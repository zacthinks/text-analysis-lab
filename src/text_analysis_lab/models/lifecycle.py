"""Thin provider-specific checks and safe cache eviction for ModelManager.

All checks are offline; model-data readiness does not guarantee that GPU/model
execution will succeed. Deletion must be confirmed and can affect other projects.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path


def verify_resource(manager, key):
    from text_analysis_lab.linguistics.model_resources import (
        ALLENNLP_SRL_BERT, project_model_paths,
    )
    status = manager.inspect(key)
    desc = status.descriptor
    if desc.provider == "spacy":
        from text_analysis_lab.models.spacy import SpacyResourceError, resolve_spacy
        try:
            found = resolve_spacy(manager.cache_paths()["models"], desc.model_id,
                                  allow_download=False)
        except (SpacyResourceError, ImportError) as exc:
            return replace(status, verified=False, compatible=False, ready=False,
                           detail=str(exc))
        return replace(
            status, present=True, verified=True, compatible=True, ready=True,
            candidate_path=Path(found.load_target) if found.source == "managed_cache" else None,
            detail=f"Loadable spaCy pipeline: {found.source}, {found.version}.",
        )
    if desc.provider == "allennlp":
        from text_analysis_lab.linguistics.srl.resources import _runtime_is_ready
        paths = project_model_paths(manager.cache_paths()["teal"], ALLENNLP_SRL_BERT)
        prepared = _runtime_is_ready(paths)
        return replace(
            status, prepared=prepared, verified=None, compatible=None,
            ready=None if prepared else False,
            detail=("Converted SRL files and format metadata are present, "
                    "but weights, dependencies and actual runtime loading "
                    "have not been validated." if prepared
                    else "SRL converted files are missing or have an incompatible "
                    "format; prepare the resource before use."),
        )
    if desc.provider == "huggingface":
        from text_analysis_lab.linguistics.cache import active_hf_cache_path
        from text_analysis_lab.linguistics.coreference.runtime import _model_is_present
        from text_analysis_lab.models import _hub_locations, _coref_materialized_locations
        cache = active_hf_cache_path(
            desc.model_id, revision=desc.revision,
            teal_root=manager.cache_paths()["teal"], cache_dir=manager._hf_cache_dir,
        )
        candidates = _coref_materialized_locations(cache, desc, "materialized")
        candidates += _hub_locations(cache, desc, "snapshot")
        valid = next(
            (loc.path for loc in candidates if loc.path is not None and
             _model_is_present(loc.path)), None,
        )
        if valid is None:
            return replace(
                status, verified=None, compatible=None, ready=False,
                detail=("No structurally complete config-and-weights snapshot "
                        "was found; resource loading has not been attempted."),
            )
        # Filenames alone are insufficient to validate weights, tokenizers,
        # model architecture or optional runtimes. This is structural presence,
        # not a successful load/compatibility check.
        return replace(
            status, present=True, candidate_path=valid, verified=None,
            compatible=None, ready=None,
            detail=("Config and recognized weight filenames exist locally. "
                    "Contents, tokenizer, dependencies and runtime compatibility "
                    "have not been validated."),
        )
    return replace(status, detail="Auxiliary cache presence only; runtime validation not applicable.")


def remove_resource(manager, key, *, version=None, digest=None, revision=None,
                    confirm=False):
    from text_analysis_lab.linguistics.cache import active_hf_cache_path, user_cache_paths
    from text_analysis_lab.linguistics.model_resources import ALLENNLP_SRL_BERT, project_model_paths
    from text_analysis_lab.models import _BUILTINS

    warning = (
        "Other projects or running processes may be using this shared asset. "
        "Frozen work requiring it may fail until restored."
    )
    if key.startswith("huggingface:"):
        from huggingface_hub import scan_cache_dir
        desc = next((item for item in _BUILTINS if item.key == key), None)
        if desc is None:
            raise KeyError(f"Unknown model {key!r}.")
        if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Removal requires an exact 40-character Hub commit revision.")
        cache = active_hf_cache_path(
            desc.model_id, revision=revision,
            teal_root=manager.cache_paths()["teal"], cache_dir=manager._hf_cache_dir,
        )
        if not cache.is_dir():
            raise FileNotFoundError(f"Hub cache not found: {cache}")
        info = scan_cache_dir(cache_dir=cache)
        selected = [
            rev for repo in info.repos if repo.repo_id == desc.model_id
            and repo.repo_type == "model"
            for rev in repo.revisions if rev.commit_hash == revision
        ]
        if len(selected) != 1:
            raise FileNotFoundError(f"Exact cached revision not found: {desc.model_id}@{revision}")
        strategy = info.delete_revisions(revision)
        result = {
            "model": key, "revision": revision, "cache": str(cache),
            "path": str(selected[0].snapshot_path),
            "expected_freed_size": strategy.expected_freed_size_str,
            "shared_cache_warning": warning + (
                " Native Hub deletion preserves blobs shared with other revisions. "
                "Legacy FastCoref materialized copies, if present, are untouched."
            ),
            "removed": False,
        }
        if confirm:
            # Do not act on a previous scan; ensure the selected repo/revision still exists.
            current = scan_cache_dir(cache_dir=cache)
            matches = [
                rev for repo in current.repos if repo.repo_id == desc.model_id
                and repo.repo_type == "model"
                for rev in repo.revisions if rev.commit_hash == revision
            ]
            if len(matches) != 1:
                raise FileNotFoundError("The selected Hub revision changed before confirmation.")
            current.delete_revisions(revision).execute()
            result["removed"] = True
        return result
    if key == f"allennlp:{ALLENNLP_SRL_BERT.model_id}" or key.startswith("teal-cache:"):
        teal = user_cache_paths(manager._cache_root, create=False)
        candidates = {
            f"allennlp:{ALLENNLP_SRL_BERT.model_id}":
                project_model_paths(teal.root, ALLENNLP_SRL_BERT).root,
            "teal-cache:wordnet": teal.wordnet,
            "teal-cache:wordnet-mwe-indices": teal.wordnet_mwe_indices,
            "teal-cache:wsd-gloss-text": teal.wsd_gloss_text,
        }
        if key not in candidates:
            raise KeyError(f"Unknown cache resource {key!r}.")
        if key == "teal-cache:wordnet":
            raise NotImplementedError(
                "The WordNet cache path is an aggregate containing the separately "
                "managed wordnet-mwe-indices resource. Delete that exact child "
                "instead; TeAL will not recursively remove the parent or wn's "
                "provider-managed lexicon."
            )
        target = candidates[key]
        if target.is_symlink() or any(
            parent.is_symlink() for parent in target.parents
            if parent == teal.root or teal.root in parent.parents
        ):
            raise ValueError("Refusing symlinked TeAL cache path.")
        result = {
            "model": key, "path": str(target),
            "shared_cache_warning": warning + (
                " This deletes TeAL-owned data only, not Python packages or wn's "
                "external lexicon installation."
            ),
            "removed": False,
        }
        if not confirm:
            return result
        if not target.is_dir():
            raise FileNotFoundError(f"TeAL-owned cache directory not found: {target}")
        shutil.rmtree(target)
        result["removed"] = True
        return result
    if key.startswith("spacy:"):
        from text_analysis_lab.models.spacy import (
            APPROVED_SPACY_MODELS, SpacyResourceError,
            spacy_cache_root, verify_managed_spacy,
        )
        from text_analysis_lab.models.spacy_install import _install_lock
        name = key.removeprefix("spacy:")
        if name not in APPROVED_SPACY_MODELS:
            raise KeyError(f"Unknown model {key!r}.")
        if not version or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version) is None:
            raise ValueError("Specify the exact spaCy version.")
        if not digest or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Specify the exact 64-character content SHA-256 digest.")
        root = spacy_cache_root(manager.cache_paths()["models"])
        target = root / name / version / digest
        plan = {
            "model": key, "version": version, "digest": digest, "path": str(target),
            "shared_cache_warning": warning, "removed": False,
        }
        if not confirm:
            return plan
        if not target.is_dir() or target.is_symlink():
            raise FileNotFoundError(f"Managed spaCy cache entry not found: {target}")
        try:
            verify_managed_spacy(manager.cache_paths()["models"], name, version, digest)
        except SpacyResourceError as exc:
            raise ValueError("Refusing deletion of an unverified spaCy entry.") from exc
        with _install_lock(root / name / ".install.lock"):
            verify_managed_spacy(manager.cache_paths()["models"], name, version, digest)
            shutil.rmtree(target)
        plan["removed"] = True
        return plan
    raise KeyError(f"Unknown model resource {key!r}.")
