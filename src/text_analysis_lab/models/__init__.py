"""Read-only, project-independent discovery of TeAL model resources.

Discovery never downloads or validates a resource. Finding files does not mean a
model is safe, compatible, or ready for scientific execution.
"""
from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path

from text_analysis_lab.linguistics.cache import standard_hf_cache_path, user_cache_paths
from text_analysis_lab.linguistics.model_resources import (
    ALLENNLP_SRL_BERT,
    project_model_paths,
)


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    """Stable resource identity, independent of local filesystem paths."""

    provider: str
    model_id: str
    storage_kind: str
    task: str
    revision: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"


@dataclass(frozen=True, slots=True)
class ResourceLocation:
    """Candidate resource files; no integrity/compatibility claim is implied."""

    source: str
    path: Path | None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    """Read-only resource discovery status, not runtime validation."""

    descriptor: ResourceDescriptor
    candidate_path: Path | None
    present: bool
    verified: bool | None = None
    compatible: bool | None = None
    ready: bool | None = None
    locations: tuple[ResourceLocation, ...] = ()
    prepared: bool = False


# Small curated set of models already used by TeAL, plus official spaCy pipelines.
# Registered names do not imply availability or compatibility with the interpreter.
_BUILTINS: tuple[ResourceDescriptor, ...] = (
    ResourceDescriptor("spacy", "en_core_web_sm", "pipeline", "linguistic-analysis"),
    ResourceDescriptor("spacy", "en_core_web_md", "pipeline", "linguistic-analysis"),
    ResourceDescriptor("spacy", "en_core_web_lg", "pipeline", "linguistic-analysis"),
    ResourceDescriptor("huggingface", "biu-nlp/f-coref", "hf_snapshot", "coreference"),
    ResourceDescriptor("huggingface", "biu-nlp/lingmess-coref", "hf_snapshot", "coreference"),
    ResourceDescriptor(
        "huggingface", "Babelscape/wsl-reader-deberta-v3-base",
        "hf_snapshot", "word-sense-disambiguation",
        "809d05bd12f261d26b42e28dc2b31db430c1585c",
    ),
    ResourceDescriptor(
        "allennlp", ALLENNLP_SRL_BERT.model_id, "archive_and_runtime",
        "semantic-role-labeling",
    ),
    ResourceDescriptor("teal-cache", "wordnet", "directory", "lexical-data"),
    ResourceDescriptor("teal-cache", "wordnet-mwe-indices", "directory", "lexical-data"),
    ResourceDescriptor("teal-cache", "wsd-gloss-text", "directory", "word-sense-disambiguation"),
)


def _hub_locations(root: Path, descriptor: ResourceDescriptor, source: str) -> list[ResourceLocation]:
    snapshots = (root / ("models--" + descriptor.model_id.replace("/", "--")) / "snapshots")
    if not snapshots.is_dir():
        return []
    if descriptor.revision is not None:
        candidates = (snapshots / descriptor.revision,)
    else:
        candidates = tuple(sorted(snapshots.iterdir()))
    return [
        ResourceLocation(source, candidate, candidate.name)
        for candidate in candidates
        if candidate.is_dir() and any(part.exists() for part in candidate.iterdir())
    ]


def _coref_materialized_locations(root: Path, descriptor: ResourceDescriptor,
                                   source: str) -> list[ResourceLocation]:
    directory = {"biu-nlp/f-coref": "fcoref", "biu-nlp/lingmess-coref": "lingmess"}.get(
        descriptor.model_id
    )
    if directory is None:
        return []
    candidate = root.parent / "materialized" / directory
    weights = (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )
    if (candidate / "config.json").is_file() and any(
        (candidate / name).is_file() for name in weights
    ):
        return [ResourceLocation(source, candidate)]
    return []


class ModelManager:
    """Inspect recognized resource locations without opening a TeAL project."""

    def __init__(self, *, cache_root: str | Path | None = None,
                 hf_cache_dir: str | Path | None = None) -> None:
        self._cache_root = cache_root
        self._hf_cache_dir = hf_cache_dir

    def cache_paths(self) -> dict[str, Path]:
        teal = user_cache_paths(self._cache_root, create=False)
        hub = (Path(self._hf_cache_dir).expanduser()
               if self._hf_cache_dir is not None else standard_hf_cache_path())
        return {"teal": teal.root, "models": teal.models,
                "huggingface_hub": hub, "legacy_teal_hf": teal.huggingface_hub}

    def list_models(self) -> tuple[ResourceStatus, ...]:
        """List known resources without downloads, optional ML imports, or writes."""
        paths = self.cache_paths()
        results: list[ResourceStatus] = []
        for descriptor in _BUILTINS:
            locations: list[ResourceLocation] = []
            prepared = False
            if descriptor.provider == "huggingface":
                roots = (("huggingface_hub", paths["huggingface_hub"]),
                         ("legacy_teal_hf", paths["legacy_teal_hf"]))
                seen_roots: set[Path] = set()
                for source, root in roots:
                    if root in seen_roots:
                        continue
                    seen_roots.add(root)
                    locations.extend(_hub_locations(root, descriptor, source))
                    locations.extend(_coref_materialized_locations(
                        root, descriptor, source + "_materialized"
                    ))
            elif descriptor.provider == "allennlp":
                resource_paths = project_model_paths(paths["teal"], ALLENNLP_SRL_BERT)
                if resource_paths.archive_path.is_file():
                    locations.append(ResourceLocation("srl_archive", resource_paths.archive_path))
                from text_analysis_lab.linguistics.srl.resources import _runtime_is_ready
                prepared = _runtime_is_ready(resource_paths)
                if prepared:
                    locations.append(ResourceLocation("srl_runtime", resource_paths.runtime_dir))
            elif descriptor.provider == "teal-cache":
                teal = user_cache_paths(self._cache_root, create=False)
                directory = {
                    "wordnet": teal.wordnet,
                    "wordnet-mwe-indices": teal.wordnet_mwe_indices,
                    "wsd-gloss-text": teal.wsd_gloss_text,
                }[descriptor.model_id]
                if directory.is_dir() and any(directory.iterdir()):
                    locations.append(ResourceLocation("teal_cache", directory))
            elif descriptor.provider == "spacy":
                from text_analysis_lab.models.spacy import available_managed_spacy, spacy_cache_root

                for version, digest in available_managed_spacy(
                    paths["models"], descriptor.model_id
                ):
                    candidate = (
                        spacy_cache_root(paths["models"])
                        / descriptor.model_id / version / digest / "pipeline"
                    )
                    locations.append(ResourceLocation("managed_cache", candidate, version))
                try:
                    version = importlib.metadata.version(descriptor.model_id)
                except importlib.metadata.PackageNotFoundError:
                    pass
                else:
                    # A distribution version alone cannot prove spacy.load works.
                    locations.append(ResourceLocation("installed_package", None, version))
            results.append(ResourceStatus(
                descriptor=descriptor,
                candidate_path=locations[0].path if locations else None,
                present=bool(locations),
                locations=tuple(locations),
                prepared=prepared,
            ))
        return tuple(results)

    def inspect(self, model_key: str) -> ResourceStatus:
        """Inspect one known provider:model_id (no remote registry access)."""
        for item in self.list_models():
            if item.descriptor.key == model_key:
                return item
        raise KeyError(f"Unknown TeAL model resource {model_key!r}.")

    def install(
        self,
        model_key: str,
        *,
        allow_download: bool = False,
        progress=None,
        cancelled=None,
    ):
        """Resolve or explicitly download one supported model without a Project.

        Non-spaCy model installation remains a future provider operation.
        """
        if model_key.startswith("spacy:"):
            from text_analysis_lab.models.spacy_install import install_official_spacy
            return install_official_spacy(
                self.cache_paths()["models"],
                model_key.removeprefix("spacy:"),
                allow_download=allow_download,
                progress=progress,
                cancelled=cancelled,
            )
        if model_key == f"allennlp:{ALLENNLP_SRL_BERT.model_id}":
            from text_analysis_lab.linguistics.srl.resources import prepare_project_srl_runtime
            return prepare_project_srl_runtime(
                self.cache_paths()["teal"],
                cache_dir=self.cache_paths()["teal"],
                download=allow_download,
                show_progress=progress is not None,
            )
        if model_key.startswith("huggingface:"):
            descriptor = next(
                (x for x in _BUILTINS if x.key == model_key), None
            )
            if descriptor is None:
                raise KeyError(f"Unknown TeAL model resource {model_key!r}.")
            from huggingface_hub import snapshot_download
            from text_analysis_lab.linguistics.cache import active_hf_cache_path
            cache = active_hf_cache_path(
                descriptor.model_id, revision=descriptor.revision,
                teal_root=self.cache_paths()["teal"],
                cache_dir=self._hf_cache_dir,
            )
            return Path(snapshot_download(
                repo_id=descriptor.model_id,
                revision=descriptor.revision,
                cache_dir=str(cache),
                local_files_only=not allow_download,
            ))
        raise KeyError(f"Unknown TeAL model resource {model_key!r}.")


    def remove(
        self,
        model_key: str,
        *,
        version: str | None = None,
        digest: str | None = None,
        confirm: bool = False,
    ) -> dict[str, str | bool]:
        """Preview, then optionally remove one exact managed spaCy cache identity.

        Installed Python packages, project assets, and provider-native caches
        are deliberately outside this operation. No implicit eviction.
        """
        if not model_key.startswith("spacy:"):
            raise NotImplementedError(
                "Removal for this provider is not supported by TeAL; "
                "provider-native caches must be managed by their own APIs."
            )
        from text_analysis_lab.models.spacy import (
            APPROVED_SPACY_MODELS, SpacyResourceError,
            spacy_cache_root, verify_managed_spacy,
        )
        from text_analysis_lab.models.spacy_install import _install_lock
        import re
        import shutil

        name = model_key.removeprefix("spacy:")
        if name not in APPROVED_SPACY_MODELS:
            raise KeyError(f"Unknown TeAL model resource {model_key!r}.")
        if not version or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version):
            raise ValueError("Specify the exact model version to remove.")
        if not digest or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Specify the exact 64-character content SHA-256 digest.")
        root = spacy_cache_root(self.cache_paths()["models"])
        directory = root / name / version / digest
        plan: dict[str, str | bool] = {
            "model": model_key,
            "version": version,
            "digest": digest,
            "path": str(directory),
            "shared_cache_warning": (
                "Other projects or interpreters may reference this exact model. "
                "Frozen operations requiring it will fail until restored."
            ),
            "removed": False,
        }
        if not confirm:
            return plan
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"Managed spaCy cache entry not found: {directory}")
        # Never delete a mismatched, corrupt or substituted directory.
        try:
            verify_managed_spacy(self.cache_paths()["models"], name, version, digest)
        except SpacyResourceError as exc:
            raise ValueError("Refusing removal of an unverified cache entry.") from exc
        lock = root / name / ".install.lock"
        with _install_lock(lock):
            verify_managed_spacy(self.cache_paths()["models"], name, version, digest)
            shutil.rmtree(directory)
        plan["removed"] = True
        return plan
