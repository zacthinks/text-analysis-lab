"""Read-only, project-independent discovery of TeAL model resources.

Discovery never downloads or validates a resource. Finding files does not mean a
model is safe, compatible, or ready for scientific execution.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
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
    detail: str | None = None


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

    def _local_registry_path(self) -> Path:
        return self.cache_paths()["models"] / "local_resources.json"

    def _local_entries(self) -> dict[str, str]:
        path = self._local_registry_path()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in data.items()
        ):
            raise ValueError("Invalid local model resource registry")
        return data

    def _save_local_entries(self, data: dict[str, str]) -> None:
        target = self._local_registry_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(prefix=".local-models-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\\n")
            os.replace(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)

    def register_local(self, name: str, path: str | Path) -> ResourceStatus:
        """Remember an external file/directory, without copying or executing it."""
        alias = str(name).strip()
        if not alias or any(c in alias for c in "/\\\\:") or alias in {".", ".."}:
            raise ValueError("Choose a nonempty model name without path separators")
        source = Path(path).expanduser().resolve()
        if not source.exists() or not (source.is_file() or source.is_dir()):
            raise FileNotFoundError(f"Local model resource not found: {source}")
        entries = self._local_entries()
        if alias in entries:
            raise FileExistsError(f"Model resource name already registered: {alias}")
        entries[alias] = str(source)
        self._save_local_entries(entries)
        return self.inspect(f"local:{alias}")

    def unregister_local(self, name: str) -> None:
        """Forget the external reference. Never delete the researcher's source."""
        entries = self._local_entries()
        if name not in entries:
            raise KeyError(f"Unknown local model resource {name!r}")
        del entries[name]
        self._save_local_entries(entries)

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
        # Existing standard and legacy Hugging Face snapshots are user assets,
        # even when they are not one of TeAL's curated built-in models.
        known = {item.descriptor.key for item in results}
        discovered: dict[str, list[ResourceLocation]] = {}
        for source, root in (
            ("huggingface_hub", paths["huggingface_hub"]),
            ("legacy_teal_hf", paths["legacy_teal_hf"]),
        ):
            if not root.is_dir():
                continue
            for repository in root.glob("models--*"):
                if not repository.is_dir():
                    continue
                identifier = repository.name.removeprefix("models--").replace("--", "/")
                if not identifier or f"huggingface:{identifier}" in known:
                    continue
                descriptor = ResourceDescriptor(
                    "huggingface", identifier, "hf_snapshot", "user-supplied"
                )
                discovered.setdefault(identifier, []).extend(
                    _hub_locations(root, descriptor, source)
                )
        for identifier, locations in sorted(discovered.items()):
            if not locations:
                continue
            results.append(ResourceStatus(
                descriptor=ResourceDescriptor(
                    "huggingface", identifier, "hf_snapshot", "user-supplied"
                ),
                candidate_path=locations[0].path,
                present=True,
                locations=tuple(locations),
            ))
        for name, path in self._local_entries().items():
            source = Path(path)
            present = source.exists() and (source.is_file() or source.is_dir())
            results.append(ResourceStatus(
                descriptor=ResourceDescriptor(
                    "local", name, "external_reference", "user-supplied"
                ),
                candidate_path=source if present else None,
                present=present,
                locations=(ResourceLocation("registered_local", source),),
                detail="Reference only: execution compatibility is not verified.",
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
        revision: str | None = None,
        progress=None,
        cancelled=None,
    ):
        """Resolve or explicitly download one supported model without a Project.

        Uses native provider functions; unsupported callback behavior is explicit.
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
        if (progress is not None or cancelled is not None) and not model_key.startswith("spacy:"):
            raise NotImplementedError("Progress/cancellation callbacks are currently spaCy-only.")
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
                # User-selected Hub resources are downloaded as data only.
                # No model code is executed or treated as a supported translator.
                model_id = model_key.removeprefix("huggingface:")
                from huggingface_hub.utils import validate_repo_id

                validate_repo_id(model_id)
                descriptor = ResourceDescriptor(
                    "huggingface", model_id, "hf_snapshot", "user-supplied",
                    revision=revision,
                )
            elif revision is not None:
                raise ValueError(
                    "Known TeAL model revisions cannot be overridden by the model manager."
                )
            from huggingface_hub import snapshot_download
            from text_analysis_lab.linguistics.cache import active_hf_cache_path
            cache = active_hf_cache_path(
                descriptor.model_id, revision=descriptor.revision,
                teal_root=self.cache_paths()["teal"],
                cache_dir=self._hf_cache_dir,
            )
            snapshot = Path(snapshot_download(
                repo_id=descriptor.model_id,
                revision=descriptor.revision,
                cache_dir=str(cache),
                local_files_only=not allow_download,
            ))
            if descriptor.model_id == "Babelscape/wsl-reader-deberta-v3-base":
                # The reader also loads a pinned DeBERTa base configuration.
                # Prefetch that native Hub file into the same shared cache.
                from huggingface_hub import hf_hub_download
                from text_analysis_lab.linguistics.wsd.wsl_reader import (
                    DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
                )
                hf_hub_download(
                    repo_id="microsoft/deberta-v3-base",
                    filename="config.json",
                    revision=DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
                    cache_dir=str(cache),
                    local_files_only=not allow_download,
                )
            return snapshot
        raise KeyError(f"Unknown TeAL model resource {model_key!r}.")


    def verify(self, model_key: str) -> ResourceStatus:
        """Run offline provider checks; local references are never called runnable."""
        if model_key.startswith("local:"):
            return self.inspect(model_key)
        if (
            model_key.startswith("huggingface:")
            and not any(item.key == model_key for item in _BUILTINS)
        ):
            # Unknown Hub files may exist but are not scientifically verified.
            return self.inspect(model_key)
        from text_analysis_lab.models.lifecycle import verify_resource
        return verify_resource(self, model_key)

    def remove(
        self, model_key: str, *, version: str | None = None,
        digest: str | None = None, revision: str | None = None,
        confirm: bool = False,
    ) -> dict[str, str | bool]:
        """Preview/confirm exact cache deletion without uninstalling packages."""
        from text_analysis_lab.models.lifecycle import remove_resource
        return remove_resource(self, model_key, version=version, digest=digest,
                               revision=revision, confirm=confirm)
