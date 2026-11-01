"""Read-only, project-independent discovery of TeAL model resources.

Discovery never downloads or validates a resource. Finding files does not mean a
model is safe, compatible, or ready for scientific execution.
"""
from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path

from text_analysis_lab.linguistics.cache import user_cache_paths
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
    verified: bool = False
    compatible: bool = False
    ready: bool = False
    locations: tuple[ResourceLocation, ...] = ()
    prepared: bool = False


def standard_hf_cache_path(*, environ: dict[str, str] | None = None) -> Path:
    """Compute Hub cache location without importing HF, changing env or writing.

    Explicit cache_dir on a Hub call takes priority over these environment settings.
    """
    env = os.environ if environ is None else environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        home = Path(env["HF_HOME"]).expanduser()
    else:
        xdg = env.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        home = base / "huggingface"
    return home / "hub"


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
                resource_paths = project_model_paths(paths["models"], ALLENNLP_SRL_BERT)
                if resource_paths.archive_path.is_file():
                    locations.append(ResourceLocation("srl_archive", resource_paths.archive_path))
                from text_analysis_lab.linguistics.srl.resources import _runtime_is_ready
                prepared = _runtime_is_ready(resource_paths)
                if prepared:
                    locations.append(ResourceLocation("srl_runtime", resource_paths.runtime_dir))
            elif descriptor.provider == "spacy":
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
