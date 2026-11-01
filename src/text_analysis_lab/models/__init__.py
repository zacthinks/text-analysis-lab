"""Project-independent discovery of TeAL model resource locations.

This module is intentionally read-only. A path being present does not prove a model
is compatible or verified; provider-specific validation is a later operation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from text_analysis_lab.linguistics.cache import user_cache_paths


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    """Stable resource identity and minimal discovery metadata."""

    provider: str
    model_id: str
    storage_kind: str
    task: str
    revision: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    """Honest discovery status, without implying successful validation."""

    descriptor: ResourceDescriptor
    candidate_path: Path | None
    present: bool
    verified: bool = False
    compatible: bool = False
    ready: bool = False


def standard_hf_cache_path(*, environ: dict[str, str] | None = None) -> Path:
    """Resolve HF Hub's ordinary cache environment precedence without side effects.

    Does not change credentials, environment variables, or create any directory.
    An explicit cache_dir passed to an HF call takes precedence over this default.
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


# These are the built-in model families TeAL actually uses, not a remote catalog.
# Revision is populated only when the existing implementation specifies one.
_BUILTINS: tuple[ResourceDescriptor, ...] = (
    ResourceDescriptor("huggingface", "biu-nlp/f-coref", "hf_snapshot", "coreference"),
    ResourceDescriptor("huggingface", "biu-nlp/lingmess-coref", "hf_snapshot", "coreference"),
    ResourceDescriptor(
        "huggingface", "Babelscape/wsl-reader-deberta-v3-base",
        "hf_snapshot", "word-sense-disambiguation",
        "809d05bd12f261d26b42e28dc2b31db430c1585c",
    ),
    ResourceDescriptor(
        "allennlp", "allennlp-srl-bert-2020-12-15", "archive_and_runtime",
        "semantic-role-labeling",
    ),
)


class ModelManager:
    """Standalone entry point; no Project or model runtime is required."""

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
        """List known resources without downloading, importing NLP, or writing.

        HF snapshot presence is deliberately not inferred from an opaque hub tree;
        inspection/verification must resolve the exact revision through HF APIs.
        """
        paths = self.cache_paths()
        result = []
        for descriptor in _BUILTINS:
            path = None
            present = False
            if descriptor.provider == "allennlp":
                path = paths["models"] / descriptor.model_id
                present = path.is_dir()
            result.append(ResourceStatus(descriptor, path, present))
        return tuple(result)
