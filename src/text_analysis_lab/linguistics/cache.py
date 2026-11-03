"""User-level cache locations for reusable TeAL model assets.

Project workspaces contain manifests and derived research artifacts. Large reusable model
weights live in one operating-system-appropriate user cache so opening a new project does
not duplicate or redownload them. Credentials are deliberately not managed here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_CACHE_ENV = "TEAL_CACHE_DIR"


@dataclass(frozen=True, slots=True)
class UserCachePaths:
    """Canonical process-independent cache locations used by TeAL."""

    root: Path
    models: Path
    huggingface_hub: Path
    wsd_gloss_text: Path
    wordnet: Path
    wordnet_mwe_indices: Path

    def ensure(self) -> UserCachePaths:
        for path in (
            self.root,
            self.models,
            self.huggingface_hub,
            self.wsd_gloss_text,
            self.wordnet,
            self.wordnet_mwe_indices,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "models": str(self.models),
            "huggingface_hub": str(self.huggingface_hub),
            "wsd_gloss_text": str(self.wsd_gloss_text),
            "wordnet": str(self.wordnet),
            "wordnet_mwe_indices": str(self.wordnet_mwe_indices),
        }


def default_user_cache_root() -> Path:
    """Return the OS-appropriate TeAL cache root.

    ``TEAL_CACHE_DIR`` overrides the default. The cache contains only reusable/downloadable
    assets; Hugging Face credentials remain in the user's normal Hugging Face credential
    location because TeAL passes a model ``cache_dir`` rather than changing
    ``HF_HOME``.
    """

    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "TextAnalysisLab" / "Cache"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "TextAnalysisLab"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "text-analysis-lab"


def user_cache_paths(
    root: str | Path | None = None, *, create: bool = True
) -> UserCachePaths:
    """Return reusable cache paths, optionally rooted at an explicit directory."""

    cache_root = (
        default_user_cache_root() if root is None else Path(root).expanduser().resolve()
    )
    paths = UserCachePaths(
        root=cache_root,
        models=cache_root / "models",
        huggingface_hub=cache_root / "huggingface" / "hub",
        wsd_gloss_text=cache_root / "wsd" / "gloss_text",
        wordnet=cache_root / "wordnet",
        wordnet_mwe_indices=cache_root / "wordnet" / "mwe_indices",
    )
    return paths.ensure() if create else paths


def standard_hf_cache_path(*, environ: dict[str, str] | None = None) -> Path:
    """Find the normal Hugging Face Hub cache without importing it or creating files."""
    env = os.environ if environ is None else environ
    override = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if override:
        return Path(override).expanduser()
    if env.get("HF_HOME"):
        home = Path(env["HF_HOME"]).expanduser()
    else:
        xdg = env.get("XDG_CACHE_HOME")
        home = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface"
    return home / "hub"


def active_hf_cache_path(
    repository: str,
    *,
    revision: str | None = None,
    teal_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Explicit directory / native environment selection > complete legacy > native default.

    Returning the cache root never migrates or modifies any cached files.
    Explicit cache_dir is also what native Hub APIs already accept per call.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    # A profile-level native Hub override is an explicit cache selection;
    # it must not be overridden by an older, otherwise valid TeAL snapshot.
    if any(os.environ.get(name) for name in (
        "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
    )):
        return standard_hf_cache_path()
    legacy = user_cache_paths(teal_root, create=False).huggingface_hub
    snapshots = legacy / ("models--" + repository.replace("/", "--")) / "snapshots"
    if snapshots.is_dir() and not snapshots.is_symlink():
        candidates = (snapshots / revision,) if revision else tuple(snapshots.iterdir())
        for candidate in candidates:
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            # A nonempty snapshot isn't necessarily usable; avoid shadowing the
            # native cache with an interrupted legacy download.
            if not (candidate / "config.json").is_file():
                continue
            weights = (
                "model.safetensors", "model.safetensors.index.json",
                "pytorch_model.bin", "pytorch_model.bin.index.json",
            )
            if any((candidate / name).is_file() for name in weights):
                return legacy
    return standard_hf_cache_path()
