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
