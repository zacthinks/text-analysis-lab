"""Content-addressed manifest cache for exact gloss text."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from text_analysis_lab.linguistics.hashing import canonical_json, fingerprint
from text_analysis_lab.linguistics.wsd.types import (
    CacheStats,
    GlossPayload,
    GlossRenderConfig,
    SenseCandidate,
)


@dataclass(frozen=True, slots=True)
class CachedGlossText:
    gloss_hash: str
    gloss_text: str
    hit: bool


def rendered_gloss_hash(
    payload: GlossPayload, render: GlossRenderConfig
) -> tuple[str, str]:
    text = render.render(payload)
    return text, fingerprint({"language": payload.language, "text": text})


class GlossTextCache:
    """Persist exact rendered glosses by content hash.

    WSL jointly re-encodes glosses with each context, so this cache deliberately stores no
    neural vector. It provides immutable content identity, change detection, and provenance.
    """

    FORMAT_VERSION = "1"

    def __init__(
        self,
        root: str | Path,
        *,
        render_config: GlossRenderConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.render_config = render_config or GlossRenderConfig()
        self.render_fingerprint = fingerprint(
            {
                "format_version": self.FORMAT_VERSION,
                "render": self.render_config.model_dump(mode="json"),
            }
        )
        self.namespace = self.root / self.render_fingerprint
        self.namespace.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "render_fingerprint": self.render_fingerprint,
            "render": self.render_config.model_dump(mode="json"),
        }
        path = self.namespace / "_cache.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise RuntimeError(f"gloss-cache namespace metadata mismatch: {path}")
        else:
            self._write_text_atomic(path, canonical_json(manifest) + "\n")

    def _path(self, gloss_hash: str) -> Path:
        return self.namespace / gloss_hash[:2] / f"{gloss_hash}.json"

    def get(self, payload: GlossPayload) -> CachedGlossText | None:
        gloss_text, gloss_hash = rendered_gloss_hash(payload, self.render_config)
        path = self._path(gloss_hash)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "format_version": self.FORMAT_VERSION,
            "gloss_hash": gloss_hash,
            "gloss_text": gloss_text,
        }
        if any(data.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"invalid gloss-cache entry: {path}")
        return CachedGlossText(gloss_hash=gloss_hash, gloss_text=gloss_text, hit=True)

    def put(self, payload: GlossPayload) -> CachedGlossText:
        gloss_text, gloss_hash = rendered_gloss_hash(payload, self.render_config)
        existing = self.get(payload)
        if existing is not None:
            return existing
        path = self._path(gloss_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(
            path,
            canonical_json(
                {
                    "format_version": self.FORMAT_VERSION,
                    "gloss_hash": gloss_hash,
                    "gloss_text": gloss_text,
                }
            )
            + "\n",
        )
        return CachedGlossText(gloss_hash=gloss_hash, gloss_text=gloss_text, hit=False)

    def resolve_candidates(
        self, candidates: Sequence[SenseCandidate]
    ) -> tuple[dict[str, CachedGlossText], CacheStats]:
        by_hash: dict[str, GlossPayload] = {}
        candidate_hashes: dict[str, str] = {}
        for candidate in candidates:
            _, gloss_hash = rendered_gloss_hash(candidate.gloss, self.render_config)
            candidate_hashes[candidate.sense_id] = gloss_hash
            by_hash.setdefault(gloss_hash, candidate.gloss)

        resolved: dict[str, CachedGlossText] = {}
        hits = 0
        misses = 0
        for gloss_hash, payload in by_hash.items():
            item = self.get(payload)
            if item is None:
                item = self.put(payload)
                misses += 1
            else:
                hits += 1
            resolved[gloss_hash] = item
        return (
            {
                candidate.sense_id: resolved[candidate_hashes[candidate.sense_id]]
                for candidate in candidates
            },
            CacheStats(hits=hits, misses=misses, unique_glosses=len(by_hash)),
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
