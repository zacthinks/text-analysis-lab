"""Shared-cache reverse indices for WordNet multiword lexical entries."""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_analysis_lab.linguistics.hashing import fingerprint

MWE_INDEX_SCHEMA_VERSION = 1
MWE_COMPONENT_NORMALIZATION = "unicode_word_or_apostrophe_components_v1"


@dataclass(frozen=True, slots=True)
class MultiwordLemmaEntry:
    """One WordNet lexical entry containing two or more lexical components."""

    word_id: str
    lemma: str
    pos: str
    components: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "word_id": self.word_id,
            "lemma": self.lemma,
            "pos": self.pos,
            "components": list(self.components),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MultiwordLemmaEntry:
        return cls(
            word_id=str(payload["word_id"]),
            lemma=str(payload["lemma"]),
            pos=str(payload["pos"]),
            components=tuple(str(item) for item in payload["components"]),
        )


class MultiwordLemmaIndex:
    """In-memory reverse lookup from ``(component, POS)`` to WordNet MWE entries."""

    def __init__(
        self,
        *,
        lexicon: str,
        entries: Iterable[MultiwordLemmaEntry],
        enumeration_method: str,
    ) -> None:
        self.lexicon = lexicon
        self.entries = tuple(entries)
        self.enumeration_method = enumeration_method
        reverse: dict[tuple[str, str], list[MultiwordLemmaEntry]] = {}
        for entry in self.entries:
            for component in dict.fromkeys(entry.components):
                reverse.setdefault((component, entry.pos), []).append(entry)
        self._reverse = {
            key: tuple(
                sorted(
                    values,
                    key=lambda item: (item.lemma.casefold(), item.word_id),
                )
            )
            for key, values in reverse.items()
        }

    def lookup(self, component: str, pos: str) -> tuple[MultiwordLemmaEntry, ...]:
        return self._reverse.get((normalize_component(component), pos), ())

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_version": MWE_INDEX_SCHEMA_VERSION,
            "lexicon": self.lexicon,
            "component_normalization": MWE_COMPONENT_NORMALIZATION,
            "enumeration_method": self.enumeration_method,
            "multiword_entries": len(self.entries),
            "reverse_keys": len(self._reverse),
            "reverse_links": sum(len(items) for items in self._reverse.values()),
        }

    def to_payload(self) -> dict[str, object]:
        descriptor = self.descriptor()
        return {
            **descriptor,
            "entries": [entry.to_dict() for entry in self.entries],
            "content_fingerprint": fingerprint(
                {
                    "lexicon": self.lexicon,
                    "entries": [entry.to_dict() for entry in self.entries],
                }
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MultiwordLemmaIndex:
        if int(payload.get("schema_version", -1)) != MWE_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported MWE reverse-index schema version")
        if payload.get("component_normalization") != MWE_COMPONENT_NORMALIZATION:
            raise ValueError("MWE reverse-index normalization policy does not match")
        entries = tuple(
            MultiwordLemmaEntry.from_dict(item) for item in payload.get("entries", ())
        )
        result = cls(
            lexicon=str(payload["lexicon"]),
            entries=entries,
            enumeration_method=str(payload.get("enumeration_method") or "unknown"),
        )
        expected = payload.get("content_fingerprint")
        if expected is not None:
            actual = fingerprint(
                {
                    "lexicon": result.lexicon,
                    "entries": [entry.to_dict() for entry in result.entries],
                }
            )
            if str(expected) != actual:
                raise ValueError("MWE reverse-index content fingerprint does not match")
        return result


def normalize_lemma(lemma: str) -> str:
    return " ".join(str(lemma).replace("_", " ").split())


def lexical_components(lemma: str) -> tuple[str, ...]:
    """Split a lexical entry into exact normalized components, not substrings."""

    normalized = normalize_lemma(lemma).casefold()
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", normalized, flags=re.UNICODE))


def normalize_component(component: str) -> str:
    parts = lexical_components(component)
    return parts[0] if len(parts) == 1 else normalize_lemma(component).casefold()


def build_multiword_lemma_index(wordnet: Any, *, lexicon: str) -> MultiwordLemmaIndex:
    """Enumerate one WordNet edition and build its component/POS reverse index."""

    try:
        word_iterator = wordnet.words()
        enumeration_method = "wordnet.words_all"
    except TypeError:
        lemmas = getattr(wordnet, "lemmas", None)
        if lemmas is None:
            # Lightweight test doubles and some imported lexicons expose only
            # form-specific lookup. Ordinary single-word WSD remains usable.
            enumeration_method = "global_enumeration_unavailable"
            word_iterator = ()
        else:
            enumeration_method = "lemmas_then_words_fallback"
            word_iterator = (
                word for lemma in lemmas() for word in wordnet.words(lemma)
            )

    seen_word_ids: set[str] = set()
    entries: list[MultiwordLemmaEntry] = []
    for word in word_iterator:
        word_id = str(_call_or_value(word, "id"))
        if word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        lemma = normalize_lemma(str(_call_or_value(word, "lemma")))
        pos = str(_call_or_value(word, "pos"))
        components = lexical_components(lemma)
        if len(components) <= 1:
            continue
        entries.append(
            MultiwordLemmaEntry(
                word_id=word_id,
                lemma=lemma,
                pos=pos,
                components=components,
            )
        )
    entries.sort(key=lambda item: (item.lemma.casefold(), item.pos, item.word_id))
    return MultiwordLemmaIndex(
        lexicon=lexicon,
        entries=entries,
        enumeration_method=enumeration_method,
    )


def load_or_build_multiword_lemma_index(
    wordnet: Any,
    *,
    lexicon: str,
    cache_dir: str | Path,
    force_rebuild: bool = False,
) -> tuple[MultiwordLemmaIndex, bool, Path]:
    """Load a shared serialized index, rebuilding atomically when unavailable or stale.

    Returns ``(index, cache_hit, path)``.
    """

    path = multiword_index_path(cache_dir, lexicon=lexicon)
    memory_key = str(path.resolve())
    if not force_rebuild:
        cached = _MEMORY_CACHE.get(memory_key)
        if cached is not None:
            return cached, True, path
        if path.exists():
            try:
                loaded = read_multiword_lemma_index(path)
                if loaded.lexicon != lexicon:
                    raise ValueError("cached MWE index belongs to another lexicon")
                _MEMORY_CACHE[memory_key] = loaded
                return loaded, True, path
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # A stale or interrupted cache is replaceable derived data.
                pass

    built = build_multiword_lemma_index(wordnet, lexicon=lexicon)
    write_multiword_lemma_index_atomic(path, built)
    _MEMORY_CACHE[memory_key] = built
    return built, False, path


def multiword_index_path(cache_dir: str | Path, *, lexicon: str) -> Path:
    safe_lexicon = re.sub(r"[^A-Za-z0-9._-]+", "_", lexicon).strip("_") or "wordnet"
    return (
        Path(cache_dir)
        / safe_lexicon
        / f"mwe_reverse_index.v{MWE_INDEX_SCHEMA_VERSION}.json.gz"
    )


def read_multiword_lemma_index(path: str | Path) -> MultiwordLemmaIndex:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("MWE reverse-index payload must be an object")
    return MultiwordLemmaIndex.from_payload(payload)


def write_multiword_lemma_index_atomic(
    path: str | Path, index: MultiwordLemmaIndex
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                text = json.dumps(
                    index.to_payload(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                compressed.write(text.encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def clear_process_multiword_index_cache() -> None:
    """Clear only the process-local object cache; serialized files remain intact."""

    _MEMORY_CACHE.clear()


def _call_or_value(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return value() if callable(value) else value


_MEMORY_CACHE: dict[str, MultiwordLemmaIndex] = {}
