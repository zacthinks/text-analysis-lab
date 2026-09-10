"""Reusable content-analysis dictionary specifications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.source import DictionarySource

ValueType = Literal["fixed", "glob", "non_whitespace_glob", "regex"]


def _normalize_valuetype(value: str) -> ValueType:
    normalized = str(value).lower()
    if normalized not in {"fixed", "glob", "non_whitespace_glob", "regex"}:
        raise ValueError(
            "valuetype must be one of 'fixed', 'glob', 'non_whitespace_glob', or 'regex'."
        )
    return normalized  # type: ignore[return-value]


def _normalize_patterns(values: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    patterns = tuple(str(value) for value in raw)
    if any(value == "" for value in patterns):
        raise ValueError("Dictionary patterns must be non-empty strings.")
    return patterns


class Dictionary:
    """Named sets of feature patterns for content analysis.

    Parameters
    ----------
    entries:
        Mapping from dictionary key/category to one or more feature patterns.
    valuetype:
        ``"fixed"`` for exact feature names, ``"glob"`` for shell-style
        wildcards such as ``"econom*"`` where wildcards may span whitespace,
        ``"non_whitespace_glob"`` for the same user-facing glob syntax with
        ``*`` and ``?`` constrained not to consume whitespace, or ``"regex"``
        for regular expressions.
    case_sensitive:
        Whether feature matching distinguishes case.  The default is False,
        matching the usual content-analysis workflow on lower-cased DTMs.
    name:
        Optional human-readable dictionary name.  It has no effect on matching.
    """

    def __init__(
        self,
        entries: Mapping[str, str | Iterable[str]],
        *,
        valuetype: ValueType | str = "glob",
        case_sensitive: bool = False,
        name: str | None = None,
        provenance: DictionaryProvenance | Mapping[str, Any] | None = None,
        source: DictionarySource | Mapping[str, Any] | None = None,
    ) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_key, raw_values in entries.items():
            key = str(raw_key)
            if not key:
                raise ValueError("Dictionary keys must be non-empty strings.")
            if key in normalized:
                raise ValueError(f"Duplicate dictionary key: {key!r}.")
            normalized[key] = _normalize_patterns(raw_values)
        if not normalized:
            raise ValueError("Dictionary requires at least one key/category.")
        self._entries = MappingProxyType(normalized)
        self.valuetype: ValueType = _normalize_valuetype(str(valuetype))
        self.case_sensitive = bool(case_sensitive)
        self.name = None if name is None else str(name)
        self.provenance = DictionaryProvenance.from_value(provenance)
        self.source = DictionarySource.from_value(source)

    @property
    def entries(self) -> Mapping[str, tuple[str, ...]]:
        return self._entries

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, selection: str | Sequence[str]) -> "Dictionary":
        keys = [selection] if isinstance(selection, str) else list(selection)
        missing = [str(key) for key in keys if str(key) not in self._entries]
        if missing:
            raise KeyError(f"Unknown dictionary key(s): {missing}.")
        return Dictionary(
            {str(key): self._entries[str(key)] for key in keys},
            valuetype=self.valuetype,
            case_sensitive=self.case_sensitive,
            name=self.name,
            provenance=self.provenance,
            source=None,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return one row per dictionary key/pattern pair."""
        rows = [
            {"key": key, "pattern": pattern}
            for key, patterns in self._entries.items()
            for pattern in patterns
        ]
        return pd.DataFrame(rows, columns=["key", "pattern"])

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        key: str = "key",
        pattern: str = "pattern",
        valuetype: ValueType | str = "glob",
        case_sensitive: bool = False,
        name: str | None = None,
        provenance: DictionaryProvenance | Mapping[str, Any] | None = None,
    ) -> "Dictionary":
        """Construct a dictionary from a tidy key/pattern data frame."""
        missing = [column for column in (key, pattern) if column not in frame.columns]
        if missing:
            raise ValueError(f"Dictionary frame is missing columns {missing}.")
        entries: dict[str, list[str]] = {}
        for key_value, group in frame.loc[:, [key, pattern]].groupby(key, sort=False):
            entries[str(key_value)] = [str(value) for value in group[pattern].tolist()]
        return cls(
            entries,
            valuetype=valuetype,
            case_sensitive=case_sensitive,
            name=name,
            provenance=provenance,
        )
