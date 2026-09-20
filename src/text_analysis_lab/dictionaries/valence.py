"""Valence / lexical-affinity dictionary specifications."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any

import pandas as pd

from text_analysis_lab.dictionaries.dictionary import ValueType, _normalize_valuetype
from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.source import DictionarySource


class ValenceDictionary:
    """Numeric feature-pattern scores for one or more named dimensions.

    ``values`` may be a single mapping from pattern to numeric score, in which
    case the dimension is named ``"valence"``, or a mapping from dimension name
    to pattern/score mappings.
    """

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        valuetype: ValueType | str = "fixed",
        case_sensitive: bool = False,
        name: str | None = None,
        provenance: DictionaryProvenance | Mapping[str, Any] | None = None,
        source: DictionarySource | Mapping[str, Any] | None = None,
    ) -> None:
        normalized = _normalize_values(values)
        self._values = MappingProxyType(
            {key: MappingProxyType(value) for key, value in normalized.items()}
        )
        self.valuetype: ValueType = _normalize_valuetype(str(valuetype))
        self.case_sensitive = bool(case_sensitive)
        self.name = None if name is None else str(name)
        self.provenance = DictionaryProvenance.from_value(provenance)
        self.source = DictionarySource.from_value(source)

    @property
    def values(self) -> Mapping[str, Mapping[str, float]]:
        return self._values

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, dimension: str | Sequence[str]) -> ValenceDictionary:
        selected = [dimension] if isinstance(dimension, str) else list(dimension)
        missing = [str(key) for key in selected if str(key) not in self._values]
        if missing:
            raise KeyError(f"Unknown valence dimension(s): {missing}.")
        return ValenceDictionary(
            {str(key): dict(self._values[str(key)]) for key in selected},
            valuetype=self.valuetype,
            case_sensitive=self.case_sensitive,
            name=self.name,
            provenance=self.provenance,
            source=None,
        )

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"dimension": dimension, "pattern": pattern, "value": value}
            for dimension, scores in self._values.items()
            for pattern, value in scores.items()
        ]
        return pd.DataFrame(rows, columns=["dimension", "pattern", "value"])

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        dimension: str = "dimension",
        pattern: str = "pattern",
        value: str = "value",
        valuetype: ValueType | str = "fixed",
        case_sensitive: bool = False,
        name: str | None = None,
        provenance: DictionaryProvenance | Mapping[str, Any] | None = None,
    ) -> ValenceDictionary:
        missing = [
            column
            for column in (dimension, pattern, value)
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Valence dictionary frame is missing columns {missing}.")
        nested: dict[str, dict[str, float]] = {}
        for _, row in frame.loc[:, [dimension, pattern, value]].iterrows():
            key = str(row[dimension])
            term = str(row[pattern])
            score = float(row[value])
            if term in nested.setdefault(key, {}):
                raise ValueError(
                    f"Duplicate valence pattern {term!r} in dimension {key!r}."
                )
            nested[key][term] = score
        return cls(
            nested,
            valuetype=valuetype,
            case_sensitive=case_sensitive,
            name=name,
            provenance=provenance,
        )


def _normalize_values(values: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    if not values:
        raise ValueError("ValenceDictionary requires at least one value.")

    raw_values = list(values.values())
    if all(_is_number(value) for value in raw_values):
        source: Mapping[str, Any] = {"valence": values}
    elif all(isinstance(value, Mapping) for value in raw_values):
        source = values
    else:
        raise TypeError(
            "ValenceDictionary values must be either pattern->number or "
            "dimension->(pattern->number)."
        )

    normalized: dict[str, dict[str, float]] = {}
    for raw_dimension, raw_scores in source.items():
        dimension = str(raw_dimension)
        if not dimension:
            raise ValueError("Valence dimensions must be non-empty strings.")
        assert isinstance(raw_scores, Mapping)
        scores: dict[str, float] = {}
        for raw_pattern, raw_value in raw_scores.items():
            pattern = str(raw_pattern)
            if not pattern:
                raise ValueError("Valence patterns must be non-empty strings.")
            if not _is_number(raw_value):
                raise TypeError(
                    f"Valence for {pattern!r} in {dimension!r} is not numeric."
                )
            score = float(raw_value)
            if not math.isfinite(score):
                raise ValueError(
                    f"Valence for {pattern!r} in {dimension!r} must be finite."
                )
            scores[pattern] = score
        if not scores:
            raise ValueError(f"Valence dimension {dimension!r} has no patterns.")
        normalized[dimension] = scores
    return normalized


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
