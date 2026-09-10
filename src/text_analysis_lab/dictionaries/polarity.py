"""Polarity dictionary specifications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from text_analysis_lab.dictionaries.dictionary import Dictionary, ValueType
from text_analysis_lab.dictionaries.provenance import DictionaryProvenance


class PolarityDictionary:
    """Assign dictionary keys to positive, negative, and optional neutral poles.

    Pole names describe the mathematical coding ``+1``, ``-1``, and ``0``;
    they need not represent sentiment. Multiple dictionary keys may be assigned
    to each pole. A feature that resolves to more than one *different* pole is
    rejected when the dictionary is translated against a concrete vocabulary.
    """

    def __init__(
        self,
        dictionary: Dictionary | Mapping[str, str | Iterable[str]],
        *,
        positive: str | Sequence[str],
        negative: str | Sequence[str],
        neutral: str | Sequence[str] | None = None,
        valuetype: ValueType | str = "glob",
        case_sensitive: bool = False,
        name: str | None = None,
        provenance: DictionaryProvenance | Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(dictionary, Dictionary):
            if name is not None or provenance is not None:
                raise ValueError(
                    "name/provenance cannot be supplied when dictionary is already a Dictionary."
                )
            base = dictionary
        else:
            base = Dictionary(
                dictionary,
                valuetype=valuetype,
                case_sensitive=case_sensitive,
                name=name,
                provenance=provenance,
            )
        self.dictionary = base
        self.positive = _normalize_keys(positive, argument="positive")
        self.negative = _normalize_keys(negative, argument="negative")
        self.neutral = (
            () if neutral is None else _normalize_keys(neutral, argument="neutral")
        )

        assignments = {
            "positive": set(self.positive),
            "negative": set(self.negative),
            "neutral": set(self.neutral),
        }
        for left, right in (("positive", "negative"), ("positive", "neutral"), ("negative", "neutral")):
            overlap = sorted(assignments[left].intersection(assignments[right]))
            if overlap:
                raise ValueError(
                    f"A dictionary key cannot belong to both {left} and {right} poles: {overlap}."
                )

        missing = [
            key
            for key in (*self.positive, *self.negative, *self.neutral)
            if key not in self.dictionary.entries
        ]
        if missing:
            raise ValueError(f"Polarity references unknown dictionary keys: {missing}.")

    @property
    def name(self) -> str | None:
        return self.dictionary.name

    @property
    def provenance(self) -> DictionaryProvenance | None:
        return self.dictionary.provenance

    @property
    def source(self):
        return self.dictionary.source


def _normalize_keys(values: str | Sequence[str], *, argument: str) -> tuple[str, ...]:
    if isinstance(values, str):
        result = (values,)
    else:
        result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{argument} requires at least one non-empty dictionary key.")
    if len(set(result)) != len(result):
        raise ValueError(f"{argument} contains duplicate dictionary keys.")
    return result
