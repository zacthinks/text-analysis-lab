"""Provenance metadata attached to reusable dictionary specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DictionaryProvenance:
    """Human-readable provenance for an external or user-defined dictionary.

    TeAL treats this metadata as provenance rather than as legal advice. External
    resource terms remain governed by the upstream provider and resource.
    """

    provider: str | None = None
    resource: str | None = None
    version: str | None = None
    citation: str | None = None
    license: str | None = None
    source: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_value(
        cls, value: "DictionaryProvenance | Mapping[str, Any] | None"
    ) -> "DictionaryProvenance | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("provenance must be DictionaryProvenance, a mapping, or None.")
        allowed = {
            "provider",
            "resource",
            "version",
            "citation",
            "license",
            "source",
            "notes",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown dictionary provenance field(s): {unknown}.")
        return cls(
            **{
                key: None if raw is None else str(raw)
                for key, raw in value.items()
            }
        )
