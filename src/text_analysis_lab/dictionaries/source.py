"""External dictionary source references.

These references describe how TeAL can reacquire an external lexicon without
embedding its full contents in every operator snapshot. They are execution
metadata, separate from human-readable :class:`DictionaryProvenance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_JSON_SCALARS = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class DictionarySource:
    """Reloadable reference to an external dictionary provider resource."""

    provider: str
    resource: str
    parameters: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = str(self.provider).strip()
        resource = str(self.resource).strip()
        if not provider or not resource:
            raise ValueError("DictionarySource provider and resource must be non-empty strings.")
        normalized: dict[str, str | int | float | bool | None] = {}
        for key, value in dict(self.parameters).items():
            name = str(key)
            if not name:
                raise ValueError("DictionarySource parameter names must be non-empty strings.")
            if not isinstance(value, _JSON_SCALARS):
                raise TypeError(
                    "DictionarySource parameters must contain only JSON scalar values; "
                    f"{name!r} has {type(value).__name__}."
                )
            normalized[name] = value
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "parameters", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "resource": self.resource,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_value(
        cls, value: "DictionarySource | Mapping[str, Any] | None"
    ) -> "DictionarySource | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("source must be DictionarySource, a mapping, or None.")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("DictionarySource parameters must be a mapping.")
        return cls(
            provider=str(value.get("provider", "")),
            resource=str(value.get("resource", "")),
            parameters=dict(parameters),
        )
