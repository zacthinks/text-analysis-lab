"""Shared field resolution for batchwise tabular Analytic Methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


FieldNamespace = Literal["data", "metadata"]


@dataclass(frozen=True)
class ResolvedField:
    requested_name: str
    output_name: str
    namespace: FieldNamespace


def resolve_tabular_fields(
    artifact: "BaseArtifact",
    *names: str,
) -> tuple[ResolvedField, ...]:
    """Resolve unambiguous data/metadata fields using the full artifact view."""
    columns = artifact.query_columns(metadata_mode="full")["columns"]
    resolved: list[ResolvedField] = []
    for raw_name in names:
        name = str(raw_name)
        if not name:
            raise ValueError("Field names must be non-empty strings.")
        candidates = [
            info
            for info in columns
            if info["namespace"] in {"data", "metadata"}
            and name
            in {
                str(info["base_name"]),
                str(info["qualified_name"]),
                str(info["output_name"]),
            }
        ]
        unique = {
            (str(info["namespace"]), str(info["output_name"])) for info in candidates
        }
        if not unique:
            available = sorted(
                str(info["output_name"])
                for info in columns
                if info["namespace"] in {"data", "metadata"}
            )
            raise ArtifactError(
                f"Field {name!r} is not available as artifact data or metadata. "
                f"Available fields: {available}."
            )
        if len(unique) > 1:
            options = sorted(
                str(info["qualified_name"])
                for info in candidates
                if info["namespace"] in {"data", "metadata"}
            )
            raise ArtifactError(
                f"Field {name!r} is ambiguous. Use one of the qualified names: {options}."
            )
        namespace, output_name = next(iter(unique))
        resolved.append(
            ResolvedField(
                requested_name=name,
                output_name=output_name,
                namespace=namespace,  # type: ignore[arg-type]
            )
        )
    return tuple(resolved)


def batch_query_columns(
    fields: tuple[ResolvedField, ...],
) -> tuple[list[str] | bool, list[str] | bool]:
    data = [field.requested_name for field in fields if field.namespace == "data"]
    metadata = [
        field.requested_name for field in fields if field.namespace == "metadata"
    ]
    return (data or False, metadata or False)
