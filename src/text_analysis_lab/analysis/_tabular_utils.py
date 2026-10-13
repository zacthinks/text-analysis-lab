"""Shared field resolution for batchwise tabular Analytic Methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


FieldNamespace = Literal["data", "metadata", "with"]


@dataclass(frozen=True)
class ResolvedField:
    requested_name: str
    output_name: str
    namespace: FieldNamespace


def resolve_tabular_fields(
    artifact: BaseArtifact,
    *names: str,
    with_artifacts: Any = None,
    with_columns: Any = True,
) -> tuple[ResolvedField, ...]:
    """Resolve unambiguous data/metadata fields using the full artifact view."""
    columns = (artifact.query_columns(metadata_mode="full") if with_artifacts is None
               else artifact.query_columns(metadata_mode="full", with_artifacts=with_artifacts))["columns"]
    resolved: list[ResolvedField] = []
    for raw_name in names:
        name = str(raw_name)
        if not name:
            raise ValueError("Field names must be non-empty strings.")
        candidates = [
            info
            for info in columns
            if info["namespace"] in {"data", "metadata", "with"}
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
                if info["namespace"] in {"data", "metadata", "with"}
            )
            raise ArtifactError(
                f"Field {name!r} is not available as artifact data or metadata. "
                f"Available fields: {available}."
            )
        if len(unique) > 1:
            options = sorted(
                str(info["qualified_name"])
                for info in candidates
                if info["namespace"] in {"data", "metadata", "with"}
            )
            raise ArtifactError(
                f"Field {name!r} is ambiguous. Use one of the qualified names: {options}."
            )
        namespace, output_name = next(iter(unique))
        if namespace == "with" and with_columns is not True:
            allowed = [] if with_columns is False else (
                [with_columns] if isinstance(with_columns, str)
                else [str(item) for item in with_columns]
            )
            choices = {name, output_name}
            choices.update(str(info["base_name"]) for info in candidates
                           if str(info["output_name"]) == output_name)
            choices.update(str(info["qualified_name"]) for info in candidates
                           if str(info["output_name"]) == output_name)
            if not choices.intersection(allowed):
                raise ArtifactError(f"Attached field {name!r} was excluded by with_columns.")
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


def batch_with_columns(fields: tuple[ResolvedField, ...]) -> list[str] | bool:
    """Project only fields needed by the aggregation."""
    selected = [field.requested_name for field in fields if field.namespace == "with"]
    return selected or False
