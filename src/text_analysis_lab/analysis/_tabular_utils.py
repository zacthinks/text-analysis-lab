"""Shared field resolution for batchwise tabular Analytic Methods."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    qualified_name: str


def resolve_tabular_fields(
    artifact: BaseArtifact,
    *names: str,
    with_artifacts: Any = None,
    with_columns: Any = True,
) -> tuple[ResolvedField, ...]:
    """Resolve field identities against the full artifact view."""
    columns = (
        artifact.query_columns(metadata_mode="full")
        if with_artifacts is None
        else artifact.query_columns(
            metadata_mode="full", with_artifacts=with_artifacts,
        )
    )["columns"]
    allowed_with: set[str] | None = None
    if with_artifacts is not None and with_columns is not True:
        # Use the exact namespace selector that BaseArtifact.query uses.
        query = artifact.project.query
        full_view = query.build_composed_view_sql(
            artifact, metadata_mode="full", include_data=True,
            with_artifacts=with_artifacts,
        )
        allowed_with = set(query._select_namespace_columns(
            full_view, namespace="with", requested=with_columns,
            label="with columns",
        ))
    resolved: list[ResolvedField] = []
    for raw_name in names:
        name = str(raw_name)
        if not name:
            raise ValueError("Field names must be non-empty strings.")
        candidates = [
            info
            for info in columns
            if info["namespace"] in {"data", "metadata", "with"}
            and name == str(info["output_name"])
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
                f"Field {name!r} is not an exact output label in this query scope. "
                f"Available fields: {available}."
            )
        if len(unique) > 1:
            options = sorted(
                str(info["qualified_name"])
                for info in candidates
                if info["namespace"] in {"data", "metadata", "with"}
            )
            raise ArtifactError(
                f"Field {name!r} is ambiguous. Available output labels: {options}."
            )
        namespace, output_name = next(iter(unique))
        if namespace == "with" and allowed_with is not None:
            if output_name not in allowed_with:
                raise ArtifactError(
                    f"Attached field {name!r} was excluded by with_columns."
                )
        resolved.append(
            ResolvedField(
                requested_name=name,
                output_name=output_name,
                namespace=namespace,  # type: ignore[arg-type]
                qualified_name=next(
                    str(info["qualified_name"]) for info in candidates
                    if str(info["output_name"]) == output_name
                    and str(info["namespace"]) == namespace
                ),
            )
        )
    return tuple(resolved)


def resolve_batch_output_names(
    artifact: BaseArtifact,
    fields: tuple[ResolvedField, ...],
    *,
    with_artifacts: Any,
) -> tuple[ResolvedField, ...]:
    """Return canonical query labels unchanged by a narrower projection.

    Selection no longer changes output names, so no per-batch renaming is
    necessary. Retain this helper for existing histogram/crosstab callers.
    """
    _ = artifact, with_artifacts
    return fields


def batch_query_columns(
    fields: tuple[ResolvedField, ...],
) -> tuple[list[str] | bool, list[str] | bool]:
    """Pass the exact output labels of the planned query view."""
    data = [field.output_name for field in fields if field.namespace == "data"]
    metadata = [
        field.output_name for field in fields if field.namespace == "metadata"
    ]
    return (data or False, metadata or False)


def batch_with_columns(fields: tuple[ResolvedField, ...]) -> list[str] | bool:
    """Project attached fields using the planned view's exact output labels."""
    selected = [field.output_name for field in fields if field.namespace == "with"]
    return selected or False
