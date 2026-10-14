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
    """Map selected field identities onto the actual narrow query view.

    A full-view output name can change when caller data or metadata are omitted.
    Qualified field identities remain stable across those projections.
    """
    if with_artifacts is None:
        return fields
    data_columns, metadata_columns = batch_query_columns(fields)
    view = artifact.project.query.build_composed_view_sql(
        artifact,
        metadata_mode="full" if metadata_columns is not False else "none",
        include_data=data_columns is not False,
        with_artifacts=with_artifacts,
    )
    remapped: list[ResolvedField] = []
    for field in fields:
        output_name = view.mapping.get(field.qualified_name)
        if output_name is None:
            raise ArtifactError(
                f"Selected field {field.qualified_name!r} is not present in "
                "the aggregation's narrow query view."
            )
        remapped.append(replace(field, output_name=output_name))
    return tuple(remapped)


def batch_query_columns(
    fields: tuple[ResolvedField, ...],
    *,
    qualified: bool = False,
) -> tuple[list[str] | bool, list[str] | bool]:
    data = [
        field.qualified_name if qualified else field.requested_name
        for field in fields if field.namespace == "data"
    ]
    metadata = [
        field.qualified_name if qualified else field.requested_name
        for field in fields if field.namespace == "metadata"
    ]
    return (data or False, metadata or False)


def batch_with_columns(fields: tuple[ResolvedField, ...]) -> list[str] | bool:
    """Project only fields needed by the aggregation."""
    selected = [field.qualified_name for field in fields if field.namespace == "with"]
    return selected or False
