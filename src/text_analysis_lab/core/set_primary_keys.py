"""Replace one artifact's key namespace without copying data or metadata.

``Project.set_primary_keys(...)`` is a structural rekey operation. It preserves
exactly the same rows in exactly the same order, writes only a new key table,
and inherits representation data/metadata lazily through ``rekeyed_key``
lineage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, QueryError
from text_analysis_lab.core.failure_cleanup import mark_operation_failed_best_effort
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.lineage import validate_primary_key_relationship
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.core.utils import quote_identifier
from text_analysis_lab.core.writer import STRUCTURAL_COLUMNS, create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


class SetPrimaryKeysOperator(BaseOperator):
    """Frozen descriptor for deterministic hierarchical key replacement."""

    operation_type = "rekey"

    def __init__(
        self,
        *,
        levels: Mapping[str, str],
        leaf_key: str,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.levels = {str(k): str(v) for k, v in levels.items()}
        self.leaf_key = str(leaf_key)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = sources[DEFAULT_SOURCE_LABEL]
        return OutputSpec(
            artifact_type=source.artifact_type,
            lineage_mode="rekeyed_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {"levels": dict(self.levels), "leaf_key": self.leaf_key}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SetPrimaryKeysOperator:
        raw_levels = state.get("levels", {})
        if not isinstance(raw_levels, Mapping):
            raise ArtifactError("Serialized set_primary_keys levels must be a mapping.")
        return cls(
            levels={str(k): str(v) for k, v in raw_levels.items()},
            leaf_key=str(state.get("leaf_key", "row_id")),
        )


def _resolve_level_columns(
    project: Project,
    artifact: BaseArtifact,
    source_names: list[str],
) -> list[tuple[str, str, str]]:
    """Resolve user field names to (namespace, qualified_name, output_name)."""
    if not source_names:
        return []
    view = project.query.build_artifact_view_sql(
        artifact,
        metadata_mode="full",
        include_data=True,
    )
    resolved: list[tuple[str, str, str]] = []
    for name in source_names:
        matches = [
            column
            for column in view.columns
            if name in {column.base_name, column.output_name, column.qualified_name}
        ]
        # Deduplicate the same column if more than one naming form matched.
        unique = {(c.namespace, c.qualified_name, c.output_name): c for c in matches}
        matches = list(unique.values())
        if not matches:
            available = sorted({column.base_name for column in view.columns})
            raise ArtifactError(
                f"set_primary_keys level column {name!r} is not available. "
                f"Available fields include: {available}."
            )
        if len(matches) > 1:
            options = sorted(column.qualified_name for column in matches)
            raise ArtifactError(
                f"set_primary_keys level column {name!r} is ambiguous. "
                f"Use one of: {options}."
            )
        column = matches[0]
        resolved.append((column.namespace, column.qualified_name, column.output_name))
    return resolved


def _sorted_mapping(values: pd.Series, *, field: str) -> dict[Any, int]:
    if values.isna().any():
        raise ArtifactError(
            f"set_primary_keys grouping field {field!r} contains null values."
        )
    unique = list(pd.unique(values))
    try:
        ordered = sorted(unique)
    except TypeError as exc:
        raise ArtifactError(
            f"set_primary_keys grouping field {field!r} contains values that cannot "
            "be deterministically sorted together."
        ) from exc
    return {value: index for index, value in enumerate(ordered)}


def _build_hierarchical_keys(
    frame: pd.DataFrame,
    *,
    source_columns: list[str],
    key_names: list[str],
    leaf_key: str,
) -> pd.DataFrame:
    """Build local 0-based integer key levels while preserving row order."""
    if len(source_columns) != len(key_names):
        raise ValueError("source_columns and key_names must have equal lengths.")

    result = pd.DataFrame(index=frame.index)
    parent_keys: list[str] = []

    for source_col, key_name in zip(source_columns, key_names, strict=True):
        values = frame[source_col]
        if not parent_keys:
            mapping = _sorted_mapping(values, field=source_col)
            result[key_name] = values.map(mapping).astype("int64")
        else:
            result[key_name] = np.zeros(len(frame), dtype="int64")
            groupby_keys = parent_keys[0] if len(parent_keys) == 1 else parent_keys
            grouped_indices = result.groupby(
                groupby_keys, sort=False, dropna=False
            ).groups
            for indices in grouped_indices.values():
                index_list = list(indices)
                group_values = values.loc[index_list]
                mapping = _sorted_mapping(group_values, field=source_col)
                result.loc[index_list, key_name] = (
                    group_values.map(mapping).astype("int64").to_numpy()
                )
            result[key_name] = result[key_name].astype("int64")
        parent_keys.append(key_name)

    if parent_keys:
        groupby_keys = parent_keys[0] if len(parent_keys) == 1 else parent_keys
        result[leaf_key] = (
            result.groupby(groupby_keys, sort=False, dropna=False)
            .cumcount()
            .astype("int64")
        )
    else:
        result[leaf_key] = np.arange(len(frame), dtype="int64")

    if result.duplicated(subset=[*key_names, leaf_key]).any():
        raise ArtifactError("set_primary_keys generated duplicate primary-key tuples.")
    return result.reset_index(drop=True)


def set_primary_keys(
    project: Project,
    source: BaseArtifact | str,
    *,
    levels: Mapping[str, str],
    leaf_key: str,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> BaseArtifact:
    """Replace an artifact's primary-key namespace using existing row fields.

    ``levels`` maps accessible source field names to new integer key names in
    outer-to-inner insertion order. At each level, distinct source values are
    sorted within the parent group and assigned local running integers starting
    at zero. ``leaf_key`` is then assigned in existing artifact row order within
    the deepest group.

    Only a keys component is written. Data and metadata remain on ancestors and
    are inherited lazily through the positional ``rekeyed_key`` bridge.
    """
    artifact = project.get_artifact(source)
    artifact.require_complete()
    if not isinstance(levels, Mapping):
        raise TypeError("levels must be a mapping of source field -> new key name.")

    normalized_levels = {str(k): str(v) for k, v in levels.items()}
    if any(not name for name in normalized_levels):
        raise ArtifactError(
            "set_primary_keys source level names must be non-empty strings."
        )
    new_level_names = list(normalized_levels.values())
    leaf_key = str(leaf_key)
    if not leaf_key:
        raise ArtifactError("leaf_key must be a non-empty string.")
    all_new_names = [*new_level_names, leaf_key]
    if any(not name for name in all_new_names):
        raise ArtifactError("Generated primary-key names must be non-empty strings.")
    if len(set(all_new_names)) != len(all_new_names):
        raise ArtifactError("Generated primary-key names must be unique.")
    reserved = [name for name in all_new_names if name in STRUCTURAL_COLUMNS]
    if reserved:
        raise ArtifactError(
            f"Generated primary-key names may not use structural columns: {reserved}."
        )

    output_label = validate_output_label(output_label)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    source_names = list(normalized_levels.keys())
    resolved = _resolve_level_columns(project, artifact, source_names)
    view = project.query.build_artifact_view_sql(
        artifact, metadata_mode="full", include_data=True
    )
    existing_base_names = {column.base_name for column in view.columns}
    collisions = [name for name in all_new_names if name in existing_base_names]
    if collisions:
        raise ArtifactError(
            "Generated primary-key names must not collide with existing accessible "
            f"fields: {collisions}."
        )

    selected_output_names = [item[2] for item in resolved]
    # Reject null grouping values explicitly. DuckDB has deterministic NULL sort
    # rules, but treating missing substantive group labels as a real key level
    # would make the generated hierarchy harder to interpret.
    for source_name, output_name in zip(
        source_names, selected_output_names, strict=True
    ):
        null_sql = (
            "SELECT 1 FROM ("
            f"{view.sql}"
            f") v WHERE v.{quote_identifier(output_name)} IS NULL LIMIT 1"
        )
        try:
            if project.query.con.execute(null_sql).fetchone() is not None:
                raise ArtifactError(
                    f"set_primary_keys grouping field {source_name!r} contains null values."
                )
        except ArtifactError:
            raise
        except Exception as exc:
            raise QueryError(
                f"set_primary_keys could not validate grouping field {source_name!r}: {exc}"
            ) from exc

    key_exprs: list[str] = []
    for index, (output_name, new_key_name) in enumerate(
        zip(selected_output_names, new_level_names, strict=True)
    ):
        partition = selected_output_names[:index]
        partition_sql = (
            "PARTITION BY "
            + ", ".join(f"v.{quote_identifier(name)}" for name in partition)
            + " "
            if partition
            else ""
        )
        key_exprs.append(
            "CAST(DENSE_RANK() OVER ("
            f"{partition_sql}ORDER BY v.{quote_identifier(output_name)}"
            ") - 1 AS BIGINT) AS "
            f"{quote_identifier(new_key_name)}"
        )

    if selected_output_names:
        leaf_partition = ", ".join(
            f"v.{quote_identifier(name)}" for name in selected_output_names
        )
        leaf_expr = (
            "CAST(ROW_NUMBER() OVER (PARTITION BY "
            f"{leaf_partition} ORDER BY v._position) - 1 AS BIGINT) AS "
            f"{quote_identifier(leaf_key)}"
        )
    else:
        leaf_expr = (
            "CAST(ROW_NUMBER() OVER (ORDER BY v._position) - 1 AS BIGINT) AS "
            f"{quote_identifier(leaf_key)}"
        )
    key_exprs.append(leaf_expr)
    key_sql = (
        "SELECT " + ", ".join(key_exprs) + f" FROM ({view.sql}) v ORDER BY v._position"
    )

    output_pk = [*new_level_names, leaf_key]
    validate_primary_key_relationship(
        basis_keys=[artifact.primary_key],
        output_key=output_pk,
        lineage_mode="rekeyed_key",
    )

    operator = SetPrimaryKeysOperator(levels=normalized_levels, leaf_key=leaf_key)
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="rekey",
        snapshot_status="pending",
    )
    try:
        operator.save_to_dir(
            project.storage.operator_dir(operator_id), operator_id=operator_id
        )
        project.catalog.mark_operator_serialized(operator_id)
    except Exception:
        project.catalog.mark_operator_snapshot_failed(operator_id)
        raise

    operation_id = next_id(project.storage.manifest_path, "operation")
    op_dir = project.storage.operation_dir(operation_id)
    op_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="rekey",
        operator_id=operator_id,
        status="incomplete",
    )
    project.catalog.add_operation_source(
        operation_id, DEFAULT_SOURCE_LABEL, artifact.artifact_id
    )

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact.artifact_type,
        label=output_label,
        lineage_mode="rekeyed_key",
        status="incomplete",
        basis_artifact_ids=(artifact.artifact_id,),
    )
    project.catalog.add_operation_output(
        operation_id, output_label, artifact_id, ordinal=0
    )
    writer = create_artifact_writer(
        artifact_type=artifact.artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=output_label,
        operation_id=operation_id,
        lineage_mode="rekeyed_key",
        basis_artifact_ids=(artifact.artifact_id,),
    )
    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "rekey",
        "operator_id": operator_id,
        "kind": "set_primary_keys",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: artifact.artifact_id},
        "output_artifact_ids": {output_label: artifact_id},
        "request": {
            "output_label": output_label,
            "levels": dict(normalized_levels),
            "leaf_key": leaf_key,
        },
    }
    (op_dir / "operation.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
    )

    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )
        written_rows = 0
        try:
            result = project.query.con.execute(key_sql)
            reader = result.to_arrow_reader(batch_size=batch_size)
            for batch in reader:
                key_batch = batch.to_pandas().reset_index(drop=True)
                writer.write({"keys": key_batch})
                written_rows += len(key_batch)
        except Exception as exc:
            raise QueryError(
                f"set_primary_keys key construction failed: {exc}"
            ) from exc

        if written_rows == 0:
            empty_keys = pd.DataFrame(
                {name: pd.Series(dtype="int64") for name in output_pk}
            )
            writer.write({"keys": empty_keys})
        if written_rows != int(artifact.n_rows or 0):
            raise ArtifactError(
                "rekeyed_key invariant violated before sealing: source/output row counts differ "
                f"({artifact.n_rows} vs {written_rows})."
            )

        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        (op_dir / "operation.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
        )
        project.storage.touch_manifest()
        output = project.get_artifact(artifact_id)
        if output.n_rows != artifact.n_rows:
            raise ArtifactError(
                "rekeyed_key invariant violated after write: source/output row counts differ."
            )
        if set(output.components) != {"keys"}:
            raise ArtifactError(
                "set_primary_keys must create a keys-only artifact; unexpected components "
                f"were written: {sorted(output.components)}."
            )
        return output
    except Exception as exc:
        mark_operation_failed_best_effort(
            project,
            writer=writer,
            artifact_id=artifact_id,
            operation_id=operation_id,
            error=exc,
        )
        descriptor["status"] = "failed"
        descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
        (op_dir / "operation.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise
