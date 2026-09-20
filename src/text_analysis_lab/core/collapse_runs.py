"""Collapse adjacent equal-valued rows into span-key relational artifacts.

``Project.collapse_runs(...)`` discovers maximal runs in current artifact order
within each parent-key prefix.  Each run is represented by the closed interval
between the minimum and maximum leaf-key values observed in that run.  Span
membership thereafter follows TeAL's normal span semantics: matching parent
keys plus ``leaf >= start`` and ``leaf <= end`` against the immediate basis.

The operation deliberately does not require monotonic leaf keys.  Monotonic,
order-preserving sources make spans easiest to interpret, but non-monotonic
inputs are permitted; users are responsible for the semantics of any resulting
overlapping or unexpectedly broad spans.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.aggregate import (
    FieldAggregationSpec,
    _apply_postprocessing,
    _concat_separator,
    _json_safe_spec,
    _normalize_field_spec,
    _resolve_rule_sources,
    _rule_sql_expression,
    _validate_output_names,
)
from text_analysis_lab.core.errors import ArtifactError, QueryError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.lineage import expected_span_key, validate_primary_key_relationship
from text_analysis_lab.core.operator import BaseOperator, OutputSpec, TranslationRequest, validate_output_label
from text_analysis_lab.core.types import ArtifactType, DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.core.utils import quote_identifier
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


class CollapseRunsOperator(BaseOperator):
    """Frozen descriptor for adjacent-run span recomposition."""

    operation_type = "translate"

    def __init__(
        self,
        *,
        by: Sequence[str],
        data_spec: Mapping[str, Any] | None = None,
        metadata_spec: Mapping[str, Any] | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.by = tuple(str(value) for value in by)
        self.data_spec = _json_safe_spec(data_spec)
        self.metadata_spec = _json_safe_spec(metadata_spec)

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        if len(sources) != 1:
            raise ArtifactError("CollapseRunsOperator expects exactly one source artifact.")
        return OutputSpec(
            artifact_type=ArtifactType.TABLE,
            lineage_mode="span_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "by": list(self.by),
            "data_spec": self.data_spec,
            "metadata_spec": self.metadata_spec,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "CollapseRunsOperator":
        raw_by = state.get("by", ())
        if isinstance(raw_by, str) or not isinstance(raw_by, Sequence):
            raise ArtifactError("Serialized collapse_runs by must be a sequence of field names.")
        data_spec = state.get("data_spec")
        metadata_spec = state.get("metadata_spec")
        return cls(
            by=tuple(str(value) for value in raw_by),
            data_spec=data_spec if isinstance(data_spec, Mapping) else None,
            metadata_spec=metadata_spec if isinstance(metadata_spec, Mapping) else None,
        )


def collapse_runs(
    project: "Project",
    source: "BaseArtifact | str",
    *,
    by: str | Sequence[str],
    data: FieldAggregationSpec | None = None,
    metadata: FieldAggregationSpec | None = None,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> "BaseArtifact":
    """Collapse maximal adjacent runs into one span-key row per run.

    Runs are discovered in current artifact position order independently within
    each parent-key prefix.  ``by`` fields are carried automatically as local
    metadata on the output.  ``data`` and ``metadata`` use the same aggregation
    grammar as :meth:`Project.aggregate`, including ``agg(...)``, ``concat(...)``,
    ``literal(...)``, and named reducers.

    The output span key replaces the source leaf key ``k`` with ``k_start`` and
    ``k_end``.  Span membership is the normal TeAL closed leaf-key interval over
    the immediate basis.  Gaps are valid.  No monotonic-key validation is
    imposed; order-preserving sources are recommended when spans are intended to
    represent ordinary contiguous stretches of the source sequence.
    """
    artifact = project.get_artifact(source)
    artifact.require_complete()
    if artifact.artifact_type not in {ArtifactType.TABLE, ArtifactType.JSONL}:
        raise ArtifactError("collapse_runs currently supports table and jsonl sources.")

    by_fields = _normalize_by(by)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    output_label = validate_output_label(output_label)

    source_pk = tuple(str(value) for value in artifact.primary_key)
    if not source_pk:
        raise ArtifactError("collapse_runs source has no primary key.")
    output_pk = tuple(expected_span_key(source_pk))
    parent_cols = source_pk[:-1]
    leaf_col = source_pk[-1]

    data_rules = _normalize_field_spec(data, label="data")
    metadata_rules = _normalize_field_spec(metadata, label="metadata")
    _validate_output_names(output_pk, data_rules, metadata_rules)

    view = project.query.build_artifact_view_sql(
        artifact,
        metadata_mode="full",
        include_data=True,
    )
    resolved_by = _resolve_by_fields(view, by_fields)
    by_output_names = [item[2] for item in resolved_by]
    _validate_run_output_names(
        output_pk=output_pk,
        by_output_names=by_output_names,
        data_names=list(data_rules),
        metadata_names=list(metadata_rules),
    )

    resolved_data = _resolve_rule_sources(view, data_rules, namespace="data")
    resolved_metadata = _resolve_rule_sources(view, metadata_rules, namespace="metadata")

    operator = CollapseRunsOperator(
        by=by_fields,
        data_spec=data,
        metadata_spec=metadata,
    )
    operation = _start_operation(
        project,
        artifact,
        operator=operator,
        output_label=output_label,
        output_pk=output_pk,
        batch_size=batch_size,
        request_state={
            "by": list(by_fields),
            "data": _json_safe_spec(data),
            "metadata": _json_safe_spec(metadata),
        },
        memo=memo,
    )
    writer = operation["writer"]

    try:
        sql, postprocess = _collapse_sql(
            view_sql=view.sql,
            source_pk=source_pk,
            resolved_by=resolved_by,
            resolved_data=resolved_data,
            resolved_metadata=resolved_metadata,
        )

        data_names = list(resolved_data)
        metadata_names = list(resolved_metadata)
        reader = project.query.con.execute(sql).to_arrow_reader(batch_size=batch_size)
        wrote_any = False
        for batch in reader:
            frame = batch.to_pandas().reset_index(drop=True)
            if frame.empty:
                continue
            _apply_postprocessing(frame, postprocess)
            payload: dict[str, Any] = {
                "keys": frame.loc[:, list(output_pk)].copy(),
            }
            if data_names:
                payload["data"] = frame.loc[:, data_names].copy()
            local_metadata = pd.DataFrame(
                {name: frame[name].reset_index(drop=True) for name in by_output_names}
            )
            local_metadata["n_rows"] = frame["__teal_group_n"].astype("int64")
            for name in metadata_names:
                local_metadata[name] = frame[name].reset_index(drop=True)
            payload["metadata"] = local_metadata
            writer.write(payload)
            wrote_any = True

        if not wrote_any:
            payload = {
                "keys": pd.DataFrame(columns=list(output_pk)),
                "metadata": pd.DataFrame(columns=[*by_output_names, "n_rows", *metadata_names]),
            }
            if data_names:
                payload["data"] = pd.DataFrame(columns=data_names)
            writer.write(payload)

        return _finish_operation(
            project,
            artifact,
            output_pk=output_pk,
            operation=operation,
        )
    except Exception as exc:
        _fail_operation(project, operation=operation, exc=exc)
        raise


def _normalize_by(by: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(by, str):
        values = (by,)
    elif isinstance(by, Sequence):
        values = tuple(str(value) for value in by)
    else:
        raise TypeError("by must be a field name or a sequence of field names.")
    if not values or any(not str(value) for value in values):
        raise ValueError("by must contain at least one non-empty field name.")
    if len(set(values)) != len(values):
        raise ValueError("by field names must be unique.")
    return tuple(str(value) for value in values)


def _resolve_by_fields(view: Any, by_fields: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return ``(namespace, resolved_output_name, local_metadata_name)`` per field."""
    resolved: list[tuple[str, str, str]] = []
    for requested in by_fields:
        exact_qualified = [c for c in view.columns if c.qualified_name == requested]
        if len(exact_qualified) == 1:
            column = exact_qualified[0]
        else:
            exact_output = [c for c in view.columns if c.output_name == requested]
            if len(exact_output) == 1:
                column = exact_output[0]
            else:
                base_matches = [c for c in view.columns if c.base_name == requested]
                if len(base_matches) == 1:
                    column = base_matches[0]
                elif len(base_matches) > 1:
                    options = sorted(c.qualified_name for c in base_matches)
                    raise QueryError(
                        f"collapse_runs by field {requested!r} is ambiguous. Use one of: {options}."
                    )
                else:
                    available = sorted({c.base_name for c in view.columns})
                    raise ArtifactError(
                        f"collapse_runs by field {requested!r} is not available; "
                        f"available fields include: {available}."
                    )
        local_name = str(column.base_name)
        resolved.append((str(column.namespace), str(column.output_name), local_name))
    local_names = [item[2] for item in resolved]
    if len(set(local_names)) != len(local_names):
        raise ArtifactError(
            "collapse_runs by fields resolve to duplicate local metadata names; "
            "use unambiguous fields with distinct base names."
        )
    return resolved


def _validate_run_output_names(
    *,
    output_pk: Sequence[str],
    by_output_names: Sequence[str],
    data_names: Sequence[str],
    metadata_names: Sequence[str],
) -> None:
    reserved = set(str(value) for value in output_pk) | {"n_rows"}
    for name in by_output_names:
        if name in reserved or name.startswith("_"):
            raise ArtifactError(
                f"collapse_runs by field output {name!r} collides with a key/reserved field."
            )
    duplicates = set(by_output_names) & (set(data_names) | set(metadata_names))
    if duplicates:
        raise ArtifactError(
            "collapse_runs by fields are carried automatically as local metadata and may not "
            f"also be declared as data/metadata outputs: {sorted(duplicates)}."
        )


def _collapse_sql(
    *,
    view_sql: str,
    source_pk: Sequence[str],
    resolved_by: Sequence[tuple[str, str, str]],
    resolved_data: Mapping[str, Any],
    resolved_metadata: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[str, str]]]:
    parent_cols = [str(value) for value in source_pk[:-1]]
    leaf_col = str(source_pk[-1])
    start_col = f"{leaf_col}_start"
    end_col = f"{leaf_col}_end"
    by_source_names = [item[1] for item in resolved_by]
    by_output_names = [item[2] for item in resolved_by]

    partition = (
        "PARTITION BY " + ", ".join(quote_identifier(name) for name in parent_cols) + " "
        if parent_cols
        else ""
    )
    window = f"{partition}ORDER BY {quote_identifier('_position')}"

    lag_parts = [
        f"LAG({quote_identifier(name)}) OVER ({window}) AS {quote_identifier(f'__teal_prev_{i}')}"
        for i, name in enumerate(by_source_names)
    ]
    lag_sql = ", " + ", ".join(lag_parts) if lag_parts else ""
    boundary_checks = [
        f"{quote_identifier(name)} IS DISTINCT FROM {quote_identifier(f'__teal_prev_{i}')}"
        for i, name in enumerate(by_source_names)
    ]
    boundary_sql = " OR ".join(boundary_checks) if boundary_checks else "FALSE"

    run_by_select = [
        f"FIRST({quote_identifier(name)} ORDER BY {quote_identifier('_position')}) "
        f"AS {quote_identifier(f'__teal_run_by_{i}')}"
        for i, name in enumerate(by_source_names)
    ]
    run_group_cols = [*parent_cols, "__teal_run_id"]
    run_group_sql = ", ".join(quote_identifier(name) for name in run_group_cols)
    run_select_prefix = (
        ", ".join(quote_identifier(name) for name in parent_cols) + ", "
        if parent_cols
        else ""
    )

    join_parts = [
        f"b.{quote_identifier(name)} = r.{quote_identifier(name)}" for name in parent_cols
    ]
    join_parts.extend(
        [
            f"b.{quote_identifier(leaf_col)} >= r.{quote_identifier(start_col)}",
            f"b.{quote_identifier(leaf_col)} <= r.{quote_identifier(end_col)}",
        ]
    )
    join_sql = " AND ".join(join_parts)

    final_select: list[str] = [
        *[f"m.{quote_identifier(name)} AS {quote_identifier(name)}" for name in parent_cols],
        f"m.{quote_identifier(start_col)} AS {quote_identifier(start_col)}",
        f"m.{quote_identifier(end_col)} AS {quote_identifier(end_col)}",
    ]
    postprocess: dict[str, tuple[str, str]] = {}
    source_order = f"m.{quote_identifier('_position')}"

    for i, output_name in enumerate(by_output_names):
        final_select.append(
            f"FIRST(m.{quote_identifier(f'__teal_run_by_{i}')} ORDER BY {source_order}) "
            f"AS {quote_identifier(output_name)}"
        )

    for output_name, rule in resolved_data.items():
        expression, post_method = _rule_sql_expression(rule, source_order=source_order)
        final_select.append(f"{expression} AS {quote_identifier(output_name)}")
        if post_method is not None:
            postprocess[output_name] = (post_method, _concat_separator(rule))

    for output_name, rule in resolved_metadata.items():
        expression, post_method = _rule_sql_expression(rule, source_order=source_order)
        final_select.append(f"{expression} AS {quote_identifier(output_name)}")
        if post_method is not None:
            postprocess[output_name] = (post_method, _concat_separator(rule))

    final_select.extend(
        [
            "COUNT(*) AS __teal_group_n",
            f"MIN(m.{quote_identifier('__teal_run_first_position')}) AS __teal_first_position",
        ]
    )

    final_group_cols = [
        *[f"m.{quote_identifier(name)}" for name in parent_cols],
        f"m.{quote_identifier(start_col)}",
        f"m.{quote_identifier(end_col)}",
        f"m.{quote_identifier('__teal_run_id')}",
    ]

    run_extra = ", " + ", ".join(run_by_select) if run_by_select else ""
    sql = f"""
WITH base AS (
    {view_sql}
),
lagged AS (
    SELECT base.*,
           ROW_NUMBER() OVER ({window}) AS __teal_seq
           {lag_sql}
    FROM base
),
marked AS (
    SELECT lagged.*,
           CASE
               WHEN __teal_seq = 1 OR {boundary_sql} THEN 1
               ELSE 0
           END AS __teal_boundary
    FROM lagged
),
numbered AS (
    SELECT marked.*,
           SUM(__teal_boundary) OVER (
               {partition}ORDER BY {quote_identifier('_position')}
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS __teal_run_id
    FROM marked
),
runs AS (
    SELECT {run_select_prefix}
           __teal_run_id,
           MIN({quote_identifier(leaf_col)}) AS {quote_identifier(start_col)},
           MAX({quote_identifier(leaf_col)}) AS {quote_identifier(end_col)},
           MIN({quote_identifier('_position')}) AS __teal_run_first_position
           {run_extra}
    FROM numbered
    GROUP BY {run_group_sql}
),
members AS (
    SELECT b.*,
           r.{quote_identifier(start_col)} AS {quote_identifier(start_col)},
           r.{quote_identifier(end_col)} AS {quote_identifier(end_col)},
           r.__teal_run_id,
           r.__teal_run_first_position,
           {', '.join(f'r.{quote_identifier(f"__teal_run_by_{i}")} AS {quote_identifier(f"__teal_run_by_{i}")}' for i in range(len(by_source_names)))}
    FROM runs r
    JOIN base b ON {join_sql}
)
SELECT {', '.join(final_select)}
FROM members m
GROUP BY {', '.join(final_group_cols)}
ORDER BY __teal_first_position
""".strip()
    return sql, postprocess


def _start_operation(
    project: "Project",
    artifact: "BaseArtifact",
    *,
    operator: CollapseRunsOperator,
    output_label: str,
    output_pk: Sequence[str],
    batch_size: int,
    request_state: Mapping[str, Any],
    memo: str | None,
) -> dict[str, Any]:
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="translate",
        snapshot_status="pending",
    )
    try:
        operator.save_to_dir(project.storage.operator_dir(operator_id), operator_id=operator_id)
        project.catalog.mark_operator_serialized(operator_id)
    except Exception:
        project.catalog.mark_operator_snapshot_failed(operator_id)
        raise

    operation_id = next_id(project.storage.manifest_path, "operation")
    operation_dir = project.storage.operation_dir(operation_id)
    operation_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="translate",
        operator_id=operator_id,
        status="incomplete",
    )
    project.catalog.add_operation_source(operation_id, DEFAULT_SOURCE_LABEL, artifact.artifact_id)

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=ArtifactType.TABLE,
        label=output_label,
        lineage_mode="span_key",
        status="incomplete",
        basis_artifact_ids=(artifact.artifact_id,),
    )
    project.catalog.add_operation_output(operation_id, output_label, artifact_id, ordinal=0)
    writer = create_artifact_writer(
        artifact_type=ArtifactType.TABLE,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=output_label,
        operation_id=operation_id,
        lineage_mode="span_key",
        basis_artifact_ids=(artifact.artifact_id,),
    )

    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "translate",
        "operator_id": operator_id,
        "kind": "collapse_runs",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: artifact.artifact_id},
        "output_artifact_ids": {output_label: artifact_id},
        "request": {
            "output_label": output_label,
            "batch_size": batch_size,
            "batch_unit": "collapsed_runs",
            "output_primary_key": list(output_pk),
            **dict(request_state),
        },
    }
    _write_descriptor(operation_dir, descriptor)
    if memo is not None:
        project.catalog.add_memo(target_type="operation", target_id=operation_id, body=memo)

    return {
        "operator_id": operator_id,
        "operation_id": operation_id,
        "operation_dir": operation_dir,
        "artifact_id": artifact_id,
        "writer": writer,
        "descriptor": descriptor,
    }


def _finish_operation(
    project: "Project",
    artifact: "BaseArtifact",
    *,
    output_pk: Sequence[str],
    operation: Mapping[str, Any],
) -> "BaseArtifact":
    writer = operation["writer"]
    writer.finalize()
    validate_primary_key_relationship(
        basis_keys=[artifact.primary_key],
        output_key=output_pk,
        lineage_mode="span_key",
    )
    artifact_id = str(operation["artifact_id"])
    operation_id = str(operation["operation_id"])
    project.catalog.mark_artifact_complete(artifact_id)
    project.catalog.mark_operation_complete(operation_id)
    descriptor = operation["descriptor"]
    descriptor["status"] = "complete"
    _write_descriptor(operation["operation_dir"], descriptor)
    project.storage.touch_manifest()
    project.query.clear_cache()
    return project.get_artifact(artifact_id)


def _fail_operation(
    project: "Project",
    *,
    operation: Mapping[str, Any],
    exc: BaseException,
) -> None:
    try:
        operation["writer"].mark_failed(exc)
    except Exception:
        pass
    try:
        project.catalog.mark_artifact_failed(str(operation["artifact_id"]))
    except Exception:
        pass
    try:
        project.catalog.mark_operation_failed(str(operation["operation_id"]), exc)
    except Exception:
        pass
    descriptor = operation["descriptor"]
    descriptor["status"] = "failed"
    descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
    _write_descriptor(operation["operation_dir"], descriptor)


def _write_descriptor(operation_dir: Any, descriptor: Mapping[str, Any]) -> None:
    (operation_dir / "operation.json").write_text(
        json.dumps(dict(descriptor), indent=2, sort_keys=True),
        encoding="utf-8",
    )
