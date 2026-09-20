"""Reduced-key aggregation across relational and matrix TeAL artifacts.

The public :func:`aggregate` operation reduces a source artifact from a finer
primary-key grain to a retained primary-key prefix.  Relational sources may
aggregate data and metadata field-by-field.  Matrix sources deliberately apply
one reducer uniformly across the full matrix representation while still
allowing field-level metadata aggregation.

Batching is group-safe by construction.  Relational aggregation happens inside
DuckDB before result batches are emitted.  Matrix aggregation first discovers
complete retained-key groups in DuckDB, batches those groups, and only then
loads all source rows belonging to each selected group batch.  A retained-key
group is therefore never split across aggregation batches.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import (
    ArtifactError,
    MetadataAggregationError,
    QueryError,
)
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.lineage import validate_primary_key_relationship
from text_analysis_lab.core.metadata_aggregation import normalize_aggregation_spec
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.query import quote_identifier
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, ArtifactType
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


ReducerName = Literal[
    "first",
    "last",
    "first_non_null",
    "last_non_null",
    "min",
    "max",
    "mean",
    "median",
    "sum",
    "count",
    "nunique",
    "any",
    "all",
    "mode",
    "all_equal",
    "concat",
    "unique",
]

_SUPPORTED_METHODS = frozenset(
    {
        "first",
        "last",
        "first_non_null",
        "last_non_null",
        "min",
        "max",
        "mean",
        "median",
        "sum",
        "count",
        "nunique",
        "any",
        "all",
        "mode",
        "all_equal",
        "concat",
        "unique",
    }
)
_MATRIX_METHODS = frozenset({"sum", "mean"})
_POSTPROCESS_METHODS = frozenset({"mode", "concat", "unique"})


@dataclass(frozen=True)
class ConcatReducer:
    """Parameterized string concatenation reducer."""

    separator: str = " "

    def __post_init__(self) -> None:
        if not isinstance(self.separator, str):
            raise TypeError("concat separator must be a string.")


@dataclass(frozen=True)
class AggregateField:
    """Aggregate ``source`` into an output field using ``reducer``."""

    source: str
    reducer: str | ConcatReducer

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("agg source must be a non-empty field name.")
        _normalize_reducer(self.reducer)


@dataclass(frozen=True)
class LiteralValue:
    """Emit one literal value for every retained-key group."""

    value: Any

    def __post_init__(self) -> None:
        _validate_literal_value(self.value)


def agg(source: str, reducer: str | ConcatReducer) -> AggregateField:
    """Describe a renamed or explicitly sourced aggregation.

    Example::

        data={
            "probability_sum": agg("probability", "sum"),
            "probability_mean": agg("probability", "mean"),
        }
    """

    return AggregateField(source=str(source), reducer=reducer)


def literal(value: Any) -> LiteralValue:
    """Describe a literal output value repeated once per aggregate group."""

    return LiteralValue(value=value)


def concat(separator: str = " ") -> ConcatReducer:
    """Return a parameterized ordered string-concatenation reducer."""

    return ConcatReducer(separator=separator)


@dataclass(frozen=True)
class _OutputRule:
    output_name: str
    source_name: str | None = None
    reducer: str | ConcatReducer | None = None
    literal_value: Any = None
    is_literal: bool = False
    resolved_source_name: str | None = None

    def resolved(self, source_name: str) -> _OutputRule:
        return _OutputRule(
            output_name=self.output_name,
            source_name=self.source_name,
            reducer=self.reducer,
            literal_value=self.literal_value,
            is_literal=self.is_literal,
            resolved_source_name=str(source_name),
        )


FieldAggregationSpec = Mapping[
    str,
    str | ConcatReducer | AggregateField | LiteralValue,
]
LegacyAggregationSpec = Mapping[str, str | Sequence[str] | Mapping[str, str]]


class AggregateOperator(BaseOperator):
    """Frozen descriptor for a reduced-key aggregation."""

    operation_type = "translate"

    def __init__(
        self,
        *,
        retained_key: Sequence[str] = (),
        data_spec: Mapping[str, Any] | str | None = None,
        metadata_spec: Mapping[str, Any] | None = None,
        legacy_aggregations: Mapping[str, Any] | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.retained_key = tuple(str(value) for value in retained_key)
        self.data_spec = _json_safe_spec(data_spec)
        self.metadata_spec = _json_safe_spec(metadata_spec)
        self.legacy_aggregations = _json_safe_legacy_aggregations(
            legacy_aggregations or {}
        )

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        if len(sources) != 1:
            raise ArtifactError(
                "AggregateOperator expects exactly one source artifact."
            )
        source = next(iter(sources.values()))
        artifact_type = (
            source.artifact_type.value
            if source.artifact_type
            in {ArtifactType.DENSE_MATRIX, ArtifactType.SPARSE_MATRIX}
            else ArtifactType.TABLE.value
        )
        return OutputSpec(
            artifact_type=artifact_type,
            lineage_mode="reduced_key",
            basis_labels="source",
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "retained_key": list(self.retained_key),
            "data_spec": self.data_spec,
            "metadata_spec": self.metadata_spec,
            "legacy_aggregations": dict(self.legacy_aggregations),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> AggregateOperator:
        # Round-27-and-earlier snapshots stored only ``aggregations``.  Preserve
        # deserialization compatibility even though new calls use data/metadata.
        raw_legacy = state.get("legacy_aggregations", state.get("aggregations", {}))
        return cls(
            retained_key=tuple(str(value) for value in state.get("retained_key", ())),
            data_spec=state.get("data_spec"),
            metadata_spec=state.get("metadata_spec"),
            legacy_aggregations=raw_legacy if isinstance(raw_legacy, Mapping) else {},
        )


def aggregate(
    project: Project,
    source: BaseArtifact | str,
    *,
    to_key: str | Sequence[str],
    data: str | FieldAggregationSpec | None = None,
    metadata: FieldAggregationSpec | None = None,
    aggregations: LegacyAggregationSpec | None = None,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> BaseArtifact:
    """Aggregate an artifact to a retained primary-key prefix.

    ``to_key`` names the last primary-key field to retain, or supplies the
    retained prefix explicitly.  The retained key must be a non-empty proper
    prefix of the source key.

    Relational sources (table/jsonl)
    --------------------------------
    ``data`` and ``metadata`` use the same output-centric field grammar::

        data={
            "probability": "mean",  # shorthand: source and output share a name
            "probability_sum": agg("probability", "sum"),
            "course_number": literal(12),
        }

    Matrix sources
    --------------
    Matrix data is intentionally homogeneous: ``data`` must be ``"sum"`` or
    ``"mean"`` and applies to every feature dimension.  Metadata may still use
    the field-level grammar above.

    ``aggregations`` is the pre-Round-27 source-centric relational syntax and is
    retained for backward compatibility.  It cannot be combined with ``data``
    or ``metadata``.

    Batching is group-safe.  ``batch_size`` counts *retained-key groups*, not
    arbitrary source rows.  Relational groups are reduced by DuckDB before
    batching.  Matrix groups are discovered first and source rows are loaded
    only after complete groups have been assigned to a batch.
    """

    artifact = project.get_artifact(source)
    artifact.require_complete()

    source_pk = tuple(str(value) for value in artifact.primary_key)
    retained_key = _resolve_retained_key(source_pk, to_key)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if aggregations is not None and (data is not None or metadata is not None):
        raise TypeError(
            "aggregate cannot combine legacy aggregations= with the new data=/metadata= API."
        )

    legacy = aggregations if aggregations is not None else None
    if legacy is not None:
        data_rules = _normalize_legacy_data_spec(legacy)
        metadata_rules: dict[str, _OutputRule] = {}
    elif artifact.artifact_type in {ArtifactType.TABLE, ArtifactType.JSONL}:
        if isinstance(data, str):
            raise TypeError(
                "For table/jsonl sources, data= must be a field mapping, not a single reducer string."
            )
        data_rules = _normalize_field_spec(data, label="data")
        metadata_rules = _normalize_field_spec(metadata, label="metadata")
    elif artifact.artifact_type in {
        ArtifactType.DENSE_MATRIX,
        ArtifactType.SPARSE_MATRIX,
    }:
        if not isinstance(data, str) or data not in _MATRIX_METHODS:
            raise ArtifactError(
                "Matrix aggregation requires data='sum' or data='mean'; the reducer applies "
                "uniformly to every matrix feature."
            )
        data_rules = {}
        metadata_rules = _normalize_field_spec(metadata, label="metadata")
    else:
        raise ArtifactError(
            "aggregate supports table, jsonl, dense_matrix, and sparse_matrix sources."
        )

    if artifact.artifact_type in {ArtifactType.TABLE, ArtifactType.JSONL} and not (
        data_rules or metadata_rules
    ):
        raise ArtifactError(
            "Relational aggregate requires at least one data or metadata aggregation rule."
        )

    _validate_output_names(retained_key, data_rules, metadata_rules)

    if artifact.artifact_type in {ArtifactType.TABLE, ArtifactType.JSONL}:
        return _aggregate_relational(
            project,
            artifact,
            retained_key=retained_key,
            data_rules=data_rules,
            metadata_rules=metadata_rules,
            raw_data_spec=data,
            raw_metadata_spec=metadata,
            legacy_aggregations=legacy,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )

    return _aggregate_matrix(
        project,
        artifact,
        retained_key=retained_key,
        pooling=str(data),
        metadata_rules=metadata_rules,
        raw_metadata_spec=metadata,
        output_label=output_label,
        batch_size=batch_size,
        memo=memo,
    )


def _aggregate_relational(
    project: Project,
    artifact: BaseArtifact,
    *,
    retained_key: tuple[str, ...],
    data_rules: Mapping[str, _OutputRule],
    metadata_rules: Mapping[str, _OutputRule],
    raw_data_spec: Any,
    raw_metadata_spec: Any,
    legacy_aggregations: LegacyAggregationSpec | None,
    output_label: str,
    batch_size: int,
    memo: str | None,
) -> BaseArtifact:
    label = validate_output_label(output_label)

    include_data = any(not rule.is_literal for rule in data_rules.values())
    view = project.query.build_artifact_view_sql(
        artifact,
        metadata_mode="full" if metadata_rules else "none",
        include_data=include_data,
    )
    resolved_data = _resolve_rule_sources(view, data_rules, namespace="data")
    resolved_metadata = _resolve_rule_sources(
        view, metadata_rules, namespace="metadata"
    )

    operator = AggregateOperator(
        retained_key=retained_key,
        data_spec=raw_data_spec,
        metadata_spec=raw_metadata_spec,
        legacy_aggregations=legacy_aggregations,
    )
    operation = _start_operation(
        project,
        artifact,
        operator=operator,
        output_artifact_type=ArtifactType.TABLE,
        output_label=label,
        retained_key=retained_key,
        batch_size=batch_size,
        request_state={
            "data": _json_safe_spec(raw_data_spec),
            "metadata": _json_safe_spec(raw_metadata_spec),
            "aggregations": _json_safe_legacy_aggregations(legacy_aggregations or {}),
        },
        memo=memo,
    )
    writer = operation["writer"]
    descriptor = operation["descriptor"]

    data_names = list(resolved_data)
    metadata_names = list(resolved_metadata)
    source_order = ", ".join(quote_identifier(name) for name in artifact.primary_key)
    group_exprs = [quote_identifier(name) for name in retained_key]

    try:
        select_exprs = list(group_exprs)
        postprocess: dict[str, tuple[str, str]] = {}

        for output_name, rule in resolved_data.items():
            expression, post_method = _rule_sql_expression(
                rule, source_order=source_order
            )
            select_exprs.append(f"{expression} AS {quote_identifier(output_name)}")
            if post_method is not None:
                postprocess[output_name] = (post_method, _concat_separator(rule))

        for output_name, rule in resolved_metadata.items():
            expression, post_method = _rule_sql_expression(
                rule, source_order=source_order
            )
            select_exprs.append(f"{expression} AS {quote_identifier(output_name)}")
            if post_method is not None:
                postprocess[output_name] = (post_method, _concat_separator(rule))

        select_exprs.append("COUNT(*) AS __teal_group_n")
        select_exprs.append("MIN(_position) AS __teal_first_position")
        sql = (
            "SELECT "
            + ", ".join(select_exprs)
            + f" FROM ({view.sql}) AS base GROUP BY "
            + ", ".join(group_exprs)
            + " ORDER BY __teal_first_position"
        )

        # Important: DuckDB has already reduced source rows to one row per
        # retained-key group before Arrow batching begins.  No group can cross a
        # batch boundary here.
        reader = project.query.con.execute(sql).to_arrow_reader(batch_size=batch_size)
        wrote_any = False
        for batch in reader:
            frame = batch.to_pandas().reset_index(drop=True)
            if frame.empty:
                continue
            _apply_postprocessing(frame, postprocess)
            keys = frame.loc[:, list(retained_key)].copy()
            payload: dict[str, Any] = {"keys": keys}
            if data_names:
                payload["data"] = frame.loc[:, data_names].copy()
            metadata_frame = pd.DataFrame(
                {"n_rows": frame["__teal_group_n"].astype("int64")}
            )
            if metadata_names:
                for name in metadata_names:
                    metadata_frame[name] = frame[name].reset_index(drop=True)
            payload["metadata"] = metadata_frame
            writer.write(payload)
            wrote_any = True

        if not wrote_any:
            payload = {
                "keys": pd.DataFrame(columns=list(retained_key)),
                "metadata": pd.DataFrame(columns=["n_rows", *metadata_names]),
            }
            if data_names:
                payload["data"] = pd.DataFrame(columns=data_names)
            writer.write(payload)

        return _finish_operation(
            project,
            artifact,
            retained_key=retained_key,
            operation=operation,
        )
    except Exception as exc:
        _fail_operation(project, operation=operation, exc=exc)
        raise


def _aggregate_matrix(
    project: Project,
    artifact: BaseArtifact,
    *,
    retained_key: tuple[str, ...],
    pooling: str,
    metadata_rules: Mapping[str, _OutputRule],
    raw_metadata_spec: Any,
    output_label: str,
    batch_size: int,
    memo: str | None,
) -> BaseArtifact:
    if pooling not in _MATRIX_METHODS:
        raise ArtifactError(f"Unsupported matrix reducer {pooling!r}.")

    label = validate_output_label(output_label)
    metadata_view = project.query.build_artifact_view_sql(
        artifact,
        metadata_mode="full" if metadata_rules else "none",
        include_data=False,
    )
    resolved_metadata = _resolve_rule_sources(
        metadata_view, metadata_rules, namespace="metadata"
    )
    metadata_names = list(resolved_metadata)
    metadata_source_names = list(
        dict.fromkeys(
            rule.resolved_source_name
            for rule in resolved_metadata.values()
            if not rule.is_literal and rule.resolved_source_name is not None
        )
    )

    operator = AggregateOperator(
        retained_key=retained_key,
        data_spec=pooling,
        metadata_spec=raw_metadata_spec,
    )
    operation = _start_operation(
        project,
        artifact,
        operator=operator,
        output_artifact_type=artifact.artifact_type,
        output_label=label,
        retained_key=retained_key,
        batch_size=batch_size,
        request_state={
            "data": pooling,
            "metadata": _json_safe_spec(raw_metadata_spec),
        },
        memo=memo,
    )
    writer = operation["writer"]

    key_view = project.query.build_artifact_view_sql(
        artifact,
        metadata_mode="none",
        include_data=False,
    )
    group_exprs = [quote_identifier(name) for name in retained_key]
    group_sql = (
        "SELECT "
        + ", ".join(group_exprs)
        + ", COUNT(*) AS __teal_group_n, MIN(_position) AS __teal_first_position"
        + f" FROM ({key_view.sql}) AS base GROUP BY "
        + ", ".join(group_exprs)
    )
    group_table = f"__teal_aggregate_units_{uuid4().hex}"

    try:
        # Materialize only the compact retained-key unit relation inside DuckDB.
        # We do *not* stream arbitrary source rows and hope group boundaries line
        # up with batch boundaries.  The group table is the batching domain.
        project.query.con.execute(
            f"CREATE TEMP TABLE {quote_identifier(group_table)} AS "
            "SELECT grouped.*, "
            "ROW_NUMBER() OVER (ORDER BY __teal_first_position) - 1 AS __teal_group_order "
            f"FROM ({group_sql}) AS grouped"
        )
        n_groups = int(
            project.query.con.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(group_table)}"
            ).fetchone()[0]
        )
        wrote_any = False
        matrix_columns = list(artifact.get_data_columns())

        for group_start in range(0, n_groups, batch_size):
            group_stop = min(group_start + batch_size, n_groups)
            groups = (
                project.query.con.execute(
                    "SELECT * FROM "
                    + quote_identifier(group_table)
                    + f" WHERE __teal_group_order >= {group_start} "
                    + f"AND __teal_group_order < {group_stop} "
                    + "ORDER BY __teal_group_order"
                )
                .fetchdf()
                .reset_index(drop=True)
            )
            if groups.empty:
                continue

            group_keys = groups.loc[:, list(retained_key)].copy()
            group_counts = groups["__teal_group_n"].astype("int64").to_numpy()
            child_frame = _matrix_child_rows_for_groups(
                project,
                artifact,
                key_view_sql=key_view.sql,
                retained_key=retained_key,
                groups=group_keys,
            )
            positions = child_frame["_position"].astype("int64").tolist()
            if len(positions) != int(group_counts.sum()):
                raise ArtifactError(
                    "Matrix aggregation group discovery mismatch: fetched source row count "
                    f"{len(positions)} != expected {int(group_counts.sum())}."
                )

            matrix = artifact._data_native_for_positions(positions, data_columns=True)
            pooled = _pool_matrix_by_complete_groups(
                matrix,
                group_counts=group_counts,
                pooling=pooling,
            )

            metadata_frame = pd.DataFrame({"n_rows": group_counts})
            if resolved_metadata:
                source_metadata = None
                if metadata_source_names:
                    source_metadata = artifact.query(
                        key_columns=False,
                        data_columns=False,
                        metadata_columns=metadata_source_names,
                        metadata_mode="full",
                        positions=positions,
                        form="table",
                    ).reset_index(drop=True)
                    if len(source_metadata) != len(positions):
                        raise ArtifactError(
                            "Matrix aggregation metadata query did not preserve source row count."
                        )
                reduced_metadata = _aggregate_group_metadata(
                    resolved_metadata,
                    source_metadata=source_metadata,
                    group_counts=group_counts,
                )
                for name in metadata_names:
                    metadata_frame[name] = reduced_metadata[name]

            writer.write(
                {
                    "keys": group_keys,
                    "metadata": metadata_frame,
                    "data": {"values": pooled, "columns": matrix_columns},
                }
            )
            wrote_any = True

        if not wrote_any:
            raise ArtifactError("Matrix aggregate cannot aggregate an empty artifact.")

        return _finish_operation(
            project,
            artifact,
            retained_key=retained_key,
            operation=operation,
        )
    except Exception as exc:
        _fail_operation(project, operation=operation, exc=exc)
        raise
    finally:
        try:
            project.query.con.execute(
                f"DROP TABLE IF EXISTS {quote_identifier(group_table)}"
            )
        except Exception:
            pass


def _matrix_child_rows_for_groups(
    project: Project,
    artifact: BaseArtifact,
    *,
    key_view_sql: str,
    retained_key: Sequence[str],
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Return every child row for a complete batch of retained-key groups.

    The temporary table contains *groups*, not arbitrary source positions.  The
    join then retrieves all children belonging to those groups and orders each
    group by the complete integer primary key.  This is the group-safe batching
    boundary required by aggregation.
    """

    temp = groups.reset_index(drop=True).copy()
    temp["__teal_group_order"] = np.arange(len(temp), dtype="int64")
    temp_name = f"__teal_aggregate_groups_{id(temp)}"
    project.query.con.register(temp_name, temp)
    try:
        join = " AND ".join(
            f"base.{quote_identifier(name)} = g.{quote_identifier(name)}"
            for name in retained_key
        )
        pk_order = ", ".join(
            f"base.{quote_identifier(name)}" for name in artifact.primary_key
        )
        sql = (
            "SELECT base._position, g.__teal_group_order, "
            + ", ".join(
                f"base.{quote_identifier(name)} AS {quote_identifier(name)}"
                for name in artifact.primary_key
            )
            + f" FROM ({key_view_sql}) AS base JOIN {quote_identifier(temp_name)} AS g ON {join}"
            + f" ORDER BY g.__teal_group_order, {pk_order}"
        )
        return project.query.con.execute(sql).fetchdf().reset_index(drop=True)
    finally:
        project.query.con.unregister(temp_name)


def _pool_matrix_by_complete_groups(
    matrix: Any,
    *,
    group_counts: np.ndarray,
    pooling: str,
) -> Any:
    counts = np.asarray(group_counts, dtype="int64")
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts <= 0):
        raise ArtifactError("Matrix aggregation requires positive group row counts.")
    expected_rows = int(counts.sum())
    if getattr(matrix, "shape", (None,))[0] != expected_rows:
        raise ArtifactError(
            "Matrix aggregation source matrix row count does not match complete group rows."
        )

    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a package dependency
        sparse = None

    if sparse is not None and sparse.issparse(matrix):
        # Sparse indicator multiplication keeps sparse count matrices sparse; it
        # never constructs one dense vector per output group.
        n_groups = len(counts)
        group_codes = np.repeat(np.arange(n_groups, dtype="int64"), counts)
        row_indices = np.arange(expected_rows, dtype="int64")
        indicator = sparse.csr_matrix(
            (
                np.ones(expected_rows, dtype="float64"),
                (group_codes, row_indices),
            ),
            shape=(n_groups, expected_rows),
        )
        pooled = (indicator @ matrix).tocsr()
        if pooling == "mean":
            scale = sparse.diags(1.0 / counts.astype("float64"), format="csr")
            pooled = (scale @ pooled).tocsr()
        return pooled

    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ArtifactError(
            "Dense matrix aggregation requires a two-dimensional matrix."
        )
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype="int64")))
    pooled = np.add.reduceat(values, starts, axis=0)
    if pooling == "mean":
        pooled = pooled / counts.reshape(-1, 1)
    return pooled


def _aggregate_group_metadata(
    rules: Mapping[str, _OutputRule],
    *,
    source_metadata: pd.DataFrame | None,
    group_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = 0
    for count in np.asarray(group_counts, dtype="int64"):
        stop = start + int(count)
        row: dict[str, Any] = {}
        for output_name, rule in rules.items():
            if rule.is_literal:
                row[output_name] = rule.literal_value
                continue
            if source_metadata is None or rule.resolved_source_name is None:
                raise ArtifactError(
                    f"Missing source metadata for aggregate output {output_name!r}."
                )
            values = (
                source_metadata[rule.resolved_source_name].iloc[start:stop].tolist()
            )
            row[output_name] = _reduce_ordered_values(
                values,
                rule.reducer,
            )
        rows.append(row)
        start = stop
    return pd.DataFrame(rows, columns=list(rules))


def _normalize_field_spec(
    spec: FieldAggregationSpec | None,
    *,
    label: str,
) -> dict[str, _OutputRule]:
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"aggregate {label}= must be a mapping of output fields to reducers."
        )

    out: dict[str, _OutputRule] = {}
    for raw_output, value in spec.items():
        output_name = str(raw_output)
        if not output_name:
            raise ValueError(
                f"aggregate {label} output names must be non-empty strings."
            )
        if isinstance(value, LiteralValue):
            rule = _OutputRule(
                output_name=output_name,
                literal_value=value.value,
                is_literal=True,
            )
        elif isinstance(value, AggregateField):
            rule = _OutputRule(
                output_name=output_name,
                source_name=value.source,
                reducer=_normalize_reducer(value.reducer),
            )
        elif isinstance(value, (str, ConcatReducer)):
            rule = _OutputRule(
                output_name=output_name,
                source_name=output_name,
                reducer=_normalize_reducer(value),
            )
        elif callable(value):
            raise TypeError(
                "Callable aggregation reducers are reserved for a future implementation; "
                "use a built-in reducer string, agg(...), concat(...), or literal(...)."
            )
        else:
            raise TypeError(
                f"Invalid aggregate {label} rule for {output_name!r}: {value!r}. "
                "Use a reducer string, agg(...), concat(...), or literal(...)."
            )
        out[output_name] = rule
    return out


def _normalize_legacy_data_spec(spec: LegacyAggregationSpec) -> dict[str, _OutputRule]:
    normalized = normalize_aggregation_spec(spec)
    out: dict[str, _OutputRule] = {}
    for input_name, outputs in normalized.items():
        for output_name, method in outputs.items():
            if output_name in out:
                raise ArtifactError(
                    f"aggregate output field {output_name!r} is declared more than once."
                )
            out[output_name] = _OutputRule(
                output_name=output_name,
                source_name=input_name,
                reducer=_normalize_reducer(method),
            )
    return out


def _normalize_reducer(value: str | ConcatReducer) -> str | ConcatReducer:
    if isinstance(value, ConcatReducer):
        return value
    method = str(value)
    if method not in _SUPPORTED_METHODS:
        raise MetadataAggregationError(
            f"aggregate method {method!r} is not supported; supported={sorted(_SUPPORTED_METHODS)}."
        )
    return method


def _validate_output_names(
    retained_key: Sequence[str],
    data_rules: Mapping[str, _OutputRule],
    metadata_rules: Mapping[str, _OutputRule],
) -> None:
    for namespace, rules in (("data", data_rules), ("metadata", metadata_rules)):
        for output_name in rules:
            if output_name in retained_key or output_name.startswith("_"):
                raise ArtifactError(
                    f"aggregate {namespace} output field {output_name!r} collides with a retained/reserved field."
                )
            if output_name == "n_rows":
                raise ArtifactError(
                    "aggregate output field 'n_rows' is reserved for TeAL's group-size metadata."
                )


def _resolve_rule_sources(
    view: Any, rules: Mapping[str, _OutputRule], *, namespace: str
) -> dict[str, _OutputRule]:
    resolved: dict[str, _OutputRule] = {}
    columns = [column for column in view.columns if column.namespace == namespace]
    for output_name, rule in rules.items():
        if rule.is_literal:
            resolved[output_name] = rule
            continue
        assert rule.source_name is not None
        name = rule.source_name
        exact_qualified = [
            column for column in columns if column.qualified_name == name
        ]
        if len(exact_qualified) == 1:
            resolved[output_name] = rule.resolved(exact_qualified[0].output_name)
            continue
        exact_output = [column for column in columns if column.output_name == name]
        if len(exact_output) == 1:
            resolved[output_name] = rule.resolved(exact_output[0].output_name)
            continue
        base_matches = [column for column in columns if column.base_name == name]
        if len(base_matches) == 1:
            resolved[output_name] = rule.resolved(base_matches[0].output_name)
            continue
        if len(base_matches) > 1:
            options = [column.qualified_name for column in base_matches]
            raise QueryError(
                f"Requested aggregate {namespace} source {name!r} is ambiguous. Use one of: {options}."
            )
        available = sorted({column.output_name for column in columns})
        raise ArtifactError(
            f"aggregate {namespace} source field {name!r} is not available; available={available}."
        )
    return resolved


def _rule_sql_expression(
    rule: _OutputRule, *, source_order: str
) -> tuple[str, str | None]:
    if rule.is_literal:
        return _sql_literal(rule.literal_value), None
    if rule.resolved_source_name is None or rule.reducer is None:
        raise ArtifactError(f"Malformed aggregation rule for {rule.output_name!r}.")
    column = quote_identifier(rule.resolved_source_name)
    method = rule.reducer
    normalized_method = "concat" if isinstance(method, ConcatReducer) else str(method)
    if normalized_method in _POSTPROCESS_METHODS:
        # Preserve exact primary-key order in an ordered list and apply the
        # reducer after DuckDB has already collapsed the group to one output row.
        return f"LIST({column} ORDER BY {source_order})", normalized_method
    return _sql_aggregate(column, normalized_method, source_order=source_order), None


def _concat_separator(rule: _OutputRule) -> str:
    if isinstance(rule.reducer, ConcatReducer):
        return rule.reducer.separator
    return " "


def _apply_postprocessing(
    frame: pd.DataFrame,
    postprocess: Mapping[str, tuple[str, str]],
) -> None:
    for output_name, (method, separator) in postprocess.items():
        frame[output_name] = frame[output_name].map(
            lambda raw: _reduce_ordered_values(
                _as_list(raw),
                ConcatReducer(separator) if method == "concat" else method,
            )
        )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def _reduce_ordered_values(
    values: Sequence[Any], reducer: str | ConcatReducer | None
) -> Any:
    if reducer is None:
        raise MetadataAggregationError("Aggregation reducer is missing.")
    method = "concat" if isinstance(reducer, ConcatReducer) else str(reducer)
    separator = reducer.separator if isinstance(reducer, ConcatReducer) else " "
    values_list = list(values)
    non_null = [value for value in values_list if not _is_null(value)]

    if method == "first":
        return values_list[0] if values_list else pd.NA
    if method == "last":
        return values_list[-1] if values_list else pd.NA
    if method == "first_non_null":
        return non_null[0] if non_null else pd.NA
    if method == "last_non_null":
        return non_null[-1] if non_null else pd.NA
    if method == "min":
        return min(non_null) if non_null else pd.NA
    if method == "max":
        return max(non_null) if non_null else pd.NA
    if method == "mean":
        numeric = pd.to_numeric(pd.Series(non_null), errors="coerce")
        return numeric.mean() if len(numeric) else pd.NA
    if method == "median":
        numeric = pd.to_numeric(pd.Series(non_null), errors="coerce")
        return numeric.median() if len(numeric) else pd.NA
    if method == "sum":
        numeric = pd.to_numeric(pd.Series(non_null), errors="coerce")
        return numeric.sum() if len(numeric) else 0
    if method == "count":
        return len(non_null)
    if method == "nunique":
        return len(_ordered_unique(non_null))
    if method == "any":
        return any(bool(value) for value in non_null) if non_null else False
    if method == "all":
        return all(bool(value) for value in non_null) if non_null else False
    if method == "mode":
        if not non_null:
            return pd.NA
        unique = _ordered_unique(non_null)
        counts = [
            sum(_values_equal(value, other) for other in non_null) for value in unique
        ]
        return unique[int(np.argmax(np.asarray(counts, dtype="int64")))]
    if method == "all_equal":
        return len(_ordered_unique(non_null)) <= 1
    if method == "concat":
        return separator.join(str(value) for value in non_null)
    if method == "unique":
        return _ordered_unique(non_null)
    raise MetadataAggregationError(f"Unknown aggregate method {method!r}.")


def _ordered_unique(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if not any(_values_equal(value, existing) for existing in out):
            out.append(value)
    return out


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        if hasattr(result, "all"):
            return bool(result.all())
    except Exception:
        pass
    return repr(left) == repr(right)


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
    return False


def _resolve_retained_key(
    source_pk: Sequence[str], to_key: str | Sequence[str]
) -> tuple[str, ...]:
    source = tuple(str(value) for value in source_pk)
    if not source:
        raise ArtifactError("aggregate source has no primary key.")
    if isinstance(to_key, str):
        if not to_key:
            raise ValueError("to_key must be a non-empty primary-key field name.")
        if to_key not in source:
            raise ArtifactError(
                f"aggregate to_key {to_key!r} is not in source primary key {list(source)}."
            )
        retained = source[: source.index(to_key) + 1]
    else:
        retained = tuple(str(value) for value in to_key)
        if not retained:
            raise ValueError("to_key sequence must contain at least one field.")
    if len(retained) >= len(source) or source[: len(retained)] != retained:
        raise ArtifactError(
            "aggregate to_key must identify a non-empty proper prefix of the source "
            f"primary key {list(source)}; got {list(retained)}."
        )
    return retained


def _sql_aggregate(column: str, method: str, *, source_order: str) -> str:
    method = str(method)
    if method == "first":
        return f"FIRST({column} ORDER BY {source_order})"
    if method == "last":
        return f"LAST({column} ORDER BY {source_order})"
    if method == "first_non_null":
        return f"FIRST({column} ORDER BY {source_order}) FILTER (WHERE {column} IS NOT NULL)"
    if method == "last_non_null":
        return f"LAST({column} ORDER BY {source_order}) FILTER (WHERE {column} IS NOT NULL)"
    if method == "min":
        return f"MIN({column})"
    if method == "max":
        return f"MAX({column})"
    if method == "mean":
        return f"AVG(TRY_CAST({column} AS DOUBLE))"
    if method == "median":
        return f"MEDIAN(TRY_CAST({column} AS DOUBLE))"
    if method == "sum":
        return f"COALESCE(SUM(TRY_CAST({column} AS DOUBLE)), 0.0)"
    if method == "count":
        return f"COUNT({column})"
    if method == "nunique":
        return f"COUNT(DISTINCT {column})"
    if method == "any":
        return f"COALESCE(BOOL_OR(TRY_CAST({column} AS BOOLEAN)), FALSE)"
    if method == "all":
        return f"COALESCE(BOOL_AND(TRY_CAST({column} AS BOOLEAN)), FALSE)"
    if method == "all_equal":
        return f"COUNT(DISTINCT {column}) <= 1"
    raise MetadataAggregationError(f"Unknown aggregate method {method!r}.")


def _sql_literal(value: Any) -> str:
    _validate_literal_value(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("literal float values must be finite.")
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _validate_literal_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("literal float values must be finite.")
        return
    raise TypeError(
        "literal(...) currently supports only None, str, bool, int, and finite float values."
    )


def _json_safe_spec(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, ConcatReducer):
        return {"kind": "concat", "separator": value.separator}
    if isinstance(value, AggregateField):
        return {
            "kind": "aggregate",
            "source": value.source,
            "reducer": _json_safe_spec(value.reducer),
        }
    if isinstance(value, LiteralValue):
        return {"kind": "literal", "value": value.value}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_spec(rule) for key, rule in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_spec(item) for item in value]
    _validate_literal_value(value)
    return value


def _json_safe_legacy_aggregations(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rule in value.items():
        key = str(key)
        if isinstance(rule, str):
            out[key] = rule
        elif isinstance(rule, Mapping):
            out[key] = {str(k): str(v) for k, v in rule.items()}
        elif isinstance(rule, Sequence) and not isinstance(
            rule, (str, bytes, bytearray)
        ):
            out[key] = [str(item) for item in rule]
        else:
            raise TypeError(
                "legacy aggregation rules must be method strings, sequences of methods, or "
                "output-name-to-method mappings."
            )
    return out


def _start_operation(
    project: Project,
    artifact: BaseArtifact,
    *,
    operator: AggregateOperator,
    output_artifact_type: ArtifactType,
    output_label: str,
    retained_key: Sequence[str],
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
        operator.save_to_dir(
            project.storage.operator_dir(operator_id), operator_id=operator_id
        )
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
    project.catalog.add_operation_source(operation_id, "source", artifact.artifact_id)

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=output_artifact_type,
        label=output_label,
        lineage_mode="reduced_key",
        status="incomplete",
        basis_artifact_ids=(artifact.artifact_id,),
    )
    project.catalog.add_operation_output(
        operation_id, output_label, artifact_id, ordinal=0
    )
    writer = create_artifact_writer(
        artifact_type=output_artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=output_label,
        operation_id=operation_id,
        lineage_mode="reduced_key",
        basis_artifact_ids=(artifact.artifact_id,),
    )

    descriptor = {
        "schema_version": 2,
        "operation_id": operation_id,
        "operation_type": "translate",
        "operator_id": operator_id,
        "kind": "aggregate",
        "status": "incomplete",
        "resumable": False,
        "sources": {"source": artifact.artifact_id},
        "output_artifact_ids": {output_label: artifact_id},
        "request": {
            "output_label": output_label,
            "batch_size": batch_size,
            "batch_unit": "retained_key_groups",
            "to_key": list(retained_key),
            **dict(request_state),
        },
    }
    _write_descriptor(operation_dir, descriptor)
    if memo is not None:
        project.catalog.add_memo(
            target_type="operation", target_id=operation_id, body=memo
        )

    return {
        "operator_id": operator_id,
        "operation_id": operation_id,
        "operation_dir": operation_dir,
        "artifact_id": artifact_id,
        "writer": writer,
        "descriptor": descriptor,
    }


def _finish_operation(
    project: Project,
    artifact: BaseArtifact,
    *,
    retained_key: Sequence[str],
    operation: Mapping[str, Any],
) -> BaseArtifact:
    writer = operation["writer"]
    writer.finalize()
    validate_primary_key_relationship(
        basis_keys=[artifact.primary_key],
        output_key=retained_key,
        lineage_mode="reduced_key",
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
    project: Project,
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
