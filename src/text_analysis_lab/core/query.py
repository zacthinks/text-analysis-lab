"""Minimal DuckDB query engine for TeAL artifacts.

QueryEngine has one job: build and execute relational views over the standard
artifact side tables. It understands keys, positions, storage locations,
table-backed representation data, and metadata lineage. Non-table representation
data belongs to artifact subclasses, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args
from uuid import uuid4

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import MissingDependencyError, QueryError
from text_analysis_lab.core.lineage import (
    basis_artifact_ids,
    iter_metadata_lineage_sources,
    lineage_mode_for_artifact,
    lineage_paths_to_ancestor,
)
from text_analysis_lab.core.types import (
    ArtifactType,
    ColumnSelect,
    MetadataMode,
    StreamingMode,
    StructuralColumn,
)
from text_analysis_lab.core.utils import (
    quote_identifier,
    resolve_names,
    sql_literal,
    str_keys,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from text_analysis_lab.core.artifact_base import BaseArtifact

METADATA_MODES: tuple[str, ...] = get_args(MetadataMode)

STRUCTURAL_COLUMNS: tuple[str, ...] = get_args(StructuralColumn)


@dataclass(frozen=True)
class Location:
    """Physical storage coordinates for one artifact row."""

    position: int
    batch: int
    row_offset: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Location:
        return cls(
            position=int(row["_position"]),
            batch=int(row["_batch"]),
            row_offset=int(row["_row_offset"]),
        )


@dataclass(frozen=True)
class MetadataJoinSpec:
    """Join plan for one metadata source."""

    artifact_id: str
    alias: str
    metadata_sql: str
    source_columns: tuple[str, ...]


@dataclass(frozen=True)
class ViewColumn:
    """One queryable column in an artifact view."""

    namespace: str
    base_name: str
    qualified_name: str
    output_name: str
    sql_expr: str
    source_artifact_id: str | None = None


@dataclass(frozen=True)
class ArtifactViewSQL:
    """A lazy artifact SQL view and its column bookkeeping."""

    sql: str
    columns: tuple[ViewColumn, ...]
    structural_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    data_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    ambiguous_columns: dict[str, tuple[str, ...]]
    mapping: dict[str, str]


def _parquet_dataset_expr(path: Path) -> str:
    glob = str(Path(path) / "*.parquet")
    # Individual Parquet parts are written batch-by-batch. A nullable column can
    # therefore be inferred as Arrow NULL in an all-null batch and as its concrete
    # type in a later batch. DuckDB's default multi-file schema inference takes
    # the first file's physical schema and can then fail on a later concrete type.
    # union_by_name derives a compatible dataset schema across all parts while the
    # writer's existing column-name checks still enforce TeAL's logical schema.
    return f"read_parquet({sql_literal(glob)}, union_by_name = true)"


def _jsonl_dataset_expr(
    path: Path, *, columns: Sequence[str], types: dict[str, str]
) -> str:
    glob = str(Path(path) / "*.jsonl")
    duckdb_columns = {"_position": "BIGINT"}
    duckdb_columns.update({str(col): str(types[str(col)]) for col in columns})
    columns_sql = ", ".join(
        f"{sql_literal(name)}: {sql_literal(dtype)}"
        for name, dtype in duckdb_columns.items()
    )
    return (
        f"read_json({sql_literal(glob)}, "
        f"columns = {{{columns_sql}}}, "
        "format = 'newline_delimited')"
    )


def _columns_requested(value: ColumnSelect) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        return True
    return len(value) > 0


def _validate_metadata_mode(metadata_mode: object) -> None:
    if metadata_mode not in METADATA_MODES:
        raise ValueError(f"metadata_mode must be one of {METADATA_MODES}.")


def _validate_metadata_query_request(
    *,
    metadata_mode: object,
    metadata_columns: ColumnSelect,
) -> None:
    _validate_metadata_mode(metadata_mode)
    if metadata_mode == "none" and _columns_requested(metadata_columns):
        raise QueryError(
            "metadata_mode='none' cannot be combined with requested metadata_columns."
        )


def _assign_view_output_names(
    pending: Sequence[dict[str, str]],
) -> tuple[tuple[ViewColumn, ...], dict[str, tuple[str, ...]], dict[str, str]]:
    names, ambiguous = resolve_names(
        [column["base_name"] for column in pending],
        [column["qualified_name"] for column in pending],
        reserved_names=STRUCTURAL_COLUMNS,
        error_cls=QueryError,
    )

    mapping: dict[str, str] = {}
    columns: list[ViewColumn] = []
    for column, name in zip(pending, names, strict=True):
        mapping[column["qualified_name"]] = name
        columns.append(
            ViewColumn(
                output_name=name,
                **column,
            )
        )

    return tuple(columns), ambiguous, mapping


def _order_clause(order_by: str | Sequence[str] | None) -> str | None:
    if order_by is None:
        return None
    if isinstance(order_by, str):
        return order_by
    return ", ".join(str(col) for col in order_by)


def _result_columns(result: Any) -> list[str]:
    description = result.description or []
    return [str(item[0]) for item in description]


def _record_from_row(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    return str_keys(dict(zip(columns, row, strict=True)))


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    columns = _result_columns(result)
    return [_record_from_row(columns, row) for row in result.fetchall()]


def _temp_name(prefix: str) -> str:
    return f"_teal_{prefix}_{uuid4().hex}"


class QueryEngine:
    """DuckDB-backed query engine for artifact keys, table data, and metadata."""

    def __init__(self, project: Any):
        self.project = project
        self._con = None
        self._relation_columns_cache: dict[str, tuple[str, ...]] = {}
        self._artifact_view_cache: dict[
            tuple[str, MetadataMode, bool], ArtifactViewSQL
        ] = {}
        self._lineage_mapping_cache: dict[tuple[str, str], str] = {}

    @property
    def con(self):
        if self._con is None:
            try:
                import duckdb
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError(
                    "DuckDB is required for TeAL query operations. Install duckdb."
                ) from exc
            self._con = duckdb.connect(database=":memory:")
        return self._con

    def clear_cache(self) -> None:
        self._relation_columns_cache.clear()
        self._artifact_view_cache.clear()
        self._lineage_mapping_cache.clear()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
        self.clear_cache()

    # ------------------------------------------------------------------
    # Key, position, and location lookup
    # ------------------------------------------------------------------

    def _by_key(
        self, artifact: BaseArtifact, key: dict[str, Any], select: str
    ) -> tuple[DuckDBPyConnection, tuple[Any, ...]]:
        primary_key = [str(col) for col in artifact.primary_key]
        if not primary_key:
            raise QueryError(f"Artifact {artifact.artifact_id} has no primary key.")

        predicates: list[str] = []
        params: list[Any] = []
        for col in primary_key:
            predicates.append(f"{quote_identifier(col)} = ?")
            params.append(key[col])

        sql = (
            f"SELECT {select} FROM {_parquet_dataset_expr(artifact.keys_dir)} "
            f"WHERE {' AND '.join(predicates)} LIMIT 1"
        )
        result = self.con.execute(sql, params)
        row = result.fetchone()
        if row is None:
            raise KeyError(key)

        return result, row

    def key_row_by_key(
        self, artifact: BaseArtifact, key: dict[str, Any]
    ) -> dict[str, Any]:
        result, row = self._by_key(artifact, key, select="*")
        return _record_from_row(_result_columns(result), row)

    def position_by_key(self, artifact: BaseArtifact, key: dict[str, Any]) -> int:
        _, row = self._by_key(artifact, key, select="_position")
        return int(row[0])

    def key_row_by_position(
        self, artifact: BaseArtifact, position: int
    ) -> dict[str, Any]:
        sql = (
            f"SELECT * FROM {_parquet_dataset_expr(artifact.keys_dir)} "
            "WHERE _position = ? LIMIT 1"
        )
        result = self.con.execute(sql, [int(position)])
        row = result.fetchone()
        if row is None:
            raise IndexError(position)
        return _record_from_row(_result_columns(result), row)

    def location_by_key(self, artifact: BaseArtifact, key: dict[str, Any]) -> Location:
        return Location.from_row(self.key_row_by_key(artifact, key))

    def location_by_position(self, artifact: BaseArtifact, position: int) -> Location:
        return Location.from_row(self.key_row_by_position(artifact, int(position)))

    def _fetch_by_positions(
        self,
        artifact: BaseArtifact,
        positions: Sequence[int],
        *,
        select_exprs: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Fetch selected key-table fields for artifact-local positions.

        This helper owns the temp-table join mechanics and request-order
        preservation. Callers choose the projection and validate the returned
        records according to what they asked for.
        """
        if not positions:
            return []

        temp = pd.DataFrame(
            {
                "_position": [int(pos) for pos in positions],
                "_request_order": list(range(len(positions))),
            }
        )
        temp_name = _temp_name("positions")
        self.con.register(temp_name, temp)
        try:
            sql = (
                "SELECT "
                + ", ".join([*select_exprs, "p._request_order AS _request_order"])
                + f" FROM {quote_identifier(temp_name)} p "
                + f"LEFT JOIN {_parquet_dataset_expr(artifact.keys_dir)} k USING (_position) "
                + "ORDER BY p._request_order"
            )
            records = _records_from_result(self.con.execute(sql))
        finally:
            self.con.unregister(temp_name)

        for record in records:
            record.pop("_request_order", None)
        return records

    def _fetch_by_keys(
        self,
        artifact: BaseArtifact,
        keys: Sequence[dict[str, Any]],
        *,
        select_exprs: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Fetch selected key-table fields for normalized primary keys.

        This helper owns the temp-table join mechanics and request-order
        preservation. Callers choose the projection and validate the returned
        records according to what they asked for.
        """
        if not keys:
            return []

        primary_key = [str(col) for col in artifact.primary_key]
        temp_records = []
        for order, key in enumerate(keys):
            record = {col: key[col] for col in primary_key}
            record["_request_order"] = order
            temp_records.append(record)

        temp = pd.DataFrame(temp_records)
        temp_name = _temp_name("keys")
        self.con.register(temp_name, temp)
        try:
            predicates = " AND ".join(
                f"k.{quote_identifier(col)} = q.{quote_identifier(col)}"
                for col in primary_key
            )
            sql = (
                "SELECT "
                + ", ".join([*select_exprs, "q._request_order AS _request_order"])
                + f" FROM {quote_identifier(temp_name)} q "
                + f"LEFT JOIN {_parquet_dataset_expr(artifact.keys_dir)} k ON {predicates} "
                + "ORDER BY q._request_order"
            )
            records = _records_from_result(self.con.execute(sql))
        finally:
            self.con.unregister(temp_name)

        for record in records:
            record.pop("_request_order", None)
        return records

    def positions_by_keys(
        self,
        artifact: BaseArtifact,
        keys: Sequence[dict[str, Any]],
    ) -> list[int]:
        records = self._fetch_by_keys(
            artifact,
            keys,
            select_exprs=["k._position AS _position"],
        )
        if len(records) != len(keys) or any(
            record.get("_position") is None for record in records
        ):
            raise KeyError("One or more keys were not found.")
        return [int(record["_position"]) for record in records]

    def key_rows_by_positions(
        self,
        artifact: BaseArtifact,
        positions: Sequence[int],
    ) -> list[dict[str, Any]]:
        primary_key = [str(col) for col in artifact.primary_key]
        select_exprs = [
            "k._position AS _position",
            "k._batch AS _batch",
            "k._row_offset AS _row_offset",
            *[
                f"k.{quote_identifier(col)} AS {quote_identifier(col)}"
                for col in primary_key
            ],
        ]
        records = self._fetch_by_positions(
            artifact,
            positions,
            select_exprs=select_exprs,
        )
        if len(records) != len(positions) or any(
            record.get("_batch") is None for record in records
        ):
            raise IndexError("One or more positions were not found.")
        return records

    def locations_by_positions(
        self,
        artifact: BaseArtifact,
        positions: Sequence[int],
    ) -> list[Location]:
        records = self._fetch_by_positions(
            artifact,
            positions,
            select_exprs=[
                "k._position AS _position",
                "k._batch AS _batch",
                "k._row_offset AS _row_offset",
            ],
        )
        if len(records) != len(positions) or any(
            record.get("_batch") is None for record in records
        ):
            raise IndexError("One or more positions were not found.")
        return [Location.from_row(record) for record in records]

    def key_rows_by_keys(
        self,
        artifact: BaseArtifact,
        keys: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        primary_key = [str(col) for col in artifact.primary_key]
        select_exprs = [
            "k._position AS _position",
            "k._batch AS _batch",
            "k._row_offset AS _row_offset",
            *[
                f"k.{quote_identifier(col)} AS {quote_identifier(col)}"
                for col in primary_key
            ],
        ]
        records = self._fetch_by_keys(
            artifact,
            keys,
            select_exprs=select_exprs,
        )
        if len(records) != len(keys) or any(
            record.get("_position") is None for record in records
        ):
            raise KeyError("One or more keys were not found.")
        return records

    def locations_by_keys(
        self,
        artifact: BaseArtifact,
        keys: Sequence[dict[str, Any]],
    ) -> list[Location]:
        records = self._fetch_by_keys(
            artifact,
            keys,
            select_exprs=[
                "k._position AS _position",
                "k._batch AS _batch",
                "k._row_offset AS _row_offset",
            ],
        )
        if len(records) != len(keys) or any(
            record.get("_position") is None for record in records
        ):
            raise KeyError("One or more keys were not found.")
        return [Location.from_row(record) for record in records]

    def positions_where_keys(
        self,
        artifact: BaseArtifact,
        *,
        where: str | None = None,
        order_by: str | Sequence[str] | None = "_position",
        limit: int | None = None,
    ) -> list[int]:
        sql = f"SELECT _position FROM {_parquet_dataset_expr(artifact.keys_dir)}"
        if where:
            sql += f" WHERE ({where})"
        order = _order_clause(order_by)
        if order:
            sql += f" ORDER BY {order}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [int(row[0]) for row in self.con.execute(sql).fetchall()]

    def sample_positions_where(
        self,
        artifact: BaseArtifact,
        *,
        where: str,
        metadata_mode: MetadataMode,
        include_data: bool,
        positions: Sequence[int] | None,
        sample_n: int | None,
        sample_frac: float | None,
        random_state: int | None,
    ) -> list[int]:
        """Sample positions from the filtered eligible population with bounded memory.

        Eligibility is the intersection of an optional explicit ``positions``
        restriction and ``where``. Only ``_position`` values are streamed during
        sampling. ``include_data`` mirrors the final artifact query's relational
        view so column naming/filter semantics stay identical; the outer query
        projects only ``_position``, allowing DuckDB to prune unneeded table data
        columns during the eligibility scan.
        """
        if sample_n is not None and sample_frac is not None:
            raise ValueError("Cannot specify both sample_n and sample_frac.")
        if sample_n is None and sample_frac is None:
            raise ValueError("Either sample_n or sample_frac must be provided.")
        if sample_n is not None and int(sample_n) < 0:
            raise ValueError("sample_n must be non-negative.")
        if sample_frac is not None and not 0 <= float(sample_frac) <= 1:
            raise ValueError("sample_frac must be between 0 and 1.")
        if sample_n == 0 or sample_frac == 0:
            return []

        _validate_metadata_query_request(
            metadata_mode=metadata_mode,
            metadata_columns=False,
        )
        registered: list[str] = []
        try:
            positions_name: str | None = None
            if positions is not None:
                # A positions restriction denotes an eligible row set. Duplicate
                # position values therefore do not create duplicate sampling units.
                unique_positions = list(dict.fromkeys(int(pos) for pos in positions))
                if not unique_positions:
                    return []
                temp = pd.DataFrame({"_position": unique_positions})
                positions_name = _temp_name("sample_positions")
                self.con.register(positions_name, temp)
                registered.append(positions_name)

            def eligibility_sql() -> str:
                view = self.build_artifact_view_sql(
                    artifact,
                    metadata_mode=metadata_mode,
                    include_data=include_data,
                )
                ctes = [f"artifact_view AS ({view.sql})"]
                from_sql = "artifact_view v"
                if positions_name is not None:
                    ctes.append(
                        "positions AS (SELECT _position FROM "
                        f"{quote_identifier(positions_name)})"
                    )
                    from_sql += " JOIN positions p USING (_position)"
                return (
                    "WITH "
                    + ", ".join(ctes)
                    + " SELECT v._position AS _position FROM "
                    + from_sql
                    + f" WHERE ({where})"
                )

            sql = eligibility_sql()
            try:
                # Bind without scanning so invalid/ambiguous filters fail before
                # any sampling pass begins.
                self.con.execute(f"EXPLAIN {sql}").fetchall()
            except Exception as exc:
                raise QueryError(f"Artifact query failed: {exc}") from exc

            if sample_frac is not None:
                try:
                    eligible_count = int(
                        self.con.execute(
                            f"SELECT COUNT(*) FROM ({sql}) AS eligible_positions"
                        ).fetchone()[0]
                    )
                except Exception as exc:
                    raise QueryError(f"Artifact query failed: {exc}") from exc
                sample_size = int(round(eligible_count * float(sample_frac)))
            else:
                sample_size = int(sample_n or 0)

            if sample_size <= 0:
                return []

            rng = np.random.default_rng(random_state)
            reservoir: list[int] = []
            seen = 0
            try:
                result = self.con.execute(sql)
                while True:
                    rows = result.fetchmany(10_000)
                    if not rows:
                        break
                    for row in rows:
                        position = int(row[0])
                        if seen < sample_size:
                            reservoir.append(position)
                        else:
                            replace_at = int(rng.integers(0, seen + 1))
                            if replace_at < sample_size:
                                reservoir[replace_at] = position
                        seen += 1
            except Exception as exc:
                raise QueryError(f"Artifact query failed: {exc}") from exc

            if not reservoir:
                return []
            rng.shuffle(reservoir)
            return reservoir
        finally:
            for name in reversed(registered):
                try:
                    self.con.unregister(name)
                except Exception:
                    pass

    def _lineage_paths_between(
        self,
        descendant: BaseArtifact,
        ancestor: BaseArtifact,
    ) -> list[tuple[BaseArtifact, ...]]:
        paths = lineage_paths_to_ancestor(self.project, descendant, ancestor)
        if not paths:
            raise QueryError(
                f"Artifact {ancestor.artifact_id!r} is not an ancestor of "
                f"artifact {descendant.artifact_id!r}."
            )
        return paths

    @staticmethod
    def _path_crosses_rekey(path: Sequence[BaseArtifact]) -> bool:
        return any(
            lineage_mode_for_artifact(artifact) == "rekeyed_key"
            for artifact in path[:-1]
        )

    def _requires_rekey_mapping(
        self,
        descendant: BaseArtifact,
        ancestor: BaseArtifact,
    ) -> bool:
        return any(
            self._path_crosses_rekey(path)
            for path in self._lineage_paths_between(descendant, ancestor)
        )

    def _mapping_sql_for_path(
        self,
        path: Sequence[BaseArtifact],
    ) -> str | None:
        """Return target-position -> source-position SQL for one lineage path.

        Ordinary key-space segments are collapsed into direct primary-key joins.
        Only ``rekeyed_key`` edges are crossed positionally. A path that cannot
        be resolved without reversing a reduced key domain returns ``None``; such
        paths do not contribute to a cross-key-space mapping.
        """
        if not path:
            return None
        descendant = path[0]
        ancestor = path[-1]
        current_pk = [str(col) for col in descendant.primary_key]
        target_keys_sql = _parquet_dataset_expr(descendant.keys_dir)
        select_keys = ", ".join(
            f"tk.{quote_identifier(col)} AS {quote_identifier(col)}"
            for col in current_pk
        )
        relation = (
            "SELECT tk._position AS _target_position"
            + (f", {select_keys}" if select_keys else "")
            + f" FROM {target_keys_sql} tk"
        )

        for index, child in enumerate(path[:-1]):
            if lineage_mode_for_artifact(child) != "rekeyed_key":
                continue
            basis = path[index + 1]
            child_pk = [str(col) for col in child.primary_key]
            basis_pk = [str(col) for col in basis.primary_key]
            missing = [col for col in child_pk if col not in current_pk]
            if missing:
                # A downstream reduction discarded key fields needed to locate
                # the exact rows on the rekeyed side. There is no unique bridge.
                return None
            if child.artifact_type != basis.artifact_type:
                raise QueryError(
                    f"Malformed rekeyed lineage {child.artifact_id} -> {basis.artifact_id}: "
                    f"artifact types differ ({child.artifact_type!r} vs {basis.artifact_type!r})."
                )
            if child.n_rows != basis.n_rows:
                raise QueryError(
                    f"Malformed rekeyed lineage {child.artifact_id} -> {basis.artifact_id}: "
                    f"row counts differ ({child.n_rows} vs {basis.n_rows})."
                )

            child_keys_sql = _parquet_dataset_expr(child.keys_dir)
            basis_keys_sql = _parquet_dataset_expr(basis.keys_dir)
            predicates = " AND ".join(
                f"m.{quote_identifier(col)} = rk.{quote_identifier(col)}"
                for col in child_pk
            )
            basis_select = ", ".join(
                f"bk.{quote_identifier(col)} AS {quote_identifier(col)}"
                for col in basis_pk
            )
            relation = (
                "SELECT m._target_position"
                + (f", {basis_select}" if basis_select else "")
                + f" FROM ({relation}) m "
                + f"LEFT JOIN {child_keys_sql} rk ON {predicates} "
                + f"LEFT JOIN {basis_keys_sql} bk ON bk._position = rk._position"
            )
            current_pk = basis_pk

        ancestor_pk = [str(col) for col in ancestor.primary_key]
        missing = [col for col in ancestor_pk if col not in current_pk]
        if missing:
            return None
        source_keys_sql = _parquet_dataset_expr(ancestor.keys_dir)
        predicates = " AND ".join(
            f"m.{quote_identifier(col)} = sk.{quote_identifier(col)}"
            for col in ancestor_pk
        )
        return (
            "SELECT m._target_position, sk._position AS _source_position "
            f"FROM ({relation}) m LEFT JOIN {source_keys_sql} sk ON {predicates}"
        )

    def lineage_position_mapping_sql(
        self,
        descendant: BaseArtifact,
        ancestor: BaseArtifact,
    ) -> str:
        """Return SQL mapping descendant positions to one ancestor's positions.

        The result has ``_target_position`` and ``_source_position``. Multiple
        lineage branches are unioned; duplicated paths to the same source row are
        de-duplicated. Ambiguous mappings are rejected rather than guessed.
        """
        cache_key = (str(descendant.artifact_id), str(ancestor.artifact_id))
        cached = self._lineage_mapping_cache.get(cache_key)
        if cached is not None:
            return cached

        path_sql: list[str] = []
        for path in self._lineage_paths_between(descendant, ancestor):
            sql = self._mapping_sql_for_path(path)
            if sql is not None:
                path_sql.append(sql)
        if not path_sql:
            raise QueryError(
                f"Cannot map artifact {descendant.artifact_id!r} to ancestor "
                f"{ancestor.artifact_id!r}; no lineage path preserves enough key "
                "information to reach every rekey boundary."
            )

        union_sql = " UNION ALL ".join(f"({sql})" for sql in path_sql)
        mapping_sql = (
            "SELECT DISTINCT _target_position, _source_position "
            f"FROM ({union_sql}) _teal_lineage_paths"
        )

        ambiguous_sql = (
            "SELECT _target_position FROM ("
            f"{mapping_sql}"
            ") m WHERE _source_position IS NOT NULL "
            "GROUP BY _target_position "
            "HAVING COUNT(DISTINCT _source_position) > 1 LIMIT 1"
        )
        row = self.con.execute(ambiguous_sql).fetchone()
        if row is not None:
            raise QueryError(
                f"Lineage mapping from {descendant.artifact_id!r} to "
                f"{ancestor.artifact_id!r} is ambiguous at target position {int(row[0])}."
            )

        self._lineage_mapping_cache[cache_key] = mapping_sql
        return mapping_sql

    def map_descendant_positions_to_ancestor_positions(
        self,
        descendant: BaseArtifact,
        ancestor: BaseArtifact,
        positions: Sequence[int],
    ) -> list[int]:
        """Resolve descendant positions to an ancestor, crossing rekeys safely."""
        if not positions:
            return []
        if not self._requires_rekey_mapping(descendant, ancestor):
            return self.map_left_positions_to_right_positions(
                descendant, ancestor, positions
            )

        mapping_sql = self.lineage_position_mapping_sql(descendant, ancestor)
        temp = pd.DataFrame(
            {
                "_position": [int(pos) for pos in positions],
                "_request_order": list(range(len(positions))),
            }
        )
        temp_name = _temp_name("lineage_positions")
        self.con.register(temp_name, temp)
        try:
            sql = (
                "SELECT p._request_order, p._position AS left_position, "
                "m._source_position AS right_position "
                f"FROM {quote_identifier(temp_name)} p "
                f"LEFT JOIN ({mapping_sql}) m ON m._target_position = p._position "
                "ORDER BY p._request_order"
            )
            records = _records_from_result(self.con.execute(sql))
        finally:
            self.con.unregister(temp_name)

        if len(records) != len(positions):
            raise QueryError(
                "Lineage position mapping did not preserve row count: "
                f"expected {len(positions)}, got {len(records)}."
            )
        missing = [
            int(record["left_position"])
            for record in records
            if record.get("right_position") is None
        ]
        if missing:
            raise QueryError(
                f"Could not map every position from {descendant.artifact_id!r} to "
                f"{ancestor.artifact_id!r}; missing target positions: {missing[:20]}."
            )
        return [int(record["right_position"]) for record in records]

    def map_left_positions_to_right_positions(
        self,
        left_artifact: BaseArtifact,
        right_artifact: BaseArtifact,
        left_positions: Sequence[int],
    ) -> list[int]:
        if not left_positions:
            return []

        left_pk = [str(col) for col in left_artifact.primary_key]
        right_pk = [str(col) for col in right_artifact.primary_key]
        if left_pk != right_pk:
            raise QueryError(
                "Cannot map positions between artifacts with different primary keys: "
                f"{left_artifact.artifact_id} has {left_pk!r}, "
                f"{right_artifact.artifact_id} has {right_pk!r}."
            )

        temp = pd.DataFrame(
            {
                "_position": [int(pos) for pos in left_positions],
                "_request_order": list(range(len(left_positions))),
            }
        )
        temp_name = _temp_name("positions")
        self.con.register(temp_name, temp)
        try:
            left_key_select = ", ".join(
                f"l.{quote_identifier(col)} AS {quote_identifier(col)}"
                for col in left_pk
            )
            predicates = " AND ".join(
                f"lk.{quote_identifier(col)} = r.{quote_identifier(col)}"
                for col in left_pk
            )
            sql = (
                "WITH left_keys AS ("
                "SELECT p._request_order, p._position AS left_position, "
                f"{left_key_select} "
                f"FROM {quote_identifier(temp_name)} p "
                f"LEFT JOIN {_parquet_dataset_expr(left_artifact.keys_dir)} l USING (_position)"
                ") "
                "SELECT lk._request_order, lk.left_position, r._position AS right_position "
                "FROM left_keys lk "
                f"LEFT JOIN {_parquet_dataset_expr(right_artifact.keys_dir)} r ON {predicates} "
                "ORDER BY lk._request_order"
            )
            records = _records_from_result(self.con.execute(sql))
        finally:
            self.con.unregister(temp_name)

        if len(records) != len(left_positions):
            raise QueryError(
                "Position mapping did not preserve row count: "
                f"expected {len(left_positions)}, got {len(records)}."
            )
        missing = [
            record["left_position"]
            for record in records
            if record.get("right_position") is None
        ]
        if missing:
            raise QueryError(
                "Could not map every left artifact position to the right artifact. "
                f"Missing left positions: {[int(pos) for pos in missing]}."
            )
        return [int(record["right_position"]) for record in records]

    # ------------------------------------------------------------------
    # Relation helpers
    # ------------------------------------------------------------------

    def _relation_columns(self, relation_sql: str) -> tuple[str, ...]:
        cached = self._relation_columns_cache.get(relation_sql)
        if cached is not None:
            return cached

        try:
            result = self.con.execute(f"DESCRIBE SELECT * FROM {relation_sql}")
            column_index = _result_columns(result).index("column_name")
            columns = tuple(str(row[column_index]) for row in result.fetchall())
        except Exception as exc:
            raise QueryError(f"Could not inspect relation schema: {exc}") from exc

        self._relation_columns_cache[relation_sql] = columns
        return columns

    def _parquet_component_sql(self, path: Path | None) -> str | None:
        if path is None or not Path(path).exists():
            return None
        if not list(Path(path).glob("*.parquet")):
            return None
        return _parquet_dataset_expr(Path(path))

    def _table_data_relation_sql(
        self, artifact: BaseArtifact
    ) -> tuple[str, tuple[str, ...]] | None:
        if artifact.artifact_type != ArtifactType.TABLE:
            return None
        if not artifact.has_own_data():
            return None
        sql = self._parquet_component_sql(getattr(artifact, "data_dir", None))
        if sql is None:
            return None
        columns = tuple(
            col for col in self._relation_columns(sql) if col != "_position"
        )
        return sql, columns

    def _jsonl_data_relation_sql(
        self, artifact: BaseArtifact
    ) -> tuple[str, tuple[str, ...]] | None:
        if artifact.artifact_type != ArtifactType.JSONL:
            return None
        if not artifact.has_own_data():
            return None

        data_dir = getattr(artifact, "data_dir", None)
        if data_dir is None or not Path(data_dir).exists():
            return None
        if not list(Path(data_dir).glob("*.jsonl")):
            return None

        component = artifact.components.get("data", {})
        columns = tuple(str(col) for col in component.get("columns", []))
        raw_types = component.get("types", {})
        if not isinstance(raw_types, dict):
            raise QueryError(
                f"JSONL artifact {artifact.artifact_id} data component has invalid types metadata."
            )
        types = {str(col): str(dtype) for col, dtype in raw_types.items()}
        missing_types = [col for col in columns if col not in types]
        if missing_types:
            raise QueryError(
                f"JSONL artifact {artifact.artifact_id} data component does not record "
                f"DuckDB types for columns: {missing_types}."
            )

        return _jsonl_dataset_expr(
            Path(data_dir), columns=columns, types=types
        ), columns

    def _data_relation_sql(
        self, artifact: BaseArtifact
    ) -> tuple[str, tuple[str, ...]] | None:
        lineage = artifact.descriptor.get("lineage", {})
        if isinstance(lineage, dict) and lineage.get("lineage_mode") in {
            "merged_key",
            "joined_key",
        }:
            view = self.build_artifact_view_sql(
                artifact,
                metadata_mode="none",
                include_data=True,
            )
            if not view.data_columns:
                return None
            selected = ", ".join(
                [
                    "_position",
                    *[quote_identifier(col) for col in view.data_columns],
                ]
            )
            return (
                f"(SELECT {selected} FROM ({view.sql}) merged_data_view)",
                tuple(view.data_columns),
            )

        table_sql = self._table_data_relation_sql(artifact)
        if table_sql is not None:
            return table_sql
        return self._jsonl_data_relation_sql(artifact)

    def _local_metadata_relation_sql(self, artifact: BaseArtifact) -> str | None:
        return self._parquet_component_sql(getattr(artifact, "metadata_dir", None))

    # ------------------------------------------------------------------
    # Artifact view planning
    # ------------------------------------------------------------------

    def _metadata_join_specs(
        self,
        artifact: BaseArtifact,
        *,
        metadata_mode: MetadataMode,
    ) -> list[MetadataJoinSpec]:
        _validate_metadata_mode(metadata_mode)
        if metadata_mode == "none":
            return []

        if metadata_mode == "local":
            sources = [artifact]
        else:
            sources = iter_metadata_lineage_sources(self.project, artifact)

        specs: list[MetadataJoinSpec] = []
        for index, source_artifact in enumerate(sources):
            metadata_sql = self._local_metadata_relation_sql(source_artifact)
            if metadata_sql is None:
                continue
            source_columns = tuple(
                col
                for col in self._relation_columns(metadata_sql)
                if not col.startswith("_")
            )
            if not source_columns:
                continue
            specs.append(
                MetadataJoinSpec(
                    artifact_id=str(source_artifact.artifact_id),
                    alias=f"m{index}",
                    metadata_sql=metadata_sql,
                    source_columns=source_columns,
                )
            )
        return specs

    def build_artifact_view_sql(
        self,
        artifact: BaseArtifact,
        *,
        metadata_mode: MetadataMode = "none",
        include_data: bool = True,
    ) -> ArtifactViewSQL:
        # Complete and failed artifacts are immutable, so their planned views can
        # be cached safely. Incomplete artifacts may still acquire new parts or
        # components while an operation is running; refresh their descriptor and
        # always rebuild the view so inspection from another process sees current
        # durable state.
        catalog_status = artifact.status
        descriptor_status = artifact.descriptor.get("status")
        cacheable = catalog_status != "incomplete"
        if not cacheable or descriptor_status != catalog_status:
            artifact.refresh()

        cache_key = (str(artifact.artifact_id), metadata_mode, bool(include_data))
        if cacheable:
            cached = self._artifact_view_cache.get(cache_key)
            if cached is not None:
                return cached

        _validate_metadata_mode(metadata_mode)

        lineage = artifact.descriptor.get("lineage", {})
        if isinstance(lineage, dict) and lineage.get("lineage_mode") == "merged_key":
            view = self._build_merged_artifact_view_sql(
                artifact,
                metadata_mode=metadata_mode,
                include_data=include_data,
            )
            if cacheable:
                self._artifact_view_cache[cache_key] = view
            return view

        if isinstance(lineage, dict) and lineage.get("lineage_mode") == "joined_key":
            view = self._build_joined_artifact_view_sql(
                artifact,
                metadata_mode=metadata_mode,
                include_data=include_data,
            )
            if cacheable:
                self._artifact_view_cache[cache_key] = view
            return view

        current_pk = [str(col) for col in artifact.primary_key]
        key_sql = _parquet_dataset_expr(artifact.keys_dir)
        ctes: list[str] = [f"keys AS (SELECT * FROM {key_sql})"]
        joins: list[str] = []
        select_exprs: list[str] = [
            f"k.{quote_identifier(col)} AS {quote_identifier(col)}"
            for col in STRUCTURAL_COLUMNS
        ]
        pending_columns: list[dict[str, str]] = []

        for col in current_pk:
            pending_columns.append(
                {
                    "namespace": "key",
                    "base_name": col,
                    "qualified_name": f"key.{col}",
                    "sql_expr": f"k.{quote_identifier(col)}",
                    "source_artifact_id": str(artifact.artifact_id),
                }
            )

        if include_data:
            data_artifact = artifact.data_artifact
            data_relation = (
                None
                if data_artifact is None
                else self._data_relation_sql(data_artifact)
            )
            if data_relation is not None and data_artifact is not None:
                data_sql, data_raw_columns = data_relation
                owner_pk = [str(col) for col in data_artifact.primary_key]
                if data_artifact is artifact:
                    selected = ", ".join(
                        [
                            "_position",
                            *[quote_identifier(col) for col in data_raw_columns],
                        ]
                    )
                    ctes.append(f"data AS (SELECT {selected} FROM {data_sql})")
                    joins.append("LEFT JOIN data d USING (_position)")
                else:
                    raw_selected = ", ".join(
                        [
                            "_position",
                            *[quote_identifier(col) for col in data_raw_columns],
                        ]
                    )
                    ctes.append(f"data_raw AS (SELECT {raw_selected} FROM {data_sql})")
                    if self._requires_rekey_mapping(artifact, data_artifact):
                        mapping_sql = self.lineage_position_mapping_sql(
                            artifact, data_artifact
                        )
                        missing_sql = (
                            "SELECT COUNT(*) FROM "
                            f"{_parquet_dataset_expr(artifact.keys_dir)} tk "
                            f"LEFT JOIN ({mapping_sql}) lm ON lm._target_position = tk._position "
                            "WHERE lm._source_position IS NULL"
                        )
                        missing_count = int(self.con.execute(missing_sql).fetchone()[0])
                        if missing_count:
                            raise QueryError(
                                f"Inherited data from {data_artifact.artifact_id!r} cannot be "
                                f"mapped to all rows of {artifact.artifact_id!r}; "
                                f"{missing_count} target row(s) are unmapped."
                            )
                        ctes.append(f"data_map AS ({mapping_sql})")
                        data_select = ", ".join(
                            [
                                "dm._target_position AS _position",
                                *[
                                    f"raw.{quote_identifier(col)} AS {quote_identifier(col)}"
                                    for col in data_raw_columns
                                ],
                            ]
                        )
                        ctes.append(
                            "data AS ("
                            f"SELECT {data_select} FROM data_map dm "
                            "LEFT JOIN data_raw raw ON raw._position = dm._source_position"
                            ")"
                        )
                    else:
                        if current_pk != owner_pk:
                            raise QueryError(
                                "Inherited table data requires identical preserved primary keys: "
                                f"{artifact.artifact_id} has {current_pk!r}, "
                                f"{data_artifact.artifact_id} has {owner_pk!r}."
                            )
                        owner_key_sql = _parquet_dataset_expr(data_artifact.keys_dir)
                        owner_key_select = ", ".join(
                            [
                                "_position AS _data_position",
                                *[quote_identifier(col) for col in owner_pk],
                            ]
                        )
                        ctes.append(
                            f"data_keys AS (SELECT {owner_key_select} FROM {owner_key_sql})"
                        )
                        data_select = ", ".join(
                            [
                                "k._position AS _position",
                                *[
                                    f"raw.{quote_identifier(col)} AS {quote_identifier(col)}"
                                    for col in data_raw_columns
                                ],
                            ]
                        )
                        predicates = " AND ".join(
                            f"k.{quote_identifier(col)} = dk.{quote_identifier(col)}"
                            for col in owner_pk
                        )
                        ctes.append(
                            "data AS ("
                            f"SELECT {data_select} FROM keys k "
                            f"LEFT JOIN data_keys dk ON {predicates} "
                            "LEFT JOIN data_raw raw ON raw._position = dk._data_position"
                            ")"
                        )
                    joins.append("LEFT JOIN data d USING (_position)")

                for col in data_raw_columns:
                    pending_columns.append(
                        {
                            "namespace": "data",
                            "base_name": col,
                            "qualified_name": f"data.{col}",
                            "sql_expr": f"d.{quote_identifier(col)}",
                            "source_artifact_id": str(data_artifact.artifact_id),
                        }
                    )

        metadata_specs = self._metadata_join_specs(
            artifact, metadata_mode=metadata_mode
        )
        seen_metadata_aliases: set[str] = set()
        metadata_internal_names: dict[tuple[str, str], str] = {}
        for spec in metadata_specs:
            if spec.alias in seen_metadata_aliases:
                continue
            if spec.artifact_id == artifact.artifact_id:
                selected = ", ".join(
                    [
                        "_position",
                        *[quote_identifier(col) for col in spec.source_columns],
                    ]
                )
                ctes.append(
                    f"{spec.alias} AS (SELECT {selected} FROM {spec.metadata_sql})"
                )
                joins.append(f"LEFT JOIN {spec.alias} USING (_position)")
                for col in spec.source_columns:
                    metadata_internal_names[(spec.alias, col)] = col
            else:
                source_artifact = self.project.get_artifact(spec.artifact_id)
                source_pk = [str(col) for col in source_artifact.primary_key]
                metadata_select_parts: list[str] = []
                for index, col in enumerate(spec.source_columns):
                    internal = f"_teal_{spec.alias}_metadata_{index}"
                    metadata_internal_names[(spec.alias, col)] = internal
                    metadata_select_parts.append(
                        f"sm.{quote_identifier(col)} AS {quote_identifier(internal)}"
                    )
                metadata_select = ", ".join(metadata_select_parts)

                if self._requires_rekey_mapping(artifact, source_artifact):
                    mapping_sql = self.lineage_position_mapping_sql(
                        artifact, source_artifact
                    )
                    ctes.append(
                        f"{spec.alias} AS ("
                        "SELECT lm._target_position AS _position, "
                        f"{metadata_select} "
                        f"FROM ({mapping_sql}) lm "
                        f"LEFT JOIN {spec.metadata_sql} sm "
                        "ON sm._position = lm._source_position"
                        ")"
                    )
                    joins.append(f"LEFT JOIN {spec.alias} USING (_position)")
                else:
                    missing = [col for col in source_pk if col not in current_pk]
                    if missing:
                        raise QueryError(
                            f"Cannot join metadata from {spec.artifact_id}; current artifact "
                            f"does not contain source key columns: {missing}."
                        )
                    source_keys_sql = _parquet_dataset_expr(source_artifact.keys_dir)
                    key_select = ", ".join(
                        f"sk.{quote_identifier(col)} AS {quote_identifier(col)}"
                        for col in source_pk
                    )
                    ctes.append(
                        f"{spec.alias} AS ("
                        "SELECT "
                        f"{key_select}, {metadata_select} "
                        f"FROM {source_keys_sql} sk "
                        f"LEFT JOIN {spec.metadata_sql} sm USING (_position)"
                        ")"
                    )
                    predicates = " AND ".join(
                        f"k.{quote_identifier(col)} = {spec.alias}.{quote_identifier(col)}"
                        for col in source_pk
                    )
                    joins.append(f"LEFT JOIN {spec.alias} ON {predicates}")
            seen_metadata_aliases.add(spec.alias)

        for spec in metadata_specs:
            for col in spec.source_columns:
                internal = metadata_internal_names[(spec.alias, col)]
                pending_columns.append(
                    {
                        "namespace": "metadata",
                        "base_name": col,
                        "qualified_name": f"metadata.{spec.artifact_id}.{col}",
                        "sql_expr": f"{spec.alias}.{quote_identifier(internal)}",
                        "source_artifact_id": spec.artifact_id,
                    }
                )

        view_columns, ambiguous_columns, mapping = _assign_view_output_names(
            pending_columns
        )
        for column in view_columns:
            select_exprs.append(
                f"{column.sql_expr} AS {quote_identifier(column.output_name)}"
            )

        sql = (
            "WITH "
            + ", ".join(ctes)
            + " SELECT "
            + ", ".join(select_exprs)
            + " FROM keys k "
            + " ".join(joins)
        )

        key_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "key"
        )
        data_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "data"
        )
        metadata_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "metadata"
        )
        output_columns = tuple(col.output_name for col in view_columns)

        view = ArtifactViewSQL(
            sql=sql,
            columns=view_columns,
            structural_columns=STRUCTURAL_COLUMNS,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            output_columns=output_columns,
            ambiguous_columns=ambiguous_columns,
            mapping=mapping,
        )
        if cacheable:
            self._artifact_view_cache[cache_key] = view
        return view

    def _build_joined_artifact_view_sql(
        self,
        artifact: BaseArtifact,
        *,
        metadata_mode: MetadataMode,
        include_data: bool,
    ) -> ArtifactViewSQL:
        """Build the basis-first lazy relational view for ``joined_key`` lineage."""
        basis_ids = basis_artifact_ids(artifact)
        if len(basis_ids) < 2:
            raise QueryError(
                f"Joined artifact {artifact.artifact_id} must have at least two basis artifacts."
            )
        bases = [self.project.get_artifact(artifact_id) for artifact_id in basis_ids]
        current_pk = tuple(str(col) for col in artifact.primary_key)
        for basis in bases:
            if tuple(str(col) for col in basis.primary_key) != current_pk:
                raise QueryError(
                    f"Joined artifact {artifact.artifact_id} expects primary key "
                    f"{list(current_pk)}, but basis {basis.artifact_id} uses "
                    f"{list(basis.primary_key)}."
                )

        key_sql = _parquet_dataset_expr(artifact.keys_dir)
        ctes: list[str] = [f"keys AS (SELECT * FROM {key_sql})"]
        joins: list[str] = []
        select_exprs: list[str] = [
            f"k.{quote_identifier(col)} AS {quote_identifier(col)}"
            for col in STRUCTURAL_COLUMNS
        ]
        pending_columns: list[dict[str, str | None]] = []
        for col in current_pk:
            pending_columns.append(
                {
                    "namespace": "key",
                    "base_name": col,
                    "qualified_name": f"key.{col}",
                    "sql_expr": f"k.{quote_identifier(col)}",
                    "source_artifact_id": str(artifact.artifact_id),
                }
            )

        # Representation data are composed horizontally from each basis. The
        # first artifact owns the result row universe; every later relation is a
        # left join on the shared stable primary key. Repeated references to the
        # exact same underlying data owner are included only once.
        seen_data_origins: set[tuple[str | None, str]] = set()
        if include_data:
            for index, basis in enumerate(bases):
                branch = self.build_artifact_view_sql(
                    basis, metadata_mode="none", include_data=True
                )
                data_columns = [
                    col for col in branch.columns if col.namespace == "data"
                ]
                selected_columns = [
                    col
                    for col in data_columns
                    if (col.source_artifact_id, col.base_name) not in seen_data_origins
                ]
                if not selected_columns:
                    continue
                alias = f"jd{index}"
                selected_names = [
                    *current_pk,
                    *[col.output_name for col in selected_columns],
                ]
                select_sql = ", ".join(
                    f"b.{quote_identifier(name)} AS {quote_identifier(name)}"
                    for name in selected_names
                )
                ctes.append(f"{alias} AS (SELECT {select_sql} FROM ({branch.sql}) b)")
                predicates = " AND ".join(
                    f"k.{quote_identifier(col)} = {alias}.{quote_identifier(col)}"
                    for col in current_pk
                )
                joins.append(f"LEFT JOIN {alias} ON {predicates}")
                for column in selected_columns:
                    seen_data_origins.add((column.source_artifact_id, column.base_name))
                    pending_columns.append(
                        {
                            "namespace": "data",
                            "base_name": column.base_name,
                            "qualified_name": f"data.{column.base_name}",
                            "sql_expr": f"{alias}.{quote_identifier(column.output_name)}",
                            "source_artifact_id": column.source_artifact_id,
                        }
                    )

        # Metadata follows ordinary lineage rules. joined_key traversal branches
        # through all bases and de-duplicates common upstream metadata artifacts.
        metadata_specs = self._metadata_join_specs(
            artifact, metadata_mode=metadata_mode
        )
        seen_metadata_aliases: set[str] = set()
        metadata_internal_names: dict[tuple[str, str], str] = {}
        for spec in metadata_specs:
            if spec.alias in seen_metadata_aliases:
                continue
            if spec.artifact_id == artifact.artifact_id:
                selected = ", ".join(
                    [
                        "_position",
                        *[quote_identifier(col) for col in spec.source_columns],
                    ]
                )
                ctes.append(
                    f"{spec.alias} AS (SELECT {selected} FROM {spec.metadata_sql})"
                )
                joins.append(f"LEFT JOIN {spec.alias} USING (_position)")
                for col in spec.source_columns:
                    metadata_internal_names[(spec.alias, col)] = col
            else:
                source_artifact = self.project.get_artifact(spec.artifact_id)
                source_pk = [str(col) for col in source_artifact.primary_key]
                metadata_select_parts: list[str] = []
                for col_index, col in enumerate(spec.source_columns):
                    internal = f"_teal_{spec.alias}_metadata_{col_index}"
                    metadata_internal_names[(spec.alias, col)] = internal
                    metadata_select_parts.append(
                        f"sm.{quote_identifier(col)} AS {quote_identifier(internal)}"
                    )
                metadata_select = ", ".join(metadata_select_parts)

                if self._requires_rekey_mapping(artifact, source_artifact):
                    mapping_sql = self.lineage_position_mapping_sql(
                        artifact, source_artifact
                    )
                    ctes.append(
                        f"{spec.alias} AS (SELECT lm._target_position AS _position, "
                        f"{metadata_select} FROM ({mapping_sql}) lm "
                        f"LEFT JOIN {spec.metadata_sql} sm "
                        "ON sm._position = lm._source_position)"
                    )
                    joins.append(f"LEFT JOIN {spec.alias} USING (_position)")
                else:
                    missing = [col for col in source_pk if col not in current_pk]
                    if missing:
                        raise QueryError(
                            f"Cannot join metadata from {spec.artifact_id}; joined artifact "
                            f"does not contain source key columns: {missing}."
                        )
                    source_keys_sql = _parquet_dataset_expr(source_artifact.keys_dir)
                    key_select = ", ".join(
                        f"sk.{quote_identifier(col)} AS {quote_identifier(col)}"
                        for col in source_pk
                    )
                    ctes.append(
                        f"{spec.alias} AS (SELECT {key_select}, {metadata_select} "
                        f"FROM {source_keys_sql} sk LEFT JOIN {spec.metadata_sql} sm USING (_position))"
                    )
                    predicates = " AND ".join(
                        f"k.{quote_identifier(col)} = {spec.alias}.{quote_identifier(col)}"
                        for col in source_pk
                    )
                    joins.append(f"LEFT JOIN {spec.alias} ON {predicates}")
            seen_metadata_aliases.add(spec.alias)

        for spec in metadata_specs:
            for col in spec.source_columns:
                internal = metadata_internal_names[(spec.alias, col)]
                pending_columns.append(
                    {
                        "namespace": "metadata",
                        "base_name": col,
                        "qualified_name": f"metadata.{spec.artifact_id}.{col}",
                        "sql_expr": f"{spec.alias}.{quote_identifier(internal)}",
                        "source_artifact_id": spec.artifact_id,
                    }
                )

        view_columns, ambiguous_columns, mapping = _assign_view_output_names(
            pending_columns  # type: ignore[arg-type]
        )
        for column in view_columns:
            select_exprs.append(
                f"{column.sql_expr} AS {quote_identifier(column.output_name)}"
            )
        sql = (
            "WITH "
            + ", ".join(ctes)
            + " SELECT "
            + ", ".join(select_exprs)
            + " FROM keys k "
            + " ".join(joins)
        )
        return ArtifactViewSQL(
            sql=sql,
            columns=view_columns,
            structural_columns=STRUCTURAL_COLUMNS,
            key_columns=tuple(
                col.output_name for col in view_columns if col.namespace == "key"
            ),
            data_columns=tuple(
                col.output_name for col in view_columns if col.namespace == "data"
            ),
            metadata_columns=tuple(
                col.output_name for col in view_columns if col.namespace == "metadata"
            ),
            output_columns=tuple(col.output_name for col in view_columns),
            ambiguous_columns=ambiguous_columns,
            mapping=mapping,
        )

    def _build_merged_artifact_view_sql(
        self,
        artifact: BaseArtifact,
        *,
        metadata_mode: MetadataMode,
        include_data: bool,
    ) -> ArtifactViewSQL:
        """Build a relational view for a keys-only ``merged_key`` artifact.

        A merged artifact owns only its regenerated keys/structural table. Its
        representation data and inherited metadata remain on the disjoint basis
        branches. Querying therefore resolves each branch as a normal artifact,
        unions the resolved non-structural columns by primary key, and joins that
        union onto the merged artifact's own key positions.

        ``metadata_mode='local'`` deliberately does not expose basis metadata:
        local means local to the merged artifact itself. The current merge
        operation writes no local metadata, so that mode returns no metadata.
        ``metadata_mode='full'`` resolves full metadata independently on every
        branch before unioning them.
        """
        basis_ids = basis_artifact_ids(artifact)
        if len(basis_ids) < 2:
            raise QueryError(
                f"Merged artifact {artifact.artifact_id} must have at least two basis artifacts."
            )

        bases = [self.project.get_artifact(artifact_id) for artifact_id in basis_ids]
        current_pk = tuple(str(col) for col in artifact.primary_key)

        branch_metadata_mode: MetadataMode = (
            "full" if metadata_mode == "full" else "none"
        )
        branch_views = [
            self.build_artifact_view_sql(
                basis,
                metadata_mode=branch_metadata_mode,
                include_data=include_data,
            )
            for basis in bases
        ]

        for basis, branch in zip(bases, branch_views, strict=True):
            if tuple(branch.key_columns) != current_pk:
                raise QueryError(
                    f"Merged artifact {artifact.artifact_id} expects primary key "
                    f"{list(current_pk)}, but basis {basis.artifact_id} resolves "
                    f"keys {list(branch.key_columns)}."
                )

        def signature(view: ArtifactViewSQL) -> tuple[tuple[str, str, str], ...]:
            # Qualified metadata names include the owning artifact ID, which may
            # legitimately differ between disjoint branches. The UNION requires
            # matching logical namespaces/base names and matching resolved output
            # names; the latter still catches ambiguity-induced incompatibility.
            return tuple(
                (
                    column.namespace,
                    column.base_name,
                    column.output_name,
                )
                for column in view.columns
                if column.namespace != "key"
            )

        expected_signature = signature(branch_views[0])
        for basis, branch in zip(bases[1:], branch_views[1:], strict=True):
            actual_signature = signature(branch)
            if actual_signature != expected_signature:
                raise QueryError(
                    "Merged artifacts require compatible effective data/metadata schemas. "
                    f"Basis {bases[0].artifact_id} resolves {expected_signature}, while "
                    f"basis {basis.artifact_id} resolves {actual_signature}."
                )

        key_sql = _parquet_dataset_expr(artifact.keys_dir)
        ctes: list[str] = [f"keys AS (SELECT * FROM {key_sql})"]
        joins: list[str] = []
        select_exprs: list[str] = [
            f"k.{quote_identifier(col)} AS {quote_identifier(col)}"
            for col in STRUCTURAL_COLUMNS
        ]
        pending_columns: list[dict[str, str | None]] = []

        for col in current_pk:
            pending_columns.append(
                {
                    "namespace": "key",
                    "base_name": col,
                    "qualified_name": f"key.{col}",
                    "sql_expr": f"k.{quote_identifier(col)}",
                    "source_artifact_id": str(artifact.artifact_id),
                }
            )

        non_key_columns = [
            column for column in branch_views[0].columns if column.namespace != "key"
        ]
        if non_key_columns:
            branch_selects: list[str] = []
            for branch in branch_views:
                selected_names = [
                    *current_pk,
                    *[col.output_name for col in non_key_columns],
                ]
                selected_sql = ", ".join(
                    f"b.{quote_identifier(name)} AS {quote_identifier(name)}"
                    for name in selected_names
                )
                branch_selects.append(f"SELECT {selected_sql} FROM ({branch.sql}) b")
            ctes.append("resolved AS (" + " UNION ALL ".join(branch_selects) + ")")
            predicates = " AND ".join(
                f"k.{quote_identifier(col)} = r.{quote_identifier(col)}"
                for col in current_pk
            )
            joins.append(f"LEFT JOIN resolved r ON {predicates}")

            for index, column in enumerate(non_key_columns):
                source_ids = {
                    branch.columns[len(current_pk) + index].source_artifact_id
                    for branch in branch_views
                }
                common_source_id = (
                    next(iter(source_ids)) if len(source_ids) == 1 else None
                )
                pending_columns.append(
                    {
                        "namespace": column.namespace,
                        "base_name": column.base_name,
                        "qualified_name": column.qualified_name,
                        "sql_expr": f"r.{quote_identifier(column.output_name)}",
                        "source_artifact_id": common_source_id,
                    }
                )

        # A merged artifact created by core.merge owns no metadata. If future
        # operations attach local metadata directly to one, expose it here while
        # keeping branch metadata exclusive to metadata_mode='full'.
        if metadata_mode in {"local", "full"}:
            local_metadata_sql = self._local_metadata_relation_sql(artifact)
            if local_metadata_sql is not None:
                local_columns = tuple(
                    col
                    for col in self._relation_columns(local_metadata_sql)
                    if not col.startswith("_")
                )
                if local_columns:
                    selected = ", ".join(
                        ["_position", *[quote_identifier(col) for col in local_columns]]
                    )
                    ctes.append(
                        f"merge_local_metadata AS (SELECT {selected} FROM {local_metadata_sql})"
                    )
                    joins.append("LEFT JOIN merge_local_metadata mlm USING (_position)")
                    for col in local_columns:
                        pending_columns.append(
                            {
                                "namespace": "metadata",
                                "base_name": col,
                                "qualified_name": f"metadata.{artifact.artifact_id}.{col}",
                                "sql_expr": f"mlm.{quote_identifier(col)}",
                                "source_artifact_id": str(artifact.artifact_id),
                            }
                        )

        # Type narrowing: _assign_view_output_names accepts dictionaries of
        # strings, while source_artifact_id is intentionally optional.
        raw_pending = [dict(item) for item in pending_columns]
        view_columns, ambiguous_columns, mapping = _assign_view_output_names(
            raw_pending
        )  # type: ignore[arg-type]
        for column in view_columns:
            select_exprs.append(
                f"{column.sql_expr} AS {quote_identifier(column.output_name)}"
            )

        sql = (
            "WITH "
            + ", ".join(ctes)
            + " SELECT "
            + ", ".join(select_exprs)
            + " FROM keys k "
            + " ".join(joins)
        )

        key_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "key"
        )
        data_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "data"
        )
        metadata_columns = tuple(
            col.output_name for col in view_columns if col.namespace == "metadata"
        )
        output_columns = tuple(col.output_name for col in view_columns)
        return ArtifactViewSQL(
            sql=sql,
            columns=view_columns,
            structural_columns=STRUCTURAL_COLUMNS,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            output_columns=output_columns,
            ambiguous_columns=ambiguous_columns,
            mapping=mapping,
        )

    def query_columns(
        self,
        artifact: BaseArtifact,
        *,
        metadata_mode: MetadataMode = "none",
    ) -> dict[str, Any]:
        view = self.build_artifact_view_sql(
            artifact,
            metadata_mode=metadata_mode,
            include_data=True,
        )
        return {
            "structural": list(view.structural_columns),
            "key": list(view.key_columns),
            "data": list(view.data_columns),
            "metadata": list(view.metadata_columns),
            "output": list(view.output_columns),
            "ambiguous": {
                key: list(value) for key, value in view.ambiguous_columns.items()
            },
            "columns": [
                {
                    "namespace": column.namespace,
                    "base_name": column.base_name,
                    "qualified_name": column.qualified_name,
                    "output_name": column.output_name,
                    "source_artifact_id": column.source_artifact_id,
                }
                for column in view.columns
            ],
            "mapping": dict(view.mapping),
        }

    def _select_namespace_columns(
        self,
        view: ArtifactViewSQL,
        *,
        namespace: str,
        requested: ColumnSelect,
        label: str,
    ) -> list[str]:
        columns = [column for column in view.columns if column.namespace == namespace]
        if requested is True:
            return [column.output_name for column in columns]
        if requested is False:
            return []

        requested_names = (
            [str(requested)]
            if isinstance(requested, str)
            else [str(col) for col in requested]
        )
        selected: list[str] = []

        def add(output_name: str) -> None:
            if output_name not in selected:
                selected.append(output_name)

        for name in requested_names:
            exact_qualified = [
                column for column in columns if column.qualified_name == name
            ]
            if len(exact_qualified) == 1:
                add(exact_qualified[0].output_name)
                continue

            exact_output = [column for column in columns if column.output_name == name]
            if len(exact_output) == 1:
                add(exact_output[0].output_name)
                continue

            base_matches = [column for column in columns if column.base_name == name]
            if len(base_matches) == 1:
                add(base_matches[0].output_name)
                continue
            if len(base_matches) > 1:
                options = [column.qualified_name for column in base_matches]
                raise QueryError(
                    f"Requested {label} {name!r} is ambiguous. Use one of: {options}."
                )

            available = sorted({column.output_name for column in columns})
            raise QueryError(
                f"Requested {label} is not available: {name!r}. "
                f"Available {label}: {available}."
            )

        return selected

    def _view_selected_columns(
        self,
        view: ArtifactViewSQL,
        *,
        key_columns: ColumnSelect,
        data_columns: ColumnSelect,
        metadata_columns: ColumnSelect,
    ) -> list[str]:
        selected: list[str] = []
        for output_name in self._select_namespace_columns(
            view,
            namespace="key",
            requested=key_columns,
            label="key columns",
        ):
            if output_name not in selected:
                selected.append(output_name)
        for output_name in self._select_namespace_columns(
            view,
            namespace="data",
            requested=data_columns,
            label="data columns",
        ):
            if output_name not in selected:
                selected.append(output_name)
        for output_name in self._select_namespace_columns(
            view,
            namespace="metadata",
            requested=metadata_columns,
            label="metadata columns",
        ):
            if output_name not in selected:
                selected.append(output_name)
        return selected

    def _artifact_select_sql(
        self,
        artifact: BaseArtifact,
        *,
        key_columns: ColumnSelect,
        data_columns: ColumnSelect,
        metadata_columns: ColumnSelect,
        metadata_mode: MetadataMode,
        where: str | None,
        order_by: str | Sequence[str] | None,
        limit: int | None,
        positions: Sequence[int] | None,
        include_position: bool,
        registered: list[str],
    ) -> str:
        _validate_metadata_query_request(
            metadata_mode=metadata_mode,
            metadata_columns=metadata_columns,
        )
        view = self.build_artifact_view_sql(
            artifact,
            metadata_mode=metadata_mode,
            include_data=_columns_requested(data_columns),
        )
        selected = self._view_selected_columns(
            view,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
        )

        ctes = [f"artifact_view AS ({view.sql})"]
        from_sql = "artifact_view v"
        select_exprs: list[str] = []

        if positions is not None:
            temp = pd.DataFrame(
                {
                    "_position": [int(pos) for pos in positions],
                    "_request_order": list(range(len(positions))),
                }
            )
            temp_name = _temp_name("query_positions")
            self.con.register(temp_name, temp)
            registered.append(temp_name)
            ctes.append(f"positions AS (SELECT * FROM {quote_identifier(temp_name)})")
            from_sql += " JOIN positions p USING (_position)"
            select_exprs.append("p._request_order AS _request_order")

        if include_position:
            select_exprs.append("v._position AS _position")
        for col in selected:
            select_exprs.append(f"v.{quote_identifier(col)} AS {quote_identifier(col)}")
        if not select_exprs:
            select_exprs.append("v._position AS _position")

        sql = (
            "WITH "
            + ", ".join(ctes)
            + " SELECT "
            + ", ".join(select_exprs)
            + " FROM "
            + from_sql
        )
        if where:
            sql += f" WHERE ({where})"

        order = _order_clause(order_by)
        if order:
            sql += f" ORDER BY {order}"
        elif positions is not None:
            sql += " ORDER BY _request_order"
        else:
            sql += " ORDER BY _position"

        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return sql

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def artifact_query(
        self,
        artifact: BaseArtifact,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        limit: int | None = None,
        positions: Sequence[int] | None = None,
        include_position: bool = False,
    ) -> pd.DataFrame:
        registered: list[str] = []
        try:
            sql = self._artifact_select_sql(
                artifact,
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                limit=limit,
                positions=positions,
                include_position=include_position,
                registered=registered,
            )
            frame = self.con.execute(sql).fetchdf()
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise QueryError(f"Artifact query failed: {exc}") from exc
        finally:
            for name in reversed(registered):
                try:
                    self.con.unregister(name)
                except Exception:
                    pass
        if "_request_order" in frame.columns:
            frame = frame.drop(columns=["_request_order"])
        return frame

    def artifact_query_batches(
        self,
        artifact: BaseArtifact,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        limit: int | None = None,
        positions: Sequence[int] | None = None,
        include_position: bool = False,
        batch_size: int = 100_000,
        streaming_mode: StreamingMode = "auto",
    ) -> Iterable[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if streaming_mode not in {"auto", "arrow", "paged"}:
            raise ValueError("streaming_mode must be 'auto', 'arrow', or 'paged'.")

        registered: list[str] = []
        try:
            base_sql = self._artifact_select_sql(
                artifact,
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                limit=limit,
                positions=positions,
                include_position=True,
                registered=registered,
            )

            def clean(frame: pd.DataFrame) -> pd.DataFrame:
                drop = []
                if not include_position and "_position" in frame.columns:
                    drop.append("_position")
                if "_request_order" in frame.columns:
                    drop.append("_request_order")
                return frame.drop(columns=drop) if drop else frame

            def yield_arrow() -> Iterable[pd.DataFrame]:
                reader = self.con.execute(base_sql).to_arrow_reader(
                    batch_size=int(batch_size)
                )
                for batch in reader:
                    yield clean(batch.to_pandas())

            def yield_paged() -> Iterable[pd.DataFrame]:
                offset = 0
                emitted = 0
                hard_limit = None if limit is None else int(limit)
                while hard_limit is None or emitted < hard_limit:
                    page_limit = (
                        int(batch_size)
                        if hard_limit is None
                        else min(int(batch_size), hard_limit - emitted)
                    )
                    if page_limit <= 0:
                        break
                    page_sql = (
                        f"WITH base_query AS ({base_sql}) "
                        f"SELECT * FROM base_query LIMIT {page_limit} OFFSET {offset}"
                    )
                    frame = clean(self.con.execute(page_sql).fetchdf())
                    if frame.empty:
                        break
                    emitted += len(frame)
                    offset += len(frame)
                    yield frame.reset_index(drop=True)
                    if len(frame) < page_limit:
                        break

            if streaming_mode == "paged":
                yield from yield_paged()
            elif streaming_mode == "arrow":
                yield from yield_arrow()
            else:
                emitted_any = False
                try:
                    reader = self.con.execute(base_sql).to_arrow_reader(
                        batch_size=int(batch_size)
                    )
                    for batch in reader:
                        emitted_any = True
                        yield clean(batch.to_pandas())
                except Exception:
                    if emitted_any:
                        raise
                    yield from yield_paged()
        finally:
            for name in reversed(registered):
                try:
                    self.con.unregister(name)
                except Exception:
                    pass

    def project_sql(
        self,
        artifacts: Sequence[BaseArtifact | str],
        sql: str,
    ) -> pd.DataFrame:
        """Run SQL against explicitly registered artifact views.

        Each entry in ``artifacts`` is resolved through ``project.get_artifact(...)`` and
        registered under ``str(entry)``. Passing a string registers that exact string,
        which may be an artifact id or alias. Passing an artifact object registers its
        artifact_id.

        Registered views expose keys, table-backed data, full metadata lineage, and
        structural columns.
        """
        if not artifacts:
            raise QueryError("project_sql requires at least one artifact to register.")

        registered: list[str] = []

        try:
            for ref in artifacts:
                sql_name = str(ref)
                if sql_name in registered:
                    raise QueryError(
                        f"Duplicate project_sql artifact name: {sql_name!r}."
                    )

                artifact = self.project.get_artifact(ref)
                view = self.build_artifact_view_sql(
                    artifact,
                    metadata_mode="full",
                    include_data=True,
                )

                self.con.execute(
                    f"CREATE OR REPLACE TEMP VIEW {quote_identifier(sql_name)} AS {view.sql}"
                )
                registered.append(sql_name)

            return self.con.execute(sql).fetchdf()

        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"Project SQL failed: {exc}") from exc
        finally:
            for name in reversed(registered):
                try:
                    self.con.execute(f"DROP VIEW IF EXISTS {quote_identifier(name)}")
                except Exception:
                    pass
