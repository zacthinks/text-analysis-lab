"""Base artifact handles and shared artifact helpers for TextAnalysisLab (TeAL)."""

from __future__ import annotations

import json
import numbers
import warnings
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, get_args
from collections.abc import Sequence, Iterable

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import (
    ArtifactError,
    DataInheritanceError,
    IncompleteArtifactError,
    MissingDataComponentError,
    UnsupportedArtifactOperationError,
)
from text_analysis_lab.core.kwic import KWICResult, keyword_in_context
from text_analysis_lab.core.lineage import (
    find_data_artifact,
    iter_metadata_lineage_sources,
)
from text_analysis_lab.core.storage import ArtifactStorage
from text_analysis_lab.core.types import (
    ArtifactStatus,
    ArtifactType,
    ColumnSelect,
    MetadataMode,
    QueryForm,
    StreamingMode,
)
from text_analysis_lab.core.utils import resolve_names, str_keys

if TYPE_CHECKING:
    from text_analysis_lab.analysis.accessor import ArtifactAnalysis
    from text_analysis_lab.visualization.accessor import ArtifactVisualization
    from text_analysis_lab.core.project import Project


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------


KeyRangeFilter = dict[str, tuple[int, ...]]

METADATA_MODES: tuple[str, ...] = get_args(MetadataMode)
ARTIFACT_STATUSES: tuple[str, ...] = get_args(ArtifactStatus)
QUERY_FORMS: tuple[str, ...] = get_args(QueryForm)
STREAMING_MODES: tuple[str, ...] = get_args(StreamingMode)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_artifact_status(status: object) -> None:
    if status not in ARTIFACT_STATUSES:
        raise ArtifactError(f"Invalid artifact status: {status!r}")


def _validate_query_form(form: object) -> None:
    if form not in QUERY_FORMS:
        raise ValueError(f"Invalid query form: {form!r}")


def _validate_sample_request(
    *, sample_n: int | None, sample_frac: float | None
) -> None:
    if sample_n is not None and sample_frac is not None:
        raise ValueError("Cannot specify both sample_n and sample_frac.")
    if sample_n is not None and sample_n < 0:
        raise ValueError("sample_n must be non-negative.")
    if sample_frac is not None and not (0 <= float(sample_frac) <= 1):
        raise ValueError("sample_frac must be between 0 and 1.")


def _columns_requested(value: ColumnSelect) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        return True
    return len(value) > 0


# ---------------------------------------------------------------------------
# File, path, and key helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _component_path(
    artifact_dir: Path, descriptor: dict[str, Any], name: str
) -> Path | None:
    component = descriptor.get("components", {}).get(name)
    if not component:
        return None
    rel = component.get("path")
    if not rel:
        return None
    return artifact_dir / rel


def _part_name(batch: int, suffix: str) -> str:
    return f"part-{int(batch):06d}.{suffix}"


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _is_int_key_value(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _records_with_str_keys(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [str_keys(record) for record in frame.to_dict(orient="records")]


def _sample_positions(
    *,
    total: int,
    candidates: Sequence[int] | None = None,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = None,
) -> list[int]:
    total = max(int(total), 0)
    if candidates is None:
        pool_size = total
    else:
        pool = np.asarray([int(pos) for pos in candidates], dtype=int)
        pool_size = len(pool)

    if pool_size == 0:
        return []

    if n is None and frac is None:
        raise ValueError("Either sample_n or sample_frac must be provided.")

    if frac is not None:
        if not 0 <= float(frac) <= 1:
            raise ValueError("sample_frac must be between 0 and 1.")
        sample_size = int(round(pool_size * float(frac)))
    else:
        sample_size = int(n or 0)

    if sample_size < 0:
        raise ValueError("sample_n must be non-negative.")

    sample_size = min(sample_size, pool_size)
    if sample_size == 0:
        return []

    rng = np.random.default_rng(random_state)
    if candidates is None:
        # Passing an integer population avoids allocating np.arange(total) for
        # large artifacts when no pre-filtered candidate list is needed.
        chosen = rng.choice(pool_size, size=sample_size, replace=False)
    else:
        chosen = rng.choice(pool, size=sample_size, replace=False)
    return [int(pos) for pos in chosen]


def _resolve_query_positions(
    *,
    total: int,
    positions: Sequence[int] | None,
    sample_n: int | None,
    sample_frac: float | None,
    random_state: int | None,
) -> list[int] | None:
    if sample_n is None and sample_frac is None:
        return None if positions is None else [int(pos) for pos in positions]
    return _sample_positions(
        total=total,
        candidates=positions,
        n=sample_n,
        frac=sample_frac,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Column merging helpers
# ---------------------------------------------------------------------------


def _data_column_rename_map(
    columns: Iterable[str],
    *,
    reserved: Sequence[str],
) -> dict[str, str]:
    """Map raw externally-fetched data column names onto names that don't
    collide with ``reserved`` (already-finalized, non-renameable) names or
    with each other."""
    columns = list(columns)
    qualified = [f"data.{col}" for col in columns]
    output_names, _ = resolve_names(
        columns,
        qualified,
        reserved,
        error_cls=ArtifactError,
    )
    return dict(zip(columns, output_names, strict=True))


def _merge_info_and_data_frames(
    info_frame: pd.DataFrame,
    data_frame: pd.DataFrame,
    *,
    include_position: bool,
) -> pd.DataFrame:
    info = info_frame.reset_index(drop=True).copy()
    data = data_frame.reset_index(drop=True).copy()

    if "_position" in data.columns:
        data = data.drop(columns=["_position"])

    if len(data) not in {0, len(info)}:
        raise ArtifactError(
            f"Data frame row count {len(data)} does not match info row count {len(info)}."
        )

    if len(data) == 0:
        out = info
    else:
        rename = _data_column_rename_map(data.columns, reserved=list(info.columns))
        data = data.rename(columns=rename)
        out = pd.concat([info, data], axis=1)

    if not include_position and "_position" in out.columns:
        out = out.drop(columns=["_position"])
    return out


def _merge_info_and_data_records(
    info_frame: pd.DataFrame,
    data_records: Sequence[dict[str, Any]],
    *,
    data_columns: Sequence[str],
    include_position: bool,
) -> list[dict[str, Any]]:
    info_records = _records_with_str_keys(info_frame)
    data_records = [str_keys(record) for record in data_records]

    if len(data_records) not in {0, len(info_records)}:
        raise ArtifactError(
            f"Data record count {len(data_records)} does not match info record count {len(info_records)}."
        )

    if len(data_records) == 0 or not data_columns:
        out = info_records
    else:
        raw_columns = [str(col) for col in data_columns if str(col) != "_position"]
        rename = _data_column_rename_map(raw_columns, reserved=list(info_frame.columns))
        out = []
        for info, data in zip(info_records, data_records, strict=True):
            merged = dict(info)
            for col in raw_columns:
                merged[rename[col]] = data.get(col)
            out.append(merged)

    if not include_position:
        for record in out:
            record.pop("_position", None)
    return out


# ---------------------------------------------------------------------------
# Artifact-reference and context-window helpers
# ---------------------------------------------------------------------------


def resolve_artifact_ref(
    project: "Project",
    artifact_ref: "BaseArtifact | str | None",
    *,
    default: "BaseArtifact | None" = None,
    parameter_name: str = "artifact",
) -> "BaseArtifact":
    """Resolve an artifact id/alias or artifact object into a BaseArtifact."""
    if artifact_ref is None:
        if default is None:
            raise ArtifactError(f"{parameter_name} cannot be None.")
        return default

    if isinstance(artifact_ref, str):
        return project.get_artifact(artifact_ref)

    if isinstance(artifact_ref, BaseArtifact):
        if artifact_ref.project != project:
            raise ArtifactError(f"{parameter_name} must belong to the same project.")
        return artifact_ref

    raise ArtifactError(
        f"{parameter_name} must be None, an artifact id/alias, or a BaseArtifact."
    )


def _context_key_filter(
    request_key: dict[str, int],
    context_primary_key: Sequence[str],
) -> KeyRangeFilter:
    """Return a KeyRangeFilter for a context artifact from a request key and context primary key."""
    out: KeyRangeFilter = {}

    for raw_col in context_primary_key:
        col = str(raw_col)

        if col in request_key:
            out[col] = (int(request_key[col]),)
            continue

        start_col = f"{col}_start"
        end_col = f"{col}_end"
        has_start = start_col in request_key
        has_end = end_col in request_key

        if has_start != has_end:
            raise ArtifactError(
                f"Malformed span key for {col!r}: expected both "
                f"{start_col!r} and {end_col!r}."
            )

        if has_start and has_end:
            start = int(request_key[start_col])
            end = int(request_key[end_col])
            if end < start:
                raise ArtifactError(
                    f"Malformed span key for {col!r}: {start_col}={start} "
                    f"is greater than {end_col}={end}."
                )
            out[col] = (start, end)

    return out


def _resolved_key_column_mapping(
    artifact: "BaseArtifact",
    key_filter: KeyRangeFilter,
    *,
    metadata_mode: MetadataMode,
) -> dict[str, str]:
    query_info = artifact.query_columns(metadata_mode=metadata_mode)
    mapping = dict(query_info.get("mapping", {}))

    out: dict[str, str] = {}
    for col in key_filter:
        resolved = mapping.get(f"key.{col}")
        if resolved is None:
            raise ArtifactError(
                f"Cannot resolve context key column {col!r} for artifact "
                f"{artifact.artifact_id}."
            )
        out[col] = str(resolved)

    return out


def _key_filter_where_clause(
    key_filter: KeyRangeFilter,
    *,
    key_column_mapping: dict[str, str],
) -> str:
    if not key_filter:
        raise ArtifactError("Cannot build context query from an empty key filter.")

    parts: list[str] = []

    for col, values in key_filter.items():
        resolved_col = key_column_mapping.get(col)
        if resolved_col is None:
            raise ArtifactError(f"Cannot resolve query column for key column {col!r}.")

        quoted_col = _quote_identifier(resolved_col)

        if len(values) == 1:
            parts.append(f"{quoted_col} = {int(values[0])}")
        elif len(values) == 2:
            start, end = values
            parts.append(f"{quoted_col} >= {int(start)} AND {quoted_col} <= {int(end)}")
        else:
            raise ArtifactError(
                f"Invalid key filter for {col!r}: expected one or two values."
            )

    return " AND ".join(parts)


def _expand_context_positions(
    matched_positions: Sequence[int],
    *,
    before: int,
    after: int,
    total: int,
    include_focus: bool = True,
) -> list[int]:
    if not matched_positions:
        return []

    matched = sorted({int(pos) for pos in matched_positions})
    low = min(matched)
    high = max(matched)

    before_positions = list(range(max(0, low - int(before)), low))
    after_positions = list(range(high + 1, min(total, high + int(after) + 1)))

    if include_focus:
        return [*before_positions, *matched, *after_positions]
    return [*before_positions, *after_positions]


def _extract_single(records: list[Any]) -> Any:
    if not records:
        raise ArtifactError("form='single' received no records.")
    if len(records) > 1:
        warnings.warn(
            "form='single' received more than one record; returning the first record.",
            stacklevel=2,
        )
    return records[0]


# ---------------------------------------------------------------------------
# Base artifact handle
# ---------------------------------------------------------------------------


class BaseArtifact(ABC):
    """Lightweight handle for a persisted TeAL artifact.

    ``query`` is the core public access method. It resolves a row set through the
    project query engine, asks the data-owning artifact for non-table data when
    necessary, and materializes the result in a requested form.
    """

    artifact_type: ClassVar[ArtifactType]

    def __init__(
        self,
        project: "Project",
        artifact_dir: str | Path,
    ) -> None:
        self.project = project
        self.artifact_dir = Path(artifact_dir)
        self.storage = ArtifactStorage.open(self.artifact_dir)
        self.descriptor = _read_json(self.storage.descriptor_path)
        self.data_artifact = find_data_artifact(self)

    def __str__(self) -> str:
        """Return a compact human-readable artifact summary."""
        display_name = self.aliases[0] if self.aliases else self.label
        rows = "?" if self.n_rows is None else f"{self.n_rows:,}"
        if self.artifact_type.value in {"sparse_matrix", "dense_matrix"}:
            shape = f"{rows} x {len(self.get_data_columns()):,}"
        else:
            shape = f"{rows} rows"
        return f"{display_name} [{self.artifact_type.value}: {shape}]"

    def __repr__(self) -> str:
        """Return an informative notebook/debug representation."""
        aliases = self.aliases
        lineage = self.descriptor.get("lineage")
        lineage_mode = lineage.get("lineage_mode") if isinstance(lineage, dict) else None
        basis: Any = None
        if isinstance(lineage, dict):
            basis = lineage.get("basis_artifact_ids", lineage.get("basis_artifact_id"))
        data_columns = self.get_data_columns()
        if len(data_columns) <= 6:
            data_summary: Any = data_columns
        else:
            data_summary = f"{len(data_columns):,} columns"
        parts = [
            f"id={self.artifact_id!r}",
            f"label={self.label!r}",
        ]
        if aliases:
            parts.append(f"alias={aliases[0]!r}")
        parts.extend(
            [
                f"type={self.artifact_type.value!r}",
                f"rows={self.n_rows!r}",
                f"key={self.primary_key!r}",
                f"data={data_summary!r}",
                f"lineage={lineage_mode!r}",
            ]
        )
        if basis:
            parts.append(f"basis={basis!r}")
        if self.status != "complete":
            parts.append(f"status={self.status!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _repr_html_(self) -> str:
        """Render a compact artifact card in Jupyter frontends."""
        from html import escape

        aliases = self.aliases
        display_name = aliases[0] if aliases else self.label
        lineage = self.descriptor.get("lineage")
        lineage_mode = lineage.get("lineage_mode") if isinstance(lineage, dict) else None
        data_columns = self.get_data_columns()
        data_text = (
            ", ".join(data_columns)
            if len(data_columns) <= 6
            else f"{len(data_columns):,} columns"
        )
        rows = "?" if self.n_rows is None else f"{self.n_rows:,}"
        fields = [
            ("Artifact", display_name),
            ("ID", self.artifact_id),
            ("Type", self.artifact_type.value),
            ("Rows", rows),
            ("Primary key", ", ".join(self.primary_key) or "-"),
            ("Data", data_text or "-"),
            ("Lineage", str(lineage_mode or "-")),
            ("Status", self.status),
        ]
        rows_html = "".join(
            f"<tr><th style='text-align:left;padding:2px 10px 2px 0'>{escape(str(key))}</th>"
            f"<td style='text-align:left;padding:2px 0'>{escape(str(value))}</td></tr>"
            for key, value in fields
        )
        return f"<table>{rows_html}</table>"

    def __getitem__(self, item: int | slice) -> Any:
        """Return row(s) by artifact-local integer position.

        ``artifact[i]`` is shorthand for ``artifact.get_by_position(i)``.
        ``artifact[start:stop]`` is shorthand for querying positions in that
        half-open Python slice range.
        """
        if isinstance(item, slice):
            if item.step == 0:
                raise ValueError("slice step cannot be zero.")

            n_rows = self.n_rows
            if n_rows is None:
                raise ArtifactError(
                    f"Cannot slice artifact {self.artifact_id}: n_rows is unknown."
                )

            positions = list(range(*item.indices(int(n_rows))))
            return self.query(
                positions=positions,
                form="records",
                iter_batches=False,
            )

        if not isinstance(item, numbers.Integral) or isinstance(item, bool):
            raise TypeError(
                f"Artifact indices must be integers or slices, not {type(item).__name__}."
            )

        position = int(item)
        if position < 0:
            n_rows = self.n_rows
            if n_rows is None:
                raise ArtifactError(
                    f"Cannot resolve negative index for artifact {self.artifact_id}: "
                    "n_rows is unknown."
                )
            position += int(n_rows)

        return self.get_by_position(position)

    # ------------------------------------------------------------------
    # Descriptor-backed identity and storage paths
    # ------------------------------------------------------------------

    @property
    def artifact_id(self) -> str:
        return str(self.descriptor["artifact_id"])

    @property
    def label(self) -> str:
        value = self.descriptor.get("label")
        if not isinstance(value, str) or not value:
            raise ArtifactError(
                f"Artifact {self.artifact_id} descriptor is missing its required label."
            )
        return value

    @property
    def status(self) -> ArtifactStatus:
        # artifact.json retains status as a human-facing record, but the catalog
        # is authoritative for live project state.
        row = self.project.catalog.resolve_artifact(
            self.artifact_id, include_deleted=True
        )
        value = row["status"]
        _validate_artifact_status(value)
        return cast(ArtifactStatus, value)

    @property
    def components(self) -> dict[str, Any]:
        return dict(self.descriptor.get("components", {}))

    @property
    def primary_key(self) -> list[str]:
        return list(self.descriptor.get("primary_key", []))

    @property
    def n_rows(self) -> int | None:
        value = self.descriptor.get("n_rows")
        return int(value) if value is not None else None

    @property
    def keys_dir(self) -> Path:
        return self.storage.keys_dir

    @property
    def data_dir(self) -> Path | None:
        if "data" not in self.components:
            return None
        return self.storage.data_dir

    @property
    def metadata_dir(self) -> Path | None:
        if "metadata" not in self.components:
            return None
        return self.storage.metadata_dir

    def refresh(self) -> None:
        """Reload this artifact handle's descriptor and data-owner resolution."""
        self.descriptor = _read_json(self.storage.descriptor_path)
        self.data_artifact = find_data_artifact(self)

    # ------------------------------------------------------------------
    # Lifecycle and project/catalog delegation
    # ------------------------------------------------------------------

    def require_complete(self) -> None:
        if self.status != "complete":
            raise IncompleteArtifactError(
                f"Artifact {self.artifact_id} has status={self.status!r}, not 'complete'."
            )

    @property
    def analysis(self) -> "ArtifactAnalysis":
        """Return Analytic Methods bound to this artifact.

        The accessor is ephemeral: calling ``artifact.analysis.<method>(...)``
        does not create a new Artifact unless that method explicitly says so.
        """
        from text_analysis_lab.analysis.accessor import ArtifactAnalysis

        return ArtifactAnalysis(self)

    @property
    def visualize(self) -> "ArtifactVisualization":
        """Return scalable visualization methods bound to this artifact."""
        from text_analysis_lab.visualization.accessor import ArtifactVisualization

        return ArtifactVisualization(self)

    @property
    def aliases(self) -> list[str]:
        """Return project-level aliases for this artifact."""
        return self.project.catalog.aliases_for_artifact(self.artifact_id)

    def add_alias(self, alias: str) -> None:
        """Add a project-level alias for this artifact."""
        self.project.catalog.add_artifact_alias(self.artifact_id, alias)

    def remove_alias(self, alias: str) -> None:
        """Remove a project-level alias."""
        self.project.catalog.remove_artifact_alias(alias)

    @property
    def operation_id(self) -> str | None:
        """Return the operation that created this artifact, if recorded."""
        value = self.descriptor.get("operation_id")
        if value is not None:
            return str(value)
        # Read older descriptors defensively if encountered.
        created_by = self.descriptor.get("created_by", {})
        if isinstance(created_by, dict) and created_by.get("operation_id") is not None:
            return str(created_by["operation_id"])
        return None

    def resume(self) -> Any:
        """Resume the operation that produced this artifact."""
        operation_id = self.operation_id
        if operation_id is None:
            raise ArtifactError(
                f"Artifact {self.artifact_id} does not record a creating operation."
            )
        return self.project.resume_operation(operation_id)

    # ------------------------------------------------------------------
    # Component availability and key/position lookup
    # ------------------------------------------------------------------

    def has_own_data(self) -> bool:
        return "data" in self.components

    def has_metadata(self) -> bool:
        return self.metadata_dir is not None and self.metadata_dir.exists()

    def _normalize_key(self, key: Any) -> dict[str, int]:
        pk = self.primary_key
        if not pk:
            raise ArtifactError(f"Artifact {self.artifact_id} has no primary_key.")

        if isinstance(key, dict):
            missing = [col for col in pk if col not in key]
            if missing:
                raise KeyError(f"Missing key columns for {self.artifact_id}: {missing}")
            values = {col: key[col] for col in pk}
        elif len(pk) == 1:
            values = {pk[0]: key}
        elif isinstance(key, tuple) or isinstance(key, list):
            if len(key) != len(pk):
                raise KeyError(
                    f"Expected {len(pk)} key values for {pk}; got {len(key)}."
                )
            values = dict(zip(pk, key, strict=True))
        else:
            raise KeyError(
                f"Composite key required for {self.artifact_id}; expected {pk}."
            )

        bad = {
            col: value for col, value in values.items() if not _is_int_key_value(value)
        }
        if bad:
            raise KeyError(
                f"TeAL primary keys must be integers. Invalid key values for "
                f"{self.artifact_id}: {bad}"
            )
        return {col: int(value) for col, value in values.items()}

    def position_by_key(self, key: Any) -> int:
        """Return the artifact-local integer position for a primary key."""
        normalized = self._normalize_key(key)
        return self.project.query.position_by_key(self, normalized)

    # ------------------------------------------------------------------
    # Data hooks for concrete artifact subclasses
    # ------------------------------------------------------------------

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        """Return owned representation data records for artifact-local positions.

        Subclasses for non-table representation data should override this. Table
        artifacts usually do not need to override it because QueryEngine can read
        their tabular data directly into the artifact SQL view.
        """
        raise UnsupportedArtifactOperationError(
            f"{self.__class__.__name__} does not expose owned data records."
        )

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        """Return owned representation data as a row-aligned DataFrame."""
        return pd.DataFrame(
            self._own_data_records_for_positions(
                positions,
                data_columns=data_columns,
            )
        )

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> Any:
        """Return owned representation data in the artifact type's native form."""
        return self._own_data_records_for_positions(
            positions,
            data_columns=data_columns,
        )

    def _require_data_artifact(self) -> "BaseArtifact":
        if self.data_artifact is None:
            raise MissingDataComponentError(
                f"Artifact {self.artifact_id} has no available data component."
            )
        return self.data_artifact

    def _query_can_include_data(self, data_columns: ColumnSelect) -> bool:
        if (
            _columns_requested(data_columns)
            and self.artifact_type in {ArtifactType.TABLE, ArtifactType.JSONL}
            and self._has_merged_relational_data()
        ):
            return True
        return (
            _columns_requested(data_columns)
            and self.data_artifact is not None
            and self.data_artifact.artifact_type
            in {ArtifactType.TABLE, ArtifactType.JSONL}
        )

    def _has_merged_relational_data(self) -> bool:
        """Return whether this keys-only artifact exposes virtual relational data.

        The historical method name is retained internally for compatibility; both
        vertical ``merged_key`` and horizontal ``joined_key`` artifacts resolve
        relational data lazily through the query engine.
        """
        descriptor = getattr(self, "descriptor", {})
        if not isinstance(descriptor, dict):
            return False
        lineage = descriptor.get("lineage", {})
        return (
            isinstance(lineage, dict)
            and lineage.get("lineage_mode") in {"merged_key", "joined_key"}
        )

    def _resolve_data_positions(self, positions: Sequence[int]) -> list[int]:
        """Return positions in ``self.data_artifact`` for current artifact positions."""
        resolved = [int(pos) for pos in positions]
        data_artifact = self._require_data_artifact()

        if data_artifact is self:
            return resolved

        return self.project.query.map_descendant_positions_to_ancestor_positions(
            self,
            data_artifact,
            resolved,
        )

    def _resolve_external_data_columns(
        self,
        data_columns: ColumnSelect,
    ) -> list[str]:
        if not _columns_requested(data_columns):
            return []

        data_artifact = self._require_data_artifact()
        available = [str(col) for col in data_artifact.get_data_columns()]
        if data_columns is True:
            return available

        if isinstance(data_columns, str):
            requested = [data_columns]
        elif isinstance(data_columns, Iterable):
            requested = [str(col) for col in data_columns]
        else:
            raise ArtifactError(
                f"Invalid data_columns argument: {data_columns!r}. "
                f"Expected True, False, a string, or an iterable of strings."
            )

        missing = [col for col in requested if col not in available]
        if missing:
            raise ArtifactError(
                f"Requested data columns are not available for artifact "
                f"{data_artifact.artifact_id}: {missing}. "
                f"Available data columns: {available}."
            )
        return requested

    def _data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        if not _columns_requested(data_columns):
            return []
        data_positions = self._resolve_data_positions(positions)
        data_artifact = self._require_data_artifact()
        return data_artifact._own_data_records_for_positions(
            data_positions,
            data_columns=data_columns,
        )

    def _data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        if not _columns_requested(data_columns):
            return pd.DataFrame()
        data_positions = self._resolve_data_positions(positions)
        data_artifact = self._require_data_artifact()
        return data_artifact._own_data_frame_for_positions(
            data_positions,
            data_columns=data_columns,
        )

    def _data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> Any:
        if not _columns_requested(data_columns):
            return None
        data_positions = self._resolve_data_positions(positions)
        data_artifact = self._require_data_artifact()
        return data_artifact._own_data_native_for_positions(
            data_positions,
            data_columns=data_columns,
        )

    def _native_from_info_and_data(
        self,
        info: pd.DataFrame,
        data: Any,
    ) -> Any:
        """Return native representation data for a query result.

        Subclasses for non-table representation data should override this. Table
        artifacts usually do not need to override it because QueryEngine can read
        their tabular data directly into the artifact SQL view.
        """
        raise UnsupportedArtifactOperationError(
            f"{self.__class__.__name__} does not expose native representation data."
        )

    # ------------------------------------------------------------------
    # Core query operation
    # ------------------------------------------------------------------

    def query_columns(
        self,
        *,
        metadata_mode: MetadataMode = "none",
    ) -> dict[str, Any]:
        """Return key/data/metadata column names available to artifact.query(...)."""
        return self.project.query.query_columns(self, metadata_mode=metadata_mode)

    def query(
        self,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        form: QueryForm = "native",
        iter_batches: bool = False,
        batch_size: int = 10_000,
        include_position: bool = False,
        streaming_mode: StreamingMode = "auto",
    ) -> Any:
        """Return rows from this artifact in a requested form.

        This is the primary public access method. ``form`` controls the returned
        shape. ``iter_batches=True`` returns an iterable rather than materializing
        the full result.
        """
        _validate_query_form(form)
        _validate_sample_request(
            sample_n=sample_n,
            sample_frac=sample_frac,
        )
        if iter_batches and form == "single" and int(batch_size) != 1:
            raise ValueError(
                "form='single' with iter_batches=True requires batch_size=1; "
                "otherwise rows would be discarded from multi-row batches."
            )

        catalog_status = self.status
        if (
            catalog_status == "incomplete"
            or self.descriptor.get("status") != catalog_status
        ):
            self.refresh()

        if where is not None and (sample_n is not None or sample_frac is not None):
            resolved_positions = self.project.query.sample_positions_where(
                self,
                where=where,
                metadata_mode=metadata_mode,
                include_data=self._query_can_include_data(data_columns),
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
            )
        else:
            resolved_positions = _resolve_query_positions(
                total=self.n_rows or 0,
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
            )

        query_kwargs = {
            "key_columns": key_columns,
            "data_columns": data_columns,
            "metadata_columns": metadata_columns,
            "metadata_mode": metadata_mode,
            "where": where,
            "order_by": order_by,
            "positions": resolved_positions,
            "limit": limit,
            "form": form,
            "include_position": include_position,
        }

        if iter_batches:
            return self._query_iter_batches(
                batch_size=batch_size, streaming_mode=streaming_mode, **query_kwargs
            )

        return self._query_materialized(**query_kwargs)

    def _query_materialized(
        self,
        *,
        key_columns: ColumnSelect,
        data_columns: ColumnSelect,
        metadata_columns: ColumnSelect,
        metadata_mode: MetadataMode,
        where: str | None,
        order_by: str | Sequence[str] | None,
        positions: Sequence[int] | None,
        limit: int | None,
        form: QueryForm,
        include_position: bool,
    ) -> Any:
        if (
            self.data_artifact is None
            and not BaseArtifact._has_merged_relational_data(self)
            and data_columns is not True
            and _columns_requested(data_columns)
        ):
            raise MissingDataComponentError(
                f"Artifact {self.artifact_id} has no available data component; "
                f"cannot select explicit data columns {data_columns!r}."
            )
        include_data_in_query = self._query_can_include_data(data_columns)
        needs_external_data = (
            _columns_requested(data_columns)
            and self.data_artifact is not None
            and not include_data_in_query
        )
        internal_position = include_position or needs_external_data

        info = self.project.query.artifact_query(
            self,
            key_columns=key_columns,
            data_columns=data_columns if include_data_in_query else False,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            where=where,
            order_by=order_by,
            limit=limit,
            positions=positions,
            include_position=internal_position,
        )

        if not needs_external_data:
            return self._format_query_frame(
                info,
                form=form,
                include_position=include_position,
            )

        current_positions = info["_position"].astype(int).tolist()

        if form == "native":
            native_data = self._data_native_for_positions(
                current_positions,
                data_columns=data_columns,
            )
            info_out = info if include_position else info.drop(columns=["_position"])
            return self._native_from_info_and_data(info_out, native_data)

        if form == "table":
            data_frame = self._data_frame_for_positions(
                current_positions,
                data_columns=data_columns,
            )
            return _merge_info_and_data_frames(
                info,
                data_frame,
                include_position=include_position,
            )

        resolved_data_columns = self._resolve_external_data_columns(data_columns)
        data_records = self._data_records_for_positions(
            current_positions,
            data_columns=resolved_data_columns,
        )
        records = _merge_info_and_data_records(
            info,
            data_records,
            data_columns=resolved_data_columns,
            include_position=include_position,
        )
        if form == "records":
            return records
        if form == "single":
            return _extract_single(records)

    def _query_iter_batches(
        self,
        *,
        key_columns: ColumnSelect,
        data_columns: ColumnSelect,
        metadata_columns: ColumnSelect,
        metadata_mode: MetadataMode,
        where: str | None,
        order_by: str | Sequence[str] | None,
        positions: Sequence[int] | None,
        limit: int | None,
        form: QueryForm,
        include_position: bool,
        batch_size: int,
        streaming_mode: StreamingMode,
    ) -> Iterable[Any]:
        if (
            self.data_artifact is None
            and not BaseArtifact._has_merged_relational_data(self)
            and data_columns is not True
            and _columns_requested(data_columns)
        ):
            raise MissingDataComponentError(
                f"Artifact {self.artifact_id} has no available data component; "
                f"cannot select explicit data columns {data_columns!r}."
            )
        include_data_in_query = self._query_can_include_data(data_columns)
        needs_external_data = (
            _columns_requested(data_columns)
            and self.data_artifact is not None
            and not include_data_in_query
        )
        internal_position = include_position or needs_external_data

        # External representation data (for example sparse/dense matrices) are
        # loaded after each query batch. Those loads may execute additional
        # DuckDB queries on the project's shared connection (for position ->
        # physical batch/row-offset resolution). Keeping an Arrow reader open
        # across those nested queries invalidates the reader after its first
        # batch. Use the existing bounded paged path internally whenever query
        # batches must be interleaved with external-data reads. Table/JSONL data
        # remain eligible for Arrow streaming because they are included directly
        # in the DuckDB relation and require no nested query.
        effective_streaming_mode: StreamingMode = (
            "paged" if needs_external_data else streaming_mode
        )

        for info in self.project.query.artifact_query_batches(
            self,
            key_columns=key_columns,
            data_columns=data_columns if include_data_in_query else False,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            where=where,
            order_by=order_by,
            limit=limit,
            positions=positions,
            include_position=internal_position,
            batch_size=batch_size,
            streaming_mode=effective_streaming_mode,
        ):
            if not needs_external_data:
                yield self._format_query_frame(
                    info,
                    form=form,
                    include_position=include_position,
                )
                continue

            current_positions = info["_position"].astype(int).tolist()

            if form == "native":
                native_data = self._data_native_for_positions(
                    current_positions,
                    data_columns=data_columns,
                )
                info_out = (
                    info if include_position else info.drop(columns=["_position"])
                )
                yield self._native_from_info_and_data(info_out, native_data)
                continue

            if form == "table":
                data_frame = self._data_frame_for_positions(
                    current_positions,
                    data_columns=data_columns,
                )
                yield _merge_info_and_data_frames(
                    info,
                    data_frame,
                    include_position=include_position,
                )
                continue

            resolved_data_columns = self._resolve_external_data_columns(data_columns)
            data_records = self._data_records_for_positions(
                current_positions,
                data_columns=resolved_data_columns,
            )
            records = _merge_info_and_data_records(
                info,
                data_records,
                data_columns=resolved_data_columns,
                include_position=include_position,
            )
            if form == "records":
                yield records
            elif form == "single":
                yield _extract_single(records)

    def _format_query_frame(
        self,
        frame: pd.DataFrame,
        *,
        form: QueryForm,
        include_position: bool,
    ) -> Any:
        out = frame
        if not include_position and "_position" in out.columns:
            out = out.drop(columns=["_position"])

        if form in ("table", "native"):
            return out

        records = _records_with_str_keys(out)
        if form == "records":
            return records
        if form == "single":
            return _extract_single(records)

    # ------------------------------------------------------------------
    # Curated public wrappers around query
    # ------------------------------------------------------------------

    def table(
        self,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        limit: int | None = None,
        include_position: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame table for common exploratory use."""
        return cast(
            pd.DataFrame,
            self.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                limit=limit,
                form="table",
                iter_batches=False,
                include_position=include_position,
            ),
        )

    def records(
        self,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        limit: int | None = None,
        include_position: bool = False,
    ) -> list[dict[str, Any]]:
        """Return flat record dictionaries for common Python use."""
        return cast(
            list[dict[str, Any]],
            self.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                limit=limit,
                form="records",
                iter_batches=False,
                include_position=include_position,
            ),
        )

    def preview(
        self,
        n: int = 20,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
    ) -> pd.DataFrame:
        """Return the first ``n`` rows as a DataFrame."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        return self.table(
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            limit=int(n),
        )

    def sample(
        self,
        n: int | None = None,
        *,
        frac: float | None = None,
        random_state: int | None = None,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        form: Literal["table", "records", "native"] = "table",
    ) -> Any:
        """Return a random sample in a curated output form."""
        if n is not None and frac is not None:
            raise ValueError("Cannot specify both n and frac.")
        sample_n = 5 if n is None and frac is None else n
        return self.query(
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            sample_n=None if sample_n is None else int(sample_n),
            sample_frac=frac,
            random_state=random_state,
            form=form,
            iter_batches=False,
        )

    def iter_batches(
        self,
        *,
        batch_size: int = 10_000,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        form: Literal["table", "records", "native"] = "table",
        include_position: bool = False,
        streaming_mode: StreamingMode = "auto",
    ) -> Iterable[Any]:
        """Iterate through query results in batches."""
        return cast(
            Iterable[Any],
            self.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
                limit=limit,
                form=form,
                iter_batches=True,
                batch_size=batch_size,
                include_position=include_position,
                streaming_mode=streaming_mode,
            ),
        )

    def iter_table_batches(
        self,
        *,
        batch_size: int = 10_000,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        include_position: bool = False,
        streaming_mode: StreamingMode = "auto",
    ) -> Iterable[pd.DataFrame]:
        """Iterate through DataFrame batches."""
        return cast(
            Iterable[pd.DataFrame],
            self.iter_batches(
                batch_size=batch_size,
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
                limit=limit,
                form="table",
                include_position=include_position,
                streaming_mode=streaming_mode,
            ),
        )

    def iter_records(
        self,
        *,
        batch_size: int = 10_000,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        include_position: bool = False,
        streaming_mode: StreamingMode = "auto",
    ) -> Iterable[list[dict[str, Any]]]:
        """Iterate through batches of flat record dictionaries."""
        return cast(
            Iterable[list[dict[str, Any]]],
            self.iter_batches(
                batch_size=batch_size,
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
                limit=limit,
                form="records",
                include_position=include_position,
                streaming_mode=streaming_mode,
            ),
        )

    def iter_rows(
        self,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        include_position: bool = False,
        streaming_mode: StreamingMode = "auto",
    ) -> Iterable[dict[str, Any]]:
        """Iterate row-by-row as flat record dictionaries."""
        return cast(
            Iterable[dict[str, Any]],
            self.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by=order_by,
                positions=positions,
                sample_n=sample_n,
                sample_frac=sample_frac,
                random_state=random_state,
                limit=limit,
                form="single",
                iter_batches=True,
                batch_size=1,
                include_position=include_position,
                streaming_mode=streaming_mode,
            ),
        )

    # ------------------------------------------------------------------
    # Single-row access
    # ------------------------------------------------------------------

    def get_by_position(
        self,
        position: int,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        include_position: bool = False,
    ) -> dict[str, Any]:
        """Return one flat record by artifact-local position."""
        return cast(
            dict[str, Any],
            self.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                positions=[int(position)],
                form="single",
                iter_batches=False,
                include_position=include_position,
            ),
        )

    def get(
        self,
        key: Any,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        include_position: bool = False,
    ) -> dict[str, Any]:
        """Return one flat record by primary key."""
        return self.get_by_position(
            self.position_by_key(key),
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            include_position=include_position,
        )

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def get_metadata_by_position(
        self,
        position: int,
        *,
        metadata_mode: MetadataMode = "local",
    ) -> dict[str, Any]:
        """Return all metadata for one row by artifact-local position."""
        if metadata_mode == "none":
            raise ValueError(
                "metadata_mode='none' is not valid for get_metadata_by_position()."
            )
        return cast(
            dict[str, Any],
            self.query(
                key_columns=False,
                data_columns=False,
                metadata_columns=True,
                metadata_mode=metadata_mode,
                positions=[int(position)],
                form="single",
                iter_batches=False,
                include_position=False,
            ),
        )

    def get_metadata(
        self,
        key: Any,
        *,
        metadata_mode: MetadataMode = "local",
    ) -> dict[str, Any]:
        """Return all metadata for one row by primary key."""
        return self.get_metadata_by_position(
            self.position_by_key(key),
            metadata_mode=metadata_mode,
        )

    def get_metadata_columns(self) -> list[str]:
        """Return local metadata columns visible through query()."""
        return list(
            self.query_columns(
                metadata_mode="local",
            )["metadata"]
        )

    def get_full_metadata_columns(self) -> list[str]:
        """Return full lineage metadata columns visible through query()."""
        return list(
            self.query_columns(
                metadata_mode="full",
            )["metadata"]
        )

    def get_data_columns(self) -> list[str]:
        """Return local representation data columns exposed by this artifact.

        Concrete artifact subclasses should implement this according to their
        storage format. Table artifacts can delegate to QueryEngine, matrix
        artifacts can read their columns component, and JSONL artifacts can read
        the writer-maintained data component column list.
        """
        raise UnsupportedArtifactOperationError(
            f"{self.__class__.__name__} does not expose named data columns."
        )

    # ------------------------------------------------------------------
    # Context windows
    # ------------------------------------------------------------------

    def get_context(
        self,
        key: Any,
        before: int = 2,
        after: int = 2,
        *,
        context_artifact: "BaseArtifact | str | None" = None,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        include_focus: bool = True,
    ) -> pd.DataFrame:
        if before < 0 or after < 0:
            raise ValueError("before and after must be non-negative.")

        context = resolve_artifact_ref(
            self.project,
            context_artifact,
            default=self,
            parameter_name="context_artifact",
        )
        normalized = self._normalize_key(key)

        key_filter = _context_key_filter(normalized, list(context.primary_key))
        if not key_filter:
            raise ArtifactError(
                f"Cannot resolve key from artifact {self.artifact_id} into context "
                f"artifact {context.artifact_id}: no shared key constraints."
            )

        key_column_mapping = _resolved_key_column_mapping(
            context,
            key_filter,
            metadata_mode=metadata_mode,
        )
        where = _key_filter_where_clause(
            key_filter,
            key_column_mapping=key_column_mapping,
        )

        if before == 0 and after == 0 and include_focus:
            return context.query(
                key_columns=key_columns,
                data_columns=data_columns,
                metadata_columns=metadata_columns,
                metadata_mode=metadata_mode,
                where=where,
                order_by="_position",
                form="table",
            )

        matched_positions = self.project.query.positions_where_keys(
            context,
            where=where,
        )
        if not matched_positions:
            return pd.DataFrame()

        positions = _expand_context_positions(
            matched_positions,
            before=int(before),
            after=int(after),
            total=context.n_rows or 0,
            include_focus=include_focus,
        )

        if not positions:
            return pd.DataFrame()

        return context.query(
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            positions=positions,
            order_by="_position",
            form="table",
        )

    def get_previous(self, key: Any, n: int = 1, **kwargs: Any) -> pd.DataFrame:
        if n < 0:
            raise ValueError("n must be non-negative.")
        return self.get_context(
            key,
            before=n,
            after=0,
            include_focus=False,
            **kwargs,
        )

    def get_next(self, key: Any, n: int = 1, **kwargs: Any) -> pd.DataFrame:
        if n < 0:
            raise ValueError("n must be non-negative.")
        return self.get_context(
            key,
            before=0,
            after=n,
            include_focus=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Text search
    # ------------------------------------------------------------------

    def kwic(
        self,
        pattern: str,
        *,
        window: int = 5,
        before: int | None = None,
        after: int | None = None,
        valuetype: Literal["fixed", "regex"] = "fixed",
        case_sensitive: bool = False,
        enforce_word_boundary: bool = True,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        batch_size: int = 100_000,
        streaming_mode: StreamingMode = "auto",
        search_columns: ColumnSelect | None = None,
        target_matches: int | None = None,
    ) -> KWICResult:
        """Return keyword-in-context concordance hits from query result columns."""
        return keyword_in_context(
            self,
            pattern,
            window=window,
            before=before,
            after=after,
            valuetype=valuetype,
            case_sensitive=case_sensitive,
            enforce_word_boundary=enforce_word_boundary,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            where=where,
            order_by=order_by,
            positions=positions,
            sample_n=sample_n,
            sample_frac=sample_frac,
            random_state=random_state,
            limit=limit,
            batch_size=batch_size,
            streaming_mode=streaming_mode,
            search_columns=search_columns,
            target_matches=target_matches,
        )
