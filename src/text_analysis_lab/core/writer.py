"""Artifact writing for TextAnalysisLab (TeAL).

The artifact writer is a strict, batch-only serialization boundary. It does
not own execution semantics, row buffering, checkpointing, resume behavior, or
operator-family coercion. Project/execution code is responsible for planning
units of work and handing this writer writer-ready batch payloads. Table
artifacts may use a schema-bearing zero-row payload to represent a legitimate
empty result.

A single writer owns the output order for one artifact. It assigns structural
columns (``_position``, ``_batch``, and ``_row_offset``) sequentially according
to the order in which ``write(...)`` is called.
"""

from __future__ import annotations

import importlib
import json
import numbers
import os
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast, get_args

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import (
    ArtifactError,
    DuplicatePrimaryKeyError,
    UnsupportedArtifactTypeError,
)
from text_analysis_lab.core.storage import ArtifactStorage
from text_analysis_lab.core.types import (
    ArtifactStatus,
    ArtifactType,
    LineageMode,
    StructuralColumn,
)

STRUCTURAL_COLUMNS: frozenset[str] = frozenset(get_args(StructuralColumn))
TABLE_TYPES = {ArtifactType.TABLE, ArtifactType.JSONL}

OtherDataSerializer = Callable[[Any, Path, int], Mapping[str, Any] | None]
CallableRef = dict[str, str]


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically publish a human-facing JSON descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _resolve_callable_ref(ref: CallableRef) -> OtherDataSerializer:
    module_name = ref["module"]
    qualname = ref["qualname"]

    try:
        module = importlib.import_module(module_name)
        resolved: Any = module
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except Exception as exc:
        raise ArtifactError(
            f"Could not resolve data_serializer_ref {module_name}:{qualname}."
        ) from exc

    if not callable(resolved):
        raise ArtifactError(
            f"Resolved data_serializer_ref {module_name}:{qualname} is not callable."
        )
    return cast(OtherDataSerializer, resolved)


def _optional_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)


def _coerce_integer_key_column(series: pd.Series, col: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = (
        numeric.isna() | ~np.isfinite(numeric) | (numeric != np.floor(numeric))
    )
    if invalid_mask.any():
        first_invalid = series.loc[invalid_mask].iloc[0]
        raise ArtifactError(
            f"Primary-key column {col!r} must contain values coercible to non-null integers; "
            f"first invalid value: {first_invalid!r}."
        )
    return numeric.astype("int64")


def _require_dataframe(
    value: Any,
    *,
    n_rows: int | None = None,
    channel: str,
) -> pd.DataFrame:
    """Require a pandas DataFrame payload for tabular channels."""
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(f"Payload channel {channel!r} must be a pandas DataFrame.")
    frame = value.copy().reset_index(drop=True)
    if n_rows is not None and len(frame) != int(n_rows):
        raise ArtifactError(
            f"Payload channel {channel!r} row count {len(frame)} != expected {n_rows}."
        )
    reserved = [str(col) for col in frame.columns if str(col) in STRUCTURAL_COLUMNS]
    if reserved:
        raise ArtifactError(
            f"{channel} payload contains reserved structural column(s): {reserved}. "
            f"Reserved columns are managed by ArtifactWriter: {sorted(STRUCTURAL_COLUMNS)}."
        )
    return frame


def _as_json_records(
    value: Any, *, n_rows: int, channel: str
) -> list[Mapping[str, Any]]:
    """Require batch-shaped JSONL data: a sequence of row mappings."""
    if not isinstance(value, (list, tuple)):
        raise ArtifactError(
            f"Payload channel {channel!r} for JSONL data must be a list/tuple of row mappings."
        )
    records = list(value)
    if len(records) != int(n_rows):
        raise ArtifactError(
            f"Payload channel {channel!r} row count {len(records)} != expected {n_rows}."
        )
    bad = [i for i, record in enumerate(records) if not isinstance(record, Mapping)]
    if bad:
        raise ArtifactError(
            f"Payload channel {channel!r} JSONL records must be mappings; first bad row: {bad[0]}."
        )
    return records


def _require_matrix_payload(value: Any, *, channel: str = "data") -> Mapping[str, Any]:
    """Require the uniform matrix payload shape: {'values': ..., 'columns': sequence}."""
    if not isinstance(value, Mapping):
        raise ArtifactError(
            f"Payload channel {channel!r} for matrix data must be a mapping."
        )
    missing = [name for name in ("values", "columns") if name not in value]
    if missing:
        raise ArtifactError(
            f"Payload channel {channel!r} for matrix data must include keys 'values' and 'columns'; "
            f"missing {missing}."
        )
    return value


def _schema(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(str(col) for col in frame.columns)


def _json_value_type(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, numbers.Integral):
        return "BIGINT"
    if isinstance(value, numbers.Real):
        return "DOUBLE"
    if isinstance(value, str):
        return "VARCHAR"
    return "JSON"


def _promote_json_value_type(existing: str, new: str) -> str:
    if existing == new:
        return existing
    if existing == "NULL":
        return new
    if new == "NULL":
        return existing

    numeric = {"BIGINT", "DOUBLE"}
    if existing in numeric and new in numeric:
        return "DOUBLE"

    if existing == "JSON" or new == "JSON":
        return "JSON"

    return "VARCHAR"


def _coerce_matrix_columns(columns: Any) -> tuple[str, ...]:
    """Coerce matrix column labels to the writer's internal tuple form."""
    if isinstance(columns, np.ndarray):
        if columns.ndim != 1:
            raise ArtifactError("Matrix columns must be a one-dimensional numpy array.")
        values = columns.tolist()
    elif isinstance(columns, (list, tuple)):
        values = columns
    else:
        raise ArtifactError(
            "Matrix columns must be a list, tuple, or one-dimensional numpy array."
        )
    return tuple(str(value) for value in values)


def _coerce_matrix_row_names(row_names: Any, *, n_rows: int) -> tuple[str, ...]:
    """Coerce one batch of optional matrix row names to strings."""
    if isinstance(row_names, (pd.Series, pd.Index)):
        values = row_names.tolist()
    elif isinstance(row_names, np.ndarray):
        if row_names.ndim != 1:
            raise ArtifactError("Matrix row_names must be one-dimensional.")
        values = row_names.tolist()
    elif isinstance(row_names, (list, tuple)):
        values = list(row_names)
    else:
        raise ArtifactError(
            "Matrix row_names must be a list, tuple, pandas Series/Index, or "
            "one-dimensional numpy array."
        )
    if len(values) != int(n_rows):
        raise ArtifactError(
            f"Matrix row_names count {len(values)} != matrix row count {n_rows}."
        )
    for value in values:
        if value is None:
            raise ArtifactError("Matrix row_names may not contain null values.")
        null = pd.isna(value)
        if not isinstance(null, (bool, np.bool_)):
            raise ArtifactError("Matrix row_names must contain scalar values.")
        if bool(null):
            raise ArtifactError("Matrix row_names may not contain null values.")
    labels = tuple(str(value) for value in values)
    if len(set(labels)) != len(labels):
        raise ArtifactError(
            "Matrix row_names must be unique within each batch after string conversion."
        )
    return labels


class ArtifactWriter:
    """Write one TeAL artifact from strict batch payloads.

    The writer receives only batch payloads using the uniform channels:
    ``keys``, optional ``metadata``, and optional ``data``. Row-wise buffering,
    execution planning, checkpointing, and parallel scheduling belong to
    ``Project.run`` or related execution code. A zero-row table payload is
    permitted when it carries explicit key/data/metadata column schemas.

    Structural columns are writer-owned. Callers must never supply ``_position``,
    ``_batch``, or ``_row_offset``.
    """

    def __init__(
        self,
        *,
        artifact_dir: str | Path,
        artifact_id: str,
        artifact_type: ArtifactType | str,
        label: str,
        lineage: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        data_serializer: OtherDataSerializer | None = None,
        data_serializer_ref: CallableRef | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_id = str(artifact_id)
        self.artifact_type = ArtifactType(artifact_type)
        self.label = str(label)
        if not self.label:
            raise ArtifactError("Artifact label must be a non-empty string.")
        self.lineage = dict(lineage or {})
        self.operation_id = None if operation_id is None else str(operation_id)

        if self.artifact_type == ArtifactType.OTHER:
            self.data_serializer_ref = data_serializer_ref
            if self.data_serializer_ref is None:
                raise ArtifactError("ArtifactType.OTHER requires data_serializer_ref.")
            if data_serializer is not None and not callable(data_serializer):
                raise ArtifactError("data_serializer must be callable or None.")
            self.data_serializer = (
                data_serializer
                if data_serializer is not None
                else _resolve_callable_ref(self.data_serializer_ref)
            )
        else:
            self.data_serializer_ref = None
            self.data_serializer = None

        self.storage = ArtifactStorage.open(self.artifact_dir)
        self.storage.ensure_artifact_dir()
        self.keys_dir = self.storage.keys_dir
        self.data_dir = self.storage.data_dir
        self.metadata_dir = self.storage.metadata_dir

        self._status: ArtifactStatus = "incomplete"
        self._closed = False
        self._failed = False
        self._next_part_index = 0
        self._next_position = 0
        self._n_rows = 0
        self._primary_key: tuple[str, ...] = ()
        self._components: dict[str, dict[str, Any]] = {}
        self._initialized_components: set[str] = set()

        self._data_schema: tuple[str, ...] | None = None
        self._metadata_schema: tuple[str, ...] | None = None
        self._matrix_n_columns: int | None = None
        self._matrix_columns: tuple[str, ...] | None = None
        self._matrix_has_row_names: bool | None = None
        self._matrix_row_name: str | None = None

        self._write_descriptor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, payload: Mapping[str, Any]) -> bool:
        """Serialize one batch-shaped payload.

        Table artifacts may receive a schema-bearing zero-row payload. This is
        useful for legitimate empty outputs such as ``failures`` or
        ``unresolved`` tables: the payload still establishes key/data schemas so
        the artifact remains queryable and can be sealed normally.

        Returns ``True`` after durable serialization. Write failures raise.
        A ``False`` return is intentionally not used; buffering belongs to
        Project/execution code, not to the writer.
        """
        self._ensure_open()
        if not isinstance(payload, Mapping):
            raise ArtifactError("ArtifactWriter.write() requires a mapping payload.")
        if "keys" not in payload:
            raise ArtifactError("Artifact payloads must include 'keys'.")

        self._write_payload(payload)
        return True

    def finalize(self) -> Path:
        """Validate, seal, and mark the artifact complete.

        Batch writes perform local structural validation as they are serialized.
        Finalization is the artifact-wide seal: checks that require seeing the
        complete key universe, including cross-batch primary-key uniqueness,
        happen here.  A seal failure preserves the written artifact but marks its
        descriptor ``failed`` instead of blessing it as ``complete``.
        """
        self._ensure_open()
        if "keys" not in self._components:
            raise ArtifactError("Cannot finalize artifact without keys.")
        try:
            self._validate_final_keys()
            self._validate_final_matrix_row_names()
        except Exception as exc:
            self.mark_failed(exc)
            raise
        self._status = "complete"
        self._closed = True
        self._write_descriptor()
        return self.artifact_dir

    def mark_failed(self, error: BaseException | str | None = None) -> None:
        """Mark the artifact failed without attempting hidden buffering or flushing."""
        self._status = "failed"
        self._failed = True
        self._closed = True
        if error is not None:
            self._components.setdefault("error", {})["message"] = str(error)
        self._write_descriptor()

    def resume_state(self) -> dict[str, Any]:
        """Return JSON-safe writer state needed to resume this writer.

        This is a writer-owned checkpoint. Execution code should not infer these
        internals from the artifact descriptor; it should store this state and
        pass it back to ``ArtifactWriter.resume(...)``.
        """
        state: dict[str, Any] = {
            "schema_version": 1,
            "artifact_dir": str(self.artifact_dir),
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "label": self.label,
            "lineage": dict(self.lineage),
            "operation_id": self.operation_id,
            "data_serializer_ref": self.data_serializer_ref,
            "status": self._status,
            "closed": self._closed,
            "failed": self._failed,
            "next_part_index": self._next_part_index,
            "next_position": self._next_position,
            "n_rows": self._n_rows,
            "primary_key": list(self._primary_key),
            "components": dict(self._components),
            "initialized_components": sorted(self._initialized_components),
            "data_schema": (
                None if self._data_schema is None else list(self._data_schema)
            ),
            "metadata_schema": (
                None if self._metadata_schema is None else list(self._metadata_schema)
            ),
            "matrix_n_columns": self._matrix_n_columns,
            "matrix_columns": (
                None if self._matrix_columns is None else list(self._matrix_columns)
            ),
            "matrix_has_row_names": self._matrix_has_row_names,
            "matrix_row_name": self._matrix_row_name,
        }
        try:
            json.dumps(state)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                "ArtifactWriter resume state must be JSON-serializable."
            ) from exc
        return state

    @classmethod
    def resume(
        cls,
        state: Mapping[str, Any] | str | Path,
        *,
        artifact_dir: str | Path | None = None,
    ) -> ArtifactWriter:
        """Rehydrate a writer from ``resume_state()`` output.

        ``artifact_dir`` may override the serialized path when a project has been
        moved and the caller wants to resolve the artifact location from project
        storage.
        """
        if isinstance(state, (str, Path)):
            state = json.loads(Path(state).read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ArtifactError("ArtifactWriter.resume() requires a state mapping.")
        if int(state.get("schema_version", 0)) != 1:
            raise ArtifactError(
                "Unsupported ArtifactWriter resume state schema_version."
            )

        writer = cls.__new__(cls)
        writer.artifact_dir = Path(
            artifact_dir if artifact_dir is not None else state["artifact_dir"]
        )
        writer.artifact_id = str(state["artifact_id"])
        writer.artifact_type = ArtifactType(state["artifact_type"])
        writer.label = str(state["label"])
        if not writer.label:
            raise ArtifactError("Artifact label must be a non-empty string.")
        writer.lineage = dict(state.get("lineage") or {})
        raw_operation_id = state.get("operation_id")
        writer.operation_id = (
            None if raw_operation_id is None else str(raw_operation_id)
        )

        if writer.artifact_type == ArtifactType.OTHER:
            writer.data_serializer_ref = cast(
                CallableRef | None,
                state.get("data_serializer_ref"),
            )
            if writer.data_serializer_ref is None:
                raise ArtifactError(
                    "ArtifactType.OTHER requires data_serializer_ref on resume."
                )
            writer.data_serializer = _resolve_callable_ref(writer.data_serializer_ref)
        else:
            writer.data_serializer_ref = None
            writer.data_serializer = None

        writer.storage = ArtifactStorage.open(writer.artifact_dir)
        writer.storage.ensure_artifact_dir()
        writer.keys_dir = writer.storage.keys_dir
        writer.data_dir = writer.storage.data_dir
        writer.metadata_dir = writer.storage.metadata_dir

        writer._status = state["status"]
        writer._closed = bool(state["closed"])
        writer._failed = bool(state["failed"])
        writer._next_part_index = int(state["next_part_index"])
        writer._next_position = int(state["next_position"])
        writer._n_rows = int(state["n_rows"])
        writer._primary_key = tuple(str(col) for col in state.get("primary_key", []))
        writer._components = {
            str(name): dict(component)
            for name, component in dict(state.get("components", {})).items()
        }
        writer._initialized_components = {
            str(name) for name in state.get("initialized_components", [])
        }
        writer._data_schema = _optional_tuple(state.get("data_schema"))
        writer._metadata_schema = _optional_tuple(state.get("metadata_schema"))
        raw_matrix_n_columns = state.get("matrix_n_columns")
        writer._matrix_n_columns = (
            None if raw_matrix_n_columns is None else int(raw_matrix_n_columns)
        )
        writer._matrix_columns = _optional_tuple(state.get("matrix_columns"))
        raw_has_row_names = state.get("matrix_has_row_names")
        writer._matrix_has_row_names = (
            None if raw_has_row_names is None else bool(raw_has_row_names)
        )
        raw_row_name = state.get("matrix_row_name")
        writer._matrix_row_name = None if raw_row_name is None else str(raw_row_name)
        return writer

    # ------------------------------------------------------------------
    # Batch payload writing
    # ------------------------------------------------------------------

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        part_index = self._next_part_index
        start_position = self._next_position

        n_rows = self._write_keys(
            payload["keys"], start_position=start_position, part_index=part_index
        )
        if n_rows == 0 and self.artifact_type != ArtifactType.TABLE:
            raise ArtifactError(
                "Zero-row schema payloads are currently supported only for table artifacts."
            )

        if "metadata" in payload and payload.get("metadata") is not None:
            self._write_metadata(
                payload["metadata"],
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )

        if "feature_indices" in payload and payload.get("feature_indices") is not None:
            self._record_feature_projection(payload["feature_indices"])

        if "data" in payload and payload.get("data") is not None:
            self._write_data(
                payload["data"],
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )

        self._next_part_index += 1
        self._next_position += n_rows
        self._n_rows += n_rows
        self._write_descriptor()

    # ------------------------------------------------------------------
    # Channel writers
    # ------------------------------------------------------------------

    def _record_feature_projection(self, raw_indices: Any) -> None:
        """Record one positional matrix-feature projection on this artifact.

        The projection is intentionally positional. TeAL does not infer feature
        identity from column labels or other feature metadata. A producer that
        replays a fitted feature-space operation is responsible for supplying a
        matrix with the same ordered feature axis used when these indices were
        learned.
        """
        if self.artifact_type not in {
            ArtifactType.SPARSE_MATRIX,
            ArtifactType.DENSE_MATRIX,
        }:
            raise ArtifactError(
                "feature_indices are supported only for sparse_matrix or "
                "dense_matrix artifacts."
            )
        if isinstance(raw_indices, np.ndarray):
            if raw_indices.ndim != 1:
                raise ArtifactError("feature_indices must be one-dimensional.")
            values = raw_indices.tolist()
        elif isinstance(raw_indices, (list, tuple)):
            values = list(raw_indices)
        else:
            raise ArtifactError(
                "feature_indices must be a list, tuple, or one-dimensional numpy array."
            )

        indices: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ArtifactError("feature_indices must contain only integers.")
            index = int(value)
            if index < 0:
                raise ArtifactError("feature_indices may not contain negative values.")
            indices.append(index)

        if not indices:
            raise ArtifactError("feature_indices must retain at least one feature.")
        if any(right <= left for left, right in pairwise(indices)):
            raise ArtifactError(
                "feature_indices must be unique and strictly increasing so the "
                "derived feature axis preserves source order."
            )

        existing = self.lineage.get("feature_indices")
        if existing is None:
            self.lineage["feature_indices"] = indices
            return
        if [int(value) for value in existing] != indices:
            raise ArtifactError(
                "feature_indices changed after the feature view was established."
            )

    def _validate_final_keys(self) -> None:
        """Run artifact-wide key checks before a writer may become complete.

        Key parts are scanned one at a time.  A temporary disk-backed SQLite
        table provides the cross-batch uniqueness check without retaining every
        primary key in Python memory.
        """
        if not self._primary_key:
            raise ArtifactError("Cannot finalize artifact without a primary key.")

        part_paths = [
            self.storage.key_part_path(part_index)
            for part_index in range(self._next_part_index)
        ]
        missing = [path for path in part_paths if not path.exists()]
        if missing:
            raise ArtifactError(f"Missing key part(s) during finalization: {missing}.")

        with tempfile.TemporaryDirectory(prefix="teal-key-validation-") as temp_dir:
            validation_db = Path(temp_dir) / "keys.sqlite"
            con = sqlite3.connect(validation_db)
            try:
                key_columns = [f"k{index}" for index in range(len(self._primary_key))]
                column_sql = ", ".join(
                    f'"{column}" INTEGER NOT NULL' for column in key_columns
                )
                pk_sql = ", ".join(f'"{column}"' for column in key_columns)
                con.execute(
                    f"CREATE TABLE seen_keys ({column_sql}, PRIMARY KEY ({pk_sql})) WITHOUT ROWID"
                )

                expected_position = 0
                total_rows = 0
                placeholders = ", ".join("?" for _ in key_columns)
                insert_sql = f"INSERT INTO seen_keys VALUES ({placeholders})"

                for part_index, path in enumerate(part_paths):
                    columns = [
                        *self._primary_key,
                        "_position",
                        "_batch",
                        "_row_offset",
                    ]
                    frame = pd.read_parquet(path, columns=columns)
                    n_rows = len(frame)

                    expected_positions = np.arange(
                        expected_position,
                        expected_position + n_rows,
                        dtype="int64",
                    )
                    if not np.array_equal(
                        frame["_position"].to_numpy(dtype="int64"),
                        expected_positions,
                    ):
                        raise ArtifactError(
                            f"Key positions are not contiguous in part {part_index}."
                        )
                    if not np.all(
                        frame["_batch"].to_numpy(dtype="int64") == part_index
                    ):
                        raise ArtifactError(
                            f"Key batch markers are invalid in part {part_index}."
                        )
                    if not np.array_equal(
                        frame["_row_offset"].to_numpy(dtype="int64"),
                        np.arange(n_rows, dtype="int64"),
                    ):
                        raise ArtifactError(
                            f"Key row offsets are invalid in part {part_index}."
                        )

                    values = frame.loc[:, list(self._primary_key)].itertuples(
                        index=False,
                        name=None,
                    )
                    try:
                        con.executemany(insert_sql, values)
                    except sqlite3.IntegrityError as exc:
                        raise DuplicatePrimaryKeyError(
                            "Duplicate primary-key values across artifact key batches."
                        ) from exc

                    expected_position += n_rows
                    total_rows += n_rows

                if total_rows != self._n_rows:
                    raise ArtifactError(
                        f"Key row count {total_rows} does not match writer row count "
                        f"{self._n_rows}."
                    )
            finally:
                con.close()

    def _validate_final_matrix_row_names(self) -> None:
        """Seal the optional named row axis and enforce global uniqueness."""
        if self._matrix_has_row_names is not True:
            return
        if self.artifact_type not in {
            ArtifactType.DENSE_MATRIX,
            ArtifactType.SPARSE_MATRIX,
        }:
            raise ArtifactError("Only matrix artifacts may define row_names.")

        part_paths = [
            self.storage.data_row_names_part_path(part_index)
            for part_index in range(self._next_part_index)
        ]
        missing = [path for path in part_paths if not path.exists()]
        if missing:
            raise ArtifactError(
                f"Missing matrix row_names part(s) during finalization: {missing}."
            )

        with tempfile.TemporaryDirectory(
            prefix="teal-row-name-validation-"
        ) as temp_dir:
            con = sqlite3.connect(Path(temp_dir) / "row_names.sqlite")
            try:
                con.execute(
                    "CREATE TABLE seen_row_names "
                    "(row_name TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID"
                )
                expected_position = 0
                total_rows = 0
                for part_index, path in enumerate(part_paths):
                    frame = pd.read_parquet(path, columns=["row_name", "_position"])
                    n_rows = len(frame)
                    expected = np.arange(
                        expected_position, expected_position + n_rows, dtype="int64"
                    )
                    if not np.array_equal(
                        frame["_position"].to_numpy(dtype="int64"), expected
                    ):
                        raise ArtifactError(
                            f"Matrix row-name positions are not contiguous in part {part_index}."
                        )
                    try:
                        con.executemany(
                            "INSERT INTO seen_row_names VALUES (?)",
                            ((str(value),) for value in frame["row_name"].tolist()),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ArtifactError(
                            "Duplicate matrix row_names across artifact batches."
                        ) from exc
                    expected_position += n_rows
                    total_rows += n_rows
                if total_rows != self._n_rows:
                    raise ArtifactError(
                        f"Matrix row_names count {total_rows} does not match writer "
                        f"row count {self._n_rows}."
                    )
            finally:
                con.close()

    def _write_keys(self, keys: Any, *, start_position: int, part_index: int) -> int:
        frame = _require_dataframe(keys, channel="keys")
        if len(frame.columns) == 0:
            raise ArtifactError(
                "Keys payload must contain at least one primary-key column."
            )
        if any(str(col).startswith("_") for col in frame.columns):
            raise ArtifactError("Primary-key columns may not start with '_'.")

        frame.columns = [str(col) for col in frame.columns]
        columns = tuple(frame.columns)
        if len(set(columns)) != len(columns):
            raise ArtifactError(
                "Key column names must be unique after string conversion."
            )

        for col in columns:
            frame[col] = _coerce_integer_key_column(frame[col], col)

        if not self._primary_key:
            self._primary_key = columns
        elif columns != self._primary_key:
            raise ArtifactError(
                f"Key columns changed from {self._primary_key} to {columns}."
            )

        if frame.duplicated(subset=list(self._primary_key)).any():
            raise DuplicatePrimaryKeyError(
                "Duplicate primary-key values within key batch."
            )

        n_rows = len(frame)
        frame["_position"] = np.arange(
            start_position, start_position + n_rows, dtype="int64"
        )
        frame["_batch"] = int(part_index)
        frame["_row_offset"] = np.arange(n_rows, dtype="int64")

        self._ensure_component(
            "keys",
            {"format": "parquet_dataset", "path": "keys/"},
            self.keys_dir,
        )
        frame.to_parquet(self.storage.key_part_path(part_index), index=False)
        return n_rows

    def _write_metadata(
        self,
        metadata: Any,
        *,
        start_position: int,
        n_rows: int,
        part_index: int,
    ) -> None:
        frame = _require_dataframe(metadata, n_rows=n_rows, channel="metadata")
        frame.columns = [str(col) for col in frame.columns]
        schema = _schema(frame)
        if self._metadata_schema is None:
            self._metadata_schema = schema
        elif schema != self._metadata_schema:
            raise ArtifactError(
                f"Metadata schema changed from {self._metadata_schema} to {schema}."
            )

        self._ensure_component(
            "metadata",
            {
                "format": "parquet_dataset",
                "path": "metadata/",
                "schema": list(self._metadata_schema),
            },
            self.metadata_dir,
        )
        frame["_position"] = np.arange(
            start_position, start_position + n_rows, dtype="int64"
        )
        frame.to_parquet(self.storage.metadata_part_path(part_index), index=False)

    def _write_data(
        self, data: Any, *, start_position: int, n_rows: int, part_index: int
    ) -> None:
        if self.artifact_type == ArtifactType.TABLE:
            self._write_table_data(
                data,
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )
        elif self.artifact_type == ArtifactType.JSONL:
            self._write_jsonl_data(
                data,
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )
        elif self.artifact_type == ArtifactType.SPARSE_MATRIX:
            self._write_sparse_matrix_data(
                data,
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )
        elif self.artifact_type == ArtifactType.DENSE_MATRIX:
            self._write_dense_matrix_data(
                data,
                start_position=start_position,
                n_rows=n_rows,
                part_index=part_index,
            )
        elif self.artifact_type == ArtifactType.OTHER:
            self._write_other_data(data, part_index=part_index)
        else:
            raise UnsupportedArtifactTypeError(
                f"Unsupported artifact_type={self.artifact_type!r}."
            )

    def _write_table_data(
        self, data: Any, *, start_position: int, n_rows: int, part_index: int
    ) -> None:
        frame = _require_dataframe(data, n_rows=n_rows, channel="data")
        frame.columns = [str(col) for col in frame.columns]
        schema = _schema(frame)
        if self._data_schema is None:
            self._data_schema = schema
        elif schema != self._data_schema:
            raise ArtifactError(
                f"Data schema changed from {self._data_schema} to {schema}."
            )

        self._ensure_component(
            "data",
            {
                "format": "parquet_dataset",
                "path": "data/",
                "schema": list(self._data_schema),
            },
            self.data_dir,
        )
        frame["_position"] = np.arange(
            start_position, start_position + n_rows, dtype="int64"
        )
        frame.to_parquet(
            self.storage.data_part_path(part_index, "parquet"), index=False
        )

    def _write_jsonl_data(
        self, data: Any, *, start_position: int, n_rows: int, part_index: int
    ) -> None:
        records = _as_json_records(data, n_rows=n_rows, channel="data")
        positions = np.arange(start_position, start_position + n_rows, dtype="int64")
        self._ensure_component(
            "data",
            {
                "format": "jsonl_dataset",
                "path": "data/",
                "columns": [],
                "types": {},
            },
            self.data_dir,
        )

        component = self._components["data"]
        columns = component["columns"]
        types = component["types"]
        seen = set(columns)

        path = self.storage.data_part_path(part_index, "jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for position, record in zip(positions.tolist(), records, strict=True):
                output = dict(record)
                for key, value in output.items():
                    col = str(key)
                    if col in STRUCTURAL_COLUMNS:
                        raise ArtifactError(
                            f"JSONL data record contains reserved structural key {col!r}. "
                            f"Reserved keys are managed by ArtifactWriter: {sorted(STRUCTURAL_COLUMNS)}."
                        )
                    value_type = _json_value_type(value)
                    if col not in seen:
                        seen.add(col)
                        columns.append(col)
                        types[col] = value_type
                    else:
                        types[col] = _promote_json_value_type(
                            str(types.get(col, "NULL")), value_type
                        )
                output["_position"] = position
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")

        component["columns"] = columns
        component["types"] = types

    def _write_sparse_matrix_data(
        self,
        data: Any,
        *,
        start_position: int,
        n_rows: int,
        part_index: int,
    ) -> None:
        matrix_payload = _require_matrix_payload(data)
        try:
            from scipy import sparse
        except ImportError as exc:
            raise ArtifactError("Sparse matrix writing requires scipy.") from exc

        values = matrix_payload["values"]
        if not sparse.issparse(values) or len(values.shape) != 2:
            raise ArtifactError(
                "Sparse matrix data 'values' must be a 2D scipy sparse matrix."
            )
        values = values.tocsr()
        if int(values.shape[0]) != int(n_rows):
            raise ArtifactError(
                f"Sparse matrix row count {values.shape[0]} != expected {n_rows}."
            )
        n_columns = int(values.shape[1])
        labels = self._validate_matrix_columns(
            matrix_payload["columns"], n_columns=n_columns
        )
        row_names = self._validate_matrix_row_names(matrix_payload, n_rows=n_rows)

        values_dir = self.storage.data_values_dir
        self._ensure_component(
            "data",
            self._matrix_component_descriptor(
                format="sparse_matrix",
                values_format="scipy_npz",
            ),
            self.data_dir,
            values_dir,
            *((self.storage.data_row_names_dir,) if row_names is not None else ()),
            on_init=lambda: self._write_matrix_columns(labels),
        )
        if row_names is not None:
            self._write_matrix_row_names(
                row_names, start_position=start_position, part_index=part_index
            )
        sparse.save_npz(self.storage.data_value_part_path(part_index, "npz"), values)

    def _write_dense_matrix_data(
        self,
        data: Any,
        *,
        start_position: int,
        n_rows: int,
        part_index: int,
    ) -> None:
        matrix_payload = _require_matrix_payload(data)
        values = np.asarray(matrix_payload["values"])
        if values.ndim == 1:
            if n_rows == 1:
                values = values.reshape(1, -1)
            else:
                values = values.reshape(n_rows, -1)
        if values.ndim != 2:
            raise ArtifactError("Dense matrix data 'values' must be a 2D array.")
        if int(values.shape[0]) != int(n_rows):
            raise ArtifactError(
                f"Dense matrix row count {values.shape[0]} != expected {n_rows}."
            )
        n_columns = int(values.shape[1])
        labels = self._validate_matrix_columns(
            matrix_payload["columns"], n_columns=n_columns
        )
        row_names = self._validate_matrix_row_names(matrix_payload, n_rows=n_rows)

        values_dir = self.storage.data_values_dir
        self._ensure_component(
            "data",
            self._matrix_component_descriptor(
                format="dense_matrix",
                values_format="npy",
            ),
            self.data_dir,
            values_dir,
            *((self.storage.data_row_names_dir,) if row_names is not None else ()),
            on_init=lambda: self._write_matrix_columns(labels),
        )
        if row_names is not None:
            self._write_matrix_row_names(
                row_names, start_position=start_position, part_index=part_index
            )
        np.save(self.storage.data_value_part_path(part_index, "npy"), values)

    def _write_other_data(self, data: Any, *, part_index: int) -> None:
        if self.data_serializer is None:
            raise ArtifactError("OTHER artifact writer has no data_serializer.")

        self._ensure_component(
            "data",
            {"format": "custom", "path": "data/"},
            self.data_dir,
        )

        result = self.data_serializer(data, self.data_dir, part_index)
        if result is not None:
            if not isinstance(result, Mapping):
                raise ArtifactError("data_serializer must return a mapping or None.")
            self._components["data"].update(dict(result))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_component(
        self,
        name: str,
        descriptor: Mapping[str, Any],
        *directories: Path,
        on_init: Callable[[], None] | None = None,
    ) -> None:
        if name in self._initialized_components:
            return

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        if on_init is not None:
            on_init()

        self._components[name] = dict(descriptor)
        self._initialized_components.add(name)

    def _validate_matrix_columns(
        self, columns: Any, *, n_columns: int
    ) -> tuple[str, ...]:
        labels = _coerce_matrix_columns(columns)

        if self._matrix_n_columns is None:
            if len(labels) != int(n_columns):
                raise ArtifactError(
                    f"Matrix column count {len(labels)} != data column count {n_columns}."
                )
            if len(set(labels)) != len(labels):
                raise ArtifactError(
                    "Matrix columns must be unique after string conversion."
                )
            reserved = [label for label in labels if label in STRUCTURAL_COLUMNS]
            if reserved:
                raise ArtifactError(
                    f"Matrix columns contain reserved structural name(s): {reserved}. "
                    f"Reserved names are managed by ArtifactWriter: {sorted(STRUCTURAL_COLUMNS)}."
                )
            self._matrix_n_columns = int(n_columns)
            self._matrix_columns = labels
            return labels

        if self._matrix_n_columns != int(n_columns):
            raise ArtifactError(
                f"Matrix column count changed from {self._matrix_n_columns} to {n_columns}."
            )
        if self._matrix_columns != labels:
            raise ArtifactError("Matrix columns changed after they were established.")
        return labels

    def _write_matrix_columns(self, labels: tuple[str, ...]) -> None:
        columns_path = self.storage.data_columns_path
        frame = pd.DataFrame(
            {
                "column_index": np.arange(len(labels), dtype="int64"),
                "column": list(labels),
            }
        )
        frame.to_parquet(columns_path, index=False)

    def _validate_matrix_row_names(
        self, matrix_payload: Mapping[str, Any], *, n_rows: int
    ) -> tuple[str, ...] | None:
        has_names = "row_names" in matrix_payload
        has_axis_name = "row_name" in matrix_payload
        if has_names != has_axis_name:
            raise ArtifactError(
                "Named matrix rows require both 'row_names' and 'row_name'."
            )
        if self._matrix_has_row_names is None:
            self._matrix_has_row_names = has_names
        elif self._matrix_has_row_names != has_names:
            raise ArtifactError(
                "Matrix batches must either all define row_names or all omit them."
            )
        if not has_names:
            return None

        axis_name = matrix_payload["row_name"]
        if not isinstance(axis_name, str) or not axis_name:
            raise ArtifactError("Matrix row_name must be a non-empty string.")
        if axis_name in STRUCTURAL_COLUMNS or axis_name.startswith("_"):
            raise ArtifactError(
                "Matrix row_name may not be a reserved structural name."
            )
        if self._matrix_row_name is None:
            self._matrix_row_name = axis_name
        elif self._matrix_row_name != axis_name:
            raise ArtifactError(
                f"Matrix row_name changed from {self._matrix_row_name!r} "
                f"to {axis_name!r}."
            )
        return _coerce_matrix_row_names(matrix_payload["row_names"], n_rows=n_rows)

    def _matrix_component_descriptor(
        self, *, format: str, values_format: str
    ) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "format": format,
            "path": "data/",
            "values": {"format": values_format, "path": "data/values/"},
            "columns": {"format": "parquet", "path": "data/columns.parquet"},
        }
        if self._matrix_has_row_names:
            descriptor["row_names"] = {
                "format": "parquet_dataset",
                "path": "data/row_names/",
                "name": self._matrix_row_name,
                "unique": True,
            }
        return descriptor

    def _write_matrix_row_names(
        self,
        row_names: tuple[str, ...],
        *,
        start_position: int,
        part_index: int,
    ) -> None:
        frame = pd.DataFrame(
            {
                "row_name": list(row_names),
                "_position": np.arange(
                    start_position,
                    start_position + len(row_names),
                    dtype="int64",
                ),
            }
        )
        frame.to_parquet(self.storage.data_row_names_part_path(part_index), index=False)

    def _descriptor(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "label": self.label,
            "status": self._status,
            "primary_key": list(self._primary_key),
            "n_rows": int(self._n_rows),
            "components": dict(self._components),
            "lineage": dict(self.lineage),
            "operation_id": self.operation_id,
            "write": {
                "mode": "batch",
                "parts": int(self._next_part_index),
            },
        }

    def _write_descriptor(self) -> None:
        _write_json(self.storage.descriptor_path, self._descriptor())

    def _ensure_open(self) -> None:
        if self._closed:
            state = "failed" if self._failed else "finalized"
            raise ArtifactError(
                f"Cannot write to {state} artifact writer {self.artifact_id!r}."
            )


def create_artifact_writer(
    *,
    artifact_type: ArtifactType | str,
    artifact_dir: str | Path,
    artifact_id: str,
    label: str,
    lineage: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
    lineage_mode: LineageMode | None = None,
    basis_artifact_ids: Sequence[str] | None = None,
    data_serializer: OtherDataSerializer | None = None,
    data_serializer_ref: CallableRef | None = None,
) -> ArtifactWriter:
    """Create a strict artifact writer."""
    try:
        kind = ArtifactType(artifact_type)
    except ValueError as exc:
        supported = ", ".join(item.value for item in ArtifactType)
        raise UnsupportedArtifactTypeError(
            f"Unsupported artifact_type={artifact_type!r}. Supported: {supported}."
        ) from exc

    resolved_lineage = dict(lineage or {})
    if lineage_mode is not None:
        resolved_lineage.setdefault("lineage_mode", lineage_mode)

    if basis_artifact_ids is not None:
        resolved_lineage.setdefault(
            "basis_artifact_ids",
            [str(artifact_id) for artifact_id in basis_artifact_ids],
        )

    return ArtifactWriter(
        artifact_dir=artifact_dir,
        artifact_id=artifact_id,
        artifact_type=kind,
        label=label,
        lineage=resolved_lineage,
        operation_id=operation_id,
        data_serializer=(data_serializer if kind == ArtifactType.OTHER else None),
        data_serializer_ref=(
            data_serializer_ref if kind == ArtifactType.OTHER else None
        ),
    )
