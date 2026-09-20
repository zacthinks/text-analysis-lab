"""Register externally computed results as durable TeAL artifacts.

The external computation is intentionally outside TeAL's execution/replay
machinery.  TeAL begins at registration: it records declared provenance
sources, validates declared structural lineage, streams one or more external
batches through the ordinary ArtifactWriter, and seals a normal artifact.

The canonical external protocol is writer-shaped batch payloads::

    {"keys": <DataFrame>, "data": ..., "metadata": <DataFrame>}

A caller may provide one payload or an iterable/generator of payloads.  Table
results additionally support pandas DataFrames and common tabular paths as
convenience adapters.  External workflows remain responsible for their own
checkpointing/resumption before registration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, LineageError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.importers import (
    _duckdb_literal,
    _duckdb_source_sql,
    _import_duckdb,
    _validate_duckdb_options,
)
from text_analysis_lab.core.lineage import (
    validate_lineage_mode,
    validate_primary_key_relationship,
)
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.types import ArtifactType, DEFAULT_OUTPUT_LABEL, LineageMode
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


TabularExternalFormat = Literal["csv", "jsonl", "parquet"]
_SUPPORTED_TYPES = {
    ArtifactType.TABLE,
    ArtifactType.JSONL,
    ArtifactType.DENSE_MATRIX,
    ArtifactType.SPARSE_MATRIX,
}
_RESERVED = {"_position", "_batch", "_row_offset"}


class RegisteredExternalOperator(BaseOperator):
    """Frozen descriptor for one non-replayable external registration."""

    operation_type = "register"

    def __init__(
        self,
        *,
        artifact_type: ArtifactType | str,
        primary_key: Sequence[str],
        lineage_mode: LineageMode | str,
        basis_labels: Sequence[str],
        output_label: str,
        external_kind: str,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.artifact_type = ArtifactType(artifact_type)
        if self.artifact_type not in _SUPPORTED_TYPES:
            raise ValueError(
                "register_external currently supports table, jsonl, dense_matrix, "
                "and sparse_matrix artifacts."
            )
        self.primary_key = tuple(str(v) for v in primary_key)
        self.lineage_mode = validate_lineage_mode(str(lineage_mode))
        self.basis_labels = tuple(str(v) for v in basis_labels)
        self.output_label = validate_output_label(output_label)
        self.external_kind = str(external_kind)

    @property
    def requires_source(self) -> bool:
        return False

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        unknown = [label for label in self.basis_labels if label not in sources]
        if unknown:
            raise ArtifactError(
                f"RegisteredExternalOperator basis_labels contain unknown source labels {unknown}."
            )
        return {
            self.output_label: OutputSpec(
                artifact_type=self.artifact_type,
                lineage_mode=self.lineage_mode,
                basis_labels=self.basis_labels,
            )
        }

    def to_json_state(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "primary_key": list(self.primary_key),
            "lineage_mode": self.lineage_mode,
            "basis_labels": list(self.basis_labels),
            "output_label": self.output_label,
            "external_kind": self.external_kind,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "RegisteredExternalOperator":
        return cls(
            artifact_type=str(state["artifact_type"]),
            primary_key=tuple(str(v) for v in state.get("primary_key", ())),
            lineage_mode=str(state.get("lineage_mode", "new_key")),
            basis_labels=tuple(str(v) for v in state.get("basis_labels", ())),
            output_label=str(state.get("output_label", DEFAULT_OUTPUT_LABEL)),
            external_kind=str(state.get("external_kind", "external")),
        )


def register_external(
    project: "Project",
    external: Any,
    *,
    artifact_type: ArtifactType | str = ArtifactType.TABLE,
    primary_key: str | Sequence[str],
    data_fields: str | Sequence[str] | None = None,
    metadata_fields: str | Sequence[str] | None = None,
    format: TabularExternalFormat | None = None,
    batch_size: int = 10_000,
    duckdb_options: Mapping[str, Any] | None = None,
    sources: Mapping[str, "BaseArtifact | str"] | None = None,
    lineage_mode: LineageMode | str = "new_key",
    basis_labels: str | Sequence[str] | None = None,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    memo: str | None = None,
) -> "BaseArtifact":
    """Register an externally produced result as a normal TeAL artifact.

    Canonical input is one writer-shaped payload mapping, or an iterable of such
    payloads.  This is batch-native and supports tables, JSONL, dense matrices,
    and sparse matrices.  Each payload must include ``keys``; optional ``data``
    and ``metadata`` use the same shapes accepted by :class:`ArtifactWriter`.

    Table artifacts additionally accept a pandas DataFrame or a CSV/JSONL/
    Parquet file.  A directory path is interpreted as a recursively discovered,
    lexically ordered Parquet dataset.  ``batch_size`` controls re-batching only
    for DataFrame/path adapters; caller-provided payload iterables retain their
    own batch boundaries.

    ``sources`` records all TeAL artifacts consumed by the external computation.
    ``basis_labels`` independently names the subset that structurally defines
    output row identity.  TeAL validates the declared key-schema relationship,
    but cannot verify or replay the external computation itself.
    """
    resolved_type = ArtifactType(artifact_type)
    if resolved_type not in _SUPPORTED_TYPES:
        raise ValueError(
            "register_external currently supports table, jsonl, dense_matrix, "
            "and sparse_matrix artifacts."
        )
    keys = _normalize_required_fields(primary_key, name="primary_key")
    data = _normalize_optional_fields(data_fields, name="data_fields")
    metadata = _normalize_optional_fields(metadata_fields, name="metadata_fields")
    label = validate_output_label(output_label)
    mode = validate_lineage_mode(str(lineage_mode))
    bases = _normalize_basis_labels(basis_labels)
    resolved_sources = _resolve_sources(project, sources)
    _validate_lineage_declaration(mode, bases, resolved_sources)

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    external_kind, external_descriptor, payloads = _normalize_external(
        external,
        artifact_type=resolved_type,
        primary_key=keys,
        data_fields=data,
        metadata_fields=metadata,
        format=format,
        batch_size=batch_size,
        duckdb_options=duckdb_options,
    )

    basis_artifacts = tuple(resolved_sources[name] for name in bases)
    validate_primary_key_relationship(
        basis_keys=[artifact.primary_key for artifact in basis_artifacts],
        output_key=keys,
        lineage_mode=mode,
    )

    operator = RegisteredExternalOperator(
        artifact_type=resolved_type,
        primary_key=keys,
        lineage_mode=mode,
        basis_labels=bases,
        output_label=label,
        external_kind=external_kind,
    )
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="register",
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
        operation_type="register",
        operator_id=operator_id,
        status="incomplete",
    )
    for source_label, artifact in resolved_sources.items():
        project.catalog.add_operation_source(operation_id, source_label, artifact.artifact_id)

    basis_ids = tuple(artifact.artifact_id for artifact in basis_artifacts)
    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=resolved_type,
        label=label,
        lineage_mode=mode,
        status="incomplete",
        basis_artifact_ids=basis_ids,
    )
    project.catalog.add_operation_output(operation_id, label, artifact_id, ordinal=0)

    writer = create_artifact_writer(
        artifact_type=resolved_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        operation_id=operation_id,
        lineage_mode=mode,
        basis_artifact_ids=basis_ids,
    )

    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "register",
        "operator_id": operator_id,
        "kind": "register_external",
        "status": "incomplete",
        "resumable": False,
        "replayable": False,
        "external": external_descriptor,
        "sources": {
            source_label: artifact.artifact_id
            for source_label, artifact in resolved_sources.items()
        },
        "basis_labels": list(bases),
        "output_artifact_ids": {label: artifact_id},
        "request": {
            "artifact_type": resolved_type.value,
            "primary_key": list(keys),
            "data_fields": list(data),
            "metadata_fields": list(metadata),
            "format": format,
            "lineage_mode": mode,
            "output_label": label,
            "batch_size": batch_size,
        },
    }
    _write_descriptor(operation_dir, descriptor)

    batches_written = 0
    rows_written = 0
    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )

        saw_payload = False
        for payload in payloads:
            saw_payload = True
            normalized = _validate_payload_primary_key(payload, keys)
            writer.write(normalized)
            batches_written += 1
            rows_written += len(normalized["keys"])

        if not saw_payload:
            raise ArtifactError(
                "External batch iterable produced no payloads. Supply at least one "
                "payload; for an empty table, provide a schema-bearing zero-row payload."
            )

        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        descriptor["registered_batches"] = batches_written
        descriptor["registered_rows"] = rows_written
        _write_descriptor(operation_dir, descriptor)
        project.storage.touch_manifest()
        project.query.clear_cache()
        return project.get_artifact(artifact_id)
    except Exception as exc:
        try:
            writer.mark_failed(exc)
        except Exception:
            pass
        try:
            project.catalog.mark_artifact_failed(artifact_id)
        except Exception:
            pass
        try:
            project.catalog.mark_operation_failed(operation_id, exc)
        except Exception:
            pass
        descriptor["status"] = "failed"
        descriptor["registered_batches"] = batches_written
        descriptor["registered_rows"] = rows_written
        descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
        _write_descriptor(operation_dir, descriptor)
        raise


def _normalize_external(
    external: Any,
    *,
    artifact_type: ArtifactType,
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
    format: TabularExternalFormat | None,
    batch_size: int,
    duckdb_options: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any], Iterable[Mapping[str, Any]]]:
    if isinstance(external, pd.DataFrame):
        if artifact_type != ArtifactType.TABLE:
            raise TypeError("DataFrame registration is supported only for table artifacts.")
        _validate_tabular_namespaces(primary_key, data_fields, metadata_fields)
        frame = _select_frame_fields(external, primary_key, data_fields, metadata_fields)
        return (
            "dataframe",
            {"kind": "dataframe", "rows": int(len(frame))},
            _frame_payloads(frame, primary_key, data_fields, metadata_fields, batch_size),
        )

    if isinstance(external, (str, Path)):
        if artifact_type != ArtifactType.TABLE:
            raise TypeError(
                "Path adapters are currently supported only for table artifacts. "
                "For matrix artifacts, pass one writer-shaped payload or an iterable "
                "of payloads so keys, matrix values, and feature columns remain explicit."
            )
        _validate_tabular_namespaces(primary_key, data_fields, metadata_fields)
        descriptor, payloads = _path_payloads(
            Path(external),
            primary_key=primary_key,
            data_fields=data_fields,
            metadata_fields=metadata_fields,
            format=format,
            batch_size=batch_size,
            duckdb_options=duckdb_options,
        )
        return "path", descriptor, payloads

    if isinstance(external, Mapping):
        if "keys" not in external:
            raise ArtifactError(
                "A single external payload mapping must contain a 'keys' channel."
            )
        return "payload", {"kind": "payload"}, iter((external,))

    if isinstance(external, Iterable):
        def payload_iter() -> Iterator[Mapping[str, Any]]:
            for index, item in enumerate(external):
                if isinstance(item, pd.DataFrame):
                    if artifact_type != ArtifactType.TABLE:
                        raise TypeError(
                            "DataFrame batch items are supported only for table artifacts."
                        )
                    _validate_tabular_namespaces(primary_key, data_fields, metadata_fields)
                    frame = _select_frame_fields(
                        item, primary_key, data_fields, metadata_fields
                    )
                    yield _frame_payload(frame, primary_key, data_fields, metadata_fields)
                    continue
                if not isinstance(item, Mapping) or "keys" not in item:
                    raise TypeError(
                        "External iterable items must be writer-shaped payload mappings "
                        "containing 'keys', or DataFrames for table artifacts; "
                        f"item {index} has type {type(item).__name__}."
                    )
                yield item

        return (
            "payload_iterable",
            {"kind": "payload_iterable", "python_type": type(external).__name__},
            payload_iter(),
        )

    raise TypeError(
        "external must be a pandas DataFrame, path, writer-shaped payload mapping, "
        "or iterable/generator of payloads."
    )


def _path_payloads(
    path: Path,
    *,
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
    format: TabularExternalFormat | None,
    batch_size: int,
    duckdb_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Iterable[Mapping[str, Any]]]:
    path = path.expanduser().resolve()
    resolved_format = _resolve_path_format(path, format)
    options = _validate_duckdb_options(duckdb_options)

    if path.is_dir():
        if resolved_format != "parquet":
            raise ArtifactError("Directory registration currently supports Parquet datasets only.")
        files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
        if not files:
            raise ArtifactError(f"Parquet dataset folder contains no .parquet files: {path}")
        source_arg = _duckdb_literal([p.as_posix() for p in files])
        option_sql = "".join(
            f", {name} = {_duckdb_literal(value)}" for name, value in options.items()
        )
        source_sql = f"read_parquet({source_arg}{option_sql})"
        descriptor = {
            "kind": "folder",
            "path": str(path),
            "format": "parquet",
            "matched_file_count": len(files),
            "matched_files": [p.relative_to(path).as_posix() for p in files],
        }
    else:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        source_sql = _duckdb_source_sql(resolved_format, path, options)
        stat = path.stat()
        descriptor = {
            "kind": "file",
            "path": str(path),
            "format": resolved_format,
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }

    selected = [*primary_key, *data_fields, *metadata_fields]
    if len(set(selected)) != len(selected):
        raise ArtifactError("External table key/data/metadata fields must not overlap.")

    def payloads() -> Iterator[Mapping[str, Any]]:
        duckdb = _import_duckdb()
        con = duckdb.connect(database=":memory:")
        try:
            con.execute("SET preserve_insertion_order = true")
            relation = con.sql(f"SELECT * FROM {source_sql}")
            columns = tuple(str(column) for column in relation.columns)
            missing = [name for name in selected if name not in columns]
            if missing:
                raise ArtifactError(
                    f"External {resolved_format} source is missing required column(s) {missing}."
                )
            projection = ", ".join(_quote_identifier(name) for name in selected)
            reader = con.sql(f"SELECT {projection} FROM {source_sql}").to_arrow_reader(
                batch_size=batch_size
            )
            emitted = False
            for record_batch in reader:
                frame = record_batch.to_pandas().reset_index(drop=True)
                if frame.empty:
                    continue
                emitted = True
                yield _frame_payload(frame, primary_key, data_fields, metadata_fields)
            if not emitted:
                # Establish an empty table schema using a zero-row query result.
                empty = con.sql(f"SELECT {projection} FROM {source_sql} LIMIT 0").df()
                yield _frame_payload(empty, primary_key, data_fields, metadata_fields)
        finally:
            con.close()

    return descriptor, payloads()


def _resolve_path_format(
    path: Path, explicit: TabularExternalFormat | None
) -> TabularExternalFormat:
    if explicit is not None:
        value = str(explicit).lower()
        if value not in {"csv", "jsonl", "parquet"}:
            raise ValueError("format must be 'csv', 'jsonl', 'parquet', or None.")
        return value  # type: ignore[return-value]
    if path.is_dir():
        return "parquet"
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    raise ValueError(
        f"Cannot infer external tabular format from {path.name!r}; pass format explicitly."
    )


def _frame_payloads(
    frame: pd.DataFrame,
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
    batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    if frame.empty:
        yield _frame_payload(frame, primary_key, data_fields, metadata_fields)
        return
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size].reset_index(drop=True)
        yield _frame_payload(batch, primary_key, data_fields, metadata_fields)


def _frame_payload(
    frame: pd.DataFrame,
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"keys": frame.loc[:, list(primary_key)].copy()}
    if data_fields:
        payload["data"] = frame.loc[:, list(data_fields)].copy()
    if metadata_fields:
        payload["metadata"] = frame.loc[:, list(metadata_fields)].copy()
    return payload


def _select_frame_fields(
    frame: pd.DataFrame,
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
) -> pd.DataFrame:
    columns = [str(v) for v in frame.columns]
    if len(set(columns)) != len(columns):
        raise ArtifactError("External DataFrame must have unique column names.")
    required = [*primary_key, *data_fields, *metadata_fields]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ArtifactError(f"External DataFrame is missing required column(s) {missing}.")
    reserved = sorted(set(required).intersection(_RESERVED))
    if reserved:
        raise ArtifactError(
            f"External registration cannot use reserved structural column(s) {reserved}."
        )
    renamed = frame.copy()
    renamed.columns = columns
    return renamed.loc[:, required].copy().reset_index(drop=True)


def _validate_tabular_namespaces(
    primary_key: Sequence[str],
    data_fields: Sequence[str],
    metadata_fields: Sequence[str],
) -> None:
    groups = [set(primary_key), set(data_fields), set(metadata_fields)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ArtifactError("External table primary_key, data_fields, and metadata_fields must not overlap.")
    reserved = sorted(set([*primary_key, *data_fields, *metadata_fields]).intersection(_RESERVED))
    if reserved:
        raise ArtifactError(
            f"External registration cannot use reserved structural column(s) {reserved}."
        )


def _validate_payload_primary_key(
    payload: Mapping[str, Any], primary_key: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("External payload must be a mapping.")
    if "keys" not in payload:
        raise ArtifactError("External payload must include 'keys'.")
    keys = payload["keys"]
    if not isinstance(keys, pd.DataFrame):
        raise ArtifactError("External payload 'keys' must be a pandas DataFrame.")
    actual = tuple(str(v) for v in keys.columns)
    expected = tuple(str(v) for v in primary_key)
    if actual != expected:
        raise ArtifactError(
            f"External payload key columns {actual} do not match declared primary_key {expected}."
        )
    return dict(payload)


def _validate_lineage_declaration(
    mode: LineageMode,
    bases: Sequence[str],
    sources: Mapping[str, "BaseArtifact"],
) -> None:
    if mode == "new_key":
        if bases:
            raise LineageError("new_key registration cannot declare basis_labels.")
        return
    if not bases:
        raise LineageError(f"{mode} registration requires at least one basis label.")
    unknown = [label for label in bases if label not in sources]
    if unknown:
        raise LineageError(
            f"basis_labels contain unknown source labels {unknown}; known sources are {tuple(sources)}."
        )


def _resolve_sources(
    project: "Project",
    sources: Mapping[str, "BaseArtifact | str"] | None,
) -> dict[str, "BaseArtifact"]:
    if sources is None:
        return {}
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping of source label to TeAL artifact.")
    resolved: dict[str, BaseArtifact] = {}
    for label, source in sources.items():
        if not isinstance(label, str) or not label:
            raise ValueError("source labels must be non-empty strings.")
        artifact = project.get_artifact(source)
        artifact.require_complete()
        resolved[label] = artifact
    return resolved


def _normalize_basis_labels(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(v) for v in value)
    else:
        raise TypeError("basis_labels must be a string, sequence of strings, or None.")
    if any(not value for value in values):
        raise ValueError("basis_labels must contain only non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError("basis_labels cannot contain duplicates.")
    return values


def _normalize_required_fields(value: str | Sequence[str], *, name: str) -> tuple[str, ...]:
    values = _normalize_optional_fields(value, name=name)
    if not values:
        raise ValueError(f"{name} must contain at least one field.")
    return values


def _normalize_optional_fields(
    value: str | Sequence[str] | None, *, name: str
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(v) for v in value)
    else:
        raise TypeError(f"{name} must be a string, sequence of strings, or None.")
    if any(not value for value in values):
        raise ValueError(f"{name} must contain only non-empty fields.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates.")
    return values


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _write_descriptor(operation_dir: Path, descriptor: Mapping[str, Any]) -> None:
    (operation_dir / "operation.json").write_text(
        json.dumps(dict(descriptor), indent=2, sort_keys=True), encoding="utf-8"
    )
