"""Project-side translation runner for TextAnalysisLab (TeAL).

This is the outsourced implementation target for Project.translate(...). TeAL
binds sources, creates a durable data-light execution plan, materializes only
one planned unit at a time, coordinates writers, gives translators an
operation-local temp directory for resumable state, and persists fitted
translators.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from text_analysis_lab.core.errors import OperatorError, OperatorNotFoundError
from text_analysis_lab.core.failure_cleanup import attempt_failure_cleanup
from text_analysis_lab.core.idempotence import (
    AliasPlan,
    AliasSpec,
    finalize_alias_plan,
    prepare_alias_plan,
    reused_outputs,
)
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.lineage import (
    validate_lineage_mode,
    validate_primary_key_relationship,
)
from text_analysis_lab.core.operator import (
    BatchResult,
    ColumnRequest,
    InputBatch,
    InputRequest,
    OutputMap,
    RunRoute,
    SourceMode,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import (
    DEFAULT_SOURCE_LABEL,
    ArtifactType,
    ColumnSelect,
    MetadataMode,
    QueryForm,
)
from text_analysis_lab.core.writer import ArtifactWriter, create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.operator import BaseTranslator, OutputSpec
    from text_analysis_lab.core.project import Project


TRANSLATION_MODES = get_args(TranslationMode)
RUN_ROUTES = get_args(RunRoute)
SOURCE_MODES = get_args(SourceMode)
QUERY_FORMS = get_args(QueryForm)
METADATA_MODES = get_args(MetadataMode)

_CHECKPOINT_FS_ATTEMPTS = 5
_CHECKPOINT_FS_BACKOFF_SECONDS = 0.02


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically publish a human-facing operation descriptor."""
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


@dataclass(frozen=True)
class _SourcePlan:
    source_label: str
    artifact_id: str
    mode: SourceMode
    source_batch_index: int
    source_batch_count: int
    start_position: int | None = None
    stop_position: int | None = None


@dataclass(frozen=True)
class _PlanUnit:
    unit_index: int
    sources: Mapping[str, _SourcePlan]


@dataclass
class _Runtime:
    operation_id: str
    operator_id: str
    mode: TranslationMode
    route: RunRoute
    output_specs: Mapping[str, OutputSpec]
    output_artifact_ids: Mapping[str, str]
    output_labels: tuple[str, ...]
    writers: Mapping[str, ArtifactWriter]
    operation_dir: Path
    operation_temp_dir: Path
    operator_temp_dir: Path
    writers_temp_dir: Path
    resumable: bool
    operator_snapshot_pending: bool


@dataclass(frozen=True)
class _PreparedOperator:
    operator_id: str
    snapshot_pending: bool


class _ExecutionPlan:
    """Operation-local durable plan for resumable translation execution."""

    def __init__(self, operation_dir: Path) -> None:
        self.operation_dir = Path(operation_dir)
        self.operation_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.operation_dir / "plan.sqlite"
        self._con: sqlite3.Connection | None = None
        self._init_db()

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self.db_path)
            self._con.row_factory = sqlite3.Row
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def _init_db(self) -> None:
        with self.con as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS plan_units (
                    unit_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    assigned_worker INTEGER,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS plan_sources (
                    unit_index INTEGER NOT NULL,
                    source_label TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    source_batch_index INTEGER NOT NULL,
                    source_batch_count INTEGER NOT NULL,
                    start_position INTEGER,
                    stop_position INTEGER,
                    PRIMARY KEY (unit_index, source_label),
                    FOREIGN KEY (unit_index) REFERENCES plan_units(unit_index)
                );
                """
            )

    def initialize(self, units: Iterable[_PlanUnit]) -> None:
        with self.con as con:
            existing = con.execute("SELECT COUNT(*) AS n FROM plan_units").fetchone()
            if existing is not None and int(existing["n"]) > 0:
                raise OperatorError("Execution plan already exists for this operation.")

            for unit in units:
                con.execute(
                    "INSERT INTO plan_units(unit_index, status) VALUES (?, 'pending')",
                    (int(unit.unit_index),),
                )
                for source in unit.sources.values():
                    con.execute(
                        """
                        INSERT INTO plan_sources(
                            unit_index,
                            source_label,
                            artifact_id,
                            mode,
                            source_batch_index,
                            source_batch_count,
                            start_position,
                            stop_position
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(unit.unit_index),
                            source.source_label,
                            source.artifact_id,
                            source.mode,
                            source.source_batch_index,
                            source.source_batch_count,
                            source.start_position,
                            source.stop_position,
                        ),
                    )

    def reset_running_to_pending(self) -> None:
        """Reset retryable non-complete units before resume.

        A unit may be left ``running`` by process interruption or explicitly
        marked ``failed`` when the operation raised.  Resume must retry both;
        completed units remain untouched.
        """
        with self.con as con:
            con.execute(
                """
                UPDATE plan_units
                SET status = 'pending', assigned_worker = NULL, started_at = NULL, error = NULL
                WHERE status IN ('running', 'failed')
                """
            )

    def iter_pending(self) -> Iterable[_PlanUnit]:
        with self.con as con:
            rows = con.execute(
                """
                SELECT unit_index
                FROM plan_units
                WHERE status = 'pending'
                ORDER BY unit_index
                """
            ).fetchall()
        for row in rows:
            yield self.get_unit(int(row["unit_index"]))

    def get_unit(self, unit_index: int) -> _PlanUnit:
        with self.con as con:
            rows = con.execute(
                """
                SELECT * FROM plan_sources
                WHERE unit_index = ?
                ORDER BY source_label
                """,
                (int(unit_index),),
            ).fetchall()
        if not rows:
            raise OperatorError(f"Plan unit {unit_index} has no source records.")
        sources = {
            str(row["source_label"]): _SourcePlan(
                source_label=str(row["source_label"]),
                artifact_id=str(row["artifact_id"]),
                mode=str(row["mode"]),  # type: ignore[arg-type]
                source_batch_index=int(row["source_batch_index"]),
                source_batch_count=int(row["source_batch_count"]),
                start_position=(
                    None
                    if row["start_position"] is None
                    else int(row["start_position"])
                ),
                stop_position=(
                    None if row["stop_position"] is None else int(row["stop_position"])
                ),
            )
            for row in rows
        }
        return _PlanUnit(unit_index=int(unit_index), sources=sources)

    def mark_running(self, unit_index: int, *, worker_index: int | None = None) -> None:
        with self.con as con:
            con.execute(
                """
                UPDATE plan_units
                SET status = 'running', assigned_worker = ?, started_at = datetime('now'), error = NULL
                WHERE unit_index = ? AND status = 'pending'
                """,
                (worker_index, int(unit_index)),
            )

    def mark_complete(self, unit_index: int) -> None:
        with self.con as con:
            con.execute(
                """
                UPDATE plan_units
                SET status = 'complete', completed_at = datetime('now'), error = NULL
                WHERE unit_index = ?
                """,
                (int(unit_index),),
            )

    def mark_failed(self, unit_index: int, error: BaseException | str) -> None:
        with self.con as con:
            con.execute(
                """
                UPDATE plan_units
                SET status = 'failed', error = ?
                WHERE unit_index = ?
                """,
                (str(error), int(unit_index)),
            )

    def complete_count(self) -> int:
        row = self.con.execute(
            "SELECT COUNT(*) AS n FROM plan_units WHERE status = 'complete'"
        ).fetchone()
        return int(row["n"] if row is not None else 0)

    def total_count(self) -> int:
        row = self.con.execute("SELECT COUNT(*) AS n FROM plan_units").fetchone()
        return int(row["n"] if row is not None else 0)


def translate(
    project: Project,
    translator: BaseTranslator,
    sources: BaseArtifact
    | str
    | Sequence[BaseArtifact | str]
    | Mapping[str, BaseArtifact | str],
    *,
    workers: int = 1,
    batch_size: int | None = None,
    max_outstanding_units: int | None = None,
    memo: str | None = None,
    alias: AliasSpec = None,
    overwrite: bool = False,
    **params: Any,
) -> Mapping[str, BaseArtifact]:
    """Run a new TeAL translation operation and return output artifacts by label."""
    source_bindings = _resolve_sources(project, sources)
    translator.prepare_for_translation(project=project, sources=source_bindings)
    mode = _select_mode(translator)
    operation_params = translator.validate_operation_params(
        params,
        sources=source_bindings,
        mode=mode,
    )
    translation_request = _validate_translation_request(
        TranslationRequest(
            workers=workers,
            batch_size=batch_size,
            max_outstanding_units=max_outstanding_units,
            params=operation_params,
        )
    )

    input_request = _normalize_input_request(
        translator.input_request(
            sources=source_bindings,
            mode=mode,
            request=translation_request,
        )
    )
    _validate_source_bindings(source_bindings, input_request)
    output_specs = dict(
        translator.output_specs_in_dependency_order(
            sources=source_bindings,
            request=translation_request,
        )
    )
    _validate_operation_label_namespace(input_request, output_specs)
    _validate_output_basis_labels(input_request, output_specs)
    alias_plan = prepare_alias_plan(
        project, tuple(output_specs), alias, overwrite=overwrite
    )
    if alias_plan is not None and alias_plan.reuse:
        return reused_outputs(project, alias_plan)
    _warn_batch_size_adjustments(translation_request, input_request)
    route, effective_workers = _select_route(
        translator,
        mode=mode,
        request=translation_request,
        input_request=input_request,
    )
    serialized_operation_params = _serialize_operation_params(
        translator,
        translation_request.params,
    )
    runtime = _prepare_runtime(
        project=project,
        translator=translator,
        sources=source_bindings,
        mode=mode,
        route=route,
        output_specs=output_specs,
    )
    plan: _ExecutionPlan | None = None
    try:
        _save_operation_metadata(
            runtime=runtime,
            translator=translator,
            mode=mode,
            route=route,
            effective_workers=effective_workers,
            request=translation_request,
            serialized_operation_params=serialized_operation_params,
            input_request=input_request,
            sources=source_bindings,
            output_specs=output_specs,
            output_artifact_ids=runtime.output_artifact_ids,
            alias_plan=alias_plan,
        )
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation",
                target_id=runtime.operation_id,
                body=memo,
            )
        plan = _create_execution_plan(
            runtime=runtime,
            sources=source_bindings,
            input_request=input_request,
        )
        translator.initialize_translation(
            mode=mode,
            route=route,
            request=translation_request,
        )
        # A resumable operation needs a recovery point even before unit 0.
        # Otherwise a failure in the first unit leaves no translator/writer
        # checkpoint for resume_translate(...) to restore.
        _save_intermediate_state(
            translator=translator,
            runtime=runtime,
            mode=mode,
            route=route,
        )
        _run_plan(
            translator=translator,
            runtime=runtime,
            plan=plan,
            sources=source_bindings,
            input_request=input_request,
            mode=mode,
            route=route,
            request=translation_request,
            workers=effective_workers,
        )

        _commit_final_outputs(
            runtime,
            translator.finalize_translation(
                mode=mode,
                request=translation_request,
            ),
        )

        _serialize_pending_operator_snapshot(project, translator, runtime)

        _complete_runtime(project, runtime)
        outputs = _load_output_artifacts(project, runtime)
        finalize_alias_plan(project, alias_plan, outputs)
        return outputs
    except Exception as exc:
        operation = project.catalog.get_operation(runtime.operation_id)
        if str(operation["status"]) != "complete":
            _fail_runtime(project, runtime, exc)
        raise
    finally:
        if plan is not None:
            plan.close()


def resume_translate(
    project: Project,
    operation_id: str,
) -> Mapping[str, BaseArtifact]:
    """Resume an incomplete translation operation from its durable plan.

    Resume restores the canonical translator from operation-local intermediate
    state. It does not call ``initialize_translation(...)``, which is reserved for
    new operations.
    """
    metadata = _load_operation_metadata(project, operation_id)
    translator = _load_translator_for_resume(project, operation_id, metadata)
    mode = _validate_translation_mode(str(metadata["mode"]))
    route = _validate_run_route(str(metadata["route"]))
    raw_input_request = metadata.get("input_request")
    if not isinstance(raw_input_request, Mapping):
        raise OperatorError("Operation descriptor is missing input_request metadata.")
    input_request = _normalize_input_request(
        _input_request_from_dict(raw_input_request)
    )

    request_data = metadata.get("request", {})
    if not isinstance(request_data, Mapping):
        raise OperatorError("Operation metadata request must be a mapping.")
    params = request_data.get("params", {})
    if not isinstance(params, Mapping):
        raise OperatorError("Operation metadata request.params must be a mapping.")
    restored_params = translator.deserialize_operation_params(params)
    if not isinstance(restored_params, Mapping):
        raise OperatorError("deserialize_operation_params() must return a mapping.")

    request = _validate_translation_request(
        TranslationRequest(
            workers=request_data.get("workers", 1),
            batch_size=request_data.get("batch_size"),
            max_outstanding_units=request_data.get("max_outstanding_units"),
            params=restored_params,
        )
    )
    effective_workers = _validate_workers(metadata.get("effective_workers"))

    if not translator.supports_resume(mode=mode, route=route):
        raise OperatorError(
            f"Operation {operation_id} cannot be resumed because "
            f"{translator.__class__.__name__} does not support resume for {route} {mode}."
        )

    raw_sources = metadata.get("sources")
    if not isinstance(raw_sources, Mapping):
        raise OperatorError("Operation descriptor is missing sources metadata.")
    source_bindings = {
        str(label): project.get_artifact(str(artifact_id))
        for label, artifact_id in raw_sources.items()
    }
    translator.prepare_for_translation(project=project, sources=source_bindings)
    _validate_source_bindings(source_bindings, input_request)
    raw_output_specs = metadata.get("output_specs")
    if not isinstance(raw_output_specs, Mapping):
        raise OperatorError("Operation descriptor is missing output_specs metadata.")
    output_specs = _output_specs_from_dict(raw_output_specs)
    _validate_operation_label_namespace(input_request, output_specs)
    _validate_output_basis_labels(input_request, output_specs)

    runtime = _runtime_from_metadata(
        project,
        translator,
        metadata,
        mode=mode,
        route=route,
        output_specs=output_specs,
    )
    plan = _ExecutionPlan(runtime.operation_dir)
    plan.reset_running_to_pending()

    try:
        _run_plan(
            translator=translator,
            runtime=runtime,
            plan=plan,
            sources=source_bindings,
            input_request=input_request,
            mode=mode,
            route=route,
            request=request,
            workers=effective_workers,
        )
        _commit_final_outputs(
            runtime,
            translator.finalize_translation(mode=mode, request=request),
        )
        _serialize_pending_operator_snapshot(project, translator, runtime)
        _complete_runtime(project, runtime)
        outputs = _load_output_artifacts(project, runtime)
        raw_alias_plan = metadata.get("alias_plan")
        if isinstance(raw_alias_plan, Mapping):
            finalize_alias_plan(project, AliasPlan.from_dict(raw_alias_plan), outputs)
        return outputs
    except Exception as exc:
        operation = project.catalog.get_operation(runtime.operation_id)
        if str(operation["status"]) != "complete":
            _fail_runtime(project, runtime, exc)
        raise
    finally:
        plan.close()


# ---------------------------------------------------------------------------
# Mode / route
# ---------------------------------------------------------------------------


def _select_mode(translator: BaseTranslator) -> TranslationMode:
    if not translator.requires_fit or translator.is_fitted:
        return "translate"
    if translator.supports_fit_translate:
        return "fit_translate"
    raise OperatorError(
        f"{translator.__class__.__name__} must be fitted before translation."
    )


def _validate_translation_request(request: TranslationRequest) -> TranslationRequest:
    if not isinstance(request.params, Mapping):
        raise OperatorError("operation params must be a mapping.")
    return TranslationRequest(
        workers=_validate_workers(request.workers),
        batch_size=_validate_batch_size(request.batch_size),
        max_outstanding_units=_validate_max_outstanding_units(
            request.max_outstanding_units
        ),
        params=dict(request.params),
    )


def _validate_workers(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperatorError("workers must be a positive integer.")
    if value <= 0:
        raise OperatorError("workers must be a positive integer.")
    return value


def _validate_batch_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OperatorError("batch_size must be a positive integer or None.")
    return value


def _validate_max_outstanding_units(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OperatorError("max_outstanding_units must be a positive integer or None.")
    return value


def _validate_translation_mode(mode: str) -> TranslationMode:
    if mode not in TRANSLATION_MODES:
        raise OperatorError(f"Invalid translation mode {mode!r}.")
    return cast(TranslationMode, mode)


def _validate_run_route(route: str) -> RunRoute:
    if route not in RUN_ROUTES:
        raise OperatorError(f"Invalid run route {route!r}.")
    return cast(RunRoute, route)


def _warn_batch_size_adjustments(
    request: TranslationRequest, input_request: InputRequest
) -> None:
    if request.batch_size is None:
        return

    batched = {
        label: source
        for label, source in input_request.items()
        if source.mode == "batches"
    }
    if not batched:
        warnings.warn(
            "batch_size was ignored because every requested source is "
            "materialized as a full artifact.",
            stacklevel=2,
        )
        return

    selected = {label: source.batch_size for label, source in batched.items()}
    if any(size != request.batch_size for size in selected.values()):
        warnings.warn(
            f"batch_size={request.batch_size} was treated as a request; "
            f"the translator selected effective batch sizes {selected}.",
            stacklevel=2,
        )


def _select_route(
    translator: BaseTranslator,
    *,
    mode: TranslationMode,
    request: TranslationRequest,
    input_request: InputRequest,
) -> tuple[RunRoute, int]:
    if request.workers == 1:
        return "sequential", 1

    if all(source.mode == "full_artifact" for source in input_request.values()):
        warnings.warn(
            "workers was ignored because every requested source is materialized "
            "as a full artifact, so there are no source batches to distribute.",
            stacklevel=2,
        )
        return "sequential", 1

    supported = (
        translator.supports_parallel_translate
        if mode == "translate"
        else translator.supports_parallel_fit_translate
    )
    if not supported:
        warnings.warn(
            f"workers={request.workers} was ignored because "
            f"{translator.__class__.__name__} does not support parallel {mode}.",
            stacklevel=2,
        )
        return "sequential", 1
    return "parallel", request.workers


# ---------------------------------------------------------------------------
# Source binding / input request validation
# ---------------------------------------------------------------------------


def _normalize_input_request(
    request: SourceRequest | Mapping[str, SourceRequest],
) -> dict[str, SourceRequest]:
    if isinstance(request, SourceRequest):
        request = {DEFAULT_SOURCE_LABEL: request}
    elif isinstance(request, Mapping):
        request = dict(request)
    else:
        raise OperatorError(
            "input_request() must return SourceRequest or label mapping."
        )

    if not request:
        raise OperatorError("input_request() must request at least one source.")

    out: dict[str, SourceRequest] = {}
    for label, source_request in request.items():
        if not isinstance(label, str) or not label:
            raise OperatorError("Input source labels must be non-empty strings.")
        if not isinstance(source_request, SourceRequest):
            raise OperatorError(f"Input source {label!r} must be a SourceRequest.")
        _validate_source_request(label, source_request)
        out[label] = source_request
    return out


def _validate_source_request(label: str, request: SourceRequest) -> None:
    _accepted_artifact_types(label, request)
    if request.mode not in SOURCE_MODES:
        raise OperatorError(
            f"Input source {label!r} has invalid mode {request.mode!r}."
        )
    if request.form not in QUERY_FORMS:
        raise OperatorError(
            f"Input source {label!r} has invalid form {request.form!r}."
        )
    if request.metadata_mode not in METADATA_MODES:
        raise OperatorError(
            f"Input source {label!r} has invalid metadata_mode "
            f"{request.metadata_mode!r}."
        )
    if not isinstance(request.columns, ColumnRequest):
        raise OperatorError(f"Input source {label!r} columns must be a ColumnRequest.")
    _validate_column_select(label, "key", request.columns.keys)
    _validate_column_select(label, "data", request.columns.data)
    _validate_column_select(label, "metadata", request.columns.metadata)
    if not isinstance(request.include_position, bool):
        raise OperatorError(
            f"Input source {label!r} include_position must be a boolean."
        )
    if request.mode == "batches":
        if isinstance(request.batch_size, bool) or not isinstance(
            request.batch_size, int
        ):
            raise OperatorError(
                f"Input source {label!r} batch_size must be a positive integer "
                "when mode='batches'."
            )
        if request.batch_size <= 0:
            raise OperatorError(f"Input source {label!r} batch_size must be positive.")
    else:
        if request.batch_size is not None:
            raise OperatorError(
                f"Input source {label!r} batch_size must be None when "
                "mode='full_artifact'."
            )


def _validate_column_select(label: str, namespace: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        if value:
            return
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (bytes, bytearray))
        and all(isinstance(column, str) and column for column in value)
    ):
        return
    raise OperatorError(
        f"Input source {label!r} {namespace} columns must be a boolean, "
        "a non-empty string, or a sequence of non-empty strings."
    )


def _accepted_artifact_types(
    label: str, request: SourceRequest
) -> tuple[ArtifactType, ...]:
    raw = request.artifact_type
    if isinstance(raw, (str, ArtifactType)):
        values = (raw,)
    else:
        values = tuple(raw)

    if not values:
        raise OperatorError(
            f"Input source {label!r} must accept at least one artifact type."
        )

    try:
        return tuple(ArtifactType(value) for value in values)
    except ValueError as exc:
        raise OperatorError(
            f"Input source {label!r} declares unsupported artifact_type {raw!r}."
        ) from exc


def _validate_source_artifact(
    *,
    label: str,
    artifact: BaseArtifact,
    request: SourceRequest,
) -> None:
    artifact.require_complete()
    accepted = _accepted_artifact_types(label, request)
    actual = ArtifactType(artifact.artifact_type)
    if actual not in accepted:
        raise OperatorError(
            f"Input source {label!r} expected artifact_type "
            f"{tuple(item.value for item in accepted)}; got {actual.value!r} "
            f"from artifact {artifact.artifact_id}."
        )


def _resolve_sources(
    project: Project,
    sources: BaseArtifact
    | str
    | Sequence[BaseArtifact | str]
    | Mapping[str, BaseArtifact | str],
) -> dict[str, BaseArtifact]:
    if isinstance(sources, Mapping):
        if not sources:
            raise OperatorError("Translation requires at least one source artifact.")
        resolved: dict[str, BaseArtifact] = {}
        for label, source in sources.items():
            if not isinstance(label, str) or not label:
                raise OperatorError("Source labels must be non-empty strings.")
            resolved[label] = project.get_artifact(source)
        return resolved
    if isinstance(sources, Sequence) and not isinstance(
        sources, (str, bytes, bytearray)
    ):
        if not sources:
            raise OperatorError("Translation requires at least one source artifact.")
        return {
            f"source_{index}": project.get_artifact(source)
            for index, source in enumerate(sources)
        }
    return {DEFAULT_SOURCE_LABEL: project.get_artifact(sources)}


def _validate_source_bindings(
    sources: Mapping[str, BaseArtifact],
    request: InputRequest,
) -> None:
    missing = [label for label in request if label not in sources]
    extra = [label for label in sources if label not in request]
    if missing or extra:
        raise OperatorError(
            "Source mapping does not match input_request labels. "
            f"Missing: {missing}; extra: {extra}."
        )
    for label, artifact in sources.items():
        _validate_source_artifact(
            label=label, artifact=artifact, request=request[label]
        )


def _validate_operation_label_namespace(
    input_request: InputRequest,
    output_specs: Mapping[str, OutputSpec],
) -> None:
    source_labels = set(input_request)
    output_labels = set(output_specs)
    overlap = source_labels & output_labels
    if overlap:
        raise OperatorError(
            "Input source labels and output labels must be distinct; "
            f"overlap: {tuple(sorted(overlap))}."
        )


def _validate_output_basis_labels(
    input_request: InputRequest,
    output_specs: Mapping[str, OutputSpec],
) -> None:
    known_labels = set(input_request) | set(output_specs)
    for label, spec in output_specs.items():
        basis_labels = cast(tuple[str, ...], spec.basis_labels)
        unknown = [basis for basis in basis_labels if basis not in known_labels]
        if unknown:
            raise OperatorError(
                f"Output {label!r} declares unknown basis_labels {unknown}. "
                f"Known labels: {tuple(sorted(known_labels))}."
            )


# ---------------------------------------------------------------------------
# Durable plan construction / materialization
# ---------------------------------------------------------------------------


def _create_execution_plan(
    *,
    runtime: _Runtime,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
) -> _ExecutionPlan:
    plan = _ExecutionPlan(runtime.operation_dir)
    units = tuple(_build_plan_units(sources, input_request))
    plan.initialize(units)
    return plan


def _build_plan_units(
    sources: Mapping[str, BaseArtifact], input_request: InputRequest
) -> Iterable[_PlanUnit]:
    planned_sources: dict[str, list[_SourcePlan]] = {}

    for label, source_request in input_request.items():
        artifact = sources[label]
        if source_request.mode == "full_artifact":
            planned_sources[label] = [
                _SourcePlan(
                    source_label=label,
                    artifact_id=artifact.artifact_id,
                    mode="full_artifact",
                    source_batch_index=0,
                    source_batch_count=1,
                )
            ]
            continue

        if source_request.batch_size is None:
            raise OperatorError(
                f"Cannot create a durable batch plan for source {label!r}: "
                "batch_size must be specified when mode='batches'."
            )
        n_rows = artifact.n_rows
        if n_rows is None:
            raise OperatorError(
                f"Cannot create a durable batch plan for source {label!r}: "
                f"artifact {artifact.artifact_id} does not record n_rows."
            )
        planned_sources[label] = _position_slices_for_source(
            label=label,
            artifact_id=artifact.artifact_id,
            n_rows=int(n_rows),
            batch_size=source_request.batch_size,
        )

    max_units = max((len(units) for units in planned_sources.values()), default=0)
    for unit_index in range(max_units):
        unit_sources = {
            label: source_units[unit_index]
            for label, source_units in planned_sources.items()
            if unit_index < len(source_units)
        }
        if unit_sources:
            yield _PlanUnit(unit_index=unit_index, sources=unit_sources)


def _position_slices_for_source(
    *,
    label: str,
    artifact_id: str,
    n_rows: int,
    batch_size: int,
) -> list[_SourcePlan]:
    ranges = tuple(enumerate(range(0, max(n_rows, 0), batch_size)))
    batch_count = len(ranges)
    return [
        _SourcePlan(
            source_label=label,
            artifact_id=artifact_id,
            mode="batches",
            source_batch_index=batch_index,
            source_batch_count=batch_count,
            start_position=start,
            stop_position=min(start + batch_size, n_rows),
        )
        for batch_index, start in ranges
    ]


def _materialize_plan_unit(
    unit: _PlanUnit,
    *,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
) -> Mapping[str, InputBatch]:
    return {
        label: _materialize_source_plan(
            source_plan,
            artifact=sources[label],
            request=input_request[label],
        )
        for label, source_plan in unit.sources.items()
    }


def _materialize_source_plan(
    source_plan: _SourcePlan,
    *,
    artifact: BaseArtifact,
    request: SourceRequest,
) -> InputBatch:
    if source_plan.mode == "full_artifact":
        data = _query_source(artifact, request, where=None)
    else:
        if source_plan.start_position is None or source_plan.stop_position is None:
            raise OperatorError(
                f"Malformed plan source for {source_plan.source_label!r}."
            )
        data = _query_source(
            artifact,
            request,
            where=_position_range_where(
                start=source_plan.start_position,
                stop=source_plan.stop_position,
            ),
        )

    return InputBatch(
        source_label=source_plan.source_label,
        artifact_id=artifact.artifact_id,
        primary_key=tuple(artifact.primary_key),
        data=data,
        batch_index=source_plan.source_batch_index,
        batch_count=source_plan.source_batch_count,
        is_first=source_plan.source_batch_index == 0,
        is_last=source_plan.source_batch_index == source_plan.source_batch_count - 1,
    )


def _position_range_where(*, start: int, stop: int) -> str:
    """Return the half-open artifact-position predicate for one planned slice."""
    start_value = int(start)
    stop_value = int(stop)
    if start_value < 0 or stop_value < start_value:
        raise OperatorError(
            f"Invalid planned position slice: start={start_value}, stop={stop_value}."
        )
    return f"_position >= {start_value} AND _position < {stop_value}"


def _query_source(
    artifact: BaseArtifact,
    request: SourceRequest,
    *,
    where: str | None,
) -> Any:
    return artifact.query(
        key_columns=request.columns.keys,
        data_columns=request.columns.data,
        metadata_columns=request.columns.metadata,
        metadata_mode=request.metadata_mode,
        where=where,
        form=request.form,
        iter_batches=False,
        include_position=request.include_position,
    )


# ---------------------------------------------------------------------------
# Catalog / writer setup
# ---------------------------------------------------------------------------


def _prepare_runtime(
    *,
    project: Project,
    translator: BaseTranslator,
    sources: Mapping[str, BaseArtifact],
    mode: TranslationMode,
    route: RunRoute,
    output_specs: Mapping[str, OutputSpec],
) -> _Runtime:
    prepared_operator = _prepare_operator_snapshot(
        project=project,
        translator=translator,
        mode=mode,
    )

    operation_id = next_id(project.storage.manifest_path, "operation")
    operation_dir = project.storage.operation_dir(operation_id)
    operation_temp_dir = project.storage.operation_temp_dir(operation_id)
    operator_temp_dir = project.storage.operation_temp_dir_for_operator(operation_id)
    writers_temp_dir = project.storage.operation_temp_dir_for_writers(operation_id)
    operation_dir.mkdir(parents=True, exist_ok=True)
    operation_temp_dir.mkdir(parents=True, exist_ok=True)
    operator_temp_dir.mkdir(parents=True, exist_ok=True)
    writers_temp_dir.mkdir(parents=True, exist_ok=True)

    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type=translator.operation_type,
        operator_id=prepared_operator.operator_id,
        status="incomplete",
    )

    for label, artifact in sources.items():
        project.catalog.add_operation_source(
            operation_id,
            label,
            artifact.artifact_id,
        )

    output_ids = {
        label: next_id(project.storage.manifest_path, "artifact")
        for label in output_specs
    }
    source_artifact_ids = {
        label: artifact.artifact_id for label, artifact in sources.items()
    }
    writers: dict[str, ArtifactWriter] = {}

    try:
        for ordinal, (label, spec) in enumerate(output_specs.items()):
            artifact_id = output_ids[label]
            basis_ids = _basis_ids(
                spec,
                source_artifact_ids=source_artifact_ids,
                output_artifact_ids=output_ids,
            )
            project.catalog.register_artifact(
                artifact_id=artifact_id,
                artifact_type=spec.artifact_type,
                label=label,
                lineage_mode=spec.lineage_mode,
                status="incomplete",
                basis_artifact_ids=basis_ids,
            )
            project.catalog.add_operation_output(
                operation_id, label, artifact_id, ordinal=ordinal
            )
            writers[label] = create_artifact_writer(
                artifact_type=spec.artifact_type,
                artifact_dir=project.storage.artifact_dir(artifact_id),
                artifact_id=artifact_id,
                label=label,
                operation_id=operation_id,
                lineage_mode=spec.lineage_mode,
                basis_artifact_ids=basis_ids,
                data_serializer=getattr(spec, "data_serializer", None),
                data_serializer_ref=getattr(spec, "data_serializer_ref", None),
            )
    except Exception as exc:
        for label, writer in writers.items():
            attempt_failure_cleanup(
                lambda writer=writer, exc=exc: writer.mark_failed(exc),
                primary_error=exc,
                label=f"writer {label!r} failure marking",
            )
            attempt_failure_cleanup(
                lambda label=label: project.catalog.mark_artifact_failed(
                    output_ids[label]
                ),
                primary_error=exc,
                label=f"artifact {output_ids[label]!r} failure marking",
            )
        attempt_failure_cleanup(
            lambda exc=exc: project.catalog.mark_operation_failed(operation_id, exc),
            primary_error=exc,
            label=f"operation {operation_id!r} failure marking",
        )
        if prepared_operator.snapshot_pending:
            attempt_failure_cleanup(
                lambda: project.catalog.mark_operator_snapshot_failed(
                    prepared_operator.operator_id
                ),
                primary_error=exc,
                label=f"operator {prepared_operator.operator_id!r} snapshot failure marking",
            )
        raise

    return _Runtime(
        operation_id=operation_id,
        operator_id=prepared_operator.operator_id,
        mode=mode,
        route=route,
        output_specs=output_specs,
        output_artifact_ids=output_ids,
        output_labels=tuple(output_specs),
        writers=writers,
        operation_dir=operation_dir,
        operation_temp_dir=operation_temp_dir,
        operator_temp_dir=operator_temp_dir,
        writers_temp_dir=writers_temp_dir,
        resumable=translator.supports_resume(mode=mode, route=route),
        operator_snapshot_pending=prepared_operator.snapshot_pending,
    )


def _runtime_from_metadata(
    project: Project,
    translator: BaseTranslator,
    metadata: Mapping[str, Any],
    *,
    mode: TranslationMode,
    route: RunRoute,
    output_specs: Mapping[str, OutputSpec],
) -> _Runtime:
    operation_id = str(metadata["operation_id"])
    operation_dir = project.storage.operation_dir(operation_id)
    operation_temp_dir = project.storage.operation_temp_dir(operation_id)
    operator_temp_dir = project.storage.operation_temp_dir_for_operator(operation_id)
    writers_temp_dir = project.storage.operation_temp_dir_for_writers(operation_id)
    output_artifact_ids = {
        str(label): str(artifact_id)
        for label, artifact_id in dict(metadata["output_artifact_ids"]).items()
    }
    writers = _open_writers_for_resume(
        project,
        output_artifact_ids,
        writers_temp_dir=writers_temp_dir,
    )
    return _Runtime(
        operation_id=operation_id,
        operator_id=str(metadata["operator_id"]),
        mode=mode,
        route=route,
        output_specs=output_specs,
        output_artifact_ids=output_artifact_ids,
        output_labels=tuple(output_artifact_ids),
        writers=writers,
        operation_dir=operation_dir,
        operation_temp_dir=operation_temp_dir,
        operator_temp_dir=operator_temp_dir,
        writers_temp_dir=writers_temp_dir,
        resumable=bool(metadata.get("resumable", True)),
        operator_snapshot_pending=(
            project.catalog.operator_snapshot_status(str(metadata["operator_id"]))
            == "pending"
        ),
    )


def _prepare_operator_snapshot(
    *,
    project: Project,
    translator: BaseTranslator,
    mode: TranslationMode,
) -> _PreparedOperator:
    """Prepare the operator catalog row and snapshot timing for this operation."""
    if translator.operator_id is None:
        operator_id = next_id(project.storage.manifest_path, "operator")
        translator.assign_operator_id(operator_id)
        project.catalog.register_operator(
            operator_id=operator_id,
            operation_type=translator.operation_type,
            snapshot_status="pending",
        )
        if mode == "translate":
            try:
                _save_operator(project, translator, operator_id)
            except Exception:
                project.catalog.mark_operator_snapshot_failed(operator_id)
                raise
            project.catalog.mark_operator_serialized(operator_id)
            return _PreparedOperator(operator_id=operator_id, snapshot_pending=False)
        return _PreparedOperator(operator_id=operator_id, snapshot_pending=True)

    operator_id = str(translator.operator_id)
    try:
        row = project.catalog.resolve_operator(operator_id, include_deleted=False)
    except OperatorNotFoundError as exc:
        raise OperatorError(
            f"Operator object has operator_id={operator_id!r}, but that operator "
            "is not registered in this project. Pass an unsaved translator or "
            "load a serialized project operator."
        ) from exc

    snapshot_status = str(row.get("snapshot_status", "serialized"))
    if snapshot_status != "serialized":
        hint = (
            _pending_operator_resume_hint(project, operator_id)
            if snapshot_status == "pending"
            else "Create a new translator."
        )
        raise OperatorError(
            f"Operator {operator_id} has snapshot_status={snapshot_status!r}; "
            "new translate operations require a serialized operator snapshot. "
            f"{hint}"
        )

    if mode == "fit_translate":
        raise OperatorError(
            f"Cannot fit_translate serialized operator {operator_id}. "
            "Create a new unsaved translator so fitting can produce a new operator snapshot."
        )

    if not translator.is_frozen:
        raise OperatorError(
            f"Operator {operator_id} is registered as serialized but the provided "
            "object is not frozen. Load the operator from the project or pass a new "
            "unsaved translator."
        )

    descriptor_path = project.storage.operator_descriptor_path(operator_id)
    if not descriptor_path.exists():
        raise OperatorError(
            f"Operator {operator_id} is registered as serialized, but missing "
            f"operator descriptor at {descriptor_path}."
        )

    return _PreparedOperator(operator_id=operator_id, snapshot_pending=False)


def _pending_operator_resume_hint(project: Project, operator_id: str) -> str:
    rows = project.catalog.operations_using_operator(operator_id)

    if not rows:
        return (
            "No operation is recorded for this pending operator; "
            "the catalog may be inconsistent. Create a new translator."
        )

    if len(rows) > 1:
        operation_ids = tuple(str(row["operation_id"]) for row in rows)
        return (
            "More than one operation is recorded for this pending operator, "
            f"which violates the pending-operator invariant. "
            f"Recorded operations: {operation_ids}."
        )

    row = rows[0]
    operation_id = str(row["operation_id"])
    status = str(row["status"])

    if status != "incomplete":
        return (
            f"The only operation recorded for this pending operator is "
            f"{operation_id!r}, but its status is {status!r}. "
            "This operator is not reusable; create a new translator."
        )

    try:
        metadata = _load_operation_metadata(project, operation_id)
    except OperatorError:
        return (
            f"The pending operator belongs to incomplete operation {operation_id!r}, "
            "but its operation descriptor is missing. Create a new translator."
        )

    if not bool(metadata.get("resumable", False)):
        return (
            f"The pending operator belongs to incomplete operation {operation_id!r}, "
            "but that operation was not marked resumable. Create a new translator."
        )

    return f"Resume the interrupted operation with resume_translate(project, {operation_id!r})."


def _load_translator_for_resume(
    project: Project,
    operation_id: str,
    metadata: Mapping[str, Any],
) -> BaseTranslator:
    operator_id = str(metadata["operator_id"])
    snapshot_status = project.catalog.operator_snapshot_status(operator_id)
    if snapshot_status not in {"serialized", "pending"}:
        raise OperatorError(
            f"Operation {operation_id} cannot be resumed because operator {operator_id} "
            f"has snapshot_status={snapshot_status!r}."
        )
    if not bool(metadata.get("resumable", False)):
        raise OperatorError(
            f"Operation {operation_id} cannot be resumed because it was not "
            "marked resumable."
        )

    translator_cls = _translator_class_from_metadata(metadata)
    intermediate_dir = project.storage.operation_temp_dir_for_operator(operation_id)
    if not intermediate_dir.exists():
        raise OperatorError(
            f"Operation {operation_id} cannot be resumed because its translator "
            f"intermediate state directory is missing: {intermediate_dir}."
        )

    loaded = translator_cls.load_intermediate_state(
        intermediate_dir,
        operator_id=operator_id,
        mode=_validate_translation_mode(metadata["mode"]),
        route=_validate_run_route(metadata["route"]),
    )
    if loaded.operator_id is None:
        loaded.assign_operator_id(operator_id)
    elif loaded.operator_id != operator_id:
        raise OperatorError(
            f"Intermediate state for operation {operation_id} loaded operator_id="
            f"{loaded.operator_id!r}; expected {operator_id!r}."
        )
    return loaded


def _translator_class_from_metadata(
    metadata: Mapping[str, Any],
) -> type[BaseTranslator]:
    class_info = metadata.get("translator_class")
    if not isinstance(class_info, Mapping):
        raise OperatorError(
            "Operation descriptor is missing translator_class metadata."
        )
    module_name = class_info.get("module")
    qualname = class_info.get("qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise OperatorError(
            "Operation translator_class metadata must include module and qualname strings."
        )

    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)

    from text_analysis_lab.core.operator import BaseTranslator

    if not isinstance(obj, type) or not issubclass(obj, BaseTranslator):
        raise OperatorError(
            f"Operation translator_class {module_name}:{qualname} is not a BaseTranslator subclass."
        )
    return obj


def _serialize_pending_operator_snapshot(
    project: Project,
    translator: BaseTranslator,
    runtime: _Runtime,
) -> None:
    if not runtime.operator_snapshot_pending:
        return
    _save_operator(project, translator, runtime.operator_id)
    project.catalog.mark_operator_serialized(runtime.operator_id)
    runtime.operator_snapshot_pending = False


def _save_operator(
    project: Project, translator: BaseTranslator, operator_id: str
) -> None:
    translator.save_to_dir(
        project.storage.operator_dir(operator_id), operator_id=operator_id
    )


def _basis_ids(
    spec: OutputSpec,
    *,
    source_artifact_ids: Mapping[str, str],
    output_artifact_ids: Mapping[str, str],
) -> list[str]:
    basis_labels = cast(tuple[str, ...], spec.basis_labels)
    if not basis_labels:
        return list(source_artifact_ids.values())

    label_to_artifact_id = {**source_artifact_ids, **output_artifact_ids}
    return [label_to_artifact_id[label] for label in basis_labels]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_plan(
    *,
    translator: BaseTranslator,
    runtime: _Runtime,
    plan: _ExecutionPlan,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
    mode: TranslationMode,
    route: RunRoute,
    request: TranslationRequest,
    workers: int,
) -> None:
    if route == "sequential":
        _run_sequential(
            translator=translator,
            runtime=runtime,
            plan=plan,
            sources=sources,
            input_request=input_request,
            mode=mode,
            route=route,
            request=request,
        )
        return
    _run_parallel(
        translator=translator,
        runtime=runtime,
        plan=plan,
        sources=sources,
        input_request=input_request,
        mode=mode,
        route=route,
        request=request,
        workers=workers,
    )


def _run_sequential(
    *,
    translator: BaseTranslator,
    runtime: _Runtime,
    plan: _ExecutionPlan,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
    mode: TranslationMode,
    route: RunRoute,
    request: TranslationRequest,
) -> None:
    for unit in plan.iter_pending():
        plan.mark_running(unit.unit_index)
        try:
            inputs = _materialize_plan_unit(
                unit,
                sources=sources,
                input_request=input_request,
            )
            result = translator.translate_batch(inputs, mode=mode, request=request)

            handled = translator.handle_batch_result(
                result,
                batch_index=unit.unit_index,
                mode=mode,
                request=request,
            )
            _commit_outputs(runtime, handled)
            _save_intermediate_state(
                translator=translator,
                runtime=runtime,
                mode=mode,
                route=route,
            )
            plan.mark_complete(unit.unit_index)
        except Exception as exc:
            plan.mark_failed(unit.unit_index, exc)
            raise


def _translate_batch_worker(
    translator: BaseTranslator,
    inputs: Mapping[str, InputBatch],
    mode: TranslationMode,
    request: TranslationRequest,
) -> BatchResult:
    return translator.translate_batch(inputs, mode=mode, request=request)


def _open_dask_client(*, workers: int) -> tuple[Any, Any, Any]:
    try:
        from dask.distributed import Client, LocalCluster, as_completed
    except ImportError as exc:
        raise OperatorError(
            "Parallel translation requires the optional dependency "
            "dask[distributed]. Install TeAL with its parallel extra or add "
            "dask[distributed] to the active environment."
        ) from exc

    cluster = LocalCluster(
        n_workers=int(workers),
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)
    return client, cluster, as_completed


def _run_parallel(
    *,
    translator: BaseTranslator,
    runtime: _Runtime,
    plan: _ExecutionPlan,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
    mode: TranslationMode,
    route: RunRoute,
    request: TranslationRequest,
    workers: int,
) -> None:
    translate_worker = translator.make_translate_worker(mode=mode, request=request)

    for unit_index, result in _parallel_results(
        translate_worker=translate_worker,
        plan=plan,
        sources=sources,
        input_request=input_request,
        mode=mode,
        request=request,
        workers=workers,
    ):
        _handle_and_commit_unit(
            translator=translator,
            runtime=runtime,
            plan=plan,
            unit_index=unit_index,
            result=result,
            mode=mode,
            route=route,
            request=request,
        )


def _parallel_results(
    *,
    translate_worker: BaseTranslator,
    plan: _ExecutionPlan,
    sources: Mapping[str, BaseArtifact],
    input_request: InputRequest,
    mode: TranslationMode,
    request: TranslationRequest,
    workers: int,
) -> Iterable[tuple[int, BatchResult]]:
    limit = request.max_outstanding_units or workers
    units = iter(plan.iter_pending())
    client, cluster, as_completed = _open_dask_client(workers=workers)
    pending: dict[Any, int] = {}
    ready: dict[int, Any] = {}
    completed = as_completed()

    try:
        worker_future = client.scatter(translate_worker, broadcast=True)

        def submit_next() -> bool:
            try:
                unit = next(units)
            except StopIteration:
                return False

            plan.mark_running(unit.unit_index)
            try:
                inputs = _materialize_plan_unit(
                    unit,
                    sources=sources,
                    input_request=input_request,
                )
                future = client.submit(
                    _translate_batch_worker,
                    worker_future,
                    inputs,
                    mode,
                    request,
                    pure=False,
                )
            except Exception as exc:
                plan.mark_failed(unit.unit_index, exc)
                raise
            pending[future] = unit.unit_index
            completed.add(future)
            return True

        def fill_window() -> None:
            while len(pending) + len(ready) < limit and submit_next():
                pass

        fill_window()
        next_unit_index = min(pending.values(), default=None)

        while pending or ready:
            while next_unit_index is not None and next_unit_index in ready:
                future = ready.pop(next_unit_index)
                try:
                    result = future.result()
                except Exception as exc:
                    plan.mark_failed(next_unit_index, exc)
                    raise
                yield next_unit_index, result
                fill_window()
                waiting = [*pending.values(), *ready]
                next_unit_index = min(waiting, default=None)

            if not pending:
                if ready:
                    raise OperatorError(
                        "Parallel results could not be restored to durable plan order."
                    )
                break

            future = next(completed)
            ready[pending.pop(future)] = future
    finally:
        try:
            if pending:
                client.cancel(list(pending))
        finally:
            client.close()
            cluster.close()


def _handle_and_commit_unit(
    *,
    translator: BaseTranslator,
    runtime: _Runtime,
    plan: _ExecutionPlan,
    unit_index: int,
    result: BatchResult,
    mode: TranslationMode,
    route: RunRoute,
    request: TranslationRequest,
) -> None:
    try:
        handled = translator.handle_batch_result(
            result,
            batch_index=unit_index,
            mode=mode,
            request=request,
        )
        _commit_outputs(runtime, handled)
        _save_intermediate_state(
            translator=translator,
            runtime=runtime,
            mode=mode,
            route=route,
        )
        plan.mark_complete(unit_index)
    except Exception as exc:
        plan.mark_failed(unit_index, exc)
        raise


# ---------------------------------------------------------------------------
# Writing / intermediate state / completion
# ---------------------------------------------------------------------------


def _commit_outputs(
    runtime: _Runtime,
    outputs: OutputMap | None,
) -> None:
    outputs = dict(outputs or {})
    _validate_output_labels(runtime, outputs)
    for label in runtime.output_labels:
        if label in outputs:
            runtime.writers[label].write(outputs[label])


def _commit_final_outputs(runtime: _Runtime, outputs: OutputMap | None) -> None:
    outputs = dict(outputs or {})
    _validate_output_labels(runtime, outputs)
    for label in runtime.output_labels:
        if label in outputs:
            runtime.writers[label].write(outputs[label])


def _save_intermediate_state(
    *,
    translator: BaseTranslator,
    runtime: _Runtime,
    mode: TranslationMode,
    route: RunRoute,
) -> None:
    if not runtime.resumable:
        return

    def write_intermediate(target: Path) -> None:
        operator_dir = target / runtime.operator_temp_dir.name
        writers_dir = target / runtime.writers_temp_dir.name
        operator_dir.mkdir(parents=True, exist_ok=True)
        writers_dir.mkdir(parents=True, exist_ok=True)
        translator.save_intermediate_state(
            operator_dir,
            operator_id=runtime.operator_id,
            mode=mode,
            route=route,
        )
        _write_writer_checkpoints(writers_dir, runtime)

    _replace_directory_with_staging(runtime.operation_temp_dir, write_intermediate)


def _writer_checkpoint_path(writers_dir: Path, label: str) -> Path:
    return Path(writers_dir) / f"{label}.json"


def _write_writer_checkpoints(writers_dir: Path, runtime: _Runtime) -> None:
    writers_dir.mkdir(parents=True, exist_ok=True)
    for label in runtime.output_labels:
        _writer_checkpoint_path(writers_dir, label).write_text(
            json.dumps(runtime.writers[label].resume_state(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _load_writer_checkpoint(writers_dir: Path, label: str) -> Mapping[str, Any]:
    path = _writer_checkpoint_path(writers_dir, label)
    if not path.exists():
        raise OperatorError(
            f"Missing writer checkpoint for output label {label!r}: {path}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise OperatorError(
            f"Writer checkpoint for output label {label!r} must contain a JSON object."
        )
    return payload


def _replace_directory_with_staging(
    target: Path,
    writer: Callable[[Path], None],
) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.__staging__")
    previous = target.with_name(f"{target.name}.__previous__")

    # Recover deterministically from an interrupted prior checkpoint swap.
    # ``target`` is authoritative whenever it exists. If only ``previous``
    # survives, it is the last complete checkpoint and must be restored before
    # a new staging generation is created.
    if target.exists():
        if previous.exists():
            _remove_checkpoint_tree(previous)
    elif previous.exists():
        _rename_checkpoint_path(previous, target)

    if staging.exists():
        _remove_checkpoint_tree(staging)
    staging.mkdir(parents=True, exist_ok=False)

    try:
        writer(staging)
    except Exception:
        try:
            _remove_checkpoint_tree(staging)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        if previous.exists():
            _remove_checkpoint_tree(previous)
        if target.exists():
            _rename_checkpoint_path(target, previous)
    except Exception:
        # The old target is still authoritative if it could not be moved. The
        # uncommitted staging generation can be discarded safely.
        try:
            _remove_checkpoint_tree(staging)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        _rename_checkpoint_path(staging, target)
    except Exception as exc:
        if previous.exists() and not target.exists():
            try:
                _rename_checkpoint_path(previous, target)
            except OSError as restore_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "TeAL could not restore the prior checkpoint after the new "
                        f"checkpoint promotion failed: {restore_exc!r}. The prior "
                        f"checkpoint remains at {previous}."
                    )
        if staging.exists():
            try:
                _remove_checkpoint_tree(staging)
            except OSError as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"TeAL could not remove failed checkpoint staging {staging}: "
                        f"{cleanup_exc!r}."
                    )
        raise

    if previous.exists():
        try:
            _remove_checkpoint_tree(previous)
        except PermissionError as exc:
            # The new target is already durably promoted. A stale ``previous``
            # directory is safe because the next checkpoint begins by removing
            # it when ``target`` exists; do not turn a successful commit into a
            # failed operation solely because Windows still has the old tree open.
            warnings.warn(
                f"Checkpoint committed at {target}, but TeAL could not yet remove "
                f"the prior checkpoint directory {previous}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _retry_checkpoint_fs(action: Callable[[], Any]) -> Any:
    """Retry short-lived Windows permission/sharing failures with bounded backoff."""
    last_error: PermissionError | None = None
    for attempt in range(_CHECKPOINT_FS_ATTEMPTS):
        try:
            return action()
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 >= _CHECKPOINT_FS_ATTEMPTS:
                raise
            time.sleep(_CHECKPOINT_FS_BACKOFF_SECONDS * (2**attempt))
    if last_error is not None:  # pragma: no cover - loop always returns or raises
        raise last_error
    return None


def _rename_checkpoint_path(source: Path, destination: Path) -> None:
    _retry_checkpoint_fs(lambda: source.rename(destination))


def _remove_checkpoint_tree(path: Path) -> None:
    if not path.exists():
        return
    _retry_checkpoint_fs(lambda: shutil.rmtree(path))


def _validate_output_labels(runtime: _Runtime, outputs: Mapping[str, Any]) -> None:
    known = set(runtime.output_specs)
    for label in outputs:
        if label not in known:
            raise OperatorError(
                f"Translator returned unknown output label {label!r}. "
                f"Known labels: {tuple(runtime.output_specs)}."
            )


def _validate_runtime_output_lineage(
    project: Project, runtime: _Runtime, label: str
) -> None:
    """Validate one finalized output's key schema against its declared basis."""
    artifact_id = runtime.output_artifact_ids[label]
    artifact = project.get_artifact(artifact_id)
    basis_rows = project.catalog.artifact_basis(artifact_id)
    basis_keys = [
        project.get_artifact(str(row["basis_artifact_id"])).primary_key
        for row in basis_rows
    ]
    validate_primary_key_relationship(
        basis_keys=basis_keys,
        output_key=artifact.primary_key,
        lineage_mode=runtime.output_specs[label].lineage_mode,
    )


def _complete_runtime(project: Project, runtime: _Runtime) -> None:
    # Writers perform artifact-wide storage/key validation first.  Lineage is a
    # project-level integrity check, so reuse the existing lineage validator here
    # before the catalog grants any output its authoritative complete status.
    for label in runtime.output_labels:
        runtime.writers[label].finalize()
    for label in runtime.output_labels:
        _validate_runtime_output_lineage(project, runtime, label)
    for label in runtime.output_labels:
        project.catalog.mark_artifact_complete(runtime.output_artifact_ids[label])
    _update_operation_metadata_status(runtime, "complete")
    project.catalog.mark_operation_complete(runtime.operation_id)
    _cleanup_completed_operation_temp(runtime)


def _cleanup_completed_operation_temp(runtime: _Runtime) -> None:
    """Remove resumable working state after an operation is durably complete."""
    paths = (
        runtime.operation_temp_dir,
        runtime.operation_temp_dir.with_name(
            f"{runtime.operation_temp_dir.name}.__staging__"
        ),
        runtime.operation_temp_dir.with_name(
            f"{runtime.operation_temp_dir.name}.__previous__"
        ),
    )
    for path in paths:
        try:
            _remove_checkpoint_tree(path)
        except OSError as exc:
            # Completion is already authoritative in the catalog. A transient
            # Windows handle must not convert committed outputs into a failed
            # operation, but surface the stale working tree loudly.
            warnings.warn(
                f"Operation {runtime.operation_id} completed, but TeAL could not "
                f"remove checkpoint state at {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _fail_runtime(project: Project, runtime: _Runtime, error: Exception) -> None:
    for label, writer in runtime.writers.items():
        try:
            writer.mark_failed(error)
        finally:
            project.catalog.mark_artifact_failed(runtime.output_artifact_ids[label])
    project.catalog.mark_operation_failed(runtime.operation_id, error)
    if runtime.operator_snapshot_pending and not runtime.resumable:
        project.catalog.mark_operator_snapshot_failed(runtime.operator_id)
    _update_operation_metadata_status(runtime, "failed", error=error)


def _load_output_artifacts(
    project: Project, runtime: _Runtime
) -> Mapping[str, BaseArtifact]:
    return {
        label: project.get_artifact(artifact_id)
        for label, artifact_id in runtime.output_artifact_ids.items()
    }


def _open_writers_for_resume(
    project: Project,
    output_artifact_ids: Mapping[str, str],
    *,
    writers_temp_dir: Path,
) -> Mapping[str, ArtifactWriter]:
    writers: dict[str, ArtifactWriter] = {}
    for label, artifact_id in output_artifact_ids.items():
        writers[label] = ArtifactWriter.resume(
            _load_writer_checkpoint(writers_temp_dir, label),
            artifact_dir=project.storage.artifact_dir(artifact_id),
        )
    return writers


# ---------------------------------------------------------------------------
# Operation metadata serialization
# ---------------------------------------------------------------------------


def _update_operation_metadata_status(
    runtime: _Runtime,
    status: str,
    *,
    error: BaseException | str | None = None,
) -> None:
    path = runtime.operation_dir / "operation.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = str(status)
    payload["operator_snapshot_pending"] = runtime.operator_snapshot_pending
    if error is not None:
        payload["error"] = str(error)
    _write_json(path, payload)


def _save_operation_metadata(
    *,
    runtime: _Runtime,
    translator: BaseTranslator,
    mode: TranslationMode,
    route: RunRoute,
    effective_workers: int,
    request: TranslationRequest,
    serialized_operation_params: Mapping[str, Any],
    input_request: InputRequest,
    sources: Mapping[str, BaseArtifact],
    output_specs: Mapping[str, OutputSpec],
    output_artifact_ids: Mapping[str, str],
    alias_plan: AliasPlan | None,
) -> None:
    payload = {
        "schema_version": 1,
        "operation_id": runtime.operation_id,
        "operation_type": translator.operation_type,
        "operator_id": runtime.operator_id,
        "operator_snapshot_pending": runtime.operator_snapshot_pending,
        "translator_class": {
            "module": translator.__class__.__module__,
            "qualname": translator.__class__.__qualname__,
        },
        "mode": mode,
        "route": route,
        "effective_workers": effective_workers,
        "status": "incomplete",
        "resumable": runtime.resumable,
        "request": {
            "workers": request.workers,
            "batch_size": request.batch_size,
            "max_outstanding_units": request.max_outstanding_units,
            "params": dict(serialized_operation_params),
        },
        "input_request": _input_request_to_dict(input_request),
        "sources": {label: artifact.artifact_id for label, artifact in sources.items()},
        "output_specs": _output_specs_to_dict(output_specs),
        "output_artifact_ids": dict(output_artifact_ids),
        "alias_plan": None if alias_plan is None else alias_plan.to_dict(),
    }
    _write_json(runtime.operation_dir / "operation.json", payload)


def _load_operation_metadata(project: Project, operation_id: str) -> Mapping[str, Any]:
    path = project.storage.operation_descriptor_path(str(operation_id))
    if not path.exists():
        raise OperatorError(f"Operation {operation_id} has no operation descriptor.")
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_operation_params(
    translator: BaseTranslator,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    serialized = translator.serialize_operation_params(params)
    if not isinstance(serialized, Mapping):
        raise OperatorError("serialize_operation_params() must return a mapping.")
    payload = dict(serialized)
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise OperatorError(
            "serialize_operation_params() must return a JSON-serializable mapping."
        ) from exc
    return payload


def _input_request_to_dict(input_request: InputRequest) -> dict[str, Any]:
    return {
        label: _source_request_to_dict(request)
        for label, request in input_request.items()
    }


def _input_request_from_dict(data: Mapping[str, Any]) -> dict[str, SourceRequest]:
    return {
        str(label): _source_request_from_dict(value) for label, value in data.items()
    }


def _output_specs_to_dict(
    output_specs: Mapping[str, OutputSpec],
) -> dict[str, Any]:
    return {label: spec.to_dict() for label, spec in output_specs.items()}


def _output_specs_from_dict(data: Mapping[str, Any]) -> dict[str, OutputSpec]:
    from text_analysis_lab.core.operator import OutputSpec

    specs: dict[str, OutputSpec] = {}
    for label, raw in data.items():
        if not isinstance(raw, Mapping):
            raise OperatorError(f"Stored output spec {label!r} must be a mapping.")
        serializer_ref = raw.get("data_serializer")
        if serializer_ref is not None and not isinstance(serializer_ref, Mapping):
            raise OperatorError(
                f"Stored output spec {label!r} has an invalid data serializer reference."
            )
        specs[str(label)] = OutputSpec(
            artifact_type=ArtifactType(raw["artifact_type"]),
            lineage_mode=validate_lineage_mode(raw.get("lineage_mode", "new_key")),
            basis_labels=tuple(raw.get("basis_labels", ())),
            data_serializer=(
                None
                if serializer_ref is None
                else _resolve_callable_ref(serializer_ref)
            ),
        )
    return specs


def _resolve_callable_ref(ref: Mapping[str, Any]) -> Callable[..., Any]:
    module_name = ref.get("module")
    qualname = ref.get("qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise OperatorError("Callable reference requires module and qualname strings.")
    try:
        value: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except Exception as exc:
        raise OperatorError(
            f"Could not resolve callable reference {module_name}:{qualname}."
        ) from exc
    if not callable(value):
        raise OperatorError(
            f"Callable reference {module_name}:{qualname} did not resolve to a callable."
        )
    return cast(Callable[..., Any], value)


def _source_request_to_dict(request: SourceRequest) -> dict[str, Any]:
    artifact_types = _accepted_artifact_types("source", request)
    return {
        "artifact_type": [artifact_type.value for artifact_type in artifact_types],
        "mode": request.mode,
        "columns": _column_request_to_dict(request.columns),
        "batch_size": request.batch_size,
        "form": request.form,
        "metadata_mode": request.metadata_mode,
        "include_position": request.include_position,
    }


def _source_request_from_dict(data: Mapping[str, Any]) -> SourceRequest:
    raw_batch_size = data.get("batch_size")
    return SourceRequest(
        artifact_type=tuple(data["artifact_type"]),
        mode=data["mode"],
        columns=_column_request_from_dict(data["columns"]),
        batch_size=cast(int | None, raw_batch_size),
        form=data["form"],
        metadata_mode=data["metadata_mode"],
        include_position=bool(data["include_position"]),
    )


def _column_request_to_dict(columns: ColumnRequest) -> dict[str, Any]:
    return {
        "keys": _jsonable_column_select(columns.keys),
        "metadata": _jsonable_column_select(columns.metadata),
        "data": _jsonable_column_select(columns.data),
    }


def _column_request_from_dict(data: Mapping[str, Any]) -> ColumnRequest:
    return ColumnRequest(
        keys=data["keys"],
        metadata=data["metadata"],
        data=data["data"],
    )


def _jsonable_column_select(value: ColumnSelect) -> bool | str | list[str]:
    if isinstance(value, bool | str):
        return value
    return list(value)
