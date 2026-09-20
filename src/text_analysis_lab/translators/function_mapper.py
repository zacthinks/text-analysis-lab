"""Batch-oriented preserved-key functional mapping for TeAL artifacts.

``FunctionMapper`` consumes one source artifact, exposes requested representation
data and/or metadata as separate DataFrame components, and lets a user callable
return new data and/or metadata for exactly those rows. TeAL retains sole authority
over keys and row alignment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    RunRoute,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import (
    DEFAULT_OUTPUT_LABEL,
    DEFAULT_SOURCE_LABEL,
    ColumnSelect,
    MetadataMode,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


MapperFunction = Callable[[Any], Any]
_SUPPORTED_TYPES = ("table", "jsonl", "sparse_matrix", "dense_matrix")
_FUNCTION_ASSET = "function.pkl"


class FunctionMapper(BaseTranslator):
    """Apply one row-preserving callable to source data/metadata batches.

    The callable receives a mapping with optional ``"data"`` and ``"metadata"``
    DataFrames. It must return a mapping with at least one of those same component
    names. The callable never receives or returns keys; TeAL writes the original
    source keys unchanged and requires exactly one returned row per input row.
    """

    operation_type = "translate"

    def __init__(
        self,
        function: MapperFunction | None,
        *,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if function is not None and not callable(function):
            raise TypeError("FunctionMapper function must be callable.")
        self.function: MapperFunction | None = function
        self._source_type: str | None = None
        self._data_columns: tuple[str, ...] | None = None
        self._metadata_columns: tuple[str, ...] | None = None
        self._metadata_mode: MetadataMode = "none"

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        _single_source(sources, name="FunctionMapper")
        return OutputSpec(
            artifact_type="table",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        unknown = sorted(
            set(params) - {"data_columns", "metadata_columns", "metadata_mode"}
        )
        if unknown:
            raise OperatorError(f"Unknown FunctionMapper parameter(s): {unknown}.")
        source = _single_source(sources, name="FunctionMapper")
        if source.artifact_type.value not in _SUPPORTED_TYPES:
            raise OperatorError(
                "FunctionMapper requires a table, jsonl, sparse_matrix, or "
                f"dense_matrix source; got {source.artifact_type.value!r}."
            )
        data_selection = _normalize_column_select(
            params.get("data_columns", True), name="data_columns"
        )
        metadata_selection = _normalize_column_select(
            params.get("metadata_columns", False), name="metadata_columns"
        )
        metadata_mode = _normalize_metadata_mode(params.get("metadata_mode", "none"))
        if metadata_selection is not False and metadata_mode == "none":
            raise OperatorError(
                "FunctionMapper metadata_columns requires metadata_mode='local' or 'full'."
            )
        if data_selection is False and metadata_selection is False:
            raise OperatorError(
                "FunctionMapper requires at least one requested data or metadata column."
            )
        return {
            "data_columns": data_selection,
            "metadata_columns": metadata_selection,
            "metadata_mode": metadata_mode,
        }

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _single_source(sources, name="FunctionMapper")
        self._source_type = source.artifact_type.value
        data_selection = cast(ColumnSelect, request.params["data_columns"])
        metadata_selection = cast(ColumnSelect, request.params["metadata_columns"])
        metadata_mode = cast(MetadataMode, request.params["metadata_mode"])
        query_info = source.query_columns(metadata_mode=metadata_mode)
        # Matrix representation columns are stored outside the relational DuckDB
        # view, so query_columns() correctly reports no SQL ``data`` namespace
        # for them. Resolve matrix features from the artifact's representation
        # schema instead; artifact.query(..., form="table") will materialize
        # exactly those requested matrix columns batchwise.
        if source.artifact_type.value in {"sparse_matrix", "dense_matrix"}:
            self._data_columns = tuple(
                _resolved_matrix_data_columns(
                    source.get_data_columns(),
                    selection=data_selection,
                    parameter_name="data_columns",
                )
            )
        else:
            self._data_columns = tuple(
                _resolved_namespace_columns(
                    query_info,
                    namespace="data",
                    selection=data_selection,
                    parameter_name="data_columns",
                )
            )
        self._metadata_columns = tuple(
            _resolved_namespace_columns(
                query_info,
                namespace="metadata",
                selection=metadata_selection,
                parameter_name="metadata_columns",
            )
        )
        self._metadata_mode = metadata_mode
        if not self._data_columns and not self._metadata_columns:
            raise OperatorError(
                "FunctionMapper selection resolved to no data or metadata columns."
            )
        return SourceRequest(
            artifact_type=_SUPPORTED_TYPES,
            mode="batches",
            columns=ColumnRequest(
                keys=True,
                data=data_selection,
                metadata=metadata_selection,
            ),
            batch_size=request.batch_size or 10_000,
            # FunctionMapper's public callable contract is DataFrame-based even
            # for matrix sources. Matrix artifacts preserve sparsity in pandas
            # sparse DataFrames when materialized in table form.
            form="table",
            metadata_mode=metadata_mode,
            include_position=False,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported FunctionMapper mode {mode!r}.")
        source_batch = _single_input(inputs, name="FunctionMapper")
        keys, callable_packet = _mapper_input_packet(
            source_batch,
            data_columns=self._data_columns or (),
            metadata_columns=self._metadata_columns or (),
        )
        raw = self._require_function()(callable_packet)
        payload = _normalize_mapper_packet(
            raw,
            expected_rows=len(keys),
            primary_key=source_batch.primary_key,
        )
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, **payload}})

    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> FunctionMapper:
        _ = request
        if mode != "translate":
            raise OperatorError("FunctionMapper workers support translate mode only.")
        worker = FunctionMapper(
            cloudpickle.loads(cloudpickle.dumps(self._require_function()))
        )
        worker._source_type = self._source_type
        worker._data_columns = self._data_columns
        worker._metadata_columns = self._metadata_columns
        worker._metadata_mode = self._metadata_mode
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> FunctionMapper:
        _ = state
        return cls(None)

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / _FUNCTION_ASSET
        payload = cloudpickle.dumps(self._require_function())
        path.write_bytes(payload)
        # Execute from the frozen callable snapshot as well as persisting it, so
        # later mutations to captured notebook/session state cannot affect the
        # operator after freeze.
        function = cloudpickle.loads(payload)
        if not callable(function):  # pragma: no cover - cloudpickle contract guard
            raise OperatorError(
                "Frozen FunctionMapper callable did not deserialize to a callable."
            )
        self.function = cast(MapperFunction, function)
        return {"function_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("function_file")
        if not isinstance(filename, str) or not filename:
            raise OperatorError(
                "FunctionMapper operator is missing its callable asset."
            )
        path = assets_dir / filename
        if not path.exists():
            raise OperatorError(f"Missing FunctionMapper callable asset: {path}.")
        function = cloudpickle.loads(path.read_bytes())
        if not callable(function):
            raise OperatorError(
                "FunctionMapper asset did not deserialize to a callable."
            )
        self.function = cast(MapperFunction, function)

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        assets = dict(self.save_assets(intermediate_dir))
        state = {
            "operator_id": operator_id,
            "source_type": self._source_type,
            "data_columns": None
            if self._data_columns is None
            else list(self._data_columns),
            "metadata_columns": (
                None if self._metadata_columns is None else list(self._metadata_columns)
            ),
            "metadata_mode": self._metadata_mode,
            "assets": assets,
        }
        (intermediate_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> FunctionMapper:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls(None, operator_id=operator_id)
        raw_source_type = state.get("source_type")
        obj._source_type = None if raw_source_type is None else str(raw_source_type)
        raw_columns = state.get("data_columns")
        if isinstance(raw_columns, Sequence) and not isinstance(
            raw_columns, (str, bytes)
        ):
            obj._data_columns = tuple(str(value) for value in raw_columns)
        raw_metadata_columns = state.get("metadata_columns")
        if isinstance(raw_metadata_columns, Sequence) and not isinstance(
            raw_metadata_columns, (str, bytes)
        ):
            obj._metadata_columns = tuple(str(value) for value in raw_metadata_columns)
        obj._metadata_mode = _normalize_metadata_mode(
            state.get("metadata_mode", "none")
        )
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("FunctionMapper intermediate assets must be a mapping.")
        obj.load_assets(intermediate_dir, assets)
        return obj

    def _require_function(self) -> MapperFunction:
        if self.function is None:
            raise OperatorError("FunctionMapper callable is unavailable.")
        return self.function


def _single_source(sources: Mapping[str, Any], *, name: str):
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(f"{name} requires exactly one source under 'source'.")
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch], *, name: str) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(f"{name} expected exactly one input under 'source'.")
    return inputs[DEFAULT_SOURCE_LABEL]


def _normalize_column_select(value: Any, *, name: str) -> ColumnSelect:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} cannot be an empty string.")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [str(item) for item in value]
        if any(not item for item in values):
            raise ValueError(f"{name} must contain non-empty strings.")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} cannot contain duplicates.")
        return False if not values else values
    raise TypeError(f"{name} must be a bool, string, or sequence of strings.")


def _normalize_metadata_mode(value: Any) -> MetadataMode:
    mode = str(value)
    if mode not in {"none", "local", "full"}:
        raise OperatorError("metadata_mode must be one of 'none', 'local', or 'full'.")
    return cast(MetadataMode, mode)


def _resolved_matrix_data_columns(
    available_columns: Sequence[str],
    *,
    selection: ColumnSelect,
    parameter_name: str,
) -> list[str]:
    available = [str(value) for value in available_columns]
    if selection is False:
        return []
    if selection is True:
        return available
    requested = (
        [selection] if isinstance(selection, str) else [str(v) for v in selection]
    )
    missing = [name for name in requested if name not in available]
    if missing:
        raise ArtifactError(
            f"FunctionMapper {parameter_name} requested unknown matrix data "
            f"column(s) {missing}; "
            f"available columns are {available}."
        )
    # _normalize_column_select() already rejects duplicate requests, but keep
    # the resolver independently deterministic.
    return list(dict.fromkeys(str(name) for name in requested))


def _resolved_namespace_columns(
    query_info: Mapping[str, Any],
    *,
    namespace: str,
    selection: ColumnSelect,
    parameter_name: str,
) -> list[str]:
    columns = [
        dict(item)
        for item in query_info.get("columns", [])
        if isinstance(item, Mapping) and item.get("namespace") == namespace
    ]
    if selection is False:
        return []
    if selection is True:
        return [str(item["output_name"]) for item in columns]
    requested = (
        [selection] if isinstance(selection, str) else [str(v) for v in selection]
    )
    out: list[str] = []
    for name in requested:
        matches = [
            item
            for item in columns
            if name
            in {
                str(item.get("qualified_name")),
                str(item.get("output_name")),
                str(item.get("base_name")),
            }
        ]
        # Prefer exact qualified/output matches before base-name resolution.
        exact = [
            item
            for item in matches
            if name in {str(item.get("qualified_name")), str(item.get("output_name"))}
        ]
        if len(exact) == 1:
            resolved = str(exact[0]["output_name"])
        elif len(matches) == 1:
            resolved = str(matches[0]["output_name"])
        elif len(matches) > 1:
            choices = [str(item.get("qualified_name")) for item in matches]
            raise ArtifactError(
                f"FunctionMapper {parameter_name} name {name!r} is ambiguous; "
                f"use one of {choices}."
            )
        else:
            available = [str(item.get("output_name")) for item in columns]
            raise ArtifactError(
                f"FunctionMapper requested unknown {namespace} column {name!r}; "
                f"available columns are {available}."
            )
        if resolved not in out:
            out.append(resolved)
    return out


def _mapper_input_packet(
    packet: InputBatch,
    *,
    data_columns: Sequence[str],
    metadata_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not isinstance(packet.data, pd.DataFrame):
        raise ArtifactError(
            "FunctionMapper input must materialize as a DataFrame packet."
        )
    frame = packet.data.reset_index(drop=True)
    key_columns = [str(value) for value in packet.primary_key]
    missing_keys = [column for column in key_columns if column not in frame.columns]
    if missing_keys:
        raise ArtifactError(
            f"FunctionMapper packet is missing key column(s) {missing_keys}."
        )
    keys = frame.loc[:, key_columns].copy().reset_index(drop=True)
    out: dict[str, pd.DataFrame] = {}
    if data_columns:
        missing = [column for column in data_columns if column not in frame.columns]
        if missing:
            raise ArtifactError(
                f"FunctionMapper packet is missing data column(s) {missing}."
            )
        out["data"] = frame.loc[:, list(data_columns)].copy().reset_index(drop=True)
    if metadata_columns:
        missing = [column for column in metadata_columns if column not in frame.columns]
        if missing:
            raise ArtifactError(
                f"FunctionMapper packet is missing metadata column(s) {missing}."
            )
        out["metadata"] = (
            frame.loc[:, list(metadata_columns)].copy().reset_index(drop=True)
        )
    return keys, out


def _normalize_mapper_packet(
    value: Any,
    *,
    expected_rows: int,
    primary_key: Sequence[str],
) -> dict[str, pd.DataFrame]:
    if not isinstance(value, Mapping):
        raise ArtifactError(
            "FunctionMapper callable must return a mapping containing 'data' and/or 'metadata'."
        )
    if "keys" in value:
        raise ArtifactError(
            "FunctionMapper preserves source keys; its callable may not return 'keys'."
        )
    out: dict[str, pd.DataFrame] = {}
    for component in ("data", "metadata"):
        if component not in value or value[component] is None:
            continue
        frame = value[component]
        if not isinstance(frame, pd.DataFrame):
            raise ArtifactError(
                f"FunctionMapper {component!r} output must be a pandas DataFrame."
            )
        frame = frame.copy().reset_index(drop=True)
        if len(frame) != int(expected_rows):
            raise ArtifactError(
                f"FunctionMapper function returned {len(frame)} {component} rows for an "
                f"input batch with {expected_rows} rows."
            )
        if frame.shape[1] == 0:
            raise ArtifactError(
                f"FunctionMapper {component!r} output must contain at least one column."
            )
        frame.columns = [str(column) for column in frame.columns]
        _validate_output_columns(frame, primary_key=primary_key, component=component)
        out[component] = frame
    if not out:
        raise ArtifactError(
            "FunctionMapper callable must return at least one non-empty 'data' or 'metadata' DataFrame."
        )
    return out


def _validate_output_columns(
    frame: pd.DataFrame,
    *,
    primary_key: Sequence[str],
    component: str,
) -> None:
    columns = [str(column) for column in frame.columns]
    if any(not column for column in columns):
        raise ArtifactError(
            f"FunctionMapper {component} output column names must be non-empty strings."
        )
    if len(set(columns)) != len(columns):
        raise ArtifactError(
            f"FunctionMapper {component} output column names must be unique."
        )
    reserved = set(primary_key).union({"_position", "_batch", "_row_offset"})
    overlap = sorted(set(columns).intersection(reserved))
    if overlap:
        raise ArtifactError(
            f"FunctionMapper output {component} columns collide with TeAL key/structural "
            f"columns: {overlap}."
        )
