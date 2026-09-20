"""Functional subset operations backed by the TeAL translation runner.

The public operation is ``Project.subset(...)``. Internally, a functional subset
is a simple translation: materialize one source packet, call a user function to
produce a boolean mask, and write the selected source keys to a keys-only child
artifact that inherits representation data through preserved-key lineage.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, get_args

import cloudpickle
import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError, QueryError
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
    QueryForm,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


SubsetFunction = Callable[[Any], Iterable[bool]]
FunctionSpec = SubsetFunction | tuple[str | Path, str]

QUERY_FORMS = get_args(QueryForm)
_FUNCTION_ASSET = "function.pkl"


def subset(
    project: Project,
    source: BaseArtifact | str,
    function: FunctionSpec,
    *,
    key_columns: ColumnSelect = True,
    data_columns: ColumnSelect = True,
    metadata_columns: ColumnSelect = False,
    metadata_mode: MetadataMode = "none",
    form: QueryForm = "table",
    iter_batches: bool = True,
    batch_size: int | None = None,
    include_position: bool = True,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    workers: int = 1,
    memo: str | None = None,
    alias: str | None = None,
    overwrite: bool = False,
) -> Mapping[str, BaseArtifact]:
    """Create a keys-only subset artifact using a boolean-mask function.

    ``function`` receives each materialized source packet in the requested
    query ``form`` and must return one boolean value per input row. The
    ``key_columns`` selection controls only which source keys are exposed to the
    callable; TeAL always carries the complete primary key internally so selected
    rows can be written to the keys-only output artifact. The output artifact has
    the same artifact type as ``source`` and uses preserved-key lineage, so
    representation data is inherited from the source.
    """
    translator = FunctionSubsetTranslator(function=function)
    return project.translate(
        translator,
        source,
        workers=workers,
        batch_size=batch_size,
        memo=memo,
        alias=alias,
        overwrite=overwrite,
        key_columns=key_columns,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode=metadata_mode,
        form=form,
        iter_batches=iter_batches,
        include_position=include_position,
        output_label=output_label,
    )


class FunctionSubsetTranslator(BaseTranslator):
    """Translator-backed functional subsetter."""

    operation_type = "subset"

    def __init__(
        self,
        function: FunctionSpec | None = None,
        *,
        function_origin: Mapping[str, str] | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if function is None and function_origin is None:
            raise ValueError("FunctionSubsetTranslator requires a callable.")
        self.function: SubsetFunction | None = None
        self.function_origin: dict[str, str] = dict(function_origin or {})
        if function is not None:
            prepared_origin, self.function = _prepare_function(function)
            if function_origin is None:
                self.function_origin = prepared_origin

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        source = _source_artifact(sources)
        output_label = str(request.params["output_label"])
        return {
            output_label: OutputSpec(
                artifact_type=source.artifact_type,
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
        }

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        """Subset execution is stateless and safe to resume by planned batch."""
        return mode == "translate" and route in {"sequential", "parallel"}

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        expected = {
            "key_columns",
            "data_columns",
            "metadata_columns",
            "metadata_mode",
            "form",
            "iter_batches",
            "include_position",
            "output_label",
        }
        unknown = sorted(set(params) - expected)
        if unknown:
            raise OperatorError(f"Unknown subset operation parameter(s): {unknown}.")

        source = _source_artifact(sources)
        column_info = _source_column_info(source)
        key_columns = _validate_column_select(
            cast(ColumnSelect, params.get("key_columns", True)),
            name="key_columns",
            namespace="key",
            info=column_info,
        )
        data_columns = _validate_column_select(
            cast(ColumnSelect, params.get("data_columns", True)),
            name="data_columns",
            namespace="data",
            info=column_info,
        )
        metadata_columns = _validate_column_select(
            cast(ColumnSelect, params.get("metadata_columns", False)),
            name="metadata_columns",
            namespace="metadata",
            info=column_info,
        )
        return {
            "key_columns": _json_column_select(key_columns),
            "data_columns": _json_column_select(data_columns),
            "metadata_columns": _json_column_select(metadata_columns),
            "metadata_mode": _validate_metadata_mode(
                cast(MetadataMode, params.get("metadata_mode", "none"))
            ),
            "form": _validate_form(str(params.get("form", "table"))),
            "iter_batches": _validate_bool(
                params.get("iter_batches", True), name="iter_batches"
            ),
            "include_position": _validate_bool(
                params.get("include_position", True), name="include_position"
            ),
            "output_label": _validate_label(
                params.get("output_label", DEFAULT_OUTPUT_LABEL),
                name="output_label",
            ),
        }

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _source_artifact(sources)
        params = request.params
        iter_batches = bool(params["iter_batches"])
        effective_batch_size = (
            (request.batch_size if request.batch_size is not None else 10_000)
            if iter_batches
            else None
        )
        return SourceRequest(
            artifact_type=source.artifact_type,
            mode="batches" if iter_batches else "full_artifact",
            columns=ColumnRequest(
                # Structural subset outputs always require the complete source
                # primary key, regardless of which key columns the user wants
                # exposed to the predicate callable. The callable-facing key
                # selection is applied inside ``translate_batch``.
                keys=True,
                data=cast(ColumnSelect, params["data_columns"]),
                metadata=cast(ColumnSelect, params["metadata_columns"]),
            ),
            batch_size=effective_batch_size,
            form=cast(QueryForm, params["form"]),
            metadata_mode=cast(MetadataMode, params["metadata_mode"]),
            include_position=bool(params["include_position"]),
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = mode
        if set(inputs) != {DEFAULT_SOURCE_LABEL}:
            raise OperatorError(
                f"FunctionSubsetTranslator expected one input label {DEFAULT_SOURCE_LABEL!r}; "
                f"got {tuple(inputs)}."
            )
        source_batch = inputs[DEFAULT_SOURCE_LABEL]
        packet = source_batch.data
        keys = _extract_key_frame(packet, primary_key=source_batch.primary_key)
        n_rows = _packet_length(packet)
        if len(keys) != n_rows:
            raise ArtifactError(
                f"Subset packet has {n_rows} rows but extracted key frame has "
                f"{len(keys)} rows."
            )

        predicate_packet = _predicate_packet(
            packet,
            primary_key=source_batch.primary_key,
            key_columns=cast(ColumnSelect, request.params["key_columns"]),
        )
        raw_mask = self.function(predicate_packet)
        mask = _normalize_mask(raw_mask, expected_len=n_rows)
        selected_keys = keys.loc[mask].reset_index(drop=True)
        output_label = str(request.params["output_label"])
        if selected_keys.empty:
            return BatchResult(outputs={})
        return BatchResult(outputs={output_label: {"keys": selected_keys}})

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
    ) -> FunctionSubsetTranslator:
        _ = mode, request
        function = cloudpickle.loads(cloudpickle.dumps(self._require_function()))
        return self.__class__(
            function,
            function_origin=self.function_origin,
        )

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
            "operator_id": str(operator_id),
            "function_origin": dict(self.function_origin),
            "assets": assets,
        }
        (intermediate_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> FunctionSubsetTranslator:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        if not isinstance(state, Mapping):
            raise OperatorError("Subset intermediate state must contain a JSON object.")
        origin = state.get("function_origin")
        if isinstance(origin, Mapping):
            obj = cls(
                None,
                function_origin={str(k): str(v) for k, v in origin.items()},
            )
            assets = state.get("assets", {})
            if not isinstance(assets, Mapping):
                raise OperatorError("Subset intermediate assets must be a mapping.")
            obj.load_assets(intermediate_dir, assets)
        else:
            # Backward compatibility for Round <=25 resume checkpoints.
            legacy_ref = state.get("function_ref")
            if not isinstance(legacy_ref, Mapping):
                raise OperatorError(
                    "Subset intermediate state is missing function_origin/function_ref."
                )
            ref = {str(k): str(v) for k, v in legacy_ref.items()}
            kind = ref.get("kind")
            if kind == "file":
                path = Path(ref.get("path", ""))
                if not path.is_absolute():
                    path = intermediate_dir / path
                function = _load_function_from_file(
                    path, str(ref.get("function_name", ""))
                )
            elif kind == "import":
                function = _resolve_legacy_function_ref(ref)
            else:
                raise OperatorError(
                    f"Unsupported legacy subset function reference: {kind!r}."
                )
            obj = cls(
                function,
                function_origin={"kind": f"legacy_{kind}", **ref},
            )
            # Detach from the live/imported object immediately for resumed work.
            obj.function = cast(
                SubsetFunction,
                cloudpickle.loads(cloudpickle.dumps(obj._require_function())),
            )
        obj.operator_id = str(operator_id)
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / _FUNCTION_ASSET
        payload = cloudpickle.dumps(self._require_function())
        path.write_bytes(payload)
        # Once the operator snapshot is written, use the serialized callable as
        # the live execution copy too. This detaches closures/globals from any
        # later mutation in the caller's Python session.
        function = cloudpickle.loads(payload)
        if not callable(function):  # pragma: no cover - cloudpickle contract guard
            raise OperatorError(
                "Frozen subset callable did not deserialize to a callable."
            )
        self.function = cast(SubsetFunction, function)
        return {"function_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        # Legacy import-backed subset operators had no frozen assets. Preserve
        # their ability to load, while all newly saved operators use cloudpickle.
        if not manifest and self.function is not None:
            return

        filename = manifest.get("function_file")
        if not isinstance(filename, str) or not filename:
            raise OperatorError("Subset operator is missing its frozen callable asset.")
        path = assets_dir / filename
        if not path.exists():
            raise OperatorError(f"Missing subset callable asset: {path}.")

        legacy_function_name = manifest.get("function_name")
        if isinstance(legacy_function_name, str) and legacy_function_name:
            # Round <=25 file-backed snapshots copied a .py file instead of a
            # cloudpickle payload. Resolve that frozen source exactly once.
            function = _load_function_from_file(path, legacy_function_name)
            function = cloudpickle.loads(cloudpickle.dumps(function))
        else:
            function = cloudpickle.loads(path.read_bytes())
        if not callable(function):
            raise OperatorError(
                "Subset callable asset did not deserialize to a callable."
            )
        self.function = cast(SubsetFunction, function)

    def to_json_state(self) -> dict[str, Any]:
        return {"function_origin": dict(self.function_origin)}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> FunctionSubsetTranslator:
        origin = state.get("function_origin")
        if isinstance(origin, Mapping):
            return cls(
                None,
                function_origin={str(key): str(value) for key, value in origin.items()},
            )

        # Backward compatibility for Round <=25 subset snapshots. New snapshots
        # never store an executable import reference.
        legacy_ref = state.get("function_ref")
        if not isinstance(legacy_ref, Mapping):
            raise OperatorError(
                "FunctionSubsetTranslator state is missing function_origin."
            )
        ref = {str(key): str(value) for key, value in legacy_ref.items()}
        kind = ref.get("kind")
        legacy_origin = {"kind": f"legacy_{kind or 'unknown'}", **ref}
        if kind == "import":
            function = _resolve_legacy_function_ref(ref)
            return cls(function, function_origin=legacy_origin)
        if kind == "file":
            # BaseOperator.load_from_dir() will hand the copied source asset to
            # load_assets(), so the absolute path in old JSON state is not needed.
            return cls(None, function_origin=legacy_origin)
        raise OperatorError(f"Unsupported legacy subset function reference: {kind!r}.")

    def _require_function(self) -> SubsetFunction:
        if self.function is None:
            raise OperatorError("Subset callable is unavailable.")
        return self.function


def _source_artifact(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            "FunctionSubsetTranslator requires exactly one source under "
            f"{DEFAULT_SOURCE_LABEL!r}; got {tuple(sources)}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


# ---------------------------------------------------------------------------
# Function reference helpers
# ---------------------------------------------------------------------------


def _prepare_function(
    function: FunctionSpec | None,
) -> tuple[dict[str, str], SubsetFunction]:
    if isinstance(function, tuple):
        if len(function) != 2:
            raise ValueError("Function file specs must be (path, function_name).")
        path, function_name = function
        resolved_path = Path(path).expanduser().resolve()
        callable_obj = _load_function_from_file(resolved_path, str(function_name))
        return (
            {
                "kind": "file",
                "path": str(resolved_path),
                "function_name": str(function_name),
            },
            cast(SubsetFunction, callable_obj),
        )

    if not callable(function):
        raise TypeError("function must be callable or a (path, function_name) tuple.")

    module_name = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    name = getattr(function, "__name__", None)
    origin = {
        "kind": "callable",
        "module": str(module_name or ""),
        "qualname": str(qualname or name or type(function).__name__),
    }
    return origin, cast(SubsetFunction, function)


def _resolve_legacy_function_ref(ref: Mapping[str, str]) -> SubsetFunction:
    module_name = str(ref.get("module", ""))
    qualname = str(ref.get("qualname", ""))
    if not module_name or not qualname:
        raise OperatorError("Legacy subset import reference is incomplete.")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise OperatorError(
            f"Legacy subset reference {module_name}:{qualname} is not callable."
        )
    return cast(SubsetFunction, obj)


def _load_function_from_file(path: Path, function_name: str) -> Callable[..., Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise OperatorError(f"Subset function file does not exist: {path}.")
    module_name = f"_teal_subset_function_{abs(hash((str(path), function_name)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OperatorError(f"Cannot import subset function file: {path}.")
    module = importlib.util.module_from_spec(spec)
    _exec_module(spec.loader, module)
    obj = getattr(module, function_name, None)
    if not callable(obj):
        raise OperatorError(
            f"Subset function file {path} has no callable {function_name!r}."
        )
    return cast(Callable[..., Any], obj)


def _exec_module(loader: Any, module: ModuleType) -> None:
    loader.exec_module(module)


# ---------------------------------------------------------------------------
# Packet/key/mask helpers
# ---------------------------------------------------------------------------


def _packet_length(packet: Any) -> int:
    if isinstance(packet, pd.DataFrame):
        return len(packet)
    if isinstance(packet, Mapping) and isinstance(packet.get("info"), pd.DataFrame):
        return len(packet["info"])
    if isinstance(packet, Sequence) and not isinstance(packet, (str, bytes, bytearray)):
        return len(packet)
    try:
        return len(packet)
    except TypeError as exc:
        raise ArtifactError(
            f"Cannot determine row count for subset packet of type {type(packet).__name__}."
        ) from exc


def _extract_key_frame(packet: Any, *, primary_key: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(packet, pd.DataFrame):
        frame = packet
    elif isinstance(packet, Mapping) and isinstance(packet.get("info"), pd.DataFrame):
        frame = packet["info"]
    elif isinstance(packet, Sequence) and not isinstance(
        packet, (str, bytes, bytearray)
    ):
        frame = pd.DataFrame(list(packet))
    else:
        raise ArtifactError(
            f"Cannot extract keys from subset packet of type {type(packet).__name__}. "
            "Use form='table' or form='records', or a native form that exposes an "
            "'info' DataFrame."
        )

    missing = [column for column in primary_key if column not in frame.columns]
    if missing:
        raise ArtifactError(
            f"Subset packet is missing primary key column(s) {missing}. "
            f"Available columns: {list(frame.columns)}."
        )
    keys = frame.loc[:, list(primary_key)].copy().reset_index(drop=True)
    if keys.isna().any().any():
        raise ArtifactError("Subset selected key frame contains NA primary key values.")
    return keys


def _normalize_mask(mask: Iterable[bool], *, expected_len: int) -> np.ndarray:
    if isinstance(mask, (bool, np.bool_)):
        raise ArtifactError(
            "Subset function must return one boolean per row, not a scalar bool."
        )
    if isinstance(mask, pd.Series):
        arr = mask.to_numpy()
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(list(mask))

    if arr.ndim != 1:
        raise ArtifactError(
            f"Subset mask must be one-dimensional; got shape {tuple(arr.shape)}."
        )
    if len(arr) != expected_len:
        raise ArtifactError(
            f"Subset mask length {len(arr)} does not match packet length {expected_len}."
        )
    if _mask_has_na(arr):
        raise ArtifactError("Subset mask contains NA values.")
    return arr.astype(bool, copy=False)


def _mask_has_na(arr: np.ndarray) -> bool:
    try:
        return bool(pd.isna(arr).any())
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _source_column_info(source: BaseArtifact) -> Mapping[str, Any]:
    try:
        return source.project.query.query_columns(source, metadata_mode="full")
    except Exception as exc:
        raise ArtifactError(
            f"Cannot inspect columns for artifact {source.artifact_id!r}."
        ) from exc


def _validate_column_select(
    value: ColumnSelect,
    *,
    name: str,
    namespace: str,
    info: Mapping[str, Any],
) -> ColumnSelect:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        requested = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        requested = tuple(value)
    else:
        raise ValueError(
            f"{name} must be a boolean, a non-empty string, or a sequence "
            "of non-empty strings."
        )

    if not requested or any(
        not isinstance(column, str) or not column for column in requested
    ):
        raise ValueError(
            f"{name} must be a non-empty string or a sequence of non-empty strings."
        )
    if len(set(requested)) != len(requested):
        raise ValueError(f"{name} contains duplicate columns: {list(requested)!r}.")

    output_names = set(info.get("output", ()))
    ambiguous = info.get("ambiguous", {})
    output_to_namespace = {
        column["output_name"]: column["namespace"] for column in info.get("columns", ())
    }
    available = sorted(
        output_name
        for output_name, column_namespace in output_to_namespace.items()
        if column_namespace == namespace
    )

    for column in requested:
        if column not in output_names:
            if column in ambiguous:
                raise QueryError(
                    f"{name} column {column!r} is ambiguous. Use one of these "
                    f"qualified names instead: {ambiguous[column]}."
                )
            raise QueryError(
                f"{name} column {column!r} is not available. Available "
                f"{namespace} columns are: {available}."
            )
        actual_namespace = output_to_namespace.get(column)
        if actual_namespace != namespace:
            raise QueryError(
                f"{name} column {column!r} belongs to the "
                f"{actual_namespace!r} namespace, not {namespace!r}."
            )
    return value


def _predicate_packet(
    packet: Any,
    *,
    primary_key: tuple[str, ...],
    key_columns: ColumnSelect,
) -> Any:
    """Hide unrequested key columns from the subset predicate input.

    Subset execution always materializes the complete source primary key so TeAL
    can write the preserved-key output. ``key_columns`` controls only what the
    user callable sees. This keeps structural key preservation independent from
    predicate input selection.
    """
    if key_columns is True:
        return packet

    if key_columns is False:
        visible_keys: set[str] = set()
    elif isinstance(key_columns, str):
        visible_keys = {key_columns}
    else:
        visible_keys = {str(column) for column in key_columns}

    hidden_keys = {str(column) for column in primary_key} - visible_keys
    if not hidden_keys:
        return packet

    if isinstance(packet, pd.DataFrame):
        return packet.drop(
            columns=[column for column in hidden_keys if column in packet.columns]
        )

    if isinstance(packet, Mapping) and isinstance(packet.get("info"), pd.DataFrame):
        copied = dict(packet)
        info = packet["info"]
        copied["info"] = info.drop(
            columns=[column for column in hidden_keys if column in info.columns]
        )
        return copied

    if isinstance(packet, Sequence) and not isinstance(packet, (str, bytes, bytearray)):
        records: list[Any] = []
        for row in packet:
            if isinstance(row, Mapping):
                records.append(
                    {key: value for key, value in row.items() if key not in hidden_keys}
                )
            else:
                records.append(row)
        return records

    return packet


def _validate_metadata_mode(value: MetadataMode) -> MetadataMode:
    if value not in {"none", "local", "full"}:
        raise ValueError("metadata_mode must be 'none', 'local', or 'full'.")
    return value


def _validate_form(value: str) -> QueryForm:
    if value not in QUERY_FORMS:
        raise ValueError("form must be 'table', 'records', or 'native'.")
    return cast(QueryForm, value)


def _validate_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if value == DEFAULT_SOURCE_LABEL:
        raise ValueError(
            f"{name} cannot be {DEFAULT_SOURCE_LABEL!r}; input and output labels must differ."
        )
    return value


def _json_column_select(value: ColumnSelect) -> bool | str | list[str]:
    if isinstance(value, bool) or isinstance(value, str):
        return value
    return [str(column) for column in value]
