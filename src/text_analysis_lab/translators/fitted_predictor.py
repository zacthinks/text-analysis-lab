"""Frozen wrapper for applying an already-fitted prediction object."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BatchResult,
    BaseTranslator,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    RunRoute,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    clone_estimator,
    dump_estimator,
    load_estimator,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_MODEL_ASSET = "predictor.pkl"
_SUPPORTED_TYPES = ("table", "jsonl", "sparse_matrix", "dense_matrix")


class FittedPredictor(BaseTranslator):
    """Freeze and apply an arbitrary already-fitted estimator-like object.

    TeAL deliberately does not record or enforce a semantic feature contract for
    the supplied model.  Compatibility is delegated to the model's ordinary
    ``predict``/``predict_proba`` behavior.
    """

    operation_type = "translate"

    def __init__(
        self,
        model: Any | None,
        *,
        probability_class: Any | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if model is not None and not callable(getattr(model, "predict", None)):
            raise TypeError("FittedPredictor model must expose a callable predict(...).")
        self.model = model
        self.probability_class = _json_scalar(probability_class)
        self._source_type: str | None = None

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = _single_source(sources)
        if source.artifact_type.value not in _SUPPORTED_TYPES:
            raise OperatorError(
                "FittedPredictor requires a table, jsonl, sparse_matrix, or "
                f"dense_matrix source; got {source.artifact_type.value!r}."
            )
        return OutputSpec(
            artifact_type="table",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                f"FittedPredictor does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _single_source(sources)
        source_type = source.artifact_type.value
        if source_type not in _SUPPORTED_TYPES:
            raise OperatorError(
                "FittedPredictor requires a table, jsonl, sparse_matrix, or dense_matrix source."
            )
        self._source_type = source_type
        return SourceRequest(
            artifact_type=_SUPPORTED_TYPES,
            mode="batches",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=request.batch_size or 10_000,
            form="native" if source_type in {"sparse_matrix", "dense_matrix"} else "table",
            metadata_mode="none",
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
            raise OperatorError(f"Unsupported FittedPredictor mode {mode!r}.")
        packet = _single_input(inputs)
        keys, values = _prediction_input(packet, source_type=self._source_type)
        model = self._require_model()

        prediction = _one_dimensional(
            model.predict(values), expected_rows=len(keys), name="model.predict"
        )
        data: dict[str, Any] = {"prediction": prediction}

        if self.probability_class is not None:
            predict_proba = getattr(model, "predict_proba", None)
            if not callable(predict_proba):
                raise OperatorError(
                    "FittedPredictor probability_class was requested, but the model "
                    "does not expose predict_proba(...)."
                )
            classes = getattr(model, "classes_", None)
            if classes is None:
                raise OperatorError(
                    "FittedPredictor probability_class was requested, but the model "
                    "does not expose classes_."
                )
            class_values = list(np.asarray(classes).reshape(-1))
            matches = [
                idx
                for idx, value in enumerate(class_values)
                if _class_equal(value, self.probability_class)
            ]
            if len(matches) != 1:
                raise OperatorError(
                    f"Requested probability_class={self.probability_class!r} is not uniquely "
                    f"present in model.classes_={class_values!r}."
                )
            probabilities = np.asarray(predict_proba(values))
            if probabilities.ndim != 2 or probabilities.shape[0] != len(keys):
                raise ArtifactError(
                    "model.predict_proba(...) must return a two-dimensional array with "
                    "one row per input observation."
                )
            class_index = matches[0]
            if class_index >= probabilities.shape[1]:
                raise ArtifactError(
                    "model.predict_proba(...) returned fewer columns than model.classes_."
                )
            data["probability"] = probabilities[:, class_index]

        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "data": pd.DataFrame(data).reset_index(drop=True),
                }
            }
        )

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
    ) -> "FittedPredictor":
        _ = request
        if mode != "translate":
            raise OperatorError("FittedPredictor workers support translate mode only.")
        worker = FittedPredictor(
            clone_estimator(self._require_model()),
            probability_class=self.probability_class,
        )
        worker._source_type = self._source_type
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {"probability_class": self.probability_class}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "FittedPredictor":
        return cls(None, probability_class=state.get("probability_class"))

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / _MODEL_ASSET
        dump_estimator(path, self._require_model())
        return {"estimator_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file")
        if not isinstance(filename, str) or not filename:
            raise OperatorError("FittedPredictor operator is missing its estimator asset.")
        self.model = load_estimator(assets_dir / filename)

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
            **self.to_json_state(),
            "operator_id": operator_id,
            "source_type": self._source_type,
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
    ) -> "FittedPredictor":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls.from_json_state(state)
        obj.operator_id = operator_id
        raw_source_type = state.get("source_type")
        obj._source_type = None if raw_source_type is None else str(raw_source_type)
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("FittedPredictor intermediate assets must be a mapping.")
        obj.load_assets(intermediate_dir, assets)
        return obj

    def _require_model(self) -> Any:
        if self.model is None:
            raise OperatorError("FittedPredictor estimator is unavailable.")
        if not callable(getattr(self.model, "predict", None)):
            raise OperatorError("FittedPredictor estimator has no callable predict(...).")
        return self.model


def _single_source(sources: Mapping[str, Any]):
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("FittedPredictor requires exactly one source under 'source'.")
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("FittedPredictor expected exactly one input under 'source'.")
    return inputs[DEFAULT_SOURCE_LABEL]


def _prediction_input(packet: InputBatch, *, source_type: str | None) -> tuple[pd.DataFrame, Any]:
    key_columns = list(packet.primary_key)
    if source_type in {"sparse_matrix", "dense_matrix"}:
        if not isinstance(packet.data, Mapping):
            raise ArtifactError("FittedPredictor expected a native matrix packet.")
        info = packet.data.get("info")
        matrix = packet.data.get("matrix")
        if not isinstance(info, pd.DataFrame):
            raise ArtifactError("FittedPredictor matrix packet is missing info rows.")
        if sparse.issparse(matrix):
            n_rows = int(matrix.shape[0])
        else:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                raise ArtifactError("FittedPredictor matrix input must be two-dimensional.")
            n_rows = int(matrix.shape[0])
        if len(info) != n_rows:
            raise ArtifactError("FittedPredictor matrix keys and values have different row counts.")
        missing = [column for column in key_columns if column not in info.columns]
        if missing:
            raise ArtifactError(f"FittedPredictor packet is missing key column(s) {missing}.")
        return info.loc[:, key_columns].reset_index(drop=True), matrix

    if not isinstance(packet.data, pd.DataFrame):
        raise ArtifactError("FittedPredictor relational input must materialize as a DataFrame.")
    frame = packet.data
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ArtifactError(f"FittedPredictor packet is missing key column(s) {missing}.")
    keys = frame.loc[:, key_columns].reset_index(drop=True)
    values = frame.drop(columns=key_columns, errors="ignore").reset_index(drop=True)
    return keys, values


def _one_dimensional(value: Any, *, expected_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ArtifactError(f"{name}(...) must return one value per row; got shape {arr.shape}.")
    if len(arr) != int(expected_rows):
        raise ArtifactError(
            f"{name}(...) returned {len(arr)} values for {expected_rows} input rows."
        )
    return arr


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("probability_class must be JSON-serializable and finite.")
        return value
    raise TypeError(
        "probability_class must be None or a JSON-scalar class label (str/int/float/bool)."
    )


def _class_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    try:
        result = left == right
        return bool(result)
    except Exception:
        return False
