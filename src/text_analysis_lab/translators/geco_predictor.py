"""Frozen TeAL execution wrapper for GeCo prediction procedures.

``GeCoPredictor`` is intentionally GeCo-specific. GeCo owns exploratory predictor
construction and fitting; TeAL owns durable execution after GeCo exports a frozen
prediction procedure. The initial implementation supports binary classifiers,
fixed classifier committees, and logistic-stacking committees over one or more
row-aligned sparse/dense TeAL matrix artifacts.
"""

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


_FORMAT_VERSION = 1
_MEMBER_ASSET_PREFIX = "member_"
_STACKER_ASSET = "stacker.pkl"
_MATRIX_TYPES = ("sparse_matrix", "dense_matrix")
_FIXED_AGGREGATIONS = {
    "mean",
    "median",
    "minimum",
    "maximum",
    "harmonic_mean",
    "geometric_mean",
}
_SUPPORTED_AGGREGATIONS = {"single", *_FIXED_AGGREGATIONS, "logistic_stack"}


class GeCoPredictor(BaseTranslator):
    """Execute one frozen GeCo predictor against ordered TeAL geometry sources.

    The class is already fitted when constructed. It never fits or retrains GeCo
    state. Source indices refer to the ordered unique geometry list frozen by GeCo.

    Version 1 supports binary classification procedures only. Every member consumes
    one sparse/dense source matrix and exposes ``predict_proba`` plus ``classes_``.
    Fixed committees aggregate member probabilities; logistic-stacking committees
    pass the ordered member-probability matrix through a fitted stacker.
    """

    operation_type = "translate"

    def __init__(
        self,
        member_models: Sequence[Any | None],
        *,
        source_specs: Sequence[Mapping[str, Any]],
        member_specs: Sequence[Mapping[str, Any]],
        aggregation: str = "single",
        stacker: Any | None = None,
        stacker_positive_class: Any | None = 1,
        threshold: float = 0.5,
        output_semantics: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        stale_at_export: bool = False,
        format_version: int = _FORMAT_VERSION,
        operator_id: str | None = None,
        _allow_unloaded_assets: bool = False,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.format_version = int(format_version)
        if self.format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported GeCoPredictor format_version={self.format_version}; "
                f"expected {_FORMAT_VERSION}."
            )

        self.source_specs = _normalize_source_specs(source_specs)
        self.member_specs = _normalize_member_specs(
            member_specs, source_count=len(self.source_specs)
        )
        self.member_models = list(member_models)
        if len(self.member_models) != len(self.member_specs):
            raise ValueError(
                "GeCoPredictor member_models and member_specs must have equal length."
            )
        if not self.member_models:
            raise ValueError("GeCoPredictor requires at least one fitted member model.")

        self.aggregation = str(aggregation)
        if self.aggregation not in _SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported GeCoPredictor aggregation={self.aggregation!r}; "
                f"expected one of {sorted(_SUPPORTED_AGGREGATIONS)}."
            )
        if self.aggregation == "single" and len(self.member_models) != 1:
            raise ValueError("aggregation='single' requires exactly one member model.")
        if self.aggregation == "logistic_stack" and stacker is None and not _allow_unloaded_assets:
            raise ValueError("aggregation='logistic_stack' requires a fitted stacker.")
        if self.aggregation != "logistic_stack" and stacker is not None:
            raise ValueError("A fitted stacker is valid only for aggregation='logistic_stack'.")
        self.stacker = stacker
        self.stacker_positive_class = _json_scalar(
            stacker_positive_class, name="stacker_positive_class"
        )

        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("GeCoPredictor threshold must lie in the closed interval [0, 1].")

        semantics = dict(output_semantics or {"kind": "binary_classification"})
        if str(semantics.get("kind", "binary_classification")) != "binary_classification":
            raise ValueError(
                "GeCoPredictor format v1 supports only output_semantics kind "
                "'binary_classification'."
            )
        self.output_semantics = _json_mapping(semantics, name="output_semantics")
        self.provenance = _json_mapping(dict(provenance or {}), name="provenance")
        self.stale_at_export = bool(stale_at_export)

        if not _allow_unloaded_assets:
            for index, model in enumerate(self.member_models):
                _require_probability_model(model, name=f"member {index}")
            if self.aggregation == "logistic_stack":
                _require_probability_model(self.stacker, name="stacker")

    @classmethod
    def single_classifier(
        cls,
        model: Any,
        *,
        geometry_id: Any | None = None,
        geometry_name: str | None = None,
        n_features: int | None = None,
        original_external_ref: Any | None = None,
        positive_class: Any = 1,
        threshold: float = 0.5,
        provenance: Mapping[str, Any] | None = None,
        stale_at_export: bool = False,
    ) -> "GeCoPredictor":
        """Construct the one-member case used by legacy GeCo classifier export."""
        source_spec = {
            "source_index": 0,
            "geometry_id": geometry_id,
            "geometry_name": geometry_name,
            "n_features": n_features,
            "original_external_ref": original_external_ref,
        }
        member_spec = {"source_index": 0, "positive_class": positive_class}
        return cls(
            [model],
            source_specs=[source_spec],
            member_specs=[member_spec],
            aggregation="single",
            threshold=threshold,
            provenance=provenance,
            stale_at_export=stale_at_export,
        )

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
        labels = self._validate_source_bindings(sources)
        # The first ordered geometry defines the prediction artifact's key basis.
        # All supplied geometries remain recorded as operation sources/provenance,
        # while the output itself owns new prediction data.  ``joined_key`` is not
        # appropriate here: it is TeAL's lazy horizontal-composition lineage and
        # therefore resolves representation data from its bases instead of the
        # translator-owned prediction columns.
        return OutputSpec(
            artifact_type="table",
            lineage_mode="preserved_key",
            basis_labels=labels[0],
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
                f"GeCoPredictor does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> Mapping[str, SourceRequest]:
        if mode != "translate":
            raise OperatorError("GeCoPredictor is inference-only and supports translate mode.")
        labels = self._validate_source_bindings(sources)
        batch_size = request.batch_size or 10_000
        return {
            label: SourceRequest(
                artifact_type=_MATRIX_TYPES,
                mode="batches",
                columns=ColumnRequest(keys=True, data=True, metadata=False),
                batch_size=batch_size,
                form="native",
                metadata_mode="none",
                include_position=False,
            )
            for label in labels
        }

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported GeCoPredictor mode {mode!r}.")
        labels = _ordered_input_labels(inputs, expected_count=len(self.source_specs))
        packets = [inputs[label] for label in labels]
        keys, matrices = _aligned_matrices(packets)

        member_probabilities: list[np.ndarray] = []
        for index, (model, spec) in enumerate(zip(self.member_models, self.member_specs, strict=True)):
            fitted = _require_probability_model(model, name=f"member {index}")
            source_index = int(spec["source_index"])
            probability = _positive_probability(
                fitted,
                matrices[source_index],
                positive_class=spec["positive_class"],
                expected_rows=len(keys),
                name=f"member {index}",
            )
            member_probabilities.append(probability)

        probability = self._combine_probabilities(member_probabilities)
        prediction = (probability >= self.threshold).astype("int64")
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "data": pd.DataFrame(
                        {
                            "prediction": prediction,
                            "probability": probability,
                        }
                    ).reset_index(drop=True),
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
    ) -> "GeCoPredictor":
        _ = request
        if mode != "translate":
            raise OperatorError("GeCoPredictor workers support translate mode only.")
        return GeCoPredictor(
            [clone_estimator(_require_probability_model(model, name=f"member {index}"))
             for index, model in enumerate(self.member_models)],
            source_specs=self.source_specs,
            member_specs=self.member_specs,
            aggregation=self.aggregation,
            stacker=(
                clone_estimator(_require_probability_model(self.stacker, name="stacker"))
                if self.aggregation == "logistic_stack"
                else None
            ),
            stacker_positive_class=self.stacker_positive_class,
            threshold=self.threshold,
            output_semantics=self.output_semantics,
            provenance=self.provenance,
            stale_at_export=self.stale_at_export,
            format_version=self.format_version,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "source_specs": [dict(spec) for spec in self.source_specs],
            "member_specs": [dict(spec) for spec in self.member_specs],
            "aggregation": self.aggregation,
            "stacker_positive_class": self.stacker_positive_class,
            "threshold": self.threshold,
            "output_semantics": dict(self.output_semantics),
            "provenance": dict(self.provenance),
            "stale_at_export": self.stale_at_export,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "GeCoPredictor":
        source_specs = state.get("source_specs", [])
        member_specs = state.get("member_specs", [])
        if not isinstance(source_specs, Sequence) or isinstance(source_specs, (str, bytes)):
            raise OperatorError("GeCoPredictor source_specs state must be a sequence.")
        if not isinstance(member_specs, Sequence) or isinstance(member_specs, (str, bytes)):
            raise OperatorError("GeCoPredictor member_specs state must be a sequence.")
        aggregation = str(state.get("aggregation", "single"))
        return cls(
            [None] * len(member_specs),
            source_specs=source_specs,
            member_specs=member_specs,
            aggregation=aggregation,
            stacker=None,
            stacker_positive_class=state.get("stacker_positive_class", 1),
            threshold=float(state.get("threshold", 0.5)),
            output_semantics=_mapping_or_empty(state.get("output_semantics")),
            provenance=_mapping_or_empty(state.get("provenance")),
            stale_at_export=bool(state.get("stale_at_export", False)),
            format_version=int(state.get("format_version", _FORMAT_VERSION)),
            _allow_unloaded_assets=True,
        )

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        member_files: list[str] = []
        for index, model in enumerate(self.member_models):
            fitted = _require_probability_model(model, name=f"member {index}")
            filename = f"{_MEMBER_ASSET_PREFIX}{index:03d}.pkl"
            dump_estimator(assets_dir / filename, fitted)
            member_files.append(filename)
        manifest: dict[str, Any] = {"member_files": member_files}
        if self.aggregation == "logistic_stack":
            stacker = _require_probability_model(self.stacker, name="stacker")
            dump_estimator(assets_dir / _STACKER_ASSET, stacker)
            manifest["stacker_file"] = _STACKER_ASSET
        return manifest

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        raw_member_files = manifest.get("member_files")
        if not isinstance(raw_member_files, Sequence) or isinstance(raw_member_files, (str, bytes)):
            raise OperatorError("GeCoPredictor operator is missing its member asset list.")
        member_files = [str(value) for value in raw_member_files]
        if len(member_files) != len(self.member_specs):
            raise OperatorError(
                "GeCoPredictor member asset count does not match serialized member specs."
            )
        self.member_models = [load_estimator(assets_dir / filename) for filename in member_files]
        if self.aggregation == "logistic_stack":
            filename = manifest.get("stacker_file")
            if not isinstance(filename, str) or not filename:
                raise OperatorError("GeCoPredictor logistic stack is missing its stacker asset.")
            self.stacker = load_estimator(assets_dir / filename)
        else:
            self.stacker = None

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
        state = {
            **self.to_json_state(),
            "operator_id": operator_id,
            "assets": dict(self.save_assets(intermediate_dir)),
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
    ) -> "GeCoPredictor":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise OperatorError("GeCoPredictor intermediate state must be a mapping.")
        obj = cls.from_json_state(state)
        obj.operator_id = operator_id
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("GeCoPredictor intermediate assets must be a mapping.")
        obj.load_assets(intermediate_dir, assets)
        return obj

    def _validate_source_bindings(self, sources: Mapping[str, "BaseArtifact"]) -> list[str]:
        labels = _ordered_source_labels(sources, expected_count=len(self.source_specs))
        first = sources[labels[0]]
        first_key = tuple(str(value) for value in first.primary_key)
        first_rows = first.n_rows
        for source_index, label in enumerate(labels):
            artifact = sources[label]
            if artifact.artifact_type.value not in _MATRIX_TYPES:
                raise OperatorError(
                    "GeCoPredictor requires sparse_matrix or dense_matrix sources; "
                    f"got {artifact.artifact_type.value!r} for {label!r}."
                )
            if tuple(str(value) for value in artifact.primary_key) != first_key:
                raise OperatorError(
                    "GeCoPredictor sources must use the same primary-key schema."
                )
            if first_rows is not None and artifact.n_rows is not None and int(artifact.n_rows) != int(first_rows):
                raise OperatorError(
                    "GeCoPredictor sources must have the same number of rows before batching; "
                    f"{labels[0]!r} has {first_rows}, {label!r} has {artifact.n_rows}."
                )
            required_width = self.source_specs[source_index].get("n_features")
            if required_width is not None:
                actual_width = len(artifact.get_data_columns())
                if int(actual_width) != int(required_width):
                    geometry_name = self.source_specs[source_index].get("geometry_name")
                    raise OperatorError(
                        f"GeCoPredictor source {source_index} ({geometry_name!r}) requires "
                        f"{required_width} features, but {label!r} has {actual_width}."
                    )
        return labels

    def _combine_probabilities(self, member_probabilities: Sequence[np.ndarray]) -> np.ndarray:
        if not member_probabilities:
            raise OperatorError("GeCoPredictor has no member probabilities to combine.")
        matrix = np.column_stack([np.asarray(value, dtype=float) for value in member_probabilities])
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ArtifactError("GeCoPredictor member probabilities must form a non-empty matrix.")
        _validate_probabilities(matrix, name="member probabilities")

        if self.aggregation == "single":
            probability = matrix[:, 0]
        elif self.aggregation == "mean":
            probability = np.mean(matrix, axis=1)
        elif self.aggregation == "median":
            probability = np.median(matrix, axis=1)
        elif self.aggregation == "minimum":
            probability = np.min(matrix, axis=1)
        elif self.aggregation == "maximum":
            probability = np.max(matrix, axis=1)
        elif self.aggregation == "harmonic_mean":
            positive = np.all(matrix > 0.0, axis=1)
            probability = np.zeros(matrix.shape[0], dtype=float)
            probability[positive] = matrix.shape[1] / np.sum(1.0 / matrix[positive], axis=1)
        elif self.aggregation == "geometric_mean":
            positive = np.all(matrix > 0.0, axis=1)
            probability = np.zeros(matrix.shape[0], dtype=float)
            probability[positive] = np.exp(np.mean(np.log(matrix[positive]), axis=1))
        elif self.aggregation == "logistic_stack":
            stacker = _require_probability_model(self.stacker, name="stacker")
            probability = _positive_probability(
                stacker,
                matrix,
                positive_class=self.stacker_positive_class,
                expected_rows=matrix.shape[0],
                name="stacker",
            )
        else:  # pragma: no cover - guarded by constructor
            raise OperatorError(f"Unsupported GeCoPredictor aggregation {self.aggregation!r}.")

        probability = np.asarray(probability, dtype=float).reshape(-1)
        _validate_probabilities(probability, name="final probability")
        return probability


def _ordered_source_labels(
    sources: Mapping[str, Any], *, expected_count: int
) -> list[str]:
    if expected_count <= 0:
        raise OperatorError("GeCoPredictor requires at least one source.")
    if expected_count == 1 and set(sources) == {DEFAULT_SOURCE_LABEL}:
        return [DEFAULT_SOURCE_LABEL]
    expected = [f"source_{index}" for index in range(expected_count)]
    if list(sources) != expected and set(sources) != set(expected):
        raise OperatorError(
            "GeCoPredictor source bindings must correspond to its ordered source list; "
            f"expected {expected}, got {list(sources)}."
        )
    return expected


def _ordered_input_labels(
    inputs: Mapping[str, InputBatch], *, expected_count: int
) -> list[str]:
    if expected_count == 1 and set(inputs) == {DEFAULT_SOURCE_LABEL}:
        return [DEFAULT_SOURCE_LABEL]
    expected = [f"source_{index}" for index in range(expected_count)]
    if set(inputs) != set(expected):
        raise OperatorError(
            "GeCoPredictor execution unit omitted or added a required source; "
            f"expected {expected}, got {list(inputs)}."
        )
    return expected


def _aligned_matrices(packets: Sequence[InputBatch]) -> tuple[pd.DataFrame, list[Any]]:
    if not packets:
        raise OperatorError("GeCoPredictor received no source packets.")
    first_keys: pd.DataFrame | None = None
    first_primary_key: tuple[str, ...] | None = None
    first_batch_index: int | None = None
    first_batch_count: int | None = None
    matrices: list[Any] = []

    for packet in packets:
        if not isinstance(packet.data, Mapping):
            raise ArtifactError("GeCoPredictor expected native matrix packets.")
        info = packet.data.get("info")
        matrix = packet.data.get("matrix")
        if not isinstance(info, pd.DataFrame):
            raise ArtifactError(
                f"GeCoPredictor source {packet.source_label!r} is missing matrix info rows."
            )
        if sparse.issparse(matrix):
            if len(matrix.shape) != 2:
                raise ArtifactError("GeCoPredictor source matrix must be two-dimensional.")
            n_rows = int(matrix.shape[0])
        else:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                raise ArtifactError("GeCoPredictor source matrix must be two-dimensional.")
            n_rows = int(matrix.shape[0])
        if len(info) != n_rows:
            raise ArtifactError(
                f"GeCoPredictor source {packet.source_label!r} has {len(info)} key rows "
                f"but {n_rows} matrix rows."
            )

        primary_key = tuple(str(value) for value in packet.primary_key)
        missing = [column for column in primary_key if column not in info.columns]
        if missing:
            raise ArtifactError(
                f"GeCoPredictor source {packet.source_label!r} is missing key columns {missing}."
            )
        keys = info.loc[:, list(primary_key)].reset_index(drop=True)

        if first_keys is None:
            first_keys = keys
            first_primary_key = primary_key
            first_batch_index = int(packet.batch_index)
            first_batch_count = int(packet.batch_count)
        else:
            if primary_key != first_primary_key:
                raise ArtifactError(
                    "GeCoPredictor source packets use different primary-key schemas."
                )
            if int(packet.batch_index) != first_batch_index or int(packet.batch_count) != first_batch_count:
                raise ArtifactError(
                    "GeCoPredictor source packets are not synchronized to the same batch index/count."
                )
            if not keys.equals(first_keys):
                raise ArtifactError(
                    "GeCoPredictor source packets are not row-aligned: primary-key values "
                    "must be exactly equal and in identical order before prediction."
                )
        matrices.append(matrix)

    assert first_keys is not None
    return first_keys, matrices


def _positive_probability(
    model: Any,
    values: Any,
    *,
    positive_class: Any,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    predict_proba = getattr(model, "predict_proba", None)
    classes = getattr(model, "classes_", None)
    if not callable(predict_proba) or classes is None:
        raise OperatorError(
            f"GeCoPredictor {name} must expose predict_proba(...) and fitted classes_."
        )
    class_values = list(np.asarray(classes).reshape(-1))
    matches = [
        index for index, value in enumerate(class_values) if _class_equal(value, positive_class)
    ]
    if len(matches) != 1:
        raise OperatorError(
            f"GeCoPredictor {name} positive_class={positive_class!r} is not uniquely "
            f"present in classes_={class_values!r}."
        )
    probabilities = np.asarray(predict_proba(values), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != int(expected_rows):
        raise ArtifactError(
            f"GeCoPredictor {name}.predict_proba(...) must return a two-dimensional "
            "array with one row per input observation."
        )
    class_index = matches[0]
    if class_index >= probabilities.shape[1]:
        raise ArtifactError(
            f"GeCoPredictor {name}.predict_proba(...) returned fewer columns than classes_."
        )
    result = probabilities[:, class_index]
    _validate_probabilities(result, name=f"{name} probability")
    return result


def _validate_probabilities(values: Any, *, name: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ArtifactError(f"GeCoPredictor {name} contains NaN or infinite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ArtifactError(f"GeCoPredictor {name} must lie in the closed interval [0, 1].")


def _require_probability_model(model: Any | None, *, name: str) -> Any:
    if model is None:
        raise OperatorError(f"GeCoPredictor fitted {name} asset is unavailable.")
    if not callable(getattr(model, "predict_proba", None)):
        raise OperatorError(f"GeCoPredictor {name} has no callable predict_proba(...).")
    if getattr(model, "classes_", None) is None:
        raise OperatorError(f"GeCoPredictor {name} has no fitted classes_.")
    return model


def _normalize_source_specs(source_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source_specs, (str, bytes)) or not isinstance(source_specs, Sequence):
        raise TypeError("GeCoPredictor source_specs must be a sequence of mappings.")
    if not source_specs:
        raise ValueError("GeCoPredictor requires at least one source specification.")
    normalized: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(source_specs):
        if not isinstance(raw, Mapping):
            raise TypeError("Each GeCoPredictor source spec must be a mapping.")
        spec = dict(raw)
        index = int(spec.get("source_index", expected_index))
        if index != expected_index:
            raise ValueError(
                "GeCoPredictor source_specs must be ordered unique sources with contiguous "
                f"source_index values starting at 0; expected {expected_index}, got {index}."
            )
        spec["source_index"] = index
        if spec.get("n_features") is not None:
            n_features = int(spec["n_features"])
            if n_features <= 0:
                raise ValueError("GeCoPredictor source n_features must be positive when provided.")
            spec["n_features"] = n_features
        normalized.append(_json_mapping(spec, name=f"source_specs[{expected_index}]"))
    return normalized


def _normalize_member_specs(
    member_specs: Sequence[Mapping[str, Any]], *, source_count: int
) -> list[dict[str, Any]]:
    if isinstance(member_specs, (str, bytes)) or not isinstance(member_specs, Sequence):
        raise TypeError("GeCoPredictor member_specs must be a sequence of mappings.")
    if not member_specs:
        raise ValueError("GeCoPredictor requires at least one member specification.")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(member_specs):
        if not isinstance(raw, Mapping):
            raise TypeError("Each GeCoPredictor member spec must be a mapping.")
        spec = dict(raw)
        if "source_index" not in spec:
            raise ValueError(f"GeCoPredictor member_specs[{index}] is missing source_index.")
        source_index = int(spec["source_index"])
        if source_index < 0 or source_index >= source_count:
            raise ValueError(
                f"GeCoPredictor member_specs[{index}] source_index={source_index} is outside "
                f"the available source range 0..{source_count - 1}."
            )
        spec["source_index"] = source_index
        spec["positive_class"] = _json_scalar(
            spec.get("positive_class", 1), name=f"member_specs[{index}].positive_class"
        )
        normalized.append(_json_mapping(spec, name=f"member_specs[{index}]"))
    return normalized


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    data = dict(value)
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"GeCoPredictor {name} must be JSON-serializable.") from exc
    return data


def _json_scalar(value: Any, *, name: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"GeCoPredictor {name} must be finite when numeric.")
        return value
    raise TypeError(f"GeCoPredictor {name} must be a JSON scalar.")


def _class_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    try:
        return bool(left == right)
    except Exception:
        return False
