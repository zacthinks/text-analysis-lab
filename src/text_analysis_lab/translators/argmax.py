"""Row-wise maximum feature label and value for named numeric matrices."""

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
    BaseTranslator, BatchResult, ColumnRequest, InputBatch, OutputMap,
    OutputSpec, RunRoute, SourceRequest, TranslationMode, TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    feature_labels, key_frame, native_matrix_packet, single_input,
    single_source, unpack_standalone_matrix,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class ArgMax(BaseTranslator):
    """Persist each matrix row's largest named feature and original numeric value.

    Ties select the lexicographically smallest canonical feature name. Completely
    zero rows have no assignment (max_feature is null; max_value is zero).
    Sparse implicit zeros participate as real values without densifying.
    """

    operation_type = "translate"
    frozen_runtime_fields = frozenset({"_features"})

    def __init__(self, *, operator_id: str | None = None) -> None:
        super().__init__(operator_id=operator_id)
        self._features: tuple[str, ...] | None = None

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ) -> OutputSpec:
        _ = request
        source = single_source(sources, name="ArgMax")
        if source.artifact_type.value not in {"dense_matrix", "sparse_matrix"}:
            raise OperatorError("ArgMax requires a dense_matrix or sparse_matrix source.")
        return OutputSpec(
            artifact_type="table", lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def validate_operation_params(
        self, params: Mapping[str, Any], *,
        sources: Mapping[str, BaseArtifact], mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(f"ArgMax does not accept operation parameters: {sorted(params)}.")
        return {}

    def input_request(
        self, *, sources: Mapping[str, BaseArtifact],
        mode: TranslationMode, request: TranslationRequest,
    ) -> SourceRequest:
        if mode != "translate":
            raise OperatorError("ArgMax supports translate mode only.")
        source = single_source(sources, name="ArgMax")
        if source.artifact_type.value not in {"dense_matrix", "sparse_matrix"}:
            raise OperatorError("ArgMax requires a matrix source.")
        self._features = _validate_features(source.get_data_columns())
        return SourceRequest(
            artifact_type=("dense_matrix", "sparse_matrix"),
            mode="batches", columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=request.batch_size or 10_000, form="native",
            metadata_mode="none", include_position=False,
        )

    def translate(
        self, matrix: Any, *, features: Sequence[str] | None = None
    ) -> pd.DataFrame:
        values, metadata, _ = unpack_standalone_matrix(
            matrix, name="ArgMax.translate(...)"
        )
        if metadata is not None:
            embedded = feature_labels(metadata, name="ArgMax")
            if features is not None and list(features) != embedded:
                raise ValueError("ArgMax received conflicting feature names.")
            features = embedded
        if features is None:
            raise ValueError("ArgMax requires feature names for standalone matrices.")
        names = _validate_features(features)
        if values.shape[1] != len(names):
            raise ValueError("ArgMax feature count does not match matrix width.")
        return _argmax_frame(values, names)

    def translate_batch(
        self, inputs: Mapping[str, InputBatch], *,
        mode: TranslationMode, request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError("ArgMax supports translate mode only.")
        packet = single_input(inputs, name="ArgMax")
        info, matrix, keys = native_matrix_packet(packet, name="ArgMax")
        if self._features is None:
            raise OperatorError("ArgMax source feature schema was not established.")
        output = self.translate(matrix, features=self._features)
        return BatchResult(outputs={
            DEFAULT_OUTPUT_LABEL: {"keys": key_frame(info, keys), "data": output}
        })

    def handle_batch_result(
        self, result: BatchResult, *, batch_index: int,
        mode: TranslationMode, request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self, *, mode: TranslationMode, request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self, *, mode: TranslationMode, request: TranslationRequest,
    ) -> ArgMax:
        _ = request
        if mode != "translate":
            raise OperatorError("ArgMax workers support translate mode only.")
        worker = ArgMax()
        worker._features = self._features
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> ArgMax:
        if state:
            raise OperatorError("ArgMax has no scientific configuration.")
        return cls()

    def save_intermediate_state(
        self, intermediate_dir: Path, *, operator_id: str,
        mode: TranslationMode, route: RunRoute,
    ) -> None:
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "state.json").write_text(
            json.dumps({"operator_id": operator_id, "features": self._features}),
            encoding="utf-8",
        )

    @classmethod
    def load_intermediate_state(
        cls, intermediate_dir: Path, *, operator_id: str,
        mode: TranslationMode, route: RunRoute,
    ) -> ArgMax:
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls(operator_id=operator_id)
        raw = state.get("features")
        obj._features = None if raw is None else _validate_features(raw)
        return obj


def _validate_features(features: Sequence[str]) -> tuple[str, ...]:
    names = tuple(features)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("ArgMax requires at least one nonempty string feature name.")
    if len(set(names)) != len(names):
        raise ValueError("ArgMax requires unique feature names.")
    return names


def _argmax_frame(matrix: Any, names: tuple[str, ...]) -> pd.DataFrame:
    width = len(names)
    labels: list[str | None] = []
    scores: list[float] = []
    # Sorted name ranks give stable, column-order-independent tie resolution.
    order = np.argsort(np.asarray(names), kind="stable")
    ranks = np.empty(width, dtype=np.int64)
    ranks[order] = np.arange(width)

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.sort_indices()
        if csr.shape[1] != width:
            raise ArtifactError("ArgMax source width differs from its feature schema.")
        if not np.issubdtype(csr.dtype, np.number) or not np.all(np.isfinite(csr.data)):
            raise ArtifactError("ArgMax requires finite numeric matrix values.")
        for i in range(csr.shape[0]):
            lo, hi = csr.indptr[i], csr.indptr[i + 1]
            indices, data = csr.indices[lo:hi], csr.data[lo:hi]
            # Find the smallest named implicit-zero column without materializing
            # any dense row (important for large vocabulary matrices).
            implicit = None
            if len(indices) < width:
                present = set(indices.tolist())
                implicit = next((int(j) for j in order if int(j) not in present), None)
            best = float(np.max(data)) if len(data) else float("-inf")
            if implicit is not None:
                best = max(best, 0.0)
            if best == 0.0 and (not len(data) or np.all(data == 0)):
                labels.append(None)
                scores.append(0.0)
                continue
            candidates = indices[data == best]
            chosen = int(candidates[np.argmin(ranks[candidates])]) if len(candidates) else None
            if implicit is not None and best == 0.0 and (
                chosen is None or ranks[implicit] < ranks[chosen]
            ):
                chosen = implicit
            labels.append(names[chosen] if chosen is not None else None)
            scores.append(best)
    else:
        arr = np.asarray(matrix)
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ArtifactError("ArgMax requires a matrix matching its feature schema.")
        if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)):
            raise ArtifactError("ArgMax requires finite numeric matrix values.")
        for row in arr:
            if not np.any(row):
                labels.append(None)
                scores.append(0.0)
                continue
            maximum = float(np.max(row))
            candidates = np.flatnonzero(row == maximum)
            chosen = int(candidates[np.argmin(ranks[candidates])])
            labels.append(names[chosen])
            scores.append(maximum)

    return pd.DataFrame({
        "max_feature": pd.Series(labels, dtype="string"),
        "max_value": np.asarray(scores, dtype=np.float64),
    })
