"""Row- or column-wise L1/L2 normalization for matrix artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize as sklearn_normalize

from text_analysis_lab.core.errors import OperatorError
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
from text_analysis_lab.translators._matrix_transform_utils import key_frame, native_matrix_packet, single_input, single_source

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class MatrixNormalizer(BaseTranslator):
    """Normalize matrix rows or columns to unit L1/L2 norm."""

    operation_type = "translate"

    def __init__(self, *, axis: str | int = "rows", norm: str = "l2", operator_id: str | None = None) -> None:
        super().__init__(operator_id=operator_id)
        self.axis = _normalize_axis(axis)
        if norm not in {"l1", "l2"}:
            raise ValueError("norm must be 'l1' or 'l2'.")
        self.norm = norm
        self._source_type: str | None = None
        self._columns: tuple[str, ...] | None = None

    @property
    def supports_parallel_translate(self) -> bool:
        return self.axis == "rows"

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        if mode != "translate":
            return False
        return route == "sequential" or (self.axis == "rows" and route == "parallel")

    def output_specs(self, *, sources: Mapping[str, "BaseArtifact"], request: TranslationRequest) -> OutputSpec:
        _ = request
        source = single_source(sources, name="MatrixNormalizer")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("MatrixNormalizer requires a sparse_matrix or dense_matrix source.")
        return OutputSpec(
            artifact_type=source.artifact_type.value,
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def validate_operation_params(self, params: Mapping[str, Any], *, sources: Mapping[str, "BaseArtifact"], mode: TranslationMode) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(f"MatrixNormalizer does not accept operation parameters; got {sorted(params)}.")
        return {}

    def input_request(self, *, sources: Mapping[str, "BaseArtifact"], mode: TranslationMode, request: TranslationRequest) -> SourceRequest:
        _ = mode
        source = single_source(sources, name="MatrixNormalizer")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("MatrixNormalizer requires a sparse_matrix or dense_matrix source.")
        self._source_type = source.artifact_type.value
        self._columns = tuple(str(value) for value in source.get_data_columns())
        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="batches" if self.axis == "rows" else "full_artifact",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=(request.batch_size or 10_000) if self.axis == "rows" else None,
            form="native",
            metadata_mode="none",
            include_position=False,
        )

    def translate_batch(self, inputs: Mapping[str, InputBatch], *, mode: TranslationMode, request: TranslationRequest) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported MatrixNormalizer mode {mode!r}.")
        packet = single_input(inputs, name="MatrixNormalizer")
        info, matrix, key_columns = native_matrix_packet(packet, name="MatrixNormalizer")
        normalized = sklearn_normalize(matrix, norm=self.norm, axis=1 if self.axis == "rows" else 0, copy=True)
        if self._source_type == "sparse_matrix":
            values = normalized.tocsr() if sparse.issparse(normalized) else sparse.csr_matrix(normalized)
        else:
            values = normalized.toarray() if sparse.issparse(normalized) else np.asarray(normalized)
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {
            "keys": key_frame(info, key_columns),
            "data": {"values": values, "columns": list(self._columns or ())},
        }})

    def transform_external_matrix(
        self,
        matrix: Any,
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ):
        """Normalize new rows when this operator was configured row-wise."""
        _ = query, params
        if self.axis != "rows":
            raise OperatorError(
                "Column-wise MatrixNormalizer cannot replay a single new document: "
                "its normalization depends on the fitted corpus rows."
            )
        values = sklearn_normalize(matrix, norm=self.norm, axis=1, copy=True)
        if sparse.issparse(matrix):
            return values.tocsr() if sparse.issparse(values) else sparse.csr_matrix(values)
        return values.toarray() if sparse.issparse(values) else np.asarray(values)

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "matrix" and self.axis == "rows"

    def handle_batch_result(self, result: BatchResult, *, batch_index: int, mode: TranslationMode, request: TranslationRequest) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(self, *, mode: TranslationMode, request: TranslationRequest) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(self, *, mode: TranslationMode, request: TranslationRequest) -> "MatrixNormalizer":
        _ = request
        if mode != "translate" or self.axis != "rows":
            raise OperatorError("Parallel MatrixNormalizer workers are available only for row-wise normalization.")
        worker = MatrixNormalizer(axis=self.axis, norm=self.norm)
        worker._source_type = self._source_type
        worker._columns = self._columns
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {"axis": self.axis, "norm": self.norm}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "MatrixNormalizer":
        return cls(axis=str(state.get("axis", "rows")), norm=str(state.get("norm", "l2")))

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        """Checkpoint the operation bindings required by resumed batches."""
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        state = self.to_json_state()
        state.update(
            {
                "operator_id": operator_id,
                "source_type": self._source_type,
                "columns": None if self._columns is None else list(self._columns),
            }
        )
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
    ) -> "MatrixNormalizer":
        """Restore configuration plus the source schema bound to the operation."""
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise OperatorError("MatrixNormalizer intermediate state must be a mapping.")

        obj = cls.from_json_state(state)
        raw_source_type = state.get("source_type")
        if raw_source_type not in {None, "sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "MatrixNormalizer intermediate state has an invalid source_type."
            )
        obj._source_type = raw_source_type

        raw_columns = state.get("columns")
        if raw_columns is not None:
            if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
                raise OperatorError(
                    "MatrixNormalizer intermediate state columns must be a sequence."
                )
            obj._columns = tuple(str(value) for value in raw_columns)
        obj.operator_id = operator_id
        return obj


def _normalize_axis(value: str | int) -> str:
    if value in {"rows", "row", 1}:
        return "rows"
    if value in {"columns", "column", "cols", "col", 0}:
        return "columns"
    raise ValueError("axis must be 'rows'/1 or 'columns'/0.")
