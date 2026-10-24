"""Lazy numerical feature matrix translator with frozen categorical encodings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.query_scope import QueryScope
from text_analysis_lab.core.operator import BaseTranslator, BatchResult
from text_analysis_lab.translators._feature_matrix_encoding import FeatureMatrixEncoding
from text_analysis_lab.translators._feature_matrix_provider import render_feature_matrix


def _selected(available: Sequence[str], request: Any) -> list[str]:
    if request is False or request is None:
        return []
    chosen = list(available) if request is True else (
        [request] if isinstance(request, str) else list(request)
    )
    missing = [name for name in chosen if name not in available]
    if missing:
        raise ArtifactError(f"FeatureMatrix fields not found: {missing}")
    return chosen


class FeatureMatrix(BaseTranslator):
    """Create a recipe-only matrix through project.translate()."""

    operation_type = "translate"

    def __init__(
        self, *,
        data_columns: Any = True,
        metadata_columns: Any = False,
        key_columns: Any = False,
        encodings: Mapping[str, Any] | None = None,
        matrix_type: str = "auto",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if matrix_type not in {"auto", "dense", "sparse"}:
            raise ValueError("matrix_type must be 'auto', 'dense' or 'sparse'.")
        self.data_columns = data_columns
        self.metadata_columns = metadata_columns
        self.key_columns = key_columns
        self.encodings = dict(encodings or {})
        self.matrix_type = matrix_type
        self._encoder = FeatureMatrixEncoding(self.encodings)
        self.matrix_columns: list[str] = []
        self.scalar_data_columns: list[str] = []
        self.scalar_metadata_columns: list[str] = []
        self.scalar_key_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.sparse_output = False

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return bool(self.feature_columns)

    @property
    def supports_fit_translate(self) -> bool:
        return True

    def _resolve(self, source: Any) -> tuple[list[str], list[str], list[str], list[str]]:
        matrix = source.artifact_type.value in {"sparse_matrix", "dense_matrix"}
        matrix_cols = _selected(source.get_data_columns(), self.data_columns) if matrix else []
        data_cols = [] if matrix else _selected(QueryScope(source, metadata_mode="full").columns()["data"], self.data_columns)
        key_cols = _selected(source.primary_key, self.key_columns)
        scope = QueryScope(source, metadata_mode="full")
        available = scope.columns()["metadata"]
        meta_names = [
            value["output_name"] if isinstance(value, dict) else value
            for value in available
        ]
        meta_cols = _selected(meta_names, self.metadata_columns)
        return matrix_cols, data_cols, meta_cols, key_cols

    def fit_source(self, source: Any, *, batch_size: int = 10_000) -> None:
        if self.is_fitted or self.is_frozen:
            raise OperatorError("FeatureMatrix is already fitted.")
        mat, data, meta, key = self._resolve(source)
        self.matrix_columns = mat
        self.scalar_data_columns = data
        self.scalar_metadata_columns = meta
        self.scalar_key_columns = key
        scope = QueryScope(source, metadata_mode="full")
        batches = scope.query(
            data_columns=data if data else False,
            metadata_columns=meta if meta else False,
            key_columns=key if key else False,
            form="table", iter_batches=True, batch_size=batch_size,
        )
        self._encoder.fit_batches(batches, columns=[*key, *data, *meta])
        self.feature_columns = [*mat, *self._encoder.columns]
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ArtifactError("FeatureMatrix needs unique nonempty feature columns.")
        self.sparse_output = (
            source.artifact_type.value == "sparse_matrix"
            if self.matrix_type == "auto" else self.matrix_type == "sparse"
        )

    def validate_source(self, source: Any) -> None:
        mat, data, meta, key = self._resolve(source)
        if (mat, data, meta, key) != (
            self.matrix_columns, self.scalar_data_columns,
            self.scalar_metadata_columns, self.scalar_key_columns,
        ):
            raise ArtifactError("FeatureMatrix source lacks frozen selected feature columns.")

    def render(self, source: Any, positions: Sequence[int], *, selected_indices: Sequence[int] | None = None) -> Any:
        return render_feature_matrix(
            source, positions=positions,
            matrix_columns=self.matrix_columns if self.matrix_columns else False,
            data_columns=self.scalar_data_columns if self.scalar_data_columns else False,
            metadata_columns=self.scalar_metadata_columns if self.scalar_metadata_columns else False,
            key_columns=self.scalar_key_columns if self.scalar_key_columns else False,
            encoder=self._encoder, sparse_output=self.sparse_output,
            output_columns=self.feature_columns,
            selected_indices=selected_indices,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "data_columns": self.data_columns, "metadata_columns": self.metadata_columns,
            "key_columns": self.key_columns, "encodings": self.encodings,
            "matrix_type": self.matrix_type,
            "matrix_columns": self.matrix_columns,
            "scalar_data_columns": self.scalar_data_columns,
            "scalar_metadata_columns": self.scalar_metadata_columns,
            "scalar_key_columns": self.scalar_key_columns,
            "feature_columns": self.feature_columns,
            "sparse_output": self.sparse_output,
            "encoder": self._encoder.to_state() if self._encoder.is_fitted else None,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> FeatureMatrix:
        obj = cls(
            data_columns=state["data_columns"],
            metadata_columns=state["metadata_columns"],
            key_columns=state["key_columns"],
            encodings=state["encodings"],
            matrix_type=state["matrix_type"],
        )
        for attr in ("matrix_columns", "scalar_data_columns",
                     "scalar_metadata_columns", "scalar_key_columns", "feature_columns"):
            setattr(obj, attr, list(state[attr]))
        obj.sparse_output = bool(state["sparse_output"])
        if state.get("encoder") is not None:
            obj._encoder = FeatureMatrixEncoding.from_state(state["encoder"])
        return obj

    def translate(self, source: Any) -> Any:
        """Apply a fitted FeatureMatrix recipe to a pandas DataFrame.

        Artifact-backed matrix composition uses Project.translate(), not this
        standalone tabular convenience method.
        """
        if not self.is_fitted:
            raise OperatorError("FeatureMatrix must be fitted before standalone translation.")
        if not isinstance(source, pd.DataFrame):
            raise TypeError("FeatureMatrix.translate expects a pandas DataFrame.")
        selected = [
            *self.scalar_key_columns,
            *self.scalar_data_columns,
            *self.scalar_metadata_columns,
        ]
        missing = [name for name in selected if name not in source.columns]
        if missing:
            raise ArtifactError(f"FeatureMatrix input is missing columns: {missing}")
        if self.matrix_columns:
            raise OperatorError(
                "Standalone FeatureMatrix.translate cannot assemble an artifact-backed "
                "matrix; use Project.translate with an artifact."
            )
        return self._encoder.transform(
            source.loc[:, selected], sparse_output=self.sparse_output
        )

    def input_request(self, *, sources, mode, request):
        raise OperatorError("FeatureMatrix uses the recipe-only Project.translate path.")

    def translate_batch(self, inputs, *, mode, request) -> BatchResult:
        raise OperatorError("FeatureMatrix writes no numerical translation batches.")

    def handle_batch_result(self, result, *, batch_index, mode, request):
        return None

    def finalize_translation(self, *, mode, request):
        return None

    def output_specs(self, *, sources, request):
        raise OperatorError("FeatureMatrix output is registered by its lazy output path.")
