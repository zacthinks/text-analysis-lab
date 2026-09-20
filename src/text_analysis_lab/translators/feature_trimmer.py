"""Post-hoc feature trimming as a lazy positional matrix view."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse

from text_analysis_lab.core.errors import (
    ArtifactError,
    OperatorError,
    OperatorNotFittedError,
)
from text_analysis_lab.core.feature_subset import slice_matrix_features
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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    key_frame,
    native_matrix_packet,
    single_input,
    single_source,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class FeatureTrimmer(BaseTranslator):
    """Fit and replay a frozen positional feature subset.

    ``min_df`` and ``max_df`` are interpreted like scikit-learn document-
    frequency thresholds: integers are row counts and floats are fractions of
    corpus rows. ``max_features`` is applied *after* DF filtering by descending
    corpus feature sum (term frequency for ordinary count DTMs), with original
    feature position as the deterministic tie-breaker. Retained positions stay
    in their original source order.

    The fitted state contains only the source width and retained integer
    positions. Feature labels are not used to establish or validate identity.
    The caller/upstream frozen vectorizer is responsible for replaying matrices
    in the same ordered feature space.

    Artifact outputs are lazy feature views: no trimmed matrix is rewritten.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        min_df: float = 1,
        max_df: float = 1.0,
        max_features: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.min_df = _validate_df_threshold(min_df, name="min_df")
        self.max_df = _validate_df_threshold(max_df, name="max_df")
        if max_features is not None and (
            isinstance(max_features, bool) or int(max_features) <= 0
        ):
            raise ValueError("max_features must be a positive integer or None.")
        self.max_features = None if max_features is None else int(max_features)
        self.source_width_: int | None = None
        self.kept_indices_: tuple[int, ...] | None = None

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return self.source_width_ is not None and self.kept_indices_ is not None

    @property
    def supports_fit_translate(self) -> bool:
        return True

    @property
    def supports_parallel_translate(self) -> bool:
        return self.is_fitted

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        if mode == "fit_translate":
            return route == "sequential"
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = single_source(sources, name="FeatureTrimmer")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "FeatureTrimmer requires a sparse_matrix or dense_matrix source."
            )
        return OutputSpec(
            artifact_type=source.artifact_type.value,
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
        _ = sources, mode
        if params:
            raise OperatorError(
                f"FeatureTrimmer does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        source = single_source(sources, name="FeatureTrimmer")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "FeatureTrimmer requires a sparse_matrix or dense_matrix source."
            )
        observed_width = len(source.get_data_columns())
        if self.source_width_ is None:
            self.source_width_ = int(observed_width)
        elif int(observed_width) != int(self.source_width_):
            raise OperatorError(
                "FeatureTrimmer replay requires the fitted source width "
                f"{self.source_width_}; got {observed_width}. TeAL does not infer "
                "feature alignment from feature labels."
            )

        if mode == "fit_translate":
            return SourceRequest(
                artifact_type=("sparse_matrix", "dense_matrix"),
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=True, metadata=False),
                batch_size=None,
                form="native",
                metadata_mode="none",
                include_position=False,
            )

        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="batches",
            columns=ColumnRequest(keys=True, data=False, metadata=False),
            batch_size=request.batch_size or 10_000,
            form="table",
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
        packet = single_input(inputs, name="FeatureTrimmer")

        if mode == "fit_translate":
            info, matrix, key_columns = native_matrix_packet(
                packet, name="FeatureTrimmer"
            )
            source_width = self._require_source_width()
            if int(matrix.shape[1]) != source_width:
                raise ArtifactError(
                    "FeatureTrimmer matrix width does not match its fitted/source width: "
                    f"{matrix.shape[1]} != {source_width}."
                )
            if self.kept_indices_ is not None:
                raise OperatorError(
                    "fit_translate received an already fitted FeatureTrimmer."
                )
            self.kept_indices_ = _fit_feature_mask(
                matrix,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
            )
            keys = key_frame(info, key_columns)
        elif mode == "translate":
            if not self.is_fitted:
                raise OperatorNotFittedError(
                    "FeatureTrimmer must be fitted before translate replay."
                )
            frame = packet.data
            if not hasattr(frame, "loc"):
                raise ArtifactError(
                    "FeatureTrimmer replay expected a tabular key packet."
                )
            keys = frame.loc[:, list(packet.primary_key)].reset_index(drop=True)
        else:  # pragma: no cover - runner validates mode
            raise OperatorError(f"Unsupported FeatureTrimmer mode {mode!r}.")

        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "feature_indices": list(self._require_kept_indices()),
                }
            }
        )

    def transform_external_matrix(
        self,
        matrix: Any,
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Apply the exact frozen positional feature subset to new matrix rows."""
        _ = query, params
        source_width = self._require_source_width()
        shape = getattr(matrix, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[1]) != source_width:
            raise OperatorError(
                "FeatureTrimmer replay requires the fitted source width "
                f"{source_width}; got shape={shape!r}."
            )
        return slice_matrix_features(matrix, self._require_kept_indices())

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "matrix" and self.is_fitted

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
    ) -> FeatureTrimmer:
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError(
                "Parallel FeatureTrimmer workers require a fitted feature mask."
            )
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "source_width": self.source_width_,
            "kept_indices": (
                None if self.kept_indices_ is None else list(self.kept_indices_)
            ),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> FeatureTrimmer:
        obj = cls(
            min_df=cast(int | float, state.get("min_df", 1)),
            max_df=cast(int | float, state.get("max_df", 1.0)),
            max_features=(
                None
                if state.get("max_features") is None
                else int(state["max_features"])
            ),
        )
        raw_width = state.get("source_width")
        if raw_width is not None:
            obj.source_width_ = int(raw_width)
        else:
            # Backward compatibility for snapshots created before positional-only
            # feature trimming. Old source feature labels establish only width.
            raw_features = state.get("source_features")
            if isinstance(raw_features, Sequence) and not isinstance(
                raw_features, (str, bytes)
            ):
                obj.source_width_ = len(raw_features)
        raw_indices = state.get("kept_indices")
        if isinstance(raw_indices, Sequence) and not isinstance(
            raw_indices, (str, bytes)
        ):
            obj.kept_indices_ = tuple(int(value) for value in raw_indices)
        return obj

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
        state = self.to_json_state()
        state["operator_id"] = operator_id
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
    ) -> FeatureTrimmer:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        if not isinstance(state, Mapping):
            raise OperatorError("FeatureTrimmer intermediate state must be a mapping.")
        obj = cls.from_json_state(state)
        obj.operator_id = operator_id
        return obj

    def _require_source_width(self) -> int:
        if self.source_width_ is None:
            raise OperatorNotFittedError(
                "FeatureTrimmer has no bound source feature width."
            )
        return int(self.source_width_)

    def _require_kept_indices(self) -> tuple[int, ...]:
        if self.kept_indices_ is None:
            raise OperatorNotFittedError("FeatureTrimmer has no fitted feature mask.")
        return self.kept_indices_


def _validate_df_threshold(value: float, *, name: str) -> int | float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be an integer count or float fraction, not bool."
        )
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} integer thresholds must be >= 1.")
        return int(value)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 1.0:
        raise ValueError(f"{name} float thresholds must satisfy 0 < {name} <= 1.")
    return numeric


def _resolve_min_df(value: float, n_rows: int) -> int:
    return (
        int(value) if isinstance(value, int) else int(math.ceil(float(value) * n_rows))
    )


def _resolve_max_df(value: float, n_rows: int) -> int:
    return (
        int(value) if isinstance(value, int) else int(math.floor(float(value) * n_rows))
    )


def _fit_feature_mask(
    matrix: Any,
    *,
    min_df: float,
    max_df: float,
    max_features: int | None,
) -> tuple[int, ...]:
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ArtifactError("FeatureTrimmer requires a two-dimensional matrix.")
    n_rows, n_features = (int(value) for value in shape)
    if n_rows <= 0:
        raise ArtifactError("FeatureTrimmer cannot fit on an empty matrix.")
    if n_features <= 0:
        raise ArtifactError("FeatureTrimmer cannot fit on an empty feature axis.")

    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix, copy=True)
        csr.eliminate_zeros()
        document_frequency = np.asarray(csr.getnnz(axis=0), dtype=np.int64).reshape(-1)
        feature_sum = np.asarray(csr.sum(axis=0), dtype=float).reshape(-1)
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ArtifactError("FeatureTrimmer requires a two-dimensional matrix.")
        document_frequency = np.count_nonzero(dense, axis=0).astype(
            np.int64, copy=False
        )
        feature_sum = np.asarray(np.sum(dense, axis=0, dtype=float)).reshape(-1)

    min_count = _resolve_min_df(min_df, n_rows)
    max_count = _resolve_max_df(max_df, n_rows)
    if max_count < min_count:
        raise OperatorError(
            f"FeatureTrimmer max_df corresponds to {max_count} rows, below min_df={min_count}."
        )
    candidate = np.flatnonzero(
        (document_frequency >= min_count) & (document_frequency <= max_count)
    ).astype(np.int64, copy=False)
    if candidate.size == 0:
        raise ArtifactError(
            "FeatureTrimmer retained no features after min_df/max_df filtering."
        )

    if max_features is not None and candidate.size > max_features:
        ranked = sorted(
            (int(index) for index in candidate.tolist()),
            key=lambda index: (-float(feature_sum[index]), index),
        )[:max_features]
        candidate = np.asarray(sorted(ranked), dtype=np.int64)

    return tuple(int(value) for value in candidate.tolist())
