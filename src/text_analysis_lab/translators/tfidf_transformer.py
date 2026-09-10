"""TF-IDF weighting for matrix artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from text_analysis_lab.core.errors import OperatorError, OperatorNotFittedError
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
    establish_or_validate_features,
    key_frame,
    load_estimator,
    native_matrix_packet,
    require_nonnegative,
    single_input,
    single_source,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class TfidfTransformer(BaseTranslator):
    """Reweight a count-like matrix with scikit-learn TF-IDF.

    Fitting learns corpus-level inverse-document-frequency weights. By default
    ``norm=None`` so TF-IDF weighting and vector normalization remain separate
    TeAL operations. Set ``norm='l1'`` or ``norm='l2'`` for sklearn-compatible
    combined weighting + row normalization.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        norm: str | None = None,
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if norm not in {None, "l1", "l2"}:
            raise ValueError("norm must be None, 'l1', or 'l2'.")
        self.norm = norm
        self.use_idf = bool(use_idf)
        self.smooth_idf = bool(smooth_idf)
        self.sublinear_tf = bool(sublinear_tf)
        self.source_features_: tuple[str, ...] | None = None
        self._transformer = None

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return self._transformer is not None

    @property
    def supports_fit_translate(self) -> bool:
        return True

    @property
    def supports_parallel_translate(self) -> bool:
        return self.is_fitted

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return route == "sequential" and mode in {"fit_translate", "translate"} or (
            mode == "translate" and route == "parallel"
        )

    def output_specs(self, *, sources: Mapping[str, "BaseArtifact"], request: TranslationRequest) -> OutputSpec:
        _ = request
        single_source(sources, name="TfidfTransformer")
        return OutputSpec(
            artifact_type="sparse_matrix",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def validate_operation_params(self, params: Mapping[str, Any], *, sources: Mapping[str, "BaseArtifact"], mode: TranslationMode) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(f"TfidfTransformer does not accept operation parameters; got {sorted(params)}.")
        return {}

    def input_request(self, *, sources: Mapping[str, "BaseArtifact"], mode: TranslationMode, request: TranslationRequest) -> SourceRequest:
        source = single_source(sources, name="TfidfTransformer")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("TfidfTransformer requires a sparse_matrix or dense_matrix source.")
        self.source_features_ = establish_or_validate_features(
            self.source_features_, source.get_data_columns(), fitted=self.is_fitted, name="TfidfTransformer"
        )
        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="full_artifact" if mode == "fit_translate" else "batches",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=None if mode == "fit_translate" else (request.batch_size or 10_000),
            form="native",
            metadata_mode="none",
            include_position=False,
        )

    def translate_batch(self, inputs: Mapping[str, InputBatch], *, mode: TranslationMode, request: TranslationRequest) -> BatchResult:
        _ = request
        packet = single_input(inputs, name="TfidfTransformer")
        info, matrix, key_columns = native_matrix_packet(packet, name="TfidfTransformer")
        require_nonnegative(matrix, name="TfidfTransformer")
        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError("fit_translate received an already fitted TfidfTransformer.")
            transformer = self._make_transformer()
            values = transformer.fit_transform(matrix)
            self._transformer = transformer
        elif mode == "translate":
            values = self._require_transformer().transform(matrix)
        else:  # pragma: no cover
            raise OperatorError(f"Unsupported TfidfTransformer mode {mode!r}.")
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {
            "keys": key_frame(info, key_columns),
            "data": {"values": values.tocsr(), "columns": list(self._require_features())},
        }})

    def transform_external_matrix(
        self,
        matrix: Any,
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ):
        """Apply the frozen IDF weighting to new matrix rows in memory."""
        _ = query, params
        require_nonnegative(matrix, name="TfidfTransformer")
        return self._require_transformer().transform(matrix).tocsr()

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "matrix" and self.is_fitted

    def handle_batch_result(self, result: BatchResult, *, batch_index: int, mode: TranslationMode, request: TranslationRequest) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(self, *, mode: TranslationMode, request: TranslationRequest) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(self, *, mode: TranslationMode, request: TranslationRequest) -> "TfidfTransformer":
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError("Parallel TfidfTransformer workers require fitted IDF state.")
        worker = TfidfTransformer(norm=self.norm, use_idf=self.use_idf, smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf)
        worker.source_features_ = self._require_features()
        worker._transformer = clone_estimator(self._require_transformer())
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "source_features": None if self.source_features_ is None else list(self.source_features_),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "TfidfTransformer":
        obj = cls(
            norm=cast(str | None, state.get("norm")),
            use_idf=bool(state.get("use_idf", True)),
            smooth_idf=bool(state.get("smooth_idf", True)),
            sublinear_tf=bool(state.get("sublinear_tf", False)),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            obj.source_features_ = tuple(str(value) for value in raw)
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.is_fitted:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "tfidf_transformer.pkl"
        dump_estimator(path, self._require_transformer())
        return {"estimator_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file") if manifest else None
        if filename is None:
            return
        if not isinstance(filename, str) or not filename:
            raise OperatorError("TfidfTransformer asset manifest has invalid estimator_file.")
        self._transformer = load_estimator(assets_dir / filename)

    def save_intermediate_state(self, intermediate_dir: Path, *, operator_id: str, mode: TranslationMode, route: RunRoute) -> None:
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        state = self.to_json_state()
        state["assets"] = dict(self.save_assets(intermediate_dir))
        state["operator_id"] = operator_id
        (intermediate_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_intermediate_state(cls, intermediate_dir: Path, *, operator_id: str, mode: TranslationMode, route: RunRoute) -> "TfidfTransformer":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls.from_json_state(state)
        obj.load_assets(intermediate_dir, state.get("assets", {}))
        obj.operator_id = operator_id
        return obj

    def _make_transformer(self):
        from sklearn.feature_extraction.text import TfidfTransformer as SklearnTfidfTransformer
        return SklearnTfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

    def _require_transformer(self):
        if self._transformer is None:
            raise OperatorNotFittedError("TfidfTransformer has no fitted IDF state.")
        return self._transformer

    def _require_features(self) -> tuple[str, ...]:
        if self.source_features_ is None:
            raise OperatorError("TfidfTransformer has no source feature schema.")
        return self.source_features_
