"""Latent Dirichlet Allocation (LDA) for lexical count matrices."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import OperatorError, OperatorNotFittedError
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

LDA_TOPICS_LABEL = "topics"


class LDA(BaseTranslator):
    """Fit sklearn Latent Dirichlet Allocation to a count matrix.

    The initial fit emits ``output`` (document-topic distributions) and
    ``topics`` (topic-by-term component weights). Reusing the fitted operator on
    a compatible count matrix emits only document-topic distributions.
    """

    operation_type = "translate"

    def __init__(
        self,
        n_components: int = 10,
        *,
        learning_method: str = "batch",
        max_iter: int = 10,
        doc_topic_prior: float | None = None,
        topic_word_prior: float | None = None,
        learning_decay: float = 0.7,
        learning_offset: float = 10.0,
        batch_size: int = 128,
        evaluate_every: int = -1,
        perp_tol: float = 0.1,
        mean_change_tol: float = 0.001,
        max_doc_update_iter: int = 100,
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = int(n_components)
        self.learning_method = str(learning_method)
        self.max_iter = int(max_iter)
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_decay = float(learning_decay)
        self.learning_offset = float(learning_offset)
        self.batch_size = int(batch_size)
        self.evaluate_every = int(evaluate_every)
        self.perp_tol = float(perp_tol)
        self.mean_change_tol = float(mean_change_tol)
        self.max_doc_update_iter = int(max_doc_update_iter)
        self.random_state = random_state
        self.source_features_: tuple[str, ...] | None = None
        self._estimator = None

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    @property
    def supports_fit_translate(self) -> bool:
        return True

    @property
    def supports_parallel_translate(self) -> bool:
        return self.is_fitted

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return (
            route == "sequential"
            and mode in {"fit_translate", "translate"}
            or (mode == "translate" and route == "parallel")
        )

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        single_source(sources, name="LDA")
        specs: dict[str, OutputSpec] = {
            DEFAULT_OUTPUT_LABEL: OutputSpec(
                artifact_type="dense_matrix",
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
        }
        if not self.is_fitted:
            specs[LDA_TOPICS_LABEL] = OutputSpec(
                artifact_type="dense_matrix",
                lineage_mode="new_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
        return specs

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
                f"LDA does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        source = single_source(sources, name="LDA")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("LDA requires a sparse_matrix or dense_matrix source.")
        self.source_features_ = establish_or_validate_features(
            self.source_features_,
            source.get_data_columns(),
            fitted=self.is_fitted,
            name="LDA",
        )
        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="full_artifact" if mode == "fit_translate" else "batches",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=None
            if mode == "fit_translate"
            else (request.batch_size or 10_000),
            form="native",
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
        packet = single_input(inputs, name="LDA")
        info, matrix, key_columns = native_matrix_packet(packet, name="LDA")
        require_nonnegative(matrix, name="LDA", integer=True)
        outputs: dict[str, Any] = {}
        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError("fit_translate received an already fitted LDA.")
            estimator = self._make_estimator()
            doc_topics = estimator.fit_transform(matrix)
            self._estimator = estimator
            outputs[LDA_TOPICS_LABEL] = {
                "keys": pd.DataFrame(
                    {"topic_id": np.arange(self.n_components, dtype=np.int64)}
                ),
                "data": {
                    "values": np.asarray(estimator.components_, dtype=float),
                    "columns": list(self._require_features()),
                },
            }
        elif mode == "translate":
            doc_topics = self._require_estimator().transform(matrix)
        else:
            raise OperatorError(f"Unsupported LDA mode {mode!r}.")
        outputs[DEFAULT_OUTPUT_LABEL] = {
            "keys": key_frame(info, key_columns),
            "data": {
                "values": np.asarray(doc_topics, dtype=float),
                "columns": [f"topic_{i}" for i in range(self.n_components)],
            },
        }
        return BatchResult(outputs=outputs)

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
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> LDA:
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError("Parallel LDA workers require a fitted model.")
        worker = self.from_json_state(self.to_json_state())
        worker._estimator = clone_estimator(self._require_estimator())
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "learning_method": self.learning_method,
            "max_iter": self.max_iter,
            "doc_topic_prior": self.doc_topic_prior,
            "topic_word_prior": self.topic_word_prior,
            "learning_decay": self.learning_decay,
            "learning_offset": self.learning_offset,
            "batch_size": self.batch_size,
            "evaluate_every": self.evaluate_every,
            "perp_tol": self.perp_tol,
            "mean_change_tol": self.mean_change_tol,
            "max_doc_update_iter": self.max_doc_update_iter,
            "random_state": self.random_state,
            "source_features": None
            if self.source_features_ is None
            else list(self.source_features_),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> LDA:
        obj = cls(
            n_components=int(state.get("n_components", 10)),
            learning_method=str(state.get("learning_method", "batch")),
            max_iter=int(state.get("max_iter", 10)),
            doc_topic_prior=state.get("doc_topic_prior"),
            topic_word_prior=state.get("topic_word_prior"),
            learning_decay=float(state.get("learning_decay", 0.7)),
            learning_offset=float(state.get("learning_offset", 10.0)),
            batch_size=int(state.get("batch_size", 128)),
            evaluate_every=int(state.get("evaluate_every", -1)),
            perp_tol=float(state.get("perp_tol", 0.1)),
            mean_change_tol=float(state.get("mean_change_tol", 0.001)),
            max_doc_update_iter=int(state.get("max_doc_update_iter", 100)),
            random_state=state.get("random_state", 42),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            obj.source_features_ = tuple(str(value) for value in raw)
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.is_fitted:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "lda.pkl"
        dump_estimator(path, self._require_estimator())
        return {"estimator_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file") if manifest else None
        if filename is not None:
            if not isinstance(filename, str) or not filename:
                raise OperatorError("LDA asset manifest has invalid estimator_file.")
            self._estimator = load_estimator(assets_dir / filename)

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
        state["assets"] = dict(self.save_assets(intermediate_dir))
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
    ) -> LDA:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(state)
        obj.load_assets(intermediate_dir, state.get("assets", {}))
        obj.operator_id = operator_id
        return obj

    def _make_estimator(self):
        from sklearn.decomposition import LatentDirichletAllocation

        return LatentDirichletAllocation(
            n_components=self.n_components,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            learning_method=self.learning_method,
            learning_decay=self.learning_decay,
            learning_offset=self.learning_offset,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            evaluate_every=self.evaluate_every,
            perp_tol=self.perp_tol,
            mean_change_tol=self.mean_change_tol,
            max_doc_update_iter=self.max_doc_update_iter,
            random_state=self.random_state,
        )

    def _require_estimator(self):
        if self._estimator is None:
            raise OperatorNotFittedError("LDA has no fitted topic model.")
        return self._estimator

    def _require_features(self) -> tuple[str, ...]:
        if self.source_features_ is None:
            raise OperatorError("LDA has no source feature schema.")
        return self.source_features_
