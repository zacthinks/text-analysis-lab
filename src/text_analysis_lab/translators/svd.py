"""Truncated singular-value decomposition (SVD/LSA) for matrix artifacts."""

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
    single_input,
    single_source,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

SVD_COMPONENTS_LABEL = "components"


class SVD(BaseTranslator):
    """Reduce sparse or dense matrices with sklearn ``TruncatedSVD``.

    The initial fit emits two durable artifacts: ``output`` contains source-row
    coordinates in component space, and ``components`` contains one row per
    component with loadings over the original features. Reusing the fitted
    operator on later compatible matrices emits only ``output``.
    """

    operation_type = "translate"

    def __init__(
        self,
        n_components: int = 2,
        *,
        algorithm: str = "randomized",
        n_iter: int = 5,
        random_state: int | None = 42,
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = int(n_components)
        self.algorithm = str(algorithm)
        self.n_iter = int(n_iter)
        self.random_state = random_state
        self.n_oversamples = int(n_oversamples)
        self.power_iteration_normalizer = str(power_iteration_normalizer)
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
        single_source(sources, name="SVD")
        specs: dict[str, OutputSpec] = {
            DEFAULT_OUTPUT_LABEL: OutputSpec(
                artifact_type="dense_matrix",
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
        }
        if not self.is_fitted:
            specs[SVD_COMPONENTS_LABEL] = OutputSpec(
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
                f"SVD does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        source = single_source(sources, name="SVD")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("SVD requires a sparse_matrix or dense_matrix source.")
        self.source_features_ = establish_or_validate_features(
            self.source_features_,
            source.get_data_columns(),
            fitted=self.is_fitted,
            name="SVD",
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
        packet = single_input(inputs, name="SVD")
        info, matrix, key_columns = native_matrix_packet(packet, name="SVD")
        outputs: dict[str, Any] = {}
        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError("fit_translate received an already fitted SVD.")
            estimator = self._make_estimator()
            reduced = estimator.fit_transform(matrix)
            self._estimator = estimator
            components = np.asarray(estimator.components_, dtype=float)
            component_meta = pd.DataFrame(
                {
                    "explained_variance": np.asarray(
                        estimator.explained_variance_, dtype=float
                    ),
                    "explained_variance_ratio": np.asarray(
                        estimator.explained_variance_ratio_, dtype=float
                    ),
                    "singular_value": np.asarray(
                        estimator.singular_values_, dtype=float
                    ),
                }
            )
            outputs[SVD_COMPONENTS_LABEL] = {
                "keys": pd.DataFrame(
                    {"component_id": np.arange(components.shape[0], dtype=np.int64)}
                ),
                "metadata": component_meta,
                "data": {
                    "values": components,
                    "columns": list(self._require_features()),
                },
            }
        elif mode == "translate":
            reduced = self._require_estimator().transform(matrix)
        else:  # pragma: no cover
            raise OperatorError(f"Unsupported SVD mode {mode!r}.")
        outputs[DEFAULT_OUTPUT_LABEL] = {
            "keys": key_frame(info, key_columns),
            "data": {
                "values": np.asarray(reduced),
                "columns": _component_columns(self.n_components),
            },
        }
        return BatchResult(outputs=outputs)

    def transform_external_matrix(
        self,
        matrix: Any,
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Project new matrix rows through the frozen decomposition in memory."""
        _ = query, params
        return np.asarray(self._require_estimator().transform(matrix))

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
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> SVD:
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError("Parallel SVD workers require a fitted model.")
        worker = self.from_json_state(self.to_json_state())
        worker._estimator = clone_estimator(self._require_estimator())
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "algorithm": self.algorithm,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "n_oversamples": self.n_oversamples,
            "power_iteration_normalizer": self.power_iteration_normalizer,
            "source_features": None
            if self.source_features_ is None
            else list(self.source_features_),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SVD:
        obj = cls(
            n_components=int(state.get("n_components", 2)),
            algorithm=str(state.get("algorithm", "randomized")),
            n_iter=int(state.get("n_iter", 5)),
            random_state=state.get("random_state", 42),
            n_oversamples=int(state.get("n_oversamples", 10)),
            power_iteration_normalizer=str(
                state.get("power_iteration_normalizer", "auto")
            ),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            obj.source_features_ = tuple(str(value) for value in raw)
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.is_fitted:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "svd.pkl"
        dump_estimator(path, self._require_estimator())
        return {"estimator_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file") if manifest else None
        if filename is not None:
            if not isinstance(filename, str) or not filename:
                raise OperatorError("SVD asset manifest has invalid estimator_file.")
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
    ) -> SVD:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(state)
        obj.load_assets(intermediate_dir, state.get("assets", {}))
        obj.operator_id = operator_id
        return obj

    def _make_estimator(self):
        from sklearn.decomposition import TruncatedSVD

        return TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            random_state=self.random_state,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
        )

    def _require_estimator(self):
        if self._estimator is None:
            raise OperatorNotFittedError("SVD has no fitted decomposition.")
        return self._estimator

    def _require_features(self) -> tuple[str, ...]:
        if self.source_features_ is None:
            raise OperatorError("SVD has no source feature schema.")
        return self.source_features_


def _component_columns(n_components: int) -> list[str]:
    return [f"component_{index}" for index in range(int(n_components))]


LSA = SVD
LatentSemanticAnalysis = SVD
