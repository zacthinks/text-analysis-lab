"""UMAP dimensionality reduction for matrix artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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


class UMAP(BaseTranslator):
    """Reduce sparse or dense matrices with ``umap-learn``."""

    operation_type = "translate"

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int | None = 42,
        transform_seed: int = 42,
        retain_estimator: bool = True,
        operator_id: str | None = None,
        **umap_kwargs: Any,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if int(n_neighbors) <= 1:
            raise ValueError("n_neighbors must be greater than 1.")
        if float(min_dist) < 0:
            raise ValueError("min_dist must be non-negative.")
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = str(metric)
        self.random_state = random_state
        self.transform_seed = int(transform_seed)
        self.retain_estimator = bool(retain_estimator)
        self.umap_kwargs = dict(umap_kwargs)
        self.source_features_: tuple[str, ...] | None = None
        self._estimator = None
        self._fit_completed = False

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    @property
    def supports_fit_translate(self) -> bool:
        return not self._fit_completed

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
    ) -> OutputSpec:
        _ = request
        single_source(sources, name="UMAP")
        return OutputSpec(
            artifact_type="dense_matrix",
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
                f"UMAP does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        source = single_source(sources, name="UMAP")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError("UMAP requires a sparse_matrix or dense_matrix source.")
        self.source_features_ = establish_or_validate_features(
            self.source_features_,
            source.get_data_columns(),
            fitted=self.is_fitted,
            name="UMAP",
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
        packet = single_input(inputs, name="UMAP")
        info, matrix, key_columns = native_matrix_packet(packet, name="UMAP")
        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError("fit_translate received an already fitted UMAP.")
            estimator = self._make_estimator()
            values = estimator.fit_transform(matrix)
            self._fit_completed = True
            if self.retain_estimator:
                self._estimator = estimator
        elif mode == "translate":
            values = self._require_estimator().transform(matrix)
        else:
            raise OperatorError(f"Unsupported UMAP mode {mode!r}.")
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": key_frame(info, key_columns),
                    "data": {
                        "values": np.asarray(values, dtype=float),
                        "columns": [f"umap_{i}" for i in range(self.n_components)],
                    },
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
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> UMAP:
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError(
                "Parallel UMAP workers require a fitted model."
            )
        worker = self.from_json_state(self.to_json_state())
        worker._estimator = clone_estimator(self._require_estimator())
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "transform_seed": self.transform_seed,
            "retain_estimator": self.retain_estimator,
            "fit_completed": self._fit_completed,
            "umap_kwargs": self.umap_kwargs,
            "source_features": None
            if self.source_features_ is None
            else list(self.source_features_),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> UMAP:
        raw_kwargs = state.get("umap_kwargs", {})
        kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {}
        obj = cls(
            n_components=int(state.get("n_components", 2)),
            n_neighbors=int(state.get("n_neighbors", 15)),
            min_dist=float(state.get("min_dist", 0.1)),
            metric=str(state.get("metric", "euclidean")),
            random_state=state.get("random_state", 42),
            transform_seed=int(state.get("transform_seed", 42)),
            retain_estimator=bool(state.get("retain_estimator", True)),
            **kwargs,
        )
        obj._fit_completed = bool(
            state.get("fit_completed", state.get("is_fitted", False))
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            obj.source_features_ = tuple(str(value) for value in raw)
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.is_fitted:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "umap.pkl"
        dump_estimator(path, self._require_estimator())
        return {"estimator_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file") if manifest else None
        if filename is not None:
            if not isinstance(filename, str) or not filename:
                raise OperatorError("UMAP asset manifest has invalid estimator_file.")
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
    ) -> UMAP:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(state)
        obj.load_assets(intermediate_dir, state.get("assets", {}))
        obj.operator_id = operator_id
        return obj

    def _make_estimator(self):
        try:
            import umap
        except ImportError as exc:
            raise OperatorError(
                "UMAP requires the declared 'umap-learn' runtime dependency. Re-sync the TeAL environment."
            ) from exc
        return umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            transform_seed=self.transform_seed,
            **self.umap_kwargs,
        )

    def _require_estimator(self):
        if self._estimator is None:
            raise OperatorNotFittedError("UMAP has no fitted reducer.")
        return self._estimator
