"""UMAP dimensionality reduction for matrix artifacts."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    dump_estimator,
    establish_or_validate_features,
    key_frame,
    load_estimator,
    native_matrix_packet,
    single_input,
    single_source,
)
from text_analysis_lab.translators._umap_persistence import (
    dump_compact_umap,
    load_compact_umap,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


UMAPReuse = Literal["none", "recompute", "stored"]
UMAPStorage = Literal["compact", "native"]
_VALID_REUSE = {"none", "recompute", "stored"}
_VALID_STORAGE = {"compact", "native"}


class UMAP(BaseTranslator):
    """Reduce sparse or dense matrices with ``umap-learn``.

    ``reuse`` controls whether and how the fitted transformer can be used later:

    - ``"none"``: keep only the output embedding and provenance.
    - ``"recompute"``: keep the fitting recipe/source reference and refit lazily
      from the immutable source artifact when later reuse is requested.
    - ``"stored"``: persist fitted transform-capable state.

    When ``reuse="stored"``, ``storage="compact"`` is the default. Compact
    persistence omits rebuildable PyNNDescent runtime callables while preserving
    exact transform behavior. ``storage="native"`` delegates to upstream object
    serialization and is retained mainly as a compatibility/debugging escape hatch.
    """

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
        reuse: UMAPReuse = "none",
        storage: UMAPStorage | None = None,
        retain_estimator: bool | None = None,
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

        # Backward compatibility for the short-lived pre-0.2 API. Old retained
        # estimators used upstream/native serialization, so preserve that exact
        # meaning rather than silently converting them to compact storage.
        if retain_estimator is not None:
            if reuse != "none" or storage is not None:
                raise ValueError(
                    "retain_estimator cannot be combined with reuse or storage; "
                    "use reuse='none' or reuse='stored' instead."
                )
            warnings.warn(
                "retain_estimator is deprecated; use reuse='none' or "
                "reuse='stored' with storage='compact'/'native'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if bool(retain_estimator):
                reuse = "stored"
                storage = "native"
            else:
                reuse = "none"

        reuse = str(reuse)
        if reuse not in _VALID_REUSE:
            raise ValueError(
                f"reuse must be one of {sorted(_VALID_REUSE)}; got {reuse!r}."
            )
        if reuse == "stored":
            storage = "compact" if storage is None else str(storage)
            if storage not in _VALID_STORAGE:
                raise ValueError(
                    f"storage must be one of {sorted(_VALID_STORAGE)} when "
                    f"reuse='stored'; got {storage!r}."
                )
        elif storage is not None:
            raise ValueError("storage is only applicable when reuse='stored'.")

        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = str(metric)
        self.random_state = random_state
        self.transform_seed = int(transform_seed)
        self.reuse = reuse
        self.storage = storage
        self.umap_kwargs = dict(umap_kwargs)
        self.source_features_: tuple[str, ...] | None = None
        self.fit_source_artifact_id_: str | None = None
        self._estimator = None
        self._fit_completed = False

    @property
    def retain_estimator(self) -> bool:
        """Backward-compatible view of whether fitted state is durably retained."""
        return self.reuse == "stored"

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
        # Do not clone large fitted UMAP/PyNNDescent object graphs into worker
        # processes. Sequential transform is deliberately preferred until TeAL
        # has a shared/read-only worker representation that does not duplicate
        # estimator state in RAM.
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return route == "sequential" and mode in {"fit_translate", "translate"}

    def prepare_for_translation(
        self,
        *,
        project: Project,
        sources: Mapping[str, BaseArtifact],
    ) -> None:
        _ = sources
        if not self._fit_completed or self.is_fitted:
            return
        if self.reuse == "none":
            raise OperatorError(
                "This UMAP was created with reuse='none' and intentionally has no "
                "reusable fitted state. Fit a new UMAP or choose reuse='recompute' "
                "or reuse='stored'."
            )
        if self.reuse == "recompute":
            self._recompute_from_project(project)

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
        if mode == "fit_translate":
            source_id = getattr(source, "artifact_id", None)
            if not isinstance(source_id, str) or not source_id:
                raise OperatorError(
                    "UMAP fitting requires a source artifact with a stable artifact_id."
                )
            self.fit_source_artifact_id_ = source_id
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
            if self.reuse == "stored":
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

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "transform_seed": self.transform_seed,
            "reuse": self.reuse,
            "storage": self.storage,
            "fit_completed": self._fit_completed,
            "umap_kwargs": self.umap_kwargs,
            "source_features": None
            if self.source_features_ is None
            else list(self.source_features_),
            "fit_source_artifact_id": self.fit_source_artifact_id_,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> UMAP:
        raw_kwargs = state.get("umap_kwargs", {})
        kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {}

        if "reuse" in state:
            reuse = str(state.get("reuse", "none"))
            raw_storage = state.get("storage")
            storage = None if raw_storage is None else str(raw_storage)
        else:
            # Legacy snapshots used retain_estimator and always wrote the raw
            # upstream pickle when retention was enabled.
            legacy_retain = bool(state.get("retain_estimator", True))
            reuse = "stored" if legacy_retain else "none"
            storage = "native" if legacy_retain else None

        obj = cls(
            n_components=int(state.get("n_components", 2)),
            n_neighbors=int(state.get("n_neighbors", 15)),
            min_dist=float(state.get("min_dist", 0.1)),
            metric=str(state.get("metric", "euclidean")),
            random_state=state.get("random_state", 42),
            transform_seed=int(state.get("transform_seed", 42)),
            reuse=reuse,  # type: ignore[arg-type]
            storage=storage,  # type: ignore[arg-type]
            **kwargs,
        )
        obj._fit_completed = bool(
            state.get("fit_completed", state.get("is_fitted", False))
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            obj.source_features_ = tuple(str(value) for value in raw)
        source_id = state.get("fit_source_artifact_id")
        if isinstance(source_id, str) and source_id:
            obj.fit_source_artifact_id_ = source_id
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if self.reuse != "stored" or not self.is_fitted:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        if self.storage == "compact":
            path = assets_dir / "umap.compact.pkl"
            dump_compact_umap(path, self._require_estimator())
            return {"estimator_file": path.name, "storage": "compact"}
        if self.storage == "native":
            path = assets_dir / "umap.pkl"
            dump_estimator(path, self._require_estimator())
            return {"estimator_file": path.name, "storage": "native"}
        raise OperatorError(f"Invalid stored UMAP storage mode {self.storage!r}.")

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("estimator_file") if manifest else None
        if filename is None:
            return
        if not isinstance(filename, str) or not filename:
            raise OperatorError("UMAP asset manifest has invalid estimator_file.")

        raw_storage = manifest.get("storage")
        persistence = (
            str(raw_storage) if raw_storage is not None else (self.storage or "native")
        )
        path = assets_dir / filename
        if persistence == "compact":
            self._estimator = load_compact_umap(path)
        elif persistence == "native":
            self._estimator = load_estimator(path)
        else:
            raise OperatorError(
                f"UMAP asset manifest has invalid storage mode {persistence!r}."
            )

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
                "UMAP requires the declared 'umap-learn' runtime dependency. "
                "Re-sync the TeAL environment."
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

    def _recompute_from_project(self, project: Project) -> None:
        source_id = self.fit_source_artifact_id_
        if not source_id:
            raise OperatorError(
                "This recomputable UMAP snapshot does not record its fitting "
                "source artifact. Fit a new UMAP operator."
            )
        try:
            source = project.get_artifact(source_id)
        except Exception as exc:
            raise OperatorError(
                f"Cannot recompute UMAP because fitting source artifact "
                f"{source_id!r} is unavailable in this project."
            ) from exc
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                f"Stored UMAP fitting source {source_id!r} is no longer a matrix artifact."
            )
        observed = tuple(str(value) for value in source.get_data_columns())
        if self.source_features_ is None or observed != self.source_features_:
            raise OperatorError(
                "Cannot recompute UMAP because the fitting source feature schema "
                "does not match the frozen operator recipe."
            )
        matrix = source.get_matrix()
        estimator = self._make_estimator()
        estimator.fit(matrix)
        self._estimator = estimator

    def _require_estimator(self):
        if self._estimator is None:
            raise OperatorNotFittedError("UMAP has no fitted reducer.")
        return self._estimator
