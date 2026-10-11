"""Native TeAL clustering translators for K-means and DBSCAN.

Both methods emit one preserved-key sparse N x K membership matrix.
DBSCAN is intentionally fit-only; K-means stores just its learned centers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import sklearn
from scipy import sparse
from sklearn.metrics import pairwise_distances_argmin

from text_analysis_lab.core.errors import ArtifactError, OperatorError, OperatorNotFittedError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    ExecutionCapabilities,
    InputBatch,
    OutputMap,
    OutputSpec,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    establish_or_validate_features,
    feature_labels,
    feature_metadata_from_columns,
    key_frame,
    native_matrix_packet,
    single_input,
    single_source,
    standalone_matrix_payload,
    unpack_standalone_matrix,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


def _cluster_columns(n_clusters: int) -> list[str]:
    return [f"cluster_{index}" for index in range(int(n_clusters))]


def _validate_matrix(matrix: Any, *, name: str) -> Any:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        values = matrix.data
    else:
        matrix = np.asarray(matrix)
        values = matrix
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ArtifactError(f"{name} requires a nonempty, two-dimensional numeric matrix.")
    try:
        finite = np.all(np.isfinite(values))
    except TypeError as exc:
        raise ArtifactError(f"{name} requires numeric matrix values.") from exc
    if not finite:
        raise ArtifactError(f"{name} requires finite matrix values.")
    return matrix


def _one_hot(labels: Any, *, n_clusters: int | None = None) -> sparse.csr_matrix:
    """Encode contiguous nonnegative labels; -1 means an all-zero/noise row."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or np.any(values < -1):
        raise OperatorError("Clustering backend returned invalid cluster labels.")
    count = int(values.max() + 1) if values.size and np.any(values >= 0) else 0
    if n_clusters is not None and count > n_clusters:
        raise OperatorError("Cluster label exceeded the frozen cluster feature schema.")
    width = count if n_clusters is None else int(n_clusters)
    assigned = np.flatnonzero(values >= 0)
    return sparse.csr_matrix(
        (np.ones(len(assigned), dtype=np.int8), (assigned, values[assigned])),
        shape=(len(values), width),
        dtype=np.int8,
    )


def _source_request(
    translator: "_ClusteringBase",
    sources: Mapping[str, "BaseArtifact"],
    *,
    name: str,
    mode: TranslationMode,
    request: TranslationRequest,
) -> SourceRequest:
    source = single_source(sources, name=name)
    if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
        raise OperatorError(f"{name} requires sparse_matrix or dense_matrix input.")
    resolved_features = establish_or_validate_features(
        translator.source_features_,
        source.get_data_columns(),
        fitted=translator.is_fitted,
        name=name,
    )
    if translator.source_features_ is None:
        translator.source_features_ = resolved_features
    # Fitting is global; assigning a row to already-frozen K-means centers is
    # independent of other rows. This changes processing only, never sampling.
    batched_prediction = mode == "translate" and translator.batched_translate
    return SourceRequest(
        artifact_type=("sparse_matrix", "dense_matrix"),
        mode="batches" if batched_prediction else "full_artifact",
        columns=ColumnRequest(keys=True, data=True, metadata=False),
        batch_size=(request.batch_size or 10_000) if batched_prediction else None,
        form="native",
        metadata_mode="none",
        include_position=False,
    )


class _ClusteringBase(BaseTranslator):
    """Only the genuinely shared TeAL output and source protocol."""

    operation_type = "translate"
    source_features_: tuple[str, ...] | None
    n_clusters_: int | None
    batched_translate = False
    sklearn_version_: str | None = None

    def _record_backend_version(self) -> None:
        """Record the upstream backend used for this particular scientific fit."""
        self.sklearn_version_ = sklearn.__version__

    def _backend_snapshot(self) -> dict[str, str | None]:
        return {"backend": "sklearn", "sklearn_version": self.sklearn_version_}

    def _restore_backend_snapshot(self, state: Mapping[str, Any]) -> None:
        # Earlier pre-Phase-3 snapshots did not carry the backend version.
        raw = state.get("sklearn_version")
        self.sklearn_version_ = None if raw is None else str(raw)

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def supports_fit_translate(self) -> bool:
        return not self.is_fitted

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        single_source(sources, name=type(self).__name__)
        return OutputSpec(
            artifact_type="sparse_matrix",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
            feature_metadata_mode="own",
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
                f"{type(self).__name__} does not accept operation parameters: {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        return _source_request(
            self, sources, name=type(self).__name__, mode=mode, request=request
        )

    def _standalone_input(self, input_matrix: Any) -> tuple[Any, Mapping[str, Any] | None, bool]:
        source = input_matrix if isinstance(input_matrix, Mapping) else None
        matrix, metadata, structured = unpack_standalone_matrix(
            input_matrix, name=type(self).__name__
        )
        matrix = _validate_matrix(matrix, name=type(self).__name__)
        if metadata is not None:
            resolved_features = establish_or_validate_features(
                self.source_features_,
                feature_labels(metadata, name=type(self).__name__),
                fitted=self.is_fitted,
                name=type(self).__name__,
            )
            if self.source_features_ is None:
                self.source_features_ = resolved_features
        if self.source_features_ is not None and len(self.source_features_) != matrix.shape[1]:
            raise ValueError(f"{type(self).__name__} requires the fitted feature width.")
        return matrix, source, structured

    def _output(self, matrix: sparse.csr_matrix, source: Mapping[str, Any] | None, structured: bool) -> Any:
        if structured:
            return standalone_matrix_payload(
                matrix,
                feature_metadata_from_columns(_cluster_columns(matrix.shape[1])),
                name=type(self).__name__,
                source=source,
            )
        return matrix

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        packet = single_input(inputs, name=type(self).__name__)
        info, matrix, columns = native_matrix_packet(packet, name=type(self).__name__)
        if mode == "fit_translate":
            membership = self.fit_transform(matrix)
        elif mode == "translate":
            membership = self.translate(matrix)
        else:
            raise OperatorError(f"{type(self).__name__} cannot execute mode {mode!r}.")
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": key_frame(info, columns),
                    "data": {
                        "values": membership,
                        "columns": _cluster_columns(membership.shape[1]),
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
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None


class KMeans(_ClusteringBase):
    """Sklearn K-means with a compact, centers-only reusable frozen state."""

    batched_translate = True

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: str = "k-means++",
        n_init: int | str = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive.")
        if init not in {"k-means++", "random"}:
            raise ValueError("KMeans init must be 'k-means++' or 'random'.")
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.source_features_ = None
        self.n_clusters_ = None
        self._centers: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._centers is not None

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("KMeans is already fitted; construct a new translator to refit.")
        matrix, source, structured = self._standalone_input(input_matrix)
        from sklearn.cluster import KMeans as SKKMeans
        estimator = SKKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        labels = estimator.fit_predict(matrix)
        self._centers = np.asarray(estimator.cluster_centers_, dtype=float).copy()
        self._record_backend_version()
        self.n_clusters_ = self.n_clusters
        return self._output(_one_hot(labels, n_clusters=self.n_clusters), source, structured)

    def translate(self, input_matrix: Any) -> Any:
        if self._centers is None:
            raise OperatorNotFittedError("KMeans must be fitted before translation.")
        matrix, source, structured = self._standalone_input(input_matrix)
        if matrix.shape[1] != self._centers.shape[1]:
            raise ValueError("KMeans requires the fitted feature width.")
        labels = pairwise_distances_argmin(matrix, self._centers)
        return self._output(_one_hot(labels, n_clusters=self.n_clusters), source, structured)

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "source_features": list(self.source_features_) if self.source_features_ is not None else None,
            "is_fitted": self.is_fitted,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> KMeans:
        result = cls(
            n_clusters=int(state["n_clusters"]),
            init=str(state.get("init", "k-means++")),
            n_init=state.get("n_init", "auto"),
            max_iter=int(state.get("max_iter", 300)),
            tol=float(state.get("tol", 1e-4)),
            random_state=state.get("random_state", 42),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        result._restore_backend_snapshot(state)
        return result

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if self._centers is None:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        np.save(assets_dir / "centers.npy", self._centers, allow_pickle=False)
        return {"centers_file": "centers.npy"}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            return
        name = manifest.get("centers_file")
        if name != "centers.npy":
            raise OperatorError("KMeans snapshot has an invalid centers file.")
        self._centers = np.load(assets_dir / name, allow_pickle=False)
        if self._centers.shape[0] != self.n_clusters or self._centers.ndim != 2:
            raise OperatorError("KMeans frozen centroid shape is invalid.")
        self.n_clusters_ = self.n_clusters


class DBSCAN(_ClusteringBase):
    """Fit-only density clustering; original membership is the durable result."""

    def __init__(
        self,
        eps: float = 0.5,
        *,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: float | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("DBSCAN eps must be positive and finite.")
        if min_samples < 1:
            raise ValueError("DBSCAN min_samples must be positive.")
        if metric == "precomputed":
            raise ValueError("DBSCAN requires feature vectors, not a precomputed distance matrix.")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = str(metric)
        self.algorithm = str(algorithm)
        self.leaf_size = int(leaf_size)
        self.p = p
        self.source_features_ = None
        self.n_clusters_ = None
        self._fit_completed = False

    @property
    def is_fitted(self) -> bool:
        return self._fit_completed

    def execution_capabilities(
        self,
        *,
        project: Project | None = None,
    ) -> ExecutionCapabilities:
        _ = project
        return ExecutionCapabilities(
            reusable=False,
            artifact=False,
            native=False,
            portable=False,
            reasons=("DBSCAN is fit-only; its learned cluster structure does not provide an inductive transform.",),
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("DBSCAN is already fitted; construct a new translator to refit.")
        matrix, source, structured = self._standalone_input(input_matrix)
        from sklearn.cluster import DBSCAN as SKDBSCAN
        estimator = SKDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
        )
        labels = estimator.fit_predict(matrix)
        membership = _one_hot(labels)
        self._record_backend_version()
        self.n_clusters_ = membership.shape[1]
        self._fit_completed = True
        return self._output(membership, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        _ = input_matrix
        raise OperatorError(
            "DBSCAN is fit-only; run a new clustering fit for other data, "
            "or fit a separate inductive classifier from X and its cluster memberships."
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "source_features": list(self.source_features_) if self.source_features_ is not None else None,
            "n_clusters_found": self.n_clusters_,
            "fit_completed": self._fit_completed,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> DBSCAN:
        result = cls(
            eps=float(state.get("eps", 0.5)),
            min_samples=int(state.get("min_samples", 5)),
            metric=str(state.get("metric", "euclidean")),
            algorithm=str(state.get("algorithm", "auto")),
            leaf_size=int(state.get("leaf_size", 30)),
            p=state.get("p"),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        raw_count = state.get("n_clusters_found")
        result.n_clusters_ = None if raw_count is None else int(raw_count)
        result._fit_completed = bool(state.get("fit_completed", False))
        result._restore_backend_snapshot(state)
        return result
