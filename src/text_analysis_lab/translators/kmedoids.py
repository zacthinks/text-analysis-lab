"""Rust-backed FasterPAM k-medoids clustering for TeAL.

The optional GPL-3.0-or-later kmedoids package performs fitting. TeAL retains
only medoid vectors and fitted provenance for reusable fixed-medoid prediction.
Neither sampling nor approximate clustering is performed implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances

from text_analysis_lab.core.errors import OperatorError, OperatorNotFittedError
from text_analysis_lab.translators.clustering import _ClusteringBase, _one_hot


class KMedoids(_ClusteringBase):
    """Exact FasterPAM optimization on a full pairwise distance matrix.

    Fit is O(N squared) in distance-matrix memory (8 * N * N bytes for
    float64, excluding the input and backend working space). No automatic
    training subsampling, distance approximation, or CLARA substitution.

    The backend is optional. Install text-analysis-lab[kmedoids] to fit;
    already-fitted snapshots can be loaded and reused without kmedoids.
    """

    batched_translate = True

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        metric: str = "euclidean",
        init: str = "build",
        max_iter: int = 100,
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if n_clusters < 1:
            raise ValueError("KMedoids n_clusters must be positive.")
        if metric not in {"euclidean", "manhattan", "cosine"}:
            raise ValueError(
                "KMedoids currently supports euclidean, manhattan or cosine "
                "distance on numeric features; precomputed input is unsupported."
            )
        if init not in {"random", "first", "build"}:
            raise ValueError("KMedoids init must be random, first or build.")
        if max_iter < 1:
            raise ValueError("KMedoids max_iter must be positive.")
        if random_state is not None and (
            not isinstance(random_state, (int, np.integer))
            or int(random_state) < 0
        ):
            raise ValueError("KMedoids random_state must be a nonnegative integer or None.")
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.init = init
        self.max_iter = int(max_iter)
        self.random_state = None if random_state is None else int(random_state)
        self.source_features_ = None
        self.n_clusters_ = None
        self.medoid_indices_: np.ndarray | None = None
        self._medoid_vectors: np.ndarray | None = None
        self.backend_version_: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self._medoid_vectors is not None

    @staticmethod
    def _backend():
        try:
            import kmedoids
        except ImportError as exc:
            raise ImportError(
                "Rust-backed KMedoids fitting needs the optional 'kmedoids' "
                "package (GPL-3.0-or-later). Install text-analysis-lab[kmedoids]."
            ) from exc
        return kmedoids

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError(
                "KMedoids is already fitted; construct a new translator to refit."
            )
        matrix, source, structured = self._standalone_input(input_matrix)
        n_rows = int(matrix.shape[0])
        if self.n_clusters > n_rows:
            raise ValueError("KMedoids n_clusters cannot exceed the input row count.")

        backend = self._backend()
        # The Rust package consumes dense, symmetric pairwise dissimilarities.
        # This is a deliberate full N x N scientific fit; never sample silently.
        distances = np.asarray(
            pairwise_distances(matrix, metric=self.metric), dtype=np.float64
        )
        if not np.all(np.isfinite(distances)):
            raise OperatorError("KMedoids requires finite pairwise distances.")
        result = backend.fasterpam(
            distances, self.n_clusters, max_iter=self.max_iter,
            init=self.init, random_state=self.random_state, n_cpu=1,
        )
        medoid_indices = np.asarray(result.medoids, dtype=np.int64)
        labels = np.asarray(result.labels, dtype=np.int64)
        if (
            medoid_indices.shape != (self.n_clusters,)
            or np.unique(medoid_indices).size != self.n_clusters
            or np.any(medoid_indices < 0)
            or np.any(medoid_indices >= n_rows)
            or labels.shape != (n_rows,)
            or np.any(labels < 0)
            or np.any(labels >= self.n_clusters)
        ):
            raise OperatorError("Rust KMedoids returned invalid medoids or assignments.")
        # Preserve only the fitted vectors, not the O(N squared) distances.
        self._medoid_vectors = np.asarray(
            matrix[medoid_indices].toarray()
            if hasattr(matrix, "toarray") else matrix[medoid_indices],
            dtype=np.float64,
        ).copy()
        self.medoid_indices_ = medoid_indices.copy()
        self.n_clusters_ = self.n_clusters
        self._record_backend_version()  # sklearn computes distances
        try:
            self.backend_version_ = version("kmedoids")
        except PackageNotFoundError:
            self.backend_version_ = None
        return self._output(
            _one_hot(labels, n_clusters=self.n_clusters), source, structured
        )

    def translate(self, input_matrix: Any) -> Any:
        if self._medoid_vectors is None:
            raise OperatorNotFittedError("KMedoids must be fitted before translation.")
        matrix, source, structured = self._standalone_input(input_matrix)
        if matrix.shape[1] != self._medoid_vectors.shape[1]:
            raise ValueError("KMedoids requires the fitted feature width.")
        # Batched prediction is independent of the original fitted corpus.
        labels = np.argmin(
            pairwise_distances(matrix, self._medoid_vectors, metric=self.metric),
            axis=1,
        )
        return self._output(
            _one_hot(labels, n_clusters=self.n_clusters), source, structured
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "metric": self.metric,
            "init": self.init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "source_features": (
                list(self.source_features_) if self.source_features_ is not None else None
            ),
            "n_clusters_found": self.n_clusters_,
            "medoid_indices": (
                self.medoid_indices_.tolist()
                if self.medoid_indices_ is not None else None
            ),
            "is_fitted": self.is_fitted,
            "backend": "rust-kmedoids",
            "backend_version": self.backend_version_,
            "sklearn_version": self.sklearn_version_,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> KMedoids:
        result = cls(
            n_clusters=int(state.get("n_clusters", 8)),
            metric=str(state.get("metric", "euclidean")),
            init=str(state.get("init", "build")),
            max_iter=int(state.get("max_iter", 100)),
            random_state=state.get("random_state", 42),
        )
        features = state.get("source_features")
        if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
            result.source_features_ = tuple(map(str, features))
        count = state.get("n_clusters_found")
        result.n_clusters_ = None if count is None else int(count)
        indices = state.get("medoid_indices")
        if indices is not None:
            result.medoid_indices_ = np.asarray(indices, dtype=np.int64)
        ver = state.get("backend_version")
        result.backend_version_ = None if ver is None else str(ver)
        skver = state.get("sklearn_version")
        result.sklearn_version_ = None if skver is None else str(skver)
        return result

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if self._medoid_vectors is None:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        np.save(assets_dir / "medoids.npy", self._medoid_vectors, allow_pickle=False)
        return {"medoids_file": "medoids.npy"}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            return
        if manifest.get("medoids_file") != "medoids.npy":
            raise OperatorError("KMedoids snapshot has an invalid medoids file.")
        medoids = np.asarray(
            np.load(assets_dir / "medoids.npy", allow_pickle=False), dtype=float
        )
        if (
            medoids.ndim != 2 or medoids.shape[0] != self.n_clusters
            or medoids.shape[1] < 1
            or not np.all(np.isfinite(medoids))
        ):
            raise OperatorError("KMedoids frozen medoid shape or values are invalid.")
        if (
            self.source_features_ is not None
            and len(self.source_features_) != medoids.shape[1]
        ):
            raise OperatorError("KMedoids medoids disagree with frozen feature schema.")
        if self.medoid_indices_ is not None and (
            self.medoid_indices_.shape != (self.n_clusters,)
            or np.any(self.medoid_indices_ < 0)
            or np.unique(self.medoid_indices_).size != self.n_clusters
        ):
            raise OperatorError("KMedoids medoid indices are invalid.")
        self._medoid_vectors = medoids.copy()
        self.n_clusters_ = self.n_clusters
