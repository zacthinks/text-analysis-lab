"""Reusable mini-batch K-means and fit-only sklearn spectral clustering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans as SkMiniBatchKMeans
from sklearn.cluster import SpectralClustering as SkSpectralClustering

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import ExecutionCapabilities
from text_analysis_lab.translators.clustering import KMeans, _ClusteringBase, _one_hot

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


class MiniBatchKMeans(KMeans):
    """Sklearn mini-batch K-means with reusable frozen center-only prediction.

    The constructor's batch_size is a scientific sklearn training parameter.
    Project.translate(batch_size=...) controls execution batching only and
    does not sample training observations or alter this estimator's batch_size.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: str = "k-means++",
        n_init: int | str = "auto",
        max_iter: int = 100,
        batch_size: int = 1024,
        tol: float = 0.0,
        max_no_improvement: int | None = 10,
        reassignment_ratio: float = 0.01,
        init_size: int | None = None,
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters, init=init, n_init=n_init,
            max_iter=max_iter, tol=tol, random_state=random_state,
            operator_id=operator_id,
        )
        if batch_size < 1 or max_iter < 1:
            raise ValueError("MiniBatchKMeans batch_size and max_iter must be positive.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("MiniBatchKMeans tol must be nonnegative and finite.")
        if max_no_improvement is not None and max_no_improvement < 1:
            raise ValueError("MiniBatchKMeans max_no_improvement must be positive or None.")
        if not np.isfinite(reassignment_ratio) or reassignment_ratio < 0:
            raise ValueError("MiniBatchKMeans reassignment_ratio must be nonnegative.")
        if init_size is not None and init_size < 1:
            raise ValueError("MiniBatchKMeans init_size must be positive.")
        self.batch_size = int(batch_size)
        self.max_no_improvement = max_no_improvement
        self.reassignment_ratio = float(reassignment_ratio)
        self.init_size = init_size

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("MiniBatchKMeans is already fitted; create a new translator.")
        matrix, source, structured = self._standalone_input(input_matrix)
        estimator = SkMiniBatchKMeans(
            n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
            max_iter=self.max_iter, batch_size=self.batch_size, tol=self.tol,
            max_no_improvement=self.max_no_improvement,
            reassignment_ratio=self.reassignment_ratio, init_size=self.init_size,
            random_state=self.random_state,
        )
        labels = estimator.fit_predict(matrix)
        self._centers = np.asarray(estimator.cluster_centers_, dtype=float).copy()
        self.n_clusters_ = self.n_clusters
        self._record_backend_version()
        return self._output(
            _one_hot(labels, n_clusters=self.n_clusters), source, structured
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            **super().to_json_state(),
            "batch_size": self.batch_size,
            "max_no_improvement": self.max_no_improvement,
            "reassignment_ratio": self.reassignment_ratio,
            "init_size": self.init_size,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> MiniBatchKMeans:
        result = cls(
            n_clusters=int(state.get("n_clusters", 8)),
            init=str(state.get("init", "k-means++")),
            n_init=state.get("n_init", "auto"),
            max_iter=int(state.get("max_iter", 100)),
            batch_size=int(state.get("batch_size", 1024)),
            tol=float(state.get("tol", 0.0)),
            max_no_improvement=state.get("max_no_improvement", 10),
            reassignment_ratio=float(state.get("reassignment_ratio", 0.01)),
            init_size=state.get("init_size"),
            random_state=state.get("random_state", 42),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        result._restore_backend_snapshot(state)
        return result


class SpectralClustering(_ClusteringBase):
    """Fit-only spectral clustering; a graph is fitted to the whole input.

    Supported feature-based affinities are rbf and nearest_neighbors. No
    precomputed affinity input, implicit sample reduction, or graph retention.
    sklearn can consume CSR features for these affinity modes; graph and
    eigenvector computation can still be memory/time intensive.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        assign_labels: str = "kmeans",
        n_init: int = 10,
        eigen_solver: str | None = None,
        eigen_tol: float | str = "auto",
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if n_clusters < 1:
            raise ValueError("SpectralClustering n_clusters must be positive.")
        if affinity not in {"rbf", "nearest_neighbors"}:
            raise ValueError(
                "SpectralClustering accepts feature-based rbf or nearest_neighbors "
                "affinity, not precomputed affinity matrices."
            )
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("SpectralClustering gamma must be positive and finite.")
        if n_neighbors < 1 or n_init < 1:
            raise ValueError("SpectralClustering n_neighbors and n_init must be positive.")
        if assign_labels not in {"kmeans", "discretize", "cluster_qr"}:
            raise ValueError("SpectralClustering assign_labels is invalid.")
        if eigen_solver not in {None, "arpack", "lobpcg"}:
            raise ValueError("SpectralClustering eigen_solver must be None, arpack or lobpcg.")
        if eigen_tol != "auto" and (
            not isinstance(eigen_tol, (int, float))
            or not np.isfinite(eigen_tol) or eigen_tol < 0
        ):
            raise ValueError("SpectralClustering eigen_tol must be auto or nonnegative.")
        self.n_clusters = int(n_clusters)
        self.affinity = affinity
        self.gamma = float(gamma)
        self.n_neighbors = int(n_neighbors)
        self.assign_labels = assign_labels
        self.n_init = int(n_init)
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.random_state = random_state
        self.source_features_ = None
        self.n_clusters_ = None
        self._fit_completed = False

    @property
    def is_fitted(self) -> bool:
        return self._fit_completed

    def execution_capabilities(
        self, *, project: Project | None = None
    ) -> ExecutionCapabilities:
        _ = project
        return ExecutionCapabilities(
            reusable=False, artifact=False, native=False, portable=False,
            reasons=("SpectralClustering is fit-only; the graph is not retained.",),
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("SpectralClustering is already fitted; create a new translator.")
        matrix, source, structured = self._standalone_input(input_matrix)
        estimator = SkSpectralClustering(
            n_clusters=self.n_clusters, affinity=self.affinity, gamma=self.gamma,
            n_neighbors=self.n_neighbors, assign_labels=self.assign_labels,
            n_init=self.n_init, eigen_solver=self.eigen_solver,
            eigen_tol=self.eigen_tol, random_state=self.random_state,
        )
        labels = estimator.fit_predict(matrix)
        membership = _one_hot(labels, n_clusters=self.n_clusters)
        self.n_clusters_ = self.n_clusters
        self._record_backend_version()
        self._fit_completed = True
        return self._output(membership, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        _ = input_matrix
        raise OperatorError(
            "SpectralClustering is fit-only; fit a separate classifier on keyed X,Y "
            "for new observations."
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "affinity": self.affinity, "gamma": self.gamma,
            "n_neighbors": self.n_neighbors, "assign_labels": self.assign_labels,
            "n_init": self.n_init, "eigen_solver": self.eigen_solver,
            "eigen_tol": self.eigen_tol, "random_state": self.random_state,
            "source_features": (
                list(self.source_features_) if self.source_features_ is not None else None
            ),
            "n_clusters_found": self.n_clusters_,
            "fit_completed": self._fit_completed,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SpectralClustering:
        result = cls(
            n_clusters=int(state.get("n_clusters", 8)),
            affinity=str(state.get("affinity", "rbf")),
            gamma=float(state.get("gamma", 1.0)),
            n_neighbors=int(state.get("n_neighbors", 10)),
            assign_labels=str(state.get("assign_labels", "kmeans")),
            n_init=int(state.get("n_init", 10)),
            eigen_solver=state.get("eigen_solver"),
            eigen_tol=state.get("eigen_tol", "auto"),
            random_state=state.get("random_state", 42),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        count = state.get("n_clusters_found")
        result.n_clusters_ = None if count is None else int(count)
        result._fit_completed = bool(state.get("fit_completed", False))
        result._restore_backend_snapshot(state)
        return result
