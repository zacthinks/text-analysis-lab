"""Fit-only hierarchical density clustering using scikit-learn's HDBSCAN.

The fitted operation records provenance and a keyed sparse N x K membership
matrix, not a predictive estimator, hierarchy, or approximate assigner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import ExecutionCapabilities
from text_analysis_lab.translators.clustering import _ClusteringBase, _one_hot

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


class HDBSCAN(_ClusteringBase):
    """One-shot density clustering; -1 denotes noise, never a cluster feature.

    Uses sklearn.cluster.HDBSCAN's parameter conventions, notably that
    min_samples *includes* the observation itself. Clustering is fit-only:
    applying it to new rows requires a separate, explicitly fitted classifier.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        *,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        allow_single_cluster: bool = False,
        algorithm: str = "auto",
        leaf_size: int = 40,
        n_jobs: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if min_cluster_size < 2:
            raise ValueError("HDBSCAN min_cluster_size must be at least 2.")
        if min_samples is not None and min_samples < 1:
            raise ValueError("HDBSCAN min_samples must be positive when specified.")
        if metric == "precomputed":
            raise ValueError("HDBSCAN requires feature vectors, not precomputed distances.")
        if cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("HDBSCAN cluster_selection_method must be 'eom' or 'leaf'.")
        if not np.isfinite(cluster_selection_epsilon) or cluster_selection_epsilon < 0:
            raise ValueError("HDBSCAN cluster_selection_epsilon must be finite and nonnegative.")
        if algorithm not in {"auto", "brute", "kd_tree", "ball_tree"}:
            raise ValueError("HDBSCAN algorithm must be auto, brute, kd_tree or ball_tree.")
        if leaf_size < 1:
            raise ValueError("HDBSCAN leaf_size must be positive.")
        if n_jobs == 0:
            raise ValueError("HDBSCAN n_jobs cannot be zero.")

        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = None if min_samples is None else int(min_samples)
        self.metric = str(metric)
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)
        self.allow_single_cluster = bool(allow_single_cluster)
        self.algorithm = algorithm
        self.leaf_size = int(leaf_size)
        self.n_jobs = n_jobs
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
            reasons=(
                "HDBSCAN is fit-only; its cluster hierarchy is not retained "
                "for assigning new observations.",
            ),
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("HDBSCAN is already fitted; construct a new translator to refit.")
        # The shared input validator rejects NaN/inf, which sklearn otherwise
        # labels -3/-2. Thus only -1 is an admissible negative label (noise).
        matrix, source, structured = self._standalone_input(input_matrix)
        estimator = SklearnHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            allow_single_cluster=self.allow_single_cluster,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
            store_centers=None,
        )
        labels = estimator.fit_predict(matrix)
        membership = _one_hot(labels)
        self.n_clusters_ = membership.shape[1]
        self._record_backend_version()
        self._fit_completed = True
        return self._output(membership, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        _ = input_matrix
        raise OperatorError(
            "HDBSCAN is fit-only; fit a new clustering model for other data, "
            "or train a separate inductive classifier from keyed X and Y."
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "allow_single_cluster": self.allow_single_cluster,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "n_jobs": self.n_jobs,
            "source_features": (
                list(self.source_features_) if self.source_features_ is not None else None
            ),
            "n_clusters_found": self.n_clusters_,
            "fit_completed": self._fit_completed,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> HDBSCAN:
        result = cls(
            min_cluster_size=int(state.get("min_cluster_size", 5)),
            min_samples=(
                None if state.get("min_samples") is None else int(state["min_samples"])
            ),
            metric=str(state.get("metric", "euclidean")),
            cluster_selection_method=str(state.get("cluster_selection_method", "eom")),
            cluster_selection_epsilon=float(state.get("cluster_selection_epsilon", 0.0)),
            allow_single_cluster=bool(state.get("allow_single_cluster", False)),
            algorithm=str(state.get("algorithm", "auto")),
            leaf_size=int(state.get("leaf_size", 40)),
            n_jobs=state.get("n_jobs"),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        count = state.get("n_clusters_found")
        result.n_clusters_ = None if count is None else int(count)
        result._fit_completed = bool(state.get("fit_completed", False))
        result._restore_backend_snapshot(state)
        return result
