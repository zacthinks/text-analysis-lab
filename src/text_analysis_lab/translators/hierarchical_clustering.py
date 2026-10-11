"""Fit-only sklearn OPTICS and agglomerative clustering translators.

Both produce one preserved-key sparse N x K membership artifact. The fitted
hierarchy/reachability graph is deliberately not retained for later prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.cluster import OPTICS as SklearnOPTICS

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import ExecutionCapabilities
from text_analysis_lab.translators.clustering import _ClusteringBase, _one_hot

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


class OPTICS(_ClusteringBase):
    """OPTICS reachability ordering with sklearn xi or DBSCAN-style extraction.

    Fitting uses the entire selected source artifact, not streamed batches.
    The reachability hierarchy is discarded after producing keyed memberships.
    sklearn's current OPTICS implementation has quadratic time complexity.
    """

    def __init__(
        self,
        min_samples: int = 5,
        *,
        max_eps: float = np.inf,
        metric: str = "minkowski",
        p: float = 2,
        cluster_method: str = "xi",
        eps: float | None = None,
        xi: float = 0.05,
        min_cluster_size: int | None = None,
        predecessor_correction: bool = True,
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if min_samples < 2:
            raise ValueError("OPTICS min_samples must be at least 2.")
        if np.isnan(max_eps) or max_eps <= 0:
            raise ValueError("OPTICS max_eps must be positive (infinity is allowed).")
        if metric == "precomputed":
            raise ValueError("OPTICS requires feature vectors, not precomputed distances.")
        if cluster_method not in {"xi", "dbscan"}:
            raise ValueError("OPTICS cluster_method must be 'xi' or 'dbscan'.")
        if not 0 < xi < 1:
            raise ValueError("OPTICS xi must be strictly between 0 and 1.")
        if min_cluster_size is not None and min_cluster_size < 2:
            raise ValueError("OPTICS min_cluster_size must be at least 2.")
        if not np.isfinite(p) or p < 1:
            raise ValueError("OPTICS p must be finite and at least 1.")
        if cluster_method == "xi" and eps is not None:
            raise ValueError("OPTICS eps applies only with cluster_method='dbscan'.")
        if cluster_method == "dbscan":
            if eps is None or not np.isfinite(eps) or eps <= 0:
                raise ValueError("OPTICS DBSCAN extraction requires a positive finite eps.")
            if eps > max_eps:
                raise ValueError("OPTICS eps cannot exceed max_eps.")
        if algorithm not in {"auto", "ball_tree", "kd_tree", "brute"}:
            raise ValueError("OPTICS algorithm must be auto, ball_tree, kd_tree or brute.")
        if leaf_size < 1:
            raise ValueError("OPTICS leaf_size must be positive.")
        if n_jobs == 0:
            raise ValueError("OPTICS n_jobs cannot be zero.")

        self.min_samples = int(min_samples)
        self.max_eps = float(max_eps)
        self.metric = str(metric)
        self.p = float(p)
        self.cluster_method = cluster_method
        self.eps = None if eps is None else float(eps)
        self.xi = float(xi)
        self.min_cluster_size = None if min_cluster_size is None else int(min_cluster_size)
        self.predecessor_correction = bool(predecessor_correction)
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
        self, *, project: Project | None = None
    ) -> ExecutionCapabilities:
        _ = project
        return ExecutionCapabilities(
            reusable=False,
            artifact=False,
            native=False,
            portable=False,
            reasons=("OPTICS is fit-only; its reachability ordering is not retained.",),
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("OPTICS is already fitted; construct a new translator to refit.")
        matrix, source, structured = self._standalone_input(input_matrix)
        estimator = SklearnOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric=self.metric,
            p=self.p,
            cluster_method=self.cluster_method,
            eps=self.eps,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size,
            predecessor_correction=self.predecessor_correction,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        try:
            labels = estimator.fit_predict(matrix)
        except TypeError as exc:
            if sparse.issparse(matrix):
                raise OperatorError(
                    "The sklearn OPTICS backend cannot process this sparse input/metric. "
                    "Use a compatible metric such as cosine, or first create a dense "
                    "embedding (e.g., SVD)."
                ) from exc
            raise
        membership = _one_hot(labels)
        self.n_clusters_ = membership.shape[1]
        self._record_backend_version()
        self._fit_completed = True
        return self._output(membership, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        _ = input_matrix
        raise OperatorError(
            "OPTICS is fit-only; fit a separate classifier from keyed X and Y "
            "to label new observations."
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "max_eps": None if np.isinf(self.max_eps) else self.max_eps,
            "metric": self.metric,
            "p": self.p,
            "cluster_method": self.cluster_method,
            "eps": self.eps,
            "xi": self.xi,
            "min_cluster_size": self.min_cluster_size,
            "predecessor_correction": self.predecessor_correction,
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
    def from_json_state(cls, state: Mapping[str, Any]) -> OPTICS:
        result = cls(
            min_samples=int(state.get("min_samples", 5)),
            max_eps=(np.inf if state.get("max_eps") is None else float(state["max_eps"])),
            metric=str(state.get("metric", "minkowski")),
            p=float(state.get("p", 2)),
            cluster_method=str(state.get("cluster_method", "xi")),
            eps=None if state.get("eps") is None else float(state["eps"]),
            xi=float(state.get("xi", 0.05)),
            min_cluster_size=(
                None if state.get("min_cluster_size") is None
                else int(state["min_cluster_size"])
            ),
            predecessor_correction=bool(state.get("predecessor_correction", True)),
            algorithm=str(state.get("algorithm", "auto")),
            leaf_size=int(state.get("leaf_size", 30)),
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


class AgglomerativeClustering(_ClusteringBase):
    """Fit-only hierarchical clustering with sklearn linkage and cut choices."""

    def __init__(
        self,
        n_clusters: int | None = 2,
        *,
        metric: str = "euclidean",
        linkage: str = "ward",
        distance_threshold: float | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if distance_threshold is None:
            if n_clusters is None or n_clusters < 1:
                raise ValueError(
                    "AgglomerativeClustering requires a positive n_clusters "
                    "when distance_threshold is not set."
                )
        else:
            if n_clusters is not None:
                raise ValueError(
                    "AgglomerativeClustering requires n_clusters=None "
                    "when using distance_threshold."
                )
            if not np.isfinite(distance_threshold) or distance_threshold < 0:
                raise ValueError(
                    "AgglomerativeClustering distance_threshold must be finite and nonnegative."
                )
        if linkage not in {"ward", "complete", "average", "single"}:
            raise ValueError("AgglomerativeClustering linkage is invalid.")
        if metric == "precomputed":
            raise ValueError(
                "AgglomerativeClustering requires feature vectors, not precomputed distances."
            )
        if linkage == "ward" and metric != "euclidean":
            raise ValueError(
                "AgglomerativeClustering ward linkage requires the euclidean metric."
            )
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        self.metric = str(metric)
        self.linkage = linkage
        self.distance_threshold = (
            None if distance_threshold is None else float(distance_threshold)
        )
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
            reusable=False,
            artifact=False,
            native=False,
            portable=False,
            reasons=("Agglomerative clustering is fit-only; the tree is not retained.",),
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError(
                "AgglomerativeClustering is already fitted; construct a new translator to refit."
            )
        matrix, source, structured = self._standalone_input(input_matrix)
        if sparse.issparse(matrix):
            raise OperatorError(
                "sklearn AgglomerativeClustering requires dense feature vectors; "
                "explicitly create a dense embedding (e.g., SVD) before fitting."
            )
        estimator = SklearnAgglomerative(
            n_clusters=self.n_clusters,
            metric=self.metric,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            compute_full_tree=True if self.distance_threshold is not None else "auto",
        )
        membership = _one_hot(estimator.fit_predict(matrix))
        self.n_clusters_ = membership.shape[1]
        self._record_backend_version()
        self._fit_completed = True
        return self._output(membership, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        _ = input_matrix
        raise OperatorError(
            "AgglomerativeClustering is fit-only; fit a separate classifier "
            "from keyed X and Y to label new observations."
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "metric": self.metric,
            "linkage": self.linkage,
            "distance_threshold": self.distance_threshold,
            "source_features": (
                list(self.source_features_) if self.source_features_ is not None else None
            ),
            "n_clusters_found": self.n_clusters_,
            "fit_completed": self._fit_completed,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> AgglomerativeClustering:
        result = cls(
            n_clusters=state.get("n_clusters", 2),
            metric=str(state.get("metric", "euclidean")),
            linkage=str(state.get("linkage", "ward")),
            distance_threshold=(
                None if state.get("distance_threshold") is None
                else float(state["distance_threshold"])
            ),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        count = state.get("n_clusters_found")
        result.n_clusters_ = None if count is None else int(count)
        result._fit_completed = bool(state.get("fit_completed", False))
        result._restore_backend_snapshot(state)
        return result
