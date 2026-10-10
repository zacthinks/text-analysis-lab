"""Reusable sklearn Gaussian mixture clustering with posterior responsibilities.

Unlike hard clustering, the TeAL output is a dense N x K posterior probability
matrix. The frozen scientific state stores numerical mixture parameters, not
a duplicate training matrix or pickled estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from text_analysis_lab.core.errors import OperatorError, OperatorNotFittedError
from text_analysis_lab.core.operator import OutputSpec, TranslationRequest
from text_analysis_lab.core.types import DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    feature_metadata_from_columns, single_source, standalone_matrix_payload,
)
from text_analysis_lab.translators.clustering import _ClusteringBase, _cluster_columns


class GaussianMixture(_ClusteringBase):
    """Fitted density mixture: dense posteriors, fixed K and reusable parameters."""

    batched_translate = True

    def __init__(
        self,
        n_components: int = 2,
        *,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if n_components < 1:
            raise ValueError("GaussianMixture n_components must be positive.")
        if covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError("GaussianMixture covariance_type must be full, tied, diag or spherical.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("GaussianMixture tol must be finite and nonnegative.")
        if not np.isfinite(reg_covar) or reg_covar < 0:
            raise ValueError("GaussianMixture reg_covar must be finite and nonnegative.")
        if max_iter < 1 or n_init < 1:
            raise ValueError("GaussianMixture max_iter and n_init must be positive.")
        if init_params not in {"kmeans", "k-means++", "random", "random_from_data"}:
            raise ValueError("GaussianMixture init_params is not a supported sklearn initializer.")
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.init_params = init_params
        self.random_state = random_state
        self.source_features_ = None
        self.n_clusters_ = None
        self._estimator: SkGaussianMixture | None = None
        self.converged_: bool | None = None
        self.n_iter_: int | None = None
        self.lower_bound_: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    def output_specs(
        self, *, sources: Mapping[str, Any], request: TranslationRequest
    ) -> OutputSpec:
        _ = request
        single_source(sources, name="GaussianMixture")
        return OutputSpec(
            artifact_type="dense_matrix",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
            feature_metadata_mode="own",
        )

    def _dense_input(
        self, input_matrix: Any
    ) -> tuple[np.ndarray, Mapping[str, Any] | None, bool]:
        matrix, source, structured = self._standalone_input(input_matrix)
        if sparse.issparse(matrix):
            raise OperatorError(
                "sklearn GaussianMixture requires dense numeric feature vectors; "
                "explicitly create a dense representation (e.g., SVD) first."
            )
        return np.asarray(matrix, dtype=float), source, structured

    def _output(
        self, values: np.ndarray, source: Mapping[str, Any] | None, structured: bool
    ) -> Any:
        if structured:
            return standalone_matrix_payload(
                values,
                feature_metadata_from_columns(_cluster_columns(values.shape[1])),
                name="GaussianMixture", source=source,
            )
        return values

    def _make_estimator(self) -> SkGaussianMixture:
        return SkGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
        )

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("GaussianMixture is already fitted; create a new translator.")
        matrix, source, structured = self._dense_input(input_matrix)
        estimator = self._make_estimator().fit(matrix)
        values = np.asarray(estimator.predict_proba(matrix), dtype=float)
        self._estimator = estimator
        self.n_clusters_ = self.n_components
        self.converged_ = bool(estimator.converged_)
        self.n_iter_ = int(estimator.n_iter_)
        self.lower_bound_ = float(estimator.lower_bound_)
        self._record_backend_version()
        return self._output(values, source, structured)

    def translate(self, input_matrix: Any) -> Any:
        if self._estimator is None:
            raise OperatorNotFittedError("GaussianMixture must be fitted before translation.")
        matrix, source, structured = self._dense_input(input_matrix)
        if matrix.shape[1] != self._estimator.means_.shape[1]:
            raise ValueError("GaussianMixture requires the fitted feature width.")
        values = np.asarray(self._estimator.predict_proba(matrix), dtype=float)
        return self._output(values, source, structured)

    def to_json_state(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "init_params": self.init_params,
            "random_state": self.random_state,
            "source_features": (
                list(self.source_features_) if self.source_features_ is not None else None
            ),
            "n_clusters_found": self.n_clusters_,
            "is_fitted": self.is_fitted,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "lower_bound": self.lower_bound_,
            **self._backend_snapshot(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> GaussianMixture:
        result = cls(
            n_components=int(state.get("n_components", 2)),
            covariance_type=str(state.get("covariance_type", "full")),
            tol=float(state.get("tol", 1e-3)),
            reg_covar=float(state.get("reg_covar", 1e-6)),
            max_iter=int(state.get("max_iter", 100)),
            n_init=int(state.get("n_init", 1)),
            init_params=str(state.get("init_params", "kmeans")),
            random_state=state.get("random_state", 42),
        )
        raw = state.get("source_features")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            result.source_features_ = tuple(map(str, raw))
        count = state.get("n_clusters_found")
        result.n_clusters_ = None if count is None else int(count)
        converged = state.get("converged")
        result.converged_ = None if converged is None else bool(converged)
        iters = state.get("n_iter")
        result.n_iter_ = None if iters is None else int(iters)
        bound = state.get("lower_bound")
        result.lower_bound_ = None if bound is None else float(bound)
        result._restore_backend_snapshot(state)
        return result

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if self._estimator is None:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "mixture.npz"
        estimator = self._estimator
        np.savez_compressed(
            path,
            weights=estimator.weights_,
            means=estimator.means_,
            covariances=estimator.covariances_,
            precisions_cholesky=estimator.precisions_cholesky_,
        )
        return {"mixture_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            return
        if manifest.get("mixture_file") != "mixture.npz":
            raise OperatorError("GaussianMixture snapshot has an invalid mixture file.")
        with np.load(assets_dir / "mixture.npz", allow_pickle=False) as stored:
            required = {"weights", "means", "covariances", "precisions_cholesky"}
            if set(stored.files) != required:
                raise OperatorError("GaussianMixture snapshot lacks fitted parameters.")
            arrays = {name: np.asarray(stored[name], dtype=float) for name in required}
        weights, means = arrays["weights"], arrays["means"]
        k = self.n_components
        if means.ndim != 2 or means.shape[0] != k or weights.shape != (k,):
            raise OperatorError("GaussianMixture fitted mixture shape is invalid.")
        d = means.shape[1]
        if d < 1 or (self.source_features_ is not None and len(self.source_features_) != d):
            raise OperatorError("GaussianMixture fitted feature width is invalid.")
        shape = {
            "full": (k, d, d), "tied": (d, d),
            "diag": (k, d), "spherical": (k,),
        }[self.covariance_type]
        if any(arrays[field].shape != shape for field in ("covariances", "precisions_cholesky")):
            raise OperatorError("GaussianMixture covariance shape is invalid.")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise OperatorError("GaussianMixture fitted mixture contains nonfinite parameters.")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise OperatorError("GaussianMixture weights are invalid.")
        estimator = self._make_estimator()
        estimator.weights_ = weights.copy()
        estimator.means_ = means.copy()
        estimator.covariances_ = arrays["covariances"].copy()
        estimator.precisions_cholesky_ = arrays["precisions_cholesky"].copy()
        estimator.n_features_in_ = d
        estimator.converged_ = bool(self.converged_)
        estimator.n_iter_ = 0 if self.n_iter_ is None else self.n_iter_
        estimator.lower_bound_ = (
            float("nan") if self.lower_bound_ is None else self.lower_bound_
        )
        self._estimator = estimator
        self.n_clusters_ = k
