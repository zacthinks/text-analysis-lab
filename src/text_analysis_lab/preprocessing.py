"""Small sklearn preprocessing components for TeAL classifier pipelines.

These transform in-memory estimator inputs, not durable TeAL geometries.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis
from sklearn.utils.validation import check_array, check_is_fitted


class FeatureScaleNormalizer(TransformerMixin, BaseEstimator):
    """Divide each feature by its fitted column scale raised to gamma.

    For a fitted column j, s_j = sqrt(mean(X[:, j]**2)) with
    norm='rms' (default), or sqrt(sum(X[:, j]**2)) with norm='l2'.
    The transformation is X[:, j] / s_j**gamma.
    These norms differ by sqrt(n_training_rows): consequently their
    interactions with a fixed classifier C/lambda differ when sample size
    changes. RMS is the default to avoid that arbitrary sample-count factor.

    All-zero and near-zero columns (scale <= zero_tol) receive unit
    scaling, and are therefore left unchanged. No centering is performed.
    Statistics are fitted exclusively on the data passed to fit;
    place this transformer inside an sklearn Pipeline for fold-local CV.

    Parameters
    ----------
    gamma : float, default=1.0
        Continuous exponent in [0, 1]; 0 is exactly no scaling, 1 fully
        corrects each feature's RMS or strict L2 magnitude.
    norm : {'rms', 'l2'}, default='rms'
        Column scale convention, with population (not sample) RMS.
    zero_tol : float, default=1e-12
        Inclusive near-zero scale threshold, in input feature units.
    copy : bool, default=True
        Copy inputs for transform. With copy=False, floating-point
        NumPy arrays and CSR/CSC matrices may be modified in place.
    """

    STATE_VERSION = 1

    def __init__(
        self, gamma: float = 1.0, *, norm: str = "rms",
        zero_tol: float = 1e-12, copy: bool = True,
    ) -> None:
        self.gamma = gamma
        self.norm = norm
        self.zero_tol = zero_tol
        self.copy = copy

    def _validate_parameters(self) -> tuple[float, float]:
        if not isinstance(self.gamma, Real) or isinstance(self.gamma, bool):
            raise ValueError("gamma must be a finite real number in [0, 1].")
        gamma = float(self.gamma)
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be a finite real number in [0, 1].")
        if self.norm not in {"rms", "l2"}:
            raise ValueError("norm must be 'rms' or 'l2'.")
        if not isinstance(self.zero_tol, Real) or isinstance(self.zero_tol, bool):
            raise ValueError("zero_tol must be a finite nonnegative real number.")
        zero_tol = float(self.zero_tol)
        if not np.isfinite(zero_tol) or zero_tol < 0:
            raise ValueError("zero_tol must be a finite nonnegative real number.")
        if not isinstance(self.copy, bool):
            raise ValueError("copy must be a bool.")
        return gamma, zero_tol

    @staticmethod
    def _check_input(X, *, copy: bool):
        return check_array(
            X, accept_sparse=("csr", "csc"), dtype=np.float64,
            ensure_all_finite=True, ensure_min_samples=1,
            ensure_min_features=1, copy=copy,
        )

    def fit(self, X, y=None):
        """Freeze training-column RMS or strict L2 scales (no label use)."""
        self._validate_parameters()
        X = self._check_input(X, copy=False)
        n_rows, n_columns = X.shape
        if sparse.issparse(X):
            # CSR/CSC may store the same coordinate more than once. These
            # entries must be summed *before* second moments are computed:
            # (100 - 99)**2 is not 100**2 + (-99)**2. Never mutate fit input.
            if not X.has_canonical_format:
                X = X.copy()
                X.sum_duplicates()
            means, variances = mean_variance_axis(X, axis=0)
            square_means = np.maximum(variances, 0.0) + np.square(means)
            rms = np.sqrt(square_means)
        else:
            # Norm per column avoids allocating a second dense squared matrix.
            rms = np.linalg.norm(X, axis=0) / np.sqrt(n_rows)
        scales = rms if self.norm == "rms" else rms * np.sqrt(n_rows)
        if not np.isfinite(scales).all():
            raise ValueError("Column scales overflowed; rescale input values.")
        self.n_features_in_ = n_columns
        self.n_samples_seen_ = n_rows
        self.column_scales_ = np.asarray(scales, dtype=np.float64)
        return self

    def transform(self, X):
        """Scale dense or sparse columns without centering or densification."""
        gamma, zero_tol = self._validate_parameters()
        check_is_fitted(self, "column_scales_")
        X = self._check_input(X, copy=self.copy)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        if gamma == 0:
            return X
        safe = np.where(self.column_scales_ <= zero_tol, 1.0, self.column_scales_)
        factors = np.power(safe, -gamma)
        if sparse.issparse(X):
            inplace_column_scale(X, factors)
        else:
            X *= factors
        if sparse.issparse(X):
            if not np.isfinite(X.data).all():
                raise ValueError("Scaled values overflowed; check input feature scales.")
        elif not np.isfinite(X).all():
            raise ValueError("Scaled values overflowed; check input feature scales.")
        return X

    def to_fitted_state(self) -> dict:
        """Export JSON-compatible state without a Python class/import path.

        GeCo can implement the same schema independently. Consumers reconstruct
        the fitted TeAL transformer before serializing an exported Pipeline.
        This does not attempt to unpickle a GeCo-specific Python class.
        """
        self._validate_parameters()
        check_is_fitted(self, "column_scales_")
        return {
            "version": self.STATE_VERSION,
            "gamma": float(self.gamma), "norm": self.norm,
            "zero_tol": float(self.zero_tol),
            "n_samples_seen": int(self.n_samples_seen_),
            "column_scales": self.column_scales_.tolist(),
        }

    @classmethod
    def from_fitted_state(cls, state: Mapping) -> FeatureScaleNormalizer:
        """Reconstitute a fitted scaler from the versioned neutral schema."""
        if not isinstance(state, Mapping) or state.get("version") != cls.STATE_VERSION:
            raise ValueError("Unsupported feature-scale state version.")
        result = cls(
            gamma=state.get("gamma"), norm=state.get("norm"),
            zero_tol=state.get("zero_tol"),
        )
        result._validate_parameters()
        n = state.get("n_samples_seen")
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError("n_samples_seen must be a positive integer.")
        raw = state.get("column_scales")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("column_scales must be a nonempty numeric list.")
        try:
            scales = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("column_scales must be numeric.") from exc
        if scales.ndim != 1 or not np.isfinite(scales).all() or (scales < 0).any():
            raise ValueError("column_scales must be finite and nonnegative.")
        result.n_features_in_ = len(scales)
        result.n_samples_seen_ = int(n)
        result.column_scales_ = scales.copy()
        return result


def portable_feature_scale_pipeline(pipeline, *, scale_steps: Mapping[str, Mapping]):
    """Replace fitted foreign scaler steps with neutral TeAL equivalents.

    Call during GeCo export, while foreign classes are still importable.
    Only explicitly listed top-level Pipeline steps are replaced. Remaining
    fitted estimators are reused without cloning so fitted state is preserved.
    The returned Pipeline can be serialized/reopened without the foreign scaler
    module, provided no other step refers to that module.
    """
    from sklearn.pipeline import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a fitted sklearn Pipeline.")
    if not isinstance(scale_steps, Mapping) or not scale_steps:
        raise ValueError("scale_steps must specify named fitted scaler states.")
    names = {name for name, _ in pipeline.steps}
    unknown = set(scale_steps) - names
    if unknown:
        raise ValueError(f"Unknown Pipeline scaler step(s): {sorted(unknown)}")
    steps = [
        (name, FeatureScaleNormalizer.from_fitted_state(scale_steps[name])
         if name in scale_steps else estimator)
        for name, estimator in pipeline.steps
    ]
    return Pipeline(
        steps, transform_input=pipeline.transform_input,
        memory=pipeline.memory, verbose=pipeline.verbose,
    )
