"""Discrete weighted-CDF inversion for direct finite-population quantiles.

These point estimators are not means. No design-based variance or confidence
interval is inferred from HT mean-variance formulas.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError


def left_weighted_quantile(values, weights, *, probability: float) -> float:
    """Smallest observed y with normalized weighted empirical CDF >= p.

    The empirical CDF is sum(w_i * 1[y_i <= t]) / sum(w_i), using strictly
    positive weights. For probability audits w_i=1/pi_i; this is Hájek
    normalization, *not* the unnormalized HT CDF with fixed denominator N.
    Equal outcomes form one CDF jump. At exact boundaries we take the lower
    observed value (the left quantile); no interpolation or averaging.
    """
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        raise ValueError("quantile probability must be strictly between 0 and 1.")
    p = float(probability)
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("quantile probability must be strictly between 0 and 1.")
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or len(y) != len(w) or not len(y):
        raise ArtifactError("Quantile values and weights must be nonempty aligned vectors.")
    if not np.all(np.isfinite(y)):
        raise ArtifactError("Quantile measurements must be finite numeric values.")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ArtifactError("Quantile weights must be finite and positive.")

    # Normalize before accumulation to avoid overflow and improve stability.
    w = w / np.max(w)
    order = np.argsort(y, kind="stable")
    sorted_y, sorted_w = y[order], w[order]
    distinct, first = np.unique(sorted_y, return_index=True)
    jumps = np.add.reduceat(sorted_w, first)
    total = float(math.fsum(jumps))
    if not math.isfinite(total) or total <= 0:
        raise ArtifactError("Quantile weight sum must be finite and positive.")
    target = p * total
    # Group ties explicitly; ties must not depend on input row order.
    mass = 0.0
    for index, jump in enumerate(jumps):
        mass += float(jump)
        # For numerical boundary cases, do not accidentally jump past equality.
        if mass >= target or np.isclose(mass, target, rtol=0.0, atol=4*np.spacing(total)):
            return float(distinct[index])
    return float(distinct[-1])


def finite_srs_quantile_interval(
    values, *, population_size: int, probability: float, level: float,
) -> tuple[float, float]:
    """Conservative exact design interval for the left finite-frame quantile.

    Under uniform SRSWOR, let k=ceil(p*N), the target population order
    statistic. The sample count of population values <= x_(k) stochastically
    dominates Hypergeom(N,k,n), including ties; the count of values strictly
    below x_(k) is dominated by Hypergeom(N,k-1,n). Choose the most
    informative lower/upper sample ranks with each tail failure <= alpha/2.
    Bonferroni gives >= 1-alpha coverage for every fixed finite population,
    including arbitrary ties and mass points. Return infinite bounds when
    no sampled order statistic can support that tail. No distribution/density
    assumptions, interpolation or observed-support truncation are needed.
    """
    from scipy.stats import hypergeom
    if not 0.0 < level < 1.0:
        raise ValueError("level must satisfy 0 < level < 1.")
    if not 0.0 < probability < 1.0:
        raise ValueError("quantile probability must satisfy 0 < p < 1.")
    ordered = np.sort(np.asarray(values, dtype=float))
    n, N = len(ordered), population_size
    if not (1 <= n <= N):
        raise ValueError("SRS quantile interval requires 1 <= n <= N.")
    if not np.all(np.isfinite(ordered)):
        raise ValueError("Quantile interval measurements must be finite.")
    if n == N:
        return (
            float(ordered[math.ceil(probability * N) - 1]),
            float(ordered[math.ceil(probability * N) - 1]),
        )
    k = math.ceil(probability * N)
    tail = (1.0 - level) / 2.0

    # Lower failure: fewer than rank a observations <= true quantile.
    lower_ranks = [
        a for a in range(1, n + 1)
        if hypergeom.cdf(a - 1, N, k, n) <= tail
    ]
    lower = float(ordered[max(lower_ranks) - 1]) if lower_ranks else -math.inf

    # Upper failure: at least b observations strictly below true quantile.
    upper_ranks = [
        b for b in range(1, n + 1)
        if hypergeom.sf(b - 1, N, k - 1, n) <= tail
    ]
    upper = float(ordered[min(upper_ranks) - 1]) if upper_ranks else math.inf
    return lower, upper


@dataclass(frozen=True)
class QuantileEstimateResult:
    """Finite quantile: exact SRS rank CI where supported, otherwise point only."""

    statistic: str
    estimate: float
    quantile_level: float
    population_size: int
    audit_size: int
    inference: str
    weighting: str
    total_weight: float
    variance_design: str | None
    variance_note: str
    standard_error: float | None = None
    _cdf: pd.DataFrame | None = None
    _srs_order: tuple[float, ...] | None = None

    def cdf_frame(self) -> pd.DataFrame:
        """Return weighted CDF estimates and design ratio-linearization SEs.

        These are pointwise CDF diagnostics, not a simultaneous band or a
        validated confidence interval for the finite-population quantile.
        """
        if self._cdf is None:
            raise ValueError("CDF diagnostics were not retained for this result.")
        return self._cdf.copy()

    @property
    def estimated_median(self) -> float:
        if self.statistic != "median":
            raise ValueError("estimated_median is available only for median results.")
        return self.estimate

    def confidence_interval(self, *, level: float = 0.95) -> tuple[float, float]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must satisfy 0 < level < 1.")
        if self.standard_error == 0.0:
            return self.estimate, self.estimate
        from text_analysis_lab.analysis.quantile_intervals import (
            cdf_quantile_interval, require_finite_quantile_interval,
        )
        if self._srs_order is not None:
            bounds = finite_srs_quantile_interval(
                self._srs_order, population_size=self.population_size,
                probability=self.quantile_level, level=level,
            )
            return require_finite_quantile_interval(
                bounds, design="srswor", method="exact finite-population SRS rank",
            )
        if self._cdf is not None and self.variance_design in {"stratified_srswor", "bernoulli"}:
            bounds = cdf_quantile_interval(
                self._cdf["threshold"], self._cdf["weighted_cdf"],
                self._cdf["cdf_se"], probability=self.quantile_level,
                level=level, population_size=self.population_size,
                point=self.estimate,
            )
            return require_finite_quantile_interval(
                bounds, design=self.variance_design,
                method="approximate design-weighted CDF inversion",
            )
        raise ValueError(
            "Quantile confidence intervals require a separately validated "
            "quantile procedure; mean-based SEs are not applicable."
        )

    def p_value(self, *, null: float = 0.0) -> float:
        raise ValueError(
            "Quantile hypothesis tests are not available without a validated "
            "quantile inference procedure."
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "statistic": self.statistic,
            "estimate": self.estimate,
            "quantile_level": self.quantile_level,
            "population_size": self.population_size,
            "audit_size": self.audit_size,
            "inference": self.inference,
            "weighting": self.weighting,
            "total_weight": self.total_weight,
            "variance_design": self.variance_design,
            "variance_note": self.variance_note,
            "standard_error": self.standard_error,
        }])
