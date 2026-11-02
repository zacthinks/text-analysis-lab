"""Finite-population CDF confidence-set inversion for quantiles.

Inverts one-sided inequalities F(m-) < p <= F(m); does NOT test F(m)=p.
Non-SRS intervals are conservative-in-multiplicity normal approximations,
not finite-sample coverage guarantees. Returns unbounded endpoints if the
observed CDF grid cannot identify a supported bound on the true quantile.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def cdf_quantile_interval(
    thresholds, cdf, cdf_se, *, probability: float, level: float,
    population_size: int, point: float | None = None,
) -> tuple[float, float]:
    """Invert threshold CDF inequalities to obtain a convex-hull interval.

    A lower endpoint is established only at a threshold t for which the
    upper CDF confidence limit is below p (therefore t < the quantile).
    An upper endpoint is established only where the lower CDF limit >= p
    (therefore the quantile <= t). Missing support remains +/- infinity.
    Bonferroni uses an upper bound of 2*N candidate fixed support values;
    because normal SEs are estimated and the grid is partly audit-dependent,
    nominal coverage is approximate, NOT an exact finite-sample assertion.

    Zero estimated SE in a partial audit is not proof of zero uncertainty:
    zero-SE knots receive no exclusion power.
    """
    if not (0 < probability < 1 and 0 < level < 1):
        raise ValueError("Probability and confidence level must lie in (0,1).")
    t = np.asarray(thresholds, dtype=float)
    f = np.asarray(cdf, dtype=float)
    se = np.asarray(cdf_se, dtype=float)
    if (not len(t) or len(t) != len(f) or len(t) != len(se)
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(f))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("CDF threshold grid must be strictly increasing and finite.")
    if population_size < 1:
        raise ValueError("population_size must be positive.")
    z = NormalDist().inv_cdf(
        1 - (1 - level) / (4 * max(1, population_size))
    )
    use = np.isfinite(se) & (se > 0)
    lo = -math.inf
    hi = math.inf
    if np.any(use):
        lower_exclusions = use & ((f + z * se) < probability)
        upper_exclusions = use & ((f - z * se) >= probability)
        if np.any(lower_exclusions):
            lo = float(t[np.flatnonzero(lower_exclusions)[-1]])
        if np.any(upper_exclusions):
            hi = float(t[np.flatnonzero(upper_exclusions)[0]])
    if point is not None:
        # A point estimate outside the raw-CDF inverted set can occur under
        # isotonic projection. Reporting the convex hull is transparent and
        # cannot reduce coverage compared with the unadjusted confidence set.
        lo = min(lo, float(point))
        hi = max(hi, float(point))
    if lo > hi:
        raise ValueError("CDF inversion produced inconsistent bounds.")
    return lo, hi
