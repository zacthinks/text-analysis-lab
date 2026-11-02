"""Finite-bound requirement for supported exact SRS quantile rank intervals."""
from __future__ import annotations

import math


def require_finite_quantile_interval(
    bounds: tuple[float, float], *, design: str | None, method: str,
) -> tuple[float, float]:
    """Expose only finite informative quantile intervals to public callers.

    An open support endpoint is a valid intermediate inversion result, but it
    cannot be advertised as an available researcher-facing confidence interval.
    Never substitute observed/proxy minima or maxima for unbounded endpoints.
    """
    lower, upper = map(float, bounds)
    missing = [
        side for side, value in (("lower", lower), ("upper", upper))
        if not math.isfinite(value)
    ]
    if missing:
        description = " and ".join(missing)
        raise ValueError(
            f"Quantile confidence interval unavailable for design {design!r} "
            f"({method}): the audit cannot establish a finite {description} "
            "bound at the requested confidence level. The point estimate and "
            "CDF diagnostics remain available; no support limits were imputed."
        )
    return lower, upper
