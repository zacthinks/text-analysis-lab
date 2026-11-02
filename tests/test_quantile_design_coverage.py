"""Reproducible fixed-frame randomization diagnostics for approximate CDF CIs.

This is NOT an exact-coverage theorem or a guarantee for arbitrary designs.
Tests record coverage and informativeness across distinct randomization laws.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from text_analysis_lab.analysis.quantile_intervals import cdf_quantile_interval


def _cdf_ci(y, q, selected, pi, design, strata, level):
    n = len(y)
    values = np.unique(y[selected] if q is None else np.concatenate((y[selected], q)))
    sample_y = y[selected]
    weights = 1. / pi[selected]
    weight_total = np.sum(weights)
    pointwise = []
    standard_errors = []
    for t in values:
        indicator = (sample_y <= t).astype(float)
        if q is None:
            cdf = float(np.sum(indicator * weights) / weight_total)
            residual = indicator - cdf
        else:
            qa = (q[selected] <= t).astype(float)
            cdf = float((np.sum(q <= t) + np.sum((indicator - qa) * weights)) / n)
            residual = indicator - qa
        if design == "bernoulli":
            var = np.sum(((1. - pi[selected]) / pi[selected]**2) * residual**2)
        elif design == "stratified_srswor":
            var = 0.
            for h in np.unique(strata):
                members = np.flatnonzero(strata == h)
                mask = strata[selected] == h
                nh, Nh = int(mask.sum()), len(members)
                if nh < 2:
                    return None
                var += Nh**2 * (1. - nh / Nh) * np.var(residual[mask], ddof=1) / nh
        else:
            if len(selected) < 2:
                return None
            var = n**2 * (1. - len(selected) / n) * np.var(residual, ddof=1) / len(selected)
        denom = weight_total if q is None else n
        pointwise.append(cdf)
        standard_errors.append(math.sqrt(max(0., var)) / denom)
    return cdf_quantile_interval(
        values, pointwise, standard_errors, probability=.5,
        level=level, population_size=n,
    )


@pytest.mark.parametrize("design", ["srswor", "stratified_srswor", "bernoulli"])
@pytest.mark.parametrize("kind", ["continuous", "tied"])
@pytest.mark.parametrize("proxy", ["direct", "good", "shifted"])
@pytest.mark.parametrize("level", [.9, .95])
def test_repeated_design_quantile_ci_coverage_diagnostics(
    design, kind, proxy, level, record_property,
):
    # Fixed finite frame. No outcomes are resampled; only audit selection is.
    N = 40
    rng = np.random.default_rng(20261102)
    y = rng.normal(size=N) + np.linspace(-.5, .5, N)
    if kind == "tied":
        y = np.floor(y * 1.5)
    q = None if proxy == "direct" else (y + rng.normal(scale=.15, size=N)
                                          if proxy == "good" else y[::-1] + 1.5)
    if design == "srswor":
        p = np.full(N, .5)
    elif design == "stratified_srswor":
        p = np.r_[np.full(N//2, .3), np.full(N//2, .7)]
    else:
        p = np.linspace(.2, .8, N)
    strata = np.r_[np.zeros(N//2, dtype=int), np.ones(N//2, dtype=int)]
    true_median = float(sorted(y)[math.ceil(.5 * N) - 1])
    total, covered, both_finite, both_infinite = 0, 0, 0, 0
    finite_widths = []
    for _ in range(160):
        if design == "srswor":
            selected = rng.choice(N, size=N//2, replace=False)
        elif design == "stratified_srswor":
            selected = np.r_[
                rng.choice(N//2, size=6, replace=False),
                rng.choice(np.arange(N//2, N), size=14, replace=False),
            ]
        else:
            selected = np.flatnonzero(rng.random(N) < p)
            if len(selected) < 2:
                continue
        ci = _cdf_ci(y, q, selected, p, design, strata, level)
        if ci is None:
            continue
        lo, hi = ci
        total += 1
        covered += int(lo <= true_median <= hi)
        both_finite += math.isfinite(lo) and math.isfinite(hi)
        both_infinite += math.isinf(lo) and math.isinf(hi)
        if math.isfinite(lo) and math.isfinite(hi):
            finite_widths.append(hi - lo)
    assert total >= 150
    coverage = covered / total
    finite_fraction = both_finite / total
    unbounded_fraction = both_infinite / total
    record_property("empirical_coverage", float(coverage))
    record_property("both_finite_fraction", float(finite_fraction))
    record_property("both_unbounded_fraction", float(unbounded_fraction))
    record_property("median_finite_width", float(np.median(finite_widths))
                    if finite_widths else None)
    # This is a diagnostic, not a pass assertion of nominal coverage. A
    # methodologically informed acceptance decision still needs the rates.
    assert 0 <= coverage <= 1
