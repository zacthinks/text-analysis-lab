"""Confidence-set inversion boundaries and conservative failure behavior."""
import math
import numpy as np
import pytest
from text_analysis_lab.analysis.quantile_intervals import cdf_quantile_interval


def test_cdf_interval_does_not_confuse_mass_point_with_cdf_equality():
    lo, hi = cdf_quantile_interval(
        [0., 1., 2.], [.1, .9, 1.], [.08, .08, .0],
        probability=.5, level=.9, population_size=100, point=1.,
    )
    assert lo <= 1 <= hi


def test_cdf_interval_returns_unbounded_sides_without_information():
    assert cdf_quantile_interval(
        [1., 2.], [.1, .9], [0., 0.],
        probability=.5, level=.95, population_size=30,
    ) == (-math.inf, math.inf)
    assert cdf_quantile_interval(
        [1., 2.], [.1, .9], [float("nan"), float("nan")],
        probability=.5, level=.95, population_size=30,
    ) == (-math.inf, math.inf)


def test_cdf_interval_probability_and_grid_validation():
    with pytest.raises(ValueError):
        cdf_quantile_interval(
            [1., 1.], [.2, .8], [.1, .1],
            probability=.5, level=.95, population_size=30,
        )


def test_quantile_interval_grid_inversion_is_conservative_on_ties():
    # With strong mass points and finite estimated SE, a rank inversion must
    # never exclude the estimated point simply because F(m) != probability.
    for p in (.25, .5, .75):
        lo, hi = cdf_quantile_interval(
            [0., 1., 2., 3.], [0., .05, .85, 1.],
            [.04, .07, .07, .04],
            probability=p, level=.95, population_size=200, point=2.,
        )
        assert lo <= 2. <= hi


def test_quantile_interval_can_report_finite_limits_with_strong_information():
    lo, hi = cdf_quantile_interval(
        [0., 1., 2., 3.], [.0, .2, .8, 1.],
        [.01, .01, .01, .01],
        probability=.5, level=.95, population_size=20, point=2.,
    )
    assert lo == 1.
    assert hi == 2.
