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
