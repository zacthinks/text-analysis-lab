"""Finite-bound guard used by the supported exact SRS rank interval."""
import math

import pytest

from text_analysis_lab.analysis.quantile_intervals import require_finite_quantile_interval


@pytest.mark.parametrize("bounds,missing", [
    ((-math.inf, math.inf), "lower and upper"),
    ((-math.inf, 2.), "lower"),
    ((1., math.inf), "upper"),
])
def test_exact_rank_interval_rejects_open_support(bounds, missing):
    with pytest.raises(ValueError, match="unavailable") as exc:
        require_finite_quantile_interval(bounds, design="srswor", method="exact rank")
    assert missing in str(exc.value)


def test_exact_rank_interval_allows_finite_limits():
    assert require_finite_quantile_interval(
        (1., 2.), design="srswor", method="exact rank",
    ) == (1., 2.)
