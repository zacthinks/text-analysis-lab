"""GD quantile point and threshold-CDF regression tests."""
import numpy as np
import pandas as pd
import pytest
import text_analysis_lab as teal
from text_analysis_lab.analysis.gd_quantiles import bounded_isotonic_cdf
from text_analysis_lab.core.errors import ArtifactError


def reg(project, ids, name, values):
    return project.register_external(
        pd.DataFrame({"row_id": list(ids), name: list(values)}),
        primary_key="row_id", data_fields=name,
    )


def test_projection_and_deterministic_boundaries():
    assert bounded_isotonic_cdf(np.array([.8, -.2, 1.])) == pytest.approx([.3, .3, 1.])
    assert bounded_isotonic_cdf(np.array([1.2, 1.4, 1.])) == pytest.approx([1., 1., 1.])
    assert bounded_isotonic_cdf(np.array([1.])) == pytest.approx([1.])


def test_gd_median_thresholdwise_matches_manual_and_mean_gd(tmp_path):
    project = teal.Project.create(tmp_path / "gd_median", name="median")
    try:
        q = reg(project, [3, 0, 2, 1], "prediction", [4., 1., 3., 2.])
        gold = reg(project, [2, 0], "label", [2., 3.])
        pi = reg(project, [0, 2], "pi", [.5, .5])
        result = q.analysis.generalized_difference(
            gold=gold, pi=pi, design="srswor", statistic="median",
        )
        table = result.cdf_frame()
        assert table["threshold"].tolist() == [1., 2., 3., 4.]
        assert result.variance_design == "srswor"
        assert result.standard_error is None
        assert result.estimate == pytest.approx(2.)
        assert table["raw_cdf"].iloc[-1] == pytest.approx(1.)
        q_source = {0: 1., 1: 2., 2: 3., 3: 4.}
        y_source = {0: 3., 2: 2.}
        for t, raw in zip(table["threshold"], table["raw_cdf"]):
            proxy_total = sum(float(v <= t) for v in q_source.values())
            correction = sum(
                (float(y_source[i] <= t) - float(q_source[i] <= t)) / .5
                for i in (0, 2)
            )
            assert raw == pytest.approx((proxy_total + correction) / 4)
        with pytest.raises(ValueError, match="unavailable"):
            result.confidence_interval()
        assert "pointwise_lower" in result.cdf_frame(level=.95)
    finally:
        project.close()


def test_gd_median_census_and_inference_restrictions(tmp_path):
    project = teal.Project.create(tmp_path / "gd_median_census", name="median")
    try:
        q = reg(project, range(4), "prediction", [9., 8., 7., 6.])
        y = reg(project, [3, 1, 0, 2], "label", [4., 2., 1., 3.])
        pi = reg(project, [0, 1, 2, 3], "pi", [1.] * 4)
        result = q.analysis.generalized_difference(
            gold=y, pi=pi, statistic="median", design="bernoulli",
        )
        assert result.estimate == 2.
        assert result.confidence_interval() == (2., 2.)
        with pytest.raises(ValueError, match="finite"):
            q.analysis.generalized_difference(
                gold=y, pi=pi, statistic="median", inference="superpopulation",
            )
        with pytest.raises(ValueError, match="known="):
            q.analysis.generalized_difference(
                gold=y, pi=pi, statistic="median", known=y,
            )
    finally:
        project.close()


def test_gd_median_rejects_invalid_asserted_srs(tmp_path):
    project = teal.Project.create(tmp_path / "gd_bad_design", name="bad")
    try:
        q = reg(project, range(4), "prediction", [1., 2., 3., 4.])
        gold = reg(project, [0, 2], "label", [2., 3.])
        pi = reg(project, [2, 0], "pi", [.7, .5])
        with pytest.raises(ArtifactError, match="SRSWOR pi contradicts"):
            q.analysis.generalized_difference(
                gold=gold, pi=pi, statistic="median", design="srswor",
            )
    finally:
        project.close()
