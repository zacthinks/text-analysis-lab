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
        lower, upper = result.confidence_interval()
        assert lower <= result.estimate <= upper
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


@pytest.mark.parametrize("design", ["stratified_srswor", "bernoulli"])
def test_threshold_cdf_independent_ordinary_gd_parity(tmp_path, design):
    project = teal.Project.create(tmp_path / design, name="threshold_parity")
    try:
        q_values = [0., 3., 1., 5., 2., 4.]
        y_values = {4: 5., 0: 2., 3: 1., 1: 6.}
        proxy = reg(project, range(6), "prediction", q_values)
        gold = reg(project, [3, 0, 4, 1], "label",
                   [y_values[i] for i in [3, 0, 4, 1]])
        if design == "stratified_srswor":
            # Two 3-unit strata, two observations per stratum.
            probs = [.6666666666666666] * 4
            h = reg(project, range(6), "stratum", [0, 0, 0, 1, 1, 1])
        else:
            probs = [.8, .25, .5, .75]
            h = None
        pi = reg(project, [1, 4, 0, 3], "pi", probs)
        result = proxy.analysis.generalized_difference(
            gold=gold, pi=pi, design=design, strata=h, statistic="median"
        )
        for row in result.cdf_frame().itertuples():
            t = row.threshold
            qt = reg(project, range(6), "prediction",
                     [float(v <= t) for v in q_values])
            yt = reg(project, [3, 0, 4, 1], "label",
                     [float(y_values[i] <= t) for i in [3, 0, 4, 1]])
            reference = qt.analysis.generalized_difference(
                gold=yt, pi=pi, design=design, strata=h
            )
            assert row.raw_cdf == pytest.approx(reference.estimated_mean)
            assert row.cdf_se**2 == pytest.approx(reference.variance_audit)
    finally:
        project.close()


def test_gd_cdf_design_validation_called_once_per_result(tmp_path, monkeypatch):
    from text_analysis_lab.analysis import gd_quantile_design
    project = teal.Project.create(tmp_path / "once", name="once")
    try:
        q = reg(project, range(10), "prediction", list(range(10)))
        y = reg(project, [0, 2, 4, 6, 8], "label", [2., 3., 1., 5., 7.])
        pi = reg(project, [8, 4, 0, 6, 2], "pi", [.5] * 5)
        from text_analysis_lab.analysis import generalized_difference as gd
        real = gd_quantile_design.prepare_threshold_design
        count = [0]
        def counted(**kwargs):
            count[0] += 1
            return real(**kwargs)
        monkeypatch.setattr(gd_quantile_design, "prepare_threshold_design", counted)
        result = q.analysis.generalized_difference(
            gold=y, pi=pi, statistic="median", design="srswor"
        )
        assert len(result.cdf_frame()) >= 10
        assert count[0] == 1
    finally:
        project.close()


def test_gd_quartile_reuses_corrected_cdf_and_preserves_median(tmp_path):
    project = teal.Project.create(tmp_path / "quartile", name="quartile")
    try:
        q = reg(project, range(6), "prediction", [1., 2., 3., 4., 5., 6.])
        gold = reg(project, range(6), "label", [1., 2., 3., 4., 5., 6.])
        pi = reg(project, range(6), "pi", [1.] * 6)
        first = q.analysis.generalized_difference(
            gold=gold, pi=pi, statistic="quantile", quantile=.25,
            design="bernoulli",
        )
        median = q.analysis.generalized_difference(
            gold=gold, pi=pi, statistic="median", design="bernoulli",
        )
        assert first.estimate == 2.
        assert median.estimate == 3.
        assert first.confidence_interval() == (2., 2.)
        assert first.quantile_level == .25
        explicit_median = q.analysis.generalized_difference(
            gold=gold, pi=pi, statistic="quantile", quantile=.5,
            design="bernoulli",
        )
        assert explicit_median.statistic == "quantile"
        assert median.statistic == "median"
    finally:
        project.close()
