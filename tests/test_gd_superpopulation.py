"""Numerical and small nested-IID-frame checks for superpopulation GD."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.analysis.gd_superpopulation import corpus_variance
from text_analysis_lab.analysis.generalized_difference import generalized_difference
from text_analysis_lab.core.errors import ArtifactError


def _register(project, ids, field, values):
    return project.register_external(
        pd.DataFrame({"row_id": list(ids), field: list(values)}),
        primary_key="row_id", data_fields=field,
    )


@pytest.mark.parametrize("design", ["srswor", "stratified_srswor", "bernoulli"])
def test_census_corpus_variance_is_sample_variance_over_n(design):
    y = np.array([0.0, 1.0, 3.0, 5.0])
    h = np.array([0, 0, 1, 1])
    var = corpus_variance(
        gold=y, inclusion=np.ones(4), population_size=4, design=design,
        stratum_population=h, stratum_audit=h,
    )
    assert var == pytest.approx(np.var(y, ddof=1) / 4)


def test_unequal_bernoulli_corpus_variance_matches_pair_sum():
    y = np.array([0., 1., 4.])
    pi = np.array([.25, .5, 1.])
    expected = sum(
        (y[i] - y[j]) ** 2 / (pi[i] * pi[j])
        for i in range(3) for j in range(i + 1, 3)
    ) / (5 * 5 * 4)
    assert corpus_variance(
        gold=y, inclusion=pi, population_size=5, design="bernoulli"
    ) == pytest.approx(expected)


def test_disproportionate_strata_corpus_variance_is_not_audit_variance():
    # A has four frame units, B two; non-proportional audit has two from each.
    y = np.array([0., 2., 10., 14.])
    h_sample = np.array(["A", "A", "B", "B"])
    h_full = np.array(["A"] * 4 + ["B"] * 2)
    estimated = corpus_variance(
        gold=y, inclusion=np.array([.5, .5, 1., 1.]),
        population_size=6, design="stratified_srswor",
        stratum_population=h_full, stratum_audit=h_sample,
    )
    # Population within pair sums: A 4*3*2, B 2*1*8.
    # Between strata: 4*2 * ((1/2)*2 + (1/2)*8 + (1-12)**2).
    expected = (24 + 16 + 8 * (1 + 4 + 121)) / (36 * 5)
    assert estimated == pytest.approx(expected)


def test_real_project_superpopulation_is_larger_than_finite_for_census(tmp_path):
    project = teal.Project.create(tmp_path / "iid", name="iid")
    try:
        ids = np.arange(12)
        source = _register(project, ids, "text", [str(i) for i in ids])
        surrogate = _register(project, ids, "prediction", np.zeros(12))
        gold_values = np.array([0.0, 1.0] * 6)
        # Old API requires nonempty remainder; Bernoulli p=1 permits census.
        selected = project.probability_sample(source, probability=1.0, random_state=3)
        gold = _register(project, ids[::-1], "label", gold_values[::-1])
        finite = generalized_difference(surrogate, gold=gold, pi=selected["pi"])
        outer = generalized_difference(
            surrogate, gold=gold, pi=selected["pi"], inference="superpopulation"
        )
        assert finite.variance_design == "bernoulli"
        assert finite.standard_error == pytest.approx(0.)
        assert outer.variance_audit == pytest.approx(0.)
        assert outer.variance_corpus == pytest.approx(
            np.var(gold_values, ddof=1) / 12
        )
        assert outer.standard_error**2 == pytest.approx(
            outer.variance_corpus
        )
        assert outer.estimated_mean == pytest.approx(finite.estimated_mean)
        assert outer.inference == "superpopulation"
    finally:
        project.close()


def test_nested_iid_srs_monte_carlo_sanity():
    rng = np.random.default_rng(95)
    N, n = 80, 28
    estimates, variances = [], []
    for _ in range(180):
        x = rng.normal(size=N)
        errors = rng.normal(scale=.5, size=N)
        y = x + errors
        sample = rng.choice(N, n, replace=False)
        e = errors[sample]
        estimate = float(np.mean(x) + np.mean(e))
        audit_var = (1 - n / N) * np.var(e, ddof=1) / n
        outer_var = float(np.var(y[sample], ddof=1) / N)
        estimates.append(estimate)
        variances.append(audit_var + outer_var)
    observed = float(np.var(estimates, ddof=1))
    predicted = float(np.mean(variances))
    assert observed == pytest.approx(predicted, rel=.35)
    assert abs(float(np.mean(estimates))) < .15



@pytest.mark.parametrize("design", ["srswor", "stratified_srswor", "bernoulli"])
def test_external_superpopulation_components_are_explicit(tmp_path, design):
    project = teal.Project.create(tmp_path / design, name="iid_external")
    try:
        ids = np.arange(12)
        surrogate = _register(project, ids, "prediction", ids / 15)
        audit_ids = np.array([0, 1, 2, 3, 6, 7, 8, 9])
        audit_labels = np.array([0., 1., 2., 1., 2., 0., 1., 3.])
        gold = _register(project, audit_ids[::-1], "label", audit_labels[::-1])
        strata = _register(project, ids, "H", ["A"] * 6 + ["B"] * 6)
        probabilities = (
            [.5, .8, .6, .7, .5, .8, .6, .7] if design == "bernoulli"
            else [2 / 3] * 8
        )
        pi = _register(project, audit_ids, "pi", probabilities)
        kwargs = {"design": design}
        if design == "stratified_srswor":
            kwargs["strata"] = strata
        finite = generalized_difference(surrogate, gold=gold, pi=pi, **kwargs)
        superpop = generalized_difference(
            surrogate, gold=gold, pi=pi,
            inference="superpopulation", **kwargs
        )
        assert superpop.inference == "superpopulation"
        assert superpop.variance_audit == pytest.approx(finite.standard_error**2)
        assert superpop.variance_corpus >= 0
        assert superpop.standard_error**2 == pytest.approx(
            superpop.variance_audit + superpop.variance_corpus
        )
        assert superpop.estimated_mean == pytest.approx(finite.estimated_mean)
        assert "variance_corpus" in superpop.to_frame().columns
        assert "variance_corpus" not in finite.to_frame().columns
    finally:
        project.close()



def test_stratified_corpus_variance_exact_enumeration():
    """Average over every valid audit: the estimated finite-frame corpus term
    must equal the known S_Y**2 / N, including the cross-stratum term."""
    from itertools import combinations

    full_y = np.array([0., 2., 4., 10., 14.])
    full_h = np.array(["A"] * 3 + ["B"] * 2)
    estimates = []
    for selected_a in combinations(range(3), 2):
        sample_ids = np.array([*selected_a, 3, 4])
        estimates.append(corpus_variance(
            gold=full_y[sample_ids],
            inclusion=np.array([2 / 3, 2 / 3, 1., 1.]),
            population_size=5, design="stratified_srswor",
            stratum_population=full_h, stratum_audit=full_h[sample_ids],
        ))
    assert np.mean(estimates) == pytest.approx(
        np.var(full_y, ddof=1) / 5, abs=1e-12
    )


@pytest.mark.parametrize("design", ["srswor", "stratified_srswor", "bernoulli"])
def test_nested_iid_interval_calibration_for_supported_designs(design):
    """Small seeded nested frame-and-audit check; no heavy simulation suite."""
    rng = np.random.default_rng({"srswor": 771, "stratified_srswor": 772,
                                  "bernoulli": 773}[design])
    N = 96
    actual = []
    predicted = []
    covered = []
    for _ in range(240):
        x = rng.normal(size=N)
        y = x + rng.normal(scale=.75, size=N)
        q = 0.7 * x
        residual = y - q
        if design == "srswor":
            idx = rng.choice(N, size=40, replace=False)
            pi = np.full(40, 40 / N)
            audit_v = (1 - 40 / N) * np.var(residual[idx], ddof=1) / 40
            corpus = corpus_variance(
                gold=y[idx], inclusion=pi, population_size=N, design=design
            )
        elif design == "stratified_srswor":
            # Deterministic frame strata of unequal sizes and allocations.
            h = np.array(["A"] * 64 + ["B"] * 32)
            idx = np.concatenate([
                rng.choice(64, size=16, replace=False),
                64 + rng.choice(32, size=24, replace=False),
            ])
            pi = np.array([.25] * 16 + [.75] * 24)
            audit_v = (
                (64 / N)**2 * (1 - .25) * np.var(residual[idx[:16]], ddof=1) / 16
                + (32 / N)**2 * (1 - .75) * np.var(residual[idx[16:]], ddof=1) / 24
            )
            corpus = corpus_variance(
                gold=y[idx], inclusion=pi, population_size=N,
                design=design, stratum_population=h, stratum_audit=h[idx],
            )
        else:
            full_pi = np.where(np.arange(N) % 3 == 0, .35, .7)
            idx = np.flatnonzero(rng.random(N) < full_pi)
            if len(idx) < 2:
                continue
            pi = full_pi[idx]
            audit_v = float(np.sum(
                (1 - pi) * (residual[idx] / pi)**2
            ) / N**2)
            corpus = corpus_variance(
                gold=y[idx], inclusion=pi, population_size=N, design=design
            )
        estimate = float(np.mean(q) + np.sum(residual[idx] / pi) / N)
        total_v = audit_v + corpus
        actual.append(estimate)
        predicted.append(total_v)
        covered.append(abs(estimate) <= 1.96 * np.sqrt(total_v))
    assert np.var(actual, ddof=1) == pytest.approx(
        np.mean(predicted), rel=.35
    )
    assert 0.84 <= np.mean(covered) <= 0.99
