"""Direct probability-audit HT estimates without surrogate predictions."""
import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.analysis import estimate
from text_analysis_lab.core.errors import ArtifactError


def _register(project, ids, name, values):
    return project.register_external(
        pd.DataFrame({"row_id": list(ids), name: list(values)}),
        primary_key="row_id", data_fields=name,
    )


@pytest.mark.parametrize("kind", ["srswor", "stratified_srswor", "bernoulli"])
def test_direct_estimation_trusted_sampling_and_reopen(tmp_path, kind):
    root = tmp_path / kind
    project = teal.Project.create(root, name="direct")
    try:
        ids = np.arange(40)
        y = (ids % 5 < 2).astype(float)
        frame = _register(project, ids, "text", [str(i) for i in ids])
        strata = _register(project, ids, "H", [0] * 20 + [1] * 20)
        if kind == "srswor":
            sampled = project.probability_sample(frame, n=20, random_state=7)
        elif kind == "stratified_srswor":
            sampled = project.probability_sample(
                frame, n=20, strata=strata, random_state=7
            )
        else:
            sampled = project.probability_sample(
                frame, probability=0.6, random_state=7
            )
        keys = sampled["sample"].query(
            key_columns=True, data_columns=False,
            metadata_columns=False, form="table"
        )["row_id"].to_numpy(dtype=int)
        gold = _register(project, keys[::-1], "label", y[keys[::-1]])
        result = estimate(gold, pi=sampled["pi"])
        assert result.estimated_total == pytest.approx(
            sum(float(y[int(i)]) / float(p)
                for i, p in zip(
                    sampled["pi"].query(
                        key_columns=True, data_columns=["pi"],
                        metadata_columns=False, form="table"
                    )["row_id"],
                    sampled["pi"].query(
                        key_columns=True, data_columns=["pi"],
                        metadata_columns=False, form="table"
                    )["pi"],
                ))
        )
        assert result.variance_design == kind
        assert result.population_size == 40
        assert result.standard_error is not None
        assert gold.analysis.estimate(pi=sampled["pi"]).estimate == pytest.approx(
            result.estimate
        )
        ids_saved = [gold.artifact_id, sampled["pi"].artifact_id]
        expected = result.estimate, result.standard_error
    finally:
        project.close()
    reopened = teal.Project.open(root)
    try:
        g, p = [reopened.get_artifact(x) for x in ids_saved]
        new = estimate(g, pi=p)
        assert (new.estimate, new.standard_error) == pytest.approx(expected)
    finally:
        reopened.close()


def test_external_srs_and_superpopulation_census(tmp_path):
    project = teal.Project.create(tmp_path / "external", name="direct")
    try:
        frame = _register(project, range(4), "text", list("abcd"))
        gold = _register(project, [3, 0], "label", [0., 1.])
        pi = _register(project, [0, 3], "pi", [0.5, 0.5])
        with pytest.raises(ValueError, match="population="):
            estimate(gold, pi=pi, design="srswor")
        finite = estimate(gold, pi=pi, population=frame, design="srswor")
        assert finite.estimate == pytest.approx(.5)
        assert finite.standard_error == pytest.approx(np.sqrt(.125))
        all_gold = _register(project, range(4), "label", [0., 1., 1., 0.])
        all_pi = _register(project, [3, 1, 0, 2], "pi", [1.] * 4)
        result = estimate(
            all_gold, pi=all_pi, population=frame,
            design="bernoulli", inference="superpopulation",
        )
        assert result.variance_audit == pytest.approx(0)
        assert result.variance_corpus == pytest.approx(1 / 12)
        assert result.estimated_prevalence == pytest.approx(.5)
    finally:
        project.close()


def test_inconsistent_external_probability_rejected(tmp_path):
    project = teal.Project.create(tmp_path / "bad", name="bad")
    try:
        frame = _register(project, range(4), "text", list("abcd"))
        gold = _register(project, [0, 2], "label", [1., 0.])
        pi = _register(project, [0, 2], "pi", [.5, .7])
        with pytest.raises(ArtifactError, match="SRSWOR"):
            estimate(gold, pi=pi, population=frame, design="srswor")
    finally:
        project.close()


def test_single_human_data_field_is_inferred_and_multiple_fields_are_ambiguous(tmp_path):
    project = teal.Project.create(tmp_path / "field_selection", name="fields")
    try:
        population = _register(project, range(4), "text", list("abcd"))
        gold = _register(project, [0, 2], "human_judgment", [1., 0.])
        pi = _register(project, [2, 0], "pi", [.5, .5])
        inferred = gold.analysis.estimate(
            pi=pi, population=population, design="srswor",
        )
        explicit = gold.analysis.estimate(
            pi=pi, population=population, design="srswor",
            field="human_judgment",
        )
        assert inferred.estimate == pytest.approx(explicit.estimate)
        assert inferred.standard_error == pytest.approx(explicit.standard_error)
        multi = project.register_external(
            pd.DataFrame({
                "row_id": [0, 2],
                "human_judgment": [1., 0.],
                "second_measure": [2., 3.],
            }),
            primary_key="row_id",
            data_fields=["human_judgment", "second_measure"],
        )
        with pytest.raises(ArtifactError, match="field="):
            multi.analysis.estimate(
                pi=pi, population=population, design="srswor",
            )
        assert multi.analysis.estimate(
            pi=pi, population=population, design="srswor",
            field="human_judgment",
        ).estimate == pytest.approx(inferred.estimate)
        with pytest.raises(ArtifactError, match="must expose data field"):
            gold.analysis.estimate(
                pi=pi, population=population, design="srswor",
                field="does_not_exist",
            )
    finally:
        project.close()


def test_full_artifact_naive_prediction_mean_and_statistical_limits(tmp_path):
    project = teal.Project.create(tmp_path / "naive", name="naive")
    try:
        predictions = _register(
            project, [3, 0, 1, 2], "model_prediction", [1., 0., 1., 1.]
        )
        result = predictions.analysis.estimate()
        assert result.estimate == pytest.approx(.75)
        assert result.estimated_total == pytest.approx(3.)
        assert result.population_size == 4
        assert result.audit_size == 0
        assert result.standard_error == pytest.approx(0.)
        assert result.variance_design == "census"
        outer = predictions.analysis.estimate(inference="superpopulation")
        assert outer.standard_error == pytest.approx(
            np.sqrt(np.var([1., 0., 1., 1.], ddof=1) / 4)
        )
        with pytest.raises(ValueError, match="require pi"):
            predictions.analysis.estimate(design="srswor")
        with pytest.raises(ValueError, match="require pi"):
            predictions.analysis.estimate(population=predictions)
    finally:
        project.close()


@pytest.mark.parametrize("kind", ["srswor", "stratified_srswor", "bernoulli"])
def test_explicit_population_must_match_recorded_frame(tmp_path, kind):
    project = teal.Project.create(tmp_path / kind, name="frame_guard")
    try:
        ids = np.arange(40)
        source = _register(project, ids, "text", [str(i) for i in ids])
        strata = _register(project, ids, "H", [0] * 20 + [1] * 20)
        if kind == "srswor":
            sampled = project.probability_sample(source, n=12, random_state=17)
        elif kind == "stratified_srswor":
            sampled = project.probability_sample(
                source, n=12, strata=strata, random_state=17
            )
        else:
            sampled = project.probability_sample(
                source, probability=.5, random_state=17
            )
        selected = sampled["sample"].query(
            key_columns=True, data_columns=False,
            metadata_columns=False, form="table"
        )["row_id"].to_numpy(dtype=int)
        labels = _register(project, selected[::-1], "label", [1.] * len(selected))
        # A supplied frame may contain every audit key but omit unsampled
        # members of the actual source frame, corrupting the denominator.
        reduced = _register(
            project, list(selected) + [i for i in ids if i not in selected][:1],
            "text", ["x"] * (len(selected) + 1),
        )
        with pytest.raises(ArtifactError, match="contradicts the recorded sampling frame"):
            labels.analysis.estimate(pi=sampled["pi"], population=reduced)
        # Equivalent keys in a different artifact and order remain valid.
        same_keys = _register(
            project, ids[::-1], "text", [str(i) for i in ids[::-1]]
        )
        original = labels.analysis.estimate(pi=sampled["pi"])
        supplied = labels.analysis.estimate(pi=sampled["pi"], population=same_keys)
        assert supplied.estimate == pytest.approx(original.estimate)
        assert supplied.standard_error == pytest.approx(original.standard_error)
    finally:
        project.close()


def test_outcome_field_named_pi_does_not_collide_with_inclusion_pi(tmp_path):
    project = teal.Project.create(tmp_path / "collision", name="collision")
    try:
        population = _register(project, range(4), "text", list("abcd"))
        outcome = _register(project, [2, 0], "pi", [1., 0.])
        inclusion = _register(project, [0, 2], "pi", [.5, .5])
        result = outcome.analysis.estimate(
            pi=inclusion, population=population, design="srswor"
        )
        assert result.estimate == pytest.approx(.5)
        assert result.estimated_total == pytest.approx(2.)
        assert result.standard_error == pytest.approx(np.sqrt(.125))
        # A distinct input field and an explicitly named probability field
        # must also work when they share the same field name.
        custom = _register(project, [0, 2], "weight", [.5, .5])
        same = outcome.analysis.estimate(
            pi=custom, pi_field="weight", population=population,
            design="srswor", field="pi"
        )
        assert same.estimate == pytest.approx(result.estimate)
    finally:
        project.close()



def test_hand_calculated_disproportionate_stratified_ht(tmp_path):
    """N_A=4, n_A=2; N_B=6, n_B=2. Hand-computed mean variance 1.04."""
    project = teal.Project.create(tmp_path / "manual_strata", name="manual")
    try:
        frame = _register(project, range(10), "text", ["x"] * 10)
        strata = _register(project, range(10), "stratum", ["A"] * 4 + ["B"] * 6)
        # Sample means 2 and 4, sample variances 2 and 8.
        gold = _register(project, [5, 0, 4, 1], "score", [6., 1., 2., 3.])
        pi = _register(project, [1, 4, 0, 5], "pi", [.5, 1 / 3, .5, 1 / 3])
        result = gold.analysis.estimate(
            pi=pi, population=frame, design="stratified_srswor", strata=strata
        )
        # HT total: 4 * 2 + 6 * 4 = 32.
        assert result.estimated_total == pytest.approx(32.)
        assert result.estimate == pytest.approx(3.2)
        # Variance(total): 4**2*(1-2/4)*2/2 + 6**2*(1-2/6)*8/2 = 104.
        assert result.variance_audit == pytest.approx(104 / 100)
        assert result.standard_error == pytest.approx(np.sqrt(1.04))
    finally:
        project.close()


def test_hand_calculated_unequal_bernoulli_ht_shuffled_rows(tmp_path):
    """The observed HT total and estimated independent-Poisson variance."""
    project = teal.Project.create(tmp_path / "manual_bernoulli", name="manual")
    try:
        frame = _register(project, range(5), "text", ["x"] * 5)
        gold = _register(project, [4, 0, 2], "score", [1., 2., 3.])
        pi = _register(project, [2, 4, 0], "pi", [.25, .8, .5])
        result = gold.analysis.estimate(pi=pi, population=frame, design="bernoulli")
        # HT total: 2/.5 + 3/.25 + 1/.8 = 17.25.
        assert result.estimated_total == pytest.approx(17.25)
        assert result.estimate == pytest.approx(3.45)
        # Variance(total): .5*(2/.5)**2 + .75*(3/.25)**2 +
        #                  .2*(1/.8)**2 = 116.3125.
        assert result.variance_audit == pytest.approx(116.3125 / 25)
        assert result.standard_error == pytest.approx(np.sqrt(4.6525))
    finally:
        project.close()


def test_known_certainty_labels_recombine_with_probability_audit(tmp_path):
    project = teal.Project.create(tmp_path / "known", name="known")
    try:
        # F=8, K={0,1}, U={2,...,7}. The four audited U units
        # have pi=4/6; sum_K y=5, sum_A y=12, total=5+18=23.
        full = _register(project, range(8), "text", ["x"] * 8)
        remainder = _register(project, range(2, 8), "text", ["x"] * 6)
        known1 = _register(project, [0], "score", [2.])
        known2 = _register(project, [1], "score", [3.])
        sample = project.probability_sample(remainder, n=4, random_state=13)
        selected = sample["sample"].query(
            key_columns=True, data_columns=False,
            metadata_columns=False, form="table"
        )["row_id"].tolist()
        outcomes = {i: float(i - 1) for i in selected}
        labels = _register(
            project, selected[::-1], "score",
            [outcomes[i] for i in selected[::-1]],
        )
        result = labels.analysis.estimate(
            pi=sample["pi"], known=[known1, known2], population=full
        )
        manual_total = 5. + sum(outcomes.values()) * 6 / 4
        assert result.estimated_total == pytest.approx(manual_total)
        assert result.estimate == pytest.approx(manual_total / 8)
        assert result.standard_error is not None
        result_u = labels.analysis.estimate(
            pi=sample["pi"], known=[known1, known2]
        )
        assert result_u.estimate == pytest.approx(result.estimate)
        with pytest.raises(ArtifactError, match="overlap"):
            labels.analysis.estimate(
                pi=sample["pi"], known=[known1, known1]
            )
        with pytest.raises(ArtifactError, match="overlap"):
            labels.analysis.estimate(
                pi=sample["pi"], known=[labels]
            )
        with pytest.raises(ValueError, match="finite"):
            labels.analysis.estimate(
                pi=sample["pi"], known=known1,
                inference="superpopulation",
            )
        incorrect = _register(project, range(7), "text", ["x"] * 7)
        with pytest.raises(ArtifactError, match="U or U union"):
            labels.analysis.estimate(
                pi=sample["pi"], known=[known1, known2],
                population=incorrect
            )
    finally:
        project.close()


def test_direct_left_median_census_even_and_ties(tmp_path):
    project = teal.Project.create(tmp_path / "median_census", name="median")
    try:
        even = _register(project, [3, 1, 0, 2], "score", [8., 4., 1., 6.])
        result = even.analysis.estimate(statistic="median")
        assert result.estimate == pytest.approx(4.)
        assert result.standard_error == 0.
        assert result.confidence_interval() == (4., 4.)
        cdf = result.cdf_frame()
        assert cdf["threshold"].tolist() == [1., 4., 6., 8.]
        assert cdf["weighted_cdf"].tolist() == pytest.approx([.25, .5, .75, 1.])
        assert cdf["cdf_se"].tolist() == [0.] * 4
        assert cdf["weighted_cdf"].is_monotonic_increasing
        pi = _register(project, [2, 0, 3, 1], "pi", [1.] * 4)
        explicit = even.analysis.estimate(
            statistic="median", pi=pi, population=even, design="bernoulli",
        )
        pd.testing.assert_frame_equal(
            cdf.reset_index(drop=True),
            explicit.cdf_frame().reset_index(drop=True),
            check_exact=False, atol=1e-12,
        )
        assert explicit.estimate == result.estimate
        ties = _register(project, range(6), "score", [1., 1., 2., 2., 2., 9.])
        tied = ties.analysis.estimate(statistic="median")
        assert tied.estimate == 2.
        assert tied.cdf_frame()["weighted_cdf"].tolist() == pytest.approx([2/6, 5/6, 1.])
        assert tied.cdf_frame()["cdf_se"].eq(0.).all()
        with pytest.raises(ValueError, match="not available"):
            result.p_value()
    finally:
        project.close()


def test_direct_weighted_median_unequal_pi_shuffled_and_no_fake_ci(tmp_path):
    project = teal.Project.create(tmp_path / "median_weighted", name="median")
    try:
        population = _register(project, range(8), "text", ["x"] * 8)
        outcome = _register(project, [4, 0, 2], "value", [8., 1., 3.])
        pi = _register(project, [2, 4, 0], "pi", [.25, .8, .5])
        # Sorted observations 1,3,8 with inverse weights 2,4,1.25.
        # Half mass=3.625, reached first at value 3.
        med = outcome.analysis.estimate(
            statistic="median", pi=pi, population=population, design="bernoulli",
        )
        assert med.estimate == pytest.approx(3.)
        assert med.total_weight == pytest.approx(7.25)
        assert med.standard_error is None
        with pytest.raises(ValueError, match="unavailable"):
            med.confidence_interval()
        assert not med.cdf_frame().empty
        with pytest.raises(ValueError, match="known="):
            med2 = outcome.analysis.estimate(
                statistic="median", pi=pi, population=population, known=outcome,
            )
        with pytest.raises(ValueError, match="superpopulation"):
            outcome.analysis.estimate(
                statistic="median", pi=pi, population=population,
                inference="superpopulation",
            )
    finally:
        project.close()


def test_direct_weighted_median_census_and_stratified(tmp_path):
    project = teal.Project.create(tmp_path / "median_designs", name="median")
    try:
        population = _register(project, range(10), "text", ["x"] * 10)
        strata = _register(project, range(10), "H", ["A"] * 4 + ["B"] * 6)
        outcome = _register(project, [5, 0, 4, 1], "score", [6., 1., 2., 3.])
        pi = _register(project, [1, 4, 0, 5], "pi", [.5, 1/3, .5, 1/3])
        # Sorted values 1,2,3,6 have weights 2,3,2,3; half mass=5.
        # Left inverse returns 2, not the arithmetic midpoint 2.5.
        result = outcome.analysis.estimate(
            statistic="median", pi=pi, population=population,
            design="stratified_srswor", strata=strata,
        )
        assert result.estimate == pytest.approx(2.)
        all_outcome = _register(project, [4, 2, 0, 1, 3], "score", [8., 3., 1., 2., 5.])
        all_pi = _register(project, [3, 1, 4, 0, 2], "pi", [1.] * 5)
        all_population = _register(project, range(5), "text", ["x"] * 5)
        exact = all_outcome.analysis.estimate(
            statistic="median", pi=all_pi, population=all_population,
            design="bernoulli",
        )
        assert exact.estimate == 3.
        assert exact.standard_error == 0.
    finally:
        project.close()


def test_median_rejects_contradictory_external_sampling_design(tmp_path):
    project = teal.Project.create(tmp_path / "median_bad_design", name="median")
    try:
        frame = _register(project, range(4), "text", list("abcd"))
        outcomes = _register(project, [2, 0], "score", [5., 1.])
        wrong_srs_pi = _register(project, [0, 2], "pi", [.5, .75])
        with pytest.raises(ArtifactError, match="SRSWOR pi contradicts"):
            outcomes.analysis.estimate(
                statistic="median", pi=wrong_srs_pi, population=frame,
                design="srswor",
            )
        # The same unequal probabilities can describe an externally
        # asserted independent Bernoulli audit, not an SRSWOR audit.
        valid = outcomes.analysis.estimate(
            statistic="median", pi=wrong_srs_pi, population=frame,
            design="bernoulli",
        )
        assert valid.estimate == 1.
        assert valid.standard_error is None

        # An asserted stratified design must also agree with observed
        # within-stratum selection rates rather than return a median.
        strata = _register(project, range(4), "stratum", ["A", "A", "B", "B"])
        with pytest.raises(ArtifactError, match="pi contradicts"):
            outcomes.analysis.estimate(
                statistic="median", pi=wrong_srs_pi, population=frame,
                design="stratified_srswor", strata=strata,
            )
    finally:
        project.close()


def test_direct_weighted_cdf_ratio_bernoulli_variance(tmp_path):
    project = teal.Project.create(tmp_path / "cdf_ratio", name="ratio")
    try:
        frame = _register(project, range(8), "text", ["x"] * 8)
        gold = _register(project, [4, 0, 2], "value", [8., 1., 3.])
        pi = _register(project, [2, 4, 0], "pi", [.25, .8, .5])
        result = gold.analysis.estimate(
            statistic="median", pi=pi, population=frame, design="bernoulli")
        cdf = result.cdf_frame().set_index("threshold")
        value = 6. / 7.25
        expected = (2*(1-value)**2 + 12*(1-value)**2 +
                    .3125*value**2) / (7.25**2)
        assert cdf.loc[3., "weighted_cdf"] == pytest.approx(value)
        assert cdf.loc[3., "cdf_se"]**2 == pytest.approx(expected)
        assert result.standard_error is None
    finally:
        project.close()


def test_finite_srs_median_rank_interval_handles_ties_and_unknown_support(tmp_path):
    from itertools import combinations
    from text_analysis_lab.analysis.quantiles import finite_srs_quantile_interval

    # Enumerate every possible SRSWOR sample: conservative coverage must
    # hold even with mass points, and with a median not observed in the audit.
    for population in (
        [0., 1., 2., 3., 4., 5., 6., 7.],
        [0., 0., 0., 0., 8., 8., 8., 8.],
        [0., 0., 0., 3., 3., 3., 3., 9.],
    ):
        target = sorted(population)[3]
        covered = 0
        designs = list(combinations(range(8), 4))
        for indices in designs:
            labels = [population[i] for i in indices]
            lo, hi = finite_srs_quantile_interval(
                labels, population_size=8, probability=.5, level=.9,
            )
            covered += lo <= target <= hi
        assert covered / len(designs) >= .9

    project = teal.Project.create(tmp_path / "srs_rank", name="rank")
    try:
        frame = _register(project, range(8), "text", ["x"] * 8)
        gold = _register(project, [0, 2, 4, 6], "score", [0., 2., 4., 6.])
        pi = _register(project, [6, 4, 0, 2], "pi", [.5] * 4)
        result = gold.analysis.estimate(
            pi=pi, population=frame, design="srswor", statistic="median",
        )
        assert result.confidence_interval(level=.9) == finite_srs_quantile_interval(
            [0., 2., 4., 6.], population_size=8, probability=.5, level=.9,
        )
        assert result.standard_error is None
    finally:
        project.close()


def test_exact_srs_rank_interval_requires_verified_internal_sample(tmp_path, monkeypatch):
    """An SRS claim cannot revive a failed recorded-sample provenance check."""
    project = teal.Project.create(tmp_path / "srs_provenance", name="rank")
    try:
        frame = _register(project, range(12), "text", ["x"] * 12)
        sampled = project.probability_sample(frame, n=6, random_state=9)
        keys = sampled["sample"].query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].tolist()
        gold = _register(project, keys[::-1], "score", [float(i) for i in keys[::-1]])
        pi = sampled["pi"]

        verified = gold.analysis.estimate(pi=pi, statistic="median")
        assert verified.variance_design == "srswor"
        assert verified._srs_order is not None

        real_outputs = project.operation_outputs
        def missing_sample(operation_id):
            outputs = real_outputs(operation_id)
            if str(operation_id) == str(pi.operation_id):
                return [row for row in outputs if row.get("output_label") != "sample"]
            return outputs
        monkeypatch.setattr(project, "operation_outputs", missing_sample)
        unverified = gold.analysis.estimate(
            pi=pi, statistic="median", design="srswor",
        )
        assert unverified.variance_design is None
        assert unverified._srs_order is None
        with pytest.raises(ValueError, match="unavailable"):
            unverified.confidence_interval()
    finally:
        project.close()


def test_direct_quantile_design_validation_is_single_pass(tmp_path, monkeypatch):
    from text_analysis_lab.analysis import gd_quantile_design
    from text_analysis_lab.analysis import estimation as estimation_module

    project = teal.Project.create(tmp_path / "once_quantile", name="single")
    try:
        frame = _register(project, range(10), "text", ["x"] * 10)
        gold = _register(project, [0, 2, 4, 6, 8], "score", [1., 2., 3., 4., 5.])
        pi = _register(project, [8, 6, 4, 2, 0], "pi", [.5] * 5)
        calls = [0]
        real = gd_quantile_design.prepare_threshold_design
        def count(**kwargs):
            calls[0] += 1
            return real(**kwargs)
        monkeypatch.setattr(gd_quantile_design, "prepare_threshold_design", count)
        result = estimation_module.estimate(
            gold, pi=pi, population=frame, design="srswor",
            statistic="quantile", quantile=.5,
        )
        assert result.variance_design == "srswor"
        assert calls == [1]
    finally:
        project.close()


def test_sampled_mean_inference_withdrawn_without_changing_standard_errors(tmp_path):
    project = teal.Project.create(tmp_path / "withdrawn_wald", name="wald")
    try:
        frame = _register(project, range(8), "text", ["x"] * 8)
        gold = _register(project, [0, 2, 4, 6], "score", [1., 3., 2., 4.])
        pi = _register(project, [6, 4, 2, 0], "pi", [.5] * 4)
        result = gold.analysis.estimate(
            pi=pi, population=frame, design="srswor",
        )
        assert result.standard_error is not None
        assert result.estimated_total == pytest.approx(20.)
        for scale in ("mean", "total"):
            with pytest.raises(ValueError, match="unavailable"):
                result.confidence_interval(scale=scale)
            with pytest.raises(ValueError, match="unavailable"):
                result.p_value(scale=scale)
        full = _register(project, range(8), "score", [0., 1., 2., 3., 4., 5., 6., 7.])
        census = full.analysis.estimate()
        assert census.confidence_interval() == (3.5, 3.5)
        assert census.p_value(null=3.5) == 1.
        assert census.p_value(null=1.) == 0.
        iid = full.analysis.estimate(inference="superpopulation")
        assert iid.standard_error is not None
        with pytest.raises(ValueError, match="unavailable"):
            iid.confidence_interval()
    finally:
        project.close()
