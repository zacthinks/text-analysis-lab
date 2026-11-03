"""Regression for certainty-known finite labels with an independent audit."""

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.analysis import generalized_difference, generalized_difference_by
from text_analysis_lab.core.errors import ArtifactError


def register(project, ids, field, values):
    return project.register_external(
        pd.DataFrame({"row_id": list(ids), field: list(values)}),
        primary_key="row_id", data_fields=field,
    )


def test_finite_known_direct_and_gd_grouped_manual_covariance(tmp_path):
    project = teal.Project.create(tmp_path / "known_finite", name="known")
    try:
        # F={0,...,4}, K={0}, U={1,2,3,4}, A={1,3}.
        frame = register(project, range(5), "text", ["x"] * 5)
        predictor = register(project, range(5), "prediction", [np.nan, 1., 1., 1., 1.])
        group = register(project, range(5), "group", ["A", "A", "A", "B", "B"])
        known = register(project, [0], "label", [5.])
        audit = register(project, [3, 1], "label", [4., 2.])
        pi = register(project, [1, 3], "pi", [.5, .5])
        direct = audit.analysis.estimate(
            pi=pi, population=frame, known=known, design="srswor",
        )
        assert direct.estimated_total == pytest.approx(5 + 2*(2 + 4))
        assert direct.population_size == 5
        gd = generalized_difference(
            predictor, gold=audit, pi=pi,
            known=known, design="srswor",
        )
        # Known 5 exactly + q_U=4 + 2*((2-1)+(4-1)) = 17.
        assert gd.estimated_total == pytest.approx(17.)
        assert gd.estimated_mean == pytest.approx(17 / 5)
        assert gd.variance_audit == pytest.approx(8 / 25)
        grouped = generalized_difference_by(
            predictor, gold=audit, pi=pi, known=known,
            group=group, group_field="group", design="srswor",
        )
        by = grouped.to_frame().set_index("group")
        assert by.loc["A", "estimated_mean"] == pytest.approx(3.)
        assert by.loc["B", "estimated_mean"] == pytest.approx(4.)
        assert by.loc["A", "population_size"] == 3
        assert by.loc["B", "population_size"] == 2
        # For contrast A - B, transformed audit residuals are
        # (2-1)/3 and -(4-1)/2. SRSWOR U=4 n=2.
        expected_variance = 16 * .5 * ((1/3 + 3/2)**2 / 2) / 2
        contrast = grouped.contrast("A", "B")
        assert contrast.difference == pytest.approx(-1.)
        assert contrast.standard_error == pytest.approx(np.sqrt(expected_variance))
    finally:
        project.close()


def test_known_with_recorded_remainder_and_forbidden_overlap(tmp_path):
    project = teal.Project.create(tmp_path / "known_provenance", name="known")
    try:
        target = register(project, range(8), "text", ["x"] * 8)
        u = register(project, range(2, 8), "text", ["x"] * 6)
        q = register(project, range(8), "prediction", [np.nan, np.nan] + [1.] * 6)
        known_a = register(project, [0], "label", [10.])
        known_b = register(project, [1], "label", [11.])
        sampled = project.probability_sample(u, n=3, random_state=3)
        keys = sampled["sample"].query(
            key_columns=True, data_columns=False,
            metadata_columns=False, form="table",
        )["row_id"].to_numpy(dtype=int)
        audit = register(project, keys[::-1], "label", [2.] * len(keys))
        gd = q.analysis.generalized_difference(
            gold=audit, pi=sampled["pi"], known=[known_a, known_b],
        )
        # 21 exact + 6 proxy + 3*(2-1)/(3/6) = 33.
        assert gd.population_size == 8
        assert gd.estimated_total == pytest.approx(33.)
        assert gd.variance_design == "srswor"
        assert gd.standard_error is not None
        with pytest.raises(ArtifactError, match="overlap"):
            q.analysis.generalized_difference(
                gold=audit, pi=sampled["pi"], known=[known_a, known_a],
            )
        bad = register(project, [2], "label", [3.])
        with pytest.raises(ArtifactError, match="overlap"):
            q.analysis.generalized_difference(
                gold=audit, pi=sampled["pi"], known=bad,
            )
        with pytest.raises(ValueError, match="finite"):
            q.analysis.generalized_difference(
                gold=audit, pi=sampled["pi"], known=known_a,
                inference="superpopulation",
            )
    finally:
        project.close()


def test_external_remainder_only_surrogate_and_population_are_valid(tmp_path):
    project = teal.Project.create(tmp_path / "external_u", name="external_u")
    try:
        # K and U are disjoint. User provides only U as the sampling frame.
        u = register(project, [2, 3, 4, 5], "prediction", [1., 1., 1., 1.])
        known1 = register(project, [0], "label", [5.])
        known2 = register(project, [1], "label", [7.])
        audit = register(project, [2, 4], "label", [3., 5.])
        pi = register(project, [4, 2], "pi", [.5, .5])
        result = u.analysis.generalized_difference(
            gold=audit, pi=pi, known=[known1, known2], design="srswor",
        )
        # 12 exact + 4 U surrogate + (2+4)*2 HT residual = 28.
        assert result.estimated_total == pytest.approx(28.)
        assert result.population_size == 6
        direct = audit.analysis.estimate(
            pi=pi, population=u, known=[known1, known2], design="srswor",
        )
        assert direct.estimated_total == pytest.approx(12 + 2*(3+5))
        # A group artifact must describe the full F, including K.
        groups = register(
            project, range(6), "group", ["A", "B", "A", "A", "B", "B"]
        )
        grouped = u.analysis.generalized_difference_by(
            gold=audit, pi=pi, known=[known1, known2],
            group=groups, group_field="group", design="srswor",
        )
        assert grouped.to_frame()["population_size"].sum() == 6
        partial = register(
            project, [0, 2, 3, 4, 5], "prediction", [1.] * 5
        )
        with pytest.raises(ArtifactError, match="either all known"):
            partial.analysis.generalized_difference(
                gold=audit, pi=pi, known=[known1, known2], design="srswor",
            )
    finally:
        project.close()


def test_gd_joins_allow_identical_gold_proxy_pi_and_group_field_names(tmp_path):
    """Shuffled keys and identical user field names must never imply row alignment."""
    project = teal.Project.create(tmp_path / "same_name_gd", name="same_name")
    try:
        proxy = register(project, range(6), "label", [1.] * 6)
        gold = register(project, [4, 0, 2], "label", [4., 2., 3.])
        # Even the pi column and the grouping column may legitimately be "label".
        pi = register(project, [2, 4, 0], "label", [.5] * 3)
        group = register(project, [5, 1, 3, 0, 4, 2], "label", [1, 0, 1, 0, 1, 0])

        direct = generalized_difference(
            proxy, gold=gold, pi=pi, surrogate_field="label",
            gold_field="label", pi_field="label", design="srswor",
        )
        # Proxy total 6, correction 2*((2-1)+(3-1)+(4-1)) = 12.
        assert direct.estimated_total == pytest.approx(18.)
        assert direct.estimated_mean == pytest.approx(3.)
        assert direct.variance_design == "srswor"
        assert direct.standard_error_total == pytest.approx(np.sqrt(6.))

        grouped = generalized_difference_by(
            proxy, gold=gold, pi=pi, group=group, group_field="label",
            surrogate_field="label", gold_field="label", pi_field="label",
            design="srswor",
        )
        by = grouped.to_frame().set_index("group")
        assert by.loc[0, "estimated_total"] == pytest.approx(9.)
        assert by.loc[1, "estimated_total"] == pytest.approx(9.)
        assert by.loc[0, "estimated_mean"] == pytest.approx(3.)
        assert by.loc[1, "estimated_mean"] == pytest.approx(3.)
        assert grouped.contrast(0, 1).difference == pytest.approx(0.)
        assert grouped.contrast(0, 1).standard_error is not None
    finally:
        project.close()


def test_gd_same_named_fields_with_finite_known_labels(tmp_path):
    project = teal.Project.create(tmp_path / "same_name_known", name="same_known")
    try:
        proxy = register(project, range(6), "label", [1.] * 6)
        known = register(project, [5], "label", [8.])
        gold = register(project, [2, 0], "label", [3., 2.])
        pi = register(project, [0, 2], "label", [.4, .4])
        group = register(project, range(6), "label", [0, 0, 0, 1, 1, 1])

        result = generalized_difference(
            proxy, gold=gold, pi=pi, known=known,
            surrogate_field="label", gold_field="label", pi_field="label",
            design="srswor",
        )
        # Remainder q total 5; known y=8; audited residuals (1+2)/.4=7.5.
        assert result.population_size == 6
        assert result.estimated_total == pytest.approx(20.5)
        assert result.variance_design == "srswor"
        assert result.standard_error_total == pytest.approx(np.sqrt(3.75))
        grouped = generalized_difference_by(
            proxy, gold=gold, pi=pi, known=known,
            group=group, group_field="label",
            surrogate_field="label", gold_field="label", pi_field="label",
            design="srswor",
        )
        by = grouped.to_frame().set_index("group")
        assert by.loc[0, "estimated_total"] == pytest.approx(10.5)
        assert by.loc[1, "estimated_total"] == pytest.approx(10.)
        assert by["estimated_total"].sum() == pytest.approx(result.estimated_total)
        assert grouped.contrast(0, 1).standard_error is not None
    finally:
        project.close()
