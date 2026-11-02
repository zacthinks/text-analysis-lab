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
        predictor = register(project, range(5), "prediction", [9., 1., 1., 1., 1.])
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
        assert gd.variance_audit == pytest.approx(16 / 25)
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
        q = register(project, range(8), "prediction", [9., 9.] + [1.] * 6)
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
