"""Real-project checks for GD design declarations and recorded sampling."""

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.analysis.generalized_difference import (
    generalized_difference,
    generalized_difference_by,
)
from text_analysis_lab.core.errors import ArtifactError


def _register(project, ids, name, values):
    return project.register_external(
        pd.DataFrame({"row_id": list(ids), name: list(values)}),
        primary_key="row_id", data_fields=name,
    )


@pytest.mark.parametrize("mode", ["srswor", "stratified_srswor", "bernoulli"])
def test_real_project_sampling_provenance_and_reopen(tmp_path, mode):
    root = tmp_path / mode
    project = teal.Project.create(root, name="gd_design")
    try:
        ids = np.arange(40)
        q = np.array([0.1 + 0.01 * int(i) for i in ids])
        y = np.array([float((int(i) % 7) < 3) for i in ids])
        source = _register(project, ids, "text", [f"doc {i}" for i in ids])
        surrogate = _register(project, ids, "prediction", q)
        strata = _register(project, ids, "H", [0] * 20 + [1] * 20)
        if mode == "srswor":
            sampled = project.probability_sample(source, n=20, random_state=5)
        elif mode == "stratified_srswor":
            sampled = project.probability_sample(
                source, n=20, strata=strata, random_state=5
            )
        else:
            prob = _register(
                project, ids[::-1], "probability",
                [0.25 + 0.5 * (int(i) % 5) / 4 for i in ids[::-1]],
            )
            sampled = project.probability_sample(
                source, probability=prob, random_state=5
            )
        audit_keys = sampled["sample"].query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].to_numpy(dtype=int)
        assert len(audit_keys) >= 2
        gold = _register(project, audit_keys[::-1], "label", y[audit_keys[::-1]])
        result = generalized_difference(
            surrogate, gold=gold, pi=sampled["pi"]
        )
        assert result.variance_design == mode
        assert result.standard_error is not None
        saved_mean, saved_se = result.estimated_mean, result.standard_error
        refs = (surrogate.artifact_id, gold.artifact_id, sampled["pi"].artifact_id)
    finally:
        project.close()
    reopened = teal.Project.open(root)
    try:
        a, g, p = [reopened.get_artifact(ref) for ref in refs]
        result = generalized_difference(a, gold=g, pi=p)
        assert result.variance_design == mode
        assert result.estimated_mean == pytest.approx(saved_mean)
        assert result.standard_error == pytest.approx(saved_se)
    finally:
        reopened.close()


def test_external_stratified_validates_missing_labels_and_pi(tmp_path):
    root = tmp_path / "external_strata"
    project = teal.Project.create(root, name="external")
    try:
        ids = range(8)
        q = _register(project, ids, "prediction", [0.0] * 8)
        gold = _register(project, [0, 1, 4, 5], "label", [1., 0., 0., 1.])
        pi = _register(project, [5, 0, 4, 1], "pi", [0.5] * 4)
        valid = _register(project, ids, "H", [0] * 4 + [1] * 4)
        result = generalized_difference(
            q, gold=gold, pi=pi, design="stratified_srswor", strata=valid
        )
        assert result.variance_design == "stratified_srswor"
        assert result.standard_error_total is not None
        invalid = _register(
            project, ids, "H", [0, 0, None, 0, 1, 1, 1, 1]
        )
        with pytest.raises(ArtifactError, match="missing or null"):
            generalized_difference(
                q, gold=gold, pi=pi,
                design="stratified_srswor", strata=invalid
            )
        wrong_pi = _register(project, [5, 0, 4, 1], "pi", [0.5, 0.7, 0.5, 0.5])
        with pytest.raises(ArtifactError, match="pi contradicts"):
            generalized_difference(
                q, gold=gold, pi=wrong_pi,
                design="stratified_srswor", strata=valid
            )
    finally:
        project.close()
