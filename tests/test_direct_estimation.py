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
            gold_field="human_judgment",
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
        with pytest.raises(ArtifactError, match="gold_field="):
            multi.analysis.estimate(
                pi=pi, population=population, design="srswor",
            )
        assert multi.analysis.estimate(
            pi=pi, population=population, design="srswor",
            gold_field="human_judgment",
        ).estimate == pytest.approx(inferred.estimate)
        with pytest.raises(ArtifactError, match="must expose data field"):
            gold.analysis.estimate(
                pi=pi, population=population, design="srswor",
                gold_field="does_not_exist",
            )
    finally:
        project.close()
