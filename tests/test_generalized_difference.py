from __future__ import annotations

import math

import pandas as pd
import pytest

from text_analysis_lab.analysis.generalized_difference import generalized_difference
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator


class _FakeArtifact:
    def __init__(
        self,
        frame: pd.DataFrame,
        data_columns: list[str],
        *,
        project: _FakeProject,
        artifact_id: str,
        operation_id: str | None = None,
    ) -> None:
        self._frame = frame.copy()
        self._data_columns = list(data_columns)
        self.project = project
        self.primary_key = ["row_id"]
        self.artifact_id = artifact_id
        self.operation_id = operation_id

    def get_data_columns(self):
        return list(self._data_columns)

    def query(self, **kwargs):
        fields = kwargs.get("data_columns")
        if fields is False:
            selected = ["row_id"]
        else:
            selected = ["row_id", *fields]
        return self._frame.loc[:, selected].copy()


class _FakeProject:
    def __init__(self) -> None:
        self.artifacts: dict[str, _FakeArtifact] = {}
        self.operations: dict[str, dict] = {}
        self.sources: dict[str, list[dict]] = {}
        self.outputs: dict[str, list[dict]] = {}
        self.operators: dict[str, object] = {}

    def add(self, artifact: _FakeArtifact) -> _FakeArtifact:
        self.artifacts[artifact.artifact_id] = artifact
        return artifact

    def get_artifact(self, value):
        if isinstance(value, str):
            return self.artifacts[value]
        return value

    def get_operation(self, operation_id: str):
        return self.operations[operation_id]

    def operation_sources(self, operation_id: str):
        return self.sources[operation_id]

    def operation_outputs(self, operation_id: str):
        return self.outputs[operation_id]

    def get_operator(self, operator_id: str):
        return self.operators[operator_id]


def _artifact(
    project: _FakeProject,
    artifact_id: str,
    frame: pd.DataFrame,
    fields: list[str],
    *,
    operation_id: str | None = None,
) -> _FakeArtifact:
    return project.add(
        _FakeArtifact(
            frame,
            fields,
            project=project,
            artifact_id=artifact_id,
            operation_id=operation_id,
        )
    )


def _record_probability_split(
    project: _FakeProject,
    *,
    pi: _FakeArtifact,
    documents: _FakeArtifact,
    translator: ProbabilitySplitTranslator,
    strata: _FakeArtifact | None = None,
) -> None:
    operation_id = pi.operation_id
    assert operation_id is not None
    operator_id = f"operator_{operation_id}"
    translator.operator_id = operator_id
    project.operations[operation_id] = {"operator_id": operator_id}
    project.operators[operator_id] = translator
    sources = [
        {
            "source_label": "documents",
            "source_artifact_id": documents.artifact_id,
        }
    ]
    if strata is not None:
        sources.append(
            {
                "source_label": "strata",
                "source_artifact_id": strata.artifact_id,
            }
        )
    project.sources[operation_id] = sources
    project.outputs[operation_id] = [
        {"output_label": "sample", "artifact_id": "audit"},
        {"output_label": "pi", "artifact_id": pi.artifact_id},
    ]


def test_generalized_difference_point_estimate_aligns_by_key() -> None:
    project = _FakeProject()
    population = _artifact(
        project,
        "population",
        pd.DataFrame({"row_id": [3, 0, 2, 1], "prediction": [1, 0, 1, 0]}),
        ["prediction"],
    )
    gold = _artifact(
        project,
        "gold",
        pd.DataFrame({"row_id": [2, 0], "label": [1, 1]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi",
        pd.DataFrame({"row_id": [0, 2], "pi": [0.5, 0.5]}),
        ["pi"],
    )

    result = generalized_difference(population, gold=gold, pi=pi)
    assert result.population_size == 4
    assert result.audit_size == 2
    assert result.surrogate_total == pytest.approx(2.0)
    assert result.correction_total == pytest.approx(2.0)
    assert result.estimated_total == pytest.approx(4.0)
    assert result.estimated_mean == pytest.approx(1.0)
    assert result.estimated_prevalence == pytest.approx(1.0)
    assert result.standard_error is None
    assert result.variance_design is None
    assert "first-order inclusion probabilities alone" in (result.variance_note or "")
    with pytest.raises(ValueError, match="Cannot form a confidence interval"):
        result.confidence_interval()


def test_generalized_difference_recovers_srswor_standard_error_from_provenance() -> (
    None
):
    project = _FakeProject()
    documents = _artifact(
        project,
        "documents",
        pd.DataFrame({"row_id": [0, 1, 2, 3]}),
        [],
    )
    population = _artifact(
        project,
        "population",
        pd.DataFrame({"row_id": [3, 0, 2, 1], "prediction": [1, 0, 1, 0]}),
        ["prediction"],
    )
    gold = _artifact(
        project,
        "gold",
        pd.DataFrame({"row_id": [2, 0], "label": [1, 1]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi",
        pd.DataFrame({"row_id": [0, 2], "pi": [0.5, 0.5]}),
        ["pi"],
        operation_id="split1",
    )
    _record_probability_split(
        project,
        pi=pi,
        documents=documents,
        translator=ProbabilitySplitTranslator(n=2, random_state=1),
    )

    result = generalized_difference(population, gold=gold, pi=pi)
    # audit residuals are [0, 1] in gold physical order -> sample variance .5.
    # Var(total) = N^2 (1-f) s^2 / n = 16 * .5 * .5 / 2 = 2.
    assert result.variance_design == "srswor"
    assert result.standard_error_total == pytest.approx(math.sqrt(2.0))
    assert result.standard_error == pytest.approx(math.sqrt(2.0) / 4.0)
    low, high = result.confidence_interval()
    assert low < result.estimated_mean < high


def test_generalized_difference_recovers_stratified_srswor_standard_error() -> None:
    project = _FakeProject()
    documents = _artifact(
        project,
        "documents",
        pd.DataFrame({"row_id": list(range(6))}),
        [],
    )
    strata = _artifact(
        project,
        "strata",
        # Deliberately shuffled to enforce stable-key alignment.
        pd.DataFrame(
            {
                "row_id": [5, 0, 3, 1, 4, 2],
                "H": [1, 0, 1, 0, 1, 0],
            }
        ),
        ["H"],
    )
    population = _artifact(
        project,
        "population",
        pd.DataFrame(
            {
                "row_id": [5, 0, 3, 1, 4, 2],
                "prediction": [1, 0, 1, 0, 1, 0],
            }
        ),
        ["prediction"],
    )
    gold = _artifact(
        project,
        "gold",
        pd.DataFrame({"row_id": [4, 0, 3, 1], "label": [0, 1, 1, 0]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi",
        pd.DataFrame({"row_id": [0, 1, 3, 4], "pi": [2 / 3] * 4}),
        ["pi"],
        operation_id="split2",
    )
    _record_probability_split(
        project,
        pi=pi,
        documents=documents,
        strata=strata,
        translator=ProbabilitySplitTranslator(
            n=4, allocation={0: 1, 1: 1}, random_state=2
        ),
    )

    result = generalized_difference(population, gold=gold, pi=pi)
    # Each stratum has N_h=3, n_h=2, sample residual variance=.5.
    # Each contributes 9*(1-2/3)*.5/2=.75, hence total variance=1.5.
    assert result.variance_design == "stratified_srswor"
    assert result.standard_error_total == pytest.approx(math.sqrt(1.5))
    assert result.standard_error == pytest.approx(math.sqrt(1.5) / 6.0)


def test_generalized_difference_stratum_with_one_noncensus_audit_has_no_se() -> None:
    project = _FakeProject()
    documents = _artifact(
        project,
        "documents",
        pd.DataFrame({"row_id": list(range(6))}),
        [],
    )
    strata = _artifact(
        project,
        "strata",
        pd.DataFrame({"row_id": list(range(6)), "H": [0, 0, 0, 1, 1, 1]}),
        ["H"],
    )
    population = _artifact(
        project,
        "population",
        pd.DataFrame({"row_id": list(range(6)), "prediction": [0, 0, 0, 1, 1, 1]}),
        ["prediction"],
    )
    gold = _artifact(
        project,
        "gold",
        pd.DataFrame({"row_id": [0, 3], "label": [1, 0]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi",
        pd.DataFrame({"row_id": [0, 3], "pi": [1 / 3, 1 / 3]}),
        ["pi"],
        operation_id="split3",
    )
    _record_probability_split(
        project,
        pi=pi,
        documents=documents,
        strata=strata,
        translator=ProbabilitySplitTranslator(
            n=2, allocation={0: 1, 1: 1}, random_state=3
        ),
    )

    result = generalized_difference(population, gold=gold, pi=pi)
    assert result.estimated_mean == pytest.approx(0.5)
    assert result.standard_error is None
    assert "only n_h=1" in (result.variance_note or "")


def test_generalized_difference_supports_probability_surrogate_and_rejects_bad_keys() -> (
    None
):
    project = _FakeProject()
    population = _artifact(
        project,
        "population",
        pd.DataFrame({"row_id": [0, 1, 2], "probability": [0.2, 0.8, 0.6]}),
        ["probability"],
    )
    gold = _artifact(
        project,
        "gold",
        pd.DataFrame({"row_id": [0, 2], "label": [0, 1]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi",
        pd.DataFrame({"row_id": [2, 0], "pi": [2 / 3, 2 / 3]}),
        ["pi"],
    )
    result = generalized_difference(
        population,
        gold=gold,
        pi=pi,
        surrogate_field="probability",
    )
    expected_total = 1.6 + ((0 - 0.2) + (1 - 0.6)) / (2 / 3)
    assert result.estimated_total == pytest.approx(expected_total)

    bad_pi = _artifact(
        project,
        "bad_pi",
        pd.DataFrame({"row_id": [0, 1], "pi": [2 / 3, 2 / 3]}),
        ["pi"],
    )
    with pytest.raises(ArtifactError, match="key set must exactly equal"):
        generalized_difference(
            population,
            gold=gold,
            pi=bad_pi,
            surrogate_field="probability",
        )


def test_generalized_difference_grouped_estimates_and_contrast_with_srswor_se() -> None:
    from text_analysis_lab.analysis.generalized_difference import (
        generalized_difference_by,
    )

    project = _FakeProject()
    documents = _artifact(
        project,
        "documents_grouped",
        pd.DataFrame({"row_id": list(range(8))}),
        [],
    )
    population = _artifact(
        project,
        "population_grouped",
        pd.DataFrame(
            {
                "row_id": list(range(8)),
                "prediction": [0, 1, 0, 1, 1, 1, 0, 0],
            }
        ),
        ["prediction"],
    )
    group = _artifact(
        project,
        "party",
        pd.DataFrame(
            {
                "row_id": [7, 0, 5, 2, 6, 1, 4, 3],
                "party": ["R", "D", "R", "D", "R", "D", "R", "D"],
            }
        ),
        ["party"],
    )
    # SRS audit has two cases per group. Residual patterns differ by group.
    gold = _artifact(
        project,
        "gold_grouped",
        pd.DataFrame({"row_id": [0, 1, 4, 5], "label": [1, 1, 0, 1]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi_grouped",
        pd.DataFrame({"row_id": [5, 0, 4, 1], "pi": [0.5] * 4}),
        ["pi"],
        operation_id="split_grouped",
    )
    _record_probability_split(
        project,
        pi=pi,
        documents=documents,
        translator=ProbabilitySplitTranslator(n=4, random_state=1),
    )

    result = generalized_difference_by(
        population,
        gold=gold,
        pi=pi,
        group=group,
        group_field="party",
    )
    frame = result.to_frame().set_index("group")
    # D: Q total=2; audit residuals [1,0], weighted correction=2 -> total 4 / 4 = 1.
    assert frame.loc["D", "estimated_mean"] == pytest.approx(1.0)
    # R: Q total=2; audit residuals [-1,0], correction=-2 -> total 0 / 4 = 0.
    assert frame.loc["R", "estimated_mean"] == pytest.approx(0.0)
    assert frame.loc["D", "standard_error"] is not None
    assert frame.loc["R", "standard_error"] is not None

    contrast = result.contrast("D", "R")
    assert contrast.difference == pytest.approx(1.0)
    assert contrast.standard_error is not None
    assert 0.0 <= contrast.p_value() <= 1.0
    low, high = contrast.confidence_interval()
    assert low < contrast.difference < high

    versus_d = result.contrasts(reference="D")
    assert versus_d[["first", "second"]].to_dict("records") == [
        {"first": "R", "second": "D"}
    ]
    assert versus_d.loc[0, "difference"] == pytest.approx(-1.0)

    pairwise = result.pairwise_contrasts()
    assert len(pairwise) == 1
    assert pairwise.loc[0, "difference"] == pytest.approx(1.0)


def test_generalized_difference_result_p_value_uses_design_se() -> None:
    project = _FakeProject()
    documents = _artifact(
        project, "documents_p", pd.DataFrame({"row_id": [0, 1, 2, 3]}), []
    )
    population = _artifact(
        project,
        "population_p",
        pd.DataFrame({"row_id": [0, 1, 2, 3], "prediction": [0, 0, 0, 0]}),
        ["prediction"],
    )
    gold = _artifact(
        project,
        "gold_p",
        pd.DataFrame({"row_id": [0, 1], "label": [0, 1]}),
        ["label"],
    )
    pi = _artifact(
        project,
        "pi_p",
        pd.DataFrame({"row_id": [0, 1], "pi": [0.5, 0.5]}),
        ["pi"],
        operation_id="split_p",
    )
    _record_probability_split(
        project,
        pi=pi,
        documents=documents,
        translator=ProbabilitySplitTranslator(n=2, random_state=4),
    )
    result = generalized_difference(population, gold=gold, pi=pi)
    assert result.standard_error is not None
    assert 0.0 <= result.p_value(null=0.0) <= 1.0
