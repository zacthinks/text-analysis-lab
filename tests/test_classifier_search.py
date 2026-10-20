"""End-to-end ClassifierSearch regression tests over real TeAL artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from text_analysis_lab import Project
from text_analysis_lab.analysis import ClassifierSearch
from text_analysis_lab.translators import FeatureMatrix


@pytest.fixture
def labeled_representations(tmp_path):
    frame = pd.DataFrame({
        "text": [f"document {i}" for i in range(12)],
        "x1": list(range(12)),
        "x2": [i % 3 for i in range(12)],
        "label": [int(i >= 6) for i in range(12)],
        "group": [i // 2 for i in range(12)],
    })
    path = tmp_path / "study.csv"
    frame.to_csv(path, index=False)
    project = Project.create(tmp_path / "project", name="classifier-search")
    try:
        table = project.read_csv(
            path, text_fields="text",
            metadata_fields=["x1", "x2", "label", "group"],
        )
        dense = project.translate(
            FeatureMatrix(data_columns=False, metadata_columns=["x1", "x2"]),
            table,
        )["output"]
        sparse_features = project.translate(
            FeatureMatrix(
                data_columns=False, metadata_columns=["x1", "x2"],
                matrix_type="sparse",
            ), table,
        )["output"]
        yield project, table, dense, sparse_features
    finally:
        project.close()


def test_plan_fit_finalize_and_teal_prediction(labeled_representations):
    project, labels, dense, sparse_features = labeled_representations
    search = ClassifierSearch(
        representations=[sparse_features, dense],
        models={
            "logistic": {"C": [0.1, 1.0]},
            "linear_svm": {"C": [1.0]},
        },
        cv=3, random_state=14,
    )
    planned = search.plan(labels, target="label")
    assert len(planned) == 6
    assert planned["planned_fits"].sum() == 18
    search.fit(labels, target="label")
    result = search.results()
    assert set(result["status"]) == {"ok"}
    assert result["mean_score"].between(0, 1).all()
    assert all(len(folds) == 3 for folds in result["fold_scores"])
    selected = int(result.sort_values("mean_score", ascending=False).iloc[0]["candidate"])
    chosen = search.finalize(candidate=selected)
    assert chosen.source.artifact_id == result.loc[selected, "representation_id"]
    assert isinstance(chosen.predictor.model, (LogisticRegression,)) or callable(chosen.predictor.model.predict)
    assert chosen.provenance["development_rows"] == 12
    assert chosen.predictor.training_provenance == chosen.provenance
    predicted = project.translate(chosen.predictor, chosen.source)["output"]
    values = predicted.query(
        data_columns=["prediction"], key_columns=False, form="table",
    )["prediction"].to_numpy()
    np.testing.assert_array_equal(
        values, chosen.predictor.model.predict(chosen.source.get_matrix())
    )


def test_budget_and_reproducible_candidate_search(labeled_representations):
    _, labels, dense, sparse_features = labeled_representations
    arguments = dict(
        representations=[dense, sparse_features],
        models={"logistic": {"C": [0.1, 1.0, 10.0]}},
        cv=3, random_state=9, n_iter=2,
    )
    first = ClassifierSearch(**arguments).fit(labels, target="label")
    second = ClassifierSearch(**arguments).fit(labels, target="label")
    left, right = first.results(), second.results()
    assert len(left) == 4
    assert left["params"].tolist() == right["params"].tolist()
    np.testing.assert_allclose(left["mean_score"], right["mean_score"])


def test_key_alignment_reorders_representation_rows(labeled_representations, monkeypatch):
    _, labels, dense, _ = labeled_representations
    original = type(dense).get_matrix
    seen_positions = []

    def instrumented(artifact, *, positions=None, data_columns=True):
        if artifact.artifact_id == dense.artifact_id:
            seen_positions.append(positions)
        return original(artifact, positions=positions, data_columns=data_columns)

    monkeypatch.setattr(type(dense), "get_matrix", instrumented)
    search = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}}, cv=3,
    )
    search.fit(labels, target="label")
    assert seen_positions
    assert all(positions is not None for positions in seen_positions)


def test_holdout_and_candidate_failure_isolation(labeled_representations):
    _, labels, dense, _ = labeled_representations
    search = ClassifierSearch(
        representations=[dense],
        models={"logistic": {"C": [0.0, 1.0]}},
        cv=None, holdout=0.25, random_state=4,
    )
    search.fit(labels, target="label")
    result = search.results()
    assert result["status"].tolist() == ["failed", "ok"]
    with pytest.raises(ValueError, match="failed"):
        search.finalize(candidate=0)
    assert search.finalize(candidate=1).predictor.model is not None


def test_invalid_label_schema_and_missing_keys(labeled_representations):
    _, labels, dense, _ = labeled_representations
    with pytest.raises(ValueError, match="exactly one"):
        ClassifierSearch(
            representations=[dense], models={"logistic": {}},
            cv=3, holdout=0.25,
        )
    with pytest.raises(TypeError, match="matrix"):
        ClassifierSearch(representations=[labels], models={"logistic": {}}, cv=3)


def test_group_aware_cv_and_holdout(labeled_representations):
    _, labels, dense, _ = labeled_representations
    grouped = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}},
        cv=3,
    )
    grouped.plan(labels, target="label", group="group")
    assert len(grouped._folds) == 3
    for train, test in grouped._folds:
        assert not set(grouped._groups[train]) & set(grouped._groups[test])

    heldout = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}},
        cv=None, holdout=0.3,
    )
    heldout.plan(labels, target="label", group="group")
    train, test = heldout._folds[0]
    assert not set(heldout._groups[train]) & set(heldout._groups[test])


def test_custom_cv_splitter(labeled_representations):
    from sklearn.model_selection import StratifiedKFold

    _, labels, dense, _ = labeled_representations
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
    search = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}},
        cv=splitter,
    )
    search.fit(labels, target="label")
    assert search.results()["status"].tolist() == ["ok"]
    assert len(search.results()["fold_scores"].iloc[0]) == 3


def test_finalize_provenance_survives_operator_reopen(labeled_representations):
    project, labels, dense, _ = labeled_representations
    search = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}},
        cv=3,
    ).fit(labels, target="label")
    chosen = search.finalize(candidate=0)
    artifact = project.translate(chosen.predictor, chosen.source)["output"]
    operator_id = project.catalog.get_operation(
        artifact.descriptor["operation_id"]
    )["operator_id"]
    restored = project.get_operator(operator_id)
    assert restored.training_provenance == chosen.provenance


def test_huge_grid_random_budget_does_not_enumerate(labeled_representations, monkeypatch):
    _, labels, dense, _ = labeled_representations
    from sklearn.model_selection import ParameterGrid
    original = ParameterGrid.__iter__
    def forbidden(self):
        raise AssertionError("Bounded search must not iterate the whole grid")
    monkeypatch.setattr(ParameterGrid, "__iter__", forbidden)
    search = ClassifierSearch(
        representations=[dense],
        models={"logistic": {"C": range(1, 2_000_001)}},
        cv=3, n_iter=2, random_state=9,
    )
    planned = search.plan(labels, target="label")
    assert len(planned) == 2
    assert all(params["C"] > 0 for params in planned["params"])


def test_custom_splitter_cannot_mix_declared_groups(labeled_representations):
    _, labels, dense, _ = labeled_representations
    search = ClassifierSearch(
        representations=[dense],
        models={"logistic": {"C": [1.0]}},
        cv=KFold(n_splits=3, shuffle=False),
    )
    with pytest.raises(ValueError, match="overlapping development groups"):
        search.plan(labels, target="label", group="group")


def test_estimator_valued_pipeline_grid_survives_frozen_reopen(labeled_representations):
    project, labels, dense, _ = labeled_representations
    pipeline = Pipeline([
        ("scale", "passthrough"),
        ("predict", LogisticRegression(max_iter=500)),
    ])
    search = ClassifierSearch(
        representations=[dense],
        models={"pipeline": (
            pipeline, {"scale": [StandardScaler(), "passthrough"]}
        )},
        cv=3,
    ).fit(labels, target="label")
    assert set(search.results()["status"]) == {"ok"}
    chosen = search.finalize(candidate=0)
    assert chosen.provenance["params"]["scale"]["__kind__"] == "sklearn_estimator"
    artifact = project.translate(chosen.predictor, chosen.source)["output"]
    op_id = project.catalog.get_operation(
        artifact.descriptor["operation_id"]
    )["operator_id"]
    restored = project.get_operator(op_id)
    assert restored.training_provenance == chosen.provenance
    np.testing.assert_array_equal(
        restored.model.predict(dense.get_matrix()),
        chosen.predictor.model.predict(dense.get_matrix()),
    )
