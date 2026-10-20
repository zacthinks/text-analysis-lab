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
        cv=KFold(n_splits=3, shuffle=True, random_state=3),
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


def test_real_reordered_keys_and_missing_development_key(tmp_path):
    source = pd.DataFrame({
        "text": [f"text {i}" for i in range(12)],
        "uid": list(range(12)),
        "x": list(range(12)),
        "label": [int(i >= 6) for i in range(12)],
    })
    labels_file = tmp_path / "labels.csv"
    reordered_file = tmp_path / "reordered.csv"
    missing_file = tmp_path / "missing.csv"
    source.to_csv(labels_file, index=False)
    source.iloc[::-1].to_csv(reordered_file, index=False)
    source.iloc[1:].to_csv(missing_file, index=False)
    project = Project.create(tmp_path / "keyed", name="keyed-search")
    try:
        labels_base = project.read_csv(
            labels_file, text_fields="text", metadata_fields=["uid", "label"],
        )
        labels = project.set_primary_keys(
            labels_base, levels={"uid": "uid_key"}, leaf_key="leaf",
        )
        def make_matrix(path):
            base = project.read_csv(
                path, text_fields="text", metadata_fields=["uid", "x"],
            )
            keyed = project.set_primary_keys(
                base, levels={"uid": "uid_key"}, leaf_key="leaf",
            )
            return project.translate(
                FeatureMatrix(data_columns=False, metadata_columns=["x"]), keyed
            )["output"]

        reverse = make_matrix(reordered_file)
        np.testing.assert_array_equal(reverse.get_matrix().ravel(), np.arange(11, -1, -1))
        search = ClassifierSearch(
            representations=[reverse], models={"logistic": {"C": [1.0]}}, cv=3,
        ).fit(labels, target="label")
        assert search.results()["status"].tolist() == ["ok"]
        assert search._matrices[0].ravel().tolist() == list(range(12))
        chosen = search.finalize(candidate=0)
        assert chosen.provenance["development_rows"] == 12

        partial = make_matrix(missing_file)
        missing = ClassifierSearch(
            representations=[partial], models={"logistic": {"C": [1.0]}}, cv=3,
        ).fit(labels, target="label")
        assert missing.results()["status"].tolist() == ["failed"]
        assert "missing development keys" in missing.results()["error"].iloc[0]
    finally:
        project.close()


def test_audit_only_matrix_rows_never_loaded_during_search(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "text": [f"text {i}" for i in range(12)],
        "x": list(range(12)),
        "label": [int(i >= 6) for i in range(12)],
    })
    devfile, fullfile = tmp_path / "dev.csv", tmp_path / "full.csv"
    frame.iloc[:8].to_csv(devfile, index=False)
    frame.to_csv(fullfile, index=False)
    project = Project.create(tmp_path / "audit", name="audit-search")
    try:
        # Default row keys are deliberately shared for the first eight rows.
        development = project.read_csv(
            devfile, text_fields="text", metadata_fields=["label"],
        )
        complete = project.read_csv(
            fullfile, text_fields="text", metadata_fields=["x"],
        )
        matrix = project.translate(
            FeatureMatrix(data_columns=False, metadata_columns=["x"]), complete,
        )["output"]
        original = type(matrix).get_matrix
        requested = []
        def guarded(artifact, *, positions=None, data_columns=True):
            if artifact.artifact_id == matrix.artifact_id:
                assert positions is not None
                assert all(0 <= p < 8 for p in positions), "audit-only matrix row queried"
                requested.extend(positions)
            return original(artifact, positions=positions, data_columns=data_columns)
        monkeypatch.setattr(type(matrix), "get_matrix", guarded)
        search = ClassifierSearch(
            representations=[matrix],
            models={"logistic": {"C": [1.0]}},
            cv=2,
        )
        search.plan(development, target="label")
        search.fit(development, target="label")
        chosen = search.finalize(candidate=0)
        assert len(requested) == 8
        assert chosen.provenance["development_rows"] == 8
        assert chosen.predictor.model.n_features_in_ == 1
    finally:
        project.close()


def test_progress_tracks_sklearn_searches_and_is_opt_in(
    labeled_representations, monkeypatch,
):
    from text_analysis_lab.analysis import classifier_search as module

    _, labels, dense, _ = labeled_representations
    seen = []

    class TrackingBar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.disabled = kwargs["disable"]
            self.completed = 0
            self.postfixes = []
            seen.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, amount):
            self.completed += amount

        def set_postfix_str(self, value, refresh=False):
            self.postfixes.append(value)

    monkeypatch.setattr(module, "tqdm", TrackingBar)
    search = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [0.1, 1.0]}},
        cv=3,
    )
    search.fit(labels, target="label", progress=True)
    assert seen[-1].total == seen[-1].completed == 1
    assert seen[-1].disabled is False
    assert any("logistic" in text for text in seen[-1].postfixes)
    assert len(search.results()) == 2

    search.fit(labels, target="label")
    assert seen[-1].disabled is True
    assert seen[-1].total == seen[-1].completed == 1



def test_fit_more_accumulates_without_replacing_existing_results(labeled_representations):
    _, labels, dense, sparse_features = labeled_representations
    search = ClassifierSearch(
        representations=[dense], models={"logistic": {"C": [1.0]}}, cv=3,
    )
    with pytest.raises(RuntimeError, match="Call fit"):
        search.fit_more(representations=[dense], models={"logistic": {"C": [0.1]}})
    search.fit(labels, target="label")
    original = search.results().copy(deep=True)
    search.fit_more(
        representations=[sparse_features],
        models={"linear_svm": {"C": [0.1, 1.0]}},
    )
    observed = search.results()
    pd.testing.assert_frame_equal(original, observed.iloc[:1].reset_index(drop=True))
    assert observed["candidate"].tolist() == [0, 1, 2]
    assert observed["search_round"].tolist() == [0, 1, 1]
    assert search.finalize(candidate=0).source.artifact_id == dense.artifact_id
    assert search.finalize(candidate=2).source.artifact_id == sparse_features.artifact_id
    search.fit_more(representations=[dense], models={"logistic": {"C": [0.1]}})
    assert len(search.representations) == 2
    assert search.results()["candidate"].tolist() == [0, 1, 2, 3]
    search.fit(labels, target="label")
    assert search.results()["candidate"].tolist() == [0]
    assert search.results()["search_round"].tolist() == [0]


def test_sklearn_random_and_halving_backends(labeled_representations):
    _, labels, dense, _ = labeled_representations
    for strategy in ("grid", "random", "halving_grid", "halving_random"):
        search = ClassifierSearch(
            representations=[dense],
            models={"logistic": {"C": [0.1, 1.0, 10.0, 100.0]}},
            cv=3, strategy=strategy,
            n_iter=3 if strategy in ("random", "halving_random") else None,
            random_state=23,
            search_kwargs={"min_resources": 12} if strategy.startswith("halving") else None,
        ).fit(labels, target="label")
        result = search.results()
        assert (result["strategy"] == strategy).all()
        assert result["status"].eq("ok").any()
        if strategy.startswith("halving"):
            assert result["search_iteration"].notna().all()
            assert result["n_resources"].notna().all()
            finalists = result.loc[result["eligible_for_selection"]]
            assert len(finalists) >= 1
            chosen = search.finalize(candidate=int(finalists.iloc[0]["candidate"]))
            assert chosen.provenance["search_strategy"] == strategy
            earlier = result.loc[~result["eligible_for_selection"]]
            if len(earlier):
                with pytest.raises(ValueError, match="intermediate"):
                    search.finalize(candidate=int(earlier.iloc[0]["candidate"]))
        else:
            assert result["eligible_for_selection"].all()


def test_halving_multiround_budget_and_selection(tmp_path):
    """A genuinely adaptive search must account for survivor refits."""
    frame = pd.DataFrame({
        "text": [f"document {i}" for i in range(120)],
        "x1": [i % 11 for i in range(120)],
        "x2": [(i * 7) % 17 for i in range(120)],
        "label": [i % 2 for i in range(120)],
    })
    data = tmp_path / "multiround.csv"
    frame.to_csv(data, index=False)
    project = Project.create(tmp_path / "multiround", name="multiround-halving")
    try:
        labels = project.read_csv(
            data, text_fields="text",
            metadata_fields=["x1", "x2", "label"],
        )
        features = project.translate(
            FeatureMatrix(data_columns=False, metadata_columns=["x1", "x2"]),
            labels,
        )["output"]
        for strategy in ("halving_grid", "halving_random"):
            search = ClassifierSearch(
                representations=[features],
                models={"logistic": {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}},
                strategy=strategy,
                n_iter=6 if strategy == "halving_random" else None,
                cv=3, random_state=7,
                search_kwargs={"factor": 2, "min_resources": 30},
            )
            plan = search.plan(labels, target="label")
            assert len(plan) == 6
            assert plan["initial_round_fits"].sum() == 18
            assert plan["planned_fits"].isna().all()
            search.fit(labels, target="label")
            result = search.results()
            rounds = result.groupby("search_iteration").agg(
                configs=("candidate", "size"),
                resources=("n_resources", "first"),
                fits=("executed_fits", "sum"),
            )
            assert rounds.index.tolist() == [0, 1, 2]
            assert rounds["configs"].tolist() == [6, 3, 2]
            assert rounds["resources"].tolist() == [30, 60, 120]
            assert rounds["fits"].tolist() == [18, 9, 6]
            assert result["executed_fits"].sum() == 33
            assert len(result) == 11
            assert result["eligible_for_selection"].sum() == 2
            early = result.loc[~result["eligible_for_selection"]].iloc[0]
            with pytest.raises(ValueError, match="intermediate"):
                search.finalize(candidate=int(early["candidate"]))
            final = result.loc[
                result["eligible_for_selection"] & result["status"].eq("ok")
            ].iloc[0]
            chosen = search.finalize(candidate=int(final["candidate"]))
            assert chosen.provenance["search_strategy"] == strategy
            assert chosen.provenance["development_rows"] == 120
    finally:
        project.close()


def test_search_kwargs_and_invalid_strategy(labeled_representations):
    _, labels, dense, _ = labeled_representations
    with pytest.raises(ValueError, match="Unknown sklearn search strategy"):
        ClassifierSearch(representations=[dense], models={"logistic": {}},
                         strategy="nonexistent")
    with pytest.raises(ValueError, match="positive candidate budget"):
        ClassifierSearch(representations=[dense], models={"logistic": {}},
                         strategy="random")
    with pytest.raises(ValueError, match="n_iter is only applicable"):
        ClassifierSearch(representations=[dense], models={"logistic": {}},
                         strategy="grid", n_iter=2)
    with pytest.raises(ValueError, match="needs cross-validation"):
        ClassifierSearch(representations=[dense], models={"logistic": {}},
                         cv=None, holdout=0.25, strategy="halving_grid")
    search = ClassifierSearch(
        representations=[dense],
        models={"logistic": {"C": [1.0]}},
        strategy="grid", search_kwargs={"verbose": 0},
        n_jobs=1,
    ).fit(labels, target="label")
    assert search.results()["status"].tolist() == ["ok"]
