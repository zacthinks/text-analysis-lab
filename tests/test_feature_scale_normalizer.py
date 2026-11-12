"""Feature-column scaling, sklearn search, and TeAL persistence regressions."""

import json
import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from text_analysis_lab.preprocessing import (
    FeatureScaleNormalizer, portable_feature_scale_pipeline,
)


@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("norm", ["rms", "l2"])
def test_dense_sparse_exact_scales_and_format(kind, gamma, norm):
    base = np.array([[3., 0., 0.], [4., 6., 0.], [0., 8., 0.]])
    X = base if kind == "dense" else getattr(sparse, kind + "_matrix")(base)
    trained = FeatureScaleNormalizer(gamma=gamma, norm=norm).fit(X)
    expected = np.linalg.norm(base, axis=0)
    if norm == "rms":
        expected /= np.sqrt(3)
    np.testing.assert_allclose(trained.column_scales_, expected)
    before = X.copy()
    transformed = trained.transform(X)
    assert sparse.issparse(transformed) == sparse.issparse(X)
    if sparse.issparse(X):
        assert transformed.getformat() == kind
    factors = np.where(expected < 1e-12, 1.0, expected) ** gamma
    np.testing.assert_allclose(
        transformed.toarray() if sparse.issparse(transformed) else transformed,
        base / factors,
    )
    np.testing.assert_allclose(
        before.toarray() if sparse.issparse(before) else before, base,
    )


def test_fitted_stats_frozen_and_gamma_zero_is_identity():
    train = np.array([[1., 10.], [2., 20.]])
    test = np.array([[3., 100.], [4., 200.]])
    model = FeatureScaleNormalizer(gamma=1.0).fit(train)
    scales = model.column_scales_.copy()
    np.testing.assert_allclose(model.transform(test), test / scales)
    np.testing.assert_array_equal(model.column_scales_, scales)
    identity = FeatureScaleNormalizer(gamma=0.0).fit(train)
    np.testing.assert_array_equal(identity.transform(test), test)
    assert identity.transform(test) is not test


def test_does_not_center_sparse_and_preserves_nnz_on_large_shape():
    x = sparse.csr_matrix(
        (np.array([1., 2., 3.]), ([0, 150_000, 299_999], [1, 100, 99_999])),
        shape=(300_000, 100_000),
    )
    m = FeatureScaleNormalizer(gamma=0.5).fit(x)
    y = m.transform(x)
    assert sparse.isspmatrix_csr(y) and y.nnz == x.nnz
    assert y.shape == x.shape
    assert m.column_scales_[50] == 0


def test_invalid_values_features_and_hyperparameters():
    for values in (np.array([[np.nan]]), np.array([[np.inf]])):
        with pytest.raises(ValueError):
            FeatureScaleNormalizer().fit(values)
    for params in ({"gamma": -0.1}, {"gamma": 1.1}, {"gamma": np.nan},
                   {"norm": "other"}, {"zero_tol": -1}, {"copy": 1}):
        with pytest.raises(ValueError):
            FeatureScaleNormalizer(**params).fit([[1., 2.]])
    with pytest.raises(ValueError, match="Expected 2 features"):
        FeatureScaleNormalizer().fit([[1., 2.]]).transform([[1., 2., 3.]])
    with pytest.raises(Exception):
        FeatureScaleNormalizer().transform([[1.]])


def test_clone_pickle_state_roundtrip_and_portable_replacement():
    x = np.array([[1., 10.], [2., 11.], [1., 12.], [2., 13.],
                  [3., 14.], [4., 15.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    pipe = Pipeline([
        ("scale", FeatureScaleNormalizer(gamma=0.5)),
        ("classifier", LogisticRegression(max_iter=500)),
    ]).fit(x, y)
    estimator = pipe.named_steps["scale"]
    assert not hasattr(clone(estimator), "column_scales_")
    state = json.loads(json.dumps(estimator.to_fitted_state()))
    reconstructed = FeatureScaleNormalizer.from_fitted_state(state)
    np.testing.assert_allclose(estimator.transform(x), reconstructed.transform(x))
    new_pipeline = portable_feature_scale_pipeline(pipe, scale_steps={"scale": state})
    np.testing.assert_allclose(new_pipeline.predict_proba(x), pipe.predict_proba(x))
    assert new_pipeline.named_steps["scale"] is not estimator
    restored = pickle.loads(pickle.dumps(new_pipeline))
    np.testing.assert_allclose(restored.predict_proba(x), pipe.predict_proba(x))
    with pytest.raises(ValueError, match="version"):
        FeatureScaleNormalizer.from_fitted_state({**state, "version": 999})
    with pytest.raises(ValueError, match="Unknown"):
        portable_feature_scale_pipeline(pipe, scale_steps={"unknown": state})


def test_joint_grid_search_fits_on_folds():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 3)) * [1, 100, 0.1]
    y = np.array([0, 1] * 15)
    pipe = Pipeline([
        ("scale", FeatureScaleNormalizer()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    search = GridSearchCV(
        pipe, {"scale__gamma": [0, 0.5, 1], "classifier__C": [0.1, 1]},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=13),
    ).fit(x, y)
    assert len(search.cv_results_["params"]) == 6
    assert search.best_estimator_.named_steps["scale"].n_samples_seen_ == len(x)
    assert isinstance(pickle.loads(pickle.dumps(search.best_estimator_)), Pipeline)


def test_classifier_search_joint_gamma_and_c_real_project(tmp_path):
    import pandas as pd

    from text_analysis_lab import Project
    from text_analysis_lab.analysis import ClassifierSearch
    from text_analysis_lab.translators import FeatureMatrix

    frame = pd.DataFrame({
        "text": [f"document {i}" for i in range(30)],
        "x1": list(range(30)),
        "x2": [i % 5 for i in range(30)],
        "label": [i >= 15 for i in range(30)],
    })
    csv_path = tmp_path / "corpus.csv"
    frame.to_csv(csv_path, index=False)
    project_path = tmp_path / "project"
    project = Project.create(project_path, name="scale-search")
    try:
        table = project.read_csv(
            csv_path, text_fields="text",
            metadata_fields=["x1", "x2", "label"],
        )
        matrix = project.translate(
            FeatureMatrix(data_columns=False, metadata_columns=["x1", "x2"],
                          matrix_type="sparse"), table,
        )["output"]
        pipeline = Pipeline([
            ("scale", FeatureScaleNormalizer()),
            ("classifier", LogisticRegression(max_iter=500)),
        ])
        search = ClassifierSearch(
            representations=[matrix],
            models={"scaled": (pipeline, {
                "scale__gamma": [0., 0.5, 1.],
                "classifier__C": [0.1, 1.],
            })},
            cv=3, random_state=7,
        ).fit(table, target="label")
        result = search.results()
        assert len(result) == 6
        assert result["status"].eq("ok").all()
        chosen = search.finalize(candidate=int(result.iloc[0]["candidate"]))
        fitted = chosen.predictor.model
        assert fitted.named_steps["scale"].n_samples_seen_ == 30
        expected = fitted.predict_proba(matrix.get_matrix())
        predicted = project.translate(chosen.predictor, chosen.source)["output"]
        operation_id = predicted.descriptor["operation_id"]
        operator_id = project.catalog.get_operation(operation_id)["operator_id"]
        reloaded_operator = project.get_operator(operator_id)
        np.testing.assert_allclose(
            reloaded_operator.model.predict_proba(matrix.get_matrix()),
            expected,
        )
    finally:
        project.close()
    reopened = Project.open(project_path)
    try:
        operator = reopened.get_operator(operator_id)
        assert isinstance(operator.model.named_steps["scale"], FeatureScaleNormalizer)
        np.testing.assert_allclose(
            operator.model.predict_proba(reopened.get_artifact(matrix.artifact_id).get_matrix()),
            expected,
        )
    finally:
        reopened.close()
