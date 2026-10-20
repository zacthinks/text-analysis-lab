"""Phase 2: adopt fitted sklearn predictors without modifying the TeAL runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

pytest.importorskip("duckdb")

from text_analysis_lab import Project
from text_analysis_lab.core.errors import OperatorNotFittedError
from text_analysis_lab.translators import FeatureMatrix, FittedPredictor


@pytest.fixture
def project_matrix(tmp_path):
    path = tmp_path / "predictors.csv"
    pd.DataFrame({
        "text": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        "first": [0, 1, 2, 3, 4, 5],
        "second": [0, 0, 1, 1, 2, 2],
    }).to_csv(path, index=False)
    project = Project.create(tmp_path / "lab", name="fitted-adoption")
    table = project.read_csv(
        path, text_fields="text",
        metadata_fields=["first", "second"], output_label="input",
    )
    matrix = project.translate(
        FeatureMatrix(
            data_columns=False, metadata_columns=["first", "second"]
        ), table
    )["output"]
    try:
        yield project, table, matrix, tmp_path / "lab"
    finally:
        project.close()


@pytest.mark.parametrize("predictor", [
    lambda: LogisticRegression(random_state=1),
    lambda: LinearSVC(random_state=1),
    lambda: make_pipeline(
        StandardScaler(), LogisticRegression(random_state=1)
    ),
])
def test_fitted_sklearn_predictors_save_and_reopen(project_matrix, predictor):
    project, _, matrix, path = project_matrix
    x = matrix.get_matrix()
    y = np.array([0, 0, 0, 1, 1, 1])
    model = predictor().fit(x, y)
    result = project.translate(model, matrix, batch_size=2)["output"]
    assert result.primary_key == matrix.primary_key
    predicted = result.query(
        data_columns=["prediction"], key_columns=False, form="table",
    )["prediction"].to_numpy()
    np.testing.assert_array_equal(predicted, model.predict(x))
    op_id = result.descriptor["operation_id"]
    # Durable fitted-object operator can be restored from the saved operation.
    operation = project.catalog.get_operation(op_id)
    reopened = Project.open(path)
    try:
        restored_model = reopened.get_operator(operation["operator_id"])
        assert isinstance(restored_model, FittedPredictor)
        assert restored_model.save_model is True
        replay = reopened.translate(restored_model, reopened.get_artifact(matrix.artifact_id))["output"]
        replay_predictions = replay.query(
            data_columns=["prediction"], key_columns=False, form="table",
        )["prediction"].to_numpy()
        np.testing.assert_array_equal(replay_predictions, model.predict(x))
    finally:
        reopened.close()


def test_sparse_fitted_predictor_and_explicit_wrapper_unchanged(project_matrix):
    project, _, matrix, path = project_matrix
    sparse_features = project.translate(
        FeatureMatrix(
            data_columns=False, metadata_columns=["first", "second"],
            matrix_type="sparse"
        ),
        project.get_artifact("input"),
    )["output"]
    x = sparse_features.get_matrix()
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(random_state=1).fit(x, y)
    direct = project.translate(model, sparse_features)["output"]
    explicit = project.translate(
        FittedPredictor(model, save_model=True), sparse_features
    )["output"]
    for output in (direct, explicit):
        actual = output.query(
            data_columns=["prediction"], key_columns=False,
            form="table",
        )["prediction"].to_numpy()
        np.testing.assert_array_equal(actual, model.predict(x))


def test_reject_unfitted_and_transform_only_without_creating_operation(project_matrix):
    project, _, matrix, _ = project_matrix
    with pytest.raises(OperatorNotFittedError, match="fitted"):
        project.translate(LogisticRegression(), matrix)
    with pytest.raises(TypeError, match="predict"):
        project.translate(StandardScaler().fit(matrix.get_matrix()), matrix)


def test_fitted_sklearn_probability_is_not_automatically_inferred(project_matrix):
    project, _, matrix, _ = project_matrix
    x = matrix.get_matrix()
    model = LogisticRegression(random_state=1).fit(
        x, np.array([0, 0, 0, 1, 1, 1])
    )
    output = project.translate(model, matrix)["output"]
    assert output.get_data_columns() == ["prediction"]
