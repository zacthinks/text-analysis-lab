"""End-to-end FeatureMatrix: ordinary matrix contract, replay, sparsity and legacy views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")

from text_analysis_lab import Project
from text_analysis_lab.translators import CountVectorizer, FeatureMatrix


def _project(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({
        "text": ["alpha beta", "beta gamma", "alpha gamma"],
        "group": ["male", "female", "male"],
        "income": [10, 20, 30],
    }).to_csv(path, index=False)
    project = Project.create(tmp_path / "lab", name="feature-matrix")
    source = project.read_csv(
        path, text_fields="text",
        metadata_fields=["group", "income"], output_label="documents",
    )
    dtm = project.translate(CountVectorizer(text_field="text"), source)["output"]
    return project, source, dtm


def test_sparse_feature_matrix_durable_replay_and_legacy_feature_subset(tmp_path):
    project, source, dtm = _project(tmp_path)
    try:
        operator = FeatureMatrix(
            metadata_columns=["group", "income"],
            encodings={"group": "onehot"},
        )
        derived = project.translate(operator, dtm)["output"]
        assert derived.artifact_type.value == "sparse_matrix"
        assert "data" not in derived.components
        assert derived.get_data_columns() == [
            *dtm.get_data_columns(), "group_female", "group_male", "income",
        ]
        target = sparse.hstack([
            dtm.get_matrix(),
            sparse.csr_matrix([[0, 1, 10], [1, 0, 20], [0, 1, 30]]),
        ], format="csr")
        assert sparse.issparse(derived.get_matrix())
        np.testing.assert_array_equal(derived.get_matrix().toarray(), target.toarray())
        np.testing.assert_array_equal(
            derived.get_matrix(positions=[2, 0], data_columns=["income"]).toarray(),
            [[30], [10]],
        )
        np.testing.assert_array_equal(
            derived.query(
                positions=[1], data_columns=["group_female"],
                key_columns=False, form="native",
            )["matrix"].toarray(),
            [[1]],
        )
        subset = project.feature_subset(
            derived, lambda meta: meta["column"].eq("group_male"),
        )
        np.testing.assert_array_equal(
            subset.get_matrix().toarray(), [[1], [0], [1]]
        )
        np.testing.assert_array_equal(
            subset.query(
                positions=[2, 0], key_columns=False,
                data_columns=["group_male"], form="native",
            )["matrix"].toarray(), [[1], [1]]
        )
        artifact_id, operator_id = derived.artifact_id, operator.operator_id
    finally:
        project.close()

    reopened = Project.open(tmp_path / "lab")
    try:
        restored = reopened.get_artifact(artifact_id)
        np.testing.assert_array_equal(restored.get_matrix().toarray(), target.toarray())
        frozen = reopened.get_operator(operator_id)
        same = reopened.translate(frozen, reopened.get_artifact(dtm.artifact_id))["output"]
        np.testing.assert_array_equal(same.get_matrix().toarray(), target.toarray())
    finally:
        reopened.close()


def test_dense_table_to_feature_matrix_without_numeric_data_duplication(tmp_path):
    project, source, dtm = _project(tmp_path)
    try:
        operator = FeatureMatrix(
            data_columns=False,
            metadata_columns=["income", "group"],
            key_columns=True,
        )
        matrix = project.translate(operator, source)["output"]
        assert matrix.artifact_type.value == "dense_matrix"
        assert matrix.components.get("data") is None
        assert matrix.n_rows == 3
        values = matrix.get_matrix()
        assert isinstance(values, np.ndarray)
        assert values.shape == (3, 4)
        np.testing.assert_array_equal(values[:, 1], [10, 20, 30])
        np.testing.assert_array_equal(
            matrix.query(data_columns=["income"], key_columns=False, form="native")["matrix"],
            [[10], [20], [30]],
        )
    finally:
        project.close()

def test_feature_matrix_frozen_standalone_table_translation(tmp_path):
    project, source, _ = _project(tmp_path)
    try:
        op = FeatureMatrix(
            data_columns=False,
            metadata_columns=["group", "income"],
        )
        project.translate(op, source)
        frame = pd.DataFrame({"group": ["female", "new"], "income": [41, 42]})
        values = op.translate(frame)
        np.testing.assert_array_equal(values, [[1, 0, 41], [0, 0, 42]])
    finally:
        project.close()
