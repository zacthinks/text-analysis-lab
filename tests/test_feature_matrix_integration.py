"""End-to-end FeatureMatrix: ordinary matrix contract, replay, sparsity and legacy views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from text_analysis_lab.translators import FittedPredictor

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


def test_frozen_replay_on_distinct_teal_artifact_with_unseen_categories(tmp_path):
    project, source, _ = _project(tmp_path)
    try:
        first = FeatureMatrix(
            data_columns=False,
            metadata_columns=["group", "income"],
            encodings={"group": "onehot"},
        )
        original = project.translate(first, source)["output"]
        expected_columns = original.get_data_columns()
        frozen_id = first.operator_id
    finally:
        project.close()

    other_path = tmp_path / "other.csv"
    pd.DataFrame({
        "text": ["different text", "other text"],
        "group": ["unknown", "female"],
        "income": [99, 47],
    }).to_csv(other_path, index=False)
    reopened = Project.open(tmp_path / "lab")
    try:
        distinct = reopened.read_csv(
            other_path, text_fields="text",
            metadata_fields=["group", "income"], output_label="other_documents",
        )
        frozen = reopened.get_operator(frozen_id)
        replay = reopened.translate(frozen, distinct)["output"]
        assert replay.get_data_columns() == expected_columns
        np.testing.assert_array_equal(
            replay.get_matrix(), [[0, 0, 99], [1, 0, 47]]
        )

        incompatible_path = tmp_path / "incompatible.csv"
        pd.DataFrame({"text": ["no field"], "other": [5]}).to_csv(
            incompatible_path, index=False
        )
        incompatible = reopened.read_csv(
            incompatible_path, text_fields="text",
            metadata_fields=["other"], output_label="incompatible_documents",
        )
        with pytest.raises(Exception, match="FeatureMatrix|group|income"):
            reopened.translate(frozen, incompatible)
    finally:
        reopened.close()


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_feature_subset_pushes_narrow_computed_columns_to_source(
    tmp_path, monkeypatch, kind,
):
    project, source, dtm = _project(tmp_path)
    try:
        computed = project.translate(
            FeatureMatrix(
                metadata_columns=["group", "income"],
                encodings={"group": "onehot"},
                matrix_type=kind,
            ), dtm,
        )["output"]
        subset = project.feature_subset(
            computed, lambda meta: meta["column"].eq("income"),
        )
        called = []
        original_query = type(dtm).query

        def recorded_query(artifact, **kwargs):
            if artifact.artifact_id == dtm.artifact_id:
                called.append(kwargs)
                if kwargs.get("form") == "native" and kwargs.get("data_columns") is not False:
                    pytest.fail("FeatureSubset income-only query fetched source DTM features")
            return original_query(artifact, **kwargs)

        monkeypatch.setattr(type(dtm), "query", recorded_query)
        assert subset.get_data_columns() == ["income"]
        got = subset.get_matrix(positions=[2, 0])
        if sparse.issparse(got):
            got = got.toarray()
        np.testing.assert_array_equal(got, [[30], [10]])
        assert any(
            entry.get("metadata_columns") == ["income"]
            and entry.get("data_columns") is False
            for entry in called
        )
    finally:
        project.close()


def test_compound_keys_survive_recipe_and_subset_queries(tmp_path):
    project, source, dtm = _project(tmp_path)
    try:
        compound = project.set_primary_keys(
            source, levels={"group": "group_id"}, leaf_key="doc_id",
        )
        assert compound.primary_key == ["group_id", "doc_id"]
        matrix = project.translate(
            FeatureMatrix(
                data_columns=False,
                metadata_columns=["income"],
                key_columns=True,
            ), compound,
        )["output"]
        assert matrix.primary_key == ["group_id", "doc_id"]
        keys = matrix.query(
            key_columns=True, data_columns=False, form="table",
        )[["group_id", "doc_id"]]
        assert keys.to_numpy().tolist() == [[1, 0], [0, 0], [1, 1]]
        assert matrix.get_data_columns() == ["group_id", "doc_id", "income"]
        np.testing.assert_array_equal(
            matrix.get_matrix(), [[1, 0, 10], [0, 0, 20], [1, 1, 30]]
        )
        subset = project.feature_subset(
            matrix, lambda features: features["column"].eq("income"),
        )
        assert subset.primary_key == ["group_id", "doc_id"]
        np.testing.assert_array_equal(subset.get_matrix(), [[10], [20], [30]])
    finally:
        project.close()


def test_fitted_predictor_consumes_recipe_backed_sparse_matrix(tmp_path):
    project, source, dtm = _project(tmp_path)
    try:
        augmented = project.translate(
            FeatureMatrix(
                metadata_columns=["group", "income"],
                encodings={"group": "onehot"},
            ), dtm,
        )["output"]
        x = augmented.get_matrix()
        assert sparse.isspmatrix_csr(x)
        y = np.array([0, 1, 0])
        fitted = LogisticRegression(random_state=0).fit(x, y)
        predicted = project.translate(
            FittedPredictor(fitted, save_model=True), augmented,
        )["output"]
        assert predicted.primary_key == augmented.primary_key
        actual = predicted.query(
            key_columns=False, data_columns=["prediction"], form="table",
        )["prediction"].to_numpy()
        np.testing.assert_array_equal(actual, fitted.predict(x))
    finally:
        project.close()


@pytest.mark.parametrize("include_metadata", [False, True])
def test_feature_matrix_scope_collision_survives_lazy_render_and_reopen(
    tmp_path, include_metadata,
):
    """The recipe's scope must not change for narrowed lazy reads."""
    project_path = tmp_path / ("with_meta" if include_metadata else "only_data")
    project = Project.create(project_path, name="recipe_scope_collision")
    source_id = computed_id = None
    expected = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"score": [11, 22]}),
            "metadata": pd.DataFrame({"score": [101, 202]}),
        }], artifact_type="table", primary_key="doc_id")
        scope_mode = "full" if include_metadata else "none"
        discovered = source.query_columns(metadata_mode=scope_mode)["columns"]
        data_name = next(
            c["output_name"] for c in discovered
            if c["namespace"] == "data" and c["base_name"] == "score"
        )
        metadata_names = (
            [next(
                c["output_name"] for c in discovered
                if c["namespace"] == "metadata" and c["base_name"] == "score"
            )] if include_metadata else False
        )
        operator = FeatureMatrix(
            data_columns=[data_name], metadata_columns=metadata_names,
            matrix_type="dense",
        )
        computed = project.translate(operator, source)["output"]
        expected = np.array(
            [[11, 101], [22, 202]] if include_metadata else [[11], [22]]
        )
        np.testing.assert_array_equal(computed.get_matrix(), expected)
        np.testing.assert_array_equal(
            computed.get_matrix(positions=[1, 0], data_columns=[data_name]),
            [[22], [11]],
        )
        source_id, computed_id = source.artifact_id, computed.artifact_id
    finally:
        project.close()

    reopened = Project.open(project_path)
    try:
        restored = reopened.get_artifact(computed_id)
        assert restored.get_data_columns()[0] == data_name
        np.testing.assert_array_equal(restored.get_matrix(), expected)
        np.testing.assert_array_equal(
            restored.get_matrix(positions=[1], data_columns=[data_name]),
            [[22]],
        )
    finally:
        reopened.close()
