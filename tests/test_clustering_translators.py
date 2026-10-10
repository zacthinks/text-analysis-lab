"""Contract tests for the first unsupervised TeAL clustering translators."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.cluster import KMeans as SkKMeans

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import DBSCAN, KMeans


X = np.asarray([
    [0.0, 0.0], [0.0, 0.2], [0.2, 0.0],
    [10.0, 10.0], [10.2, 10.0], [10.0, 10.2],
    [60.0, 60.0],
])


def _source():
    return SimpleNamespace(
        artifact_id="source-1",
        artifact_type=ArtifactType.DENSE_MATRIX,
        primary_key=["doc_id"],
        get_data_columns=lambda: ["x", "y"],
    )


def _packet(x=X):
    return InputBatch(
        source_label="source", artifact_id="source-1",
        primary_key=("doc_id",),
        data={"info": pd.DataFrame({"doc_id": [23, 2, 42, 9, 10, 1, 19][:len(x)]}),
              "matrix": x},
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )


def test_kmeans_preserves_keys_and_stores_only_centers(tmp_path):
    model = KMeans(n_clusters=2, random_state=17, n_init=10)
    req = TranslationRequest()
    assert model.input_request(sources={"source": _source()}, mode="fit_translate", request=req).mode == "full_artifact"
    result = model.translate_batch({"source": _packet()}, mode="fit_translate", request=req).outputs["output"]
    values = result["data"]["values"]
    assert sparse.isspmatrix_csr(values)
    assert values.shape == (len(X), 2)
    assert np.all(np.asarray(values.sum(axis=1)).ravel() == 1)
    assert result["keys"]["doc_id"].tolist() == [23, 2, 42, 9, 10, 1, 19]
    assert result["data"]["columns"] == ["cluster_0", "cluster_1"]
    upstream = SkKMeans(n_clusters=2, random_state=17, n_init=10).fit(X)
    np.testing.assert_array_equal(values.argmax(axis=1).A1 if hasattr(values.argmax(axis=1), "A1") else np.asarray(values.argmax(axis=1)).ravel(), upstream.labels_)
    location = tmp_path / "assets"
    manifest = model.save_assets(location)
    assert manifest == {"centers_file": "centers.npy"}
    assert len(list(location.iterdir())) == 1
    clone = KMeans.from_json_state(model.to_json_state())
    clone.load_assets(location, manifest)
    np.testing.assert_array_equal(model.translate(X).toarray(), clone.translate(X).toarray())
    assert clone.execution_capabilities().reusable is True
    with pytest.raises(OperatorError, match="already fitted"):
        model.fit_transform(X)


def test_kmeans_feature_order_validation():
    model = KMeans(n_clusters=2, random_state=17)
    source = {"values": X, "columns": ["x", "y"]}
    model.fit_transform(source)
    with pytest.raises(OperatorError, match="ordered feature schema"):
        model.translate({"values": X, "columns": ["y", "x"]})


def test_dbscan_matches_sklearn_and_keeps_noise_unassigned(tmp_path):
    model = DBSCAN(eps=.4, min_samples=2)
    req = TranslationRequest()
    model.input_request(sources={"source": _source()}, mode="fit_translate", request=req)
    result = model.translate_batch({"source": _packet()}, mode="fit_translate", request=req).outputs["output"]
    values = result["data"]["values"]
    assert sparse.isspmatrix_csr(values)
    assert values.shape == (len(X), 2)
    assert result["keys"]["doc_id"].tolist() == [23, 2, 42, 9, 10, 1, 19]
    assert values[-1].nnz == 0
    upstream = SkDBSCAN(eps=.4, min_samples=2).fit_predict(X)
    np.testing.assert_array_equal(np.asarray(values.argmax(axis=1)).ravel()[:-1], upstream[:-1])
    assert model.save_assets(tmp_path) == {}
    assert model.execution_capabilities().reusable is False
    clone = DBSCAN.from_json_state(model.to_json_state())
    assert clone.is_fitted
    assert not clone.execution_capabilities().reusable
    with pytest.raises(OperatorError, match="fit-only"):
        model.translate(X)


def test_dbscan_all_noise_has_zero_cluster_columns():
    values = DBSCAN(eps=.01, min_samples=3).fit_transform(X)
    assert values.shape == (len(X), 0)
    assert values.nnz == 0


def test_sparse_inputs_and_nonfinite_rejected():
    KMeans(n_clusters=2).fit_transform(sparse.csr_matrix(X))
    DBSCAN(eps=.4, min_samples=2).fit_transform(sparse.csr_matrix(X))
    with pytest.raises(Exception, match="finite"):
        DBSCAN().fit_transform(np.array([[np.nan, 0], [1, 2]]))

def _register_numeric_matrix(project, *, artifact_id, matrix, keys):
    from text_analysis_lab.core.writer import create_artifact_writer
    writer = create_artifact_writer(
        artifact_type="sparse_matrix" if sparse.issparse(matrix) else "dense_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id, label="training",
        lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({
        "keys": pd.DataFrame({"doc_id": keys}),
        "data": {"values": matrix, "columns": ["x", "y"]},
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix" if sparse.issparse(matrix) else "dense_matrix",
        label="training", lineage_mode="new_key",
        status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_real_project_kmeans_reload_and_frozen_feature_schema(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "clustering_kmeans"
    project = teal.Project.create(location, name="clustering_kmeans")
    ids = [23, 2, 42, 9, 10, 1, 19]
    try:
        source = _register_numeric_matrix(
            project, artifact_id="clustering_kmeans_training",
            matrix=sparse.csr_matrix(X), keys=ids,
        )
        model = KMeans(n_clusters=2, random_state=17, n_init=10)
        artifact = project.translate(model, source)["output"]
        assert artifact.artifact_type.value == "sparse_matrix"
        assert artifact.primary_key == ["doc_id"]
        assert artifact.get_data_columns() == ["cluster_0", "cluster_1"]
        assert artifact.get_matrix().shape == (7, 2)
        assert artifact.get_matrix().nnz == 7
        assert artifact.lineage_mode == "preserved_key"
        first_id = artifact.artifact_id
        model_id = model.operator_id
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        persisted = reopened.get_artifact(first_id)
        assert persisted.status == "complete"
        assert persisted.get_matrix().shape == (7, 2)
        assert persisted.get_feature_metadata()["column"].tolist() == [
            "cluster_0", "cluster_1"
        ]
        matrix_packet = persisted.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
            form="native", include_position=False,
        )
        assert matrix_packet["matrix"].shape == (7, 2)
        op = reopened.get_operator(str(model_id))
        assert op.is_frozen
        features = pd.DataFrame({"column_index": [0, 1], "column": ["x", "y"]})
        np.testing.assert_array_equal(
            op.translate({"values": X, "feature_metadata": features})["values"].toarray(),
            op.translate(X).toarray(),
        )
        with pytest.raises((OperatorError, ValueError), match="ordered feature schema"):
            op.translate({"values": X, "columns": ["y", "x"]})
        second_source = _register_numeric_matrix(
            reopened, artifact_id="clustering_kmeans_second",
            matrix=X, keys=ids,
        )
        replayed = reopened.translate(op, second_source)["output"]
        np.testing.assert_array_equal(
            replayed.get_matrix().toarray(), persisted.get_matrix().toarray()
        )
    finally:
        reopened.close()


def test_real_project_dbscan_noise_and_zero_width_round_trip(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "clustering_dbscan"
    project = teal.Project.create(location, name="clustering_dbscan")
    ids = [23, 2, 42, 9, 10, 1, 19]
    try:
        source = _register_numeric_matrix(
            project, artifact_id="clustering_dbscan_training",
            matrix=X, keys=ids,
        )
        model = DBSCAN(eps=.01, min_samples=3)
        artifact = project.translate(model, source)["output"]
        assert artifact.primary_key == ["doc_id"]
        assert artifact.lineage_mode == "preserved_key"
        assert artifact.get_data_columns() == []
        assert artifact.get_matrix().shape == (7, 0)
        first_id = artifact.artifact_id
        model_id = model.operator_id
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        persisted = reopened.get_artifact(first_id)
        assert persisted.status == "complete"
        assert persisted.get_matrix().shape == (7, 0)
        assert persisted.get_data_columns() == []
        packet = persisted.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
            form="native", include_position=False,
        )
        assert packet["matrix"].shape == (7, 0)
        op = reopened.get_operator(str(model_id))
        assert op.is_fitted
        assert not op.execution_capabilities().reusable
        with pytest.raises(OperatorError, match="fit-only"):
            op.translate(X)
        with pytest.raises(Exception):
            reopened.translate(op, reopened.get_artifact("clustering_dbscan_training"))
    finally:
        reopened.close()


def test_dbscan_duplicate_and_singleton_and_bad_metric():
    matrix = np.asarray([[0, 0], [0, 0], [9, 9]], dtype=float)
    result = DBSCAN(eps=.25, min_samples=2).fit_transform(matrix)
    assert result.shape == (3, 1)
    assert result[:2].nnz == 2
    assert result[-1].nnz == 0
    with pytest.raises(Exception):
        DBSCAN(metric="not_a_distance").fit_transform(matrix)


def test_kmeans_fit_is_full_artifact_but_reuse_is_batched():
    request = TranslationRequest(batch_size=2)
    model = KMeans(n_clusters=2)
    source = _source()
    fit_request = model.input_request(
        sources={"source": source}, mode="fit_translate", request=request
    )
    assert fit_request.mode == "full_artifact"
    assert fit_request.batch_size is None
    model.fit_transform(X)
    predict_request = model.input_request(
        sources={"source": source}, mode="translate", request=request
    )
    assert predict_request.mode == "batches"
    assert predict_request.batch_size == 2
    dbscan = DBSCAN()
    assert dbscan.input_request(
        sources={"source": source}, mode="fit_translate", request=request
    ).mode == "full_artifact"


def test_kmeans_batched_prediction_matches_full_prediction_with_keys():
    trained = KMeans(n_clusters=2, n_init=10, random_state=17)
    trained.fit_transform(X)
    expected = trained.translate(X).toarray()
    collected = []
    collected_keys = []
    for start, stop in ((0, 2), (2, 5), (5, 7)):
        part = X[start:stop]
        keys = [23, 2, 42, 9, 10, 1, 19][start:stop]
        packet = InputBatch(
            source_label="source", artifact_id="source-1",
            primary_key=("doc_id",),
            data={"info": pd.DataFrame({"doc_id": keys}), "matrix": part},
            batch_index=start, batch_count=3,
            is_first=start == 0, is_last=stop == 7,
        )
        output = trained.translate_batch(
            {"source": packet}, mode="translate",
            request=TranslationRequest(batch_size=2),
        ).outputs["output"]
        collected.append(output["data"]["values"])
        collected_keys.extend(output["keys"]["doc_id"].tolist())
    np.testing.assert_array_equal(sparse.vstack(collected).toarray(), expected)
    assert collected_keys == [23, 2, 42, 9, 10, 1, 19]


def test_real_project_kmeans_reuse_batched_matches_standalone(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "batch_reuse"
    project = teal.Project.create(location, name="batch_reuse")
    try:
        source = _register_numeric_matrix(
            project, artifact_id="batch_fit", matrix=X,
            keys=[23, 2, 42, 9, 10, 1, 19],
        )
        trained = KMeans(n_clusters=2, n_init=10, random_state=17)
        fitted = project.translate(trained, source)["output"]
        fresh = _register_numeric_matrix(
            project, artifact_id="batch_apply",
            matrix=sparse.csr_matrix(X),
            keys=[19, 1, 10, 9, 42, 2, 23],
        )
        result = project.translate(trained, fresh, batch_size=2)["output"]
        np.testing.assert_array_equal(
            result.get_matrix().toarray(), fitted.get_matrix().toarray()
        )
        assert result.get_matrix().shape == (7, 2)
        assert result.get_data_columns() == ["cluster_0", "cluster_1"]
    finally:
        project.close()
