"""Rust FasterPAM KMedoids: backend parity, durable keys and frozen reuse."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import pairwise_distances

kmedoids = pytest.importorskip("kmedoids")

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import KMedoids
from test_clustering_translators import X, _source, _register_numeric_matrix


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_rust_fasterpam_parity_and_centers_roundtrip(tmp_path, metric):
    kwargs = dict(n_clusters=2, metric=metric, init="build", random_state=17)
    obj = KMedoids(**kwargs)
    result = obj.fit_transform({"values": X, "columns": ["x", "y"]})
    distances = pairwise_distances(X, metric=metric)
    upstream = kmedoids.fasterpam(
        np.asarray(distances, dtype=np.float64), 2, init="build",
        max_iter=100, random_state=17, n_cpu=1,
    )
    actual = result["values"]
    assert sparse.isspmatrix_csr(actual)
    assert actual.shape == (len(X), 2)
    np.testing.assert_array_equal(
        np.asarray(actual.argmax(axis=1)).ravel(), np.asarray(upstream.labels)
    )
    np.testing.assert_array_equal(obj.medoid_indices_, upstream.medoids)
    assert result["feature_metadata"]["column"].tolist() == ["cluster_0", "cluster_1"]
    assert obj.execution_capabilities().reusable
    assert obj.to_json_state()["backend"] == "rust-kmedoids"
    assert obj.to_json_state()["backend_version"] is not None
    manifest = obj.save_assets(tmp_path)
    assert manifest == {"medoids_file": "medoids.npy"}
    clone = KMedoids.from_json_state(obj.to_json_state())
    clone.load_assets(tmp_path, manifest)
    np.testing.assert_array_equal(
        clone.translate(X).toarray(), obj.translate(X).toarray()
    )
    with pytest.raises(OperatorError, match="ordered feature schema"):
        clone.translate({"values": X, "columns": ["y", "x"]})
    with pytest.raises(OperatorError, match="already fitted"):
        obj.fit_transform(X)


def test_kmedoids_sparse_and_full_artifact_project_roundtrip(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "kmedoids"
    project = teal.Project.create(location, name="kmedoids")
    keys = [23, 2, 42, 9, 10, 1, 19]
    try:
        source = _register_numeric_matrix(
            project, artifact_id="source_x", matrix=sparse.csr_matrix(X),
            keys=keys,
        )
        obj = KMedoids(n_clusters=2, metric="euclidean", init="build")
        assert obj.input_request(
            sources={"source": source}, mode="fit_translate",
            request=TranslationRequest(batch_size=2),
        ).mode == "full_artifact"
        artifact = project.translate(obj, source)["output"]
        assert artifact.get_data_columns() == ["cluster_0", "cluster_1"]
        assert artifact.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        oid, aid = str(obj.operator_id), artifact.artifact_id
        expected = artifact.get_matrix().toarray()
    finally:
        project.close()
    reopened = teal.Project.open(location)
    try:
        loaded = reopened.get_operator(oid)
        assert loaded.is_frozen and loaded.is_fitted
        assert loaded.execution_capabilities().reusable
        assert loaded.input_request(
            sources={"source": reopened.get_artifact("source_x")},
            mode="translate", request=TranslationRequest(batch_size=2),
        ).mode == "batches"
        saved = reopened.get_artifact(aid)
        packet = saved.query(
            key_columns=True, data_columns=True, metadata_columns=False,
            metadata_mode="none", form="native", include_position=False,
        )
        assert packet["info"]["doc_id"].tolist() == keys
        np.testing.assert_array_equal(packet["matrix"].toarray(), expected)
        replay = reopened.translate(
            loaded, reopened.get_artifact("source_x"), batch_size=2
        )["output"]
        np.testing.assert_array_equal(replay.get_matrix().toarray(), expected)
    finally:
        reopened.close()


def test_kmedoids_validates_requests():
    with pytest.raises(ValueError, match="positive"):
        KMedoids(n_clusters=0)
    with pytest.raises(ValueError, match="precomputed"):
        KMedoids(metric="precomputed")
    with pytest.raises(ValueError, match="exceed"):
        KMedoids(n_clusters=100).fit_transform(X)
    with pytest.raises(Exception, match="finite"):
        KMedoids(n_clusters=2).fit_transform(
            np.array([[1., 2.], [np.nan, 5.]])
        )
