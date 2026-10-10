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
