"""HDBSCAN Phase 2: sklearn semantics, fitted provenance and keyed artifacts."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import HDBSCAN
from test_clustering_translators import X, _register_numeric_matrix, _source


def test_hdbscan_sklearn_parity_and_keyed_noise():
    model = HDBSCAN(min_cluster_size=2, min_samples=2)
    assert model.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    ).mode == "full_artifact"
    keys = [23, 2, 42, 9, 10, 1, 19]
    packet = InputBatch(
        source_label="source", artifact_id="source-1", primary_key=("doc_id",),
        data={"info": pd.DataFrame({"doc_id": keys}), "matrix": X},
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )
    actual = model.translate_batch(
        {"source": packet}, mode="fit_translate", request=TranslationRequest()
    ).outputs["output"]
    labels = SklearnHDBSCAN(min_cluster_size=2, min_samples=2).fit_predict(X)
    values = actual["data"]["values"]
    assert sparse.isspmatrix_csr(values)
    assert values.shape == (len(X), 2)
    assert actual["data"]["columns"] == ["cluster_0", "cluster_1"]
    assert actual["keys"]["doc_id"].tolist() == keys
    for i, label in enumerate(labels):
        assert values[i].nnz == (0 if label == -1 else 1)
        if label >= 0:
            assert values[i, label] == 1
    assert model.execution_capabilities().reusable is False
    assert model.save_assets(None) == {}
    with pytest.raises(OperatorError, match="already fitted"):
        model.fit_transform(X)
    with pytest.raises(OperatorError, match="fit-only"):
        model.translate(X)


def test_hdbscan_sparse_structured_and_invalid_values():
    labels = SklearnHDBSCAN(min_cluster_size=2, min_samples=2).fit_predict(
        sparse.csr_matrix(X)
    )
    actual = HDBSCAN(min_cluster_size=2, min_samples=2).fit_transform(
        {"values": sparse.csr_matrix(X), "columns": ["x", "y"]}
    )
    assert actual["values"].shape == (len(X), 2)
    assert actual["feature_metadata"]["column"].tolist() == ["cluster_0", "cluster_1"]
    for i, label in enumerate(labels):
        assert actual["values"][i].nnz == (0 if label == -1 else 1)
    with pytest.raises(Exception, match="finite"):
        HDBSCAN(min_cluster_size=2).fit_transform(
            np.asarray([[0.0, np.nan], [1.0, 2.0], [2.0, 4.0]])
        )
    with pytest.raises(ValueError, match="precomputed"):
        HDBSCAN(metric="precomputed")
    with pytest.raises(Exception):
        HDBSCAN(min_cluster_size=2, metric="not_a_distance").fit_transform(X)


def test_hdbscan_all_noise_and_provenance_roundtrip(tmp_path):
    import text_analysis_lab as teal

    keys = [23, 2, 42, 9, 10, 1, 19]
    location = tmp_path / "hdbscan_project"
    project = teal.Project.create(location, name="hdbscan_project")
    try:
        source = _register_numeric_matrix(
            project, artifact_id="hdbscan_input", matrix=X, keys=keys
        )
        model = HDBSCAN(min_cluster_size=5, min_samples=5)
        artifact = project.translate(model, source)["output"]
        assert artifact.get_matrix().shape == (7, 0)
        assert artifact.get_data_columns() == []
        assert artifact.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        model_id, artifact_id = model.operator_id, artifact.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        saved = reopened.get_artifact(artifact_id)
        assert saved.get_matrix().shape == (7, 0)
        assert saved.get_data_columns() == []
        packet = saved.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
            form="native", include_position=False,
        )
        assert packet["matrix"].shape == (7, 0)
        assert packet["info"]["doc_id"].tolist() == keys
        saved_model = reopened.get_operator(str(model_id))
        assert saved_model.is_frozen and saved_model.is_fitted
        assert saved_model.n_clusters_ == 0
        assert saved_model.execution_capabilities().reusable is False
        assert saved_model.save_assets(tmp_path / "assets") == {}
        with pytest.raises(OperatorError, match="fit-only"):
            saved_model.translate(X)
        with pytest.raises(Exception):
            reopened.translate(saved_model, reopened.get_artifact("hdbscan_input"))
    finally:
        reopened.close()


def test_hdbscan_normal_project_columns_and_snapshot(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "hdbscan_assigned"
    project = teal.Project.create(location, name="hdbscan_assigned")
    try:
        source = _register_numeric_matrix(
            project, artifact_id="hdbscan_dense", matrix=sparse.csr_matrix(X),
            keys=[23, 2, 42, 9, 10, 1, 19],
        )
        model = HDBSCAN(min_cluster_size=2, min_samples=2)
        artifact = project.translate(model, source)["output"]
        assert artifact.get_data_columns() == ["cluster_0", "cluster_1"]
        assert artifact.get_matrix().shape == (7, 2)
        assert artifact.get_matrix()[-1].nnz == 0
        assert artifact.get_feature_metadata()["column"].tolist() == [
            "cluster_0", "cluster_1"
        ]
        assert artifact.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert model.to_json_state()["n_clusters_found"] == 2
        obj = HDBSCAN.from_json_state(model.to_json_state())
        assert obj.is_fitted and obj.n_clusters_ == 2
    finally:
        project.close()


def test_hdbscan_edge_cases():
    values = np.asarray([[0, 0], [0, 0], [0.1, 0], [10, 10], [10, 10], [10.1, 10]])
    labels = SklearnHDBSCAN(min_cluster_size=2, min_samples=2).fit_predict(values)
    actual = HDBSCAN(min_cluster_size=2, min_samples=2).fit_transform(values)
    for i, label in enumerate(labels):
        assert actual[i].nnz == (0 if label == -1 else 1)
    with pytest.raises(ValueError, match="at least 2"):
        HDBSCAN(min_cluster_size=1)
