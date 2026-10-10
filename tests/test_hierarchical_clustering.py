"""Phase 3 sklearn clustering contracts and fitted-backend provenance."""

import numpy as np
import pandas as pd
import pytest
import sklearn
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.cluster import OPTICS as SklearnOPTICS

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    OPTICS,
)
from test_clustering_translators import X, _register_numeric_matrix, _source


def _match_sklearn(actual, labels):
    """Test exact upstream label-to-CSR semantics, including unassigned rows."""
    expected_k = max((int(label) + 1 for label in labels if label >= 0), default=0)
    assert sparse.isspmatrix_csr(actual)
    assert actual.shape == (len(labels), expected_k)
    for row, label in enumerate(labels):
        assert actual[row].nnz == (0 if label == -1 else 1)
        if label >= 0:
            assert actual[row, label] == 1


@pytest.mark.parametrize("method, kwargs", [
    ("xi", {"min_samples": 2, "cluster_method": "xi", "min_cluster_size": 2}),
    ("dbscan", {
        "min_samples": 2, "cluster_method": "dbscan",
        "max_eps": 1.0, "eps": 0.4,
    }),
])
def test_optics_exact_sklearn_parity_dense_and_unassigned(method, kwargs):
    expected = SklearnOPTICS(**kwargs).fit_predict(X)
    obj = OPTICS(**kwargs)
    actual = obj.fit_transform(X)
    _match_sklearn(actual, expected)
    assert obj.n_clusters_ == actual.shape[1]
    assert obj.execution_capabilities().reusable is False
    assert obj.save_assets(None) == {}
    with pytest.raises(OperatorError, match="already fitted"):
        obj.fit_transform(X)
    with pytest.raises(OperatorError, match="fit-only"):
        obj.translate(X)
    assert obj.to_json_state()["sklearn_version"] == sklearn.__version__


def test_optics_sparse_metric_constraints_and_source_schema():
    shifted = sparse.csr_matrix(X + 1.0)
    kwargs = {"min_samples": 2, "metric": "cosine", "cluster_method": "xi"}
    expected = SklearnOPTICS(**kwargs).fit_predict(shifted)
    obj = OPTICS(**kwargs)
    actual = obj.fit_transform({"values": shifted, "columns": ["x", "y"]})
    _match_sklearn(actual["values"], expected)
    assert actual["feature_metadata"]["column"].tolist() == [
        f"cluster_{i}" for i in range(actual["values"].shape[1])
    ]
    with pytest.raises(OperatorError, match="sparse"):
        OPTICS(min_samples=2).fit_transform(sparse.csr_matrix(X))
    with pytest.raises(ValueError, match="precomputed"):
        OPTICS(metric="precomputed")
    with pytest.raises(ValueError, match="eps"):
        OPTICS(min_samples=2, cluster_method="dbscan", eps=None)
    with pytest.raises(ValueError, match="max_eps"):
        OPTICS(min_samples=2, cluster_method="dbscan", eps=2, max_eps=1)


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 2, "linkage": "ward"},
    {"n_clusters": 3, "linkage": "complete", "metric": "manhattan"},
    {"n_clusters": None, "linkage": "average", "distance_threshold": 0.45},
])
def test_agglomerative_matches_sklearn_on_dense_inputs(kwargs):
    expected = SklearnAgglomerative(**kwargs).fit_predict(X)
    obj = AgglomerativeClustering(**kwargs)
    actual = obj.fit_transform(X)
    _match_sklearn(actual, expected)
    assert actual.nnz == len(X)
    assert obj.n_clusters_ == actual.shape[1]
    assert obj.execution_capabilities().reusable is False
    assert obj.save_assets(None) == {}
    with pytest.raises(OperatorError, match="already fitted"):
        obj.fit_transform(X)
    with pytest.raises(OperatorError, match="fit-only"):
        obj.translate(X)


def test_agglomerative_rejects_unsupported_modes_without_implicit_densification():
    with pytest.raises(OperatorError, match="dense"):
        AgglomerativeClustering().fit_transform(sparse.csr_matrix(X))
    with pytest.raises(ValueError, match="ward"):
        AgglomerativeClustering(metric="cosine", linkage="ward")
    with pytest.raises(ValueError, match="n_clusters=None"):
        AgglomerativeClustering(n_clusters=2, distance_threshold=0.3)
    with pytest.raises(ValueError, match="positive n_clusters"):
        AgglomerativeClustering(n_clusters=0)
    with pytest.raises(ValueError, match="precomputed"):
        AgglomerativeClustering(metric="precomputed", linkage="average")


@pytest.mark.parametrize("factory", [
    lambda: OPTICS(min_samples=2, cluster_method="dbscan", eps=0.4, max_eps=1),
    lambda: AgglomerativeClustering(n_clusters=2),
])
def test_phase3_preserves_shuffled_keys_and_full_fit_mode(factory):
    obj = factory()
    request = TranslationRequest(batch_size=2)
    source_req = obj.input_request(
        sources={"source": _source()}, mode="fit_translate", request=request
    )
    assert source_req.mode == "full_artifact"
    keys = [23, 2, 42, 9, 10, 1, 19]
    batch = InputBatch(
        source_label="source",
        artifact_id="source-1",
        primary_key=("doc_id",),
        data={"info": pd.DataFrame({"doc_id": keys}), "matrix": X},
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )
    output = obj.translate_batch(
        {"source": batch}, mode="fit_translate", request=request
    ).outputs["output"]
    assert output["keys"]["doc_id"].tolist() == keys
    _match_sklearn(output["data"]["values"], (
        SklearnOPTICS(
            min_samples=2, cluster_method="dbscan", eps=.4, max_eps=1
        ).fit_predict(X)
        if isinstance(obj, OPTICS)
        else SklearnAgglomerative(n_clusters=2).fit_predict(X)
    ))
    assert output["data"]["columns"] == [
        f"cluster_{i}" for i in range(output["data"]["values"].shape[1])
    ]


@pytest.mark.parametrize("name,factory", [
    ("optics", lambda: OPTICS(min_samples=2, cluster_method="dbscan", eps=.4, max_eps=1)),
    ("agglomerative", lambda: AgglomerativeClustering(n_clusters=2)),
])
def test_phase3_project_persists_keyed_results_and_fit_only_operator(
    tmp_path, name, factory
):
    import text_analysis_lab as teal

    location = tmp_path / name
    project = teal.Project.create(location, name=name)
    ids = [23, 2, 42, 9, 10, 1, 19]
    try:
        source = _register_numeric_matrix(
            project, artifact_id="input", matrix=X, keys=ids
        )
        model = factory()
        output = project.translate(model, source)["output"]
        assert output.primary_key == ["doc_id"]
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.get_data_columns() == [
            f"cluster_{i}" for i in range(output.get_matrix().shape[1])
        ]
        op_id, artifact_id = str(model.operator_id), output.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        output = reopened.get_artifact(artifact_id)
        packet = output.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
            form="native", include_position=False,
        )
        assert packet["info"]["doc_id"].tolist() == ids
        assert packet["matrix"].shape == (len(X), 2)
        obj = reopened.get_operator(op_id)
        assert obj.is_frozen and obj.is_fitted
        assert obj.n_clusters_ == 2
        assert obj.execution_capabilities().reusable is False
        assert obj.to_json_state()["sklearn_version"] == sklearn.__version__
        assert obj.save_assets(tmp_path / "assets") == {}
        with pytest.raises(OperatorError, match="fit-only"):
            obj.translate(X)
        with pytest.raises(Exception):
            reopened.translate(obj, reopened.get_artifact("input"))
    finally:
        reopened.close()


def test_optics_all_noise_zero_width_project_roundtrip(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "optics_noise"
    project = teal.Project.create(location, name="optics_noise")
    try:
        source = _register_numeric_matrix(
            project, artifact_id="input", matrix=X,
            keys=[23, 2, 42, 9, 10, 1, 19],
        )
        obj = OPTICS(
            min_samples=3, max_eps=0.01,
            cluster_method="dbscan", eps=0.01,
        )
        output = project.translate(obj, source)["output"]
        assert output.get_matrix().shape == (7, 0)
        assert output.get_data_columns() == []
        aid = output.artifact_id
        oid = str(obj.operator_id)
    finally:
        project.close()
    reopened = teal.Project.open(location)
    try:
        output = reopened.get_artifact(aid)
        assert output.get_matrix().shape == (7, 0)
        assert reopened.get_operator(oid).n_clusters_ == 0
    finally:
        reopened.close()


def test_backend_version_provenance_across_clustering_family():
    models = [
        KMeans(n_clusters=2, random_state=17),
        DBSCAN(min_samples=2, eps=.4),
        HDBSCAN(min_samples=2, min_cluster_size=2),
        OPTICS(min_samples=2, cluster_method="dbscan", eps=.4, max_eps=1),
        AgglomerativeClustering(n_clusters=2),
    ]
    for model in models:
        assert model.to_json_state()["sklearn_version"] is None
        model.fit_transform(X)
        state = model.to_json_state()
        assert state["backend"] == "sklearn"
        assert state["sklearn_version"] == sklearn.__version__
        clone = type(model).from_json_state(state)
        assert clone.to_json_state()["sklearn_version"] == sklearn.__version__
        # Old snapshots remain loadable, but their historical backend version
        # must not be falsely assigned the version of the current environment.
        legacy_state = {k: v for k, v in state.items() if k not in (
            "backend", "sklearn_version",
        )}
        legacy = type(model).from_json_state(legacy_state)
        assert legacy.to_json_state()["sklearn_version"] is None


def test_optics_infinite_default_is_json_serializable():
    import json

    obj = OPTICS(min_samples=2)
    obj.fit_transform(X)
    state = obj.to_json_state()
    assert state["max_eps"] is None
    assert json.loads(json.dumps(state, allow_nan=False)) == state
    clone = OPTICS.from_json_state(state)
    assert np.isinf(clone.max_eps)
    assert clone.is_fitted
