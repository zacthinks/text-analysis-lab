"""Phase 4 clustering scientific parity and persistent TeAL contracts."""

import numpy as np
import pandas as pd
import pytest
import sklearn
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans as SkMini
from sklearn.cluster import SpectralClustering as SkSpectral
from sklearn.mixture import GaussianMixture as SkGMM

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import MiniBatchKMeans, GaussianMixture, SpectralClustering
from test_clustering_translators import X, _source, _register_numeric_matrix


def test_minibatch_upstream_parity_and_reuse(tmp_path):
    kwargs = dict(n_clusters=2, random_state=12, n_init=4, batch_size=4,
                  reassignment_ratio=0.0)
    model = MiniBatchKMeans(**kwargs)
    fitted = model.fit_transform(sparse.csr_matrix(X))
    upstream = SkMini(**kwargs).fit_predict(sparse.csr_matrix(X))
    np.testing.assert_array_equal(np.asarray(fitted.argmax(axis=1)).ravel(), upstream)
    assert model.execution_capabilities().reusable
    np.testing.assert_array_equal(
        model.translate(X).toarray(), model.translate(sparse.csr_matrix(X)).toarray()
    )
    assert model.to_json_state()["batch_size"] == 4
    assert model.to_json_state()["sklearn_version"] == sklearn.__version__
    manifest = model.save_assets(tmp_path)
    assert manifest == {"centers_file": "centers.npy"}
    clone = MiniBatchKMeans.from_json_state(model.to_json_state())
    clone.load_assets(tmp_path, manifest)
    np.testing.assert_array_equal(clone.translate(X).toarray(), model.translate(X).toarray())
    with pytest.raises(OperatorError, match="already fitted"):
        model.fit_transform(X)


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_gaussian_mixture_upstream_parity_roundtrip(tmp_path, covariance_type):
    kwargs = dict(n_components=2, covariance_type=covariance_type,
                  random_state=7, n_init=2, reg_covar=1e-4)
    upstream = SkGMM(**kwargs).fit(X)
    model = GaussianMixture(**kwargs)
    actual = model.fit_transform({"values": X, "columns": ["x", "y"]})
    values = actual["values"]
    assert isinstance(values, np.ndarray)
    assert values.shape == (len(X), 2)
    assert actual["feature_metadata"]["column"].tolist() == ["cluster_0", "cluster_1"]
    np.testing.assert_allclose(values, upstream.predict_proba(X), atol=1e-8)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-10)
    assert model.execution_capabilities().reusable
    assert model.to_json_state()["sklearn_version"] == sklearn.__version__
    assert model.output_specs(sources={"source": _source()}, request=TranslationRequest()).artifact_type == "dense_matrix"
    with pytest.raises(OperatorError, match="dense"):
        GaussianMixture(**kwargs).fit_transform(sparse.csr_matrix(X))
    manifest = model.save_assets(tmp_path)
    assert manifest == {"mixture_file": "mixture.npz"}
    clone = GaussianMixture.from_json_state(model.to_json_state())
    clone.load_assets(tmp_path, manifest)
    np.testing.assert_allclose(clone.translate(X), model.translate(X), atol=1e-10)
    with pytest.raises(OperatorError, match="ordered feature schema"):
        clone.translate({"values": X, "columns": ["y", "x"]})
    with pytest.raises(OperatorError, match="already fitted"):
        model.fit_transform(X)


@pytest.mark.parametrize("affinity", ["rbf", "nearest_neighbors"])
def test_spectral_upstream_and_fit_only(affinity):
    kwargs = dict(n_clusters=2, affinity=affinity, n_neighbors=3,
                  random_state=2, assign_labels="cluster_qr")
    upstream = SkSpectral(**kwargs).fit_predict(X)
    obj = SpectralClustering(**kwargs)
    result = obj.fit_transform(X)
    assert sparse.isspmatrix_csr(result)
    assert result.shape == (len(X), 2)
    np.testing.assert_array_equal(np.asarray(result.argmax(axis=1)).ravel(), upstream)
    clone = SpectralClustering.from_json_state(obj.to_json_state())
    assert clone.is_fitted and clone.n_clusters_ == 2
    assert clone.to_json_state()["sklearn_version"] == sklearn.__version__
    assert obj.save_assets(None) == {}
    assert not obj.execution_capabilities().reusable
    with pytest.raises(OperatorError, match="fit-only"):
        clone.translate(X)


def test_spectral_rejects_unimplemented_precomputed():
    with pytest.raises(ValueError, match="precomputed"):
        SpectralClustering(affinity="precomputed")


@pytest.mark.parametrize("factory,kind", [
    (lambda: MiniBatchKMeans(n_clusters=2, random_state=7, n_init=3, batch_size=4, reassignment_ratio=0), "sparse"),
    (lambda: GaussianMixture(n_components=2, random_state=7, n_init=2), "dense"),
    (lambda: SpectralClustering(n_clusters=2, random_state=7, assign_labels="cluster_qr"), "sparse"),
])
def test_phase4_project_persistence_and_keys(tmp_path, factory, kind):
    import text_analysis_lab as teal

    project = teal.Project.create(tmp_path / "phase4", name="phase4")
    ids = [23, 2, 42, 9, 10, 1, 19]
    try:
        source = _register_numeric_matrix(
            project, artifact_id="phase4_source", matrix=X, keys=ids
        )
        obj = factory()
        assert obj.input_request(
            sources={"source": source}, mode="fit_translate",
            request=TranslationRequest(batch_size=2),
        ).mode == "full_artifact"
        output = project.translate(obj, source)["output"]
        assert output.artifact_type.value == kind + "_matrix"
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.get_data_columns() == ["cluster_0", "cluster_1"]
        packet = output.query(key_columns=True, data_columns=True,
                              metadata_columns=False, metadata_mode="none",
                              form="native", include_position=False)
        assert packet["info"]["doc_id"].tolist() == ids
        np.testing.assert_allclose(
            packet["matrix"].sum(axis=1),
            np.ones((len(X), 1)) if kind == "sparse" else np.ones(len(X)),
            atol=1e-8,
        )
        artifact_id, operator_id = output.artifact_id, str(obj.operator_id)
    finally:
        project.close()
    reopened = teal.Project.open(tmp_path / "phase4")
    try:
        assert reopened.get_artifact(artifact_id).get_matrix().shape == (7, 2)
        saved = reopened.get_operator(operator_id)
        assert saved.is_fitted and saved.is_frozen
        assert saved.to_json_state()["sklearn_version"] == sklearn.__version__
        if kind == "dense":
            np.testing.assert_allclose(saved.translate(X).sum(axis=1), 1.0)
        elif isinstance(saved, MiniBatchKMeans):
            assert saved.execution_capabilities().reusable
            assert saved.input_request(
                sources={"source": reopened.get_artifact("phase4_source")},
                mode="translate", request=TranslationRequest(batch_size=2),
            ).mode == "batches"
        else:
            with pytest.raises(OperatorError, match="fit-only"):
                saved.translate(X)
    finally:
        reopened.close()
