"""Spherical K-means science and TeAL integration.

The directional objective and centroid normalization follow spherecluster's
SphericalKMeans reference implementation (jasonlaska/spherecluster) while
using a separate implementation independent of sklearn private internals.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import pairwise_distances_argmin

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import SphericalKMeans
from text_analysis_lab.translators.spherical_kmeans import _assign, _unit_rows
from test_clustering_translators import _register_numeric_matrix, _source


DIRECTIONS = np.asarray([
    [1.0, 0.01], [1.0, 0.12], [1.0, -0.12], [0.9, 0.18],
    [0.01, 1.0], [-0.15, 1.0], [0.13, 1.0], [-0.01, 0.9],
])


def _hard_labels(result):
    return np.asarray(result.argmax(axis=1)).ravel()


def test_dense_directional_fit_unit_centers_and_objective():
    model = SphericalKMeans(n_clusters=2, n_init=4, random_state=6)
    result = model.fit_transform(DIRECTIONS)
    assert sparse.isspmatrix_csr(result)
    assert result.shape == (len(DIRECTIONS), 2)
    assert result.nnz == len(DIRECTIONS)
    assert len(set(_hard_labels(result)[:4])) == 1
    assert len(set(_hard_labels(result)[4:])) == 1
    assert _hard_labels(result)[0] != _hard_labels(result)[-1]
    np.testing.assert_allclose(
        np.linalg.norm(model._centers, axis=1), np.ones(2), atol=1e-12,
    )
    unit = _unit_rows(DIRECTIONS)
    scores = np.max(unit @ model._centers.T, axis=1)
    np.testing.assert_allclose(model.inertia_, np.maximum(0, 1-scores).sum())
    np.testing.assert_array_equal(
        model.translate(DIRECTIONS).toarray(), result.toarray()
    )
    assert model.execution_capabilities().reusable
    assert 1 <= model.n_iter_ <= model.max_iter
    assert model.to_json_state()["backend"] == "numpy-scipy"
    assert model.to_json_state()["numpy_version"]
    assert model.to_json_state()["scipy_version"]


def test_spherical_direction_is_distinct_from_euclidean_centroid_norms():
    # This is the precise failure mode of simply L2-normalizing the rows
    # before running *ordinary* Euclidean KMeans: learned center magnitudes
    # change which center is nearest under Euclidean distance.
    sample = np.asarray([[0.8, 0.6]])
    nonunit_centers = np.asarray([[0.2, 0.0], [0.0, 0.8]])
    assert pairwise_distances_argmin(sample, nonunit_centers).tolist() == [1]
    directional_centers = _unit_rows(nonunit_centers)
    labels, _ = _assign(_unit_rows(sample), directional_centers)
    assert labels.tolist() == [0]


def test_dense_csr_equivalence_and_repeated_seed():
    kwargs = dict(n_clusters=2, n_init=5, random_state=12)
    dense = SphericalKMeans(**kwargs)
    csr = SphericalKMeans(**kwargs)
    a = dense.fit_transform(DIRECTIONS)
    b = csr.fit_transform(sparse.csr_matrix(DIRECTIONS))
    np.testing.assert_array_equal(a.toarray(), b.toarray())
    np.testing.assert_allclose(dense._centers, csr._centers, atol=1e-11)
    repeat = SphericalKMeans(**kwargs)
    np.testing.assert_array_equal(
        a.toarray(), repeat.fit_transform(DIRECTIONS).toarray()
    )
    # Prediction on new rows is directional and independent of their norms.
    new = np.asarray([[100.0, 2.0], [0.03, 10.0]])
    scaled = np.asarray([[1.0, 0.02], [0.3, 100.0]])
    np.testing.assert_array_equal(
        dense.translate(new).toarray(),
        dense.translate(scaled).toarray()
    )


def test_sparse_rows_and_shape_protection():
    rows = sparse.csr_matrix(np.hstack(
        [DIRECTIONS, np.zeros((len(DIRECTIONS), 1000))]
    ))
    model = SphericalKMeans(n_clusters=2, random_state=1, n_init=3)
    labels = model.fit_transform(rows)
    assert labels.shape == (len(DIRECTIONS), 2)
    assert model._centers.shape == (2, 1002)
    np.testing.assert_array_equal(model.translate(rows).toarray(), labels.toarray())
    with pytest.raises(ValueError, match="width"):
        model.translate(sparse.csr_matrix(DIRECTIONS))


def test_zero_vector_invalid_and_degenerate_reseeding():
    with pytest.raises(OperatorError, match="zero-length"):
        SphericalKMeans(n_clusters=2).fit_transform(
            np.asarray([[1.0, 0.0], [0, 0], [0, 1]])
        )
    with pytest.raises(Exception, match="finite"):
        SphericalKMeans(n_clusters=2).fit_transform(
            np.asarray([[np.inf, 0], [0, 1]])
        )
    with pytest.raises(ValueError, match="n_samples"):
        SphericalKMeans(n_clusters=4).fit_transform(DIRECTIONS[:3])
    # Fewer distinct directions than K: empty clusters must not produce NaNs
    # or non-unit stored centers, even when K cannot all be occupied.
    repeats = np.tile(np.asarray([[1.0, 0.0]]), (7, 1))
    fitted = SphericalKMeans(n_clusters=3, n_init=2, random_state=3)
    result = fitted.fit_transform(repeats)
    assert result.shape == (7, 3)
    assert np.all(np.isfinite(fitted._centers))
    np.testing.assert_allclose(np.linalg.norm(fitted._centers, axis=1), 1.0)
    assert result.nnz == 7


def test_parameter_validation():
    with pytest.raises(ValueError, match="positive integer"):
        SphericalKMeans(n_init=0)
    with pytest.raises(ValueError, match="positive"):
        SphericalKMeans(max_iter=0)
    with pytest.raises(ValueError, match="nonnegative"):
        SphericalKMeans(tol=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        SphericalKMeans(random_state=-5)


def test_structured_standalone_metadata_and_snapshots(tmp_path):
    model = SphericalKMeans(n_clusters=2, random_state=23, n_init=4)
    packet = {
        "values": DIRECTIONS,
        "columns": ["x", "y"],
        "metadata": pd.DataFrame({"description": list("abcdefgh")}),
    }
    result = model.fit_transform(packet)
    assert result["feature_metadata"]["column"].tolist() == [
        "cluster_0", "cluster_1",
    ]
    assert result["metadata"]["description"].tolist() == list("abcdefgh")
    manifest = model.save_assets(tmp_path)
    assert manifest == {"centers_file": "centers.npy"}
    clone = SphericalKMeans.from_json_state(model.to_json_state())
    clone.load_assets(tmp_path, manifest)
    assert clone.is_fitted
    assert clone.n_init == 4
    assert clone.to_json_state()["backend"] == "numpy-scipy"
    assert clone.to_json_state()["numpy_version"] == model.numpy_version_
    np.testing.assert_array_equal(
        clone.translate(packet)["values"].toarray(),
        result["values"].toarray()
    )
    with pytest.raises(OperatorError, match="ordered feature schema"):
        clone.translate({"values": DIRECTIONS, "columns": ["y", "x"]})
    with pytest.raises(OperatorError, match="already fitted"):
        model.fit_transform(DIRECTIONS)


def test_full_fit_preserves_nontrivial_source_keys():
    keys = [23, 2, 42, 9, 10, 1, 19, 7]
    model = SphericalKMeans(n_clusters=2, n_init=4, random_state=3)
    request = TranslationRequest(batch_size=3)
    assert model.input_request(
        sources={"source": _source()}, mode="fit_translate", request=request
    ).mode == "full_artifact"
    packet = InputBatch(
        source_label="source", artifact_id="source-1", primary_key=("doc_id",),
        data={"info": pd.DataFrame({"doc_id": keys}), "matrix": DIRECTIONS},
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )
    output = model.translate_batch(
        {"source": packet}, mode="fit_translate", request=request
    ).outputs["output"]
    assert output["keys"]["doc_id"].tolist() == keys
    assert output["data"]["columns"] == ["cluster_0", "cluster_1"]
    assert sparse.isspmatrix_csr(output["data"]["values"])


def test_project_persistence_frozen_reuse_and_batches(tmp_path):
    import text_analysis_lab as teal

    location = tmp_path / "spherical"
    ids = [23, 2, 42, 9, 10, 1, 19, 7]
    project = teal.Project.create(location, name="spherical")
    try:
        source = _register_numeric_matrix(
            project, artifact_id="directional_train",
            matrix=sparse.csr_matrix(DIRECTIONS), keys=ids,
        )
        model = SphericalKMeans(n_clusters=2, random_state=9, n_init=4)
        output = project.translate(model, source)["output"]
        assert output.artifact_type.value == "sparse_matrix"
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.get_data_columns() == ["cluster_0", "cluster_1"]
        original = output.get_matrix().toarray()
        op_id, artifact_id = str(model.operator_id), output.artifact_id
    finally:
        project.close()

    project = teal.Project.open(location)
    try:
        output = project.get_artifact(artifact_id)
        queried = output.query(
            key_columns=True, data_columns=True, metadata_columns=False,
            metadata_mode="none", form="native", include_position=False,
        )
        assert queried["info"]["doc_id"].tolist() == ids
        np.testing.assert_array_equal(queried["matrix"].toarray(), original)
        frozen = project.get_operator(op_id)
        assert frozen.is_fitted and frozen.is_frozen
        assert frozen.execution_capabilities().reusable
        assert frozen.input_request(
            sources={"source": project.get_artifact("directional_train")},
            mode="translate", request=TranslationRequest(batch_size=3),
        ).mode == "batches"
        second = project.translate(
            frozen, project.get_artifact("directional_train"), batch_size=3,
        )["output"]
        np.testing.assert_array_equal(second.get_matrix().toarray(), original)
        assert frozen.to_json_state()["numpy_version"]
    finally:
        project.close()
