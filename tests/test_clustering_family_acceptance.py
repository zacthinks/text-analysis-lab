"""Cross-family contract acceptance for all ten TeAL clustering translators.

Tests one durable workflow per method, not ten isolated backend demonstrations:
key alignment, matrix schema, snapshot/reload, scientific fit-only distinction,
and deterministic new-row application when the method supports it.
Synthetic numeric fixtures only; no AERA/course data in TeAL CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    AgglomerativeClustering,
    DBSCAN,
    GaussianMixture,
    HDBSCAN,
    KMeans,
    KMedoids,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
    SphericalKMeans,
)
from test_clustering_translators import _register_numeric_matrix


X = np.asarray(
    [
        [3.0, 0.1], [2.8, 0.2], [3.2, 0.1],
        [0.2, 3.0], [0.1, 2.9], [0.3, 3.1],
        [5.0, 5.0],
    ],
    dtype=np.float64,
)
KEYS = [23, 2, 42, 9, 10, 1, 19]


CASES = [
    ("kmeans", lambda: KMeans(n_clusters=2, n_init=4, random_state=4), True, False),
    ("minibatch", lambda: MiniBatchKMeans(
        n_clusters=2, batch_size=4, n_init=4,
        reassignment_ratio=0, random_state=4,
    ), True, False),
    ("kmedoids", lambda: KMedoids(n_clusters=2, init="build"), True, False),
    ("spherical", lambda: SphericalKMeans(n_clusters=2, n_init=3, random_state=4), True, False),
    ("dbscan", lambda: DBSCAN(eps=0.5, min_samples=2), False, False),
    ("hdbscan", lambda: HDBSCAN(min_cluster_size=2, min_samples=2), False, False),
    ("optics", lambda: OPTICS(
        min_samples=2, max_eps=1.0, cluster_method="dbscan", eps=0.5,
    ), False, False),
    ("agglomerative", lambda: AgglomerativeClustering(n_clusters=2), False, False),
    ("gaussian", lambda: GaussianMixture(n_components=2, n_init=2, random_state=4), True, True),
    ("spectral", lambda: SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors",
        n_neighbors=3, assign_labels="cluster_qr", random_state=4,
    ), False, False),
]


def _values(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("name,factory,reusable,soft", CASES, ids=[v[0] for v in CASES])
def test_clustering_family_project_freeze_and_lineage(
    tmp_path, name, factory, reusable, soft
):
    if name == "kmedoids":
        pytest.importorskip("kmedoids", reason="optional Rust FasterPAM extra")
    location = tmp_path / name
    project = teal.Project.create(location, name=name)
    try:
        source = _register_numeric_matrix(
            project, artifact_id="input", matrix=X, keys=KEYS
        )
        model = factory()
        first = project.translate(model, source)["output"]
        matrix = _values(first.get_matrix())
        assert matrix.shape[0] == len(KEYS)
        assert first.primary_key == ["doc_id"]
        assert first.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert first.get_data_columns() == [
            f"cluster_{i}" for i in range(matrix.shape[1])
        ]
        if soft:
            assert first.artifact_type.value == "dense_matrix"
            assert np.all(np.isfinite(matrix))
            assert np.all((0 <= matrix) & (matrix <= 1))
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)
        else:
            assert first.artifact_type.value == "sparse_matrix"
            assert sparse.isspmatrix_csr(first.get_matrix())
            assert set(np.unique(matrix)).issubset({0, 1})
            # Density methods may leave some observations without a cluster.
            if name in {"dbscan", "hdbscan", "optics"}:
                assert np.all(matrix.sum(axis=1) <= 1)
            else:
                assert np.all(matrix.sum(axis=1) == 1)
        saved_id, op_id = first.artifact_id, str(model.operator_id)
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        first = reopened.get_artifact(saved_id)
        packet = first.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
            form="native", include_position=False,
        )
        assert packet["info"]["doc_id"].tolist() == KEYS
        np.testing.assert_allclose(_values(packet["matrix"]), matrix, atol=1e-10)
        assert first.get_feature_metadata()["column"].tolist() == [
            f"cluster_{i}" for i in range(matrix.shape[1])
        ]

        frozen = reopened.get_operator(op_id)
        assert frozen.is_frozen and frozen.is_fitted
        assert frozen.execution_capabilities().reusable is reusable
        state = frozen.to_json_state()
        assert state.get("backend") in {"sklearn", "rust-kmedoids", "numpy-scipy"}
        if state["backend"] == "sklearn":
            assert state["sklearn_version"]
        elif state["backend"] == "rust-kmedoids":
            assert state["backend_version"]
        else:
            assert state["numpy_version"] and state["scipy_version"]

        source = reopened.get_artifact("input")
        if reusable:
            replayed = reopened.translate(frozen, source, batch_size=3)["output"]
            assert replayed.primary_key == ["doc_id"]
            np.testing.assert_allclose(
                _values(replayed.get_matrix()), matrix, atol=1e-8
            )
            assert frozen.input_request(
                sources={"source": source}, mode="translate",
                request=teal.core.operator.TranslationRequest(batch_size=3),
            ).mode == "batches"
        else:
            with pytest.raises(Exception):
                reopened.translate(frozen, source, batch_size=3)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "name,factory",
    [
        ("kmeans", lambda: KMeans(n_clusters=2, random_state=5, n_init=3)),
        ("minibatch", lambda: MiniBatchKMeans(
            n_clusters=2, random_state=5, n_init=3,
            batch_size=4, reassignment_ratio=0,
        )),
        ("kmedoids", lambda: KMedoids(n_clusters=2, init="build")),
        ("spherical", lambda: SphericalKMeans(n_clusters=2, random_state=5, n_init=3)),
        ("dbscan", lambda: DBSCAN(eps=0.5, min_samples=2)),
        ("hdbscan", lambda: HDBSCAN(min_cluster_size=2, min_samples=2)),
        ("spectral", lambda: SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors",
            n_neighbors=3, assign_labels="cluster_qr",
        )),
    ],
)
def test_clustering_family_sparse_inputs_without_changing_output_kind(name, factory):
    if name == "kmedoids":
        pytest.importorskip("kmedoids")
    model = factory()
    output = model.fit_transform(sparse.csr_matrix(X))
    assert sparse.isspmatrix_csr(output)
    assert output.shape[0] == len(X)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AgglomerativeClustering(n_clusters=2),
        lambda: GaussianMixture(n_components=2, random_state=5),
    ],
)
def test_dense_only_clustering_never_silently_densifies(factory):
    from text_analysis_lab.core.errors import OperatorError

    with pytest.raises(OperatorError, match="dense"):
        factory().fit_transform(sparse.csr_matrix(X))


def test_density_noise_is_not_an_extra_cluster_feature():
    noise = np.asarray([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0], [30.0, 1.0]])
    for model in (
        DBSCAN(eps=0.01, min_samples=2),
        OPTICS(min_samples=2, cluster_method="dbscan",
               max_eps=0.01, eps=0.01),
    ):
        values = model.fit_transform(noise)
        assert values.shape == (4, 0)
        assert values.nnz == 0
