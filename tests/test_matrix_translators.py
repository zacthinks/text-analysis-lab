from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer as SklearnTfidf

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import (
    LDA,
    LSA,
    SVD,
    UMAP,
    MatrixNormalizer,
    MatrixTranspose,
    TfidfTransformer,
)

COUNTS = sparse.csr_matrix(
    np.array(
        [
            [3, 0, 1, 0, 2],
            [0, 2, 0, 1, 1],
            [1, 1, 0, 3, 0],
            [0, 0, 4, 1, 0],
            [2, 1, 1, 0, 1],
            [0, 3, 0, 2, 0],
        ],
        dtype=float,
    )
)
FEATURES = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _source(kind: str = "sparse_matrix"):
    return SimpleNamespace(
        artifact_id="art_counts",
        artifact_type=ArtifactType(kind),
        primary_key=["doc_id"],
        get_data_columns=lambda: list(FEATURES),
    )


def _packet(matrix=COUNTS) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_counts",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": list(range(matrix.shape[0]))}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_matrix_transpose_promotes_feature_frame_to_row_metadata() -> None:
    feature_frame = pd.DataFrame(
        {
            "column_index": np.arange(len(FEATURES), dtype=np.int64),
            "column": FEATURES,
            "family": ["a", "a", "b", "b", "c"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    source = SimpleNamespace(
        artifact_type=ArtifactType.SPARSE_MATRIX,
        primary_key=["doc_id"],
        get_data_columns=lambda: list(FEATURES),
        get_feature_frame=lambda: feature_frame.copy(),
        has_row_names=False,
    )
    translator = MatrixTranspose()
    request = translator.input_request(
        sources={"source": source},
        mode="translate",
        request=TranslationRequest(),
    )
    assert request.mode == "full_artifact"

    result = translator.translate_batch(
        {"source": _packet()},
        mode="translate",
        request=TranslationRequest(),
    )
    output = result.outputs["output"]
    assert output["keys"]["feature_id"].tolist() == list(range(len(FEATURES)))
    assert output["data"]["row_names"] == FEATURES
    assert output["data"]["row_name"] == "feature"
    assert sparse.isspmatrix_csr(output["data"]["values"])
    assert output["data"]["values"].shape == (len(FEATURES), COUNTS.shape[0])

    metadata = output["metadata"]
    assert metadata.columns.tolist() == ["column", "family", "score"]
    assert metadata["column"].tolist() == FEATURES
    assert metadata["family"].tolist() == ["a", "a", "b", "b", "c"]
    assert metadata["score"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_tfidf_defaults_to_weighting_without_implicit_normalization() -> None:
    translator = TfidfTransformer()
    request = translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(batch_size=2),
    )
    assert request.mode == "full_artifact"
    assert request.batch_size is None

    result = translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    observed = result.outputs["output"]["data"]["values"]
    expected = SklearnTfidf(norm=None).fit_transform(COUNTS)
    assert sparse.isspmatrix_csr(observed)
    assert np.allclose(observed.toarray(), expected.toarray())
    assert result.outputs["output"]["data"]["columns"] == FEATURES


def test_tfidf_fitted_schema_is_reused_and_mismatch_rejected() -> None:
    translator = TfidfTransformer(norm="l2")
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    batch = translator.translate_batch(
        {"source": _packet(COUNTS[:2])}, mode="translate", request=TranslationRequest()
    )
    norms = np.sqrt(
        np.asarray(
            batch.outputs["output"]["data"]["values"].power(2).sum(axis=1)
        ).reshape(-1)
    )
    assert norms.tolist() == pytest.approx([1.0, 1.0])

    bad = SimpleNamespace(
        artifact_type=ArtifactType.SPARSE_MATRIX,
        primary_key=["doc_id"],
        get_data_columns=lambda: [*FEATURES[:-1], "changed"],
    )
    with pytest.raises(OperatorError, match="same ordered feature schema"):
        translator.input_request(
            sources={"source": bad}, mode="translate", request=TranslationRequest()
        )


def test_matrix_normalizer_row_and_column_l1_l2() -> None:
    row = MatrixNormalizer(axis="rows", norm="l2")
    request = row.input_request(
        sources={"source": _source()},
        mode="translate",
        request=TranslationRequest(batch_size=2),
    )
    assert request.mode == "batches"
    result = row.translate_batch(
        {"source": _packet()}, mode="translate", request=TranslationRequest()
    )
    values = result.outputs["output"]["data"]["values"]
    row_norms = np.sqrt(np.asarray(values.power(2).sum(axis=1)).reshape(-1))
    assert row_norms.tolist() == pytest.approx([1.0] * COUNTS.shape[0])

    col = MatrixNormalizer(axis="columns", norm="l1")
    request = col.input_request(
        sources={"source": _source()},
        mode="translate",
        request=TranslationRequest(batch_size=2),
    )
    assert request.mode == "full_artifact"
    assert request.batch_size is None
    result = col.translate_batch(
        {"source": _packet()}, mode="translate", request=TranslationRequest()
    )
    values = result.outputs["output"]["data"]["values"]
    col_norms = np.asarray(abs(values).sum(axis=0)).reshape(-1)
    assert col_norms.tolist() == pytest.approx([1.0] * COUNTS.shape[1])


def test_matrix_normalizer_parallel_resume_state_round_trip(tmp_path) -> None:
    translator = MatrixNormalizer(axis="rows", norm="l2")
    translator.input_request(
        sources={"source": _source()},
        mode="translate",
        request=TranslationRequest(batch_size=2),
    )
    assert translator.supports_resume(mode="translate", route="parallel")

    state_path = tmp_path / "normalizer_state"
    translator.save_intermediate_state(
        state_path,
        operator_id="optr_normalizer",
        mode="translate",
        route="parallel",
    )
    restored = MatrixNormalizer.load_intermediate_state(
        state_path,
        operator_id="optr_normalizer",
        mode="translate",
        route="parallel",
    )

    assert restored.operator_id == "optr_normalizer"
    assert restored.axis == "rows"
    assert restored.norm == "l2"
    assert restored._source_type == "sparse_matrix"
    assert restored._columns == tuple(FEATURES)
    result = restored.translate_batch(
        {"source": _packet(COUNTS[:2])},
        mode="translate",
        request=TranslationRequest(),
    )
    assert result.outputs["output"]["data"]["columns"] == FEATURES
    assert sparse.isspmatrix_csr(result.outputs["output"]["data"]["values"])


def test_svd_emits_document_coordinates_and_component_loadings_and_lsa_is_alias() -> (
    None
):
    assert LSA is SVD
    translator = SVD(n_components=2, random_state=7)
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    specs = translator.output_specs(
        sources={"source": _source()}, request=TranslationRequest()
    )
    assert set(specs) == {"output", "components"}
    result = translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["values"].shape == (6, 2)
    components = result.outputs["components"]
    assert components["data"]["values"].shape == (2, 5)
    assert components["data"]["columns"] == FEATURES
    assert components["keys"]["component_id"].tolist() == [0, 1]
    assert set(components["metadata"].columns) == {
        "explained_variance",
        "explained_variance_ratio",
        "singular_value",
    }
    # Fitted reuse does not redundantly emit another component artifact.
    assert set(
        translator.output_specs(
            sources={"source": _source()}, request=TranslationRequest()
        )
    ) == {"output"}


def test_lda_emits_document_topic_distribution_and_topic_term_weights() -> None:
    translator = LDA(n_components=2, max_iter=4, random_state=7)
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    result = translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    doc_topics = result.outputs["output"]["data"]["values"]
    assert doc_topics.shape == (6, 2)
    assert doc_topics.sum(axis=1).tolist() == pytest.approx([1.0] * 6)
    topics = result.outputs["topics"]
    assert topics["data"]["values"].shape == (2, 5)
    assert topics["data"]["columns"] == FEATURES
    assert topics["keys"]["topic_id"].tolist() == [0, 1]


def test_lda_rejects_weighted_or_negative_inputs() -> None:
    weighted = COUNTS.copy().astype(float)
    weighted[0, 0] = 0.5
    lda = LDA(n_components=2)
    with pytest.raises(ArtifactError, match="integer-valued count"):
        lda.translate_batch(
            {"source": _packet(weighted)},
            mode="fit_translate",
            request=TranslationRequest(),
        )


def test_umap_lifecycle_without_optional_runtime(monkeypatch) -> None:
    class FakeUMAP:
        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components
            self.kwargs = kwargs

        def fit_transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            self.offset_ = arr.mean(axis=0)
            return arr[:, : self.n_components]

        def transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return arr[:, : self.n_components]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    translator = UMAP(
        n_components=2,
        n_neighbors=3,
        random_state=7,
        reuse="stored",
        storage="native",
    )
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    result = translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["values"].shape == (6, 2)
    reused = translator.translate_batch(
        {"source": _packet(COUNTS[:2])}, mode="translate", request=TranslationRequest()
    )
    assert reused.outputs["output"]["data"]["values"].shape == (2, 2)


def test_umap_view_only_mode_does_not_persist_estimator(monkeypatch, tmp_path) -> None:
    class FakeUMAP:
        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components
            self.kwargs = kwargs

        def fit_transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            self.large_runtime_state_ = bytearray(1024)
            return arr[:, : self.n_components]

        def transform(self, X):
            raise AssertionError("view-only UMAP should not be reusable")

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    translator = UMAP(n_components=2, n_neighbors=3, random_state=7, reuse="none")
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    result = translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["values"].shape == (6, 2)
    assert not translator.is_fitted
    assert not translator.supports_fit_translate

    path = tmp_path / "umap_view_only"
    translator.save_to_dir(path, operator_id="op_umap_view_only")
    assert not (path / "assets" / "umap.pkl").exists()

    restored = translator.load_from_dir(path)
    assert not restored.is_fitted
    assert not restored.supports_fit_translate
    assert restored.reuse == "none"
    assert restored.storage is None
    assert restored.retain_estimator is False


def test_umap_reuse_storage_contract_validation() -> None:
    assert UMAP().reuse == "none"
    assert UMAP().storage is None

    recompute = UMAP(reuse="recompute")
    assert recompute.reuse == "recompute"
    assert recompute.storage is None

    stored = UMAP(reuse="stored")
    assert stored.reuse == "stored"
    assert stored.storage == "compact"

    native = UMAP(reuse="stored", storage="native")
    assert native.storage == "native"

    with pytest.raises(ValueError, match="only applicable"):
        UMAP(reuse="none", storage="compact")
    with pytest.raises(ValueError, match="only applicable"):
        UMAP(reuse="recompute", storage="native")
    with pytest.raises(ValueError, match="reuse must be one of"):
        UMAP(reuse="mystery")  # type: ignore[arg-type]


def test_umap_recompute_rebuilds_from_fitting_artifact(monkeypatch) -> None:
    class FakeUMAP:
        fit_calls = 0

        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components
            self.kwargs = kwargs

        def fit_transform(self, X):
            type(self).fit_calls += 1
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return arr[:, : self.n_components]

        def fit(self, X):
            type(self).fit_calls += 1
            self.seen_shape = X.shape
            return self

        def transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return arr[:, : self.n_components]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    translator = UMAP(n_components=2, n_neighbors=3, random_state=7, reuse="recompute")
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert not translator.is_fitted
    assert translator.fit_source_artifact_id_ == "art_counts"

    class FitSource:
        artifact_type = ArtifactType.SPARSE_MATRIX

        def get_data_columns(self):
            return list(FEATURES)

        def get_matrix(self):
            return COUNTS

    class FakeProject:
        def get_artifact(self, artifact_id):
            assert artifact_id == "art_counts"
            return FitSource()

    translator.prepare_for_translation(
        project=FakeProject(),  # type: ignore[arg-type]
        sources={"source": _source()},
    )
    assert translator.is_fitted
    assert FakeUMAP.fit_calls == 2
    reused = translator.translate_batch(
        {"source": _packet(COUNTS[:2])}, mode="translate", request=TranslationRequest()
    )
    assert reused.outputs["output"]["data"]["values"].shape == (2, 2)


def test_umap_recompute_snapshot_keeps_recipe_without_estimator(
    monkeypatch, tmp_path
) -> None:
    class FakeUMAP:
        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components

        def fit_transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return arr[:, : self.n_components]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    translator = UMAP(n_components=2, n_neighbors=3, reuse="recompute")
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )

    path = tmp_path / "umap_recompute"
    translator.save_to_dir(path, operator_id="op_umap_recompute")
    assert list((path / "assets").iterdir()) == []

    restored = translator.load_from_dir(path)
    assert restored.reuse == "recompute"
    assert restored.storage is None
    assert restored.fit_source_artifact_id_ == "art_counts"
    assert restored.source_features_ == tuple(FEATURES)
    assert restored._fit_completed is True
    assert not restored.is_fitted


def test_umap_legacy_state_maps_to_previous_persistence_semantics() -> None:
    retained = UMAP.from_json_state(
        {
            "n_components": 2,
            "n_neighbors": 15,
            "retain_estimator": True,
            "fit_completed": True,
        }
    )
    assert retained.reuse == "stored"
    assert retained.storage == "native"

    view_only = UMAP.from_json_state(
        {
            "n_components": 2,
            "n_neighbors": 15,
            "retain_estimator": False,
            "fit_completed": True,
        }
    )
    assert view_only.reuse == "none"
    assert view_only.storage is None


def test_umap_none_rejects_reuse_after_fit(monkeypatch) -> None:
    class FakeUMAP:
        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components

        def fit_transform(self, X):
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            return arr[:, : self.n_components]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    translator = UMAP(n_components=2, n_neighbors=3, reuse="none")
    translator.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    translator.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )

    with pytest.raises(OperatorError, match="reuse='none'"):
        translator.prepare_for_translation(
            project=SimpleNamespace(),  # type: ignore[arg-type]
            sources={"source": _source()},
        )


def test_fitted_sklearn_matrix_operators_round_trip_assets(tmp_path) -> None:
    fitted = []

    tfidf = TfidfTransformer()
    tfidf.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    tfidf.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    fitted.append((tfidf, "tfidf"))

    svd = SVD(n_components=2, random_state=7)
    svd.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    svd.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    fitted.append((svd, "svd"))

    lda = LDA(n_components=2, max_iter=3, random_state=7)
    lda.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    lda.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    fitted.append((lda, "lda"))

    for operator, name in fitted:
        path = tmp_path / name
        operator.save_to_dir(path, operator_id=f"op_{name}")
        restored = operator.load_from_dir(path)
        assert restored.is_fitted
        result = restored.translate_batch(
            {"source": _packet(COUNTS[:2])},
            mode="translate",
            request=TranslationRequest(),
        )
        assert result.outputs["output"]["data"]["values"].shape[0] == 2


def test_estimator_assets_stream_to_and_from_disk_without_full_bytes_buffer(
    tmp_path, monkeypatch
) -> None:
    from text_analysis_lab.translators import _matrix_transform_utils as utils

    estimator = {"weights": np.arange(100, dtype=float)}
    path = tmp_path / "estimator.pkl"

    def fail_dumps(*args, **kwargs):
        raise AssertionError(
            "dump_estimator must not materialize the full pickle bytes in memory"
        )

    monkeypatch.setattr(utils.cloudpickle, "dumps", fail_dumps)
    utils.dump_estimator(path, estimator)
    assert path.exists()

    def fail_loads(*args, **kwargs):
        raise AssertionError(
            "load_estimator must not read the full pickle bytes into memory"
        )

    monkeypatch.setattr(utils.cloudpickle, "loads", fail_loads)
    restored = utils.load_estimator(path)
    assert np.array_equal(restored["weights"], estimator["weights"])
