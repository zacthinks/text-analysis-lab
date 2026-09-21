from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import (
    LDA,
    SVD,
    UMAP,
    MatrixNormalizer,
    TfidfTransformer,
)

WORKERS = 2 if importlib.util.find_spec("distributed") is not None else 1

VALUES = np.array(
    [
        [3, 0, 1, 0, 2],
        [0, 2, 0, 1, 1],
        [1, 1, 0, 3, 0],
        [0, 0, 4, 1, 0],
        [2, 1, 1, 0, 1],
        [0, 3, 0, 2, 0],
        [2, 0, 2, 1, 0],
        [0, 1, 0, 2, 3],
    ],
    dtype=float,
)
FEATURES = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _register_counts(project: teal.Project):
    artifact_id = "art_counts_round14"
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="counts",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    for start in (0, 4):
        stop = start + 4
        writer.write(
            {
                "keys": pd.DataFrame({"doc_id": list(range(start, stop))}),
                "data": {
                    "values": sparse.csr_matrix(VALUES[start:stop]),
                    "columns": FEATURES,
                },
            }
        )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix",
        label="counts",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_real_tfidf_normalization_svd_lda_round_trip(tmp_path: Path) -> None:
    project_path = tmp_path / "round14"
    project = teal.Project.create(project_path, name="round14")
    try:
        counts = _register_counts(project)

        tfidf_op = TfidfTransformer()
        tfidf = project.translate(tfidf_op, counts)["output"]
        assert tfidf.artifact_type.value == "sparse_matrix"
        assert tfidf.primary_key == ["doc_id"]
        assert tfidf.get_data_columns() == FEATURES
        assert tfidf.get_matrix().shape == VALUES.shape

        normalized = project.translate(
            MatrixNormalizer(axis="rows", norm="l2"),
            tfidf,
            batch_size=3,
            workers=WORKERS,
        )["output"]
        norms = np.sqrt(
            np.asarray(normalized.get_matrix().power(2).sum(axis=1)).reshape(-1)
        )
        assert norms.tolist() == pytest.approx([1.0] * len(VALUES))

        column_normalized = project.translate(
            MatrixNormalizer(axis="columns", norm="l1"), counts, batch_size=3
        )["output"]
        col_norms = np.asarray(abs(column_normalized.get_matrix()).sum(axis=0)).reshape(
            -1
        )
        assert col_norms.tolist() == pytest.approx([1.0] * len(FEATURES))

        svd_op = SVD(n_components=2, random_state=7)
        svd_outputs = project.translate(svd_op, normalized)
        reduced = svd_outputs["output"]
        components = svd_outputs["components"]
        assert reduced.get_matrix().shape == (len(VALUES), 2)
        assert components.get_matrix().shape == (2, len(FEATURES))
        assert components.primary_key == ["component_id"]
        component_metadata = components.query(
            metadata_columns=True,
            metadata_mode="local",
            data_columns=False,
        )
        assert component_metadata["explained_variance_ratio"].notna().all()

        lda_op = LDA(n_components=2, max_iter=4, random_state=7)
        lda_outputs = project.translate(lda_op, counts)
        doc_topics = lda_outputs["output"]
        topics = lda_outputs["topics"]
        assert doc_topics.get_matrix().shape == (len(VALUES), 2)
        assert doc_topics.get_matrix().sum(axis=1).tolist() == pytest.approx(
            [1.0] * len(VALUES)
        )
        assert topics.get_matrix().shape == (2, len(FEATURES))
        assert topics.primary_key == ["topic_id"]

        tfidf_operator_id = tfidf_op.operator_id
        svd_operator_id = svd_op.operator_id
        lda_operator_id = lda_op.operator_id
        ids = {
            name: artifact.artifact_id
            for name, artifact in {
                "tfidf": tfidf,
                "normalized": normalized,
                "reduced": reduced,
                "components": components,
                "doc_topics": doc_topics,
                "topics": topics,
            }.items()
        }
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        for artifact_id in ids.values():
            assert reopened.get_artifact(artifact_id).status == "complete"

        fitted_tfidf = reopened.get_operator(str(tfidf_operator_id))
        reused_tfidf = reopened.translate(
            fitted_tfidf,
            reopened.get_artifact("art_counts_round14"),
            batch_size=3,
            workers=WORKERS,
        )["output"]
        assert np.allclose(
            reused_tfidf.get_matrix().toarray(),
            reopened.get_artifact(ids["tfidf"]).get_matrix().toarray(),
        )

        fitted_svd = reopened.get_operator(str(svd_operator_id))
        reused_svd = reopened.translate(
            fitted_svd,
            reopened.get_artifact(ids["normalized"]),
            batch_size=3,
            workers=WORKERS,
        )
        assert set(reused_svd) == {"output"}
        assert np.allclose(
            reused_svd["output"].get_matrix(),
            reopened.get_artifact(ids["reduced"]).get_matrix(),
        )

        fitted_lda = reopened.get_operator(str(lda_operator_id))
        reused_lda = reopened.translate(
            fitted_lda,
            reopened.get_artifact("art_counts_round14"),
            batch_size=3,
            workers=WORKERS,
        )
        assert set(reused_lda) == {"output"}
        assert np.allclose(
            reused_lda["output"].get_matrix(),
            reopened.get_artifact(ids["doc_topics"]).get_matrix(),
        )
    finally:
        reopened.close()


def test_real_umap_round_trip_if_installed(tmp_path: Path) -> None:
    pytest.importorskip("umap")
    project_path = tmp_path / "round14_umap"
    project = teal.Project.create(project_path, name="round14_umap")
    try:
        counts = _register_counts(project)
        op = UMAP(
            n_components=2,
            n_neighbors=3,
            min_dist=0.1,
            metric="cosine",
            random_state=7,
            reuse="stored",
        )
        output = project.translate(op, counts)["output"]
        values = output.get_matrix()
        assert values.shape == (len(VALUES), 2)
        assert np.isfinite(values).all()
        operator_id = op.operator_id
        artifact_id = output.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        fitted = reopened.get_operator(str(operator_id))
        reused = reopened.translate(
            fitted, reopened.get_artifact("art_counts_round14"), batch_size=3
        )["output"]
        assert reused.get_matrix().shape == (len(VALUES), 2)
        assert np.isfinite(reused.get_matrix()).all()
        assert reopened.get_artifact(artifact_id).status == "complete"
    finally:
        reopened.close()


def test_real_umap_recompute_round_trip_if_installed(tmp_path: Path) -> None:
    pytest.importorskip("umap")
    project_path = tmp_path / "round27_umap_recompute"
    project = teal.Project.create(project_path, name="round27_umap_recompute")
    try:
        counts = _register_counts(project)
        op = UMAP(
            n_components=2,
            n_neighbors=3,
            min_dist=0.1,
            metric="cosine",
            random_state=7,
            reuse="recompute",
        )
        project.translate(op, counts)["output"]
        operator_id = str(op.operator_id)
        operator_dir = project.storage.operator_dir(operator_id)
        assert list((operator_dir / "assets").iterdir()) == []
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        recomputable = reopened.get_operator(operator_id)
        assert recomputable.reuse == "recompute"
        assert not recomputable.is_fitted
        reused = reopened.translate(
            recomputable, reopened.get_artifact("art_counts_round14"), batch_size=3
        )["output"]
        values = reused.get_matrix()
        assert values.shape == (len(VALUES), 2)
        assert np.isfinite(values).all()
        # Refit state is transient; reopening the frozen Operator still begins
        # from the recipe-only form.
        again = reopened.get_operator(operator_id)
        assert not again.is_fitted
    finally:
        reopened.close()


def test_real_feature_trimmer_round_trip_and_reuse(tmp_path: Path) -> None:
    from text_analysis_lab.translators import FeatureTrimmer

    project_path = tmp_path / "round25_4_trim"
    project = teal.Project.create(project_path, name="round25_4_trim")
    try:
        counts = _register_counts(project)
        op = FeatureTrimmer(min_df=5)
        trimmed = project.translate(op, counts)["output"]
        assert trimmed.primary_key == ["doc_id"]
        assert trimmed.get_data_columns() == ["beta", "delta"]
        assert trimmed.get_matrix().shape == (len(VALUES), 2)
        operator_id = str(op.operator_id)
        trimmed_id = trimmed.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        fitted = reopened.get_operator(operator_id)
        reused = reopened.translate(
            fitted,
            reopened.get_artifact("art_counts_round14"),
            batch_size=3,
            workers=WORKERS,
        )["output"]
        assert (
            reused.get_data_columns()
            == reopened.get_artifact(trimmed_id).get_data_columns()
        )
        assert np.array_equal(
            reused.get_matrix().toarray(),
            reopened.get_artifact(trimmed_id).get_matrix().toarray(),
        )
    finally:
        reopened.close()
