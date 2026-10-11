"""Release smoke coverage: persisted DTM, LDA topics and SVD components.

Unlike the focused top_features unit fixtures, these are real TeAL Project
artifacts written, transformed, closed and reopened from disk.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import LDA, SVD


COUNTS = sparse.csr_matrix(np.asarray([
    [4, 0, 2, 0, 1],
    [1, 3, 0, 1, 0],
    [2, 0, 4, 1, 0],
    [0, 3, 1, 2, 1],
    [3, 1, 2, 0, 2],
    [1, 2, 1, 3, 0],
], dtype=float))
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon"]
KEYS = [91, 17, 63, 5, 44, 2]


def _registered_dtm(project):
    artifact_id = "dtm_source"
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="dtm_source", lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({
        "keys": pd.DataFrame({"doc_id": KEYS}),
        "data": {"values": COUNTS, "columns": WORDS},
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id, artifact_type="sparse_matrix",
        label="dtm_source", lineage_mode="new_key",
        status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_release_top_features_real_project_dtm_lda_svd_roundtrip(tmp_path):
    path = tmp_path / "top_features_release"
    project = teal.Project.create(path, name="top_features_release")
    try:
        dtm = _registered_dtm(project)
        lda = project.translate(
            LDA(n_components=2, max_iter=4, random_state=7), dtm
        )
        svd = project.translate(
            SVD(n_components=2, random_state=7), dtm
        )
        artifact_ids = {
            "dtm": dtm.artifact_id,
            "lda_topics": lda["topics"].artifact_id,
            "svd_components": svd["components"].artifact_id,
        }
        before = {}
        before["dtm"] = dtm.analysis.top_features(
            n=3, methods=["value", "term_frequency", "document_frequency"],
            batch_size=2,
        )
        before["lda_topics"] = lda["topics"].analysis.top_features(
            n=3, methods=["value", "probability", "lift", "frex"],
            reference=dtm, batch_size=2,
        )
        before["svd_components"] = svd["components"].analysis.top_features(
            n=3, methods=["value", "absolute"], batch_size=2,
        )
        for name, artifact in [
            ("lda_topics", lda["topics"]),
            ("svd_components", svd["components"]),
        ]:
            assert artifact.get_data_columns() == WORDS
            assert artifact.get_feature_metadata()["column"].tolist() == WORDS
            assert before[name]["rank"].min() == 1
            assert len(before[name]) == 6
        assert before["dtm"]["doc_id"].drop_duplicates().tolist() == KEYS
        assert len(before["dtm"]) == len(KEYS) * 3
    finally:
        project.close()

    reopened = teal.Project.open(path)
    try:
        dtm = reopened.get_artifact(artifact_ids["dtm"])
        topics = reopened.get_artifact(artifact_ids["lda_topics"])
        components = reopened.get_artifact(artifact_ids["svd_components"])
        after = {
            "dtm": dtm.analysis.top_features(
                n=3, methods=["value", "term_frequency", "document_frequency"],
                batch_size=3,
            ),
            "lda_topics": topics.analysis.top_features(
                n=3, methods=["value", "probability", "lift", "frex"],
                reference=dtm, batch_size=1,
            ),
            "svd_components": components.analysis.top_features(
                n=3, methods=["value", "absolute"], batch_size=1,
            ),
        }
        for name in before:
            pd.testing.assert_frame_equal(
                before[name].reset_index(drop=True),
                after[name].reset_index(drop=True),
                check_dtype=False,
            )
        assert topics.get_feature_metadata()["column"].tolist() == WORDS
        assert components.get_feature_metadata()["column"].tolist() == WORDS
    finally:
        reopened.close()


def test_release_top_features_reference_feature_mismatch_in_project(tmp_path):
    path = tmp_path / "top_features_reference_mismatch"
    project = teal.Project.create(path, name="top_features_reference_mismatch")
    try:
        dtm = _registered_dtm(project)
        lda_topics = project.translate(
            LDA(n_components=2, max_iter=4, random_state=7), dtm
        )["topics"]
        result = lda_topics.analysis.top_features(
            n=2, methods=["lift"], reference=dtm
        )
        assert result["lift_score"].notna().all()
        assert np.isfinite(result["lift_score"]).all()
    finally:
        project.close()
