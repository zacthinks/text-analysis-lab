"""Opt-in persisted TeAL <-> real GeCo Topic Model round-trip.

Run explicitly with TEAL_RUN_REAL_GECO=1; excluded from ordinary CI.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TEAL_RUN_REAL_GECO") != "1",
    reason="Real GeCo round-trip is opt-in (TEAL_RUN_REAL_GECO=1).",
)


def test_real_geco_topic_registration_reopen_and_partial_coverage(tmp_path):
    import geometric_coder
    import text_analysis_lab as teal
    from scipy import sparse
    from text_analysis_lab.core.writer import create_artifact_writer
    from text_analysis_lab.translators import ArgMax, CountVectorizer, LDA

    root = tmp_path / "teal_project"
    project = teal.Project.create(root, name="real_geco_topics")
    rows = pd.DataFrame({
        "row_id": [0, 1, 2, 3, 4, 5],
        "text": [
            "apple banana apple", "banana apple orange",
            "car train bus", "train car bus",
            "apple orange banana", "train bus car",
        ],
    })
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_docs"),
        artifact_id="art_docs", label="documents", lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["row_id"]], "data": rows[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_docs", artifact_type="table", label="documents",
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    try:
        documents = project.get_artifact("art_docs")
        counts = project.translate(
            CountVectorizer(text_field="text", min_df=1), documents
        )["output"]
        fitted = project.translate(
            LDA(n_components=2, max_iter=4, random_state=7), counts
        )
        memberships, descriptions = fitted["output"], fitted["topics"]
        dominant = project.translate(ArgMax(), memberships)["output"]
        linked = project.geco.create(
            "linked", documents=documents, text_field="text",
            geometry=counts,
            topic_models={"LDA": {
                "membership": memberships, "dominant": dominant,
                "description": descriptions,
                "topic_labels": {"topic_0": "Apple"},
            }},
        )
        assert linked.coder.topic_models()[0]["membership_artifact_id"] == memberships.artifact_id
        result = linked.coder.topic_model_rows("LDA", include_description=True)
        assert result.matched_positions == tuple(range(6))
        assert result.topic_ids == ("topic_0", "topic_1")
        assert result.memberships.shape == (6, 2)
        assert result.descriptions is not None
        assert result.descriptions.values.shape[0] == 2
        np.testing.assert_allclose(
            result.memberships.toarray() if sparse.issparse(result.memberships) else result.memberships,
            memberships.get_matrix(),
        )
        partial = linked.coder.topic_model_rows("LDA", positions=[5, 1, 3])
        assert partial.matched_positions == (0, 1, 2)
        assert partial.memberships.shape == (3, 2)
        linked.close()
    finally:
        project.close()

    reopened = teal.Project.open(root)
    try:
        linked = reopened.geco.open("linked")
        assert linked.coder.topic_model("LDA")["topic_labels"] == {"topic_0": "Apple"}
        result = linked.coder.topic_model_rows("LDA", positions=[0, 2])
        assert result.matched_positions == (0, 1)
        assert result.memberships.shape == (2, 2)
        linked.close()
    finally:
        reopened.close()
