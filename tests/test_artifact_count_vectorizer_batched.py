"""End-to-end opt-in complete-group streaming for ArtifactCountVectorizer."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.translators import ArtifactCountVectorizer
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.core.errors import OperatorError


def _source(project, *, all_empty=False):
    rows = pd.DataFrame({
        "corpus": [1, 2, 1, 2, 1, 1, 2, 1, 1],
        "document": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        "sentence": [1, 1, 1, 1, 1, 1, 1, 2, 2],
        "token": [0, 0, 1, 1, 2, 0, 0, 0, 1],
        "lemma": ["alpha", "beta", "beta", "beta", "alpha",
                  None, "beta", "gamma", "beta"],
    })
    if all_empty:
        rows["lemma"] = None
    writer = create_artifact_writer(
        artifact_type="table", artifact_dir=project.storage.artifact_dir("art_tokens"),
        artifact_id="art_tokens", label="tokens", lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({
        "keys": rows[["corpus", "document", "sentence", "token"]],
        "data": rows[["lemma"]],
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_tokens", artifact_type="table", label="tokens",
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact("art_tokens")


@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2)])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_streaming_matches_whole_artifact_science(tmp_path, ngram_range, batch_size):
    project = Project.create(tmp_path / "project", name="compare")
    try:
        source = _source(project)
        kwargs = dict(
            field="lemma", group_by=("corpus", "document"),
            ngram_range=ngram_range, max_features=4, min_df=1,
        )
        reference = project.translate(ArtifactCountVectorizer(**kwargs), source)["output"]
        streaming = ArtifactCountVectorizer(**kwargs)
        output = project.translate(
            streaming, source, batch_by="document", batch_size=batch_size
        )["output"]
        assert output.get_data_columns() == reference.get_data_columns()
        assert (output.get_matrix() != reference.get_matrix()).nnz == 0
        assert output.primary_key == reference.primary_key
        assert output.query(key_columns=True, data_columns=False, form="table").equals(
            reference.query(key_columns=True, data_columns=False, form="table")
        )
        assert streaming.is_fitted
        op_id = str(project.list_operations()[-1]["operation_id"])
        assert project.get_operator(streaming.operator_id).vocabulary_ == streaming.vocabulary_
        import json
        meta = json.loads((project.storage.operation_dir(op_id) / "operation.json").read_text())
        assert meta["request"]["pass_count"] == 2
    finally:
        project.close()


def test_explicit_fitted_vocabulary_can_stream_without_replay(tmp_path):
    project = Project.create(tmp_path / "project", name="reuse")
    try:
        source = _source(project)
        reference = ArtifactCountVectorizer(field="lemma", group_by=("corpus", "document"))
        project.translate(reference, source)
        reused = ArtifactCountVectorizer(
            field="lemma", group_by=("corpus", "document"),
            vocabulary=reference.vocabulary_,
        )
        output = project.translate(reused, source, batch_by="document", batch_size=1)["output"]
        assert output.n_rows == 4
        import json
        op_id = str(project.list_operations()[-1]["operation_id"])
        meta = json.loads((project.storage.operation_dir(op_id) / "operation.json").read_text())
        assert meta["request"]["pass_count"] == 1
    finally:
        project.close()


def test_grouping_finer_than_output_grain_is_rejected(tmp_path):
    project = Project.create(tmp_path / "project", name="unsafe")
    try:
        source = _source(project)
        with pytest.raises(OperatorError, match="finer"):
            project.translate(
                ArtifactCountVectorizer(field="lemma", group_by=("corpus", "document")),
                source, batch_by="token", batch_size=1,
            )
    finally:
        project.close()
