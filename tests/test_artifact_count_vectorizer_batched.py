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
        assert output.primary_key == reference.primary_key
        # Compare keyed rows, not physical positions: complete-group batching
        # may legitimately change row order for interleaved groups.
        def keyed_counts(artifact):
            keys = artifact.query(
                key_columns=True, data_columns=False, form="table"
            )
            return {
                tuple(row): tuple(artifact.get_matrix().getrow(i).toarray().ravel())
                for i, row in enumerate(keys.to_numpy())
            }
        assert keyed_counts(output) == keyed_counts(reference)
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


def test_cross_batch_ngram_collision_is_rejected_for_fitted_reuse(tmp_path):
    """An atomic new_york must not alias the two-token new_york bigram."""
    project = Project.create(tmp_path / "collision", name="collision")
    try:
        rows = pd.DataFrame({
            "doc": [1, 2, 2],
            "sent": [0, 0, 0],
            "token": [0, 0, 1],
            "lemma": ["new_york", "new", "york"],
        })
        writer = create_artifact_writer(
            artifact_type="table", artifact_dir=project.storage.artifact_dir("art_collision"),
            artifact_id="art_collision", label="tokens", lineage_mode="new_key",
            basis_artifact_ids=(),
        )
        writer.write({
            "keys": rows[["doc", "sent", "token"]],
            "data": rows[["lemma"]],
        })
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id="art_collision", artifact_type="table", label="tokens",
            lineage_mode="new_key", status="complete", basis_artifact_ids=(),
        )
        source = project.get_artifact("art_collision")
        fitted = ArtifactCountVectorizer(
            field="lemma", group_by="doc", ngram_range=(1, 2),
            vocabulary={"new_york": 0, "new": 1, "york": 2},
        )
        with pytest.raises(Exception, match="ambiguous"):
            project.translate(fitted, source, batch_by="doc", batch_size=1)
    finally:
        project.close()


def test_custom_separator_remains_configurable():
    translator = ArtifactCountVectorizer(
        field="lemma", group_by="doc", ngram_range=(1, 2),
        ngram_separator="__",
    )
    assert translator.ngram_separator == "__"
    restored = ArtifactCountVectorizer.from_json_state(translator.to_json_state())
    assert restored.ngram_separator == "__"


def _keyed_matrix(artifact):
    keys = artifact.query(key_columns=True, data_columns=False, form="table")
    matrix = artifact.get_matrix()
    return {
        tuple(row): tuple(matrix.getrow(i).toarray().ravel())
        for i, row in enumerate(keys.to_numpy())
    }


def _assert_equivalent(actual, expected):
    assert actual.get_data_columns() == expected.get_data_columns()
    assert actual.primary_key == expected.primary_key
    assert _keyed_matrix(actual) == _keyed_matrix(expected)


def test_coarser_corpus_group_batches_preserve_document_science(tmp_path):
    """Interleaved documents may reorder physically; compare by their keys."""
    project = Project.create(tmp_path / "coarse", name="coarse_groups")
    try:
        source = _source(project)
        config = dict(
            field="lemma", group_by=("corpus", "document"),
            ngram_range=(1, 2), min_df=0.25, max_df=1.0,
            max_features=4,
        )
        reference_vectorizer = ArtifactCountVectorizer(**config)
        reference = project.translate(reference_vectorizer, source)["output"]
        for size in (1, 2):
            translator = ArtifactCountVectorizer(**config)
            result = project.translate(
                translator, source, batch_by="corpus", batch_size=size
            )["output"]
            _assert_equivalent(result, reference)
            assert translator.vocabulary_ == reference_vectorizer.vocabulary_
    finally:
        project.close()


@pytest.mark.parametrize("failure_pass", [0, 1])
def test_resume_streaming_fitting_and_matrix_writing_matches_reference(
    tmp_path, monkeypatch, failure_pass
):
    """A checkpointed fitted contribution or output must never be repeated."""
    from text_analysis_lab.core.translate import _ExecutionPlan

    path = tmp_path / f"resumed_{failure_pass}"
    project = Project.create(path, name=f"resume_{failure_pass}")
    original_mark = _ExecutionPlan.mark_complete
    triggered = {"done": False}

    try:
        source = _source(project)
        config = dict(
            field="lemma", group_by=("corpus", "document"),
            ngram_range=(1, 2), min_df=0.25,
            max_features=4,
        )
        reference_vectorizer = ArtifactCountVectorizer(**config)
        reference = project.translate(reference_vectorizer, source)["output"]
        reference_vocabulary = dict(reference_vectorizer.vocabulary_)
        reference_columns = reference.get_data_columns()
        reference_keys_and_counts = _keyed_matrix(reference)

        def interrupt_after_checkpoint(plan, unit_index):
            if not triggered["done"] and unit_index == 0:
                cursor = plan.con.execute(
                    "SELECT current_pass FROM plan_pass_state WHERE id=1"
                ).fetchone()
                if cursor is not None and cursor[0] == failure_pass:
                    triggered["done"] = True
                    raise RuntimeError(f"interrupted pass {failure_pass}")
            return original_mark(plan, unit_index)

        monkeypatch.setattr(_ExecutionPlan, "mark_complete", interrupt_after_checkpoint)
        translator = ArtifactCountVectorizer(**config)
        with pytest.raises(RuntimeError, match=f"interrupted pass {failure_pass}"):
            project.translate(
                translator, source, batch_by="document", batch_size=1
            )
        assert triggered["done"]
        operation_id = str(
            project.catalog.operations_using_operator(translator.operator_id)[0]["operation_id"]
        )
        assert project.get_operation(operation_id)["status"] == "failed"
    finally:
        project.close()
        monkeypatch.undo()

    reopened = Project.open(path)
    try:
        resumed = reopened.resume_operation(operation_id)["output"]
        assert resumed.get_data_columns() == reference_columns
        assert _keyed_matrix(resumed) == reference_keys_and_counts
        frozen = reopened.get_operator(translator.operator_id)
        assert frozen.vocabulary_ == reference_vocabulary
        assert reopened.get_operation(operation_id)["status"] == "complete"
    finally:
        reopened.close()


def test_feature_identity_tracking_is_operation_local(tmp_path):
    """Reuse of one Python object agrees with freshly reloaded frozen state."""
    project = Project.create(tmp_path / "lifecycle", name="lifecycle")
    try:
        source = _source(project)
        vectorizer = ArtifactCountVectorizer(
            field="lemma", group_by=("corpus", "document"),
            ngram_range=(1, 2), vocabulary={"alpha_beta": 0, "beta": 1},
        )
        first = project.translate(
            vectorizer, source, batch_by="document", batch_size=1
        )["output"]
        assert first.n_rows == 4
        assert vectorizer._fit_labels == {}

        # This corpus presents a different internal feature identity for the
        # same displayed column; no cross-operation history should leak.
        rows = pd.DataFrame({
            "corpus": [3], "document": [1], "sentence": [0],
            "token": [0], "lemma": ["alpha_beta"],
        })
        writer = create_artifact_writer(
            artifact_type="table",
            artifact_dir=project.storage.artifact_dir("art_second_corpus"),
            artifact_id="art_second_corpus", label="second",
            lineage_mode="new_key", basis_artifact_ids=(),
        )
        writer.write({
            "keys": rows[["corpus", "document", "sentence", "token"]],
            "data": rows[["lemma"]],
        })
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id="art_second_corpus", artifact_type="table",
            label="second", lineage_mode="new_key", status="complete",
            basis_artifact_ids=(),
        )
        second = project.get_artifact("art_second_corpus")
        reused = project.translate(
            vectorizer, second, batch_by="document", batch_size=1
        )["output"]
        assert reused.n_rows == 1
        assert reused.get_matrix()[0, 0] == 1
        assert vectorizer._fit_labels == {}
    finally:
        project.close()
