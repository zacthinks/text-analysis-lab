"""CountVectorizer two-pass fitting: sklearn parity, naming and recovery."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from text_analysis_lab import Project
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.operator import TranslationRequest
from text_analysis_lab.core.translate import _ExecutionPlan
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import CountVectorizer
from text_analysis_lab.translators.count_vectorizer import _select_global_vocabulary


TEXTS = [
    "Alpha beta beta",
    "Alpha gamma",
    "beta gamma delta",
    "delta epsilon",
    "and the and",
    None,
    "alpha gamma alpha",
    "beta delta alpha",
]


def _source(project, texts=TEXTS, *, name="art_count_source"):
    frame = pd.DataFrame({
        "document": list(range(len(texts))),
        "text": list(texts),
    })
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(name),
        artifact_id=name, label="documents", lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": frame[["document"]], "data": frame[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=name, artifact_type="table", label="documents",
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(name)


def _keyed_matrix(artifact):
    keys = artifact.query(key_columns=True, data_columns=False, form="table")
    matrix = artifact.get_matrix()
    return {
        tuple(key): tuple(matrix.getrow(i).toarray().ravel())
        for i, key in enumerate(keys.to_numpy())
    }


@pytest.mark.parametrize("config", [
    dict(ngram_range=(1, 2), max_features=4),
    dict(ngram_range=(1, 2), min_df=2, max_df=0.7),
    dict(ngram_range=(1, 2), min_df=0.25, binary=True, max_features=3),
    dict(ngram_range=(1, 2), stop_words="english", max_features=5),
    dict(ngram_range=(1, 2), stemmer="porter", max_features=5),
    dict(analyzer="char", ngram_range=(2, 3), max_features=8),
    dict(analyzer="char_wb", ngram_range=(2, 3), binary=True, max_features=8),
])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_two_pass_project_matches_sklearn_whole_artifact(
    tmp_path, config, batch_size
):
    project = Project.create(tmp_path / "project", name="count_parity")
    try:
        source = _source(project)
        whole = CountVectorizer(**config)
        expected = project.translate(whole, source)["output"]

        streamed = CountVectorizer(**config)
        actual = project.translate(
            streamed, source, batch_size=batch_size
        )["output"]

        assert streamed.vocabulary_ == whole.vocabulary_
        assert actual.get_data_columns() == expected.get_data_columns()
        assert actual.primary_key == expected.primary_key
        assert _keyed_matrix(actual) == _keyed_matrix(expected)
        op_id = str(
            project.catalog.operations_using_operator(streamed.operator_id)[0]["operation_id"]
        )
        metadata = json.loads(
            (project.storage.operation_dir(op_id) / "operation.json").read_text()
        )
        assert metadata["request"]["pass_count"] == 2
        restored = project.get_operator(streamed.operator_id)
        assert restored.vocabulary_ == whole.vocabulary_
        assert restored.ngram_separator == "_"
    finally:
        project.close()


def test_corpus_document_count_includes_batches_without_features(tmp_path):
    project = Project.create(tmp_path / "empty_batch", name="empty_batch")
    try:
        # The first batch is entirely empty; filtering uses all 5 documents.
        source = _source(project, [None, "", "alpha beta", "beta", "alpha"])
        config = dict(min_df=0.4, max_df=0.6)
        whole = CountVectorizer(**config)
        expected = project.translate(whole, source)["output"]
        streamed = CountVectorizer(**config)
        actual = project.translate(streamed, source, batch_size=2)["output"]
        assert streamed.vocabulary_ == whole.vocabulary_
        assert _keyed_matrix(actual) == _keyed_matrix(expected)
    finally:
        project.close()


def test_max_features_ties_match_sklearn_indices():
    terms = {"alpha": 2, "beta": 2, "gamma": 2, "delta": 2}
    from sklearn.feature_extraction.text import CountVectorizer as SKCount
    texts = ["alpha beta", "gamma delta"]
    sk = SKCount(max_features=2)
    sk.fit(texts)
    actual = _select_global_vocabulary(
        term_frequency=terms,
        document_frequency={term: 1 for term in terms},
        n_documents=2, min_df=1, max_df=1.0, max_features=2,
    )
    assert actual == sk.vocabulary_


def test_display_separator_is_configurable_and_legacy_state_is_compatible():
    fitted = CountVectorizer(
        ngram_range=(1, 2), vocabulary={"new": 0, "new york": 1, "york": 2}
    )
    assert fitted._feature_names() == ["new", "new_york", "york"]
    custom = CountVectorizer.from_json_state(
        fitted.to_json_state(include_vocabulary=True) | {"ngram_separator": "__"}
    )
    assert custom._feature_names() == ["new", "new__york", "york"]
    old = fitted.to_json_state(include_vocabulary=True)
    del old["ngram_separator"]
    restored = CountVectorizer.from_json_state(old)
    assert restored.ngram_separator == " "
    assert restored._feature_names() == ["new", "new york", "york"]


def test_ambiguous_display_name_rejected_without_mutating_sklearn_vocabulary():
    fitted = CountVectorizer(
        ngram_range=(1, 2),
        vocabulary={"new york": 0, "new_york": 1},
    )
    with pytest.raises(ArtifactError, match="ambiguous"):
        fitted._feature_names()
    assert fitted.vocabulary_ == {"new york": 0, "new_york": 1}


@pytest.mark.parametrize("failed_pass", [0, 1])
def test_interrupted_fit_or_write_recovers_identical_vocabulary_and_rows(
    tmp_path, monkeypatch, failed_pass
):
    path = tmp_path / "recovered"
    project = Project.create(path, name="count_resume")
    original_mark = _ExecutionPlan.mark_complete
    fired = {"value": False}
    try:
        source = _source(project)
        config = dict(ngram_range=(1, 2), min_df=2, max_features=4, binary=True)
        whole = CountVectorizer(**config)
        reference = project.translate(whole, source)["output"]
        wanted_vocabulary = dict(whole.vocabulary_)
        wanted_matrix = _keyed_matrix(reference)
        wanted_columns = reference.get_data_columns()

        def fail_after_checkpoint(plan, unit_index):
            if unit_index == 0 and not fired["value"]:
                cursor = plan.con.execute(
                    "SELECT current_pass FROM plan_pass_state WHERE id=1"
                ).fetchone()
                if cursor is not None and int(cursor[0]) == failed_pass:
                    fired["value"] = True
                    raise RuntimeError("injected count-vectorizer failure")
            return original_mark(plan, unit_index)

        monkeypatch.setattr(_ExecutionPlan, "mark_complete", fail_after_checkpoint)
        translator = CountVectorizer(**config)
        with pytest.raises(RuntimeError, match="injected count-vectorizer failure"):
            project.translate(translator, source, batch_size=2)
        assert fired["value"]
        operation_id = str(
            project.catalog.operations_using_operator(translator.operator_id)[0]["operation_id"]
        )
        assert project.get_operation(operation_id)["status"] == "failed"
    finally:
        project.close()
        monkeypatch.undo()

    reopened = Project.open(path)
    try:
        output = reopened.resume_operation(operation_id)["output"]
        assert _keyed_matrix(output) == wanted_matrix
        assert output.get_data_columns() == wanted_columns
        restored = reopened.get_operator(translator.operator_id)
        assert restored.vocabulary_ == wanted_vocabulary
        assert reopened.get_operation(operation_id)["status"] == "complete"
    finally:
        reopened.close()


def test_fitted_vocab_remains_one_pass_and_reopens(tmp_path):
    path = tmp_path / "reuse"
    project = Project.create(path, name="frozen_count")
    try:
        source = _source(project)
        whole = CountVectorizer(ngram_range=(1, 2))
        expected = project.translate(whole, source)["output"]
        frozen = project.get_operator(whole.operator_id)
        output = project.translate(frozen, source, batch_size=2)["output"]
        assert _keyed_matrix(output) == _keyed_matrix(expected)
        assert output.get_data_columns() == expected.get_data_columns()
        op_id = str(
            project.catalog.operations_using_operator(frozen.operator_id)[-1]["operation_id"]
        )
        metadata = json.loads(
            (project.storage.operation_dir(op_id) / "operation.json").read_text()
        )
        assert metadata["request"]["pass_count"] == 1
        output_id = output.artifact_id
    finally:
        project.close()
    reopened = Project.open(path)
    try:
        assert reopened.get_artifact(output_id).get_data_columns() == expected.get_data_columns()
    finally:
        reopened.close()
