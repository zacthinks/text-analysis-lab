from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.translators import ArtifactCountVectorizer, SpacyTranslator


def _external_modules():
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    return pytest.importorskip("spacy")


def _seed_documents(project: teal.Project, rows: pd.DataFrame):
    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_documents"),
        artifact_id="art_documents",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": rows[["document_id"]].reset_index(drop=True),
            "data": rows[["text"]].reset_index(drop=True),
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_documents",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_documents")


def _saved_sentencizer_model(tmp_path: Path) -> Path:
    spacy = _external_modules()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model_dir = tmp_path / "spacy_test_model"
    nlp.to_disk(model_dir)
    return model_dir


def test_spacy_tokens_subset_to_artifact_count_dtm_with_sentence_bounded_ngrams(
    tmp_path: Path,
) -> None:
    _external_modules()
    model_dir = _saved_sentencizer_model(tmp_path)
    rows = pd.DataFrame(
        {
            "document_id": [1, 2],
            "text": [
                "alpha the beta. gamma delta.",
                "beta beta. delta alpha.",
            ],
        }
    )

    subset_file = tmp_path / "keep_content.py"
    subset_file.write_text(
        "def keep_content(frame):\n"
        "    return frame['is_alpha'] & ~frame['is_stop']\n",
        encoding="utf-8",
    )

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="artifact_count_external")
    try:
        source = _seed_documents(project, rows)
        tokens = project.translate(
            SpacyTranslator(model=str(model_dir), spacy_batch_size=2),
            source,
            workers=1,
            batch_size=2,
        )["tokens"]

        filtered = project.subset(
            tokens,
            (subset_file, "keep_content"),
            key_columns=True,
            data_columns=["is_alpha", "is_stop"],
            form="table",
            iter_batches=True,
            batch_size=3,
            workers=1,
        )["output"]
        assert filtered.primary_key == ["document_id", "sentence_id", "token_id"]
        assert filtered.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert filtered.n_rows < tokens.n_rows

        vectorizer = ArtifactCountVectorizer(
            field="text",
            group_by=("document_id",),
            ngram_range=(1, 2),
        )
        dtm = project.translate(vectorizer, filtered)["output"]
        assert dtm.artifact_type.value == "sparse_matrix"
        assert dtm.primary_key == ["document_id"]
        assert dtm.descriptor["lineage"]["lineage_mode"] == "reduced_key"
        assert dtm.descriptor["lineage"]["basis_artifact_ids"] == [filtered.artifact_id]
        assert dtm.n_rows == 2

        columns = dtm.get_data_columns()
        matrix = dtm.get_matrix()
        assert sparse.issparse(matrix)
        assert "alpha beta" in columns
        assert "gamma delta" in columns
        assert "beta gamma" not in columns  # sentence boundary
        assert "beta beta" in columns
        expected = {
            1: {
                "alpha": 1,
                "beta": 1,
                "gamma": 1,
                "delta": 1,
                "alpha beta": 1,
                "gamma delta": 1,
            },
            2: {
                "alpha": 1,
                "beta": 2,
                "delta": 1,
                "beta beta": 1,
                "delta alpha": 1,
            },
        }
        dense = matrix.toarray()
        for row_index, document_id in enumerate([1, 2]):
            observed = {
                feature: int(dense[row_index, column_index])
                for column_index, feature in enumerate(columns)
                if int(dense[row_index, column_index]) != 0
            }
            assert observed == expected[document_id]

        # Frozen vocabulary/state reload should produce exactly the same feature
        # order and counts on the same compatible token artifact.
        frozen = project.get_operator(vectorizer.operator_id)
        assert isinstance(frozen, ArtifactCountVectorizer)
        assert frozen.is_fitted
        dtm_reused = project.translate(frozen, filtered)["output"]
        assert dtm_reused.get_data_columns() == columns
        np.testing.assert_array_equal(dtm_reused.get_matrix().toarray(), dense)

        dtm_id = dtm.artifact_id
        operator_id = vectorizer.operator_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        reopened_dtm = reopened.get_artifact(dtm_id)
        assert reopened_dtm.primary_key == ["document_id"]
        assert reopened_dtm.get_data_columns() == columns
        reopened_operator = reopened.get_operator(operator_id)
        assert isinstance(reopened_operator, ArtifactCountVectorizer)
        assert reopened_operator.is_fitted
        assert reopened_operator.vocabulary_ == vectorizer.vocabulary_
    finally:
        reopened.close()


def test_artifact_count_vectorizer_fit_translate_resume_freezes_vocabulary(
    tmp_path: Path,
) -> None:
    _external_modules()
    from types import MethodType

    model_dir = _saved_sentencizer_model(tmp_path)
    rows = pd.DataFrame(
        {
            "document_id": [1, 2, 3],
            "text": [
                "alpha beta. gamma.",
                "beta delta.",
                "alpha delta epsilon.",
            ],
        }
    )
    project_path = tmp_path / "resume_project"
    project = teal.Project.create(project_path, name="artifact_count_resume")
    try:
        source = _seed_documents(project, rows)
        tokens = project.translate(
            SpacyTranslator(model=str(model_dir)),
            source,
            workers=1,
            batch_size=2,
        )["tokens"]

        vectorizer = ArtifactCountVectorizer(
            field="text",
            group_by=("document_id",),
            ngram_range=(1, 2),
        )
        original_translate_batch = vectorizer.translate_batch
        failed = {"done": False}

        def fail_after_fit_once(self, inputs, *, mode, request):
            result = original_translate_batch(inputs, mode=mode, request=request)
            if not failed["done"]:
                failed["done"] = True
                raise RuntimeError("intentional ArtifactCountVectorizer interruption")
            return result

        vectorizer.translate_batch = MethodType(fail_after_fit_once, vectorizer)
        with pytest.raises(RuntimeError, match="intentional ArtifactCountVectorizer"):
            project.translate(vectorizer, tokens)

        operations = project.catalog.operations_using_operator(vectorizer.operator_id)
        assert len(operations) == 1
        operation_id = str(operations[0]["operation_id"])
        assert operations[0]["status"] == "failed"

        project.close()
        project = teal.Project.open(project_path)
        outputs = project.resume_operation(operation_id)
        dtm = outputs["output"]
        assert dtm.status == "complete"
        assert dtm.primary_key == ["document_id"]
        assert dtm.n_rows == 3
        assert sparse.issparse(dtm.get_matrix())
        assert "alpha beta" in dtm.get_data_columns()

        frozen = project.get_operator(vectorizer.operator_id)
        assert isinstance(frozen, ArtifactCountVectorizer)
        assert frozen.is_fitted
        assert frozen.vocabulary_
    finally:
        project.close()
