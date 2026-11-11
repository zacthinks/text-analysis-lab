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
        "def keep_content(frame):\n    return frame['is_alpha'] & ~frame['is_stop']\n",
        encoding="utf-8",
    )

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="artifact_count_external")
    try:
        source = _seed_documents(project, rows)
        tokens = project.translate(
            SpacyTranslator(model=str(model_dir), spacy_batch_size=2, save_model=True),
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
        assert "alpha_beta" in columns
        assert "gamma_delta" in columns
        assert "beta_gamma" not in columns  # sentence boundary
        assert "beta_beta" in columns
        expected = {
            1: {
                "alpha": 1,
                "beta": 1,
                "gamma": 1,
                "delta": 1,
                "alpha_beta": 1,
                "gamma_delta": 1,
            },
            2: {
                "alpha": 1,
                "beta": 2,
                "delta": 1,
                "beta_beta": 1,
                "delta_alpha": 1,
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
            SpacyTranslator(model=str(model_dir), save_model=True),
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
        assert "alpha_beta" in dtm.get_data_columns()

        frozen = project.get_operator(vectorizer.operator_id)
        assert isinstance(frozen, ArtifactCountVectorizer)
        assert frozen.is_fitted
        assert frozen.vocabulary_
    finally:
        project.close()


def test_frozen_spacy_dtm_with_metadata_filters_and_native_replay(
    tmp_path: Path,
) -> None:
    """Token-only DTM with filtered zero groups; no target artifact."""
    _external_modules()
    model_dir = _saved_sentencizer_model(tmp_path)
    source_rows = pd.DataFrame({
        "document_id": [1, 2, 3],
        "text": ["alpha the beta", "and the", "beta beta"],
    })
    path = tmp_path / "token_only"
    project = teal.Project.create(path, name="token_only_count")
    try:
        source = _seed_documents(project, source_rows)
        tokens = project.translate(
            SpacyTranslator(model=str(model_dir), save_model=True),
            source, workers=1, batch_size=2,
        )["tokens"]
        vectorizer = ArtifactCountVectorizer(
            field="text", group_by=("document_id",),
            metadata_filters={"is_alpha": True, "is_stop": False},
            min_df=1,
        )
        matrix = project.translate(
            vectorizer, tokens, workers=1,
            batch_by="document_id", batch_size=2,
        )["output"]
        assert matrix.primary_key == ["document_id"]
        assert matrix.n_rows == 3
        assert matrix.descriptor["lineage"]["lineage_mode"] == "reduced_key"
        assert matrix.get_data_columns() == ["alpha", "beta"]
        assert matrix.get_matrix().toarray().tolist() == [
            [1, 1], [0, 0], [0, 2],
        ]
        pipeline = project.pipeline(start=source, end=matrix)
        values = pipeline.translate(["alpha the", "beta beta"])["output"]["values"]
        assert values.toarray().tolist() == [[1, 0], [0, 2]]
        artifact_id = matrix.artifact_id
        source_id = source.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(path)
    try:
        recovered = reopened.pipeline(start=source_id, end=artifact_id)
        values = recovered.translate(["beta", "alpha"])["output"]["values"]
        assert values.toarray().tolist() == [[0, 1], [1, 0]]
    finally:
        reopened.close()


def test_custom_spacy_sentence_key_preserves_frozen_bigram_boundaries(
    tmp_path: Path,
) -> None:
    """A known bigram must never arise only by joining two sentence edges."""
    _external_modules()
    model_dir = _saved_sentencizer_model(tmp_path)
    project = teal.Project.create(tmp_path / "custom_sentences", name="custom_sentences")
    try:
        source = _seed_documents(project, pd.DataFrame({
            "document_id": [1, 2],
            "text": ["alpha beta. gamma delta.", "beta gamma."],
        }))
        tokens = project.translate(
            SpacyTranslator(
                model=str(model_dir), save_model=True, sentence_key="segment_id",
            ), source, workers=1,
        )["tokens"]
        assert tokens.primary_key == ["document_id", "segment_id", "token_id"]
        output = project.translate(
            ArtifactCountVectorizer(
                field="text", group_by=("document_id",),
                metadata_filters={"is_alpha": True, "is_stop": False},
                ngram_range=(2, 2),
            ), tokens, workers=1, batch_by="document_id", batch_size=2,
        )["output"]
        columns = output.get_data_columns()
        assert "beta_gamma" in columns
        original = output.get_matrix().toarray()
        beta_gamma = columns.index("beta_gamma")
        assert original[0, beta_gamma] == 0
        assert original[1, beta_gamma] == 1
        replay = project.pipeline(start=source, end=output).translate(
            ["alpha beta. gamma delta."]
        )["output"]["values"].toarray()
        assert replay.shape == (1, len(columns))
        np.testing.assert_array_equal(replay[0], original[0])
    finally:
        project.close()


def test_spacy_dummy_token_is_learned_as_vocab_feature_and_can_be_filtered(
    tmp_path: Path,
) -> None:
    from text_analysis_lab.translators.spacy_translator import EMPTY_TOKEN_TEXT

    _external_modules()
    model_dir = _saved_sentencizer_model(tmp_path)
    project = teal.Project.create(tmp_path / "dummy_vocab", name="dummy_vocab")
    try:
        documents = _seed_documents(project, pd.DataFrame({
            "document_id": [1, 2, 3],
            "text": ["apple", "", "banana"],
        }))
        tokens = project.translate(
            SpacyTranslator(
                model=str(model_dir), save_model=True, empty_doc="dummy",
            ),
            documents, workers=1,
        )["tokens"]
        assert tokens.n_rows == 3
        matrix = project.translate(
            ArtifactCountVectorizer(
                field="text", group_by=("document_id",),
            ),
            tokens, workers=1,
        )["output"]
        assert matrix.n_rows == 3
        columns = matrix.get_data_columns()
        assert EMPTY_TOKEN_TEXT in columns
        assert matrix.get_matrix().toarray()[1, columns.index(EMPTY_TOKEN_TEXT)] == 1

        filtered = project.translate(
            ArtifactCountVectorizer(
                field="text", group_by=("document_id",),
                metadata_filters={"is_placeholder": False},
            ), tokens, workers=1,
        )["output"]
        assert filtered.n_rows == 3
        assert EMPTY_TOKEN_TEXT not in filtered.get_data_columns()
        assert filtered.get_matrix()[1].nnz == 0
    finally:
        project.close()
