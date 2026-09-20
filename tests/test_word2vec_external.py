from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("gensim")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import EmbeddingLookup, MatrixRowAggregator, Word2Vec


def _register_tokens(project: teal.Project, *, artifact_id: str, rows: pd.DataFrame):
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": rows[["doc_id", "sentence_id", "token_id"]].reset_index(drop=True),
            "data": rows[["lemma"]].reset_index(drop=True),
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def _rows(sentences) -> pd.DataFrame:
    records = []
    for doc_id, sentence_id, words in sentences:
        records.extend(
            {
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "token_id": token_id,
                "lemma": word,
            }
            for token_id, word in enumerate(words)
        )
    return pd.DataFrame(records)


def test_real_word2vec_named_rows_lookup_pool_and_reopen(tmp_path: Path) -> None:
    training_rows = _rows(
        [
            (1, 0, ["king", "queen", "king", "royal", "queen"]),
            (1, 1, ["dog", "cat", "dog", "pet", "cat"]),
            (2, 0, ["king", "queen", "royal"]),
            (2, 1, ["dog", "cat", "pet"]),
        ]
    )
    confirmation_rows = _rows([(3, 0, ["king", "unseen", "cat"])])
    project_path = tmp_path / "word2vec_project"
    project = teal.Project.create(project_path, name="word2vec_named_rows")
    try:
        training = _register_tokens(
            project, artifact_id="art_training_tokens", rows=training_rows
        )
        confirmation = _register_tokens(
            project, artifact_id="art_confirmation_tokens", rows=confirmation_rows
        )
        trainer = Word2Vec(
            field="lemma",
            vector_size=12,
            window=2,
            min_count=2,
            negative_samples=3,
            epochs=4,
            sample=0.0,
            shrink_windows=False,
            seed=17,
            training_batch_size=5,
        )
        word_vectors = project.translate(trainer, training)["output"]
        assert word_vectors.primary_key == ["word_id"]
        assert word_vectors.has_row_names
        assert word_vectors.row_name == "word"
        assert word_vectors.get_row_names() == [
            "cat",
            "dog",
            "king",
            "queen",
            "pet",
            "royal",
        ]
        assert word_vectors.get_row_by_name("king").shape == (1, 12)
        assert word_vectors.position_by_row_name("king") == 2
        metadata = word_vectors.query(
            data_columns=False,
            metadata_columns=["count"],
            metadata_mode="local",
        )
        assert metadata["count"].tolist() == [3, 3, 3, 3, 2, 2]

        neighbors = word_vectors.analysis.nearest_neighbors(row_name="king", k=3)
        assert "word" in neighbors.columns
        assert "king" not in neighbors["word"].tolist()

        token_vectors = project.translate(
            EmbeddingLookup(field="lemma"),
            {"tokens": confirmation, "embeddings": word_vectors},
        )["output"]
        assert token_vectors.primary_key == ["doc_id", "sentence_id", "token_id"]
        matrix = token_vectors.get_matrix()
        assert matrix.shape == (3, 12)
        assert np.array_equal(matrix[0], word_vectors.get_row_by_name("king")[0])
        assert np.array_equal(matrix[1], np.zeros(12, dtype=np.float32))

        sentence_vectors = project.translate(
            MatrixRowAggregator(group_by=["doc_id", "sentence_id"], pooling="mean"),
            token_vectors,
        )["output"]
        assert sentence_vectors.primary_key == ["doc_id", "sentence_id"]
        assert np.allclose(sentence_vectors.get_matrix()[0], matrix.mean(axis=0))

        operator_id = str(trainer.operator_id)
        word_vectors_id = word_vectors.artifact_id
        expected_king = word_vectors.get_row_by_name("king").copy()
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        restored_trainer = reopened.get_operator(operator_id)
        assert isinstance(restored_trainer, Word2Vec)
        assert not hasattr(restored_trainer, "vectors_")
        restored_vectors = reopened.get_artifact(word_vectors_id)
        assert restored_vectors.get_row_names()[2] == "king"
        assert np.array_equal(restored_vectors.get_row_by_name("king"), expected_king)
    finally:
        reopened.close()
