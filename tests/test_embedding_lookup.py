from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import EmbeddingLookup, MatrixRowAggregator


def _packet(label, artifact_id, key, data):
    return InputBatch(label, artifact_id, tuple(key), data, 0, 1, True, True)


def _tokens():
    return pd.DataFrame(
        {
            "doc_id": [1, 1, 1],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 2],
            "lemma": ["b", "missing", "a"],
        }
    )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_embedding_lookup_maps_named_rows_and_zero_fills_oov(sparse_input: bool) -> None:
    values = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    matrix = sparse.csr_matrix(values) if sparse_input else values
    op = EmbeddingLookup(field="lemma")
    result = op.translate_batch(
        {
            "tokens": _packet(
                "tokens",
                "art_tokens",
                ("doc_id", "sentence_id", "token_id"),
                _tokens(),
            ),
            "embeddings": _packet(
                "embeddings",
                "art_embeddings",
                ("word_id",),
                {
                    "info": pd.DataFrame({"_position": [0, 1]}),
                    "matrix": matrix,
                    "columns": ["d0", "d1"],
                    "row_names": ["a", "b"],
                },
            ),
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    actual = result["data"]["values"]
    actual = actual.toarray() if sparse.issparse(actual) else actual
    assert np.array_equal(actual, [[3, 4], [0, 0], [1, 2]])
    assert result["metadata"].to_dict("list") == {
        "lemma": ["b", "missing", "a"],
        "in_vocabulary": [True, False, True],
    }


def test_embedding_lookup_validates_named_source_and_strict_oov() -> None:
    tokens = SimpleNamespace(artifact_type=ArtifactType.TABLE)
    unnamed = SimpleNamespace(
        artifact_type=ArtifactType.DENSE_MATRIX, has_row_names=False
    )
    with pytest.raises(OperatorError, match="named rows"):
        EmbeddingLookup(field="lemma").output_specs(
            sources={"tokens": tokens, "embeddings": unnamed},
            request=TranslationRequest(),
        )

    op = EmbeddingLookup(field="lemma", oov_policy="error")
    with pytest.raises(ArtifactError, match="1 out-of-vocabulary"):
        op.translate_batch(
            {
                "tokens": _packet(
                    "tokens",
                    "art_tokens",
                    ("doc_id", "sentence_id", "token_id"),
                    _tokens().iloc[:2],
                ),
                "embeddings": _packet(
                    "embeddings",
                    "art_embeddings",
                    ("word_id",),
                    {
                        "info": pd.DataFrame({"_position": [0, 1]}),
                        "matrix": np.asarray([[1, 2], [3, 4]]),
                        "columns": ["d0", "d1"],
                        "row_names": ["a", "b"],
                    },
                ),
            },
            mode="translate",
            request=TranslationRequest(),
        )


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("sum", [[4.0, 6.0], [5.0, 6.0]]), ("mean", [[2.0, 3.0], [5.0, 6.0]])],
)
def test_matrix_row_aggregator_pools_by_key_prefix(pooling, expected) -> None:
    info = pd.DataFrame(
        {
            "doc_id": [1, 1, 2],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 0],
        }
    )
    op = MatrixRowAggregator(group_by=["doc_id"], pooling=pooling)
    payload = op.translate_batch(
        {
            "source": _packet(
                "source",
                "art_matrix",
                ("doc_id", "sentence_id", "token_id"),
                {
                    "info": info,
                    "matrix": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float),
                    "columns": ["d0", "d1"],
                },
            )
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert payload["keys"].to_dict("list") == {"doc_id": [1, 2]}
    assert payload["metadata"]["n_rows"].tolist() == [2, 1]
    assert np.allclose(payload["data"]["values"], expected)
