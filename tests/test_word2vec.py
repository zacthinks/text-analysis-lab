from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import LocalWord2Vec, Word2Vec
from text_analysis_lab.translators.word2vec import _prepare_sequences

PRIMARY_KEY = ("doc_id", "sentence_id", "token_id")
requires_gensim = pytest.mark.skipif(
    importlib.util.find_spec("gensim") is None,
    reason="Word2Vec training requires the optional Gensim dependency.",
)


def _training_frame() -> pd.DataFrame:
    sentences = [
        (1, 0, ["king", "queen", "king", "royal", "queen"]),
        (1, 1, ["dog", "cat", "dog", "pet", "cat"]),
        (2, 0, ["king", "queen", "royal"]),
        (2, 1, ["dog", "cat", "pet"]),
    ]
    rows = []
    for doc_id, sentence_id, words in sentences:
        rows.extend(
            {
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "token_id": token_id,
                "lemma": word,
            }
            for token_id, word in enumerate(words)
        )
    return pd.DataFrame(rows)


def _source(primary_key=PRIMARY_KEY):
    return SimpleNamespace(
        artifact_type=ArtifactType.TABLE, primary_key=list(primary_key)
    )


def _packet(frame: pd.DataFrame, *, primary_key=PRIMARY_KEY) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_tokens",
        primary_key=tuple(primary_key),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _model(*, architecture="skipgram", seed=11) -> Word2Vec:
    return Word2Vec(
        field="lemma",
        vector_size=8,
        window=2,
        min_count=2,
        architecture=architecture,
        negative_samples=3,
        epochs=4,
        learning_rate=0.04,
        min_learning_rate=0.01,
        shrink_windows=False,
        seed=seed,
        training_batch_size=5,
    )


def test_word2vec_is_one_shot_named_row_matrix_contract() -> None:
    model = Word2Vec(field="lemma", min_count=1)
    sources = {"source": _source()}
    spec = model.output_specs(sources=sources, request=TranslationRequest())
    assert spec.artifact_type == ArtifactType.DENSE_MATRIX
    assert spec.lineage_mode == "new_key"
    assert not model.requires_fit
    assert model.is_fitted
    assert not model.supports_fit_translate
    request = model.input_request(
        sources=sources,
        mode="translate",
        request=TranslationRequest(batch_size=7),
    )
    assert request.mode == "full_artifact"
    assert request.batch_size is None
    assert request.columns.data == "lemma"
    assert LocalWord2Vec is Word2Vec
    with pytest.raises(OperatorError, match="one-shot"):
        model.input_request(
            sources=sources,
            mode="fit_translate",
            request=TranslationRequest(),
        )


def test_training_payload_uses_row_names_not_word_metadata(
    monkeypatch, tmp_path
) -> None:
    model = _model()
    vectors = np.arange(48, dtype=np.float32).reshape(6, 8)
    monkeypatch.setattr(
        model,
        "_train",
        lambda sequences: (
            ["cat", "dog", "king", "queen", "pet", "royal"],
            np.asarray([3, 3, 3, 3, 2, 2], dtype=np.int64),
            vectors,
            (10.0, 7.0, 5.0, 3.0),
            "4.4.0",
        ),
    )
    result = model.translate_batch(
        {"source": _packet(_training_frame())},
        mode="translate",
        request=TranslationRequest(),
    )
    payload = result.outputs["output"]
    assert payload["keys"].to_dict("list") == {"word_id": list(range(6))}
    assert payload["metadata"].to_dict("list") == {"count": [3, 3, 3, 3, 2, 2]}
    assert payload["data"]["row_name"] == "word"
    assert payload["data"]["row_names"] == [
        "cat",
        "dog",
        "king",
        "queen",
        "pet",
        "royal",
    ]
    assert np.array_equal(payload["data"]["values"], vectors)
    assert model.training_loss_ == (10.0, 7.0, 5.0, 3.0)
    state = model.to_json_state()
    assert "is_fitted" not in state
    assert model.save_assets(tmp_path) == {}


@pytest.mark.parametrize("architecture", ["skipgram", "cbow"])
@requires_gensim
def test_real_training_is_seeded_and_emits_finite_vectors(architecture: str) -> None:
    first = _model(architecture=architecture, seed=3)
    second = _model(architecture=architecture, seed=3)
    first_payload = first.translate_batch(
        {"source": _packet(_training_frame())},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    second_payload = second.translate_batch(
        {"source": _packet(_training_frame())},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert first_payload["data"]["row_names"] == second_payload["data"]["row_names"]
    assert np.array_equal(
        first_payload["data"]["values"], second_payload["data"]["values"]
    )
    assert np.isfinite(first_payload["data"]["values"]).all()


def test_sequence_preparation_preserves_boundaries_and_validation() -> None:
    frame = _training_frame()
    sequences = _prepare_sequences(
        frame,
        field="lemma",
        sequence_by=("doc_id", "sentence_id"),
        drop_empty=True,
    )
    assert len(sequences) == 4
    assert sequences[0] == ["king", "queen", "king", "royal", "queen"]
    with pytest.raises(ArtifactError, match="empty source"):
        _prepare_sequences(
            frame.iloc[:0],
            field="lemma",
            sequence_by=("doc_id", "sentence_id"),
            drop_empty=True,
        )
    with pytest.raises(OperatorError, match="hierarchical"):
        Word2Vec(field="lemma").input_request(
            sources={"source": _source(("token_id",))},
            mode="translate",
            request=TranslationRequest(),
        )
