from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import (
    CountVectorizer,
    DelimiterDecomposer,
    RegexCleaner,
    RegexReplaceRule,
    TextLength,
)


def _packet(frame: pd.DataFrame, *, primary_key=("doc_id",)) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_source",
        primary_key=tuple(primary_key),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_delimiter_decomposer_vectorized_explode_and_extended_keys() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [10, 11, 12],
            "text": ["first\n\n second ", None, "third\nfourth"],
        }
    )
    translator = DelimiterDecomposer(delimiter="\n", new_key="line_id")
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    assert payload["keys"].to_dict("records") == [
        {"doc_id": 10, "line_id": 0},
        {"doc_id": 10, "line_id": 1},
        {"doc_id": 12, "line_id": 0},
        {"doc_id": 12, "line_id": 1},
    ]
    assert payload["data"]["text"].tolist() == ["first", "second", "third", "fourth"]


def test_delimiter_decomposer_empty_batch_emits_no_writer_payload() -> None:
    frame = pd.DataFrame({"doc_id": [1, 2], "text": ["", None]})
    translator = DelimiterDecomposer(delimiter="\n")
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    )
    assert result.outputs == {}
    assert (
        translator.handle_batch_result(
            result, batch_index=0, mode="translate", request=TranslationRequest()
        )
        is None
    )


def test_regex_cleaner_vectorized_ordered_rules_nulls_and_flags() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["  HELLO   world!!! ", None, "Hello---TeAL"],
        }
    )
    translator = RegexCleaner(
        rules=[
            RegexReplaceRule(r"hello", "hi", flags=("IGNORECASE",)),
            {"pattern": r"[!-]+", "replacement": " "},
            {"pattern": r"\s+", "replacement": " "},
        ]
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    assert payload["keys"]["doc_id"].tolist() == [1, 2, 3]
    assert payload["data"]["text"].tolist()[0] == "hi world"
    assert pd.isna(payload["data"]["text"].tolist()[1])
    assert payload["data"]["text"].tolist()[2] == "hi TeAL"


def test_count_vectorizer_fit_translate_and_frozen_transform_match_sklearn_semantics() -> (
    None
):
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": [
                "cats chase mice",
                "dogs chase cats",
                "cats sleep",
            ],
        }
    )
    translator = CountVectorizer(
        min_df=1,
        lowercase=True,
        ngram_range=(1, 2),
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    matrix = payload["data"]["values"]
    columns = payload["data"]["columns"]
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (3, len(columns))
    assert translator.is_fitted
    assert "cats" in columns
    assert "chase cats" in columns
    assert payload["keys"]["doc_id"].tolist() == [1, 2, 3]

    frozen = CountVectorizer.from_json_state(
        translator.to_json_state(include_vocabulary=True)
    )
    transformed = frozen.translate_batch(
        {"source": _packet(frame.iloc[[2, 0]].reset_index(drop=True))},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]
    np.testing.assert_array_equal(
        transformed["values"].toarray(), matrix[[2, 0], :].toarray()
    )
    assert transformed["columns"] == columns


@pytest.mark.parametrize("analyzer", ["word", "char", "char_wb"])
def test_count_vectorizer_public_analyzer_modes_round_trip(analyzer: str) -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "text": ["alpha beta", "beta gamma"],
        }
    )
    translator = CountVectorizer(analyzer=analyzer, ngram_range=(1, 2))
    payload = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    ).outputs["output"]["data"]
    assert payload["values"].shape[0] == 2
    assert payload["values"].shape[1] == len(payload["columns"])
    assert payload["columns"]

    frozen = CountVectorizer.from_json_state(
        translator.to_json_state(include_vocabulary=True)
    )
    assert frozen.analyzer == analyzer
    replay = frozen.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    ).outputs["output"]["data"]
    np.testing.assert_array_equal(
        replay["values"].toarray(), payload["values"].toarray()
    )
    assert replay["columns"] == payload["columns"]


def test_count_vectorizer_feature_trimming_and_binary_counts() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["alpha alpha beta", "alpha gamma", "alpha delta"],
        }
    )
    translator = CountVectorizer(min_df=2, binary=True)
    payload = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    ).outputs["output"]["data"]
    assert payload["columns"] == ["alpha"]
    assert payload["values"].toarray().tolist() == [[1], [1], [1]]


def test_text_length_translator_emits_multiple_metadata_columns_without_data(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["One two", "", None],
            "title": ["A title", "B", None],
        }
    )
    translator = TextLength({"text": ["characters", "words"], "title": "words"})
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    assert set(payload) == {"keys", "metadata"}
    assert payload["keys"]["doc_id"].tolist() == [1, 2, 3]
    assert payload["metadata"].to_dict("list") == {
        "text_characters": [7, 0, 0],
        "text_words": [2, 0, 0],
        "title_words": [2, 1, 0],
    }
    restored = TextLength.from_json_state(translator.to_json_state())
    assert restored.lengths == translator.lengths
    translator.save_intermediate_state(
        tmp_path,
        operator_id="optr_text_length",
        mode="translate",
        route="sequential",
    )
    resumed = TextLength.load_intermediate_state(
        tmp_path,
        operator_id="optr_text_length",
        mode="translate",
        route="sequential",
    )
    assert resumed.lengths == translator.lengths
    assert resumed.operator_id == "optr_text_length"


def test_translator_specs_and_source_modes_match_lineage_contracts() -> None:
    source = SimpleNamespace(
        artifact_type=ArtifactType.TABLE,
        primary_key=["doc_id"],
    )
    sources = {"source": source}
    request = TranslationRequest()

    decomposer = DelimiterDecomposer(new_key="paragraph_id")
    decomposer_spec = decomposer.output_specs(sources=sources, request=request)
    assert decomposer_spec.lineage_mode == "extended_key"
    decomposer_request = decomposer.input_request(
        sources=sources, mode="translate", request=request
    )
    assert decomposer_request.mode == "batches"
    assert decomposer_request.batch_size == 10_000
    requested_decomposer = decomposer.input_request(
        sources=sources,
        mode="translate",
        request=TranslationRequest(batch_size=37),
    )
    assert requested_decomposer.batch_size == 37

    cleaner = RegexCleaner()
    cleaner_spec = cleaner.output_specs(sources=sources, request=request)
    assert cleaner_spec.lineage_mode == "preserved_key"
    cleaner_request = cleaner.input_request(
        sources=sources, mode="translate", request=request
    )
    assert cleaner_request.mode == "batches"
    assert cleaner_request.batch_size == 10_000

    vectorizer = CountVectorizer()
    vectorizer_spec = vectorizer.output_specs(sources=sources, request=request)
    assert vectorizer_spec.lineage_mode == "preserved_key"
    fit_request = vectorizer.input_request(
        sources=sources, mode="fit_translate", request=request
    )
    assert fit_request.mode == "full_artifact"
    assert fit_request.batch_size is None

    fitted = CountVectorizer(vocabulary={"alpha": 0, "beta": 1})
    transform_request = fitted.input_request(
        sources=sources,
        mode="translate",
        request=TranslationRequest(batch_size=53),
    )
    assert transform_request.mode == "batches"
    assert transform_request.batch_size == 53
    assert fitted.supports_parallel_translate


def test_translator_json_states_round_trip_without_rowwise_state() -> None:
    import json

    decomposer = DelimiterDecomposer(
        delimiter="\n\n", new_key="paragraph_id", output_text_field="paragraph_text"
    )
    cleaner = RegexCleaner(rules=[{"pattern": "foo", "replacement": "bar", "flags": 2}])
    fitted = CountVectorizer(vocabulary={"alpha": 0, "beta": 1}, ngram_range=(1, 2))

    assert (
        DelimiterDecomposer.from_json_state(
            json.loads(json.dumps(decomposer.to_json_state()))
        ).to_json_state()
        == decomposer.to_json_state()
    )
    assert (
        RegexCleaner.from_json_state(
            json.loads(json.dumps(cleaner.to_json_state()))
        ).to_json_state()
        == cleaner.to_json_state()
    )
    restored = CountVectorizer.from_json_state(
        json.loads(json.dumps(fitted.to_json_state(include_vocabulary=True)))
    )
    assert restored.vocabulary_ == {"alpha": 0, "beta": 1}
    assert restored.ngram_range == (1, 2)


def test_count_vectorizer_porter_stemming_is_persisted_and_applied() -> None:
    translator = CountVectorizer(
        stemmer="porter",
        stop_words="english",
        ngram_range=(1, 1),
    )
    # Fit through the underlying sklearn vectorizer so this dependency-light unit
    # test isolates analyzer behavior from artifact execution.
    vectorizer = translator._make_vectorizer()
    vectorizer.fit_transform(["running runs runner", "the cats cat"])
    translator._vectorizer = vectorizer
    translator.vocabulary_ = dict(vectorizer.vocabulary_)
    names = set(translator._feature_names())
    assert "run" in names
    assert "cat" in names
    assert "the" not in names
    restored = CountVectorizer.from_json_state(
        translator.to_json_state(include_vocabulary=True)
    )
    assert restored.stemmer == "porter"
    transformed = restored.transform_external_texts(["runs cats"])
    assert transformed.shape == (1, len(restored._feature_names()))
