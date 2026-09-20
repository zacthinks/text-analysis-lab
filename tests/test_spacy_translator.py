from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators.spacy_translator import (
    SENTENCE_DATA_COLUMNS,
    SENTENCES_LABEL,
    TOKEN_DATA_COLUMNS,
    TOKENS_LABEL,
    SpacyTranslator,
)


def _packet(
    frame: pd.DataFrame, *, primary_key=("collection_id", "document_id")
) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_documents",
        primary_key=tuple(primary_key),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _annotated_doc():
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import Doc

    nlp = spacy.blank("en")
    words = ["Alice", "sees", "Bob", ".", "They", "wave", "."]
    spaces = [True, True, False, True, True, False, False]
    return Doc(
        nlp.vocab,
        words=words,
        spaces=spaces,
        heads=[1, 1, 1, 1, 5, 5, 5],
        deps=["nsubj", "ROOT", "dobj", "punct", "nsubj", "ROOT", "punct"],
        pos=["PROPN", "VERB", "PROPN", "PUNCT", "PRON", "VERB", "PUNCT"],
        tags=["NNP", "VBZ", "NNP", ".", "PRP", "VBP", "."],
        morphs=[
            "Number=Sing",
            "VerbForm=Fin",
            "Number=Sing",
            "",
            "Number=Plur",
            "VerbForm=Fin",
            "",
        ],
        lemmas=["Alice", "see", "Bob", ".", "they", "wave", "."],
        sent_starts=[True, False, False, False, True, False, False],
        ents=["B-PERSON", "O", "B-PERSON", "O", "O", "O", "O"],
    )


class _PipeOnlyNLP:
    def __init__(self, docs_by_text):
        self.docs_by_text = docs_by_text
        self.pipe_calls = []

    def __call__(self, text):  # pragma: no cover - should never be reached
        raise AssertionError("SpacyTranslator must use nlp.pipe(), not nlp(text).")

    def pipe(self, texts, *, batch_size, n_process):
        texts = list(texts)
        self.pipe_calls.append(
            {"texts": texts, "batch_size": batch_size, "n_process": n_process}
        )
        for text in texts:
            yield self.docs_by_text[text]


def test_spacy_output_specs_form_exact_two_level_hierarchy() -> None:
    source = SimpleNamespace(
        artifact_type=ArtifactType.TABLE,
        primary_key=["collection_id", "document_id"],
    )
    translator = SpacyTranslator(model="test_model")
    specs = translator.output_specs(
        sources={"source": source}, request=TranslationRequest()
    )

    assert list(specs) == [SENTENCES_LABEL, TOKENS_LABEL]
    assert specs[SENTENCES_LABEL].artifact_type == ArtifactType.TABLE
    assert specs[SENTENCES_LABEL].lineage_mode == "extended_key"
    assert specs[SENTENCES_LABEL].basis_labels == ("source",)
    assert specs[TOKENS_LABEL].artifact_type == ArtifactType.TABLE
    assert specs[TOKENS_LABEL].lineage_mode == "extended_key"
    assert specs[TOKENS_LABEL].basis_labels == (SENTENCES_LABEL,)

    source_request = translator.input_request(
        sources={"source": source},
        mode="translate",
        request=TranslationRequest(batch_size=19),
    )
    assert source_request.mode == "batches"
    assert source_request.batch_size == 19
    assert source_request.form == "table"
    assert source_request.columns.keys is True
    assert source_request.columns.data == "text"


def test_spacy_translator_uses_pipe_and_emits_exact_normalized_rows(
    monkeypatch,
) -> None:
    import text_analysis_lab.translators.spacy_translator as module

    doc = _annotated_doc()
    fake_nlp = _PipeOnlyNLP({doc.text: doc})
    monkeypatch.setattr(module, "_load_spacy_pipeline", lambda model, disable: fake_nlp)

    frame = pd.DataFrame(
        {
            "collection_id": [7],
            "document_id": [42],
            "text": [doc.text],
        }
    )
    translator = SpacyTranslator(model="fake", spacy_batch_size=23)
    result = translator.translate_batch(
        {"source": _packet(frame)},
        mode="translate",
        request=TranslationRequest(),
    )

    assert fake_nlp.pipe_calls == [
        {"texts": [doc.text], "batch_size": 23, "n_process": 1}
    ]
    assert set(result.outputs) == {SENTENCES_LABEL, TOKENS_LABEL}

    sentences = result.outputs[SENTENCES_LABEL]
    assert list(sentences["keys"].columns) == [
        "collection_id",
        "document_id",
        "sentence_id",
    ]
    assert sentences["keys"].to_dict("records") == [
        {"collection_id": 7, "document_id": 42, "sentence_id": 0},
        {"collection_id": 7, "document_id": 42, "sentence_id": 1},
    ]
    assert tuple(sentences["data"].columns) == SENTENCE_DATA_COLUMNS
    assert sentences["data"]["text"].tolist() == ["Alice sees Bob.", "They wave."]
    for row in sentences["data"].to_dict("records"):
        assert doc.text[row["char_start"] : row["char_end"]] == row["text"]

    tokens = result.outputs[TOKENS_LABEL]
    assert list(tokens["keys"].columns) == [
        "collection_id",
        "document_id",
        "sentence_id",
        "token_id",
    ]
    assert tokens["keys"][["sentence_id", "token_id"]].to_records(
        index=False
    ).tolist() == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert tuple(tokens["data"].columns) == TOKEN_DATA_COLUMNS

    token_frame = pd.concat(
        [tokens["keys"].reset_index(drop=True), tokens["data"].reset_index(drop=True)],
        axis=1,
    )
    assert token_frame["head_token_id"].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert token_frame["dep"].tolist() == [
        "nsubj",
        "ROOT",
        "dobj",
        "punct",
        "nsubj",
        "ROOT",
        "punct",
    ]
    assert token_frame["lemma"].tolist() == [
        "Alice",
        "see",
        "Bob",
        ".",
        "they",
        "wave",
        ".",
    ]
    assert token_frame["pos"].tolist() == [
        "PROPN",
        "VERB",
        "PROPN",
        "PUNCT",
        "PRON",
        "VERB",
        "PUNCT",
    ]
    assert token_frame["tag"].tolist() == ["NNP", "VBZ", "NNP", ".", "PRP", "VBP", "."]
    assert token_frame["ent_iob"].tolist() == ["B", "O", "B", "O", "O", "O", "O"]
    assert token_frame["ent_type"].tolist() == ["PERSON", "", "PERSON", "", "", "", ""]
    assert token_frame["morph"].tolist()[0] == "Number=Sing"
    assert token_frame["morph"].tolist()[4] == "Number=Plur"

    direct_tokens = list(doc)
    for row, token in zip(token_frame.to_dict("records"), direct_tokens, strict=True):
        assert row["text"] == token.text
        assert row["lemma"] == token.lemma_
        assert row["norm"] == token.norm_
        assert row["shape"] == token.shape_
        assert row["pos"] == token.pos_
        assert row["tag"] == token.tag_
        assert row["morph"] == str(token.morph)
        assert row["dep"] == token.dep_
        assert row["ent_iob"] == token.ent_iob_
        assert row["ent_type"] == token.ent_type_
        assert row["char_start"] == token.idx
        assert row["char_end"] == token.idx + len(token.text)
        assert doc.text[row["char_start"] : row["char_end"]] == row["text"]
        for flag in (
            "is_alpha",
            "is_ascii",
            "is_digit",
            "is_lower",
            "is_upper",
            "is_title",
            "is_punct",
            "is_left_punct",
            "is_right_punct",
            "is_space",
            "is_bracket",
            "is_quote",
            "is_currency",
            "is_stop",
            "is_oov",
            "like_url",
            "like_num",
            "like_email",
        ):
            assert row[flag] is bool(getattr(token, flag))
        assert row["is_sent_start"] is token.is_sent_start
        assert row["is_sent_end"] is token.is_sent_end


def test_spacy_head_ids_are_sentence_local_and_roots_self_reference(
    monkeypatch,
) -> None:
    import text_analysis_lab.translators.spacy_translator as module

    doc = _annotated_doc()
    fake_nlp = _PipeOnlyNLP({doc.text: doc})
    monkeypatch.setattr(module, "_load_spacy_pipeline", lambda model, disable: fake_nlp)
    frame = pd.DataFrame({"collection_id": [1], "document_id": [1], "text": [doc.text]})
    output = (
        SpacyTranslator(model="fake")
        .translate_batch(
            {"source": _packet(frame)},
            mode="translate",
            request=TranslationRequest(),
        )
        .outputs[TOKENS_LABEL]
    )
    joined = pd.concat([output["keys"], output["data"]], axis=1)

    for _, sentence in joined.groupby(["collection_id", "document_id", "sentence_id"]):
        token_ids = set(sentence["token_id"].astype(int))
        assert set(sentence["head_token_id"].astype(int)).issubset(token_ids)
        roots = sentence.loc[sentence["dep"] == "ROOT"]
        assert len(roots) == 1
        assert int(roots.iloc[0]["token_id"]) == int(roots.iloc[0]["head_token_id"])


def test_spacy_translator_rejects_pipeline_without_sentence_boundaries(
    monkeypatch,
) -> None:
    spacy = pytest.importorskip("spacy")
    import text_analysis_lab.translators.spacy_translator as module

    nlp = spacy.blank("en")
    doc = nlp.make_doc("No sentence annotations here.")
    fake_nlp = _PipeOnlyNLP({doc.text: doc})
    monkeypatch.setattr(module, "_load_spacy_pipeline", lambda model, disable: fake_nlp)
    frame = pd.DataFrame({"collection_id": [1], "document_id": [1], "text": [doc.text]})
    with pytest.raises(ArtifactError, match="sentence boundaries"):
        SpacyTranslator(model="fake").translate_batch(
            {"source": _packet(frame)},
            mode="translate",
            request=TranslationRequest(),
        )


def test_spacy_translator_state_worker_and_resume_round_trip(tmp_path: Path) -> None:
    translator = SpacyTranslator(
        model="en_core_web_sm",
        text_field="body",
        sentence_key="sent_id",
        token_key="tok_id",
        spacy_batch_size=64,
        disable=("textcat",),
    )
    state = translator.to_json_state()
    restored = SpacyTranslator.from_json_state(state)
    assert restored.to_json_state() == state
    worker = translator.make_translate_worker(
        mode="translate", request=TranslationRequest()
    )
    assert worker.to_json_state() == state
    assert translator.supports_parallel_translate
    assert translator.supports_resume(mode="translate", route="sequential")
    assert translator.supports_resume(mode="translate", route="parallel")

    intermediate = tmp_path / "intermediate"
    translator.save_intermediate_state(
        intermediate,
        operator_id="optr_000001",
        mode="translate",
        route="parallel",
    )
    resumed = SpacyTranslator.load_intermediate_state(
        intermediate,
        operator_id="optr_000001",
        mode="translate",
        route="parallel",
    )
    assert resumed.operator_id == "optr_000001"
    assert resumed.to_json_state() == state


def test_installed_en_core_web_sm_matches_direct_spacy_output() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        reference_nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("optional en_core_web_sm model is not installed")

    text = "Barack Obama visited New York in 2015. He later returned to Washington."
    reference_doc = reference_nlp(text)
    frame = pd.DataFrame({"collection_id": [9], "document_id": [3], "text": [text]})
    output = SpacyTranslator(
        model="en_core_web_sm", spacy_batch_size=8
    ).translate_batch(
        {"source": _packet(frame)},
        mode="translate",
        request=TranslationRequest(),
    )
    emitted_sentences = output.outputs[SENTENCES_LABEL]["data"]
    emitted_tokens = output.outputs[TOKENS_LABEL]["data"]

    assert emitted_sentences["text"].tolist() == [
        span.text for span in reference_doc.sents
    ]
    expected_tokens = list(reference_doc)
    assert emitted_tokens["text"].tolist() == [token.text for token in expected_tokens]
    assert emitted_tokens["lemma"].tolist() == [
        token.lemma_ for token in expected_tokens
    ]
    assert emitted_tokens["pos"].tolist() == [token.pos_ for token in expected_tokens]
    assert emitted_tokens["tag"].tolist() == [token.tag_ for token in expected_tokens]
    assert emitted_tokens["dep"].tolist() == [token.dep_ for token in expected_tokens]
    assert emitted_tokens["ent_iob"].tolist() == [
        token.ent_iob_ for token in expected_tokens
    ]
    assert emitted_tokens["ent_type"].tolist() == [
        token.ent_type_ for token in expected_tokens
    ]
    assert any(emitted_tokens["dep"].astype(str).ne(""))
    assert any(emitted_tokens["pos"].astype(str).ne(""))
    assert any(emitted_tokens["lemma"].astype(str).ne(""))


def test_real_saved_spacy_pipeline_loads_and_processes_batch(tmp_path: Path) -> None:
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model_dir = tmp_path / "blank_en_with_sentencizer"
    nlp.to_disk(model_dir)

    frame = pd.DataFrame(
        {
            "collection_id": [1, 1],
            "document_id": [10, 11],
            "text": ["First sentence. Second sentence.", "Another document."],
        }
    )
    output = SpacyTranslator(model=str(model_dir), spacy_batch_size=2).translate_batch(
        {"source": _packet(frame)},
        mode="translate",
        request=TranslationRequest(),
    )
    sentence_keys = output.outputs[SENTENCES_LABEL]["keys"]
    token_keys = output.outputs[TOKENS_LABEL]["keys"]
    assert sentence_keys.to_dict("records") == [
        {"collection_id": 1, "document_id": 10, "sentence_id": 0},
        {"collection_id": 1, "document_id": 10, "sentence_id": 1},
        {"collection_id": 1, "document_id": 11, "sentence_id": 0},
    ]
    assert (
        token_keys.groupby(["collection_id", "document_id", "sentence_id"])["token_id"]
        .min()
        .eq(0)
        .all()
    )


def test_spacy_pipeline_cache_reuses_loaded_pipeline_within_process(
    tmp_path: Path,
) -> None:
    spacy = pytest.importorskip("spacy")
    import text_analysis_lab.translators.spacy_translator as module

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model_dir = tmp_path / "cached_model"
    nlp.to_disk(model_dir)

    module._load_spacy_pipeline.cache_clear()
    first = module._load_spacy_pipeline(str(model_dir), ())
    second = module._load_spacy_pipeline(str(model_dir), ())
    assert first is second
    assert module._load_spacy_pipeline.cache_info().hits >= 1
