from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from text_analysis_lab.translators.coreference_resolver import (
    FAILURE_DATA_COLUMNS as COREF_FAILURE_DATA_COLUMNS,
    MENTION_DATA_COLUMNS,
)
from text_analysis_lab.translators.semantic_role_labeler import (
    FAILURE_DATA_COLUMNS as SRL_FAILURE_DATA_COLUMNS,
    PREDICATE_DATA_COLUMNS,
    ROLE_SPAN_DATA_COLUMNS,
)
from text_analysis_lab.translators.word_sense_disambiguator import (
    CANDIDATE_DATA_COLUMNS,
    UNRESOLVED_DATA_COLUMNS,
)


SENTENCE_KEYS = ("row_id", "sentence_id")
TOKEN_KEYS = (*SENTENCE_KEYS, "token_id")
PREDICATE_KEYS = (*SENTENCE_KEYS, "predicate_id")
ROLE_SPAN_KEYS = (*PREDICATE_KEYS, "role_id")
CANDIDATE_KEYS = (*TOKEN_KEYS, "candidate_id")
MENTION_KEYS = ("row_id", "cluster_id", "mention_id")


_SENTENCES = (
    (0, 0, "John and Mary ate bread and apples."),
    (0, 1, "John likes all of the apples."),
    (1, 0, "A dog with a very green hat barked."),
    (1, 1, "Green apples."),
    (2, 0, "John fell thanks to the rain."),
    (2, 1, "She picked up an apple."),
    (3, 0, "Those three apples fell."),
    (3, 1, "Alice saw the book on the mat."),
    (4, 0, "Alice read the book on number theory."),
)


def _t(
    text: str,
    lemma: str,
    pos: str,
    tag: str,
    dep: str,
    head: int,
    ent_type: str = "",
    morph: str = "",
) -> tuple[str, str, str, str, str, int, str, str]:
    return text, lemma, pos, tag, dep, head, ent_type, morph


_TOKEN_SPECS = {
    (0, 0): (
        _t("John", "John", "PROPN", "NNP", "nsubj", 3, "PERSON", "Number=Sing"),
        _t("and", "and", "CCONJ", "CC", "cc", 0),
        _t("Mary", "Mary", "PROPN", "NNP", "conj", 0, "PERSON", "Number=Sing"),
        _t("ate", "eat", "VERB", "VBD", "ROOT", 3),
        _t("bread", "bread", "NOUN", "NN", "dobj", 3, morph="Number=Sing"),
        _t("and", "and", "CCONJ", "CC", "cc", 4),
        _t("apples", "apple", "NOUN", "NNS", "conj", 4, morph="Number=Plur"),
        _t(".", ".", "PUNCT", ".", "punct", 3),
    ),
    (0, 1): (
        _t("John", "John", "PROPN", "NNP", "nsubj", 1, "PERSON", "Number=Sing"),
        _t("likes", "like", "VERB", "VBZ", "ROOT", 1),
        _t("all", "all", "PRON", "DT", "dobj", 1),
        _t("of", "of", "ADP", "IN", "prep", 2),
        _t("the", "the", "DET", "DT", "det", 5, morph="Definite=Def"),
        _t("apples", "apple", "NOUN", "NNS", "pobj", 3, morph="Number=Plur"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
    (1, 0): (
        _t("A", "a", "DET", "DT", "det", 1, morph="Definite=Ind"),
        _t("dog", "dog", "NOUN", "NN", "nsubj", 7, morph="Number=Sing"),
        _t("with", "with", "ADP", "IN", "prep", 1),
        _t("a", "a", "DET", "DT", "det", 6, morph="Definite=Ind"),
        _t("very", "very", "ADV", "RB", "advmod", 5),
        _t("green", "green", "ADJ", "JJ", "amod", 6),
        _t("hat", "hat", "NOUN", "NN", "pobj", 2, morph="Number=Sing"),
        _t("barked", "bark", "VERB", "VBD", "ROOT", 7),
        _t(".", ".", "PUNCT", ".", "punct", 7),
    ),
    (1, 1): (
        _t("Green", "green", "ADJ", "JJ", "amod", 1),
        _t("apples", "apple", "NOUN", "NNS", "ROOT", 1, morph="Number=Plur"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
    (2, 0): (
        _t("John", "John", "PROPN", "NNP", "nsubj", 1, "PERSON", "Number=Sing"),
        _t("fell", "fall", "VERB", "VBD", "ROOT", 1),
        _t("thanks", "thank", "ADP", "IN", "prep", 1),
        _t("to", "to", "ADP", "IN", "prep", 2),
        _t("the", "the", "DET", "DT", "det", 5, morph="Definite=Def"),
        _t("rain", "rain", "NOUN", "NN", "pobj", 3, morph="Number=Sing"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
    (2, 1): (
        _t("She", "she", "PRON", "PRP", "nsubj", 1),
        _t("picked", "pick", "VERB", "VBD", "ROOT", 1),
        _t("up", "up", "ADP", "RP", "prt", 1),
        _t("an", "an", "DET", "DT", "det", 4, morph="Definite=Ind"),
        _t("apple", "apple", "NOUN", "NN", "dobj", 1, morph="Number=Sing"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
    (3, 0): (
        _t("Those", "those", "DET", "DT", "det", 2, morph="Number=Plur|PronType=Dem"),
        _t("three", "three", "NUM", "CD", "nummod", 2, morph="NumType=Card"),
        _t("apples", "apple", "NOUN", "NNS", "nsubj", 3, morph="Number=Plur"),
        _t("fell", "fall", "VERB", "VBD", "ROOT", 3),
        _t(".", ".", "PUNCT", ".", "punct", 3),
    ),
    (3, 1): (
        _t("Alice", "Alice", "PROPN", "NNP", "nsubj", 1, "PERSON", "Number=Sing"),
        _t("saw", "see", "VERB", "VBD", "ROOT", 1),
        _t("the", "the", "DET", "DT", "det", 3, morph="Definite=Def"),
        _t("book", "book", "NOUN", "NN", "dobj", 1, morph="Number=Sing"),
        _t("on", "on", "ADP", "IN", "prep", 3),
        _t("the", "the", "DET", "DT", "det", 6, morph="Definite=Def"),
        _t("mat", "mat", "NOUN", "NN", "pobj", 4, morph="Number=Sing"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
    (4, 0): (
        _t("Alice", "Alice", "PROPN", "NNP", "nsubj", 1, "PERSON", "Number=Sing"),
        _t("read", "read", "VERB", "VBD", "ROOT", 1),
        _t("the", "the", "DET", "DT", "det", 3, morph="Definite=Def"),
        _t("book", "book", "NOUN", "NN", "dobj", 1, morph="Number=Sing"),
        _t("on", "on", "ADP", "IN", "prep", 3),
        _t("number", "number", "NOUN", "NN", "compound", 6, morph="Number=Sing"),
        _t("theory", "theory", "NOUN", "NN", "pobj", 4, morph="Number=Sing"),
        _t(".", ".", "PUNCT", ".", "punct", 1),
    ),
}


_PREDICATES = (
    (0, 0, 0, 3),
    (0, 1, 0, 1),
    (1, 0, 0, 7),
    (2, 0, 0, 1),
    (2, 1, 0, 1),
    (3, 0, 0, 3),
    (3, 1, 0, 1),
)


_ROLE_SPANS = (
    (0, 0, 0, 0, "ARG0", 0, 3),
    (0, 0, 0, 1, "V", 3, 4),
    (0, 0, 0, 2, "ARG1", 4, 7),
    (0, 1, 0, 0, "ARG0", 0, 1),
    (0, 1, 0, 1, "V", 1, 2),
    (0, 1, 0, 2, "ARG1", 2, 6),
    (1, 0, 0, 0, "ARG0", 0, 7),
    (1, 0, 0, 1, "V", 7, 8),
    (2, 0, 0, 0, "ARG0", 0, 1),
    (2, 0, 0, 1, "V", 1, 2),
    (2, 0, 0, 2, "ARGM-CAU", 2, 6),
    (2, 1, 0, 0, "ARG0", 0, 1),
    (2, 1, 0, 1, "V", 1, 2),
    (2, 1, 0, 2, "ARG1", 3, 5),
    (3, 0, 0, 0, "ARG0", 0, 3),
    (3, 0, 0, 1, "V", 3, 4),
    (3, 1, 0, 0, "ARG0", 0, 1),
    (3, 1, 0, 1, "V", 1, 2),
    (3, 1, 0, 2, "ARG1", 2, 7),
)


_CANDIDATES = (
    ((0, 1, 5), (("apple.n.01", "apple", 0.85), ("apple.n.02", "apple", 0.15))),
    ((1, 0, 1), (("dog.n.01", "dog", 0.90), ("dog.n.02", "dog", 0.10))),
    ((1, 0, 5), (("green.a.01", "green", 0.70), ("green.a.02", "green", 0.30))),
    ((2, 0, 5), (("rain.n.01", "rain", 0.80), ("rain.n.02", "rain", 0.20))),
    ((2, 1, 4), (("apple.n.01", "apple", 0.75), ("apple.n.02", "apple", 0.25))),
    ((3, 1, 3), (("book.n.01", "book", 0.65), ("book.n.02", "book", 0.35))),
    ((4, 0, 6), (("theory.n.01", "theory", 0.88), ("theory.n.02", "theory", 0.12))),
)


@dataclass
class BoiLinguisticFixture:
    documents: pd.DataFrame
    sentences: pd.DataFrame
    tokens: pd.DataFrame
    predicates: pd.DataFrame
    role_spans: pd.DataFrame
    candidates: pd.DataFrame
    wsd_unresolved: pd.DataFrame
    mentions: pd.DataFrame
    srl_failures: pd.DataFrame
    coref_failures: pd.DataFrame


def _sentence_positions() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_document: dict[int, list[tuple[int, str]]] = {}
    for row_id, sentence_id, text in _SENTENCES:
        by_document.setdefault(row_id, []).append((sentence_id, text))

    document_rows: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []
    for row_id, entries in sorted(by_document.items()):
        entries = sorted(entries)
        document_text = " ".join(text for _, text in entries)
        document_rows.append({"row_id": row_id, "text": document_text})
        cursor = 0
        for sentence_id, text in entries:
            start = document_text.index(text, cursor)
            end = start + len(text)
            sentence_rows.append(
                {
                    "row_id": row_id,
                    "sentence_id": sentence_id,
                    "text": text,
                    "char_start": start,
                    "char_end": end,
                }
            )
            cursor = end

    return pd.DataFrame(document_rows), pd.DataFrame(sentence_rows)


def _tokens(sentences: pd.DataFrame) -> pd.DataFrame:
    sentence_lookup = {
        (int(row.row_id), int(row.sentence_id)): row
        for row in sentences.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for sentence_key, specs in _TOKEN_SPECS.items():
        sentence = sentence_lookup[sentence_key]
        local_cursor = 0
        for token_id, spec in enumerate(specs):
            text, lemma, pos, tag, dep, head, ent_type, morph = spec
            local_start = sentence.text.index(text, local_cursor)
            local_end = local_start + len(text)
            rows.append(
                {
                    "row_id": sentence_key[0],
                    "sentence_id": sentence_key[1],
                    "token_id": token_id,
                    "text": text,
                    "lemma": lemma,
                    "pos": pos,
                    "tag": tag,
                    "morph": morph,
                    "dep": dep,
                    "head_token_id": head,
                    "ent_type": ent_type,
                    "char_start": int(sentence.char_start) + local_start,
                    "char_end": int(sentence.char_start) + local_end,
                }
            )
            local_cursor = local_end
    return pd.DataFrame(rows)


def _predicates(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_id, sentence_id, predicate_id, token_id in _PREDICATES:
        token = tokens.loc[
            (tokens["row_id"] == row_id)
            & (tokens["sentence_id"] == sentence_id)
            & (tokens["token_id"] == token_id)
        ].iloc[0]
        rows.append(
            {
                "row_id": row_id,
                "sentence_id": sentence_id,
                "predicate_id": predicate_id,
                "predicate_token_id": token_id,
                "char_start": int(token["char_start"]),
                "char_end": int(token["char_end"]),
                "text": str(token["text"]),
                "lemma": str(token["lemma"]),
                "pos": str(token["pos"]),
            }
        )
    return pd.DataFrame(rows, columns=[*PREDICATE_KEYS, *PREDICATE_DATA_COLUMNS])


def _role_spans(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_id, sentence_id, predicate_id, role_id, role, start_id, end_id in _ROLE_SPANS:
        selected = tokens.loc[
            (tokens["row_id"] == row_id)
            & (tokens["sentence_id"] == sentence_id)
            & tokens["token_id"].between(start_id, end_id - 1)
        ].sort_values("token_id")
        rows.append(
            {
                "row_id": row_id,
                "sentence_id": sentence_id,
                "predicate_id": predicate_id,
                "role_id": role_id,
                "role": role,
                "token_start_id": start_id,
                "token_end_id": end_id,
                "char_start": int(selected.iloc[0]["char_start"]),
                "char_end": int(selected.iloc[-1]["char_end"]),
                "text": " ".join(selected["text"].astype(str)),
                "score": 0.95,
            }
        )
    return pd.DataFrame(rows, columns=[*ROLE_SPAN_KEYS, *ROLE_SPAN_DATA_COLUMNS])


def _candidate_rows(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    token_lookup = {
        (int(row.row_id), int(row.sentence_id), int(row.token_id)): row
        for row in tokens.itertuples(index=False)
    }
    for token_key, senses in _CANDIDATES:
        token = token_lookup[token_key]
        top_margin = float(senses[0][2] - senses[1][2]) if len(senses) > 1 else None
        for candidate_id, (sense_id, candidate_lemma, score) in enumerate(senses):
            rank = candidate_id + 1
            model_input = f"{token.text} :: {sense_id}"
            row = {
                "row_id": token_key[0],
                "sentence_id": token_key[1],
                "token_id": token_key[2],
                "candidate_id": candidate_id,
                "surface_form": token.text,
                "parser_lemma": token.lemma,
                "pos": token.pos,
                "sense_id": sense_id,
                "synset_id": sense_id,
                "sense_label": sense_id,
                "candidate_lemma": candidate_lemma,
                "aliases": json.dumps([candidate_lemma]),
                "ili": None,
                "ontology_id": "fixture-wordnet",
                "ontology_version": "1",
                "gloss_text": f"fixture gloss for {sense_id}",
                "gloss_hash": hashlib.sha256(sense_id.encode()).hexdigest(),
                "model_input_text": model_input,
                "model_input_hash": hashlib.sha256(model_input.encode()).hexdigest(),
                "raw_score": float(score),
                "score_type": "fixture_probability",
                "normalized_score": float(score),
                "rank": rank,
                "selected": rank == 1,
                "top1_margin": top_margin,
                "candidate_source": "fixture",
                "candidate_kind": "singleword",
                "candidate_components": "[]",
                "candidate_trigger_lemmas": "[]",
                "candidate_component_token_indices": "[]",
                "reader_target_start": 0,
                "reader_target_end": 1,
                "reader_target_text": token.text,
                "reader_candidate_heading": token.text,
                "reader_target_span_policy": "carrier_only",
                "model_name": "fixture-wsd",
                "model_revision": "fixture-v1",
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=[*CANDIDATE_KEYS, *CANDIDATE_DATA_COLUMNS])


def _wsd_unresolved(tokens: pd.DataFrame) -> pd.DataFrame:
    token = tokens.loc[
        (tokens["row_id"] == 1)
        & (tokens["sentence_id"] == 1)
        & (tokens["token_id"] == 0)
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "row_id": 1,
                "sentence_id": 1,
                "token_id": 0,
                "surface_form": token["text"],
                "parser_lemma": token["lemma"],
                "pos": token["pos"],
                "reason": "no_candidates",
            }
        ],
        columns=[*TOKEN_KEYS, *UNRESOLVED_DATA_COLUMNS],
    )


def _mentions(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mention_id, sentence_id in enumerate((0, 1)):
        token = tokens.loc[
            (tokens["row_id"] == 0)
            & (tokens["sentence_id"] == sentence_id)
            & (tokens["token_id"] == 0)
        ].iloc[0]
        rows.append(
            {
                "row_id": 0,
                "cluster_id": 0,
                "mention_id": mention_id,
                "sentence_id": sentence_id,
                "char_start": int(token["char_start"]),
                "char_end": int(token["char_end"]),
                "token_start_id": 0,
                "token_end_id": 1,
                "text": token["text"],
                "head_token_id": 0,
                "head_text": token["text"],
                "head_lemma": token["lemma"],
                "head_pos": token["pos"],
                "ent_type": token["ent_type"],
                "is_first_mention": mention_id == 0,
            }
        )
    return pd.DataFrame(rows, columns=[*MENTION_KEYS, *MENTION_DATA_COLUMNS])


def build_boi_linguistic_fixture() -> BoiLinguisticFixture:
    documents, sentences = _sentence_positions()
    tokens = _tokens(sentences)
    predicates = _predicates(tokens)
    role_spans = _role_spans(tokens)
    candidates = _candidate_rows(tokens)
    unresolved = _wsd_unresolved(tokens)
    mentions = _mentions(tokens)

    srl_failures = pd.DataFrame(
        [
            {
                "row_id": 4,
                "sentence_id": 0,
                "reason": "cuda_out_of_memory",
                "detail": json.dumps(
                    [{"reason": "cuda_out_of_memory", "detail": "fixture failure"}]
                ),
            }
        ],
        columns=[*SENTENCE_KEYS, *SRL_FAILURE_DATA_COLUMNS],
    )
    coref_failures = pd.DataFrame(
        [
            {
                "row_id": 4,
                "reason": "document_too_long",
                "detail": "fixture document-level model failure",
                "model_tokens": 9999,
                "max_model_tokens": 4096,
            }
        ],
        columns=["row_id", *COREF_FAILURE_DATA_COLUMNS],
    )

    return BoiLinguisticFixture(
        documents=documents,
        sentences=sentences,
        tokens=tokens,
        predicates=predicates,
        role_spans=role_spans,
        candidates=candidates,
        wsd_unresolved=unresolved,
        mentions=mentions,
        srl_failures=srl_failures,
        coref_failures=coref_failures,
    )
