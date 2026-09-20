from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators.coreference_resolver import CoreferenceResolver
from text_analysis_lab.translators.semantic_role_labeler import SemanticRoleLabeler
from text_analysis_lab.translators.word_sense_disambiguator import WordSenseDisambiguator


def _packet(label: str, primary_key: tuple[str, ...], frame: pd.DataFrame) -> InputBatch:
    return InputBatch(
        source_label=label,
        artifact_id=f"artifact_{label}",
        primary_key=primary_key,
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_coreference_resolver_keeps_exact_spans_and_aligns_dependency_head(monkeypatch):
    import text_analysis_lab.translators.coreference_resolver as module
    from text_analysis_lab._linguistics.coreference.runtime import (
        CorefBatchPrediction,
        CorefMention,
        CorefPrediction,
    )

    class FakeRuntime:
        def document_token_counts(self, texts):
            return [len(texts[0].split())], None

        def predict_texts(self, texts, **kwargs):
            assert kwargs["max_tokens_in_batch"] == 2000
            mentions = (
                CorefMention(0, 0, 0, 5, "Alice"),
                CorefMention(0, 1, 12, 15, "She"),
            )
            return CorefBatchPrediction(
                model="lingmess",
                model_repository="fake",
                device="cpu",
                elapsed_seconds=0.0,
                documents=(CorefPrediction(0, texts[0], mentions),),
            )

    monkeypatch.setattr(module, "_make_runtime", lambda **kwargs: FakeRuntime())
    documents = pd.DataFrame({"row_id": [7], "text": ["Alice left. She slept."]})
    tokens = pd.DataFrame(
        {
            "row_id": [7, 7, 7, 7, 7, 7],
            "sentence_id": [0, 0, 0, 1, 1, 1],
            "token_id": [0, 1, 2, 0, 1, 2],
            "text": ["Alice", "left", ".", "She", "slept", "."],
            "lemma": ["Alice", "leave", ".", "she", "sleep", "."],
            "pos": ["PROPN", "VERB", "PUNCT", "PRON", "VERB", "PUNCT"],
            "dep": ["nsubj", "ROOT", "punct", "nsubj", "ROOT", "punct"],
            "head_token_id": [1, 1, 1, 1, 1, 1],
            "ent_type": ["PERSON", "", "", "", "", ""],
            "char_start": [0, 6, 10, 12, 16, 21],
            "char_end": [5, 10, 11, 15, 21, 22],
        }
    )
    result = CoreferenceResolver().translate_batch(
        {
            "documents": _packet("documents", ("row_id",), documents),
            "tokens": _packet("tokens", ("row_id", "sentence_id", "token_id"), tokens),
        },
        mode="translate",
        request=TranslationRequest(),
    )
    keys = result.outputs["mentions"]["keys"]
    data = result.outputs["mentions"]["data"]
    assert keys.to_dict("records") == [
        {"row_id": 7, "cluster_id": 0, "mention_id": 0},
        {"row_id": 7, "cluster_id": 0, "mention_id": 1},
    ]
    assert data["sentence_id"].tolist() == [0, 1]
    assert data["head_token_id"].tolist() == [0, 0]
    assert data["text"].tolist() == ["Alice", "She"]
    assert data["is_first_mention"].tolist() == [True, False]


def test_semantic_role_labeler_emits_predicates_roles_and_heads(monkeypatch):
    import text_analysis_lab.translators.semantic_role_labeler as module
    from text_analysis_lab._linguistics.srl.runtime import SrlTokenPrediction
    from text_analysis_lab._linguistics.srl.structures import BioSpan

    class FakeRuntime:
        def encode_tokens(self, tokens):
            return tuple(tokens)

        def predict_encoded_batch(self, instances):
            assert [predicate for _, predicate in instances] == [1]
            tokens = tuple(instances[0][0])
            return (
                SrlTokenPrediction(
                    tokens=tokens,
                    predicate_index=1,
                    predicate="runs",
                    wordpieces=tokens,
                    input_ids=(1, 2, 3, 4),
                    predicate_indicator=(0, 1, 0, 0),
                    wordpiece_offsets=(0, 1, 2, 3),
                    wordpiece_tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    raw_tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    bio_repairs=(),
                    word_scores=(0.9, 0.95, 0.8, 0.99),
                    spans=(BioSpan("ARG0", 0, 1), BioSpan("V", 1, 2), BioSpan("ARGM-MNR", 2, 3)),
                    description="",
                ),
            )

    monkeypatch.setattr(module, "_make_runtime", lambda **kwargs: FakeRuntime())
    sentences = pd.DataFrame(
        {"row_id": [2], "sentence_id": [0], "text": ["Alice runs quickly."], "char_start": [0], "char_end": [19]}
    )
    tokens = pd.DataFrame(
        {
            "row_id": [2, 2, 2, 2],
            "sentence_id": [0, 0, 0, 0],
            "token_id": [0, 1, 2, 3],
            "text": ["Alice", "runs", "quickly", "."],
            "lemma": ["Alice", "run", "quickly", "."],
            "pos": ["PROPN", "VERB", "ADV", "PUNCT"],
            "tag": ["NNP", "VBZ", "RB", "."],
            "dep": ["nsubj", "ROOT", "advmod", "punct"],
            "head_token_id": [1, 1, 1, 1],
            "ent_type": ["PERSON", "", "", ""],
            "char_start": [0, 6, 11, 18],
            "char_end": [5, 10, 18, 19],
        }
    )
    result = SemanticRoleLabeler().translate_batch(
        {
            "sentences": _packet("sentences", ("row_id", "sentence_id"), sentences),
            "tokens": _packet("tokens", ("row_id", "sentence_id", "token_id"), tokens),
        },
        mode="translate",
        request=TranslationRequest(),
    )
    assert result.outputs["predicates"]["keys"].to_dict("records") == [
        {"row_id": 2, "sentence_id": 0, "predicate_id": 0}
    ]
    roles = result.outputs["roles"]["data"]
    assert roles["role"].tolist() == ["ARG0", "V", "ARGM-MNR"]
    assert roles["head_text"].tolist() == ["Alice", "runs", "quickly"]
    assert roles["token_start_id"].tolist() == [0, 1, 2]


@dataclass
class _CachePaths:
    root: Path
    models: Path
    huggingface_hub: Path
    wsd_gloss_text: Path
    wordnet: Path
    wordnet_mwe_indices: Path


def test_word_sense_disambiguator_targets_all_wordnet_eligible_tokens(monkeypatch, tmp_path):
    import text_analysis_lab.translators.word_sense_disambiguator as module
    from text_analysis_lab._linguistics.wsd.types import GlossPayload, SenseCandidate

    cache = _CachePaths(
        root=tmp_path,
        models=tmp_path / "models",
        huggingface_hub=tmp_path / "hf",
        wsd_gloss_text=tmp_path / "gloss",
        wordnet=tmp_path / "wn",
        wordnet_mwe_indices=tmp_path / "mwe",
    )
    for value in cache.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "user_cache_paths", lambda: cache)

    class FakeOntology:
        def descriptor(self):
            return {"type": "fake"}

        def candidates_for_target(self, target):
            base = target.lookup_form.casefold()
            return (
                SenseCandidate(
                    sense_id=f"{base}-1",
                    synset_id=f"{base}-synset-1",
                    ontology_id="fake",
                    ontology_version="1",
                    lemma=base,
                    pos=target.pos,
                    gloss=GlossPayload(definition=f"first {base} sense"),
                    sense_label=f"{base}.01",
                ),
                SenseCandidate(
                    sense_id=f"{base}-2",
                    synset_id=f"{base}-synset-2",
                    ontology_id="fake",
                    ontology_version="1",
                    lemma=base,
                    pos=target.pos,
                    gloss=GlossPayload(definition=f"second {base} sense"),
                    sense_label=f"{base}.02",
                ),
            )

    class FakeBackend:
        def descriptor(self):
            return {"type": "fake"}

        def score_target(self, target, candidate_texts):
            return {text: (0.8 if index == 0 else 0.2) for index, text in enumerate(candidate_texts)}

    monkeypatch.setattr(module, "_make_ontology", lambda **kwargs: FakeOntology())
    monkeypatch.setattr(module, "_make_backend", lambda **kwargs: FakeBackend())
    tokens = pd.DataFrame(
        {
            "row_id": [0, 0, 0],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 2],
            "text": ["Dogs", "run", "."],
            "lemma": ["dog", "run", "."],
            "pos": ["NOUN", "VERB", "PUNCT"],
            "ent_type": ["", "", ""],
        }
    )
    translator = WordSenseDisambiguator(acknowledge_noncommercial_license=True)
    result = translator.translate_batch(
        {"tokens": _packet("tokens", ("row_id", "sentence_id", "token_id"), tokens)},
        mode="translate",
        request=TranslationRequest(),
    )
    sense_keys = result.outputs["senses"]["keys"]
    assert sense_keys["token_id"].tolist() == [0, 1]
    senses = result.outputs["senses"]["data"]
    assert senses["surface_form"].tolist() == ["Dogs", "run"]
    assert senses["normalized_score"].tolist() == pytest.approx([0.8, 0.8])
    candidates = result.outputs["candidates"]["keys"]
    assert candidates.shape[0] == 4
    assert "unresolved" in result.outputs
    assert result.outputs["unresolved"]["keys"].empty
    assert result.outputs["unresolved"]["data"].empty



def test_word_sense_disambiguator_excludes_space_tokens_from_wsl_reader_context(monkeypatch, tmp_path):
    import text_analysis_lab.translators.word_sense_disambiguator as module
    from text_analysis_lab._linguistics.wsd.types import GlossPayload, SenseCandidate

    cache = _CachePaths(
        root=tmp_path,
        models=tmp_path / "models",
        huggingface_hub=tmp_path / "hf",
        wsd_gloss_text=tmp_path / "gloss",
        wordnet=tmp_path / "wn",
        wordnet_mwe_indices=tmp_path / "mwe",
    )
    for value in cache.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "user_cache_paths", lambda: cache)

    class FakeOntology:
        def descriptor(self):
            return {"type": "fake"}

        def candidates_for_target(self, target):
            base = target.lookup_form.casefold()
            return (
                SenseCandidate(
                    sense_id=f"{base}-1",
                    synset_id=f"{base}-synset-1",
                    ontology_id="fake",
                    ontology_version="1",
                    lemma=base,
                    pos=target.pos,
                    gloss=GlossPayload(definition=f"{base} sense"),
                    sense_label=f"{base}.01",
                ),
            )

    seen = []

    class FakeBackend:
        def descriptor(self):
            return {"type": "fake"}

        def score_target(self, target, candidate_texts):
            seen.append((target.surface_form, target.tokens, target.target_start, target.target_end))
            return {text: 1.0 for text in candidate_texts}

    monkeypatch.setattr(module, "_make_ontology", lambda **kwargs: FakeOntology())
    monkeypatch.setattr(module, "_make_backend", lambda **kwargs: FakeBackend())
    tokens = pd.DataFrame(
        {
            "row_id": [0, 0, 0],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 2],
            "text": ["Participant", "\f", "see"],
            "lemma": ["Participant", "", "see"],
            "pos": ["PROPN", "SPACE", "VERB"],
            "ent_type": ["", "", ""],
        }
    )
    translator = WordSenseDisambiguator(acknowledge_noncommercial_license=True)
    result = translator.translate_batch(
        {"tokens": _packet("tokens", ("row_id", "sentence_id", "token_id"), tokens)},
        mode="translate",
        request=TranslationRequest(),
    )

    assert seen == [
        ("Participant", ("Participant", "see"), 0, 1),
        ("see", ("Participant", "see"), 1, 2),
    ]
    assert result.outputs["senses"]["keys"]["token_id"].tolist() == [0, 2]
    assert result.outputs["unresolved"]["keys"].empty

def test_word_sense_disambiguator_binds_request_to_tokens_source():
    translator = WordSenseDisambiguator(acknowledge_noncommercial_license=True)
    token_source = SimpleNamespace(
        artifact_type=SimpleNamespace(value="table"),
        primary_key=("row_id", "sentence_id", "token_id"),
    )
    request = translator.input_request(
        sources={"tokens": token_source},
        mode="translate",
        request=TranslationRequest(),
    )
    assert set(request) == {"tokens"}
    assert request["tokens"].mode == "full_artifact"


def test_word_sense_disambiguator_requires_explicit_noncommercial_acknowledgement():
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        WordSenseDisambiguator()


def test_wsl_reader_empty_subword_context_is_target_input_error():
    from text_analysis_lab._linguistics.wsd.wsl_reader import build_wsl_reader_input

    class FakeTokenizer:
        unk_token_id = -1

        def __call__(self, value, **kwargs):
            _ = kwargs
            if isinstance(value, list):
                return {"input_ids": [[101], []]}
            return {"input_ids": [101]}

        def convert_tokens_to_ids(self, token):
            return 101

    with pytest.raises(ValueError, match="no subwords"):
        build_wsl_reader_input(
            FakeTokenizer(),
            tokens=("word", "\f"),
            target_start=0,
            target_end=1,
            candidates=("candidate",),
        )
