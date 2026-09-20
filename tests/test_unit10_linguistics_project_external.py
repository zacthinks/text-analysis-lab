from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    CoreferenceResolver,
    SemanticRoleLabeler,
    WordSenseDisambiguator,
)


def _write_table(project, artifact_id, label, key_frame, data_frame, *, lineage_mode="new_key", basis=()):
    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=tuple(basis),
    )
    writer.write({"keys": key_frame.reset_index(drop=True), "data": data_frame.reset_index(drop=True)})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=label,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=tuple(basis),
    )
    return project.get_artifact(artifact_id)


def _frame(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        include_position=False,
        form="table",
    )


@dataclass
class _CachePaths:
    root: Path
    models: Path
    huggingface_hub: Path
    wsd_gloss_text: Path
    wordnet: Path
    wordnet_mwe_indices: Path


def test_unit10_translators_write_teal_native_lineage(monkeypatch, tmp_path: Path) -> None:
    import text_analysis_lab.translators.coreference_resolver as coref_module
    import text_analysis_lab.translators.semantic_role_labeler as srl_module
    import text_analysis_lab.translators.word_sense_disambiguator as wsd_module
    from text_analysis_lab._linguistics.coreference.runtime import (
        CorefBatchPrediction,
        CorefMention,
        CorefPrediction,
    )
    from text_analysis_lab._linguistics.srl.runtime import SrlTokenPrediction
    from text_analysis_lab._linguistics.srl.structures import BioSpan
    from text_analysis_lab._linguistics.wsd.types import GlossPayload, SenseCandidate

    class FakeCorefRuntime:
        def document_token_counts(self, texts):
            return [len(texts[0].split())], None

        def predict_texts(self, texts, **kwargs):
            _ = kwargs
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

    class FakeSrlRuntime:
        def encode_tokens(self, tokens):
            return tuple(tokens)

        def predict_encoded_batch(self, instances):
            rows = []
            for tokens, predicate in instances:
                tags = tuple("B-V" if i == predicate else ("B-ARG0" if i == 0 else "O") for i in range(len(tokens)))
                spans = [BioSpan("ARG0", 0, 1), BioSpan("V", predicate, predicate + 1)]
                rows.append(
                    SrlTokenPrediction(
                        tokens=tuple(tokens),
                        predicate_index=predicate,
                        predicate=tokens[predicate],
                        wordpieces=tuple(tokens),
                        input_ids=tuple(range(len(tokens))),
                        predicate_indicator=tuple(1 if i == predicate else 0 for i in range(len(tokens))),
                        wordpiece_offsets=tuple(range(len(tokens))),
                        wordpiece_tags=tags,
                        raw_tags=tags,
                        tags=tags,
                        bio_repairs=(),
                        word_scores=tuple(0.9 for _ in tokens),
                        spans=tuple(spans),
                        description="",
                    )
                )
            return tuple(rows)

    cache = _CachePaths(
        root=tmp_path / "cache",
        models=tmp_path / "cache" / "models",
        huggingface_hub=tmp_path / "cache" / "hf",
        wsd_gloss_text=tmp_path / "cache" / "gloss",
        wordnet=tmp_path / "cache" / "wn",
        wordnet_mwe_indices=tmp_path / "cache" / "mwe",
    )
    for value in cache.__dict__.values():
        value.mkdir(parents=True, exist_ok=True)

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
            _ = target
            return {text: (0.8 if index == 0 else 0.2) for index, text in enumerate(candidate_texts)}

    monkeypatch.setattr(coref_module, "_make_runtime", lambda **kwargs: FakeCorefRuntime())
    monkeypatch.setattr(srl_module, "_make_runtime", lambda **kwargs: FakeSrlRuntime())
    monkeypatch.setattr(wsd_module, "user_cache_paths", lambda: cache)
    monkeypatch.setattr(wsd_module, "_make_ontology", lambda **kwargs: FakeOntology())
    monkeypatch.setattr(wsd_module, "_make_backend", lambda **kwargs: FakeBackend())

    project = teal.Project.create(tmp_path / "project", name="unit10_artifacts")
    try:
        documents = _write_table(
            project,
            "art_docs",
            "documents",
            pd.DataFrame({"row_id": [7]}),
            pd.DataFrame({"text": ["Alice left. She slept."]}),
        )
        sentences = _write_table(
            project,
            "art_sentences",
            "sentences",
            pd.DataFrame({"row_id": [7, 7], "sentence_id": [0, 1]}),
            pd.DataFrame(
                {
                    "text": ["Alice left.", "She slept."],
                    "char_start": [0, 12],
                    "char_end": [11, 22],
                }
            ),
            lineage_mode="extended_key",
            basis=(documents.artifact_id,),
        )
        tokens = _write_table(
            project,
            "art_tokens",
            "tokens",
            pd.DataFrame(
                {
                    "row_id": [7] * 6,
                    "sentence_id": [0, 0, 0, 1, 1, 1],
                    "token_id": [0, 1, 2, 0, 1, 2],
                }
            ),
            pd.DataFrame(
                {
                    "text": ["Alice", "left", ".", "She", "slept", "."],
                    "lemma": ["Alice", "leave", ".", "she", "sleep", "."],
                    "pos": ["PROPN", "VERB", "PUNCT", "PRON", "VERB", "PUNCT"],
                    "tag": ["NNP", "VBD", ".", "PRP", "VBD", "."],
                    "dep": ["nsubj", "ROOT", "punct", "nsubj", "ROOT", "punct"],
                    "head_token_id": [1, 1, 1, 1, 1, 1],
                    "ent_type": ["PERSON", "", "", "", "", ""],
                    "char_start": [0, 6, 10, 12, 16, 21],
                    "char_end": [5, 10, 11, 15, 21, 22],
                }
            ),
            lineage_mode="extended_key",
            basis=(sentences.artifact_id,),
        )

        coref = project.translate(
            CoreferenceResolver(),
            {"documents": documents, "tokens": tokens},
        )
        assert coref["mentions"].descriptor["lineage"]["basis_artifact_ids"] == [documents.artifact_id]
        assert _frame(coref["mentions"])["text"].tolist() == ["Alice", "She"]
        assert _frame(coref["failures"]).empty

        srl = project.translate(
            SemanticRoleLabeler(mixed_precision=False),
            {"sentences": sentences, "tokens": tokens},
        )
        assert srl["predicates"].descriptor["lineage"]["basis_artifact_ids"] == [sentences.artifact_id]
        assert srl["roles"].descriptor["lineage"]["basis_artifact_ids"] == [srl["predicates"].artifact_id]
        assert set(_frame(srl["roles"])["role"]) >= {"ARG0", "V"}
        assert _frame(srl["failures"]).empty

        wsd = project.translate(
            WordSenseDisambiguator(
                include_multiword_candidates=False,
                acknowledge_noncommercial_license=True,
            ),
            {"tokens": tokens},
        )
        assert wsd["senses"].descriptor["lineage"]["basis_artifact_ids"] == [tokens.artifact_id]
        assert wsd["candidates"].descriptor["lineage"]["basis_artifact_ids"] == [tokens.artifact_id]
        sense_frame = _frame(wsd["senses"])
        assert set(sense_frame["surface_form"].tolist()) == {"Alice", "left", "slept"}
        assert "She" not in sense_frame["surface_form"].tolist()
        assert _frame(wsd["unresolved"]).empty
    finally:
        project.close()
