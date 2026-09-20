from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    CoreferenceResolver,
    SemanticRoleLabeler,
    WordSenseDisambiguator,
)


def _enabled() -> bool:
    return os.environ.get("TEAL_RUN_LINGUISTICS_EXTERNAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _write_table(
    project,
    artifact_id,
    label,
    key_frame,
    data_frame,
    *,
    lineage_mode="new_key",
    basis=(),
):
    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=tuple(basis),
    )
    writer.write(
        {
            "keys": key_frame.reset_index(drop=True),
            "data": data_frame.reset_index(drop=True),
        }
    )
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


@pytest.mark.skipif(
    not _enabled(),
    reason="set TEAL_RUN_LINGUISTICS_EXTERNAL=1 to allow real Unit 10 model/lexicon downloads",
)
def test_real_unit10_models_on_one_sentence(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("fastcoref")
    wn = pytest.importorskip("wn")
    try:
        wn.Wordnet("oewn:2025+")
    except Exception:  # noqa: BLE001 - wn uses provider-specific missing-resource errors
        wn.download("oewn:2025+")

    text = "Alice runs quickly. She likes the race."
    project = teal.Project.create(tmp_path / "project", name="unit10_live")
    try:
        documents = _write_table(
            project,
            "art_docs",
            "documents",
            pd.DataFrame({"row_id": [0]}),
            pd.DataFrame({"text": [text]}),
        )
        sentences = _write_table(
            project,
            "art_sentences",
            "sentences",
            pd.DataFrame({"row_id": [0, 0], "sentence_id": [0, 1]}),
            pd.DataFrame(
                {
                    "text": ["Alice runs quickly.", "She likes the race."],
                    "char_start": [0, 20],
                    "char_end": [19, len(text)],
                }
            ),
            lineage_mode="extended_key",
            basis=(documents.artifact_id,),
        )
        token_rows = pd.DataFrame(
            {
                "row_id": [0] * 9,
                "sentence_id": [0, 0, 0, 0, 1, 1, 1, 1, 1],
                "token_id": [0, 1, 2, 3, 0, 1, 2, 3, 4],
            }
        )
        token_data = pd.DataFrame(
            {
                "text": [
                    "Alice",
                    "runs",
                    "quickly",
                    ".",
                    "She",
                    "likes",
                    "the",
                    "race",
                    ".",
                ],
                "lemma": [
                    "Alice",
                    "run",
                    "quickly",
                    ".",
                    "she",
                    "like",
                    "the",
                    "race",
                    ".",
                ],
                "pos": [
                    "PROPN",
                    "VERB",
                    "ADV",
                    "PUNCT",
                    "PRON",
                    "VERB",
                    "DET",
                    "NOUN",
                    "PUNCT",
                ],
                "tag": ["NNP", "VBZ", "RB", ".", "PRP", "VBZ", "DT", "NN", "."],
                "dep": [
                    "nsubj",
                    "ROOT",
                    "advmod",
                    "punct",
                    "nsubj",
                    "ROOT",
                    "det",
                    "dobj",
                    "punct",
                ],
                "head_token_id": [1, 1, 1, 1, 1, 1, 3, 1, 1],
                "ent_type": ["PERSON", "", "", "", "", "", "", "", ""],
                "char_start": [0, 6, 11, 18, 20, 24, 30, 34, 38],
                "char_end": [5, 10, 18, 19, 23, 29, 33, 38, 39],
            }
        )
        tokens = _write_table(
            project,
            "art_tokens",
            "tokens",
            token_rows,
            token_data,
            lineage_mode="extended_key",
            basis=(sentences.artifact_id,),
        )

        coref = project.translate(
            CoreferenceResolver(model="fcoref", device="cpu"),
            {"documents": documents, "tokens": tokens},
        )
        assert set(coref) == {"mentions", "failures"}

        srl = project.translate(
            SemanticRoleLabeler(device="cpu", mixed_precision=False),
            {"sentences": sentences, "tokens": tokens},
        )
        assert srl["predicates"].n_rows >= 2
        assert srl["roles"].n_rows >= 2

        tiny_tokens = project.subset(
            tokens,
            lambda frame: (frame["row_id"] == 0) & (frame["sentence_id"] == 0),
            key_columns=True,
            data_columns=True,
        )["output"]
        wsd = project.translate(
            WordSenseDisambiguator(
                device="cpu",
                include_multiword_candidates=False,
                acknowledge_noncommercial_license=True,
            ),
            {"tokens": tiny_tokens},
        )
        assert wsd["candidates"].n_rows > 0
        assert wsd["senses"].n_rows > 0
    finally:
        project.close()


@pytest.mark.skipif(
    not _enabled(),
    reason="set TEAL_RUN_LINGUISTICS_EXTERNAL=1 to allow real Unit 10 model/lexicon downloads",
)
def test_real_wsd_ignores_spacy_space_token_in_reader_context(tmp_path: Path) -> None:
    """Regression for AERA title/form-feed/abstract tokenization.

    spaCy can emit a SPACE token for the form-feed separator. The WSL reader must
    ignore that context token rather than abort because SentencePiece produces no
    subwords for it.
    """
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    wn = pytest.importorskip("wn")
    try:
        wn.Wordnet("oewn:2025+")
    except Exception:  # noqa: BLE001 - wn uses provider-specific missing-resource errors
        wn.download("oewn:2025+")

    project = teal.Project.create(
        tmp_path / "project_space_wsd", name="unit10_space_wsd"
    )
    try:
        tokens = _write_table(
            project,
            "art_space_tokens",
            "tokens",
            pd.DataFrame(
                {
                    "row_id": [0] * 7,
                    "sentence_id": [0] * 7,
                    "token_id": list(range(7)),
                }
            ),
            pd.DataFrame(
                {
                    "text": [
                        "Participant",
                        "\f",
                        "(",
                        "see",
                        "symposium",
                        "abstract",
                        ")",
                    ],
                    "lemma": [
                        "Participant",
                        "",
                        "(",
                        "see",
                        "symposium",
                        "abstract",
                        ")",
                    ],
                    "pos": ["PROPN", "SPACE", "PUNCT", "VERB", "NOUN", "ADJ", "PUNCT"],
                    "ent_type": [""] * 7,
                }
            ),
        )
        wsd = project.translate(
            WordSenseDisambiguator(
                device="cpu",
                include_multiword_candidates=False,
                acknowledge_noncommercial_license=True,
            ),
            {"tokens": tokens},
        )
        resolved = wsd["senses"].query(
            key_columns=True,
            data_columns=["surface_form"],
            metadata_columns=False,
            metadata_mode="none",
            form="table",
        )
        assert 1 not in set(resolved["token_id"].astype(int))
        assert set(resolved["surface_form"]).intersection(
            {"see", "symposium", "abstract"}
        )
    finally:
        project.close()
