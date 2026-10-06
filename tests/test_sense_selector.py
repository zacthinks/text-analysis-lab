from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators.sense_selector import SenseSelector
from text_analysis_lab.translators.word_sense_disambiguator import SENSE_DATA_COLUMNS


TOKEN_KEYS = ["row_id", "sentence_id", "token_id"]
CANDIDATE_KEYS = [*TOKEN_KEYS, "candidate_id"]


def _candidates() -> pd.DataFrame:
    rows = []
    for token_id, surface, lemma, senses in [
        (
            0,
            "Banks",
            "bank",
            [
                ("bank-finance", "bank.n.01", "bank", 0.8, 1, True),
                ("bank-river", "bank.n.02", "bank", 0.2, 2, False),
            ],
        ),
        (
            1,
            "runs",
            "run",
            [
                ("run-move", "run.v.01", "run", 0.7, 1, True),
                ("run-operate", "run.v.02", "run", 0.3, 2, False),
            ],
        ),
    ]:
        for candidate_id, (sense_id, sense_label, candidate_lemma, score, rank, selected) in enumerate(senses):
            rows.append(
                {
                    "row_id": 4,
                    "sentence_id": 0,
                    "token_id": token_id,
                    "candidate_id": candidate_id,
                    "surface_form": surface,
                    "parser_lemma": lemma,
                    "pos": "NOUN" if token_id == 0 else "VERB",
                    "sense_id": sense_id,
                    "synset_id": f"{sense_id}-synset",
                    "sense_label": sense_label,
                    "candidate_lemma": candidate_lemma,
                    "aliases": json.dumps([candidate_lemma]),
                    "ili": None,
                    "ontology_id": "fake",
                    "ontology_version": "1",
                    "gloss_text": f"gloss for {sense_id}",
                    "normalized_score": score,
                    "rank": rank,
                    "selected": selected,
                    "top1_margin": 0.6 if token_id == 0 else 0.4,
                    "candidate_kind": "singleword",
                    "model_name": "fake-model",
                    "model_revision": "fake-revision",
                }
            )
    return pd.DataFrame.from_records(rows)


def _packet(frame: pd.DataFrame) -> InputBatch:
    return InputBatch(
        source_label="candidates",
        artifact_id="art_candidates",
        primary_key=tuple(CANDIDATE_KEYS),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_sense_selector_default_reproduces_model_selected_senses() -> None:
    candidates = _candidates()
    selector = SenseSelector()

    direct = selector.translate(candidates, candidate_keys=CANDIDATE_KEYS)["senses"]

    assert direct["sense_id"].tolist() == ["bank-finance", "run-move"]
    assert direct["model_selected"].tolist() == [True, True]
    assert direct["selection_reason"].tolist() == ["top_ranked", "top_ranked"]

    expected = []
    for _, row in candidates[candidates["selected"]].sort_values(TOKEN_KEYS).iterrows():
        expected.append(
            {
                "surface_form": row["surface_form"],
                "parser_lemma": row["parser_lemma"],
                "resolved_lemma": row["candidate_lemma"],
                "lemma_overrides_parser": False,
                "pos": row["pos"],
                "sense_id": row["sense_id"],
                "synset_id": row["synset_id"],
                "sense_label": row["sense_label"],
                "aliases": row["aliases"],
                "ili": row["ili"],
                "ontology_id": row["ontology_id"],
                "ontology_version": row["ontology_version"],
                "gloss_text": row["gloss_text"],
                "normalized_score": row["normalized_score"],
                "top1_margin": row["top1_margin"],
                "candidate_kind": row["candidate_kind"],
                "model_name": row["model_name"],
                "model_revision": row["model_revision"],
            }
        )
    expected_frame = pd.DataFrame.from_records(expected, columns=list(SENSE_DATA_COLUMNS))
    pd.testing.assert_frame_equal(
        direct[list(SENSE_DATA_COLUMNS)].reset_index(drop=True),
        expected_frame.reset_index(drop=True),
        check_dtype=False,
    )


def test_sense_selector_supports_exclusion_restrictions_and_forced_tokens() -> None:
    candidates = _candidates()

    excluded = SenseSelector(excluded_sense_ids=["bank-finance"]).translate(
        candidates, candidate_keys=CANDIDATE_KEYS
    )["senses"]
    assert excluded.loc[excluded["token_id"] == 0, "sense_id"].item() == "bank-river"
    assert (
        excluded.loc[excluded["token_id"] == 0, "selection_reason"].item()
        == "top_ranked_after_exclusion"
    )
    assert not excluded.loc[excluded["token_id"] == 0, "model_selected"].item()

    lemma_restricted = SenseSelector(
        allowed_sense_ids_by_lemma={"bank": ["bank-river"]}
    ).translate(candidates, candidate_keys=CANDIDATE_KEYS)["senses"]
    assert lemma_restricted.loc[
        lemma_restricted["token_id"] == 0, "sense_id"
    ].item() == "bank-river"
    assert lemma_restricted.loc[
        lemma_restricted["token_id"] == 0, "selection_reason"
    ].item() == "lemma_restriction"

    token_key = {"row_id": 4, "sentence_id": 0, "token_id": 1}
    token_restricted = SenseSelector(
        allowed_sense_ids_by_token=[
            {"key": token_key, "sense_ids": ["run-operate"]}
        ]
    ).translate(candidates, candidate_keys=CANDIDATE_KEYS)["senses"]
    assert token_restricted.loc[
        token_restricted["token_id"] == 1, "sense_id"
    ].item() == "run-operate"

    forced = SenseSelector(
        excluded_sense_ids=["run-operate"],
        forced_sense_ids_by_token=[
            {"key": token_key, "sense_id": "run-operate"}
        ],
    ).translate(candidates, candidate_keys=CANDIDATE_KEYS)["senses"]
    forced_row = forced.loc[forced["token_id"] == 1].iloc[0]
    assert forced_row["sense_id"] == "run-operate"
    assert forced_row["selection_reason"] == "forced_token"
    assert not bool(forced_row["model_selected"])


def test_sense_selector_drops_tokens_when_filters_remove_all_candidates() -> None:
    selected = SenseSelector(
        allowed_sense_ids_by_lemma={"bank": ["not-present"]}
    ).translate(_candidates(), candidate_keys=CANDIDATE_KEYS)["senses"]
    assert selected["token_id"].tolist() == [1]


def test_sense_selector_rejects_missing_forced_candidate() -> None:
    selector = SenseSelector(
        forced_sense_ids_by_token=[
            {
                "key": {"row_id": 4, "sentence_id": 0, "token_id": 0},
                "sense_id": "missing-sense",
            }
        ]
    )
    with pytest.raises(ArtifactError, match="not present among candidates"):
        selector.translate(_candidates(), candidate_keys=CANDIDATE_KEYS)


def test_sense_selector_standalone_matches_teal_batch() -> None:
    candidates = _candidates()
    selector = SenseSelector(
        excluded_sense_ids=["bank-finance"],
        forced_sense_ids_by_token=[
            {
                "key": {"row_id": 4, "sentence_id": 0, "token_id": 1},
                "sense_id": "run-operate",
            }
        ],
    )
    direct = selector.translate(candidates, candidate_keys=CANDIDATE_KEYS)["senses"]

    batch = selector.translate_batch(
        {"candidates": _packet(candidates)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["senses"]
    combined = pd.concat([batch["keys"], batch["data"]], axis=1)

    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True),
        direct.reset_index(drop=True),
    )


def test_sense_selector_serialization_round_trip_and_output_contract() -> None:
    selector = SenseSelector(
        excluded_sense_ids=["sense-z"],
        allowed_sense_ids_by_lemma={"Bank": ["sense-b", "sense-a"]},
        allowed_sense_ids_by_token=[
            {
                "key": {"row_id": 4, "sentence_id": 0, "token_id": 0},
                "sense_ids": ["sense-b"],
            }
        ],
        forced_sense_ids_by_token=[
            {
                "key": {"row_id": 4, "sentence_id": 0, "token_id": 1},
                "sense_id": "sense-c",
            }
        ],
    )
    state = json.loads(json.dumps(selector.to_json_state()))
    restored = SenseSelector.from_json_state(state)
    assert restored.to_json_state() == selector.to_json_state()

    source = SimpleNamespace(
        artifact_type=SimpleNamespace(value="table"),
        primary_key=tuple(CANDIDATE_KEYS),
    )
    spec = selector.output_specs(
        sources={"candidates": source},
        request=TranslationRequest(),
    )["senses"]
    assert spec.lineage_mode == "reduced_key"
    assert spec.basis_labels == ("candidates",)
