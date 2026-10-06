from __future__ import annotations

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.linguistics.contracts import validate_linguistic_contracts
from text_analysis_lab.translators import SemanticRoleHeadResolver, SenseSelector
from _boi_linguistic_fixture import (
    CANDIDATE_KEYS,
    ROLE_SPAN_KEYS,
    TOKEN_KEYS,
    build_boi_linguistic_fixture,
)


def _derived_frames():
    fixture = build_boi_linguistic_fixture()
    role_heads = SemanticRoleHeadResolver().translate(
        fixture.role_spans,
        fixture.tokens,
    )["role_heads"]
    senses = SenseSelector().translate(
        fixture.candidates,
        candidate_keys=CANDIDATE_KEYS,
    )["senses"]
    return fixture, role_heads, senses


def test_shared_boi_fixture_satisfies_cross_artifact_contract() -> None:
    fixture, role_heads, senses = _derived_frames()

    validate_linguistic_contracts(
        sentences=fixture.sentences,
        tokens=fixture.tokens,
        predicates=fixture.predicates,
        role_spans=fixture.role_spans,
        role_heads=role_heads,
        candidates=fixture.candidates,
        senses=senses,
        wsd_unresolved=fixture.wsd_unresolved,
        mentions=fixture.mentions,
        srl_failures=fixture.srl_failures,
        coref_failures=fixture.coref_failures,
    )


def test_shared_fixture_locks_semantic_head_examples_for_later_boi_phases() -> None:
    fixture, role_heads, _ = _derived_frames()

    def heads(row_id: int, sentence_id: int, role: str) -> list[tuple[int, str]]:
        rows = role_heads.loc[
            (role_heads["row_id"] == row_id)
            & (role_heads["sentence_id"] == sentence_id)
            & (role_heads["role"] == role)
        ].sort_values("head_id")
        return list(
            zip(
                rows["head_token_id"].astype(int).tolist(),
                rows["head_lemma"].astype(str).tolist(),
                strict=True,
            )
        )

    assert heads(0, 0, "ARG0") == [(0, "John"), (2, "Mary")]
    assert heads(0, 0, "ARG1") == [(4, "bread"), (6, "apple")]
    assert heads(0, 1, "ARG1") == [(5, "apple")]
    assert heads(2, 0, "ARGM-CAU") == [(5, "rain")]

    causal = role_heads.loc[
        (role_heads["row_id"] == 2)
        & (role_heads["sentence_id"] == 0)
        & (role_heads["role"] == "ARGM-CAU")
    ].iloc[0]
    assert causal["rule"] == "causal_thanks_to"
    assert "thanks" in fixture.sentences.loc[
        (fixture.sentences["row_id"] == 2)
        & (fixture.sentences["sentence_id"] == 0),
        "text",
    ].item()


def test_token_id_is_sentence_local_not_globally_unique() -> None:
    fixture, role_heads, senses = _derived_frames()

    token_zero = fixture.tokens.loc[fixture.tokens["token_id"] == 0]
    assert len(token_zero) == len(fixture.sentences)
    assert token_zero[["row_id", "sentence_id", "token_id"]].drop_duplicates().shape[0] == len(
        token_zero
    )

    validate_linguistic_contracts(
        sentences=fixture.sentences,
        tokens=fixture.tokens,
        role_spans=fixture.role_spans,
        role_heads=role_heads,
        candidates=fixture.candidates,
        senses=senses,
        mentions=fixture.mentions,
    )


def test_contract_rejects_candidate_rebound_to_different_existing_token() -> None:
    fixture = build_boi_linguistic_fixture()
    broken = fixture.candidates.copy()
    target = (
        (broken["row_id"] == 2)
        & (broken["sentence_id"] == 1)
        & (broken["token_id"] == 4)
    )
    broken.loc[target, "sentence_id"] = 0

    with pytest.raises(ArtifactError, match="disagrees with token"):
        validate_linguistic_contracts(
            sentences=fixture.sentences,
            tokens=fixture.tokens,
            candidates=broken,
        )


def test_contract_rejects_copied_token_field_drift() -> None:
    fixture = build_boi_linguistic_fixture()
    broken = fixture.candidates.copy()
    broken.loc[broken.index[0], "parser_lemma"] = "not-the-parser-lemma"

    with pytest.raises(ArtifactError, match="disagrees with token"):
        validate_linguistic_contracts(
            sentences=fixture.sentences,
            tokens=fixture.tokens,
            candidates=broken,
        )


def test_contract_rejects_role_span_pointing_outside_sentence_tokens() -> None:
    fixture = build_boi_linguistic_fixture()
    broken = fixture.role_spans.copy()
    broken.loc[broken.index[0], "token_end_id"] = 99

    with pytest.raises(ArtifactError, match="missing token identities"):
        validate_linguistic_contracts(
            sentences=fixture.sentences,
            tokens=fixture.tokens,
            predicates=fixture.predicates,
            role_spans=broken,
        )


def test_contract_rejects_role_head_pointing_to_nonexistent_token() -> None:
    fixture, role_heads, _ = _derived_frames()
    broken = role_heads.copy()
    broken.loc[broken.index[0], "head_token_id"] = 999

    with pytest.raises(ArtifactError, match="head_token_id references missing token"):
        validate_linguistic_contracts(
            sentences=fixture.sentences,
            tokens=fixture.tokens,
            role_spans=fixture.role_spans,
            role_heads=broken,
        )


def test_contract_rejects_resolved_and_unresolved_wsd_overlap() -> None:
    fixture, _, senses = _derived_frames()
    unresolved = fixture.wsd_unresolved.copy()
    resolved_key = senses.iloc[0][list(TOKEN_KEYS)]
    for key in TOKEN_KEYS:
        unresolved.loc[unresolved.index[0], key] = resolved_key[key]

    with pytest.raises(ArtifactError, match="both WSD-resolved and WSD-unresolved"):
        validate_linguistic_contracts(
            sentences=fixture.sentences,
            tokens=fixture.tokens,
            senses=senses,
            wsd_unresolved=unresolved,
        )


def test_failure_states_are_not_flattened_into_generic_nulls() -> None:
    fixture = build_boi_linguistic_fixture()

    unresolved = fixture.wsd_unresolved.iloc[0]
    assert unresolved["reason"] == "no_candidates"
    assert unresolved["surface_form"] == "Green"

    srl_failure = fixture.srl_failures.iloc[0]
    assert srl_failure["reason"] == "cuda_out_of_memory"
    assert isinstance(srl_failure["detail"], str)
    assert srl_failure["detail"]

    coref_failure = fixture.coref_failures.iloc[0]
    assert coref_failure["reason"] == "document_too_long"
    assert int(coref_failure["model_tokens"]) > int(coref_failure["max_model_tokens"])

    punctuation = fixture.tokens.loc[fixture.tokens["pos"] == "PUNCT"]
    observed_wsd_keys = {
        tuple(row)
        for row in pd.concat(
            [
                fixture.candidates[list(TOKEN_KEYS)],
                fixture.wsd_unresolved[list(TOKEN_KEYS)],
            ]
        ).itertuples(index=False, name=None)
    }
    assert all(
        tuple(row) not in observed_wsd_keys
        for row in punctuation[list(TOKEN_KEYS)].itertuples(index=False, name=None)
    )


def test_structural_validation_does_not_require_shared_lineage_metadata() -> None:
    fixture, role_heads, senses = _derived_frames()

    # Ordinary frames carry no lineage at all. Contract validity is determined by
    # stable identities and required data, not by a shared provenance authorization
    # gate. Project-level provenance is tested separately.
    validate_linguistic_contracts(
        sentences=fixture.sentences.copy(),
        tokens=fixture.tokens.copy(),
        predicates=fixture.predicates.copy(),
        role_spans=fixture.role_spans.copy(),
        role_heads=role_heads.copy(),
        candidates=fixture.candidates.copy(),
        senses=senses.copy(),
        mentions=fixture.mentions.copy(),
    )


def test_fixture_primary_key_columns_are_unique() -> None:
    fixture = build_boi_linguistic_fixture()

    assert not fixture.tokens.duplicated(list(TOKEN_KEYS)).any()
    assert not fixture.role_spans.duplicated(list(ROLE_SPAN_KEYS)).any()
    assert not fixture.candidates.duplicated(list(CANDIDATE_KEYS)).any()
