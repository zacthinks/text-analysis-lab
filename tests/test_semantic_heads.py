from __future__ import annotations

import pandas as pd

from text_analysis_lab.linguistics.heads import (
    SemanticHeadRules,
    resolve_semantic_head_indices,
)
from text_analysis_lab.linguistics.srl.structures import content_head_indices
from text_analysis_lab.translators.coreference_resolver import CoreferenceResolver
from text_analysis_lab.translators.semantic_role_head_resolver import (
    SemanticRoleHeadResolver,
)


def _quantifier_tokens() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0] * 7,
            "sentence_id": [0] * 7,
            "token_id": list(range(7)),
            "text": ["John", "likes", "all", "of", "the", "apples", "."],
            "lemma": ["John", "like", "all", "of", "the", "apple", "."],
            "pos": ["PROPN", "VERB", "PRON", "ADP", "DET", "NOUN", "PUNCT"],
            "dep": ["nsubj", "ROOT", "dobj", "prep", "det", "pobj", "punct"],
            "head_token_id": [1, 1, 1, 2, 5, 3, 1],
            "ent_type": ["", "", "", "", "", "", ""],
        }
    )


def _quantifier_span() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0],
            "sentence_id": [0],
            "predicate_id": [0],
            "role_id": [0],
            "role": ["ARG1"],
            "token_start_id": [2],
            "token_end_id": [6],
        }
    )


def test_default_rules_reproduce_quantifier_normalization() -> None:
    tokens = _quantifier_tokens()
    kwargs = {
        "start": 2,
        "end": 6,
        "token_ids": tokens["token_id"].tolist(),
        "head_token_ids": tokens["head_token_id"].tolist(),
        "dependencies": tokens["dep"].tolist(),
        "pos": tokens["pos"].tolist(),
        "text": tokens["text"].tolist(),
        "role": "ARG1",
    }

    assert resolve_semantic_head_indices(**kwargs) == (5,)
    assert content_head_indices(**kwargs) == (5,)


def test_head_policy_override_changes_heads_without_touching_raw_span(
    monkeypatch,
) -> None:
    import text_analysis_lab.translators.semantic_role_labeler as srl_module

    monkeypatch.setattr(
        srl_module,
        "_make_runtime",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("SRL runtime must not be invoked by head resolution")
        ),
    )
    tokens = _quantifier_tokens()
    spans = _quantifier_span()
    original = spans.copy(deep=True)

    default = SemanticRoleHeadResolver().translate(
        spans,
        tokens,
        sentence_keys=["row_id", "sentence_id"],
    )["role_heads"]
    overridden = SemanticRoleHeadResolver(
        head_rules=SemanticHeadRules(
            rewrite_quantifier_links=False,
            move_off_function_pos=False,
        )
    ).translate(
        spans,
        tokens,
        sentence_keys=["row_id", "sentence_id"],
    )["role_heads"]

    assert default["head_text"].tolist() == ["apples"]
    assert overridden["head_text"].tolist() == ["all"]
    pd.testing.assert_frame_equal(spans, original)


def test_coordination_preserves_each_content_head() -> None:
    token_ids = list(range(8))
    text = ["John", "and", "Mary", "ate", "bread", "and", "apples", "."]
    pos = ["PROPN", "CCONJ", "PROPN", "VERB", "NOUN", "CCONJ", "NOUN", "PUNCT"]
    dep = ["nsubj", "cc", "conj", "ROOT", "dobj", "cc", "conj", "punct"]
    heads = [3, 2, 0, 3, 3, 6, 4, 3]

    subject = resolve_semantic_head_indices(
        start=0,
        end=3,
        token_ids=token_ids,
        head_token_ids=heads,
        dependencies=dep,
        pos=pos,
        text=text,
        role="ARG0",
    )
    object_ = resolve_semantic_head_indices(
        start=4,
        end=7,
        token_ids=token_ids,
        head_token_ids=heads,
        dependencies=dep,
        pos=pos,
        text=text,
        role="ARG1",
    )

    assert subject == (0, 2)
    assert object_ == (4, 6)


def test_semantic_head_rules_and_translators_round_trip_json_state() -> None:
    rules = SemanticHeadRules(
        function_pos=("DET", "ADP"),
        move_off_function_pos=False,
        quantifiers=("all", "each"),
        quantifier_links=("of",),
        rewrite_quantifier_links=False,
        preserve_coordination=False,
        coordination_excluded_roles=("V", "ARGM"),
    )
    assert SemanticHeadRules.from_dict(rules.to_dict()) == rules

    resolver = SemanticRoleHeadResolver(
        sentence_key="sent",
        token_key="tok",
        head_rules=rules,
    )
    restored_resolver = SemanticRoleHeadResolver.from_json_state(
        resolver.to_json_state()
    )
    assert restored_resolver.to_json_state() == resolver.to_json_state()
    assert restored_resolver.head_rules == rules

    coref = CoreferenceResolver(
        sentence_key="sent",
        token_key="tok",
        head_rules=rules,
    )
    restored_coref = CoreferenceResolver.from_json_state(coref.to_json_state())
    assert restored_coref.to_json_state() == coref.to_json_state()
    assert restored_coref.head_rules == rules
