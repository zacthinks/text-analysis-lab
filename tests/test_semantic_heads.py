from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from text_analysis_lab.core.operator import BaseOperator
from text_analysis_lab.core.pipeline import Pipeline
from text_analysis_lab.linguistics.heads import (
    SemanticHeadRules,
    export_default_semantic_head_rules,
    get_default_semantic_head_rules,
    load_semantic_head_rules,
    resolve_semantic_head_indices,
    resolve_semantic_heads,
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


def _resolve(
    *,
    text,
    lemmas,
    pos,
    dep,
    heads,
    start=0,
    end=None,
    role=None,
    source="srl",
    rules=None,
):
    end = len(text) if end is None else end
    return resolve_semantic_head_indices(
        start=start,
        end=end,
        token_ids=list(range(len(text))),
        head_token_ids=heads,
        dependencies=dep,
        pos=pos,
        text=text,
        lemmas=lemmas,
        role=role,
        source=source,
        rules=rules,
    )


def test_default_rule_document_matches_stabilized_boi_contract() -> None:
    document = get_default_semantic_head_rules()

    assert document["schema_version"] == 1
    assert [rule["id"] for rule in document["rules"]] == [
        "causal_thanks_to",
        "causal_owing_to",
        "causal_due_to",
        "causal_because_of",
        "causal_result_of",
        "manner_by_means_of",
        "quantifier_of_object",
        "collection_quantifier_of_object",
        "preposition_object",
    ]
    assert document["rules"][0] == {
        "id": "causal_thanks_to",
        "when": {"role": ["ARGM-CAU"], "lemma": ["thank", "thanks"]},
        "move": [
            {"within": {"lemma": ["to"]}},
            {"child": {"dep": ["pobj", "obj", "obl"]}},
        ],
    }


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

    assert resolve_semantic_head_indices(
        **kwargs,
        lemmas=tokens["lemma"].tolist(),
        source="srl",
    ) == (5,)
    assert content_head_indices(**kwargs) == (5,)


def test_role_aware_boi_rule_moves_only_with_matching_role() -> None:
    common = dict(
        text=["thanks", "to", "rain"],
        lemmas=["thanks", "to", "rain"],
        pos=["NOUN", "ADP", "NOUN"],
        dep=["advmod", "prep", "pobj"],
        heads=[3, 0, 1],
    )

    assert _resolve(**common, role="ARGM-CAU") == (2,)
    assert _resolve(**common, role="ARGM-ADV") == (0,)
    assert _resolve(**common, role=None, source="coreference") == (0,)


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

    custom = get_default_semantic_head_rules()
    custom["name"] = "no-quantifier-rewrite"
    custom["rules"] = [
        rule
        for rule in custom["rules"]
        if rule["id"] != "quantifier_of_object"
    ]
    overridden = SemanticRoleHeadResolver(head_rules=custom).translate(
        spans,
        tokens,
        sentence_keys=["row_id", "sentence_id"],
    )["role_heads"]

    assert default["head_text"].tolist() == ["apples"]
    assert overridden["head_text"].tolist() == ["all"]
    pd.testing.assert_frame_equal(spans, original)


def test_json_file_override_uses_original_boi_rule_shape(tmp_path) -> None:
    custom = get_default_semantic_head_rules()
    custom["name"] = "custom-according-to-complement"
    custom["rules"].insert(
        0,
        {
            "id": "according_to_complement",
            "when": {"lemma": ["accord"]},
            "move": [
                {"within": {"lemma": ["to"]}},
                {"child": {"dep": ["pobj", "obj", "obl"]}},
            ],
        },
    )
    path = tmp_path / "semantic_head_rules.json"
    path.write_text(json.dumps(custom), encoding="utf-8")

    loaded = load_semantic_head_rules(path)
    assert loaded.to_dict() == custom

    assert _resolve(
        text=["According", "to", "report"],
        lemmas=["accord", "to", "report"],
        pos=["VERB", "ADP", "NOUN"],
        dep=["advmod", "prep", "pobj"],
        heads=[3, 0, 1],
        role="ARGM-ADV",
        rules=path,
    ) == (2,)


def test_rule_conditions_support_any_not_and_and_or_semantics() -> None:
    rules = {
        "schema_version": 1,
        "name": "boolean-conditions",
        "description": "",
        "rules": [
            {
                "id": "allowed_a_or_x",
                "when": {
                    "any": [{"lemma": ["a"]}, {"text": ["X"]}],
                    "not": {"role": ["BLOCKED"]},
                    "pos": ["NOUN", "PROPN"],
                },
                "move": [{"child": {"lemma": ["b", "bee"]}}],
            }
        ],
    }
    common = dict(
        text=["A", "B"],
        lemmas=["a", "b"],
        pos=["NOUN", "NOUN"],
        dep=["ROOT", "dobj"],
        heads=[0, 0],
        rules=rules,
    )

    assert _resolve(**common, role="ARG1") == (1,)
    assert _resolve(**common, role="BLOCKED") == (0,)


def test_move_path_is_atomic_when_later_step_fails() -> None:
    rules = {
        "schema_version": 1,
        "name": "atomic-path",
        "description": "",
        "rules": [
            {
                "id": "incomplete",
                "when": {"lemma": ["a"]},
                "move": [
                    {"child": {"lemma": ["b"]}},
                    {"child": {"lemma": ["missing"]}},
                ],
            }
        ],
    }

    assert _resolve(
        text=["a", "b"],
        lemmas=["a", "b"],
        pos=["NOUN", "NOUN"],
        dep=["ROOT", "dobj"],
        heads=[0, 0],
        role="ARG1",
        rules=rules,
    ) == (0,)


def test_successful_move_restarts_rule_evaluation_from_first_rule() -> None:
    rules = {
        "schema_version": 1,
        "name": "restart",
        "description": "",
        "rules": [
            {
                "id": "b_to_c",
                "when": {"lemma": ["b"]},
                "move": [{"child": {"lemma": ["c"]}}],
            },
            {
                "id": "a_to_b",
                "when": {"lemma": ["a"]},
                "move": [{"child": {"lemma": ["b"]}}],
            },
        ],
    }

    assert _resolve(
        text=["a", "b", "c"],
        lemmas=["a", "b", "c"],
        pos=["NOUN", "NOUN", "NOUN"],
        dep=["ROOT", "dobj", "compound"],
        heads=[0, 0, 1],
        role="ARG1",
        rules=rules,
    ) == (2,)


def test_parent_and_within_moves_are_supported() -> None:
    parent_rules = {
        "schema_version": 1,
        "name": "parent",
        "description": "",
        "rules": [
            {
                "id": "child_parent_within",
                "when": {"lemma": ["a"]},
                "move": [
                    {"child": {"lemma": ["b"]}},
                    {"parent": {"lemma": ["a"]}},
                    {"within": {"lemma": ["c"]}},
                ],
            }
        ],
    }
    assert _resolve(
        text=["a", "b", "c"],
        lemmas=["a", "b", "c"],
        pos=["NOUN", "NOUN", "NOUN"],
        dep=["ROOT", "dobj", "appos"],
        heads=[0, 0, 0],
        role="ARG1",
        rules=parent_rules,
    ) == (2,)

    within_rules = {
        "schema_version": 1,
        "name": "within",
        "description": "",
        "rules": [
            {
                "id": "find_marker",
                "when": {"lemma": ["a"]},
                "move": [{"within": {"lemma": ["marker"]}}],
            }
        ],
    }
    assert _resolve(
        text=["a", "marker", "x"],
        lemmas=["a", "marker", "x"],
        pos=["NOUN", "ADP", "NOUN"],
        dep=["ROOT", "prep", "pobj"],
        heads=[0, 0, 1],
        role="ARG1",
        rules=within_rules,
    ) == (1,)


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
        lemmas=[value.casefold() for value in text],
        role="ARG0",
        source="srl",
    )
    object_ = resolve_semantic_head_indices(
        start=4,
        end=7,
        token_ids=token_ids,
        head_token_ids=heads,
        dependencies=dep,
        pos=pos,
        text=text,
        lemmas=[value.casefold() for value in text],
        role="ARG1",
        source="srl",
    )

    assert subject == (0, 2)
    assert object_ == (4, 6)



def test_coordination_branches_before_rule_moves_can_skip_intermediate_heads() -> None:
    rules = {
        "schema_version": 1,
        "name": "preposition-coordination",
        "description": "",
        "rules": [
            {
                "id": "preposition_object",
                "when": {"pos": ["ADP"]},
                "move": [{"child": {"dep": ["pobj"]}}],
            }
        ],
    }

    heads = _resolve(
        text=["from", "Boston", "and", "from", "New York"],
        lemmas=["from", "Boston", "and", "from", "New York"],
        pos=["ADP", "PROPN", "CCONJ", "ADP", "PROPN"],
        dep=["ROOT", "pobj", "cc", "conj", "pobj"],
        heads=[0, 0, 3, 0, 3],
        role="ARGM-LOC",
        rules=rules,
    )

    assert heads == (1, 4)


def test_predicate_role_does_not_expand_coordinated_heads() -> None:
    assert _resolve(
        text=["eat", "drink"],
        lemmas=["eat", "drink"],
        pos=["VERB", "VERB"],
        dep=["ROOT", "conj"],
        heads=[0, 0],
        role="V",
        rules={
            "schema_version": 1,
            "name": "no-moves",
            "description": "",
            "rules": [],
        },
    ) == (0,)


def test_legacy_content_head_wrapper_preserves_function_word_fallback() -> None:
    assert content_head_indices(
        start=0,
        end=2,
        token_ids=[0, 1],
        head_token_ids=[0, 0],
        dependencies=["ROOT", "det"],
        pos=["DET", "NOUN"],
        text=["the", "dog"],
        role="ARG1",
    ) == (1,)


def test_legacy_content_head_wrapper_preserves_predicate_coordination_rule() -> None:
    common = {
        "start": 0,
        "end": 2,
        "token_ids": [0, 1],
        "head_token_ids": [0, 0],
        "dependencies": ["ROOT", "conj"],
        "pos": ["VERB", "VERB"],
        "text": ["eat", "drink"],
    }

    assert content_head_indices(**common, role="V") == (0,)
    assert content_head_indices(**common, role="ARG1") == (0, 1)


def test_standalone_resolver_infers_sentence_keys() -> None:
    resolved = SemanticRoleHeadResolver().translate(
        _quantifier_span(),
        _quantifier_tokens(),
    )["role_heads"]

    assert resolved[
        ["row_id", "sentence_id", "predicate_id", "role_id", "head_id"]
    ].to_dict("records") == [
        {
            "row_id": 0,
            "sentence_id": 0,
            "predicate_id": 0,
            "role_id": 0,
            "head_id": 0,
        }
    ]
    assert resolved["head_text"].tolist() == ["apples"]


def test_standalone_resolver_rejects_ambiguous_key_layout() -> None:
    spans = _quantifier_span()[
        [
            "row_id",
            "sentence_id",
            "role",
            "predicate_id",
            "role_id",
            "token_start_id",
            "token_end_id",
        ]
    ]

    with pytest.raises(ValueError, match="expected role-span key columns"):
        SemanticRoleHeadResolver().translate(spans, _quantifier_tokens())


def test_semantic_head_resolver_executes_inside_native_pipeline() -> None:
    pipeline = Pipeline(inputs=("role_spans", "tokens"))
    inputs = pipeline.input
    stage = pipeline.add(
        "heads",
        SemanticRoleHeadResolver(),
        sources={
            "role_spans": inputs["role_spans"],
            "tokens": inputs["tokens"],
        },
    )
    pipeline.output("role_heads", stage["role_heads"])

    result = pipeline.translate(
        inputs={
            "role_spans": _quantifier_span(),
            "tokens": _quantifier_tokens(),
        }
    )

    assert result["role_heads"]["head_text"].tolist() == ["apples"]


def test_role_head_rows_preserve_resolution_provenance() -> None:
    resolver = SemanticRoleHeadResolver()
    resolved = resolver.translate(
        _quantifier_span(),
        _quantifier_tokens(),
    )["role_heads"]

    row = resolved.iloc[0]
    assert row["syntactic_root_token_id"] == 2
    assert row["head_token_id"] == 5
    assert row["rule"] == "quantifier_of_object"
    assert row["rules_fingerprint"] == resolver.head_rules.fingerprint
    assert json.loads(row["resolution_path"]) == [
        {
            "rule": "quantifier_of_object",
            "kind": "child",
            "from_token_id": 2,
            "to_token_id": 3,
        },
        {
            "rule": "quantifier_of_object",
            "kind": "child",
            "from_token_id": 3,
            "to_token_id": 5,
        },
    ]


def test_rich_resolution_records_keep_coordination_and_rule_trace() -> None:
    results = resolve_semantic_heads(
        start=0,
        end=5,
        token_ids=list(range(5)),
        head_token_ids=[0, 0, 3, 0, 3],
        dependencies=["ROOT", "pobj", "cc", "conj", "pobj"],
        pos=["ADP", "PROPN", "CCONJ", "ADP", "PROPN"],
        text=["from", "Boston", "and", "from", "New York"],
        lemmas=["from", "Boston", "and", "from", "New York"],
        role="ARGM-LOC",
        source="srl",
    )

    assert [result.semantic_head_index for result in results] == [1, 4]
    assert results[0].syntactic_root_index == 0
    assert results[0].rule_id == "preposition_object"
    assert [move.kind for move in results[0].resolution_path] == ["child"]
    assert [move.kind for move in results[1].resolution_path] == [
        "coordination",
        "child",
    ]


def test_semantic_head_rules_are_deeply_immutable() -> None:
    rules = SemanticHeadRules()
    fingerprint = rules.fingerprint

    with pytest.raises(TypeError):
        rules.rules[0]["id"] = "mutated"
    with pytest.raises(TypeError):
        rules.rules[0]["when"]["role"] = ("ARG0",)
    with pytest.raises(TypeError):
        rules.rules[0]["when"]["role"][0] = "ARG0"

    assert rules.fingerprint == fingerprint
    assert rules.to_dict() == get_default_semantic_head_rules()


def test_boi_defaults_intentionally_differ_from_legacy_function_word_fallback() -> None:
    common = {
        "start": 0,
        "end": 2,
        "token_ids": [0, 1],
        "head_token_ids": [0, 0],
        "dependencies": ["ROOT", "det"],
        "pos": ["DET", "NOUN"],
        "text": ["the", "dog"],
        "role": "ARG1",
    }

    assert content_head_indices(**common) == (1,)
    assert resolve_semantic_head_indices(
        **common,
        lemmas=["the", "dog"],
        source="srl",
    ) == (0,)


def test_default_export_and_rule_fingerprint_are_stable(tmp_path) -> None:
    first = SemanticHeadRules()
    second = SemanticHeadRules.from_dict(first.to_dict())
    assert first == second
    assert first.fingerprint == second.fingerprint

    path = export_default_semantic_head_rules(tmp_path / "rules.json")
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported == first.to_dict()
    assert load_semantic_head_rules(path).fingerprint == first.fingerprint

    with pytest.raises(FileExistsError):
        export_default_semantic_head_rules(path)


def test_semantic_head_rules_and_translators_round_trip_json_state(tmp_path) -> None:
    custom = copy.deepcopy(get_default_semantic_head_rules())
    custom["name"] = "custom"
    custom["rules"].insert(
        0,
        {
            "id": "custom_rule",
            "when": {"any": [{"lemma": ["accord"]}, {"lemma": ["according"]}]},
            "move": [{"within": {"lemma": ["to"]}}],
        },
    )
    rules = SemanticHeadRules.from_dict(custom)

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

    snapshot = tmp_path / "operator"
    resolver.save_to_dir(snapshot, operator_id="semantic-head-test")
    frozen = BaseOperator.load_from_dir(snapshot)
    assert isinstance(frozen, SemanticRoleHeadResolver)
    assert frozen.head_rules == rules
    assert frozen.head_rules.fingerprint == rules.fingerprint

    coref = CoreferenceResolver(
        sentence_key="sent",
        token_key="tok",
        head_rules=rules,
    )
    restored_coref = CoreferenceResolver.from_json_state(coref.to_json_state())
    assert restored_coref.to_json_state() == coref.to_json_state()
    assert restored_coref.head_rules == rules


def test_invalid_rule_documents_fail_early() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SemanticHeadRules.from_dict({"schema_version": 2, "rules": []})
    with pytest.raises(ValueError, match="unsupported move"):
        SemanticHeadRules.from_dict(
            {
                "schema_version": 1,
                "name": "bad",
                "description": "",
                "rules": [
                    {
                        "id": "bad",
                        "when": {},
                        "move": [{"descendant": {"lemma": ["x"]}}],
                    }
                ],
            }
        )
