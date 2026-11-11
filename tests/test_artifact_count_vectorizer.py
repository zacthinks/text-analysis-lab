from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import ArtifactCountVectorizer


def _packet(
    frame: pd.DataFrame, *, primary_key=("doc_id", "sentence_id", "token_id")
) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_tokens",
        primary_key=tuple(primary_key),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _source(primary_key=("doc_id", "sentence_id", "token_id")):
    return SimpleNamespace(
        artifact_type=ArtifactType.TABLE, primary_key=list(primary_key)
    )


def test_artifact_count_vectorizer_declares_reduced_key_full_artifact_source() -> None:
    translator = ArtifactCountVectorizer(field="lemma", group_by=("doc_id",))
    sources = {"source": _source()}
    spec = translator.output_specs(sources=sources, request=TranslationRequest())
    assert spec.artifact_type == ArtifactType.SPARSE_MATRIX
    assert spec.lineage_mode == "reduced_key"
    assert spec.basis_labels == ("source",)
    request = translator.input_request(
        sources=sources, mode="fit_translate", request=TranslationRequest(batch_size=7)
    )
    assert request.mode == "full_artifact"
    assert request.batch_size is None
    assert request.columns.keys is True
    assert request.columns.data == "lemma"


def test_artifact_count_vectorizer_counts_arbitrary_field_and_keeps_zero_feature_groups() -> (
    None
):
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 2, 2, 3],
            "sentence_id": [0, 0, 0, 0, 0, 0],
            "token_id": [0, 1, 2, 0, 1, 0],
            "pos": ["NOUN", "VERB", "NOUN", "NOUN", "NOUN", ""],
        }
    )
    translator = ArtifactCountVectorizer(field="pos", group_by=("doc_id",))
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    assert payload["keys"].to_dict("records") == [
        {"doc_id": 1},
        {"doc_id": 2},
        {"doc_id": 3},
    ]
    assert payload["data"]["columns"] == ["NOUN", "VERB"]
    assert payload["data"]["values"].toarray().tolist() == [[2, 1], [2, 0], [0, 0]]


def test_group_by_can_retain_multiple_parent_key_levels() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 2],
            "sentence_id": [0, 0, 1, 0],
            "token_id": [0, 1, 0, 0],
            "lemma": ["alpha", "beta", "gamma", "delta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id", "sentence_id")
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    payload = result.outputs["output"]
    assert payload["keys"].to_dict("records") == [
        {"doc_id": 1, "sentence_id": 0},
        {"doc_id": 1, "sentence_id": 1},
        {"doc_id": 2, "sentence_id": 0},
    ]
    assert payload["data"]["values"].shape == (3, 4)


def test_ngrams_default_to_immediate_parent_boundary_not_aggregation_boundary() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 1],
            "sentence_id": [0, 0, 1, 1],
            "token_id": [0, 1, 0, 1],
            "lemma": ["alpha", "beta", "gamma", "delta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",), ngram_range=(1, 2)
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    columns = result.outputs["output"]["data"]["columns"]
    assert "alpha_beta" in columns
    assert "gamma_delta" in columns
    assert "beta_gamma" not in columns
    values = result.outputs["output"]["data"]["values"].toarray()[0]
    assert dict(zip(columns, values, strict=True)) == {
        "alpha": 1,
        "alpha_beta": 1,
        "beta": 1,
        "delta": 1,
        "gamma": 1,
        "gamma_delta": 1,
    }


def test_sequence_by_can_intentionally_allow_cross_sentence_ngrams() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 1],
            "sentence_id": [0, 0, 1, 1],
            "token_id": [0, 1, 0, 1],
            "lemma": ["alpha", "beta", "gamma", "delta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma",
        group_by=("doc_id",),
        sequence_by=("doc_id",),
        ngram_range=(2, 2),
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["columns"] == [
        "alpha_beta",
        "beta_gamma",
        "gamma_delta",
    ]


def test_ngrams_follow_filtered_artifact_row_sequence_even_when_token_ids_have_gaps() -> (
    None
):
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1],
            "sentence_id": [0, 0],
            "token_id": [0, 2],
            "lemma": ["alpha", "beta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",), ngram_range=(2, 2)
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["columns"] == ["alpha_beta"]
    assert result.outputs["output"]["data"]["values"].toarray().tolist() == [[1]]


def test_frozen_vocabulary_reuse_ignores_unseen_features_and_binary_caps_counts() -> (
    None
):
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 2],
            "sentence_id": [0, 0, 0, 0],
            "token_id": [0, 1, 2, 0],
            "lemma": ["alpha", "alpha", "unknown", "beta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma",
        group_by=("doc_id",),
        binary=True,
        vocabulary={"alpha": 0, "beta": 1},
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["values"].toarray().tolist() == [
        [1, 0],
        [0, 1],
    ]


def test_df_trimming_and_max_features_are_group_based() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 2, 2, 3, 3],
            "sentence_id": [0] * 7,
            "token_id": [0, 1, 2, 0, 1, 0, 1],
            "lemma": ["common", "common", "rare1", "common", "mid", "common", "mid"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",), min_df=2, max_features=1
    )
    result = translator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["columns"] == ["common"]
    assert result.outputs["output"]["data"]["values"].toarray().tolist() == [
        [2],
        [1],
        [1],
    ]


def test_source_key_contract_rejects_non_prefix_group_and_bad_sequence_boundary() -> (
    None
):
    with pytest.raises(OperatorError, match="group_by"):
        ArtifactCountVectorizer(field="lemma", group_by=("sentence_id",)).output_specs(
            sources={"source": _source()}, request=TranslationRequest()
        )
    with pytest.raises(OperatorError, match="sequence_by"):
        ArtifactCountVectorizer(
            field="lemma",
            group_by=("doc_id",),
            sequence_by=("doc_id", "token_id"),
        ).output_specs(sources={"source": _source()}, request=TranslationRequest())


def test_ngram_display_collision_fails_instead_of_merging_distinct_features() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 2],
            "lemma": ["a_b", "a", "b"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",), ngram_range=(1, 2)
    )
    with pytest.raises(ArtifactError, match="ambiguous"):
        translator.translate_batch(
            {"source": _packet(frame)},
            mode="fit_translate",
            request=TranslationRequest(),
        )


def test_json_state_round_trip_preserves_ngram_and_boundary_configuration() -> None:
    translator = ArtifactCountVectorizer(
        field="lemma",
        group_by=("collection_id", "doc_id"),
        sequence_by=("collection_id", "doc_id", "sentence_id"),
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        max_features=100,
        binary=True,
        ngram_separator="__",
        vocabulary={"alpha": 0, "beta": 1},
    )
    restored = ArtifactCountVectorizer.from_json_state(
        json.loads(json.dumps(translator.to_json_state(include_vocabulary=True)))
    )
    assert restored.to_json_state(include_vocabulary=True) == translator.to_json_state(
        include_vocabulary=True
    )


def test_boolean_metadata_filters_preserve_empty_groups_and_fitted_state() -> None:
    frame = pd.DataFrame({
        "doc_id": [1, 1, 2, 2, 3],
        "sentence_id": [0, 0, 0, 0, 0],
        "token_id": [0, 1, 0, 1, 0],
        "lemma": ["alpha", "the", "beta", "gamma", "the"],
        "is_alpha": [True, True, True, True, True],
        "is_stop": [False, True, False, False, True],
    })
    operator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",),
        metadata_filters={"is_alpha": True, "is_stop": False},
    )
    result = operator.translate_batch(
        {"source": _packet(frame)}, mode="fit_translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert result["keys"]["doc_id"].tolist() == [1, 2, 3]
    assert result["data"]["columns"] == ["alpha", "beta", "gamma"]
    assert result["data"]["values"].toarray().tolist() == [
        [1, 0, 0], [0, 1, 1], [0, 0, 0],
    ]
    assert operator.to_json_state()["metadata_filters"] == {
        "is_alpha": True, "is_stop": False,
    }
    assert operator.from_json_state(
        operator.to_json_state(include_vocabulary=True)
    ).metadata_filters == operator.metadata_filters
    request = operator.input_request(
        sources={"source": _source()}, mode="translate",
        request=TranslationRequest(),
    )
    assert request.columns.data == ["lemma", "is_alpha", "is_stop"]


def test_metadata_filter_rejects_non_boolean_values_and_missing_columns() -> None:
    for value in (1, 0, "true", None):
        with pytest.raises(ValueError, match="True or False"):
            ArtifactCountVectorizer(
                field="lemma", group_by=("doc_id",),
                metadata_filters={"is_alpha": value},
            )
    with pytest.raises(ArtifactError, match="is_alpha"):
        ArtifactCountVectorizer(
            field="lemma", group_by=("doc_id",),
            metadata_filters={"is_alpha": True},
        ).translate_batch(
            {"source": _packet(pd.DataFrame({
                "doc_id": [1], "sentence_id": [0], "token_id": [0],
                "lemma": ["one"],
            }))}, mode="fit_translate", request=TranslationRequest(),
        )


def test_target_artifact_adds_zero_rows_at_parent_key() -> None:
    source = pd.DataFrame({
        "doc_id": [1, 1, 2], "sentence_id": [0, 0, 0],
        "token_id": [0, 1, 0], "lemma": ["a", "b", "b"],
    })
    target = pd.DataFrame({"doc_id": [3, 1, 2]})
    operator = ArtifactCountVectorizer(field="lemma", group_by=("doc_id",))
    sources = {"source": _source(), "target": _source(("doc_id",))}
    spec = operator.output_specs(sources=sources, request=TranslationRequest())
    assert spec.lineage_mode == "preserved_key"
    assert spec.basis_labels == ("target",)
    req = operator.input_request(
        sources=sources, mode="fit_translate", request=TranslationRequest(),
    )
    assert set(req) == {"source", "target"}
    assert req["target"].columns.keys is True
    assert req["target"].columns.data is False
    result = operator.translate_batch(
        {
            "source": _packet(source),
            "target": InputBatch(
                source_label="target", artifact_id="art_reference",
                primary_key=("doc_id",), data=target,
                batch_index=0, batch_count=1, is_first=True, is_last=True,
            ),
        },
        mode="fit_translate", request=TranslationRequest(),
    ).outputs["output"]
    assert result["keys"]["doc_id"].tolist() == [3, 1, 2]
    assert result["data"]["columns"] == ["a", "b"]
    assert result["data"]["values"].toarray().tolist() == [
        [0, 0], [1, 1], [0, 1],
    ]


def test_target_artifact_rejects_mismatched_grain_and_unexpected_groups() -> None:
    operator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",), vocabulary={"a": 0},
    )
    with pytest.raises(OperatorError, match="exactly match"):
        operator.output_specs(
            sources={"source": _source(), "target": _source()},
            request=TranslationRequest(),
        )
    with pytest.raises(ArtifactError, match="absent from the target"):
        operator.translate_batch(
            {
                "source": _packet(pd.DataFrame({
                    "doc_id": [1], "sentence_id": [0], "token_id": [0],
                    "lemma": ["a"],
                })),
                "target": InputBatch(
                    source_label="target", artifact_id="art_target",
                    primary_key=("doc_id",),
                    data=pd.DataFrame({"doc_id": [2]}),
                    batch_index=0, batch_count=1, is_first=True, is_last=True,
                ),
            }, mode="translate", request=TranslationRequest(),
        )


def test_standalone_spacy_tokens_replay_zero_for_empty_query() -> None:
    from scipy import sparse
    operator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id",),
        metadata_filters={"is_alpha": True},
        vocabulary={"apple": 0, "banana": 1},
    )
    token_rows = pd.DataFrame({
        "source_position": [0, 0], "sentence_id": [0, 0],
        "token_id": [0, 1], "lemma": ["apple", "x"],
        "is_alpha": [True, False],
    })
    result = operator.translate(token_rows, target=["apple x", ""])
    assert result["groups"]["source_position"].tolist() == [0, 1]
    assert sparse.issparse(result["values"])
    assert result["values"].toarray().tolist() == [[1, 0], [0, 0]]
    all_empty = token_rows.iloc[:0]
    result = operator.translate(all_empty, target=[""])
    assert result["values"].shape == (1, 2)
    assert result["values"].nnz == 0


def test_target_can_be_sentence_grain_with_composite_primary_key() -> None:
    source = pd.DataFrame({
        "doc_id": [1, 1, 2],
        "sentence_id": [0, 0, 0],
        "token_id": [0, 1, 0],
        "lemma": ["alpha", "beta", "beta"],
        "is_alpha": [True, True, True],
    })
    target = pd.DataFrame({
        "doc_id": [1, 1, 2, 3],
        "sentence_id": [0, 1, 0, 0],
    })
    operator = ArtifactCountVectorizer(
        field="lemma", group_by=("doc_id", "sentence_id"),
        metadata_filters={"is_alpha": True},
    )
    inputs = {
        "source": _packet(source),
        "target": InputBatch(
            source_label="target", artifact_id="art_sentences",
            primary_key=("doc_id", "sentence_id"), data=target,
            batch_index=0, batch_count=1, is_first=True, is_last=True,
        ),
    }
    result = operator.translate_batch(
        inputs, mode="fit_translate", request=TranslationRequest(),
    ).outputs["output"]
    assert result["keys"].to_dict("records") == target.to_dict("records")
    assert result["data"]["columns"] == ["alpha", "beta"]
    assert result["data"]["values"].toarray().tolist() == [
        [1, 1], [0, 0], [0, 1], [0, 0],
    ]
