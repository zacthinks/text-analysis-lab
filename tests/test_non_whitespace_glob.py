from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab import dictionaries
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import DictionaryTranslator

_FEATURES = [
    "survey",
    "surveys",
    "survey data",
    "surveys were",
    "national survey",
    "case study",
    "case studies",
    "case study design",
]


class _IdentityMatrixFixture:
    artifact_id = "art_non_whitespace_glob"
    label = "dtm"
    status = "complete"
    n_rows = len(_FEATURES)
    primary_key = ("row_id",)
    artifact_type = ArtifactType.SPARSE_MATRIX

    def __init__(self) -> None:
        self.matrix = sparse.identity(len(_FEATURES), dtype=float, format="csr")

    @property
    def analysis(self):
        from text_analysis_lab.analysis.accessor import ArtifactAnalysis

        return ArtifactAnalysis(self)

    def get_data_columns(self):
        return list(_FEATURES)

    def get_matrix(self, *, positions=None, data_columns=True):
        _ = data_columns
        if positions is None:
            return self.matrix
        return self.matrix[[int(value) for value in positions], :]

    def iter_batches(self, *, batch_size, key_columns, include_position, **kwargs):
        _ = kwargs, batch_size
        info = (
            pd.DataFrame({"row_id": list(range(self.n_rows))})
            if key_columns
            else pd.DataFrame(index=range(self.n_rows))
        )
        if include_position:
            info["_position"] = list(range(self.n_rows))
        yield {"info": info, "matrix": self.matrix}


class _TranslatorSource:
    artifact_type = ArtifactType.SPARSE_MATRIX
    primary_key = ("row_id",)

    def get_data_columns(self):
        return list(_FEATURES)


def _translator_hits(dictionary: dictionaries.Dictionary) -> list[str]:
    translator = DictionaryTranslator(dictionary)
    request = TranslationRequest(batch_size=len(_FEATURES))
    translator.input_request(
        sources={"source": _TranslatorSource()}, mode="translate", request=request
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="source",
        primary_key=("row_id",),
        data={
            "info": pd.DataFrame({"row_id": list(range(len(_FEATURES)))}),
            "matrix": sparse.identity(len(_FEATURES), dtype=float, format="csr"),
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    result = translator.translate_batch(
        {"source": packet}, mode="translate", request=request
    ).outputs["output"]
    values = result["data"]["values"].toarray().reshape(-1)
    return [_FEATURES[index] for index in np.flatnonzero(values)]


def test_normal_glob_preserves_native_whitespace_spanning_semantics() -> None:
    artifact = _IdentityMatrixFixture()
    dictionary = dictionaries.Dictionary(
        {"method": ["survey*"]},
        valuetype="glob",
    )
    matches = artifact.analysis.dictionary_matches(dictionary)
    assert matches["feature"].tolist() == [
        "survey",
        "surveys",
        "survey data",
        "surveys were",
    ]


def test_non_whitespace_glob_keeps_star_inside_final_token() -> None:
    artifact = _IdentityMatrixFixture()
    dictionary = dictionaries.Dictionary(
        {"method": ["survey*", "case stud*"]},
        valuetype="non_whitespace_glob",
    )
    matches = artifact.analysis.dictionary_matches(dictionary)
    assert matches["feature"].tolist() == [
        "survey",
        "surveys",
        "case study",
        "case studies",
    ]


def test_dictionary_matches_and_dictionary_translator_share_resolution() -> None:
    artifact = _IdentityMatrixFixture()
    dictionary = dictionaries.Dictionary(
        {"method": ["survey*", "case stud*"]},
        valuetype="non_whitespace_glob",
    )
    audited = artifact.analysis.dictionary_matches(dictionary)["feature"].tolist()
    translated = _translator_hits(dictionary)
    assert translated == audited


def test_non_whitespace_glob_question_mark_does_not_match_space() -> None:
    from text_analysis_lab.dictionaries.matching import category_membership

    features = ["axb", "a b"]
    ordinary = dictionaries.Dictionary({"x": ["a?b"]}, valuetype="glob")
    bounded = dictionaries.Dictionary({"x": ["a?b"]}, valuetype="non_whitespace_glob")
    _, ordinary_membership = category_membership(features, ordinary)
    _, bounded_membership = category_membership(features, bounded)
    assert ordinary_membership.toarray().reshape(-1).tolist() == [1.0, 1.0]
    assert bounded_membership.toarray().reshape(-1).tolist() == [1.0, 0.0]


def test_non_whitespace_glob_keeps_explicit_character_classes() -> None:
    from text_analysis_lab.dictionaries.matching import category_membership

    dictionary = dictionaries.Dictionary(
        {"x": ["survey[ s]"]},
        valuetype="non_whitespace_glob",
    )
    _, membership = category_membership(["surveys", "survey "], dictionary)
    assert membership.toarray().reshape(-1).tolist() == [1.0, 1.0]


def test_non_whitespace_glob_is_a_first_class_valuetype() -> None:
    dictionary = dictionaries.Dictionary(
        {"x": ["survey*"]}, valuetype="non_whitespace_glob"
    )
    assert dictionary.valuetype == "non_whitespace_glob"

    valence = dictionaries.ValenceDictionary(
        {"survey*": 1.0}, valuetype="non_whitespace_glob"
    )
    assert valence.valuetype == "non_whitespace_glob"

    with pytest.raises(ValueError, match="valuetype must be one of"):
        dictionaries.Dictionary({"x": ["survey*"]}, valuetype="not_a_type")


def test_non_whitespace_glob_survives_selection_frames_and_operator_state() -> None:
    dictionary = dictionaries.Dictionary(
        {"method": ["survey*"], "other": ["case stud*"]},
        valuetype="non_whitespace_glob",
        case_sensitive=False,
    )
    assert dictionary["method"].valuetype == "non_whitespace_glob"

    restored_from_frame = dictionaries.Dictionary.from_frame(
        dictionary.to_frame(),
        valuetype="non_whitespace_glob",
        case_sensitive=False,
    )
    assert restored_from_frame.valuetype == "non_whitespace_glob"

    translator = DictionaryTranslator(dictionary)
    restored_translator = DictionaryTranslator.from_json_state(
        translator.to_json_state()
    )
    assert restored_translator.dictionary is not None
    assert restored_translator.dictionary.valuetype == "non_whitespace_glob"


def test_round27_4_serialized_state_migrates_to_new_valuetype() -> None:
    from text_analysis_lab.translators.dictionary_translator import (
        _deserialize_dictionary,
    )

    restored = _deserialize_dictionary(
        {
            "kind": "categorical",
            "entries": {"method": ["survey*"]},
            "valuetype": "glob",
            "case_sensitive": False,
            "non_whitespace_glob": True,
            "name": None,
            "provenance": None,
            "source": None,
        }
    )
    assert isinstance(restored, dictionaries.Dictionary)
    assert restored.valuetype == "non_whitespace_glob"


def test_valence_glob_uses_same_non_whitespace_matching_engine() -> None:
    from text_analysis_lab.dictionaries.matching import valence_vectors

    dictionary = dictionaries.ValenceDictionary(
        {"survey*": 2.0},
        valuetype="non_whitespace_glob",
    )
    scores, matched = valence_vectors(["survey", "survey data"], dictionary)["valence"]
    assert scores.tolist() == [2.0, 0.0]
    assert matched.tolist() == [True, False]


def test_regex_matching_forces_python_engine_for_string_extension_series() -> None:
    from text_analysis_lab.dictionaries.matching import _match_regex_chunks

    features = pd.Series(["survey", "survey data"], dtype="string")
    # ``\\Z`` is emitted by fnmatch.translate() and is valid in Python ``re``.
    result = _match_regex_chunks(
        features,
        (r"(?s:survey\S*)\Z",),
        flags=0,
        contains=False,
    )
    assert result.tolist() == [True, False]
