from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab import analysis, dictionaries
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import DictionaryTranslator

_VALUES = np.array(
    [
        [2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 4.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
    ]
)
_FEATURES = ["Good", "great", "bad", "awful", "economy", "economic", "neutral"]


class _SourceFixture:
    artifact_type = ArtifactType.SPARSE_MATRIX
    primary_key = ("doc_id",)

    def get_data_columns(self):
        return list(_FEATURES)


class _MatrixFixture:
    artifact_id = "art_dictionary"
    label = "dtm"
    status = "complete"
    n_rows = 4
    primary_key = ("doc_id",)
    artifact_type = ArtifactType.SPARSE_MATRIX

    def __init__(self) -> None:
        self.matrix = sparse.csr_matrix(_VALUES)

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
        for start in range(0, self.n_rows, 2):
            positions = list(range(start, min(start + 2, self.n_rows)))
            info: dict[str, object] = {}
            if key_columns:
                info["doc_id"] = positions
            if include_position:
                info["_position"] = positions
            yield {
                "info": pd.DataFrame(info),
                "matrix": self.matrix[positions, :],
            }


class _TranslatedFixture:
    artifact_id = "art_dictionary_translation"
    label = "dictionary_counts"
    status = "complete"
    primary_key = ("doc_id",)
    artifact_type = ArtifactType.SPARSE_MATRIX

    def __init__(self, matrix, columns, metadata):
        self.matrix = sparse.csr_matrix(matrix)
        self.columns = list(columns)
        self.metadata = pd.DataFrame(metadata)
        self.n_rows = self.matrix.shape[0]

    @property
    def analysis(self):
        from text_analysis_lab.analysis.accessor import ArtifactAnalysis

        return ArtifactAnalysis(self)

    def get_data_columns(self):
        return list(self.columns)

    def get_matrix(self, *, positions=None, data_columns=True):
        _ = data_columns
        if positions is None:
            return self.matrix
        return self.matrix[[int(value) for value in positions], :]

    def get_metadata_columns(self, *, mode="local"):
        _ = mode
        return list(self.metadata.columns)

    def iter_batches(
        self,
        *,
        batch_size,
        key_columns,
        metadata_columns,
        include_position,
        **kwargs,
    ):
        _ = kwargs, batch_size
        for start in range(0, self.n_rows, 2):
            positions = list(range(start, min(start + 2, self.n_rows)))
            info = pd.DataFrame(index=range(len(positions)))
            if key_columns:
                info["doc_id"] = positions
            if include_position:
                info["_position"] = positions
            for column in metadata_columns:
                info[column] = self.metadata.iloc[positions][column].to_numpy()
            yield {"info": info, "matrix": self.matrix[positions, :]}


def _content_dictionary(*, valuetype: str = "glob") -> dictionaries.Dictionary:
    return dictionaries.Dictionary(
        {
            "positive": ["good", "great"],
            "negative": ["bad", "awful"],
            "neutral": ["neutral"],
            "economy": ["econom*"],
        },
        valuetype=valuetype,
        case_sensitive=False,
    )


def _run_translator(translator: DictionaryTranslator, values=_VALUES):
    source = _SourceFixture()
    request = TranslationRequest(batch_size=10)
    translator.input_request(
        sources={"source": source}, mode="translate", request=request
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="source",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": list(range(len(values)))}),
            "matrix": sparse.csr_matrix(values),
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    result = translator.translate_batch(
        {"source": packet}, mode="translate", request=request
    )
    return result.outputs["output"]


def test_dictionary_objects_support_mapping_tidy_frames_and_neutral_pole() -> None:
    dictionary = _content_dictionary()
    reconstructed = dictionaries.Dictionary.from_frame(dictionary.to_frame())
    assert reconstructed.entries == dictionary.entries

    polarity = dictionaries.PolarityDictionary(
        dictionary,
        positive="positive",
        negative="negative",
        neutral="neutral",
    )
    assert polarity.neutral == ("neutral",)

    values = dictionaries.ValenceDictionary({"good": 2.0, "bad": -2.0})
    assert values.dimensions == ("valence",)
    multi = dictionaries.ValenceDictionary(
        {
            "pleasantness": {"good": 2.0, "bad": -2.0},
            "arousal": {"good": 1.0, "bad": 3.0},
        }
    )
    restored = dictionaries.ValenceDictionary.from_frame(multi.to_frame())
    assert restored.values == multi.values


def test_dictionary_counts_and_matches_remain_pretranslation_audits() -> None:
    artifact = _MatrixFixture()
    dictionary = _content_dictionary()
    counts = artifact.analysis.dictionary_counts(dictionary, batch_size=2)
    assert counts["positive"].tolist() == pytest.approx([3, 0, 0, 1])
    assert counts["negative"].tolist() == pytest.approx([0, 3, 0, 1])
    assert counts["economy"].tolist() == pytest.approx([3, 1, 0, 0])

    matches = artifact.analysis.dictionary_matches(dictionary)
    assert ("economy", "economic") in set(
        map(tuple, matches[["key", "feature"]].to_numpy())
    )


def test_dictionary_translator_categorical_outputs_category_hit_counts() -> None:
    payload = _run_translator(DictionaryTranslator(_content_dictionary()))
    assert payload["data"]["columns"] == (
        "positive",
        "negative",
        "neutral",
        "economy",
    )
    assert payload["data"]["values"].toarray().tolist() == [
        [3, 0, 4, 3],
        [0, 3, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 2, 0],
    ]
    assert payload["metadata"].to_dict("list") == {
        "matched": [10, 5, 0, 4],
        "unmatched": [0, 0, 0, 0],
        "total": [10, 5, 0, 4],
    }


def test_dictionary_translator_polarity_histogram_and_invariants() -> None:
    polarity = dictionaries.PolarityDictionary(
        _content_dictionary(),
        positive="positive",
        negative="negative",
        neutral="neutral",
    )
    payload = _run_translator(DictionaryTranslator(polarity))
    assert payload["data"]["columns"] == ("positive", "negative", "neutral")
    translated = payload["data"]["values"].toarray()
    assert translated.tolist() == [
        [3, 0, 4],
        [0, 3, 1],
        [0, 0, 0],
        [1, 1, 2],
    ]
    metadata = payload["metadata"]
    assert metadata["matched"].tolist() == [7, 4, 0, 4]
    assert metadata["unmatched"].tolist() == [3, 1, 0, 0]
    assert np.array_equal(translated.sum(axis=1).astype(int), metadata["matched"])


def test_polarity_resolution_deduplicates_within_pole_and_rejects_cross_pole() -> None:
    base = dictionaries.Dictionary(
        {
            "positive_a": ["good"],
            "positive_b": ["good", "great"],
            "negative": ["bad"],
        },
        valuetype="fixed",
        case_sensitive=False,
    )
    payload = _run_translator(
        DictionaryTranslator(
            dictionaries.PolarityDictionary(
                base,
                positive=["positive_a", "positive_b"],
                negative="negative",
            )
        )
    )
    assert payload["data"]["values"].toarray()[0, 0] == 3

    conflicting = dictionaries.PolarityDictionary(
        dictionaries.Dictionary(
            {"positive": ["good"], "negative": ["g*"]},
            valuetype="glob",
            case_sensitive=False,
        ),
        positive="positive",
        negative="negative",
    )
    with pytest.raises(OperatorError, match="multiple different poles"):
        _run_translator(DictionaryTranslator(conflicting))


def test_dictionary_translator_valence_groups_counts_by_distinct_score() -> None:
    valence = dictionaries.ValenceDictionary(
        {
            "good": 2.0,
            "great": 3.0,
            "bad": -2.0,
            "awful": -4.0,
        },
        case_sensitive=False,
    )
    payload = _run_translator(DictionaryTranslator(valence))
    assert payload["data"]["columns"] == ("-4", "-2", "2", "3")
    assert payload["data"]["values"].toarray().tolist() == [
        [0, 0, 2, 1],
        [2, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ]
    assert payload["metadata"]["matched"].tolist() == [3, 3, 0, 2]
    assert payload["metadata"]["unmatched"].tolist() == [7, 2, 0, 2]


def test_valence_histogram_preserves_six_minus_ones_vs_one_minus_six() -> None:
    source = _SourceFixture()
    source.get_data_columns = lambda: ["mildly_bad", "extremely_bad"]
    translator = DictionaryTranslator(
        dictionaries.ValenceDictionary({"mildly_bad": -1.0, "extremely_bad": -6.0})
    )
    request = TranslationRequest()
    translator.input_request(
        sources={"source": source}, mode="translate", request=request
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="source",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": [0, 1]}),
            "matrix": sparse.csr_matrix([[6, 0], [0, 1]]),
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    payload = translator.translate_batch(
        {"source": packet}, mode="translate", request=request
    ).outputs["output"]
    assert payload["data"]["columns"] == ("-6", "-1")
    assert payload["data"]["values"].toarray().tolist() == [[0, 6], [1, 0]]


def test_dictionary_translator_rejects_non_count_matrices() -> None:
    translator = DictionaryTranslator(
        dictionaries.ValenceDictionary({"good": 1.0}, case_sensitive=False)
    )
    weighted = _VALUES.copy()
    weighted[0, 0] = 0.5
    with pytest.raises(ArtifactError, match="integer-valued counts"):
        _run_translator(translator, weighted)


def test_polarity_analysis_uses_durable_histogram_without_dictionary() -> None:
    artifact = _TranslatedFixture(
        [[3, 0, 4], [0, 3, 1], [0, 0, 0], [1, 1, 2]],
        ["positive", "negative", "neutral"],
        {
            "matched": [7, 4, 0, 4],
            "unmatched": [3, 1, 0, 0],
            "total": [10, 5, 0, 4],
        },
    )

    def coverage_weighted(difference, coverage):
        return difference * coverage

    result = artifact.analysis.polarity(
        smoothing=0.5,
        zero_division=0.0,
        custom={"coverage_weighted": coverage_weighted},
    )
    assert result["positive"].tolist() == pytest.approx([3, 0, 0, 1])
    assert result["neutral"].tolist() == pytest.approx([4, 1, 0, 2])
    assert result["matched"].tolist() == pytest.approx([7, 4, 0, 4])
    assert result["coverage"].tolist() == pytest.approx([0.7, 0.8, 0, 1])
    assert result["difference"].tolist() == pytest.approx([3, -3, 0, 0])
    assert result["matched_difference"].tolist() == pytest.approx([3 / 7, -3 / 4, 0, 0])
    assert result["total_difference"].tolist() == pytest.approx([0.3, -0.6, 0, 0])
    assert result["proportional_difference"].tolist() == pytest.approx([1, -1, 0, 0])
    assert result["log_ratio"].iloc[0] == pytest.approx(math.log(3.5 / 0.5))
    assert result["coverage_weighted"].tolist() == pytest.approx([2.1, -2.4, 0, 0])


def test_valence_analysis_summarizes_preserved_distribution() -> None:
    artifact = _TranslatedFixture(
        [[0, 0, 2, 1], [2, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
        ["-4", "-2", "2", "3"],
        {
            "matched": [3, 3, 0, 2],
            "unmatched": [7, 2, 0, 2],
            "total": [10, 5, 0, 4],
        },
    )
    result = artifact.analysis.valence(zero_division=0.0)
    assert result["weighted_sum"].tolist() == pytest.approx([7, -10, 0, 0])
    assert result["mean_matched"].tolist() == pytest.approx([7 / 3, -10 / 3, 0, 0])
    assert result["mean_all"].tolist() == pytest.approx([0.7, -2, 0, 0])
    assert result["std_matched"].tolist() == pytest.approx(
        [math.sqrt(2 / 9), math.sqrt(8 / 9), 0, 2]
    )
    assert result["negative"].tolist() == pytest.approx([0, 3, 0, 1])
    assert result["positive"].tolist() == pytest.approx([3, 0, 0, 1])
    assert result["median"].tolist() == pytest.approx([2, -4, 0, 0])


def test_valence_analysis_distinguishes_equal_weighted_sums() -> None:
    artifact = _TranslatedFixture(
        [[0, 6], [1, 0]],
        ["-6", "-1"],
        {"matched": [6, 1], "unmatched": [0, 0], "total": [6, 1]},
    )
    result = analysis.valence(artifact)
    assert result["weighted_sum"].tolist() == pytest.approx([-6, -6])
    assert result["mean_matched"].tolist() == pytest.approx([-1, -6])
    assert result["std_matched"].tolist() == pytest.approx([0, 0])
    assert result["median"].tolist() == pytest.approx([-1, -6])


def test_valence_dimension_is_chosen_at_translation_time() -> None:
    dictionary = dictionaries.ValenceDictionary(
        {
            "pleasantness": {"good": 2.0, "bad": -2.0},
            "arousal": {"great": 3.0, "awful": 4.0},
        },
        case_sensitive=False,
    )
    with pytest.raises(ValueError, match="dimension is required"):
        DictionaryTranslator(dictionary)
    translator = DictionaryTranslator(dictionary, dimension="arousal")
    assert translator.dimension == "arousal"


def test_valence_conflicting_patterns_fail_when_resolved_against_vocabulary() -> None:
    dictionary = dictionaries.ValenceDictionary(
        {"x": {"econom*": 1.0, "economic": 2.0}}, valuetype="glob"
    )
    with pytest.raises(ValueError, match="conflicting values"):
        _run_translator(DictionaryTranslator(dictionary, dimension="x"))
