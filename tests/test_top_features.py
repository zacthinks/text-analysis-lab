"""Focused tests for generic row-wise top-feature rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab import analysis
from text_analysis_lab.analysis.accessor import ArtifactAnalysis
from text_analysis_lab.core.types import ArtifactType


class MatrixArtifact:
    artifact_type = ArtifactType.SPARSE_MATRIX
    primary_key = ("document_id", "part_id")
    project = None

    def __init__(self, values, names=("a", "b", "c"), *, labels=None):
        self.matrix = sparse.csr_matrix(np.asarray(values, dtype=float))
        self.names = list(names)
        self.labels = list(labels or names)
        self.n_rows = self.matrix.shape[0]
        self.analysis = ArtifactAnalysis(self)

    def get_data_columns(self):
        return self.names

    def get_feature_metadata(self):
        return pd.DataFrame({
            "column_index": np.arange(len(self.names)),
            "column": self.names,
            "display_name": self.labels,
        })

    def iter_batches(self, *, batch_size=1000, **kwargs):
        for start in range(0, self.n_rows, batch_size):
            end = min(self.n_rows, start + batch_size)
            yield {
                "info": pd.DataFrame({
                    "document_id": np.arange(start, end),
                    "part_id": np.zeros(end - start, dtype=int),
                    "_position": np.arange(start, end),
                }),
                "matrix": self.matrix[start:end],
            }


def test_multiple_scores_wide_ties_and_keys():
    artifact = MatrixArtifact([[3, 1, 0], [0, 2, 2]])
    result = artifact.analysis.top_features(
        n=10, methods=["value", "probability", "absolute"], batch_size=1,
    )
    assert result.shape == (6, 9)
    assert result.columns.tolist() == [
        "document_id", "part_id", "rank", "value_ranking", "value_score",
        "probability_ranking", "probability_score", "absolute_ranking",
        "absolute_score",
    ]
    assert result.query("document_id == 0")["value_ranking"].tolist() == ["a", "b", "c"]
    assert result.query("document_id == 1")["value_ranking"].tolist() == ["b", "c", "a"]
    assert result.iloc[0]["probability_score"] == pytest.approx(0.75)


def test_signed_absolute_and_invalid_distribution():
    artifact = MatrixArtifact([[-1, 4, -6]])
    result = analysis.top_features(artifact, methods="absolute", n=2)
    assert result["absolute_ranking"].tolist() == ["c", "b"]
    assert result["absolute_score"].tolist() == [6.0, 4.0]
    with pytest.raises(ValueError, match="nonnegative"):
        artifact.analysis.top_features(methods="probability")


def test_zero_mass_invalid_for_probability_but_allowed_for_value():
    artifact = MatrixArtifact([[0, 0, 0]])
    assert artifact.analysis.top_features(methods="value", n=2)["value_ranking"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="positive row mass"):
        artifact.analysis.top_features(methods="probability")


def test_reference_alignment_frequency_and_lift():
    subject = MatrixArtifact([[1, 1, 0]], names=["b", "a", "c"])
    reference = MatrixArtifact([[2, 1, 0], [0, 1, 0]], names=["a", "b", "c"])
    result = subject.analysis.top_features(
        methods=["term_frequency", "document_frequency", "lift"],
        reference=reference,
        n=3,
    )
    assert result["term_frequency_ranking"].tolist() == ["a", "b", "c"]
    assert result["term_frequency_score"].tolist() == [2.0, 2.0, 0.0]
    assert result["document_frequency_ranking"].tolist() == ["b", "a", "c"]
    assert result["lift_ranking"].iloc[0] == "b"
    assert result["lift_score"].iloc[0] == pytest.approx(1.0)


def test_frex_cross_row_exclusivity_and_stable_ties():
    artifact = MatrixArtifact([[9, 1, 0], [1, 9, 0]])
    result = artifact.analysis.top_features(methods="frex", frex_weight=0.7)
    assert result.query("document_id == 0")["frex_ranking"].iloc[0] == "a"
    assert result.query("document_id == 1")["frex_ranking"].iloc[0] == "b"
    assert result["frex_score"].between(0, 1).all()


def test_custom_scores_kwargs_labels_and_rejections():
    artifact = MatrixArtifact([[1, 3, 2]], labels=["alpha", "beta", "gamma"])
    def scoring(values, multiplier):
        return -values * multiplier
    result = artifact.analysis.top_features(
        methods=[], custom={"reverse": scoring}, multiplier=2,
        label_field="display_name", n=2,
    )
    assert result["reverse_ranking"].tolist() == ["alpha", "gamma"]
    assert result["reverse_score"].tolist() == [-2.0, -4.0]
    with pytest.raises(ValueError, match="collide"):
        artifact.analysis.top_features(methods=["value"], custom={"value": scoring})
    with pytest.raises(ValueError, match="finite"):
        artifact.analysis.top_features(methods=[], custom={"bad": lambda values: [float("nan")] * 3})


def test_reference_mismatch_and_zero_background():
    subject = MatrixArtifact([[1, 0, 1]])
    bad_reference = MatrixArtifact([[1, 1, 1]], names=("a", "b", "different"))
    with pytest.raises(ValueError, match="canonical feature set"):
        subject.analysis.top_features(methods="lift", reference=bad_reference)
    with pytest.raises(ValueError, match="absent from reference"):
        subject.analysis.top_features(methods="lift", reference=MatrixArtifact([[1, 1, 0]]))


def test_sparse_batches_without_whole_matrix_conversion():
    artifact = MatrixArtifact([[3, 0, 1]] * 21)
    result = analysis.top_features(artifact, n=1, methods=["value"], batch_size=3)
    assert len(result) == 21
    assert result["value_ranking"].eq("a").all()
