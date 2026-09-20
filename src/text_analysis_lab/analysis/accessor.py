"""Artifact-bound convenience access to ephemeral Analytic Methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class ArtifactAnalysis:
    """Bind Analytic Methods to one artifact without mutating its artifact graph."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

    def text_diagnostics(self, **kwargs: Any):
        from text_analysis_lab.analysis.text_diagnostics import text_diagnostics

        return text_diagnostics(self._artifact, **kwargs)

    def summarize(self):
        from text_analysis_lab.analysis.summary import summarize

        return summarize(self._artifact)

    def cosine_similarity(self, **kwargs: Any):
        from text_analysis_lab.analysis.cosine_similarity import cosine_similarity

        return cosine_similarity(self._artifact, **kwargs)

    def distance(self, **kwargs: Any):
        from text_analysis_lab.analysis.distance import distance

        return distance(self._artifact, **kwargs)

    def matrix_summary(self, **kwargs: Any):
        from text_analysis_lab.analysis.matrix_summary import matrix_summary

        return matrix_summary(self._artifact, **kwargs)

    def row_summary(self, **kwargs: Any):
        from text_analysis_lab.analysis.row_summary import row_summary

        return row_summary(self._artifact, **kwargs)

    def feature_summary(self, **kwargs: Any):
        from text_analysis_lab.analysis.feature_summary import feature_summary

        return feature_summary(self._artifact, **kwargs)

    def nearest_neighbors(self, **kwargs: Any):
        from text_analysis_lab.analysis.neighbors import nearest_neighbors

        return nearest_neighbors(self._artifact, **kwargs)

    def dictionary_counts(self, dictionary: Any, **kwargs: Any):
        from text_analysis_lab.analysis.dictionary_counts import dictionary_counts

        return dictionary_counts(self._artifact, dictionary, **kwargs)

    def dictionary_matches(self, dictionary: Any):
        from text_analysis_lab.analysis.dictionary_matches import dictionary_matches

        return dictionary_matches(self._artifact, dictionary)

    def polarity(self, **kwargs: Any):
        from text_analysis_lab.analysis.polarity import polarity

        return polarity(self._artifact, **kwargs)

    def valence(self, **kwargs: Any):
        from text_analysis_lab.analysis.valence import valence

        return valence(self._artifact, **kwargs)

    def classification(self, **kwargs: Any):
        from text_analysis_lab.analysis.classification import classification

        return classification(self._artifact, **kwargs)

    def generalized_difference(self, **kwargs: Any):
        from text_analysis_lab.analysis.generalized_difference import (
            generalized_difference,
        )

        return generalized_difference(self._artifact, **kwargs)

    def generalized_difference_by(self, **kwargs: Any):
        from text_analysis_lab.analysis.generalized_difference import (
            generalized_difference_by,
        )

        return generalized_difference_by(self._artifact, **kwargs)

    def crosstab(self, rows: str, columns: str, **kwargs: Any):
        from text_analysis_lab.analysis.crosstab import crosstab

        return crosstab(self._artifact, rows=rows, columns=columns, **kwargs)

    def kwic(self, pattern: str, **kwargs: Any):
        from text_analysis_lab.core.kwic import keyword_in_context

        return keyword_in_context(self._artifact, pattern, **kwargs)
