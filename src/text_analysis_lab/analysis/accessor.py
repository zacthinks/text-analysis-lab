"""Artifact-bound convenience access to ephemeral Analytic Methods."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from text_analysis_lab.analysis.classification import classification
from text_analysis_lab.analysis.cosine_similarity import cosine_similarity
from text_analysis_lab.analysis.crosstab import crosstab
from text_analysis_lab.analysis.dictionary_counts import dictionary_counts
from text_analysis_lab.analysis.dictionary_matches import dictionary_matches
from text_analysis_lab.analysis.distance import distance
from text_analysis_lab.analysis.feature_summary import feature_summary
from text_analysis_lab.analysis.generalized_difference import (
    generalized_difference,
    generalized_difference_by,
)
from text_analysis_lab.analysis.matrix_summary import matrix_summary
from text_analysis_lab.analysis.neighbors import nearest_neighbors
from text_analysis_lab.analysis.polarity import polarity
from text_analysis_lab.analysis.row_summary import row_summary
from text_analysis_lab.analysis.summary import summarize
from text_analysis_lab.analysis.text_diagnostics import text_diagnostics
from text_analysis_lab.analysis.valence import valence
from text_analysis_lab.core.kwic import keyword_in_context

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class ArtifactAnalysis:
    """Bind Analytic Methods to one artifact without mutating its artifact graph."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

    @wraps(text_diagnostics)
    def text_diagnostics(self, **kwargs: Any):
        return text_diagnostics(self._artifact, **kwargs)

    @wraps(summarize)
    def summarize(self):
        return summarize(self._artifact)

    @wraps(cosine_similarity)
    def cosine_similarity(self, **kwargs: Any):
        return cosine_similarity(self._artifact, **kwargs)

    @wraps(distance)
    def distance(self, **kwargs: Any):
        return distance(self._artifact, **kwargs)

    @wraps(matrix_summary)
    def matrix_summary(self, **kwargs: Any):
        return matrix_summary(self._artifact, **kwargs)

    @wraps(row_summary)
    def row_summary(self, **kwargs: Any):
        return row_summary(self._artifact, **kwargs)

    @wraps(feature_summary)
    def feature_summary(self, **kwargs: Any):
        return feature_summary(self._artifact, **kwargs)

    @wraps(nearest_neighbors)
    def nearest_neighbors(self, **kwargs: Any):
        return nearest_neighbors(self._artifact, **kwargs)

    @wraps(dictionary_counts)
    def dictionary_counts(self, dictionary: Any, **kwargs: Any):
        return dictionary_counts(self._artifact, dictionary, **kwargs)

    @wraps(dictionary_matches)
    def dictionary_matches(self, dictionary: Any):
        return dictionary_matches(self._artifact, dictionary)

    @wraps(polarity)
    def polarity(self, **kwargs: Any):
        return polarity(self._artifact, **kwargs)

    @wraps(valence)
    def valence(self, **kwargs: Any):
        return valence(self._artifact, **kwargs)

    @wraps(classification)
    def classification(self, **kwargs: Any):
        return classification(self._artifact, **kwargs)

    @wraps(generalized_difference)
    def generalized_difference(self, **kwargs: Any):
        return generalized_difference(self._artifact, **kwargs)

    @wraps(generalized_difference_by)
    def generalized_difference_by(self, **kwargs: Any):
        return generalized_difference_by(self._artifact, **kwargs)

    @wraps(crosstab)
    def crosstab(self, rows: str, columns: str, **kwargs: Any):
        return crosstab(self._artifact, rows=rows, columns=columns, **kwargs)

    @wraps(keyword_in_context)
    def kwic(self, pattern: str, **kwargs: Any):
        return keyword_in_context(self._artifact, pattern, **kwargs)
