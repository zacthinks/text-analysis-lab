"""Ephemeral analytic methods over TeAL artifacts.

Analytic methods return ordinary in-memory results by default. Existing
hand-reviewed implementations are reused directly where available; each new
method lives in its own module rather than introducing a parallel analysis
hierarchy.
"""

from text_analysis_lab.analysis.accessor import ArtifactAnalysis
from text_analysis_lab.analysis.cosine_similarity import cosine_similarity
from text_analysis_lab.analysis.classification import ClassificationEvaluation, classification
from text_analysis_lab.analysis.generalized_difference import (
    GeneralizedDifferenceResult,
    generalized_difference,
)
from text_analysis_lab.analysis.distance import distance
from text_analysis_lab.analysis.dictionary_counts import dictionary_counts
from text_analysis_lab.analysis.dictionary_matches import dictionary_matches
from text_analysis_lab.analysis.feature_summary import feature_summary
from text_analysis_lab.analysis.matrix_summary import MatrixSummary, matrix_summary
from text_analysis_lab.analysis.neighbors import nearest_neighbors
from text_analysis_lab.analysis.polarity import polarity
from text_analysis_lab.analysis.polarity_scores import (
    polarity_difference,
    polarity_log_ratio,
    polarity_matched_difference,
    polarity_proportional_difference,
    polarity_total_difference,
)
from text_analysis_lab.analysis.row_summary import row_summary
from text_analysis_lab.analysis.summary import ArtifactSummary, summarize
from text_analysis_lab.analysis.crosstab import crosstab
from text_analysis_lab.analysis.valence import valence
from text_analysis_lab.core.kwic import KWICResult, keyword_in_context

# Public analytic-method name. Keep the existing implementation as the single
# source of truth; artifact.kwic(...) remains the convenience method.
kwic = keyword_in_context

__all__ = [
    "ArtifactAnalysis",
    "ArtifactSummary",
    "MatrixSummary",
    "KWICResult",
    "ClassificationEvaluation",
    "classification",
    "GeneralizedDifferenceResult",
    "generalized_difference",
    "cosine_similarity",
    "distance",
    "dictionary_counts",
    "dictionary_matches",
    "feature_summary",
    "kwic",
    "matrix_summary",
    "nearest_neighbors",
    "polarity",
    "polarity_difference",
    "polarity_log_ratio",
    "polarity_matched_difference",
    "polarity_proportional_difference",
    "polarity_total_difference",
    "row_summary",
    "summarize",
    "crosstab",
    "valence",
]
