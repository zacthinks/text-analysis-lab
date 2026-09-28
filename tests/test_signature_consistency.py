"""Guard the explicit convenience-method signatures against delegated API drift."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

from text_analysis_lab.analysis.accessor import ArtifactAnalysis
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
from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.artifact_subclasses import _MatrixArtifact
from text_analysis_lab.core.kwic import keyword_in_context
from text_analysis_lab.visualization.accessor import ArtifactVisualization
from text_analysis_lab.visualization.histogram import histogram

ParameterShape = tuple[str, inspect._ParameterKind, Any]


def _shape(
    callable_: Callable[..., Any],
    *,
    drop_first: bool = False,
    positionalize: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[ParameterShape]:
    parameters = list(inspect.signature(callable_).parameters.values())
    if drop_first:
        parameters = parameters[1:]
    positionalize = set(positionalize)
    exclude = set(exclude)

    result: list[ParameterShape] = []
    for parameter in parameters:
        if parameter.name in exclude:
            continue
        kind = parameter.kind
        if parameter.name in positionalize and kind is inspect.Parameter.KEYWORD_ONLY:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        result.append((parameter.name, kind, parameter.default))
    return result


def test_analysis_accessor_signatures_track_canonical_functions() -> None:
    delegated = [
        (ArtifactAnalysis.text_diagnostics, text_diagnostics, ()),
        (ArtifactAnalysis.summarize, summarize, ()),
        (ArtifactAnalysis.cosine_similarity, cosine_similarity, ()),
        (ArtifactAnalysis.distance, distance, ()),
        (ArtifactAnalysis.matrix_summary, matrix_summary, ()),
        (ArtifactAnalysis.row_summary, row_summary, ()),
        (ArtifactAnalysis.feature_summary, feature_summary, ()),
        (ArtifactAnalysis.nearest_neighbors, nearest_neighbors, ()),
        (ArtifactAnalysis.dictionary_counts, dictionary_counts, ()),
        (ArtifactAnalysis.dictionary_matches, dictionary_matches, ()),
        (ArtifactAnalysis.polarity, polarity, ()),
        (ArtifactAnalysis.valence, valence, ()),
        (ArtifactAnalysis.classification, classification, ()),
        (ArtifactAnalysis.generalized_difference, generalized_difference, ()),
        (
            ArtifactAnalysis.generalized_difference_by,
            generalized_difference_by,
            (),
        ),
        # The accessor deliberately makes these required selectors positional
        # conveniences even though the canonical functions make them keyword-only.
        (ArtifactAnalysis.crosstab, crosstab, ("rows", "columns")),
        (ArtifactAnalysis.kwic, keyword_in_context, ()),
    ]

    for method, source, positionalize in delegated:
        assert _shape(method, drop_first=True) == _shape(
            source,
            drop_first=True,
            positionalize=positionalize,
        )


def test_visualization_accessor_signature_tracks_histogram() -> None:
    assert _shape(ArtifactVisualization.histogram, drop_first=True) == _shape(
        histogram,
        drop_first=True,
        positionalize=("field",),
    )


def test_context_convenience_signatures_track_get_context() -> None:
    expected = _shape(
        BaseArtifact.get_context,
        exclude=("before", "after", "include_focus"),
    )
    expected.insert(
        2,
        ("n", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1),
    )

    assert _shape(BaseArtifact.get_previous) == expected
    assert _shape(BaseArtifact.get_next) == expected


def test_matrix_kwic_override_tracks_base_kwic() -> None:
    assert _shape(_MatrixArtifact.kwic) == _shape(BaseArtifact.kwic)


def test_accessor_methods_are_not_transparent_wrapped_functions() -> None:
    methods = [
        ArtifactAnalysis.text_diagnostics,
        ArtifactAnalysis.summarize,
        ArtifactAnalysis.cosine_similarity,
        ArtifactAnalysis.distance,
        ArtifactAnalysis.matrix_summary,
        ArtifactAnalysis.row_summary,
        ArtifactAnalysis.feature_summary,
        ArtifactAnalysis.nearest_neighbors,
        ArtifactAnalysis.dictionary_counts,
        ArtifactAnalysis.dictionary_matches,
        ArtifactAnalysis.polarity,
        ArtifactAnalysis.valence,
        ArtifactAnalysis.classification,
        ArtifactAnalysis.generalized_difference,
        ArtifactAnalysis.generalized_difference_by,
        ArtifactAnalysis.crosstab,
        ArtifactAnalysis.kwic,
        ArtifactVisualization.histogram,
        BaseArtifact.get_previous,
        BaseArtifact.get_next,
        _MatrixArtifact.kwic,
    ]

    for method in methods:
        assert not hasattr(method, "__wrapped__")
