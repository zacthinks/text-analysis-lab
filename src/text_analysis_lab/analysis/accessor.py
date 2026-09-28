"""Artifact-bound convenience access to ephemeral Analytic Methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from text_analysis_lab.analysis.classification import ClassificationEvaluation
    from text_analysis_lab.analysis.generalized_difference import (
        GeneralizedDifferenceGroupedResult,
        GeneralizedDifferenceResult,
    )
    from text_analysis_lab.analysis.matrix_summary import MatrixSummary
    from text_analysis_lab.analysis.polarity import PolarityScorer
    from text_analysis_lab.analysis.summary import ArtifactSummary
    from text_analysis_lab.analysis.text_diagnostics import TextDiagnosticsResult
    from text_analysis_lab.analysis.valence import ValenceScorer
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.kwic import KWICResult
    from text_analysis_lab.core.types import ColumnSelect, MetadataMode, StreamingMode
    from text_analysis_lab.dictionaries.dictionary import Dictionary


class ArtifactAnalysis:
    """Bind Analytic Methods to one artifact without mutating its artifact graph."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

    def text_diagnostics(
        self,
        *,
        text_field: str = "text",
        compare_to: BaseArtifact | str | None = None,
        strip: bool = True,
    ) -> TextDiagnosticsResult:
        from text_analysis_lab.analysis.text_diagnostics import text_diagnostics

        return text_diagnostics(
            self._artifact,
            text_field=text_field,
            compare_to=compare_to,
            strip=strip,
        )

    def summarize(self) -> ArtifactSummary:
        from text_analysis_lab.analysis.summary import summarize

        return summarize(self._artifact)

    def cosine_similarity(
        self,
        *,
        key: Any | None = None,
        position: int | None = None,
        row_name: str | None = None,
        other_key: Any | None = None,
        other_position: int | None = None,
        other_row_name: str | None = None,
    ) -> float:
        from text_analysis_lab.analysis.cosine_similarity import cosine_similarity

        return cosine_similarity(
            self._artifact,
            key=key,
            position=position,
            row_name=row_name,
            other_key=other_key,
            other_position=other_position,
            other_row_name=other_row_name,
        )

    def distance(
        self,
        *,
        key: Any | None = None,
        position: int | None = None,
        row_name: str | None = None,
        other_key: Any | None = None,
        other_position: int | None = None,
        other_row_name: str | None = None,
        metric: str | Any = "cosine",
        **metric_kwargs: Any,
    ) -> float:
        from text_analysis_lab.analysis.distance import distance

        return distance(
            self._artifact,
            key=key,
            position=position,
            row_name=row_name,
            other_key=other_key,
            other_position=other_position,
            other_row_name=other_row_name,
            metric=metric,
            **metric_kwargs,
        )

    def matrix_summary(
        self,
        *,
        batch_size: int = 10_000,
    ) -> MatrixSummary:
        from text_analysis_lab.analysis.matrix_summary import matrix_summary

        return matrix_summary(self._artifact, batch_size=batch_size)

    def row_summary(
        self,
        *,
        batch_size: int = 10_000,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.row_summary import row_summary

        return row_summary(self._artifact, batch_size=batch_size)

    def feature_summary(
        self,
        *,
        batch_size: int = 10_000,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.feature_summary import feature_summary

        return feature_summary(self._artifact, batch_size=batch_size)

    def nearest_neighbors(
        self,
        *,
        key: Any | None = None,
        position: int | None = None,
        row_name: str | None = None,
        k: int = 10,
        metric: str = "cosine",
        include_self: bool = False,
        batch_size: int = 10_000,
        context: BaseArtifact | str | None = None,
        context_data_columns: Any = False,
        context_metadata_columns: Any = False,
        context_metadata_mode: str = "none",
        **metric_kwargs: Any,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.neighbors import nearest_neighbors

        return nearest_neighbors(
            self._artifact,
            key=key,
            position=position,
            row_name=row_name,
            k=k,
            metric=metric,
            include_self=include_self,
            batch_size=batch_size,
            context=context,
            context_data_columns=context_data_columns,
            context_metadata_columns=context_metadata_columns,
            context_metadata_mode=context_metadata_mode,
            **metric_kwargs,
        )

    def dictionary_counts(
        self,
        dictionary: Dictionary,
        *,
        batch_size: int = 10_000,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.dictionary_counts import dictionary_counts

        return dictionary_counts(
            self._artifact,
            dictionary,
            batch_size=batch_size,
        )

    def dictionary_matches(self, dictionary: Dictionary) -> pd.DataFrame:
        from text_analysis_lab.analysis.dictionary_matches import dictionary_matches

        return dictionary_matches(self._artifact, dictionary)

    def polarity(
        self,
        *,
        smoothing: float = 0.5,
        zero_division: float = 0.0,
        custom: Mapping[str, PolarityScorer] | None = None,
        batch_size: int = 10_000,
        **custom_kwargs: Any,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.polarity import polarity

        return polarity(
            self._artifact,
            smoothing=smoothing,
            zero_division=zero_division,
            custom=custom,
            batch_size=batch_size,
            **custom_kwargs,
        )

    def valence(
        self,
        *,
        zero_division: float = 0.0,
        custom: Mapping[str, ValenceScorer] | None = None,
        batch_size: int = 10_000,
        **custom_kwargs: Any,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.valence import valence

        return valence(
            self._artifact,
            zero_division=zero_division,
            custom=custom,
            batch_size=batch_size,
            **custom_kwargs,
        )

    def classification(
        self,
        *,
        gold: BaseArtifact | str,
        pi: BaseArtifact | str | None = None,
        prediction_field: str = "prediction",
        gold_field: str = "label",
        pi_field: str = "pi",
    ) -> ClassificationEvaluation:
        from text_analysis_lab.analysis.classification import classification

        return classification(
            self._artifact,
            gold=gold,
            pi=pi,
            prediction_field=prediction_field,
            gold_field=gold_field,
            pi_field=pi_field,
        )

    def generalized_difference(
        self,
        *,
        gold: BaseArtifact | str,
        pi: BaseArtifact | str,
        surrogate_field: str = "prediction",
        gold_field: str = "label",
        pi_field: str = "pi",
    ) -> GeneralizedDifferenceResult:
        from text_analysis_lab.analysis.generalized_difference import (
            generalized_difference,
        )

        return generalized_difference(
            self._artifact,
            gold=gold,
            pi=pi,
            surrogate_field=surrogate_field,
            gold_field=gold_field,
            pi_field=pi_field,
        )

    def generalized_difference_by(
        self,
        *,
        gold: BaseArtifact | str,
        pi: BaseArtifact | str,
        group: BaseArtifact | str | None = None,
        group_field: str,
        surrogate_field: str = "prediction",
        gold_field: str = "label",
        pi_field: str = "pi",
    ) -> GeneralizedDifferenceGroupedResult:
        from text_analysis_lab.analysis.generalized_difference import (
            generalized_difference_by,
        )

        return generalized_difference_by(
            self._artifact,
            gold=gold,
            pi=pi,
            group=group,
            group_field=group_field,
            surrogate_field=surrogate_field,
            gold_field=gold_field,
            pi_field=pi_field,
        )

    def crosstab(
        self,
        rows: str,
        columns: str,
        *,
        margins: bool = False,
        margins_name: str = "All",
        dropna: bool = True,
        where: str | None = None,
        positions: Sequence[int] | None = None,
        limit: int | None = None,
        batch_size: int = 10_000,
    ) -> pd.DataFrame:
        from text_analysis_lab.analysis.crosstab import crosstab

        return crosstab(
            self._artifact,
            rows=rows,
            columns=columns,
            margins=margins,
            margins_name=margins_name,
            dropna=dropna,
            where=where,
            positions=positions,
            limit=limit,
            batch_size=batch_size,
        )

    def kwic(
        self,
        pattern: str,
        *,
        window: int = 5,
        before: int | None = None,
        after: int | None = None,
        valuetype: Literal["fixed", "regex"] = "fixed",
        case_sensitive: bool = False,
        enforce_word_boundary: bool = True,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        search_columns: ColumnSelect | None = None,
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        batch_size: int = 100_000,
        streaming_mode: StreamingMode = "auto",
        target_matches: int | None = None,
    ) -> KWICResult:
        from text_analysis_lab.core.kwic import keyword_in_context

        return keyword_in_context(
            self._artifact,
            pattern,
            window=window,
            before=before,
            after=after,
            valuetype=valuetype,
            case_sensitive=case_sensitive,
            enforce_word_boundary=enforce_word_boundary,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            search_columns=search_columns,
            where=where,
            order_by=order_by,
            positions=positions,
            sample_n=sample_n,
            sample_frac=sample_frac,
            random_state=random_state,
            limit=limit,
            batch_size=batch_size,
            streaming_mode=streaming_mode,
            target_matches=target_matches,
        )
