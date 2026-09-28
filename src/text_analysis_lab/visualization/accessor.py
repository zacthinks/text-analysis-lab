"""Artifact-bound convenience access to visualization methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.types import ColumnSelect, MetadataMode
    from text_analysis_lab.visualization.histogram import HistogramOutput


class ArtifactVisualization:
    """Bind scalable visualization methods to one artifact."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

    def scatter(
        self,
        *,
        embeddings: BaseArtifact,
        color: str | None = None,
        hover_fields: str | Sequence[str] | None = None,
        include_coordinates=False,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        max_length: int | None = 600,
        wrap_length: int | None = 200,
        width: int | str | None = None,
        height: int | None = 700,
    ) -> Any:
        from text_analysis_lab.visualization.scatter import scatter

        return scatter(
            self._artifact,
            embeddings=embeddings,
            color=color,
            hover_fields=hover_fields,
            include_coordinates=include_coordinates,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            where=where,
            order_by=order_by,
            positions=positions,
            sample_n=sample_n,
            sample_frac=sample_frac,
            random_state=random_state,
            limit=limit,
            max_length=max_length,
            wrap_length=wrap_length,
            width=width,
            height=height,
        )

    def histogram(
        self,
        field: str,
        *,
        by: str | None = None,
        bins: int | Sequence[float] = 10,
        range: tuple[float, float] | None = None,
        dropna: bool = True,
        where: str | None = None,
        positions: Sequence[int] | None = None,
        limit: int | None = None,
        batch_size: int = 10_000,
        output: HistogramOutput = "plot",
        ax: Any | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str = "Records",
        alpha: float = 0.45,
        figsize: tuple[float, float] = (9.0, 4.0),
    ) -> Any:
        from text_analysis_lab.visualization.histogram import histogram

        return histogram(
            self._artifact,
            field=field,
            by=by,
            bins=bins,
            range=range,
            dropna=dropna,
            where=where,
            positions=positions,
            limit=limit,
            batch_size=batch_size,
            output=output,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            alpha=alpha,
            figsize=figsize,
        )
