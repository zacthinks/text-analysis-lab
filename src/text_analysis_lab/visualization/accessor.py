"""Artifact-bound convenience access to visualization methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.visualization.histogram import HistogramOutput


class ArtifactVisualization:
    """Bind scalable visualization methods to one artifact."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

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
