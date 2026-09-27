"""Artifact-bound convenience access to visualization methods."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from text_analysis_lab.visualization.histogram import histogram

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class ArtifactVisualization:
    """Bind scalable visualization methods to one artifact."""

    def __init__(self, artifact: BaseArtifact) -> None:
        self._artifact = artifact

    @wraps(histogram)
    def histogram(self, field: str, **kwargs: Any):
        return histogram(self._artifact, field=field, **kwargs)
