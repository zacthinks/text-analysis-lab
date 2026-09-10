"""Small descriptive summaries for TeAL artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


@dataclass(frozen=True)
class ArtifactSummary:
    """Ephemeral structural summary of one artifact.

    The summary intentionally reflects durable descriptor/catalog state rather
    than materializing representation data.  Representation-specific numerical
    summaries can be added separately when a concrete method requires them.
    """

    artifact_id: str
    label: str
    artifact_type: str
    status: str
    n_rows: int | None
    primary_key: tuple[str, ...]
    lineage_mode: str | None
    basis_artifact_ids: tuple[str, ...]
    operation_id: str | None
    components: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as an ordinary dictionary."""
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        """Return a one-row DataFrame representation."""
        return pd.DataFrame([self.to_dict()])


def summarize(artifact: "BaseArtifact") -> ArtifactSummary:
    """Return a lightweight in-memory structural summary of ``artifact``."""
    lineage = artifact.descriptor.get("lineage", {})
    components = artifact.descriptor.get("components", {})
    return ArtifactSummary(
        artifact_id=str(artifact.artifact_id),
        label=str(artifact.label),
        artifact_type=str(artifact.artifact_type),
        status=str(artifact.status),
        n_rows=artifact.n_rows,
        primary_key=tuple(str(value) for value in artifact.primary_key),
        lineage_mode=(
            None
            if lineage.get("lineage_mode") is None
            else str(lineage.get("lineage_mode"))
        ),
        basis_artifact_ids=tuple(
            str(value) for value in lineage.get("basis_artifact_ids", [])
        ),
        operation_id=artifact.operation_id,
        components=tuple(str(value) for value in components),
    )
