"""Cosine-similarity Analytic Method for matrix artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from text_analysis_lab.analysis._matrix_utils import require_matrix_artifact, resolve_position
from text_analysis_lab.core.errors import UnsupportedArtifactOperationError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def cosine_similarity(
    artifact: "BaseArtifact",
    *,
    key: Any | None = None,
    position: int | None = None,
    row_name: str | None = None,
    other_key: Any | None = None,
    other_position: int | None = None,
    other_row_name: str | None = None,
) -> float:
    """Return cosine similarity between two rows of one matrix artifact."""
    require_matrix_artifact(artifact, "cosine_similarity()")
    left_position = resolve_position(
        artifact,
        key=key,
        position=position,
        row_name=row_name,
        argument_name="row",
    )
    right_position = resolve_position(
        artifact,
        key=other_key,
        position=other_position,
        row_name=other_row_name,
        argument_name="other_row",
    )

    try:
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise UnsupportedArtifactOperationError(
            "cosine_similarity() requires scikit-learn."
        ) from exc

    left = artifact.get_matrix(positions=[left_position])
    right = artifact.get_matrix(positions=[right_position])
    value = sklearn_cosine_similarity(left, right)
    return float(value[0, 0])
