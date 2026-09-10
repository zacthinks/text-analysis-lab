"""Pairwise distance Analytic Method for matrix artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from text_analysis_lab.analysis._matrix_utils import require_matrix_artifact, resolve_position
from text_analysis_lab.core.errors import UnsupportedArtifactOperationError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def distance(
    artifact: "BaseArtifact",
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
    """Return the distance between two rows of one matrix artifact.

    ``metric`` and additional keyword arguments are passed to scikit-learn's
    ``pairwise_distances``. Sparse artifacts therefore support the sparse-safe
    metrics accepted there; dense artifacts may additionally use compatible
    SciPy metrics.
    """
    require_matrix_artifact(artifact, "distance()")
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
        from sklearn.metrics import pairwise_distances
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise UnsupportedArtifactOperationError(
            "distance() requires scikit-learn."
        ) from exc

    left = artifact.get_matrix(positions=[left_position])
    right = artifact.get_matrix(positions=[right_position])
    value = pairwise_distances(left, right, metric=metric, **metric_kwargs)
    return float(value[0, 0])
