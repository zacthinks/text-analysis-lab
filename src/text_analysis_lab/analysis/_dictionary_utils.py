"""Matrix helpers for dictionary Analytic Methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from text_analysis_lab.analysis._matrix_utils import (
    as_1d,
    is_sparse_matrix,
    require_matrix_artifact,
)
from text_analysis_lab.dictionaries.matching import category_membership, valence_vectors

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

__all__ = [
    "category_membership",
    "category_scores",
    "matched_sums",
    "matrix_features",
    "row_sums",
    "valence_vectors",
    "weighted_scores",
]


def matrix_features(artifact: BaseArtifact, method: str) -> list[str]:
    require_matrix_artifact(artifact, method)
    return [str(value) for value in artifact.get_data_columns()]


def category_scores(matrix: Any, membership: sparse.csr_matrix) -> np.ndarray:
    values = matrix @ membership
    if sparse.issparse(values):
        return values.toarray().astype(float, copy=False)
    return np.asarray(values, dtype=float)


def weighted_scores(matrix: Any, weights: np.ndarray) -> np.ndarray:
    values = matrix @ np.asarray(weights, dtype=float).reshape(-1, 1)
    return as_1d(values).astype(float, copy=False)


def row_sums(matrix: Any) -> np.ndarray:
    if is_sparse_matrix(matrix):
        return as_1d(matrix.sum(axis=1)).astype(float, copy=False)
    return np.sum(np.asarray(matrix, dtype=float), axis=1, dtype=float)


def matched_sums(matrix: Any, matched: np.ndarray) -> np.ndarray:
    return weighted_scores(matrix, np.asarray(matched, dtype=float))
