"""Validation/helpers for analyses over dictionary-translated count matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from text_analysis_lab.analysis._matrix_utils import require_matrix_artifact

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

REQUIRED_COUNT_METADATA = ("matched", "unmatched", "total")


def require_dictionary_count_artifact(
    artifact: BaseArtifact,
    *,
    method: str,
) -> list[str]:
    require_matrix_artifact(artifact, method)
    available = set(artifact.get_metadata_columns())
    missing = [name for name in REQUIRED_COUNT_METADATA if name not in available]
    if missing:
        raise ValueError(
            f"{method} requires a dictionary-translated count matrix with local "
            f"metadata {list(REQUIRED_COUNT_METADATA)}; missing {missing}."
        )
    return [str(value) for value in artifact.get_data_columns()]


def row_sums_int(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.int64, copy=False)
    return np.asarray(matrix).sum(axis=1).astype(np.int64, copy=False)


def validate_batch_counts(
    matrix: Any, info: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name in REQUIRED_COUNT_METADATA:
        if name not in info.columns:
            raise ValueError(
                f"Dictionary-translated artifact batch is missing {name!r} metadata."
            )
    matched = np.asarray(info["matched"], dtype=np.int64)
    unmatched = np.asarray(info["unmatched"], dtype=np.int64)
    total = np.asarray(info["total"], dtype=np.int64)
    if np.any(matched < 0) or np.any(unmatched < 0) or np.any(total < 0):
        raise ValueError("Dictionary count metadata must be non-negative.")
    if not np.array_equal(matched + unmatched, total):
        raise ValueError(
            "Dictionary count metadata invariant failed: matched + unmatched != total."
        )
    if not np.array_equal(row_sums_int(matrix), matched):
        raise ValueError(
            "Dictionary count matrix invariant failed: row sum != matched."
        )
    return matched, unmatched, total


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    zero_division: float,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(numerator.shape, float(zero_division), dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result
