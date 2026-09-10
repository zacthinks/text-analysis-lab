"""Shared internal helpers for matrix Analytic Methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from text_analysis_lab.core.errors import UnsupportedArtifactOperationError
from text_analysis_lab.core.types import ArtifactType

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_MATRIX_TYPES = {ArtifactType.SPARSE_MATRIX, ArtifactType.DENSE_MATRIX}


def require_matrix_artifact(artifact: "BaseArtifact", method: str) -> None:
    if artifact.artifact_type not in _MATRIX_TYPES or not hasattr(artifact, "get_matrix"):
        raise UnsupportedArtifactOperationError(
            f"{method} requires a sparse_matrix or dense_matrix artifact."
        )


def resolve_position(
    artifact: "BaseArtifact",
    *,
    key: Any | None,
    position: int | None,
    row_name: str | None = None,
    argument_name: str = "row",
) -> int:
    selectors = sum(value is not None for value in (key, position, row_name))
    if selectors != 1:
        raise ValueError(
            f"Specify exactly one key, position, or row_name for {argument_name}."
        )
    if row_name is not None:
        resolver = getattr(artifact, "position_by_row_name", None)
        if resolver is None:
            raise UnsupportedArtifactOperationError(
                f"Artifact does not support named rows for {argument_name}."
            )
        return int(resolver(row_name))
    if position is None:
        return int(artifact.position_by_key(key))

    resolved = int(position)
    n_rows = artifact.n_rows
    if resolved < 0 or (n_rows is not None and resolved >= int(n_rows)):
        raise IndexError(f"Artifact position is out of range: {resolved}.")
    return resolved


def as_1d(value: Any) -> np.ndarray:
    """Normalize dense/sparse one-row/one-column reductions to a 1-D ndarray."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value).reshape(-1)


def is_sparse_matrix(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a package dependency
        return False
    return bool(sparse.issparse(value))


def matrix_block_stats(matrix: Any) -> dict[str, np.ndarray]:
    """Return vectorized per-row statistics for one matrix block."""
    n_rows, n_features = (int(v) for v in matrix.shape)
    if n_rows == 0:
        empty = np.empty(0, dtype=float)
        return {
            "nonzero": empty.astype(np.int64),
            "sum": empty,
            "min": empty,
            "max": empty,
            "l1": empty,
            "l2": empty,
        }

    if is_sparse_matrix(matrix):
        csr = matrix.tocsr(copy=True)
        csr.eliminate_zeros()
        nonzero = np.asarray(csr.getnnz(axis=1), dtype=np.int64)
        sums = as_1d(csr.sum(axis=1)).astype(float, copy=False)
        abs_values = csr.copy()
        abs_values.data = np.abs(abs_values.data)
        l1 = as_1d(abs_values.sum(axis=1)).astype(float, copy=False)
        squared = csr.astype(float, copy=True)
        squared.data = np.square(squared.data)
        l2 = np.sqrt(as_1d(squared.sum(axis=1)).astype(float, copy=False))
        if n_features == 0:
            mins = np.full(n_rows, np.nan)
            maxs = np.full(n_rows, np.nan)
        else:
            mins = as_1d(csr.min(axis=1)).astype(float, copy=False)
            maxs = as_1d(csr.max(axis=1)).astype(float, copy=False)
    else:
        dense = np.asarray(matrix, dtype=float)
        nonzero = np.count_nonzero(dense, axis=1).astype(np.int64, copy=False)
        sums = np.sum(dense, axis=1, dtype=float)
        l1 = np.sum(np.abs(dense), axis=1, dtype=float)
        l2 = np.sqrt(np.sum(np.square(dense), axis=1, dtype=float))
        if n_features == 0:
            mins = np.full(n_rows, np.nan)
            maxs = np.full(n_rows, np.nan)
        else:
            mins = np.min(dense, axis=1).astype(float, copy=False)
            maxs = np.max(dense, axis=1).astype(float, copy=False)

    return {
        "nonzero": nonzero,
        "sum": sums,
        "min": mins,
        "max": maxs,
        "l1": l1,
        "l2": l2,
    }
