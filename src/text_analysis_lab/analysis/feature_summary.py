"""Per-feature descriptive summaries for matrix artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._matrix_utils import as_1d, is_sparse_matrix, require_matrix_artifact

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def feature_summary(
    artifact: "BaseArtifact",
    *,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Return descriptive statistics for each matrix feature/column.

    For a count matrix, ``sum`` is term frequency and ``nonzero_rows`` is
    document/row frequency. The generic names remain valid for weighted and
    dense representations as well.
    """
    require_matrix_artifact(artifact, "feature_summary()")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    features = [str(value) for value in artifact.get_data_columns()]
    n_features = len(features)
    columns = [
        "feature",
        "feature_index",
        "nonzero_rows",
        "nonzero_fraction",
        "sum",
        "mean",
        "min",
        "max",
        "l1_norm",
        "l2_norm",
    ]
    if n_features == 0:
        return pd.DataFrame(columns=columns)

    nonzero = np.zeros(n_features, dtype=np.int64)
    sums = np.zeros(n_features, dtype=float)
    l1 = np.zeros(n_features, dtype=float)
    squares = np.zeros(n_features, dtype=float)
    minima = np.full(n_features, np.inf, dtype=float)
    maxima = np.full(n_features, -np.inf, dtype=float)
    n_rows = 0

    for batch in artifact.iter_batches(
        batch_size=int(batch_size),
        key_columns=False,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        form="native",
        include_position=False,
    ):
        matrix = batch["matrix"]
        rows = int(matrix.shape[0])
        if rows == 0:
            continue
        if int(matrix.shape[1]) != n_features:
            raise RuntimeError(
                "Matrix feature count changed across batches: "
                f"expected {n_features}, got {matrix.shape[1]}."
            )

        if is_sparse_matrix(matrix):
            csr = matrix.tocsr(copy=True)
            csr.eliminate_zeros()
            nonzero += np.asarray(csr.getnnz(axis=0), dtype=np.int64)
            sums += as_1d(csr.sum(axis=0)).astype(float, copy=False)
            abs_values = csr.copy()
            abs_values.data = np.abs(abs_values.data)
            l1 += as_1d(abs_values.sum(axis=0)).astype(float, copy=False)
            squared = csr.astype(float, copy=True)
            squared.data = np.square(squared.data)
            squares += as_1d(squared.sum(axis=0)).astype(float, copy=False)
            batch_min = as_1d(csr.min(axis=0)).astype(float, copy=False)
            batch_max = as_1d(csr.max(axis=0)).astype(float, copy=False)
        else:
            dense = np.asarray(matrix, dtype=float)
            nonzero += np.count_nonzero(dense, axis=0).astype(np.int64, copy=False)
            sums += np.sum(dense, axis=0, dtype=float)
            l1 += np.sum(np.abs(dense), axis=0, dtype=float)
            squares += np.sum(np.square(dense), axis=0, dtype=float)
            batch_min = np.min(dense, axis=0).astype(float, copy=False)
            batch_max = np.max(dense, axis=0).astype(float, copy=False)

        minima = np.minimum(minima, batch_min)
        maxima = np.maximum(maxima, batch_max)
        n_rows += rows

    if n_rows == 0:
        minima[:] = np.nan
        maxima[:] = np.nan

    result = pd.DataFrame(
        {
            "feature": features,
            "feature_index": np.arange(n_features, dtype=np.int64),
            "nonzero_rows": nonzero,
            "nonzero_fraction": nonzero / float(n_rows) if n_rows else np.nan,
            "sum": sums,
            "mean": sums / float(n_rows) if n_rows else np.nan,
            "min": minima,
            "max": maxima,
            "l1_norm": l1,
            "l2_norm": np.sqrt(squares),
        }
    )
    return result.loc[:, columns]
