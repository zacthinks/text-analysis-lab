"""Whole-matrix descriptive summary Analytic Method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._matrix_utils import (
    as_1d,
    is_sparse_matrix,
    require_matrix_artifact,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


@dataclass(frozen=True)
class MatrixSummary:
    artifact_id: str
    n_rows: int
    n_features: int
    nonzero_values: int
    density: float
    zero_rows: int
    zero_features: int
    sum: float
    mean: float
    min: float
    max: float
    l1_norm: float
    frobenius_norm: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


def matrix_summary(
    artifact: BaseArtifact,
    *,
    batch_size: int = 10_000,
) -> MatrixSummary:
    """Return a bounded-pass descriptive summary of a matrix artifact."""
    require_matrix_artifact(artifact, "matrix_summary()")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    n_features = len(artifact.get_data_columns())
    feature_nonzero = np.zeros(n_features, dtype=np.int64)
    n_rows = 0
    nonzero_values = 0
    zero_rows = 0
    value_sum = 0.0
    l1_norm = 0.0
    squares = 0.0
    minimum = np.inf
    maximum = -np.inf

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
        rows, features = (int(value) for value in matrix.shape)
        if rows == 0:
            continue
        if features != n_features:
            raise RuntimeError(
                "Matrix feature count changed across batches: "
                f"expected {n_features}, got {features}."
            )

        if is_sparse_matrix(matrix):
            csr = matrix.tocsr(copy=True)
            csr.eliminate_zeros()
            row_nonzero = np.asarray(csr.getnnz(axis=1), dtype=np.int64)
            feature_nonzero += np.asarray(csr.getnnz(axis=0), dtype=np.int64)
            nonzero_values += int(csr.nnz)
            value_sum += float(csr.sum())
            l1_norm += float(np.abs(csr.data).sum())
            data_float = csr.data.astype(float, copy=False)
            squares += float(np.square(data_float).sum())
            if features:
                minimum = min(minimum, float(as_1d(csr.min()).item()))
                maximum = max(maximum, float(as_1d(csr.max()).item()))
        else:
            dense = np.asarray(matrix, dtype=float)
            row_nonzero = np.count_nonzero(dense, axis=1)
            feature_nonzero += np.count_nonzero(dense, axis=0).astype(
                np.int64, copy=False
            )
            nonzero_values += int(np.count_nonzero(dense))
            value_sum += float(np.sum(dense, dtype=float))
            l1_norm += float(np.sum(np.abs(dense), dtype=float))
            squares += float(np.sum(np.square(dense), dtype=float))
            if features:
                minimum = min(minimum, float(np.min(dense)))
                maximum = max(maximum, float(np.max(dense)))

        zero_rows += int(np.count_nonzero(row_nonzero == 0))
        n_rows += rows

    n_cells = n_rows * n_features
    if n_cells == 0:
        minimum = float("nan")
        maximum = float("nan")
    return MatrixSummary(
        artifact_id=str(artifact.artifact_id),
        n_rows=int(n_rows),
        n_features=int(n_features),
        nonzero_values=int(nonzero_values),
        density=float(nonzero_values / n_cells) if n_cells else 0.0,
        zero_rows=int(zero_rows),
        zero_features=int(np.count_nonzero(feature_nonzero == 0)),
        sum=float(value_sum),
        mean=float(value_sum / n_cells) if n_cells else float("nan"),
        min=float(minimum),
        max=float(maximum),
        l1_norm=float(l1_norm),
        frobenius_norm=float(np.sqrt(squares)),
    )
