"""Ephemeral singular-value spectrum diagnostics for matrix artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from text_analysis_lab.analysis._matrix_utils import require_matrix_artifact

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def spectrum(
    artifact: BaseArtifact,
    *,
    n_components: int = 10,
    algorithm: str = "randomized",
    n_iter: int = 5,
    random_state: int | None = 42,
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
) -> pd.DataFrame:
    """Estimate the leading SVD spectrum without creating persistent artifacts.

    Energy ratios measure squared Frobenius energy of the *uncentered* input;
    unlike explained variance, this denominator includes the full source
    matrix, so a truncated spectrum does not normalize its last row to one.

    Like the existing SVD translator, this fits sklearn TruncatedSVD and
    materializes the input matrix. It is not a streaming/batched fit.
    """
    require_matrix_artifact(artifact, "spectrum()")
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be a positive integer.")
    if n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if algorithm not in {"randomized", "arpack"}:
        raise ValueError("algorithm must be 'randomized' or 'arpack'.")

    matrix = artifact.get_matrix()
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError("spectrum() requires a two-dimensional matrix.")
    rows, columns = (int(value) for value in matrix.shape)
    if rows < 2 or columns < 2:
        raise ValueError("spectrum() requires at least two rows and two features.")
    limit = columns if algorithm == "randomized" else min(rows, columns) - 1
    if n_components > limit:
        raise ValueError(
            f"n_components={n_components} exceeds the {algorithm} limit "
            f"of {limit} for matrix shape {matrix.shape}."
        )
    if sparse.issparse(matrix):
        values = matrix.data
        total_energy = float(np.sum(np.square(values.astype(float)), dtype=float))
    else:
        values = np.asarray(matrix, dtype=float)
        total_energy = float(np.sum(np.square(values), dtype=float))
    if not np.all(np.isfinite(values)) or not np.isfinite(total_energy):
        raise ValueError("spectrum() requires a matrix with finite numeric values.")

    # A zero input has no defined decomposition importance; provide explicitly
    # zero contributions instead of scipy/sklearn's potential NaNs.
    if total_energy == 0:
        singular = np.zeros(n_components, dtype=float)
        variance = np.zeros(n_components, dtype=float)
    else:
        estimator = TruncatedSVD(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
        )
        estimator.fit(matrix)
        singular = np.asarray(estimator.singular_values_, dtype=float)
        variance = np.asarray(estimator.explained_variance_ratio_, dtype=float)
        if not np.all(np.isfinite(variance)):
            # For constant rows every feature has zero sample variance,
            # despite potentially nonzero uncentered matrix energy.
            if np.all(np.var(estimator.transform(matrix), axis=0) == 0):
                variance = np.zeros_like(singular)
            else:
                raise ValueError("TruncatedSVD returned nonfinite explained variance.")
    energy = np.square(singular) / total_energy if total_energy else np.zeros_like(singular)
    energy = np.clip(energy, 0.0, 1.0)
    variance = np.clip(variance, 0.0, 1.0)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(singular) + 1, dtype=int),
            "singular_value": singular,
            "energy_ratio": energy,
            "cumulative_energy_ratio": np.minimum(np.cumsum(energy), 1.0),
            "explained_variance_ratio": variance,
            "cumulative_explained_variance_ratio": np.minimum(np.cumsum(variance), 1.0),
        }
    )
