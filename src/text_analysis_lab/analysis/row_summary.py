"""Per-row descriptive summaries for matrix artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from text_analysis_lab.analysis._matrix_utils import matrix_block_stats, require_matrix_artifact

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def row_summary(
    artifact: "BaseArtifact",
    *,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Return keyed descriptive statistics for every matrix row."""
    require_matrix_artifact(artifact, "row_summary()")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    primary_key = [str(value) for value in artifact.primary_key]
    columns = [
        *primary_key,
        "_position",
        "nonzero_features",
        "nonzero_fraction",
        "sum",
        "mean",
        "min",
        "max",
        "l1_norm",
        "l2_norm",
    ]
    n_features = len(artifact.get_data_columns())
    frames: list[pd.DataFrame] = []

    for batch in artifact.iter_batches(
        batch_size=int(batch_size),
        key_columns=True,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        form="native",
        include_position=True,
    ):
        info = batch["info"].reset_index(drop=True)
        matrix = batch["matrix"]
        if info.empty:
            continue
        stats = matrix_block_stats(matrix)
        result = info.loc[:, [*primary_key, "_position"]].copy()
        result["nonzero_features"] = stats["nonzero"]
        result["nonzero_fraction"] = (
            stats["nonzero"] / float(n_features) if n_features else 0.0
        )
        result["sum"] = stats["sum"]
        result["mean"] = stats["sum"] / float(n_features) if n_features else float("nan")
        result["min"] = stats["min"]
        result["max"] = stats["max"]
        result["l1_norm"] = stats["l1"]
        result["l2_norm"] = stats["l2"]
        frames.append(result)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, columns]
