"""Category counts/sums from matrix artifacts using content dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from text_analysis_lab.analysis._dictionary_utils import (
    category_membership,
    category_scores,
    matrix_features,
)
from text_analysis_lab.dictionaries import Dictionary

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def dictionary_counts(
    artifact: "BaseArtifact",
    dictionary: Dictionary,
    *,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Sum matrix values for each dictionary key/category by artifact row.

    On a count DTM these are ordinary dictionary term counts.  On a weighted
    matrix the same operation sums the stored matrix weights instead.
    """
    if not isinstance(dictionary, Dictionary):
        raise TypeError("dictionary_counts() requires a teal.dictionaries.Dictionary.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    features = matrix_features(artifact, "dictionary_counts()")
    keys, membership = category_membership(features, dictionary)
    primary_key = [str(value) for value in artifact.primary_key]
    collisions = sorted(set(keys).intersection({*primary_key, "_position"}))
    if collisions:
        raise ValueError(
            "Dictionary keys collide with structural output columns: "
            f"{collisions}."
        )
    columns = [*primary_key, "_position", *keys]
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
        if info.empty:
            continue
        scores = category_scores(batch["matrix"], membership)
        result = info.loc[:, [*primary_key, "_position"]].copy()
        for index, key in enumerate(keys):
            result[key] = scores[:, index]
        frames.append(result)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, columns]
