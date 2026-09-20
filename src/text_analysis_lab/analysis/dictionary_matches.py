"""Inspect which matrix features match a content dictionary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from text_analysis_lab.analysis._dictionary_utils import (
    category_membership,
    matrix_features,
)
from text_analysis_lab.dictionaries import Dictionary

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def dictionary_matches(
    artifact: BaseArtifact,
    dictionary: Dictionary,
) -> pd.DataFrame:
    """Return one row for each matrix-feature/dictionary-key match."""
    if not isinstance(dictionary, Dictionary):
        raise TypeError("dictionary_matches() requires a teal.dictionaries.Dictionary.")
    features = matrix_features(artifact, "dictionary_matches()")
    keys, membership = category_membership(features, dictionary)
    coo = membership.tocoo()
    rows = [
        {
            "key": keys[int(column)],
            "feature": features[int(row)],
            "feature_index": int(row),
        }
        for row, column in zip(coo.row, coo.col, strict=True)
    ]
    if not rows:
        return pd.DataFrame(columns=["key", "feature", "feature_index"])
    return pd.DataFrame(rows).sort_values(
        ["key", "feature_index"], kind="stable", ignore_index=True
    )
