"""Ephemeral text-length diagnostics for TeAL artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


LengthUnit = Literal["characters", "words"]


def text_length(
    artifact: BaseArtifact,
    *,
    text_field: str = "text",
    unit: LengthUnit = "characters",
    log: bool = False,
    key_columns: bool | str | Sequence[str] = True,
    include_position: bool = True,
    where: str | None = None,
    positions: Sequence[int] | None = None,
    limit: int | None = None,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Return row-level text lengths without creating an Artifact.

    ``unit='characters'`` uses pandas string length. ``unit='words'`` uses the
    same simple whitespace-token definition as TeAL's legacy TextLengthMapper.
    When ``log=True``, the returned ``text_length`` is ``log1p(length)`` so zero
    lengths remain defined.
    """
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string.")
    if unit not in {"characters", "words"}:
        raise ValueError("unit must be 'characters' or 'words'.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    outputs: list[pd.DataFrame] = []
    for batch in artifact.iter_table_batches(
        batch_size=int(batch_size),
        key_columns=key_columns,
        data_columns=text_field,
        metadata_columns=False,
        metadata_mode="none",
        where=where,
        positions=positions,
        limit=limit,
        include_position=include_position,
    ):
        if text_field not in batch.columns:
            raise ArtifactError(
                f"Artifact {artifact.artifact_id} query did not provide text field {text_field!r}."
            )
        source = batch[text_field].fillna("").astype("string")
        if unit == "characters":
            lengths = source.str.len().astype("int64")
        else:
            # ``str.split`` with no explicit pattern follows whitespace runs,
            # matching Python's ``str.split()`` behavior used by the legacy mapper.
            lengths = source.str.split().str.len().astype("int64")

        structural = batch.drop(columns=[text_field]).reset_index(drop=True)
        values = lengths.reset_index(drop=True)
        structural["text_length"] = (
            np.log1p(values.to_numpy(dtype="float64")) if log else values.to_numpy()
        )
        outputs.append(structural)

    if outputs:
        return pd.concat(outputs, ignore_index=True)

    columns: list[str] = []
    if key_columns is True:
        columns.extend(str(name) for name in artifact.primary_key)
    elif isinstance(key_columns, str):
        columns.append(key_columns)
    elif key_columns is not False:
        columns.extend(str(name) for name in key_columns)
    if include_position:
        columns.append("_position")
    columns.append("text_length")
    return pd.DataFrame(columns=columns)
