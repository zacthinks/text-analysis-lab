"""Batchwise categorical cross-tabulation for TeAL artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.analysis._tabular_utils import (
    batch_query_columns,
    resolve_tabular_fields,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_MISSING = object()


def crosstab(
    artifact: BaseArtifact,
    *,
    rows: str,
    columns: str,
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    where: str | None = None,
    positions: Sequence[int] | None = None,
    limit: int | None = None,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """Count combinations of two fields without materializing corpus rows.

    The result has the familiar pandas crosstab shape, but TeAL requests only the
    two selected fields and accumulates counts one artifact batch at a time.
    Result memory therefore grows with the number of observed category pairs.
    """
    if not isinstance(margins_name, str) or not margins_name:
        raise ValueError("margins_name must be a non-empty string.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    row_field, column_field = resolve_tabular_fields(artifact, rows, columns)
    data_columns, metadata_columns = batch_query_columns((row_field, column_field))

    counts: dict[tuple[Any, Any], int] = defaultdict(int)
    row_values: list[Any] = []
    column_values: list[Any] = []
    seen_rows: set[Any] = set()
    seen_columns: set[Any] = set()
    for batch in artifact.iter_table_batches(
        batch_size=int(batch_size),
        key_columns=False,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode="full" if metadata_columns is not False else "none",
        where=where,
        positions=positions,
        limit=limit,
        include_position=False,
    ):
        left = batch[row_field.output_name]
        right = batch[column_field.output_name]
        for left_value, right_value in zip(left, right, strict=True):
            left_missing = bool(pd.isna(left_value))
            right_missing = bool(pd.isna(right_value))
            if dropna and (left_missing or right_missing):
                continue
            left_key = _MISSING if left_missing else left_value
            right_key = _MISSING if right_missing else right_value
            counts[(left_key, right_key)] += 1
            if left_key not in seen_rows:
                seen_rows.add(left_key)
                row_values.append(left_key)
            if right_key not in seen_columns:
                seen_columns.add(right_key)
                column_values.append(right_key)

    frame = pd.DataFrame(
        [
            [counts.get((row_value, column_value), 0) for column_value in column_values]
            for row_value in row_values
        ],
        index=pd.Index([_display_value(value) for value in row_values], name=rows),
        columns=pd.Index(
            [_display_value(value) for value in column_values], name=columns
        ),
        dtype="int64",
    )
    if margins:
        frame[margins_name] = frame.sum(axis=1).astype("int64")
        totals = frame.sum(axis=0).astype("int64")
        frame.loc[margins_name] = totals
    return frame


def _display_value(value: Any) -> Any:
    return "<NA>" if value is _MISSING else value
