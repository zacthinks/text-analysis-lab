"""Built-in metadata aggregation utilities for reduced/span recompositions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from text_analysis_lab.core.errors import MetadataAggregationError

AggregationSpec = Mapping[str, str | Sequence[str] | Mapping[str, str]]


def _series_values(series: pd.Series) -> pd.Series:
    return series.dropna()


def aggregate_series(series: pd.Series, method: str) -> Any:
    """Aggregate one metadata column using a named built-in method."""
    method = str(method)
    values = _series_values(series)
    if method == "first":
        return series.iloc[0] if len(series) else pd.NA
    if method == "last":
        return series.iloc[-1] if len(series) else pd.NA
    if method == "first_non_null":
        return values.iloc[0] if len(values) else pd.NA
    if method == "last_non_null":
        return values.iloc[-1] if len(values) else pd.NA
    if method == "min":
        return values.min() if len(values) else pd.NA
    if method == "max":
        return values.max() if len(values) else pd.NA
    if method == "mean":
        return pd.to_numeric(values, errors="coerce").mean() if len(values) else pd.NA
    if method == "median":
        return pd.to_numeric(values, errors="coerce").median() if len(values) else pd.NA
    if method == "sum":
        return pd.to_numeric(values, errors="coerce").sum() if len(values) else 0
    if method == "count":
        return int(values.count())
    if method == "nunique":
        return int(values.nunique(dropna=True))
    if method == "any":
        return bool(values.astype(bool).any()) if len(values) else False
    if method == "all":
        return bool(values.astype(bool).all()) if len(values) else False
    if method == "mode":
        mode = values.mode(dropna=True)
        return mode.iloc[0] if len(mode) else pd.NA
    if method == "concat":
        return "".join(str(value) for value in values.tolist())
    if method == "concat_unique":
        seen: list[str] = []
        for value in values.tolist():
            text = str(value)
            if text not in seen:
                seen.append(text)
        return "".join(seen)
    if method == "list":
        return values.tolist()
    if method == "unique_list":
        out: list[Any] = []
        for value in values.tolist():
            if value not in out:
                out.append(value)
        return out
    if method == "all_equal":
        return bool(values.nunique(dropna=True) <= 1)
    raise MetadataAggregationError(f"Unknown metadata aggregation method {method!r}.")


def normalize_aggregation_spec(
    spec: AggregationSpec | None,
) -> dict[str, dict[str, str]]:
    """Normalize aggregation spec to {input_column: {output_column: method}}."""
    if not spec:
        return {}
    out: dict[str, dict[str, str]] = {}
    for input_col, rule in spec.items():
        input_col = str(input_col)
        if isinstance(rule, str):
            out[input_col] = {input_col: rule}
        elif isinstance(rule, Mapping):
            out[input_col] = {
                str(output_col): str(method) for output_col, method in rule.items()
            }
        else:
            out[input_col] = {f"{input_col}_{method}": str(method) for method in rule}
    return out


def aggregate_metadata_by_group(
    frame: pd.DataFrame,
    *,
    group_by: Sequence[str],
    spec: AggregationSpec | None,
) -> pd.DataFrame:
    """Aggregate metadata rows by output key columns for reduced-key recomposers."""
    normalized = normalize_aggregation_spec(spec)
    group_cols = [str(col) for col in group_by]
    if not group_cols:
        raise MetadataAggregationError("group_by must contain at least one column.")
    missing_group = [col for col in group_cols if col not in frame.columns]
    if missing_group:
        raise MetadataAggregationError(f"Missing group_by columns: {missing_group}")
    missing_input = [col for col in normalized if col not in frame.columns]
    if missing_input:
        raise MetadataAggregationError(
            f"Missing metadata columns for aggregation: {missing_input}"
        )

    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_cols, key_values, strict=True))
        for input_col, outputs in normalized.items():
            for output_col, method in outputs.items():
                row[output_col] = aggregate_series(group[input_col], method)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metadata_by_span(
    source_frame: pd.DataFrame,
    output_keys: pd.DataFrame,
    *,
    source_primary_key: Sequence[str],
    spec: AggregationSpec | None,
) -> pd.DataFrame:
    """Aggregate source metadata into span-key output rows.

    ``output_keys`` must contain the parent key columns plus ``<last>_start`` and
    ``<last>_end`` where ``<last>`` is the final source primary-key column.
    """
    normalized = normalize_aggregation_spec(spec)
    source_pk = [str(col) for col in source_primary_key]
    if not source_pk:
        raise MetadataAggregationError(
            "span aggregation requires a non-empty source primary key."
        )
    span_col = source_pk[-1]
    parent_cols = source_pk[:-1]
    start_col = f"{span_col}_start"
    end_col = f"{span_col}_end"
    required_source = [*parent_cols, span_col, *normalized.keys()]
    required_output = [*parent_cols, start_col, end_col]
    missing_source = [col for col in required_source if col not in source_frame.columns]
    missing_output = [col for col in required_output if col not in output_keys.columns]
    if missing_source:
        raise MetadataAggregationError(
            f"Missing source columns for span aggregation: {missing_source}"
        )
    if missing_output:
        raise MetadataAggregationError(
            f"Missing output span key columns: {missing_output}"
        )

    rows: list[dict[str, Any]] = []
    for _, out_key in output_keys.reset_index(drop=True).iterrows():
        mask = pd.Series(True, index=source_frame.index)
        for col in parent_cols:
            mask &= source_frame[col] == out_key[col]
        mask &= source_frame[span_col] >= out_key[start_col]
        mask &= source_frame[span_col] <= out_key[end_col]
        group = source_frame.loc[mask]
        row = {col: out_key[col] for col in required_output}
        for input_col, outputs in normalized.items():
            for output_col, method in outputs.items():
                row[output_col] = aggregate_series(group[input_col], method)
        rows.append(row)
    return pd.DataFrame(rows)
