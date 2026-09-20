"""Batchwise histogram data and plotting for TeAL artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._tabular_utils import (
    ResolvedField,
    batch_query_columns,
    resolve_tabular_fields,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


HistogramOutput = Literal["plot", "data", "result"]
_ALL = object()
_MISSING = object()


@dataclass(frozen=True)
class HistogramResult:
    field: str
    by: str | None
    bin_edges: np.ndarray
    counts: dict[Any, np.ndarray]
    observed_count: int
    missing_count: int
    underflow_count: int
    overflow_count: int

    def to_frame(self) -> pd.DataFrame:
        """Return one compact row per group/bin."""
        rows: list[dict[str, Any]] = []
        for group, values in self.counts.items():
            for index, count in enumerate(values):
                record: dict[str, Any] = {
                    "bin_left": float(self.bin_edges[index]),
                    "bin_right": float(self.bin_edges[index + 1]),
                    "count": int(count),
                }
                if self.by is not None:
                    record[self.by] = _display_group(group)
                rows.append(record)
        columns = ([] if self.by is None else [self.by]) + [
            "bin_left",
            "bin_right",
            "count",
        ]
        return pd.DataFrame(rows, columns=columns)

    def plot(
        self,
        *,
        ax: Any | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str = "Records",
        alpha: float = 0.45,
        figsize: tuple[float, float] = (9.0, 4.0),
    ) -> Any:
        """Draw the already-aggregated histogram and return its Matplotlib Axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "Plot output requires matplotlib. Use output='data' without it."
            ) from exc
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        widths = np.diff(self.bin_edges)
        for group, values in self.counts.items():
            label = None if self.by is None else str(_display_group(group))
            ax.bar(
                self.bin_edges[:-1],
                values,
                width=widths,
                align="edge",
                alpha=1.0 if self.by is None else alpha,
                label=label,
            )
        ax.set_xlabel(xlabel or self.field.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if self.by is not None and self.counts:
            ax.legend(title=self.by)
        return ax


def histogram(
    artifact: BaseArtifact,
    *,
    field: str,
    by: str | None = None,
    bins: int | Sequence[float] = 10,
    range: tuple[float, float] | None = None,
    dropna: bool = True,
    where: str | None = None,
    positions: Sequence[int] | None = None,
    limit: int | None = None,
    batch_size: int = 10_000,
    output: HistogramOutput = "plot",
    ax: Any | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Records",
    alpha: float = 0.45,
    figsize: tuple[float, float] = (9.0, 4.0),
) -> Any:
    """Aggregate a numeric histogram batchwise and return data, result, or plot.

    Integer ``bins`` uses a two-pass scan: the first pass finds the finite global
    range, and the second accumulates exact counts using shared edges. Explicit
    bin edges require only one counting pass. Corpus rows are never concatenated.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if output not in {"plot", "data", "result"}:
        raise ValueError("output must be 'plot', 'data', or 'result'.")
    resolved = resolve_tabular_fields(artifact, field, *([] if by is None else [by]))
    value_field = resolved[0]
    group_field = None if by is None else resolved[1]
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("Integer bins must be positive.")
        observed_range = range or _numeric_range(
            artifact,
            value_field=value_field,
            group_field=group_field,
            dropna=dropna,
            where=where,
            positions=positions,
            limit=limit,
            batch_size=int(batch_size),
        )
        if observed_range is None:
            edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            edges = np.histogram_bin_edges(
                np.asarray(observed_range, dtype="float64"), bins=bins, range=range
            )
    else:
        edges = np.asarray(list(bins), dtype="float64")
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(
                "Explicit histogram bins must be finite one-dimensional edges."
            )
        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                "Explicit histogram bin edges must be strictly increasing."
            )

    counts, observed, missing, underflow, overflow = _count_bins(
        artifact,
        value_field=value_field,
        group_field=group_field,
        edges=edges,
        dropna=dropna,
        where=where,
        positions=positions,
        limit=limit,
        batch_size=int(batch_size),
    )
    result = HistogramResult(
        field=field,
        by=by,
        bin_edges=edges,
        counts=dict(counts),
        observed_count=observed,
        missing_count=missing,
        underflow_count=underflow,
        overflow_count=overflow,
    )
    if output == "result":
        return result
    if output == "data":
        return result.to_frame()
    return result.plot(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        alpha=alpha,
        figsize=figsize,
    )


def _iter_values(
    artifact: BaseArtifact,
    *,
    value_field: ResolvedField,
    group_field: ResolvedField | None,
    where: str | None,
    positions: Sequence[int] | None,
    limit: int | None,
    batch_size: int,
):
    fields = (value_field,) if group_field is None else (value_field, group_field)
    data_columns, metadata_columns = batch_query_columns(fields)
    yield from artifact.iter_table_batches(
        batch_size=batch_size,
        key_columns=False,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode="full" if metadata_columns is not False else "none",
        where=where,
        positions=positions,
        limit=limit,
        include_position=False,
    )


def _numeric_range(
    artifact: BaseArtifact,
    *,
    value_field: ResolvedField,
    group_field: ResolvedField | None,
    dropna: bool,
    where: str | None,
    positions: Sequence[int] | None,
    limit: int | None,
    batch_size: int,
) -> tuple[float, float] | None:
    minimum: float | None = None
    maximum: float | None = None
    for batch in _iter_values(
        artifact,
        value_field=value_field,
        group_field=group_field,
        where=where,
        positions=positions,
        limit=limit,
        batch_size=batch_size,
    ):
        numeric = _numeric_values(
            batch[value_field.output_name], field=value_field.requested_name
        )
        valid = np.isfinite(numeric)
        if group_field is not None and dropna:
            valid &= ~batch[group_field.output_name].isna().to_numpy()
        if not valid.any():
            continue
        current = numeric[valid]
        batch_min = float(current.min())
        batch_max = float(current.max())
        minimum = batch_min if minimum is None else min(minimum, batch_min)
        maximum = batch_max if maximum is None else max(maximum, batch_max)
    return None if minimum is None else (minimum, cast(float, maximum))


def _count_bins(
    artifact: BaseArtifact,
    *,
    value_field: ResolvedField,
    group_field: ResolvedField | None,
    edges: np.ndarray,
    dropna: bool,
    where: str | None,
    positions: Sequence[int] | None,
    limit: int | None,
    batch_size: int,
) -> tuple[dict[Any, np.ndarray], int, int, int, int]:
    counts: dict[Any, np.ndarray] = defaultdict(
        lambda: np.zeros(len(edges) - 1, dtype="int64")
    )
    observed = missing = underflow = overflow = 0
    for batch in _iter_values(
        artifact,
        value_field=value_field,
        group_field=group_field,
        where=where,
        positions=positions,
        limit=limit,
        batch_size=batch_size,
    ):
        numeric = _numeric_values(
            batch[value_field.output_name], field=value_field.requested_name
        )
        finite = np.isfinite(numeric)
        missing += int((~finite).sum())
        groups = (
            np.full(len(batch), _ALL, dtype=object)
            if group_field is None
            else batch[group_field.output_name].astype("object").to_numpy()
        )
        if group_field is not None:
            group_missing = pd.isna(groups)
            if dropna:
                missing += int((finite & group_missing).sum())
                finite &= ~group_missing
            else:
                groups[group_missing] = _MISSING
        if not finite.any():
            continue
        values = numeric[finite]
        selected_groups = groups[finite]
        observed += len(values)
        underflow += int((values < edges[0]).sum())
        overflow += int((values > edges[-1]).sum())
        for group in dict.fromkeys(selected_groups.tolist()):
            mask = selected_groups == group
            counts[group] += np.histogram(values[mask], bins=edges)[0].astype("int64")
    if group_field is None and _ALL not in counts:
        counts[_ALL] = np.zeros(len(edges) - 1, dtype="int64")
    return counts, observed, missing, underflow, overflow


def _numeric_values(series: pd.Series, *, field: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        value = series.loc[invalid].iloc[0]
        raise ValueError(
            f"Histogram field {field!r} contains non-numeric value {value!r}."
        )
    return numeric.to_numpy(dtype="float64", na_value=np.nan)


def _display_group(value: Any) -> Any:
    if value is _ALL:
        return None
    if value is _MISSING:
        return "<NA>"
    return value
