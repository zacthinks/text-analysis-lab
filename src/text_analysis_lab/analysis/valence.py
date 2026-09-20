"""Distributional valence diagnostics over dictionary-translated count matrices."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.analysis._dictionary_translation_utils import (
    require_dictionary_count_artifact,
    safe_divide,
    validate_batch_counts,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

ValenceScorer = Callable[..., Any]


def valence(
    artifact: BaseArtifact,
    *,
    zero_division: float = 0.0,
    custom: Mapping[str, ValenceScorer] | None = None,
    batch_size: int = 10_000,
    **custom_kwargs: Any,
) -> pd.DataFrame:
    """Summarize the empirical distribution of matched dictionary values.

    Matrix columns are numeric dictionary scores and cells are hit counts. The
    analysis therefore retains the distinction between, for example, six hits
    at -1 and one hit at -6 while still providing familiar aggregate summaries.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    features = require_dictionary_count_artifact(artifact, method="valence()")
    try:
        values = np.asarray([float(feature) for feature in features], dtype=float)
    except ValueError as exc:
        raise ValueError(
            "valence() requires numeric DictionaryTranslator value columns."
        ) from exc
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("valence() requires finite numeric value columns.")
    if values.size and np.any(np.diff(values) < 0):
        raise ValueError(
            "valence() expects dictionary value columns in ascending order."
        )

    custom = {} if custom is None else dict(custom)
    standard = {
        "negative",
        "neutral",
        "positive",
        "matched",
        "unmatched",
        "coverage",
        "total",
        "weighted_sum",
        "mean_matched",
        "mean_all",
        "std_matched",
        "min_matched",
        "q25",
        "median",
        "q75",
        "max_matched",
    }
    protected = set(artifact.primary_key) | {"_position", *standard}
    custom_columns = [str(name) for name in custom]
    conflicts = [name for name in custom_columns if not name or name in protected]
    if conflicts:
        raise ValueError(
            f"Invalid/conflicting custom valence formula names: {conflicts}."
        )
    for name, scorer in custom.items():
        if not callable(scorer):
            raise TypeError(f"Custom valence formula {name!r} must be callable.")

    primary_key = [str(value) for value in artifact.primary_key]
    result_columns = [
        *primary_key,
        "_position",
        "negative",
        "neutral",
        "positive",
        "matched",
        "unmatched",
        "coverage",
        "total",
        "weighted_sum",
        "mean_matched",
        "mean_all",
        "std_matched",
        "min_matched",
        "q25",
        "median",
        "q75",
        "max_matched",
        *custom_columns,
    ]
    frames: list[pd.DataFrame] = []

    for batch in artifact.iter_batches(
        batch_size=int(batch_size),
        key_columns=True,
        data_columns=True,
        metadata_columns=list(("matched", "unmatched", "total")),
        metadata_mode="local",
        form="native",
        include_position=True,
    ):
        info = batch["info"].reset_index(drop=True)
        if info.empty:
            continue
        matrix = batch["matrix"]
        counts = (
            matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
        )
        counts = counts.astype(float, copy=False)
        counts.sort_indices()
        matched, unmatched, total = validate_batch_counts(counts, info)
        weighted_sum = np.asarray(counts @ values.reshape(-1, 1)).reshape(-1)
        squared_sum = np.asarray(counts @ np.square(values).reshape(-1, 1)).reshape(-1)
        mean_matched = safe_divide(weighted_sum, matched, zero_division=zero_division)
        mean_all = safe_divide(weighted_sum, total, zero_division=zero_division)
        second_moment = safe_divide(squared_sum, matched, zero_division=zero_division)
        variance = np.maximum(second_moment - np.square(mean_matched), 0.0)
        std_matched = np.where(matched == 0, float(zero_division), np.sqrt(variance))

        negative = _column_mass(counts, values < 0)
        neutral = _column_mass(counts, values == 0)
        positive = _column_mass(counts, values > 0)
        min_matched = _observed_extreme(counts, values, first=True, empty=zero_division)
        max_matched = _observed_extreme(
            counts, values, first=False, empty=zero_division
        )
        q25 = _expanded_quantile(counts, values, 0.25, empty=zero_division)
        median = _expanded_quantile(counts, values, 0.50, empty=zero_division)
        q75 = _expanded_quantile(counts, values, 0.75, empty=zero_division)
        coverage = safe_divide(matched, total, zero_division=zero_division)

        context: dict[str, Any] = {
            "counts": counts,
            "values": values,
            "negative": negative,
            "neutral": neutral,
            "positive": positive,
            "matched": matched.astype(float),
            "unmatched": unmatched.astype(float),
            "coverage": coverage,
            "total": total.astype(float),
            "weighted_sum": weighted_sum,
            "mean_matched": mean_matched,
            "mean_all": mean_all,
            "std_matched": std_matched,
            "min_matched": min_matched,
            "q25": q25,
            "median": median,
            "q75": q75,
            "max_matched": max_matched,
        }
        result = info.loc[:, [*primary_key, "_position"]].copy()
        for name in (
            "negative",
            "neutral",
            "positive",
            "matched",
            "unmatched",
            "coverage",
            "total",
            "weighted_sum",
            "mean_matched",
            "mean_all",
            "std_matched",
            "min_matched",
            "q25",
            "median",
            "q75",
            "max_matched",
        ):
            result[name] = context[name]
        for raw_name, scorer in custom.items():
            name = str(raw_name)
            raw = _call_custom_scorer(
                scorer, context=context, extra_kwargs=custom_kwargs
            )
            score = np.asarray(raw, dtype=float).reshape(-1)
            if len(score) != len(info):
                raise ValueError(
                    f"Custom valence formula {name!r} must return one value per row."
                )
            result[name] = score
        frames.append(result)

    if not frames:
        return pd.DataFrame(columns=result_columns)
    return pd.concat(frames, ignore_index=True).loc[:, result_columns]


def _column_mass(counts: sparse.csr_matrix, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(counts.shape[0], dtype=float)
    return np.asarray(counts[:, np.flatnonzero(mask)].sum(axis=1)).reshape(-1)


def _observed_extreme(
    counts: sparse.csr_matrix,
    values: np.ndarray,
    *,
    first: bool,
    empty: float,
) -> np.ndarray:
    result = np.full(counts.shape[0], float(empty), dtype=float)
    for row_index in range(counts.shape[0]):
        row = counts.getrow(row_index)
        present = row.indices[np.asarray(row.data) > 0]
        if present.size:
            result[row_index] = values[present[0] if first else present[-1]]
    return result


def _expanded_quantile(
    counts: sparse.csr_matrix,
    values: np.ndarray,
    q: float,
    *,
    empty: float,
) -> np.ndarray:
    """Match numpy's linear quantile on the conceptual expanded hit values."""
    result = np.full(counts.shape[0], float(empty), dtype=float)
    for row_index in range(counts.shape[0]):
        row = counts.getrow(row_index)
        if row.nnz == 0:
            continue
        indices = row.indices
        frequencies = np.rint(row.data).astype(np.int64)
        positive = frequencies > 0
        indices = indices[positive]
        frequencies = frequencies[positive]
        n = int(frequencies.sum())
        if n <= 0:
            continue
        h = (n - 1) * float(q)
        lower = int(np.floor(h))
        upper = int(np.ceil(h))
        lower_value = _value_at_expanded_rank(indices, frequencies, values, lower)
        upper_value = _value_at_expanded_rank(indices, frequencies, values, upper)
        result[row_index] = lower_value + (h - lower) * (upper_value - lower_value)
    return result


def _value_at_expanded_rank(
    indices: np.ndarray,
    frequencies: np.ndarray,
    values: np.ndarray,
    rank: int,
) -> float:
    cumulative = np.cumsum(frequencies)
    offset = int(np.searchsorted(cumulative, int(rank) + 1, side="left"))
    return float(values[int(indices[offset])])


def _call_custom_scorer(
    scorer: ValenceScorer,
    *,
    context: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
) -> Any:
    try:
        signature = inspect.signature(scorer)
    except (TypeError, ValueError):
        return scorer(context["counts"], context["values"], **dict(extra_kwargs))
    available: dict[str, Any] = {**context, **dict(extra_kwargs)}
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ):
        return scorer(**available)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in available:
            value = available[parameter.name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise ValueError(
                "Custom valence formulas may request only computed valence quantities "
                f"{sorted(context)} plus supplied custom kwargs; missing {parameter.name!r}."
            )
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value
    return scorer(*args, **kwargs)
