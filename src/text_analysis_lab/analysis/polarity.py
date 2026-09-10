"""Polarity diagnostics over dictionary-translated count matrices."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._dictionary_translation_utils import (
    require_dictionary_count_artifact,
    safe_divide,
    validate_batch_counts,
)
from text_analysis_lab.analysis.polarity_scores import (
    polarity_difference,
    polarity_log_ratio,
    polarity_matched_difference,
    polarity_proportional_difference,
    polarity_total_difference,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

PolarityScorer = Callable[..., Any]
_STANDARD_COLUMNS = (
    "difference",
    "log_ratio",
    "matched_difference",
    "total_difference",
    "proportional_difference",
)
_REQUIRED_COLUMNS = ("positive", "negative", "neutral")


def polarity(
    artifact: "BaseArtifact",
    *,
    smoothing: float = 0.5,
    zero_division: float = 0.0,
    custom: Mapping[str, PolarityScorer] | None = None,
    batch_size: int = 10_000,
    **custom_kwargs: Any,
) -> pd.DataFrame:
    """Summarize a polarity DictionaryTranslator output.

    The translated artifact is the durable evidence: counts of positive,
    negative, and neutral matches plus local matched/unmatched/total metadata.
    This Analytic Method returns those primitives, coverage, several standard
    alternative formulas, and optional named custom formulas side-by-side.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    features = require_dictionary_count_artifact(artifact, method="polarity()")
    if tuple(features) != _REQUIRED_COLUMNS:
        raise ValueError(
            "polarity() requires DictionaryTranslator polarity columns exactly "
            f"{list(_REQUIRED_COLUMNS)}; got {features}."
        )

    custom = {} if custom is None else dict(custom)
    custom_columns = [str(name) for name in custom]
    protected = {
        *artifact.primary_key,
        "_position",
        "positive",
        "negative",
        "neutral",
        "matched",
        "unmatched",
        "coverage",
        "total",
        *_STANDARD_COLUMNS,
    }
    conflicts = [name for name in custom_columns if not name or name in protected]
    if conflicts:
        raise ValueError(f"Invalid/conflicting custom polarity formula names: {conflicts}.")
    for name, scorer in custom.items():
        if not callable(scorer):
            raise TypeError(f"Custom polarity formula {name!r} must be callable.")

    primary_key = [str(value) for value in artifact.primary_key]
    columns = [
        *primary_key,
        "_position",
        "positive",
        "negative",
        "neutral",
        "matched",
        "unmatched",
        "coverage",
        "total",
        *_STANDARD_COLUMNS,
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
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        dense = np.asarray(dense, dtype=float)
        positive, negative, neutral = (dense[:, i] for i in range(3))
        matched, unmatched, total = validate_batch_counts(matrix, info)
        if not np.array_equal(
            np.rint(positive + negative + neutral).astype(np.int64), matched
        ):
            raise ValueError(
                "Polarity artifact invariant failed: positive + negative + neutral != matched."
            )
        coverage = safe_divide(matched, total, zero_division=zero_division)
        context: dict[str, np.ndarray] = {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "matched": matched.astype(float),
            "unmatched": unmatched.astype(float),
            "coverage": coverage,
            "total": total.astype(float),
            "difference": polarity_difference(positive, negative),
            "log_ratio": polarity_log_ratio(
                positive, negative, smoothing=smoothing
            ),
            "matched_difference": polarity_matched_difference(
                positive, negative, matched, zero_division=zero_division
            ),
            "total_difference": polarity_total_difference(
                positive, negative, total, zero_division=zero_division
            ),
            "proportional_difference": polarity_proportional_difference(
                positive, negative, zero_division=zero_division
            ),
        }

        result = info.loc[:, [*primary_key, "_position"]].copy()
        for name in (
            "positive",
            "negative",
            "neutral",
            "matched",
            "unmatched",
            "coverage",
            "total",
            *_STANDARD_COLUMNS,
        ):
            result[name] = context[name]
        for raw_name, scorer in custom.items():
            name = str(raw_name)
            raw = _call_custom_scorer(scorer, context=context, extra_kwargs=custom_kwargs)
            score = np.asarray(raw, dtype=float).reshape(-1)
            if len(score) != len(info):
                raise ValueError(
                    f"Custom polarity formula {name!r} must return one value per row."
                )
            result[name] = score
        frames.append(result)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, columns]


def _call_custom_scorer(
    scorer: PolarityScorer,
    *,
    context: Mapping[str, np.ndarray],
    extra_kwargs: Mapping[str, Any],
) -> Any:
    try:
        signature = inspect.signature(scorer)
    except (TypeError, ValueError):
        return scorer(context["positive"], context["negative"], context["total"], **dict(extra_kwargs))
    available: dict[str, Any] = {**context, **dict(extra_kwargs)}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return scorer(**available)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.name in available:
            value = available[parameter.name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise ValueError(
                "Custom polarity formulas may request only computed polarity quantities "
                f"{sorted(context)} plus supplied custom kwargs; missing {parameter.name!r}."
            )
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value
    return scorer(*args, **kwargs)
