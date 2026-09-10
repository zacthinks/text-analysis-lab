"""Built-in polarity scoring functions."""

from __future__ import annotations

from typing import Any

import numpy as np


def polarity_log_ratio(
    positive: np.ndarray,
    negative: np.ndarray,
    total: np.ndarray | None = None,
    *,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Return log((positive + smoothing) / (negative + smoothing))."""
    _ = total
    smoothing = float(smoothing)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((positive + smoothing) / (negative + smoothing))


def polarity_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    total: np.ndarray | None = None,
) -> np.ndarray:
    """Return the unnormalized positive-minus-negative difference."""
    _ = total
    return np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float)


def polarity_proportional_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    total: np.ndarray | None = None,
    *,
    zero_division: float = np.nan,
) -> np.ndarray:
    """Return (positive - negative) / (positive + negative)."""
    _ = total
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    return _safe_divide(
        positive - negative,
        positive + negative,
        zero_division=zero_division,
    )


def polarity_matched_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    matched: np.ndarray,
    *,
    zero_division: float = np.nan,
) -> np.ndarray:
    """Return (positive - negative) / matched, including neutral matches."""
    return _safe_divide(
        np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float),
        np.asarray(matched, dtype=float),
        zero_division=zero_division,
    )


def polarity_total_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    total: np.ndarray | None,
    *,
    zero_division: float = np.nan,
) -> np.ndarray:
    """Return (positive - negative) / total source count for each row."""
    if total is None:
        raise ValueError("polarity_total_difference requires total row values.")
    return _safe_divide(
        np.asarray(positive, dtype=float) - np.asarray(negative, dtype=float),
        np.asarray(total, dtype=float),
        zero_division=zero_division,
    )


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    zero_division: float,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(numerator.shape, float(zero_division), dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


POLARITY_METHODS: dict[str, Any] = {
    "log_ratio": polarity_log_ratio,
    "logratio": polarity_log_ratio,
    "logit": polarity_log_ratio,
    "difference": polarity_difference,
    "proportional_difference": polarity_proportional_difference,
    "proportion": polarity_proportional_difference,
    "relpropdiff": polarity_proportional_difference,
    "matched_difference": polarity_matched_difference,
    "matched": polarity_matched_difference,
    "total_difference": polarity_total_difference,
    "total": polarity_total_difference,
    "abspropdiff": polarity_total_difference,
}
