"""Bounded, deterministic scalar statistics for FeatureMatrix recipes.

Only selected scalar metadata columns reach this module; the source geometry is
not scanned or materialized for numerical fitting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

MAX_BINS = 4096
QUANTILE_SAMPLE_SIZE = 8192


def _fixed_width_edges(minimum: float, maximum: float, *, width: float, anchor: float) -> list[float]:
    """Exact decimal bin alignment followed by representable float edges.

    The shortest decimal spellings of the supplied float inputs define the
    requested widths, anchor and values. Converting them to exact fractions
    avoids both floor(0.3 / 0.1) == 2 and shifted-anchor cancellation.
    The float edges are what the frozen renderer compares with float data.
    """
    step = Fraction(str(width))
    origin = Fraction(str(anchor))
    start = (Fraction(str(minimum)) - origin) // step
    stop = (Fraction(str(maximum)) - origin) // step
    n_bins = stop - start + 1
    if n_bins > MAX_BINS:
        raise ArtifactError(f"Fixed-width field exceeds the {MAX_BINS}-bin safety limit.")

    try:
        edges = [float(origin + index * step) for index in range(start, stop + 2)]
    except OverflowError as exc:
        raise ArtifactError("Fixed-width bin edges exceed finite float range.") from exc

    # At extreme magnitudes, two distinct decimal edges can round to the
    # same binary float. Reject rather than freeze a non-increasing schema.
    if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ArtifactError(
            "Fixed-width bin boundaries cannot be represented distinctly as "
            "float64; increase width or change anchor."
        )
    return edges


def parse_transform(name, request):
    """Return a normalized new-style scalar transform or None for legacy."""
    if not isinstance(request, Mapping) or "kind" not in request:
        if request in ("raw", "numeric"):
            return {"kind": "raw"}
        if request == "zscore":
            return {"kind": "zscore"}
        if request == "minmax":
            return {"kind": "minmax"}
        return None
    kind = request.get("kind")
    # A legacy explicit category mapping may itself have the category "kind".
    # Preserve that old mapping syntax when all mapped values are numeric.
    # Match the accepted legacy mapping types, including numpy scalars.
    numeric_mapping_types = (int, float, np.integer, np.floating)
    if isinstance(kind, numeric_mapping_types) and not isinstance(kind, bool) and all(
        isinstance(v, numeric_mapping_types) and not isinstance(v, bool)
        for v in request.values()
    ):
        return None
    if kind in {"raw", "numeric", "zscore", "minmax"}:
        if set(request) != {"kind"}:
            raise ArtifactError(f"Unrecognized options for {name!r} {kind!r}.")
        return {"kind": "raw" if kind == "numeric" else kind}
    if kind != "bin":
        raise ArtifactError(f"Unsupported FeatureMatrix transform for {name!r}: {kind!r}")
    allowed = {"kind", "strategy", "width", "anchor", "n_bins", "boundaries", "output"}
    if set(request) - allowed:
        raise ArtifactError(f"Unknown binning options for {name!r}: {sorted(set(request) - allowed)}")
    if request.get("output", "onehot") != "onehot":
        raise ArtifactError("Binned FeatureMatrix fields require onehot output.")
    strategy = request.get("strategy")
    if strategy not in {"fixed_width", "quantile", "equal_width", "boundaries"}:
        raise ArtifactError(f"Invalid binning strategy for {name!r}: {strategy!r}")
    result = {"kind": "bin", "strategy": strategy}
    if strategy == "fixed_width":
        width = request.get("width")
        anchor = request.get("anchor", 0)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not np.isfinite(v) for v in (width, anchor)) or width <= 0:
            raise ArtifactError("Fixed-width binning requires finite width > 0 and anchor.")
        result.update(width=float(width), anchor=float(anchor))
    elif strategy in {"quantile", "equal_width"}:
        n = request.get("n_bins", 10)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_BINS:
            raise ArtifactError(f"n_bins must be an integer from 1 to {MAX_BINS}.")
        result["n_bins"] = n
    else:
        boundaries = request.get("boundaries")
        if not isinstance(boundaries, (list, tuple)) or not boundaries or len(boundaries) >= MAX_BINS:
            raise ArtifactError("Explicit boundaries must be a nonempty ordered list.")
        try:
            edges = np.asarray(boundaries, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("Bin boundaries must be numeric.") from exc
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ArtifactError("Bin boundaries must be finite and strictly increasing.")
        result["boundaries"] = edges.tolist()
    return result


class ScalarStats:
    """Fit only a constant amount of data per selected numeric scalar."""

    def __init__(self, config):
        self.config = config
        self.count = 0
        # Welford moments are accumulated relative to the first finite value.
        # Keeping this offset separate is essential for e.g. [1e16, 1e16+2].
        self.offset = None
        self.mean = 0.0  # centered mean, not the rounded absolute mean
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sample = []
        self.seen = 0
        self.rng = np.random.default_rng(0)

    def update(self, values):
        try:
            nums = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("Numeric FeatureMatrix transform received nonnumeric data.") from exc
        if np.isinf(nums).any():
            raise ArtifactError("Numeric FeatureMatrix transforms require finite values or missing.")
        valid = nums[np.isfinite(nums)]
        if not len(valid):
            return
        count = len(valid)
        if self.config["kind"] == "zscore":
            if self.offset is None:
                self.offset = float(valid[0])
            with np.errstate(over="ignore", invalid="ignore"):
                shifted = valid - self.offset
                batch_mean = float(np.mean(shifted))
                residuals = shifted - batch_mean
                batch_m2 = float(np.dot(residuals, residuals))
            if not np.isfinite(shifted).all() or not np.isfinite(
                [batch_mean, batch_m2]
            ).all():
                raise ArtifactError(
                    "Z-score moments exceed stable float64 precision; "
                    "rescale the input or use another encoding."
                )
            if self.count == 0:
                self.mean = batch_mean
                self.m2 = batch_m2
            else:
                total = self.count + count
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = batch_mean - self.mean
                    new_mean = self.mean + difference * (count / total)
                    new_m2 = (
                        self.m2 + batch_m2
                        + (difference * difference) * (self.count / total) * count
                    )
                if not np.isfinite([new_mean, new_m2]).all():
                    raise ArtifactError(
                        "Z-score moments exceed stable float64 precision; "
                        "rescale the input or use another encoding."
                    )
                self.mean = new_mean
                self.m2 = new_m2
        self.count += count
        self.minimum = min(self.minimum, float(valid.min()))
        self.maximum = max(self.maximum, float(valid.max()))
        if self.config["kind"] == "bin" and self.config["strategy"] == "quantile":
            for value in valid:
                self.seen += 1
                if len(self.sample) < QUANTILE_SAMPLE_SIZE:
                    self.sample.append(float(value))
                else:
                    index = int(self.rng.integers(0, self.seen))
                    if index < QUANTILE_SAMPLE_SIZE:
                        self.sample[index] = float(value)

    def finish(self):
        c = self.config
        kind = c["kind"]
        if kind != "bin":
            fitted = dict(
                c, count=self.count, mean=0.0, sd=0.0,
                minimum=self.minimum if self.count else 0.0,
                maximum=self.maximum if self.count else 0.0,
            )
            if kind == "zscore":
                # The rounded absolute mean is retained for compatibility;
                # new renderers use the stable offset + centered mean.
                offset = self.offset if self.offset is not None else 0.0
                fitted.update(
                    mean=offset + self.mean,
                    offset=offset,
                    mean_shift=self.mean,
                    sd=math.sqrt(self.m2 / self.count) if self.count else 0.0,
                )
            return fitted
        strategy = c["strategy"]
        if strategy == "boundaries":
            edges = [-math.inf, *c["boundaries"], math.inf]
        elif not self.count:
            # A missing-only field yields one permanently inactive bin.
            edges = [-math.inf, math.inf]
        elif strategy == "fixed_width":
            edges = _fixed_width_edges(
                self.minimum, self.maximum,
                width=c["width"], anchor=c["anchor"],
            )
        elif self.minimum == self.maximum:
            edges = [-math.inf, math.inf]
        elif strategy == "equal_width":
            requested_edges = np.linspace(
                self.minimum, self.maximum, c["n_bins"] + 1,
            )
            if (not np.isfinite(requested_edges).all()
                    or not np.all(np.diff(requested_edges) > 0)):
                raise ArtifactError(
                    "Equal-width bin boundaries cannot be represented distinctly "
                    "as float64; reduce n_bins or rescale the input."
                )
            edges = [-math.inf, *requested_edges[1:-1].tolist(), math.inf]
        else:
            # Linear interpolation of the bounded reservoir's sorted values.
            # Exact decimal fractions avoid overflow in (upper - lower)
            # for finite values of opposite signs near float64 limits.
            ordered = np.sort(np.asarray(self.sample, dtype=float))
            cutpoints = []
            for rank in range(1, c["n_bins"]):
                lower_index, remainder = divmod(
                    (len(ordered) - 1) * rank, c["n_bins"],
                )
                if remainder:
                    left = Fraction(str(float(ordered[lower_index])))
                    right = Fraction(str(float(ordered[lower_index + 1])))
                    cut = float(
                        left + (right - left) * Fraction(remainder, c["n_bins"])
                    )
                else:
                    cut = float(ordered[lower_index])
                if not math.isfinite(cut):
                    raise ArtifactError(
                        "Quantile interpolation produced a nonfinite boundary; "
                        "rescale the input."
                    )
                cutpoints.append(cut)
            # Legitimately tied float64 cutpoints collapse to fewer bins.
            inner = np.unique(np.asarray(cutpoints, dtype=float))
            if not np.isfinite(inner).all() or not np.all(inner[1:] > inner[:-1]):
                raise ArtifactError("Quantile boundaries must be finite and ordered.")
            edges = [-math.inf, *inner.tolist(), math.inf]
        # Fixed-width edges are finite; out-of-range observations encode as all zero.
        return dict(
            c, edges=[None if not math.isfinite(e) else e for e in edges],
            missing="all_zero", count=self.count,
            # Data-fitted bins with no observed values have no learned cutpoints.
            # Keep one feature for schema stability, but never activate it on replay.
            # Explicit boundaries do not require data, so remain active.
            inactive=(self.count == 0 and strategy != "boundaries"),
        )
