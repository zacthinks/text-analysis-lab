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
# Integers outside this interval are not guaranteed to survive float64
# conversion and cannot safely use float-based frozen bin boundaries.
MAX_EXACT_FLOAT_INTEGER = 1 << 53


def _integer_metadata_values(values):
    """Return exact Python integers and row positions, or None for floats.

    This handles NumPy int64/uint64, pandas nullable integer extension arrays,
    and object-backed Python/NumPy integers without first calling to_numpy(float).
    """
    nonmissing = values.dropna()
    entries = nonmissing.tolist()
    if not entries:
        return None
    if all(
        isinstance(item, (int, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in entries
    ):
        return nonmissing.index, [int(item) for item in entries]
    return None


def _require_float_safe_integer_bins(values, *, usage="binning"):
    """Reject unsafe integer-to-float features instead of collapsing values."""
    exact = _integer_metadata_values(values)
    if exact is not None and any(
        abs(value) > MAX_EXACT_FLOAT_INTEGER for value in exact[1]
    ):
        raise ArtifactError(
            f"Integer metadata exceeds the lossless float64 {usage} range; "
            "convert to a smaller relative unit or use a supported numeric "
            "scaler."
        )


def _lossless_integer_differences(values, offset):
    """Convert *integer differences*, never absolute integer observations."""
    result = []
    for value in values:
        difference = value - offset
        rounded = float(difference)
        if not math.isfinite(rounded) or int(rounded) != difference:
            raise ArtifactError(
                "Integer z-score differences cannot be represented exactly "
                "as float64; rescale the input."
            )
        result.append(rounded)
    return np.asarray(result, dtype=float)


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
        self.integer_mode = None
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
        integers = _integer_metadata_values(values)
        if integers is not None:
            if self.config["kind"] == "bin":
                # Frozen bin boundaries are float64. Never learn cutpoints
                # from silently rounded distinct int64/uint64 observations.
                _require_float_safe_integer_bins(values)
            _, exact_values = integers
            # Never coerce large integer observations to float64 at fit time.
            # The count/extrema use exact integers, z-scores use exact
            # integer differences, and safe quantile bins cast individually.
            valid = exact_values
            is_integer = True
        else:
            try:
                nums = pd.to_numeric(values, errors="raise").to_numpy(
                    dtype=float, na_value=np.nan,
                )
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    "Numeric FeatureMatrix transform received nonnumeric data."
                ) from exc
            if np.isinf(nums).any():
                raise ArtifactError(
                    "Numeric FeatureMatrix transforms require finite values or missing."
                )
            valid = nums[np.isfinite(nums)]
            exact_values = None
            is_integer = False
        if not len(valid):
            return
        if self.integer_mode is None:
            self.integer_mode = is_integer
        elif self.integer_mode != is_integer:
            raise ArtifactError(
                "Numerical FeatureMatrix batches must retain their numeric "
                "integer/float dtype to preserve fitted precision."
            )
        count = len(valid)
        if self.config["kind"] == "zscore":
            if self.offset is None:
                self.offset = exact_values[0] if is_integer else float(valid[0])
            with np.errstate(over="ignore", invalid="ignore"):
                shifted = (
                    _lossless_integer_differences(exact_values, self.offset)
                    if is_integer else valid - self.offset
                )
                batch_mean = float(np.mean(shifted))
                residuals = shifted - batch_mean
                batch_m2 = float(np.dot(residuals, residuals))
            if (not np.isfinite(shifted).all()
                    or not np.isfinite([batch_mean, batch_m2]).all()
                    or (batch_m2 == 0.0 and np.any(residuals != 0.0))):
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
        if is_integer:
            self.minimum = min(self.minimum, min(exact_values))
            self.maximum = max(self.maximum, max(exact_values))
        else:
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
            if self.integer_mode:
                fitted["integer_source"] = True
            if kind == "zscore":
                # The rounded absolute mean is retained for compatibility;
                # new renderers use the stable offset + centered mean.
                offset = self.offset if self.offset is not None else 0.0
                variance = self.m2 / self.count if self.count else 0.0
                if self.m2 > 0 and variance == 0.0:
                    raise ArtifactError(
                        "Z-score variance underflowed float64; rescale the input."
                    )
                fitted.update(
                    mean=offset + self.mean,
                    offset=offset,
                    mean_shift=self.mean,
                    sd=math.sqrt(variance),
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
            # Never subtract the two float64 endpoints: a finite span can
            # overflow, even though all interior cutpoints are representable.
            # n_bins=1 has no internal cutpoint and must always be supported.
            count = c["n_bins"]
            lo, hi = Fraction(str(self.minimum)), Fraction(str(self.maximum))
            interior = [
                float(lo + (hi - lo) * Fraction(index, count))
                for index in range(1, count)
            ]
            boundaries = [float(self.minimum), *interior, float(self.maximum)]
            if (not np.isfinite(boundaries).all()
                    or any(right <= left for left, right in zip(
                        boundaries, boundaries[1:],
                    ))):
                raise ArtifactError(
                    "Equal-width bin boundaries cannot be represented distinctly "
                    "as float64; reduce n_bins or rescale the input."
                )
            edges = [-math.inf, *interior, math.inf]
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
