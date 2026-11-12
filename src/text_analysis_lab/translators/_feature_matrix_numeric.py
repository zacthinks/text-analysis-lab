"""Bounded, deterministic scalar statistics for FeatureMatrix recipes.

Only selected scalar metadata columns reach this module; the source geometry is
not scanned or materialized for numerical fitting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

MAX_BINS = 4096
QUANTILE_SAMPLE_SIZE = 8192


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
        self.mean = 0.0
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
        mean = float(valid.mean())
        delta = mean - self.mean
        new_count = self.count + count
        self.m2 += float(np.square(valid - mean).sum()) + delta * delta * self.count * count / new_count
        self.mean += delta * count / new_count
        self.count = new_count
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
            return dict(c, count=self.count, mean=self.mean if self.count else 0.0,
                        sd=math.sqrt(max(self.m2, 0.0) / self.count) if self.count else 0.0,
                        minimum=self.minimum if self.count else 0.0,
                        maximum=self.maximum if self.count else 0.0)
        strategy = c["strategy"]
        if strategy == "boundaries":
            edges = [-math.inf, *c["boundaries"], math.inf]
        elif not self.count:
            # A missing-only field yields one permanently inactive bin.
            edges = [-math.inf, math.inf]
        elif strategy == "fixed_width":
            width, anchor = c["width"], c["anchor"]
            start = math.floor((self.minimum - anchor) / width)
            stop = math.floor((self.maximum - anchor) / width)
            if stop - start + 1 > MAX_BINS:
                raise ArtifactError(f"Fixed-width field exceeds the {MAX_BINS}-bin safety limit.")
            edges = [anchor + i * width for i in range(start, stop + 2)]
        elif self.minimum == self.maximum:
            edges = [-math.inf, math.inf]
        elif strategy == "equal_width":
            edges = [-math.inf, *np.linspace(self.minimum, self.maximum,
                     c["n_bins"] + 1)[1:-1].tolist(), math.inf]
        else:
            # Deterministic bounded-reservoir quantile estimate. Duplicate
            # cutpoints are collapsed; actual output width can be < n_bins.
            inner = np.quantile(self.sample, np.linspace(0, 1, c["n_bins"] + 1)[1:-1])
            edges = [-math.inf, *np.unique(inner).tolist(), math.inf]
        # Fixed-width edges are finite; out-of-range observations encode as all zero.
        return dict(c, edges=[None if not math.isfinite(e) else e for e in edges],
                    missing="all_zero", count=self.count)
