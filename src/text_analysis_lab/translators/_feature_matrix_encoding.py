"""Numeric feature encoding for the opt-in FeatureMatrix pathway.

The state is deliberately small: feature labels, category vocabularies and explicit
mappings. Numerical output rows are never retained.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.translators._feature_matrix_numeric import (
    ScalarStats, parse_transform, _integer_metadata_values,
    _require_float_safe_integer_bins, _lossless_integer_differences,
)


def _scaling_numbers(values, fitted, kind):
    """Preserve original integer values for exact scaling arithmetic.

    A frozen integer source must be replayed using integer metadata, never
    reinterpreted through float64. Legacy float-only recipes retain their
    original behavior.
    """
    integers = _integer_metadata_values(values)
    if fitted.get("integer_source") and integers is not None:
        positions = values.notna().to_numpy(dtype=bool)
        return positions, integers[1]
    if fitted.get("integer_source") and values.notna().any():
        raise ArtifactError(
            "Frozen integer metadata scaling requires integer values on replay; "
            "float conversion may lose precision."
        )
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(
            dtype=float, na_value=np.nan,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            f"FeatureMatrix {kind} metadata is not numeric."
        ) from exc
    if np.isinf(numeric).any():
        raise ArtifactError("FeatureMatrix metadata contains infinity.")
    return np.isfinite(numeric), numeric[np.isfinite(numeric)]


def _scaled_integer_minmax(values, minimum, maximum):
    """Keep signed/unsigned integer differences exact until normalization."""
    span = int(maximum) - int(minimum)
    if span == 0:
        return np.zeros(len(values), dtype=float)
    from fractions import Fraction

    scaled = np.array([
        float(Fraction(value - int(minimum), span)) for value in values
    ], dtype=float)
    if not np.isfinite(scaled).all():
        raise ArtifactError(
            "Min-max integer metadata cannot be represented as finite float64."
        )
    return scaled


def _bin_is_inactive(fitted: Mapping[str, Any]) -> bool:
    """No observed values => no learned bins for data-fitted strategies.

    Also honor pre-fix frozen state with count=0 but no inactive flag.
    Explicit boundaries remain meaningful even without observed values.
    """
    return bool(fitted.get(
        "inactive",
        fitted.get("count") == 0 and fitted.get("strategy") != "boundaries",
    ))


class FeatureMatrixEncoding:
    """Freeze scalar-column encodings and emit numeric blocks on demand.

    Inputs must be ordinary aligned pandas columns, already resolved by TeAL's
    query engine. This helper intentionally does not own artifact I/O.
    """

    def __init__(self, encodings: Mapping[str, Any] | None = None) -> None:
        self.encodings = dict(encodings or {})
        self._specs: list[dict[str, Any]] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._specs is not None

    @property
    def columns(self) -> tuple[str, ...]:
        if self._specs is None:
            raise ArtifactError("FeatureMatrix encoding has not been fitted.")
        return tuple(
            output for spec in self._specs for output in spec["outputs"]
        )

    def fit(self, frame: pd.DataFrame) -> FeatureMatrixEncoding:
        """Fit from an in-memory frame (the standalone encoding contract)."""
        return self.fit_batches((frame,), columns=frame.columns)

    def fit_batches(
        self, batches: Iterable[pd.DataFrame], *, columns: Sequence[str],
    ) -> FeatureMatrixEncoding:
        """Learn bounded scalar statistics and category vocabularies in batches.

        A column's type is determined from its first nonempty batch. This is
        stable for TeAL/Arrow-backed fields, whose schema is fixed throughout
        the query. Numeric fields retain no row-wise state.
        """
        if self._specs is not None:
            raise ArtifactError("FeatureMatrix encoding is already fitted.")
        names = [str(name) for name in columns]
        unknown = set(self.encodings).difference(names)
        if unknown:
            raise ArtifactError(f"Unknown FeatureMatrix encoding columns: {sorted(unknown)}")
        if len(names) != len(set(names)):
            raise ArtifactError("FeatureMatrix input scalar columns must be unique.")
        specs: list[dict[str, Any]] = []
        vocabularies: dict[str, set[str]] = {}
        numeric_stats: dict[str, ScalarStats] = {}
        for name in names:
            requested = self.encodings.get(name)
            transform = parse_transform(name, requested)
            if transform is not None:
                kind = transform["kind"]
            elif requested is None:
                kind = "auto"
            elif requested in ("onehot", "numeric"):
                kind = requested
            elif isinstance(requested, Mapping):
                kind = "mapping"
            else:
                raise ArtifactError(f"Unsupported encoding for {name!r}: {requested!r}")
            spec: dict[str, Any] = {"name": name, "kind": kind}
            if transform is not None:
                spec["config"] = transform
                if kind in {"zscore", "minmax", "bin"}:
                    numeric_stats[name] = ScalarStats(transform)
            if kind == "mapping":
                if any(not isinstance(v, (int, float, np.integer, np.floating))
                       or isinstance(v, bool) for v in requested.values()):
                    raise ArtifactError(f"Feature mapping for {name!r} must be numeric.")
                spec["pairs"] = [
                    [key.item() if isinstance(key, np.generic) else key,
                     value.item() if isinstance(value, np.generic) else value]
                    for key, value in requested.items()
                ]
            if kind == "onehot":
                vocabularies[name] = set()
            specs.append(spec)

        for frame in batches:
            if list(frame.columns) != names:
                raise ArtifactError(
                    "FeatureMatrix batches must have the same ordered scalar columns."
                )
            if frame.empty:
                continue
            for spec in specs:
                name = spec["name"]
                kind = spec["kind"]
                values = frame[name]
                if kind == "auto":
                    kind = (
                        "onehot"
                        if pd.api.types.is_object_dtype(values.dtype)
                        or isinstance(values.dtype, pd.CategoricalDtype)
                        or pd.api.types.is_string_dtype(values.dtype)
                        else "numeric"
                    )
                    spec["kind"] = kind
                    if kind == "onehot":
                        vocabularies[name] = set()
                if kind == "onehot":
                    vocabularies[name].update(
                        str(value) for value in values.dropna().tolist()
                    )
                    if len(vocabularies[name]) > 50_000:
                        raise ArtifactError(
                            f"FeatureMatrix onehot column {name!r} exceeds 50,000 categories."
                        )
                if name in numeric_stats:
                    numeric_stats[name].update(values)

        for spec in specs:
            name = spec["name"]
            if spec["kind"] == "auto":
                spec["kind"] = "numeric"
            if spec["kind"] == "onehot":
                categories = sorted(vocabularies[name])
                spec["categories"] = categories
                spec["outputs"] = [f"{name}_{cat}" for cat in categories]
            elif name in numeric_stats:
                spec["fitted"] = numeric_stats[name].finish()
                if spec["kind"] == "bin":
                    n_bins = len(spec["fitted"]["edges"]) - 1
                    spec["outputs"] = [f"{name}_bin_{index}" for index in range(n_bins)]
                else:
                    spec["outputs"] = [name]
            else:
                spec["outputs"] = [name]
        outputs = [output for spec in specs for output in spec["outputs"]]
        if len(outputs) != len(set(outputs)):
            raise ArtifactError("FeatureMatrix output column names must be unique.")
        self._specs = specs
        return self

    def _selected_specs(
        self, indices: Sequence[int] | None,
    ) -> list[tuple[dict[str, Any], int]]:
        if self._specs is None:
            raise ArtifactError("FeatureMatrix encoding has not been fitted.")
        features = [
            (spec, local_index)
            for spec in self._specs
            for local_index in range(len(spec["outputs"]))
        ]
        selected = range(len(features)) if indices is None else indices
        result = []
        for raw in selected:
            index = int(raw)
            if index < 0 or index >= len(features):
                raise ArtifactError(
                    f"FeatureMatrix encoded feature index {index} is out of range."
                )
            result.append(features[index])
        return result

    def required_columns(self, indices: Sequence[int] | None = None) -> list[str]:
        """Scalar inputs needed for precisely the requested encoded features."""
        selected = self._selected_specs(indices)
        return list(dict.fromkeys(spec["name"] for spec, _ in selected))

    def transform(
        self, frame: pd.DataFrame, *, sparse_output: bool = False,
        indices: Sequence[int] | None = None,
    ) -> Any:
        """Materialize only selected encoded features, in the requested order."""
        selected = self._selected_specs(indices)
        if sparse_output and any(spec["kind"] == "bin" for spec, _ in selected):
            # Build each bin field's CSR indicators once, rather than creating
            # O(n_rows * n_bins) intermediate dense columns. Column slicing
            # preserves arbitrary projections, repetition and output order.
            offsets: dict[int, int] = {}
            offset = 0
            for spec in self._specs:
                offsets[id(spec)] = offset
                offset += len(spec["outputs"])
            cached_bins: dict[int, Any] = {}
            groups: list[Any] = []
            position = 0
            while position < len(selected):
                spec, _ = selected[position]
                end = position + 1
                if spec["kind"] == "bin":
                    while end < len(selected) and selected[end][0] is spec:
                        end += 1
                    if id(spec) not in cached_bins:
                        name = spec["name"]
                        if name not in frame.columns:
                            raise ArtifactError(f"FeatureMatrix input is missing {name!r}.")
                        _require_float_safe_integer_bins(frame[name])
                        try:
                            numbers = pd.to_numeric(
                                frame[name], errors="raise"
                            ).to_numpy(dtype=float, na_value=np.nan)
                        except (TypeError, ValueError) as exc:
                            raise ArtifactError(
                                f"FeatureMatrix numeric column {name!r} is not numeric."
                            ) from exc
                        if np.isinf(numbers).any():
                            raise ArtifactError(
                                f"FeatureMatrix numeric column {name!r} contains infinity."
                            )
                        edges = np.asarray([
                            (-np.inf if i == 0 else np.inf) if edge is None else edge
                            for i, edge in enumerate(spec["fitted"]["edges"])
                        ], dtype=float)
                        if _bin_is_inactive(spec["fitted"]):
                            cached_bins[id(spec)] = sparse.csr_matrix(
                                (len(frame), len(edges) - 1), dtype=float,
                            )
                        else:
                            valid_rows = np.flatnonzero(np.isfinite(numbers))
                            assigned = np.searchsorted(
                                edges, numbers[valid_rows], side="right"
                            ) - 1
                            in_range = (assigned >= 0) & (assigned < len(edges) - 1)
                            rows = valid_rows[in_range]
                            columns = assigned[in_range]
                            cached_bins[id(spec)] = sparse.csr_matrix(
                                (np.ones(len(rows), dtype=float), (rows, columns)),
                                shape=(len(frame), len(edges) - 1),
                            )
                    requested = [local for _, local in selected[position:end]]
                    groups.append(cached_bins[id(spec)][:, requested])
                else:
                    while end < len(selected) and selected[end][0]["kind"] != "bin":
                        end += 1
                    nonbin_indices = [
                        offsets[id(other)] + local
                        for other, local in selected[position:end]
                    ]
                    groups.append(self.transform(
                        frame, sparse_output=True, indices=nonbin_indices,
                    ))
                position = end
            return sparse.hstack(groups, format="csr") if groups else sparse.csr_matrix(
                (len(frame), 0)
            )
        blocks: list[Any] = []
        string_values: dict[str, pd.Series] = {}
        for spec, local_index in selected:
            name = spec["name"]
            if name not in frame.columns:
                raise ArtifactError(f"FeatureMatrix input is missing {name!r}.")
            values = frame[name]
            if spec["kind"] == "onehot":
                if name not in string_values:
                    string_values[name] = values.astype("string")
                category = spec["categories"][local_index]
                block = (
                    string_values[name].eq(category).fillna(False)
                    .to_numpy(dtype=float).reshape(-1, 1)
                )
            elif spec["kind"] == "mapping":
                block = (
                    values.map(dict(spec["pairs"]))
                    .to_numpy(dtype=float).reshape(-1, 1)
                )
            elif spec["kind"] in {"bin", "zscore", "minmax"}:
                fitted = spec["fitted"]
                if spec["kind"] == "bin":
                    _require_float_safe_integer_bins(values)
                    try:
                        numeric = pd.to_numeric(
                            values, errors="raise"
                        ).to_numpy(dtype=float, na_value=np.nan)
                    except (TypeError, ValueError) as exc:
                        raise ArtifactError(
                            f"FeatureMatrix numeric column {name!r} is not numeric."
                        ) from exc
                    if np.isinf(numeric).any():
                        raise ArtifactError(
                            f"FeatureMatrix numeric column {name!r} contains infinity."
                        )
                    valid = np.isfinite(numeric)
                    observed = numeric[valid]
                else:
                    valid, observed = _scaling_numbers(
                        values, fitted, spec["kind"],
                    )
                if spec["kind"] == "bin":
                    block = np.zeros((len(values), 1), dtype=float)
                    if not _bin_is_inactive(fitted):
                        edges = np.asarray([
                            (-np.inf if i == 0 else np.inf) if edge is None else edge
                            for i, edge in enumerate(fitted["edges"])
                        ], dtype=float)
                        # Lower edges inclusive, upper edges exclusive. Missing
                        # and out-of-range values encode as all-zero features.
                        target = np.searchsorted(
                            edges, observed, side="right",
                        ) - 1
                        rows = np.flatnonzero(valid)
                        mask = (
                            (target == local_index) & (target >= 0)
                            & (target < len(edges) - 1)
                        )
                        block[rows[mask], 0] = 1.0
                elif spec["kind"] == "zscore":
                    sd = fitted["sd"]
                    block = np.zeros((len(values), 1), dtype=float)
                    if sd > 0:
                        # New fitted states retain the offset and centered mean
                        # separately. Old recipes with only "mean" still replay.
                        offset = fitted.get("offset", fitted["mean"])
                        mean_shift = fitted.get("mean_shift", 0.0)
                        with np.errstate(over="ignore", invalid="ignore"):
                            if fitted.get("integer_source"):
                                centered = (
                                    _lossless_integer_differences(observed, int(offset))
                                    - mean_shift
                                )
                            else:
                                centered = (observed - offset) - mean_shift
                            scaled = centered / sd
                        if not np.isfinite(scaled).all():
                            raise ArtifactError(
                                f"Z-score values for {name!r} cannot be represented "
                                "as finite float64; rescale the input."
                            )
                        block[valid, 0] = scaled
                else:
                    minimum = fitted["minimum"]
                    maximum = fitted["maximum"]
                    block = np.zeros((len(values), 1), dtype=float)
                    if maximum > minimum:
                        if fitted.get("integer_source"):
                            scaled = _scaled_integer_minmax(observed, minimum, maximum)
                        else:
                            with np.errstate(over="ignore", invalid="ignore"):
                                span = maximum - minimum
                                differences = observed - minimum
                                if np.isfinite(span) and np.isfinite(differences).all():
                                    scaled = differences / span
                                else:
                                    # Halving before subtraction protects the
                                    # opposite-sign float64 extremes.
                                    half_span = maximum / 2.0 - minimum / 2.0
                                    scaled = (
                                        (observed / 2.0 - minimum / 2.0)
                                        / half_span
                                    )
                        if not np.isfinite(scaled).all():
                            raise ArtifactError(
                                f"Min-max values for {name!r} cannot be represented "
                                "as finite float64; rescale the input."
                            )
                        block[valid, 0] = scaled
            else:
                try:
                    block = (
                        pd.to_numeric(values, errors="raise")
                        .to_numpy(dtype=float).reshape(-1, 1)
                    )
                except (TypeError, ValueError) as exc:
                    raise ArtifactError(
                        f"FeatureMatrix numeric column {name!r} contains nonnumeric values."
                    ) from exc
            blocks.append(sparse.csr_matrix(block) if sparse_output else block)
        if sparse_output:
            return (
                sparse.hstack(blocks, format="csr") if blocks
                else sparse.csr_matrix((len(frame), 0))
            )
        return np.hstack(blocks) if blocks else np.empty((len(frame), 0))

    def to_state(self) -> dict[str, Any]:
        if self._specs is None:
            raise ArtifactError("FeatureMatrix encoding has not been fitted.")
        return {"specs": self._specs}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FeatureMatrixEncoding:
        obj = cls()
        obj._specs = [dict(spec) for spec in state["specs"]]
        return obj


def append_feature_columns(
    matrix: Any | None,
    extra: Any,
    *,
    sparse_output: bool,
) -> Any:
    """Combine already aligned native matrix values and encoded scalar columns.

    Only the requested batch is combined; sparse matrix inputs never densify
    unless the caller explicitly requested dense output.
    """
    if matrix is None:
        return extra.tocsr() if sparse_output and sparse.issparse(extra) else (
            sparse.csr_matrix(extra) if sparse_output else np.asarray(extra)
        )
    if matrix.ndim != 2 or extra.ndim != 2 or matrix.shape[0] != extra.shape[0]:
        raise ArtifactError("FeatureMatrix blocks must have matching row counts.")
    if sparse_output:
        return sparse.hstack([sparse.csr_matrix(matrix), sparse.csr_matrix(extra)], format="csr")
    left = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    right = extra.toarray() if sparse.issparse(extra) else np.asarray(extra)
    return np.hstack([left, right])
