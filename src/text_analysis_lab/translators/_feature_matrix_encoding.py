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
        """Learn only category vocabularies; never accumulate source rows.

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
        for name in names:
            requested = self.encodings.get(name)
            if requested is None:
                kind = "auto"
            elif requested in ("onehot", "numeric"):
                kind = requested
            elif isinstance(requested, Mapping):
                kind = "mapping"
            else:
                raise ArtifactError(f"Unsupported encoding for {name!r}: {requested!r}")
            spec: dict[str, Any] = {"name": name, "kind": kind}
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

        for spec in specs:
            name = spec["name"]
            if spec["kind"] == "auto":
                spec["kind"] = "numeric"
            if spec["kind"] == "onehot":
                categories = sorted(vocabularies[name])
                spec["categories"] = categories
                spec["outputs"] = [f"{name}_{cat}" for cat in categories]
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
