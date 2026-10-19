"""Numeric feature encoding for the opt-in FeatureMatrix pathway.

The state is deliberately small: feature labels, category vocabularies and explicit
mappings. Numerical output rows are never retained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        if self._specs is not None:
            raise ArtifactError("FeatureMatrix encoding is already fitted.")
        unknown = set(self.encodings).difference(frame.columns)
        if unknown:
            raise ArtifactError(f"Unknown FeatureMatrix encoding columns: {sorted(unknown)}")
        specs: list[dict[str, Any]] = []
        for column in frame.columns:
            values = frame[column]
            requested = self.encodings.get(column)
            if requested is None:
                requested = (
                    "onehot"
                    if pd.api.types.is_object_dtype(values.dtype)
                    or isinstance(values.dtype, pd.CategoricalDtype)
                    or pd.api.types.is_string_dtype(values.dtype)
                    else "numeric"
                )
            if requested == "onehot":
                # Category ordering is frozen; missing categories do not become
                # features, and unseen categories at replay produce all zeros.
                categories = sorted(
                    {str(value) for value in values.dropna().tolist()}
                )
                specs.append({
                    "name": str(column), "kind": "onehot",
                    "categories": categories,
                    "outputs": [f"{column}_{category}" for category in categories],
                })
            elif requested == "numeric":
                specs.append({
                    "name": str(column), "kind": "numeric",
                    "outputs": [str(column)],
                })
            elif isinstance(requested, Mapping):
                if any(not isinstance(v, (int, float, np.integer, np.floating))
                       for v in requested.values()):
                    raise ArtifactError(f"Feature mapping for {column!r} must be numeric.")
                specs.append({
                    "name": str(column), "kind": "mapping",
                    "pairs": [[key.item() if isinstance(key, np.generic) else key,
                               value.item() if isinstance(value, np.generic) else value]
                              for key, value in requested.items()],
                    "outputs": [str(column)],
                })
            else:
                raise ArtifactError(f"Unsupported encoding for {column!r}: {requested!r}")
        outputs = [name for spec in specs for name in spec["outputs"]]
        if len(outputs) != len(set(outputs)):
            raise ArtifactError("FeatureMatrix output column names must be unique.")
        self._specs = specs
        return self

    def transform(self, frame: pd.DataFrame, *, sparse_output: bool = False) -> Any:
        if self._specs is None:
            raise ArtifactError("FeatureMatrix encoding has not been fitted.")
        blocks: list[Any] = []
        for spec in self._specs:
            name = spec["name"]
            if name not in frame.columns:
                raise ArtifactError(f"FeatureMatrix input is missing {name!r}.")
            values = frame[name]
            if spec["kind"] == "onehot":
                cats = spec["categories"]
                # Equality with a category is false for NA and unseen values.
                if cats:
                    block = np.column_stack([
                        values.astype('string').eq(category).fillna(False).to_numpy(dtype=float)
                        for category in cats
                    ])
                else:
                    block = np.empty((len(frame), 0), dtype=float)
            elif spec["kind"] == "mapping":
                block = values.map(dict(spec["pairs"])).to_numpy(dtype=float).reshape(-1, 1)
            else:
                try:
                    block = pd.to_numeric(values, errors="raise").to_numpy(dtype=float).reshape(-1, 1)
                except (TypeError, ValueError) as exc:
                    raise ArtifactError(
                        f"FeatureMatrix numeric column {name!r} contains nonnumeric values."
                    ) from exc
            blocks.append(sparse.csr_matrix(block) if sparse_output else block)
        if sparse_output:
            return sparse.hstack(blocks, format="csr") if blocks else sparse.csr_matrix((len(frame), 0))
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
