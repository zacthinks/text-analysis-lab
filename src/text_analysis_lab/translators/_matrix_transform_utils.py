"""Shared helpers for fitted matrix translators.

This module is intentionally small. Public translators live in their own files;
these helpers only centralize packet validation, feature-compatibility checks,
and durable estimator assets.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch


def single_source(sources: Mapping[str, Any], *, name: str):
    if set(sources) != {"source"}:
        raise OperatorError(f"{name} requires exactly one source under 'source'.")
    return sources["source"]


def single_input(inputs: Mapping[str, InputBatch], *, name: str) -> InputBatch:
    if set(inputs) != {"source"}:
        raise OperatorError(f"{name} expected exactly one input under 'source'.")
    return inputs["source"]


def native_matrix_packet(packet: InputBatch, *, name: str) -> tuple[pd.DataFrame, Any, list[str]]:
    if not isinstance(packet.data, Mapping):
        raise ArtifactError(f"{name} expected a native matrix packet.")
    info = packet.data.get("info")
    matrix = packet.data.get("matrix")
    if not isinstance(info, pd.DataFrame):
        raise ArtifactError(f"{name} native packet is missing info rows.")
    if not sparse.issparse(matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ArtifactError(f"{name} source matrix must be two-dimensional.")
    elif len(matrix.shape) != 2:
        raise ArtifactError(f"{name} source matrix must be two-dimensional.")
    if len(info) != int(matrix.shape[0]):
        raise ArtifactError(
            f"{name} key row count {len(info)} != matrix row count {matrix.shape[0]}."
        )
    key_columns = [str(value) for value in packet.primary_key]
    missing = [column for column in key_columns if column not in info.columns]
    if missing:
        raise ArtifactError(f"{name} source batch is missing key columns {missing}.")
    return info, matrix, key_columns


def key_frame(info: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    return info.loc[:, list(key_columns)].reset_index(drop=True)


def feature_tuple(features: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in features)


def establish_or_validate_features(
    current: tuple[str, ...] | None,
    observed: Sequence[str],
    *,
    fitted: bool,
    name: str,
) -> tuple[str, ...]:
    observed_tuple = feature_tuple(observed)
    if current is None:
        if fitted:
            raise OperatorError(
                f"{name} is fitted but its source feature schema is missing from operator state."
            )
        return observed_tuple
    if current != observed_tuple:
        raise OperatorError(
            f"{name} requires the same ordered feature schema used during fitting. "
            f"Expected {len(current)} features, got {len(observed_tuple)}."
        )
    return current


def require_nonnegative(matrix: Any, *, name: str, integer: bool = False) -> None:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data, dtype=float)
    else:
        values = np.asarray(matrix, dtype=float).reshape(-1)
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        raise ArtifactError(f"{name} requires finite matrix values.")
    if np.any(values < 0):
        raise ArtifactError(f"{name} requires non-negative matrix values.")
    if integer:
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
            raise ArtifactError(f"{name} requires integer-valued count data.")


def dump_estimator(path: Path, estimator: Any) -> None:
    """Serialize an estimator directly to disk without buffering the full pickle in RAM."""
    with path.open("wb") as handle:
        cloudpickle.dump(estimator, handle)


def load_estimator(path: Path) -> Any:
    """Load an estimator directly from disk without first reading the whole pickle into RAM."""
    if not path.exists():
        raise OperatorError(f"Missing fitted estimator asset: {path}.")
    with path.open("rb") as handle:
        return cloudpickle.load(handle)


def clone_estimator(estimator: Any) -> Any:
    return cloudpickle.loads(cloudpickle.dumps(estimator))


def write_intermediate_json(path: Path, state: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(state), indent=2, sort_keys=True), encoding="utf-8")


def read_intermediate_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise OperatorError(f"Intermediate state {path} must contain a JSON object.")
    return value
