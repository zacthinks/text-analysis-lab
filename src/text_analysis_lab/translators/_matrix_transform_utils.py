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


def feature_metadata_from_columns(columns: Sequence[str]) -> pd.DataFrame:
    """Build canonical Feature Metadata from ordered feature labels."""
    labels = [str(value) for value in columns]
    return pd.DataFrame(
        {
            "column_index": np.arange(len(labels), dtype="int64"),
            "column": labels,
        }
    )


def normalize_feature_metadata(
    feature_metadata: pd.DataFrame,
    *,
    width: int,
    name: str,
) -> pd.DataFrame:
    """Validate and normalize Feature Metadata for an in-memory matrix."""
    if not isinstance(feature_metadata, pd.DataFrame):
        raise TypeError(f"{name} feature_metadata must be a pandas DataFrame.")
    if len(feature_metadata) != int(width):
        raise ValueError(
            f"{name} Feature Metadata width {len(feature_metadata)} does not match "
            f"matrix width {int(width)}."
        )
    frame = feature_metadata.copy().reset_index(drop=True)
    if "column_index" in frame.columns:
        frame["column_index"] = np.arange(len(frame), dtype="int64")
    else:
        frame.insert(0, "column_index", np.arange(len(frame), dtype="int64"))
    return frame


def unpack_standalone_matrix(
    value: Any,
    *,
    name: str,
) -> tuple[Any, pd.DataFrame | None, bool]:
    """Unpack a raw matrix or canonical standalone matrix mapping."""
    structured = isinstance(value, Mapping) and "values" in value
    if structured:
        matrix = value["values"]
        feature_metadata = value.get("feature_metadata")
        if feature_metadata is None and "columns" in value:
            feature_metadata = feature_metadata_from_columns(value["columns"])
        if feature_metadata is None:
            raise ValueError(
                f"{name} standalone matrix mappings require feature_metadata."
            )
    else:
        matrix = value
        feature_metadata = None

    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} requires a two-dimensional matrix.")
    normalized = (
        None
        if feature_metadata is None
        else normalize_feature_metadata(
            feature_metadata,
            width=int(shape[1]),
            name=name,
        )
    )
    return matrix, normalized, structured


def standalone_matrix_payload(
    values: Any,
    feature_metadata: pd.DataFrame,
    *,
    name: str,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical standalone matrix representation.

    Row-preserving transforms carry ordinary row metadata and its optional
    privileged row-name designation forward unchanged.
    """
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} requires a two-dimensional matrix.")
    payload: dict[str, Any] = {
        "values": values,
        "feature_metadata": normalize_feature_metadata(
            feature_metadata,
            width=int(shape[1]),
            name=name,
        ),
    }
    if source is not None:
        metadata = source.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError(f"{name} standalone metadata must be a pandas DataFrame.")
            if len(metadata) != int(shape[0]):
                raise ValueError(
                    f"{name} row metadata count {len(metadata)} does not match "
                    f"matrix row count {int(shape[0])}."
                )
            payload["metadata"] = metadata.reset_index(drop=True).copy()
        row_name_column = source.get("row_name_column")
        if row_name_column is not None:
            if not isinstance(row_name_column, str) or not row_name_column:
                raise ValueError(f"{name} row_name_column must be a non-empty string.")
            if "metadata" not in payload or row_name_column not in payload["metadata"].columns:
                raise ValueError(
                    f"{name} row_name_column {row_name_column!r} is not present in metadata."
                )
            payload["row_name_column"] = row_name_column
    return payload


def feature_labels(feature_metadata: pd.DataFrame, *, name: str) -> list[str]:
    if "column" not in feature_metadata.columns:
        raise ValueError(
            f"{name} Feature Metadata must contain a 'column' field when feature "
            "labels are required."
        )
    return [str(value) for value in feature_metadata["column"].tolist()]


def single_source(sources: Mapping[str, Any], *, name: str):
    if set(sources) != {"source"}:
        raise OperatorError(f"{name} requires exactly one source under 'source'.")
    return sources["source"]


def single_input(inputs: Mapping[str, InputBatch], *, name: str) -> InputBatch:
    if set(inputs) != {"source"}:
        raise OperatorError(f"{name} expected exactly one input under 'source'.")
    return inputs["source"]


def native_matrix_packet(
    packet: InputBatch, *, name: str
) -> tuple[pd.DataFrame, Any, list[str]]:
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
