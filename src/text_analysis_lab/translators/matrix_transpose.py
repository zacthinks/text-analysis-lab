"""Transpose/reindex a TeAL matrix so features become named rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._matrix_transform_utils import (
    feature_labels,
    feature_metadata_from_columns,
    native_matrix_packet,
    single_input,
    single_source,
    unpack_standalone_matrix,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_RESERVED_KEY_NAMES = frozenset({"_position", "_batch", "_row_offset"})
_RESERVED_FEATURE_METADATA_NAMES = frozenset({"column_index", "column"})


class MatrixTranspose(BaseTranslator):
    """Transpose a matrix and eagerly swap the semantic roles of its axes.

    Source feature labels become the output's privileged row-name metadata.
    Source row names become output feature labels when available; otherwise
    deterministic labels are derived from the source primary key.

    Additional source key/metadata columns may be promoted eagerly into the
    output Feature Metadata with the feature_metadata_columns operation
    parameter. Cross-axis lazy inheritance is intentionally not implemented.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        key_name: str = "feature_id",
        row_name: str = "feature",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.key_name = _validate_axis_name(key_name, label="key_name", reserved=True)
        self.row_name = _validate_axis_name(row_name, label="row_name", reserved=False)
        self._source_type: str | None = None
        self._source_features: tuple[str, ...] | None = None
        self._source_feature_metadata: pd.DataFrame | None = None
        self._source_row_label_column: str | None = None
        self._promoted_feature_columns: tuple[str, ...] = ()

    def translate(
        self,
        matrix: Any,
        *,
        features: Sequence[str] | None = None,
        row_labels: Sequence[str] | None = None,
        feature_metadata: pd.DataFrame | None = None,
        row_metadata: pd.DataFrame | None = None,
        row_name_column: str | None = None,
        feature_metadata_columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Transpose a matrix with an eager, explicit axis-metadata swap."""
        payload = matrix if isinstance(matrix, Mapping) else None
        matrix, embedded_feature_metadata, _ = unpack_standalone_matrix(
            matrix,
            name="MatrixTranspose.translate(...)",
        )
        shape = matrix.shape

        if embedded_feature_metadata is not None:
            if (
                feature_metadata is not None
                and not feature_metadata.reset_index(drop=True).equals(
                    embedded_feature_metadata.reset_index(drop=True)
                )
            ):
                raise ValueError(
                    "MatrixTranspose received conflicting feature_metadata values."
                )
            feature_metadata = embedded_feature_metadata
            embedded_features = feature_labels(
                embedded_feature_metadata,
                name="MatrixTranspose",
            )
            if (
                features is not None
                and tuple(str(value) for value in features)
                != tuple(embedded_features)
            ):
                raise ValueError(
                    "MatrixTranspose received conflicting features and Feature Metadata."
                )
            features = embedded_features

        if payload is not None:
            embedded_row_metadata = payload.get("metadata")
            if embedded_row_metadata is not None:
                if not isinstance(embedded_row_metadata, pd.DataFrame):
                    raise TypeError(
                        "MatrixTranspose standalone metadata must be a pandas DataFrame."
                    )
                if (
                    row_metadata is not None
                    and not row_metadata.reset_index(drop=True).equals(
                        embedded_row_metadata.reset_index(drop=True)
                    )
                ):
                    raise ValueError(
                        "MatrixTranspose received conflicting row metadata values."
                    )
                row_metadata = embedded_row_metadata
            embedded_row_name = payload.get("row_name_column")
            if embedded_row_name is not None:
                if not isinstance(embedded_row_name, str) or not embedded_row_name:
                    raise ValueError(
                        "MatrixTranspose row_name_column must be a non-empty string."
                    )
                if (
                    row_name_column is not None
                    and row_name_column != embedded_row_name
                ):
                    raise ValueError(
                        "MatrixTranspose received conflicting row_name_column values."
                    )
                row_name_column = embedded_row_name

        if features is None:
            raise ValueError(
                "MatrixTranspose.translate(...) requires features for raw matrices "
                "or feature_metadata in a standalone matrix mapping."
            )
        feature_names = [str(value) for value in features]
        if len(feature_names) != int(shape[1]):
            raise ArtifactError(
                "MatrixTranspose features must match the matrix feature width: "
                f"{len(feature_names)} != {int(shape[1])}."
            )

        if row_labels is None and row_name_column is not None:
            if row_metadata is None or row_name_column not in row_metadata.columns:
                raise ValueError(
                    "MatrixTranspose row_name_column must identify a column in row_metadata."
                )
            row_labels = row_metadata[row_name_column].tolist()

        if row_labels is None:
            columns = [str(index) for index in range(int(shape[0]))]
        else:
            columns = [str(value) for value in row_labels]
            if len(columns) != int(shape[0]):
                raise ArtifactError(
                    "MatrixTranspose row_labels must match the matrix row count: "
                    f"{len(columns)} != {int(shape[0])}."
                )
            if len(set(columns)) != len(columns):
                raise ArtifactError(
                    "MatrixTranspose requires unique row_labels after string conversion."
                )

        try:
            from scipy import sparse
        except ImportError:  # pragma: no cover
            sparse = None
        if sparse is not None and sparse.issparse(matrix):
            values = matrix.transpose().tocsr()
        else:
            values = np.asarray(matrix).T.copy()

        promoted = feature_metadata_from_columns(columns)
        requested_promotions = tuple(
            str(value) for value in (feature_metadata_columns or ())
        )
        if requested_promotions:
            if row_metadata is None:
                raise ValueError(
                    "MatrixTranspose feature_metadata_columns requires row_metadata."
                )
            row_metadata = row_metadata.reset_index(drop=True)
            if len(row_metadata) != int(shape[0]):
                raise ValueError(
                    "MatrixTranspose row_metadata rows must match the matrix row count."
                )
            for column in requested_promotions:
                if column == row_name_column:
                    continue
                if column in _RESERVED_FEATURE_METADATA_NAMES:
                    raise ValueError(
                        f"MatrixTranspose cannot promote reserved Feature Metadata "
                        f"field {column!r}."
                    )
                if column not in row_metadata.columns:
                    raise ValueError(
                        f"MatrixTranspose row metadata is missing requested field {column!r}."
                    )
                promoted[column] = row_metadata[column].to_numpy(copy=True)

        result: dict[str, Any] = {
            "values": values,
            "columns": columns,
            "feature_metadata": promoted,
        }
        if feature_metadata is not None:
            if not isinstance(feature_metadata, pd.DataFrame):
                raise TypeError(
                    "MatrixTranspose feature_metadata must be a pandas DataFrame."
                )
            if len(feature_metadata) != len(feature_names):
                raise ValueError(
                    "MatrixTranspose feature_metadata rows must match the feature "
                    f"width: {len(feature_metadata)} != {len(feature_names)}."
                )
            metadata = _transpose_feature_metadata(
                feature_metadata,
                row_name=self.row_name,
            )
            result["metadata"] = metadata
            result["row_name_column"] = self.row_name
        return result

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = single_source(sources, name="MatrixTranspose")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "MatrixTranspose requires a sparse_matrix or dense_matrix source."
            )
        return OutputSpec(
            artifact_type=source.artifact_type.value,
            lineage_mode="new_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
            feature_metadata_mode="own",
        )

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        unknown = sorted(set(params) - {"feature_metadata_columns"})
        if unknown:
            raise OperatorError(
                f"Unknown MatrixTranspose parameter(s): {unknown}."
            )
        source = single_source(sources, name="MatrixTranspose")
        promoted, metadata_request = _resolve_promoted_columns(
            source,
            params.get("feature_metadata_columns", False),
        )
        return {
            "feature_metadata_columns": list(promoted),
            "metadata_columns": list(metadata_request),
        }

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        if mode != "translate":
            raise OperatorError(f"Unsupported MatrixTranspose mode {mode!r}.")
        source = single_source(sources, name="MatrixTranspose")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "MatrixTranspose requires a sparse_matrix or dense_matrix source."
            )
        self._source_type = source.artifact_type.value
        feature_getter = getattr(source, "get_feature_metadata", None)
        if not callable(feature_getter):
            feature_getter = getattr(source, "get_feature_frame", None)
        if callable(feature_getter):
            feature_metadata = feature_getter()
            if not isinstance(feature_metadata, pd.DataFrame):
                raise OperatorError(
                    "MatrixTranspose source get_feature_metadata() must return a pandas DataFrame."
                )
            feature_metadata = feature_metadata.reset_index(drop=True).copy()
        else:
            feature_metadata = pd.DataFrame(
                {"column": [str(value) for value in source.get_data_columns()]}
            )

        self._source_features = tuple(str(value) for value in source.get_data_columns())
        if len(feature_metadata) != len(self._source_features):
            raise OperatorError(
                "MatrixTranspose source Feature Metadata length does not match source "
                f"feature width: {len(feature_metadata)} != {len(self._source_features)}."
            )
        self._source_feature_metadata = feature_metadata

        promoted = tuple(
            str(value) for value in request.params.get("feature_metadata_columns", ())
        )
        metadata_request = [
            str(value) for value in request.params.get("metadata_columns", ())
        ]
        self._promoted_feature_columns = promoted

        self._source_row_label_column = None
        source_row_name = getattr(source, "row_name", None)
        if isinstance(source_row_name, str) and source_row_name:
            query_info = source.query_columns(metadata_mode="full")
            mapping = dict(query_info.get("mapping", {}))
            resolved = mapping.get(f"metadata.{source_row_name}", source_row_name)
            self._source_row_label_column = str(resolved)
            if resolved not in metadata_request:
                metadata_request.append(str(resolved))

        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="full_artifact",
            columns=ColumnRequest(
                keys=True,
                data=True,
                metadata=metadata_request or False,
            ),
            batch_size=None,
            form="native",
            metadata_mode="full" if metadata_request else "none",
            include_position=True,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported MatrixTranspose mode {mode!r}.")

        packet = single_input(inputs, name="MatrixTranspose")
        info, matrix, key_columns = native_matrix_packet(packet, name="MatrixTranspose")
        source_features = self._require_source_features()
        feature_metadata = self._require_source_feature_metadata()
        if int(matrix.shape[1]) != len(source_features):
            raise ArtifactError(
                "MatrixTranspose source feature count changed between planning and "
                f"execution: expected {len(source_features)}, got {matrix.shape[1]}."
            )

        row_labels = self._source_row_labels(info, key_columns)
        translated = self.translate(
            matrix,
            features=source_features,
            row_labels=row_labels,
            feature_metadata=feature_metadata,
            row_metadata=info,
            feature_metadata_columns=self._promoted_feature_columns,
        )
        keys = pd.DataFrame(
            {self.key_name: np.arange(len(source_features), dtype=np.int64)}
        )
        data = {
            "values": translated["values"],
            "columns": translated["columns"],
            "feature_metadata": translated["feature_metadata"],
        }
        output: dict[str, Any] = {
            "keys": keys,
            "metadata": translated["metadata"],
            "row_name_column": translated["row_name_column"],
            "data": data,
        }
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: output})

    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def to_json_state(self) -> dict[str, Any]:
        return {"key_name": self.key_name, "row_name": self.row_name}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "MatrixTranspose":
        return cls(
            key_name=str(state.get("key_name", "feature_id")),
            row_name=str(state.get("row_name", "feature")),
        )

    def _require_source_features(self) -> tuple[str, ...]:
        if self._source_features is None:
            raise OperatorError("MatrixTranspose has no bound source feature schema.")
        return self._source_features

    def _require_source_feature_metadata(self) -> pd.DataFrame:
        if self._source_feature_metadata is None:
            raise OperatorError("MatrixTranspose has no bound source feature metadata.")
        return self._source_feature_metadata

    def _source_row_labels(
        self,
        info: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> list[str]:
        if self._source_row_label_column is not None:
            column = self._source_row_label_column
            if column not in info.columns:
                raise ArtifactError(
                    f"MatrixTranspose source packet is missing row-name column {column!r}."
                )
            names = [str(value) for value in info[column].tolist()]
            if len(set(names)) != len(names):
                raise ArtifactError(
                    "MatrixTranspose source row names must be unique."
                )
            return names

        if not key_columns:
            raise ArtifactError("MatrixTranspose source has no primary-key columns.")
        missing = [column for column in key_columns if column not in info.columns]
        if missing:
            raise ArtifactError(
                f"MatrixTranspose source packet is missing key columns {missing}."
            )

        if len(key_columns) == 1:
            column = key_columns[0]
            return [f"{column}={_scalar_key(value)}" for value in info[column].tolist()]

        labels: list[str] = []
        for values in info.loc[:, list(key_columns)].itertuples(index=False, name=None):
            labels.append(
                "|".join(
                    f"{name}={_scalar_key(value)}"
                    for name, value in zip(key_columns, values, strict=True)
                )
            )
        return labels


def _resolve_promoted_columns(
    source: "BaseArtifact",
    selection: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve row-level key/metadata fields selected for eager promotion."""
    if selection is None or selection is False:
        return (), ()

    query_info = source.query_columns(metadata_mode="full")
    columns = tuple(query_info.get("columns", ()))
    by_output = {str(column["output_name"]): column for column in columns}

    if selection is True:
        requested = [
            str(column["output_name"])
            for column in columns
            if column["namespace"] in {"key", "metadata"}
        ]
    elif isinstance(selection, str):
        requested = [selection]
    elif isinstance(selection, Sequence):
        requested = [str(value) for value in selection]
    else:
        raise OperatorError(
            "MatrixTranspose feature_metadata_columns must be a column name, "
            "sequence of names, True, False, or None."
        )

    if len(set(requested)) != len(requested):
        raise OperatorError(
            f"Duplicate MatrixTranspose feature_metadata_columns: {requested!r}."
        )

    source_row_name = getattr(source, "row_name", None)
    mapping = dict(query_info.get("mapping", {}))
    resolved_row_name = (
        None
        if not isinstance(source_row_name, str) or not source_row_name
        else str(mapping.get(f"metadata.{source_row_name}", source_row_name))
    )

    promoted: list[str] = []
    metadata_request: list[str] = []
    for name in requested:
        column = by_output.get(name)
        if column is None:
            ambiguous = query_info.get("ambiguous", {})
            if name in ambiguous:
                raise OperatorError(
                    f"MatrixTranspose column {name!r} is ambiguous; use one of "
                    f"{ambiguous[name]}."
                )
            raise OperatorError(
                f"MatrixTranspose column {name!r} is not available. Available "
                f"row-level columns are {sorted(by_output)}."
            )
        namespace = str(column["namespace"])
        if namespace not in {"key", "metadata"}:
            raise OperatorError(
                f"MatrixTranspose can promote only row key/metadata fields; "
                f"{name!r} resolves to {namespace!r}."
            )
        if name == resolved_row_name:
            continue
        if name in _RESERVED_FEATURE_METADATA_NAMES:
            raise OperatorError(
                f"MatrixTranspose cannot promote reserved Feature Metadata field "
                f"{name!r}."
            )
        promoted.append(name)
        if namespace == "metadata":
            metadata_request.append(name)

    return tuple(promoted), tuple(metadata_request)


def _transpose_feature_metadata(
    feature_metadata: pd.DataFrame,
    *,
    row_name: str,
) -> pd.DataFrame:
    """Eagerly promote source Feature Metadata to output row metadata."""
    metadata = feature_metadata.reset_index(drop=True).copy()
    if "column_index" in metadata.columns:
        metadata = metadata.drop(columns=["column_index"])
    if "column" not in metadata.columns:
        raise ValueError("MatrixTranspose Feature Metadata must contain 'column'.")
    if row_name != "column" and row_name in metadata.columns:
        raise ValueError(
            f"MatrixTranspose row_name {row_name!r} conflicts with Feature Metadata."
        )
    metadata = metadata.rename(columns={"column": row_name})
    return metadata


def _scalar_key(value: Any) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _validate_axis_name(value: str, *, label: str, reserved: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    normalized = value.strip()
    if reserved and normalized in _RESERVED_KEY_NAMES:
        raise ValueError(
            f"{label} may not use reserved structural name {normalized!r}."
        )
    return normalized
