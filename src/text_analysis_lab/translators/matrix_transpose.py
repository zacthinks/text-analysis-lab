"""Transpose/reindex a TeAL matrix so features become named rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BatchResult,
    BaseTranslator,
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
    native_matrix_packet,
    single_input,
    single_source,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_RESERVED_KEY_NAMES = frozenset({"_position", "_batch", "_row_offset"})


class MatrixTranspose(BaseTranslator):
    """Transpose a dense or sparse matrix and reindex features as rows.

    The source matrix's feature/column labels become unique human-readable row
    names on the output. The output receives a fresh integer primary key
    (``feature_id`` by default), because transposition changes the observation
    universe rather than preserving source row identity.

    Output columns identify source rows. If the source already has unique named
    rows, those names are reused. Otherwise TeAL derives deterministic labels
    from the source primary-key values.
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
        self._source_has_row_names: bool = False

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
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
        )

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                "MatrixTranspose does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported MatrixTranspose mode {mode!r}.")
        source = single_source(sources, name="MatrixTranspose")
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "MatrixTranspose requires a sparse_matrix or dense_matrix source."
            )
        self._source_type = source.artifact_type.value
        self._source_features = tuple(str(value) for value in source.get_data_columns())
        self._source_has_row_names = bool(getattr(source, "has_row_names", False))
        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="full_artifact",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=None,
            form="native",
            metadata_mode="none",
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
        info, matrix, key_columns = native_matrix_packet(
            packet, name="MatrixTranspose"
        )
        source_features = self._require_source_features()
        if int(matrix.shape[1]) != len(source_features):
            raise ArtifactError(
                "MatrixTranspose source feature count changed between planning and "
                f"execution: expected {len(source_features)}, got {matrix.shape[1]}."
            )

        row_labels = self._source_row_labels(packet, info, key_columns)
        if len(row_labels) != int(matrix.shape[0]):
            raise ArtifactError(
                "MatrixTranspose source row-label count does not match matrix rows."
            )
        if len(set(row_labels)) != len(row_labels):
            raise ArtifactError(
                "MatrixTranspose requires unique source-row labels after string conversion."
            )

        if self._source_type == "sparse_matrix":
            values = matrix.transpose().tocsr()
        else:
            values = np.asarray(matrix).T.copy()

        keys = pd.DataFrame(
            {self.key_name: np.arange(len(source_features), dtype=np.int64)}
        )
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "data": {
                        "values": values,
                        "columns": row_labels,
                        "row_names": list(source_features),
                        "row_name": self.row_name,
                    },
                }
            }
        )

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

    def _source_row_labels(
        self,
        packet: InputBatch,
        info: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> list[str]:
        if self._source_has_row_names and isinstance(packet.data, Mapping):
            raw_names = packet.data.get("row_names")
            if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
                names = [str(value) for value in raw_names]
                if len(names) == len(info):
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


def _scalar_key(value: Any) -> str:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    # TeAL primary keys are currently integer-only, but keep deterministic JSON
    # formatting at this boundary if that contract broadens later.
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
