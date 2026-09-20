"""Aggregate matrix rows at a retained primary-key prefix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

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

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


Pooling = Literal["mean", "sum"]


class MatrixRowAggregator(BaseTranslator):
    """Pool matrix rows by a proper primary-key prefix using mean or sum."""

    operation_type = "translate"

    def __init__(
        self,
        *,
        group_by: str | Sequence[str],
        pooling: Pooling = "mean",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.group_by = _normalize_columns(group_by)
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'.")
        self.pooling = pooling

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = _single_source(sources)
        self._validate_key(source.primary_key)
        return OutputSpec(
            artifact_type=source.artifact_type,
            lineage_mode="reduced_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode, request
        source = _single_source(sources)
        self._validate_key(source.primary_key)
        return SourceRequest(
            artifact_type=("dense_matrix", "sparse_matrix"),
            mode="full_artifact",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=None,
            form="native",
            metadata_mode="none",
            include_position=False,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = mode, request
        packet = _single_input(inputs)
        native = packet.data
        if not isinstance(native, Mapping):
            raise ArtifactError("MatrixRowAggregator expected native matrix data.")
        info = native.get("info")
        matrix = native.get("matrix")
        columns = native.get("columns")
        if not isinstance(info, pd.DataFrame) or matrix is None or columns is None:
            raise ArtifactError("MatrixRowAggregator received malformed matrix data.")
        if info.empty:
            raise ArtifactError(
                "MatrixRowAggregator cannot aggregate an empty artifact."
            )
        self._validate_key(packet.primary_key)
        missing = [name for name in self.group_by if name not in info.columns]
        if missing:
            raise ArtifactError(
                f"Matrix source is missing group key columns {missing}."
            )

        group_frame = info.loc[:, list(self.group_by)].reset_index(drop=True)
        group_index = pd.MultiIndex.from_frame(group_frame)
        codes, uniques = pd.factorize(group_index, sort=False)
        n_groups = len(uniques)
        pooled_rows = []
        counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
        for group_id in range(n_groups):
            block = matrix[codes == group_id]
            row = block.sum(axis=0)
            if self.pooling == "mean":
                row = row / int(counts[group_id])
            pooled_rows.append(row)
        pooled = _stack_rows(pooled_rows, matrix)
        keys = group_frame.loc[~group_index.duplicated(keep="first")].reset_index(
            drop=True
        )
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "metadata": pd.DataFrame({"n_rows": counts}),
                    "data": {"values": pooled, "columns": list(columns)},
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
        return {"group_by": list(self.group_by), "pooling": self.pooling}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> MatrixRowAggregator:
        return cls(
            group_by=cast(Sequence[str], state["group_by"]),
            pooling=cast(Pooling, state.get("pooling", "mean")),
        )

    def _validate_key(self, source_key: Sequence[str]) -> None:
        source_key = tuple(str(value) for value in source_key)
        if (
            len(self.group_by) >= len(source_key)
            or source_key[: len(self.group_by)] != self.group_by
        ):
            raise OperatorError(
                "group_by must be a non-empty proper prefix of the matrix primary "
                f"key {list(source_key)}; got {list(self.group_by)}."
            )


def _stack_rows(rows: list[Any], source: Any) -> Any:
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a package dependency
        sparse = None
    if sparse is not None and sparse.issparse(source):
        return sparse.vstack([sparse.csr_matrix(row) for row in rows], format="csr")
    return np.vstack([np.asarray(row).reshape(1, -1) for row in rows])


def _normalize_columns(value: str | Sequence[str]) -> tuple[str, ...]:
    columns = (value,) if isinstance(value, str) else tuple(value)
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError("group_by must contain non-empty string column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("group_by cannot contain duplicate columns.")
    return columns


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"MatrixRowAggregator expects exactly source label {DEFAULT_SOURCE_LABEL!r}; "
            f"got {sorted(sources)}."
        )
    source = sources[DEFAULT_SOURCE_LABEL]
    if source.artifact_type.value not in {"dense_matrix", "sparse_matrix"}:
        raise OperatorError("MatrixRowAggregator requires a matrix artifact source.")
    return source


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"MatrixRowAggregator expects exactly input label {DEFAULT_SOURCE_LABEL!r}; "
            f"got {sorted(inputs)}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]
