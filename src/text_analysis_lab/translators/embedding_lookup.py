"""Generic lookup of named matrix rows from table values."""

from __future__ import annotations

from collections.abc import Mapping
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

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


TOKEN_SOURCE = "tokens"
EMBEDDING_SOURCE = "embeddings"
OovPolicy = Literal["zero", "error"]


class EmbeddingLookup(BaseTranslator):
    """Map a table field to vectors in any named-row matrix artifact.

    The token table and embedding matrix are explicit sources, so the resulting
    matrix has complete provenance without coupling lookup to the model that
    originally produced the embeddings.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        field: str,
        oov_policy: OovPolicy = "zero",
        drop_empty: bool = True,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(field, str) or not field:
            raise ValueError("field must be a non-empty string.")
        if oov_policy not in {"zero", "error"}:
            raise ValueError("oov_policy must be 'zero' or 'error'.")
        self.field = field
        self.oov_policy = oov_policy
        self.drop_empty = bool(drop_empty)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        tokens, embeddings = _validate_sources(sources)
        _ = tokens
        return OutputSpec(
            artifact_type=embeddings.artifact_type,
            lineage_mode="preserved_key",
            basis_labels=TOKEN_SOURCE,
        )

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                f"EmbeddingLookup does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> Mapping[str, SourceRequest]:
        _ = mode, request
        _validate_sources(sources)
        # Both are materialized once. This avoids retaining a large vocabulary
        # inside operator state and keeps every lookup operation self-contained.
        return {
            TOKEN_SOURCE: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=self.field, metadata=False),
                batch_size=None,
                form="table",
                metadata_mode="none",
                include_position=False,
            ),
            EMBEDDING_SOURCE: SourceRequest(
                artifact_type=("dense_matrix", "sparse_matrix"),
                mode="full_artifact",
                columns=ColumnRequest(keys=False, data=True, metadata=False),
                batch_size=None,
                form="native",
                metadata_mode="none",
                include_position=True,
            ),
        }

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = mode, request
        if set(inputs) != {TOKEN_SOURCE, EMBEDDING_SOURCE}:
            raise OperatorError(
                f"EmbeddingLookup expects inputs {TOKEN_SOURCE!r} and "
                f"{EMBEDDING_SOURCE!r}; got {sorted(inputs)}."
            )
        token_packet = inputs[TOKEN_SOURCE]
        embedding_packet = inputs[EMBEDDING_SOURCE]
        tokens = token_packet.data
        native = embedding_packet.data
        if not isinstance(tokens, pd.DataFrame):
            raise ArtifactError("EmbeddingLookup expected table-form token data.")
        if self.field not in tokens.columns:
            raise ArtifactError(f"Token source is missing field {self.field!r}.")
        if not isinstance(native, Mapping):
            raise ArtifactError("EmbeddingLookup expected native matrix data.")
        matrix = native.get("matrix")
        row_names = native.get("row_names")
        columns = native.get("columns")
        if matrix is None or row_names is None or columns is None:
            raise ArtifactError(
                "EmbeddingLookup requires a matrix source with named rows."
            )
        row_names = [str(value) for value in row_names]
        if len(row_names) != int(matrix.shape[0]):
            raise ArtifactError("Embedding row_names are not aligned with matrix rows.")
        position_by_name = {name: position for position, name in enumerate(row_names)}
        if len(position_by_name) != len(row_names):
            raise ArtifactError("Embedding matrix row_names must be unique.")

        raw_words = tokens[self.field]
        words = raw_words.astype("string")
        valid = raw_words.notna()
        if self.drop_empty:
            valid &= words.str.len().fillna(0).gt(0)
        positions = words.map(position_by_name).fillna(-1).to_numpy(dtype=np.int64)
        positions[~valid.to_numpy(dtype=bool)] = -1
        in_vocabulary = positions >= 0
        if self.oov_policy == "error" and not bool(np.all(in_vocabulary)):
            examples = (
                words.loc[~pd.Series(in_vocabulary, index=tokens.index)]
                .fillna("<null>")
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            raise ArtifactError(
                f"EmbeddingLookup encountered {int((~in_vocabulary).sum())} "
                f"out-of-vocabulary row(s). Example(s): {examples}."
            )

        output = _lookup_rows(matrix, positions, in_vocabulary)
        key_columns = list(token_packet.primary_key)
        return BatchResult(
            outputs={
                "output": {
                    "keys": tokens.loc[:, key_columns].reset_index(drop=True),
                    "metadata": pd.DataFrame(
                        {
                            self.field: words.fillna("<null>").astype(str).to_numpy(),
                            "in_vocabulary": in_vocabulary,
                        }
                    ),
                    "data": {"values": output, "columns": list(columns)},
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
        return {
            "field": self.field,
            "oov_policy": self.oov_policy,
            "drop_empty": self.drop_empty,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> EmbeddingLookup:
        return cls(
            field=str(state["field"]),
            oov_policy=cast(OovPolicy, state.get("oov_policy", "zero")),
            drop_empty=bool(state.get("drop_empty", True)),
        )


def _lookup_rows(matrix: Any, positions: np.ndarray, valid: np.ndarray) -> Any:
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a package dependency
        sparse = None
    if sparse is not None and sparse.issparse(matrix):
        matrix = matrix.tocsr()
        if int(matrix.shape[0]) == 0:
            return sparse.csr_matrix(
                (len(positions), int(matrix.shape[1])), dtype=matrix.dtype
            )
        safe_positions = positions.copy()
        safe_positions[~valid] = 0
        output = matrix[safe_positions].copy().tocsr()
        for row in np.flatnonzero(~valid):
            start = int(output.indptr[row])
            stop = int(output.indptr[row + 1])
            output.data[start:stop] = 0
        output.eliminate_zeros()
        return output.tocsr()
    dense = np.asarray(matrix)
    output = np.zeros((len(positions), int(dense.shape[1])), dtype=dense.dtype)
    output[valid] = dense[positions[valid]]
    return output


def _validate_sources(
    sources: Mapping[str, BaseArtifact],
) -> tuple[BaseArtifact, BaseArtifact]:
    if set(sources) != {TOKEN_SOURCE, EMBEDDING_SOURCE}:
        raise OperatorError(
            f"EmbeddingLookup expects source labels {TOKEN_SOURCE!r} and "
            f"{EMBEDDING_SOURCE!r}; got {sorted(sources)}."
        )
    tokens = sources[TOKEN_SOURCE]
    embeddings = sources[EMBEDDING_SOURCE]
    if tokens.artifact_type.value != "table":
        raise OperatorError("EmbeddingLookup tokens source must be a table artifact.")
    if embeddings.artifact_type.value not in {"dense_matrix", "sparse_matrix"}:
        raise OperatorError(
            "EmbeddingLookup embeddings source must be a matrix artifact."
        )
    if not getattr(embeddings, "has_row_names", False):
        raise OperatorError("EmbeddingLookup embeddings source must define named rows.")
    return tokens, embeddings
