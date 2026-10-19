"""Opt-in computed matrix assembly, independent of stored and inherited read paths.

Only requested source rows are materialized. Persist the recipe separately;
this module is intentionally pure, and never writes numeric artifact data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.translators._feature_matrix_encoding import (
    FeatureMatrixEncoding, append_feature_columns,
)


def render_feature_matrix(
    source: Any,
    *,
    positions: Sequence[int],
    matrix_columns: Sequence[str] | bool = True,
    data_columns: Sequence[str] | bool = False,
    metadata_columns: Sequence[str] | bool = False,
    key_columns: Sequence[str] | bool = False,
    encoder: FeatureMatrixEncoding,
    sparse_output: bool,
    output_columns: Sequence[str] | None = None,
) -> Any:
    """Render selected positions using the caller's existing artifact query API.

    Called *only* by recipe-backed matrix artifacts, never by the ordinary
    stored or inherited matrix readers. A source matrix's numerical values stay
    in its native sparse representation. Key columns selected as features are
    duplicated deliberately while still remaining structural identifiers.
    """
    if not encoder.is_fitted:
        raise ArtifactError("FeatureMatrix requires a frozen fitted encoder.")

    source_type = source.artifact_type.value
    is_matrix = source_type in {"dense_matrix", "sparse_matrix"}
    chosen_positions = [int(p) for p in positions]

    if is_matrix:
        base = source.query(
            positions=chosen_positions, key_columns=False,
            data_columns=matrix_columns, metadata_columns=False,
            form="native",
        )["matrix"] if matrix_columns is not False else None
        scalar_data = False
    else:
        base = None
        scalar_data = data_columns

    # The query result is at the requested row positions in requested order;
    # its columns are resolved using TeAL's own full metadata semantics.
    fields = source.query(
        positions=chosen_positions,
        key_columns=key_columns,
        data_columns=scalar_data,
        metadata_columns=metadata_columns,
        metadata_mode="full" if metadata_columns is not False else "none",
        form="table",
    )
    if not isinstance(fields, pd.DataFrame):
        raise ArtifactError("FeatureMatrix scalar query must return a table.")
    if len(fields) != len(chosen_positions):
        raise ArtifactError("FeatureMatrix source returned an unexpected row count.")

    numeric = encoder.transform(fields, sparse_output=sparse_output)
    output = append_feature_columns(base, numeric, sparse_output=sparse_output)
    if output_columns is not None and output.shape[1] != len(output_columns):
        raise ArtifactError(
            "FeatureMatrix frozen output feature width does not match its recipe."
        )
    if sparse_output and not sparse.issparse(output):
        raise ArtifactError("FeatureMatrix sparse output unexpectedly densified.")
    if not sparse_output and not isinstance(output, np.ndarray):
        raise ArtifactError("FeatureMatrix dense output must be an ndarray.")
    return output
