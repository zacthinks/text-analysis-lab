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
    selected_indices: Sequence[int] | None = None,
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

    matrix_names = list(matrix_columns) if matrix_columns is not False and matrix_columns is not True else (
        (list(output_columns[:len(output_columns) - len(encoder.columns)]) if output_columns is not None else list(source.get_data_columns())) if is_matrix and matrix_columns is True else []
    )
    matrix_width = len(matrix_names)
    full_width = matrix_width + len(encoder.columns)
    indices = list(range(full_width)) if selected_indices is None else [
        int(index) for index in selected_indices
    ]
    if any(index < 0 or index >= full_width for index in indices):
        raise ArtifactError("FeatureMatrix selected feature index is out of range.")

    matrix_indices = [i for i in indices if i < matrix_width]
    scalar_indices = [i - matrix_width for i in indices if i >= matrix_width]
    needed_matrix_names = list(dict.fromkeys(matrix_names[i] for i in matrix_indices))
    needed_scalar_names = encoder.required_columns(scalar_indices)

    base = None
    if needed_matrix_names:
        base = source.query(
            positions=chosen_positions, key_columns=False,
            data_columns=needed_matrix_names, metadata_columns=False,
            form="native",
        )["matrix"]

    keys = [name for name in needed_scalar_names if name in (key_columns if key_columns is not False else [])]
    data = [name for name in needed_scalar_names if name in (data_columns if data_columns is not False else [])]
    meta = [name for name in needed_scalar_names if name in (metadata_columns if metadata_columns is not False else [])]
    if needed_scalar_names:
        fields = source.query(
            positions=chosen_positions,
            key_columns=keys if keys else False,
            data_columns=data if data else False,
            metadata_columns=meta if meta else False,
            metadata_mode="full",
            form="table",
        )
        if len(fields) != len(chosen_positions):
            raise ArtifactError("FeatureMatrix source returned an unexpected row count.")
        fields = fields.loc[:, needed_scalar_names]
    else:
        fields = pd.DataFrame(index=range(len(chosen_positions)))
    numeric = encoder.transform(fields, sparse_output=sparse_output, indices=scalar_indices)

    from text_analysis_lab.translators._feature_matrix_encoding import append_feature_columns
    combined = append_feature_columns(base, numeric, sparse_output=sparse_output)
    # Reconstitute exact requested ordering and duplicates after querying each
    # distinct source column only once.
    matrix_lookup = {name: index for index, name in enumerate(needed_matrix_names)}
    scalar_lookup = {index: j for j, index in enumerate(scalar_indices)}
    base_count = len(needed_matrix_names)
    permutation = [
        matrix_lookup[matrix_names[index]] if index < matrix_width
        else base_count + scalar_lookup[index - matrix_width]
        for index in indices
    ]
    output = combined[:, permutation]
    if output_columns is not None and full_width != len(output_columns):
        raise ArtifactError("FeatureMatrix frozen output feature width does not match its recipe.")
    if sparse_output and not sparse.issparse(output):
        raise ArtifactError("FeatureMatrix sparse output unexpectedly densified.")
    if not sparse_output and not isinstance(output, np.ndarray):
        raise ArtifactError("FeatureMatrix dense output must be an ndarray.")
    return output
