from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from text_analysis_lab import agg, concat, literal
from text_analysis_lab.core.aggregate import (
    _normalize_field_spec,
    _pool_matrix_by_complete_groups,
    _reduce_ordered_values,
)


def test_public_aggregate_spec_helpers_are_typed_and_output_centric():
    rules = _normalize_field_spec(
        {
            "probability": "mean",
            "probability_sum": agg("probability", "sum"),
            "course_number": literal(12),
            "joined": agg("tag", concat("|")),
        },
        label="data",
    )
    assert rules["probability"].source_name == "probability"
    assert rules["probability"].reducer == "mean"
    assert rules["probability_sum"].source_name == "probability"
    assert rules["probability_sum"].reducer == "sum"
    assert rules["course_number"].is_literal is True
    assert rules["course_number"].literal_value == 12
    assert rules["joined"].source_name == "tag"
    assert rules["joined"].reducer.separator == "|"


def test_literal_rejects_implicit_mini_language_and_arbitrary_objects():
    assert literal("mean").value == "mean"
    with pytest.raises(TypeError, match="supports only"):
        literal({"not": "a scalar"})


def test_ordered_reducers_use_source_order_for_mode_first_last_unique_concat():
    values = ["b", "a", "a", "b"]
    assert _reduce_ordered_values(values, "first") == "b"
    assert _reduce_ordered_values(values, "last") == "b"
    # b and a tie 2-2; first appearance in source-primary-key order wins.
    assert _reduce_ordered_values(values, "mode") == "b"
    assert _reduce_ordered_values(values, "unique") == ["b", "a"]
    assert _reduce_ordered_values(values, concat("|")) == "b|a|a|b"


def test_sparse_matrix_pooling_is_group_safe_and_stays_sparse():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2],
                [0, 1, 0],
                [4, 0, 0],
                [0, 2, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
    )
    counts = np.array([2, 3], dtype=np.int64)

    summed = _pool_matrix_by_complete_groups(matrix, group_counts=counts, pooling="sum")
    assert sparse.issparse(summed)
    assert summed.toarray() == pytest.approx(
        np.array([[1, 1, 2], [5, 3, 2]], dtype=float)
    )

    meaned = _pool_matrix_by_complete_groups(
        matrix, group_counts=counts, pooling="mean"
    )
    assert sparse.issparse(meaned)
    assert meaned.toarray() == pytest.approx(
        np.array([[0.5, 0.5, 1.0], [5 / 3, 1.0, 2 / 3]], dtype=float)
    )


def test_dense_matrix_pooling_uses_whole_group_boundaries():
    matrix = np.arange(18, dtype=float).reshape(6, 3)
    counts = np.array([1, 2, 3], dtype=np.int64)
    pooled = _pool_matrix_by_complete_groups(matrix, group_counts=counts, pooling="sum")
    assert pooled == pytest.approx(
        np.vstack(
            [
                matrix[0],
                matrix[1:3].sum(axis=0),
                matrix[3:6].sum(axis=0),
            ]
        )
    )
