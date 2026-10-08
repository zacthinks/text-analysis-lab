"""Matrix query SQL clauses must fail clearly rather than reaching DuckDB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError


@pytest.fixture(params=["dense_matrix", "sparse_matrix"])
def matrix_artifact(tmp_path: Path, request):
    project = teal.Project.create(tmp_path / "project", name="matrix_query_guard")
    values = np.array([[7, 8], [2, 9], [6, 7], [8, 2], [5, 6]], dtype=np.int64)
    if request.param == "sparse_matrix":
        values = sparse.csr_matrix(values)
    matrix = project.register_external(
        [
            {
                "keys": pd.DataFrame({"doc_id": range(5)}),
                "metadata": pd.DataFrame({"year": [2020, 2021, 2022, 2023, 2024]}),
                "data": {"values": values, "columns": ["apple", "orange"]},
            }
        ],
        artifact_type=request.param,
        primary_key="doc_id",
    )
    try:
        yield matrix
    finally:
        project.close()


@pytest.mark.parametrize(
    "clauses",
    [
        {"where": '"apple" >= 5 AND "orange" >= 6'},
        {"order_by": '"orange" DESC'},
        {"where": '"apple" >= 5', "order_by": '"orange" DESC'},
    ],
)
def test_matrix_feature_sql_fails_with_actionable_error(matrix_artifact, clauses):
    with pytest.raises(QueryError, match="Matrix feature") as exc:
        matrix_artifact.query(**clauses)
    message = str(exc.value)
    assert "data_columns" in message
    assert "include_position=True" in message
    assert "query(positions=...)" in message


def test_matrix_feature_sql_fails_in_streaming_query(matrix_artifact):
    with pytest.raises(QueryError, match="Matrix feature"):
        list(matrix_artifact.query(where='"apple" > 0', iter_batches=True))


@pytest.mark.parametrize("condition,expected", [
    ("doc_id >= 2", [2, 3, 4]),
    ("year >= 2022", [2, 3, 4]),
    ("_position < 2", [0, 1]),
])
def test_matrix_sql_filter_on_relational_columns(matrix_artifact, condition, expected):
    frame = matrix_artifact.query(
        where=condition,
        metadata_mode="full",
        metadata_columns=True,
        form="table",
    )
    assert frame["doc_id"].tolist() == expected


@pytest.mark.parametrize("order", ["doc_id DESC", "year DESC"])
def test_matrix_sql_sort_on_relational_columns(matrix_artifact, order):
    frame = matrix_artifact.query(
        order_by=order,
        metadata_mode="full",
        metadata_columns=True,
        form="table",
    )
    assert frame["doc_id"].tolist() == [4, 3, 2, 1, 0]


def test_matrix_sql_where_and_order_together(matrix_artifact):
    frame = matrix_artifact.query(
        where="year >= 2022",
        order_by="doc_id DESC",
        metadata_mode="full",
        form="table",
    )
    assert frame["doc_id"].tolist() == [4, 3, 2]


def test_missing_nonfeature_column_retains_normal_error(matrix_artifact):
    with pytest.raises(QueryError, match="Artifact query failed"):
        matrix_artifact.query(where='"not_a_feature" >= 5')


def test_matrix_feature_table_filter_then_position_query(matrix_artifact):
    table = matrix_artifact.query(
        key_columns=True,
        data_columns=["apple", "orange"],
        form="table",
        include_position=True,
    )
    selected = table.loc[(table["apple"] >= 5) & (table["orange"] >= 6)]
    assert selected["doc_id"].tolist() == [0, 2, 4]
    result = matrix_artifact.query(
        positions=selected["_position"].astype(int).tolist(),
        form="native",
    )
    assert result["info"]["doc_id"].tolist() == [0, 2, 4]
    values = result["matrix"]
    if sparse.issparse(values):
        values = values.toarray()
    np.testing.assert_array_equal(values, [[7, 8], [6, 7], [5, 6]])


def test_matrix_positional_order_is_still_valid(matrix_artifact):
    for order_by in (None, "_position", ["_position"]):
        result = matrix_artifact.query(
            data_columns=False, order_by=order_by, form="table"
        )
        assert result["doc_id"].tolist() == [0, 1, 2, 3, 4]


def test_table_sql_clauses_remain_supported(tmp_path: Path):
    project = teal.Project.create(tmp_path / "project", name="table_query_guard")
    try:
        artifact = project.register_external(
            pd.DataFrame({"doc_id": [0, 1, 2], "count": [7, 2, 6]}),
            primary_key="doc_id",
            data_fields="count",
        )
        result = artifact.query(
            where='"count" >= 5', order_by='"count" DESC', form="table"
        )
        assert result["doc_id"].tolist() == [0, 2]
    finally:
        project.close()


def test_matrix_position_window_is_valid_sql(matrix_artifact):
    frame = matrix_artifact.query(
        where="_position >= 1 AND _position < 3", form="table"
    )
    assert frame["doc_id"].tolist() == [1, 2]
