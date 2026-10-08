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
        {"where": "doc_id >= 2"},
        {"where": "year >= 2022", "metadata_mode": "full"},
        {"order_by": '"orange" DESC'},
        {"order_by": "doc_id DESC"},
        {"where": '"apple" >= 5', "order_by": '"orange" DESC'},
    ],
)
def test_matrix_sql_clauses_fail_with_actionable_error(matrix_artifact, clauses):
    with pytest.raises(QueryError, match="Select the needed features with") as exc:
        matrix_artifact.query(**clauses)
    message = str(exc.value)
    assert "dense or sparse" in message
    assert "data_columns" in message
    assert "include_position=True" in message
    assert "query(positions=...)" in message


def test_matrix_sql_clauses_fail_for_streaming_queries(matrix_artifact):
    with pytest.raises(QueryError, match="matrix feature values"):
        matrix_artifact.query(where='"apple" > 0', iter_batches=True)


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
