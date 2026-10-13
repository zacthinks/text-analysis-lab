"""Phase 1B: keyed matrix attachment, feature projection, nulls and SQL metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError


@pytest.mark.parametrize("representation", ["dense_matrix", "sparse_matrix"])
@pytest.mark.parametrize("caller_type", ["table", "dense_matrix", "sparse_matrix"])
def test_attached_matrix_values_zero_and_missing(tmp_path, representation, caller_type):
    project = teal.Project.create(tmp_path / "project", name="matrix_attach")
    try:
        left_frame = pd.DataFrame({
            "doc_id": [1, 1, 2, 3], "sentence_id": [0, 1, 0, 0],
            "text": ["a", "b", "c", "d"],
        })
        if caller_type != "table":
            caller_values = np.arange(8).reshape((4, 2))
            if caller_type == "sparse_matrix":
                caller_values = sparse.csr_matrix(caller_values)
            left = project.register_external([{
                "keys": left_frame[["doc_id", "sentence_id"]],
                "data": {"values": caller_values,
                         "columns": ["x", "y"]},
            }], artifact_type=caller_type,
                primary_key=["doc_id", "sentence_id"])
        else:
            left = project.register_external(
                left_frame, primary_key=["doc_id", "sentence_id"],
                data_fields="text",
            )
        values = np.asarray([[0, 4], [7, 0]], dtype=float)
        if representation == "sparse_matrix":
            values = sparse.csr_matrix(values)
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "metadata": pd.DataFrame({"rank": [10, 20]}),
            "data": {"values": values, "columns": ["topic_0", "topic_1"]},
        }], artifact_type=representation, primary_key="doc_id")

        fields = left.query_columns(with_artifacts=right)
        assert "topic_0" in fields["with"]
        assert "rank" in fields["with"]
        features = [c for c in fields["columns"] if c["base_name"] == "topic_0"]
        assert len(features) == 1
        assert features[0]["sql_addressable"] is False

        result = left.query(
            with_artifacts=right,
            with_columns=["topic_0", "topic_1", "rank"],
            form="table",
        )
        assert len(result) == 4
        assert result["topic_0"].iloc[:3].tolist() == [0, 0, 7]
        assert result["topic_1"].iloc[:3].tolist() == [4, 4, 0]
        assert pd.isna(result["topic_0"].iloc[3])
        assert result["rank"].iloc[:3].tolist() == [10, 10, 20]
        assert not any(col.startswith("_teal_with_position") for col in result.columns)

        filtered = left.query(
            with_artifacts=right, with_columns="topic_0",
            where="rank = 20", form="table",
        )
        assert filtered["doc_id"].tolist() == [2]
        assert filtered["topic_0"].tolist() == [7]
        if caller_type != "table":
            native = left.query(with_artifacts=right, with_columns="topic_0")
            assert native["info"]["topic_0"].iloc[:3].tolist() == [0, 0, 7]
            assert native["matrix"].shape == (4, 2)
    finally:
        project.close()


def test_selected_matrix_features_respect_projection_and_batching(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="matrix_features")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3], "text": ["a", "b", "c"]}),
            primary_key="doc_id", data_fields="text",
        )
        matrix = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": np.asarray([[0, 1, 2], [3, 4, 5]]),
                     "columns": ["a", "b", "c"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        batches = list(left.query(
            with_artifacts=matrix, with_columns=["b"],
            form="table", iter_batches=True, batch_size=2,
            streaming_mode="arrow",
        ))
        assert [len(b) for b in batches] == [2, 1]
        assert [c for c in batches[0] if c in {"a", "b", "c"}] == ["b"]
        assert batches[0]["b"].tolist() == [1, 4]
        assert pd.isna(batches[1]["b"].iloc[0])
        hidden = left.query(with_artifacts=matrix, with_columns=False, form="table")
        assert not any(c in hidden for c in ["a", "b", "c"])
        with pytest.raises(QueryError):
            left.query(with_artifacts=matrix, where="b > 1", form="table")
    finally:
        project.close()



@pytest.mark.parametrize("representation", ["dense_matrix", "sparse_matrix"])
@pytest.mark.parametrize("has_missing", [False, True])
def test_attached_matrix_large_integer_precision(tmp_path, representation, has_missing):
    """Joining integer matrix features must not round values through float64."""
    project = teal.Project.create(tmp_path / "integer_values", name="matrix_ints")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3] if has_missing else [1, 2]}),
            primary_key="doc_id",
        )
        large_values = np.asarray(
            [[2**53 + 1, 0], [2**53 + 3, 7]], dtype="int64"
        )
        payload = (
            sparse.csr_matrix(large_values)
            if representation == "sparse_matrix" else large_values
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": payload, "columns": ["big_count", "zero_count"]},
        }], artifact_type=representation, primary_key="doc_id")
        result = left.query(
            with_artifacts=right,
            with_columns=["big_count", "zero_count"],
            form="table",
        )
        assert int(result["big_count"].iloc[0]) == 2**53 + 1
        assert int(result["big_count"].iloc[1]) == 2**53 + 3
        assert result["zero_count"].iloc[0] == 0
        if has_missing:
            assert pd.isna(result["big_count"].iloc[2])
            assert pd.isna(result["zero_count"].iloc[2])
            assert str(result["big_count"].dtype) == "Int64"
        else:
            assert result["big_count"].dtype == np.dtype("int64")
    finally:
        project.close()
