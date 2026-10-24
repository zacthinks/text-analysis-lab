"""Focused contract tests for operation-local QueryScope."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.operator import ColumnRequest
from text_analysis_lab.core.query_scope import QueryScope


@pytest.fixture
def source(tmp_path):
    project = teal.Project.create(tmp_path / "scope_project", name="scope_project")
    try:
        artifact = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"score": [11, 22], "text": ["a", "b"]}),
            "metadata": pd.DataFrame({"score": [101, 202], "group": ["x", "y"]}),
        }], artifact_type="table", primary_key="doc_id")
        yield artifact
    finally:
        project.close()


def test_scope_discovery_and_multiple_projections_share_labels(source):
    scope = QueryScope(source, include_data=True, metadata_mode="full")
    inventory = scope.columns()
    data_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "data" and c["base_name"] == "score"
    )
    metadata_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "metadata" and c["base_name"] == "score"
    )
    assert data_score != metadata_score
    data = scope.query(
        key_columns=False, data_columns=[data_score], metadata_columns=False,
        form="table",
    )
    metadata = scope.query(
        key_columns=False, data_columns=False,
        metadata_columns=[metadata_score], form="table",
    )
    assert data[data_score].tolist() == [11, 22]
    assert metadata[metadata_score].tolist() == [101, 202]


def test_scope_generates_source_request_with_same_naming_scope(source):
    scope = QueryScope(source, include_data=True, metadata_mode="full")
    inventory = scope.columns()
    data_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "data" and c["base_name"] == "score"
    )
    request = scope.source_request(
        columns=ColumnRequest(keys=True, data=[data_score], metadata=False),
        mode="batches", batch_size=2, form="table",
    )
    assert request.metadata_mode == "full"
    assert request.include_data is True
    assert request.columns.data == [data_score]


def test_scope_rejects_scope_overrides(source):
    scope = QueryScope(source, metadata_mode="full")
    with pytest.raises(TypeError, match="naming-scope"):
        scope.query(metadata_mode="none", data_columns=False)
    with pytest.raises(TypeError, match="naming-scope"):
        scope.source_request(
            columns=ColumnRequest(data=False),
            mode="full_artifact",
            metadata_mode="none",
        )


def test_scope_can_exclude_data_without_changing_key_names(source):
    scope = QueryScope(source, include_data=False, metadata_mode="none")
    assert scope.columns()["key"] == ["doc_id"]
    frame = scope.query(
        key_columns=True, data_columns=False, metadata_columns=False,
        form="table",
    )
    assert frame["doc_id"].tolist() == [1, 2]
