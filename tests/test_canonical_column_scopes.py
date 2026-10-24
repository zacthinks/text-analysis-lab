"""Query naming scope is independent of projections, and keys never rename."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError


@pytest.fixture
def corpus(tmp_path):
    project = teal.Project.create(tmp_path / "scoped_names", name="scoped_names")
    try:
        artifact = project.register_external(
            [{
                "keys": pd.DataFrame({"doc_id": [1, 2]}),
                "data": pd.DataFrame({
                    "doc_id": ["datum1", "datum2"],
                    "text": ["alpha", "beta"],
                }),
                "metadata": pd.DataFrame({
                    "doc_id": [101, 102],
                    "group": ["A", "B"],
                }),
            }],
            artifact_type="table",
            primary_key="doc_id",
        )
        yield artifact
    finally:
        project.close()


@pytest.mark.parametrize("include_data", [False, True])
@pytest.mark.parametrize("metadata_mode", ["none", "local", "full"])
def test_immutable_keys_and_projected_output_names(corpus, include_data, metadata_mode):
    inventory = corpus.query_columns(
        include_data=include_data, metadata_mode=metadata_mode
    )
    assert inventory["key"] == ["doc_id"]
    assert next(c for c in inventory["columns"] if c["namespace"] == "key")["output_name"] == "doc_id"

    frame = corpus.query(
        include_data=include_data, metadata_mode=metadata_mode,
        key_columns=True, data_columns=False, metadata_columns=False,
        form="table",
    )
    assert frame.columns.tolist() == ["doc_id"]
    assert frame["doc_id"].tolist() == [1, 2]

    if include_data:
        data_label = next(
            c["output_name"] for c in inventory["columns"]
            if c["namespace"] == "data" and c["base_name"] == "doc_id"
        )
        assert data_label != "doc_id"
        result = corpus.query(
            include_data=include_data, metadata_mode=metadata_mode,
            key_columns=True, data_columns=[data_label], metadata_columns=False,
            form="table",
        )
        assert result["doc_id"].tolist() == [1, 2]
        assert result[data_label].tolist() == ["datum1", "datum2"]

    if metadata_mode != "none":
        metadata_label = next(
            c["output_name"] for c in inventory["columns"]
            if c["namespace"] == "metadata" and c["base_name"] == "doc_id"
        )
        assert metadata_label != "doc_id"
        result = corpus.query(
            include_data=include_data, metadata_mode=metadata_mode,
            key_columns=False, data_columns=False,
            metadata_columns=[metadata_label], form="table",
        )
        assert result[metadata_label].tolist() == [101, 102]


def test_data_is_scoped_before_projection(corpus):
    with pytest.raises(QueryError, match="include_data=False"):
        corpus.query(include_data=False, data_columns=True)


def test_query_scope_filtering_and_sampling(corpus):
    frame = corpus.query(
        include_data=True, data_columns=False, metadata_columns=False,
        where='"data.doc_id" = \'datum2\'', sample_n=1,
        form="table",
    )
    assert frame["doc_id"].tolist() == [2]


def test_keys_only_scope_ignores_data_collision(corpus):
    inventory = corpus.query_columns(include_data=False, metadata_mode="none")
    assert inventory["output"] == ["doc_id"]
    frame = corpus.query(
        include_data=False, key_columns=True, data_columns=False,
        metadata_columns=False, form="table",
    )
    assert frame["doc_id"].tolist() == [1, 2]
