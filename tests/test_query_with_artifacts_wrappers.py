"""Acceptance for query-time composition through curated artifact wrappers."""
from __future__ import annotations

import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError


@pytest.fixture
def composed(tmp_path):
    project = teal.Project.create(tmp_path / "wrapper-project", name="wrappers")
    left = project.register_external(
        pd.DataFrame({
            "doc_id": [1, 1, 2, 3],
            "sentence_id": [0, 1, 0, 0],
            "text": ["a", "b", "c", "d"],
        }),
        primary_key=["doc_id", "sentence_id"], data_fields="text",
    )
    right = project.register_external([{
        "keys": pd.DataFrame({"doc_id": [1, 2]}),
        "metadata": pd.DataFrame({"rank": [10, 20]}),
        "data": {
            "values": sparse.csr_matrix([[0, 4], [7, 0]]),
            "columns": ["topic_0", "topic_1"],
        },
    }], artifact_type="sparse_matrix", primary_key="doc_id")
    try:
        yield project, left, right
    finally:
        project.close()


def test_materialized_wrappers_match_query_and_keep_unmatched_null(composed):
    _, left, right = composed
    kw = dict(with_artifacts=right, with_columns=["topic_0"])
    expected = left.query(**kw, form="table")
    pd.testing.assert_frame_equal(left.table(**kw), expected)
    assert expected["topic_0"].iloc[:3].tolist() == [0, 0, 7]
    assert pd.isna(expected["topic_0"].iloc[3])
    assert left.records(**kw) == left.query(**kw, form="records")
    pd.testing.assert_frame_equal(
        left.preview(2, **kw), left.query(**kw, form="table", limit=2)
    )
    pd.testing.assert_frame_equal(
        left.table(with_artifacts=right, with_columns=False),
        left.query(with_artifacts=right, with_columns=False, form="table"),
    )


@pytest.mark.parametrize("form", ["table", "records", "native"])
def test_sample_wrapper_matches_canonical_query(composed, form):
    _, left, right = composed
    kw = dict(with_artifacts=right, with_columns="topic_0")
    result = left.sample(2, random_state=17, form=form, **kw)
    expected = left.query(
        sample_n=2, random_state=17, form=form, **kw
    )
    if form == "table":
        pd.testing.assert_frame_equal(result, expected)
    else:
        assert repr(result) == repr(expected)


@pytest.mark.parametrize("method,form,shape", [
    ("iter_batches", "table", "table"),
    ("iter_table_batches", "table", "table"),
    ("iter_records", "records", "records"),
    ("iter_rows", "single", "single"),
])
def test_streamed_wrappers_match_canonical_batches(composed, method, form, shape):
    _, left, right = composed
    kw = dict(
        with_artifacts=right, with_columns="topic_0",
        where="rank >= 10", streaming_mode="arrow",
    )
    if method == "iter_rows":
        actual = list(left.iter_rows(**kw))
        expected = list(left.query(form="single", iter_batches=True, batch_size=1, **kw))
    else:
        actual = list(getattr(left, method)(batch_size=2, **kw))
        expected = list(left.query(form=form, iter_batches=True, batch_size=2, **kw))
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if shape == "table":
            pd.testing.assert_frame_equal(a, b)
        else:
            assert a == b


def test_wrapper_hidden_projection_and_matrix_sql_guard(composed):
    _, left, right = composed
    batches = list(left.iter_table_batches(
        batch_size=1, with_artifacts=right, with_columns=False,
        where="rank = 10",
    ))
    assert sum(len(batch) for batch in batches) == 2
    assert all("topic_0" not in batch for batch in batches)
    with pytest.raises(QueryError):
        list(left.iter_table_batches(
            batch_size=1, with_artifacts=right, where="topic_0 > 0",
        ))


def test_wrapper_baselines_and_no_persistent_join(composed):
    project, left, right = composed
    before = {a["artifact_id"] for a in project.list_artifacts()}
    pd.testing.assert_frame_equal(left.table(), left.query(form="table"))
    assert left.records() == left.query(form="records")
    _ = left.table(with_artifacts=right, with_columns="topic_0")
    _ = list(left.iter_rows(with_artifacts=right, with_columns="topic_0"))
    assert {a["artifact_id"] for a in project.list_artifacts()} == before
