"""Phase 1A: explicit relational field attachments are ephemeral keyed left joins."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError


@pytest.fixture
def artifacts(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="attached")
    left = project.register_external(
        pd.DataFrame({
            "doc_id": [1, 1, 2, 3],
            "sentence_id": [0, 1, 0, 0],
            "text": ["a", "b", "c", "d"],
        }),
        primary_key=["doc_id", "sentence_id"],
        data_fields="text",
    )
    right = project.register_external(
        pd.DataFrame({"doc_id": [2, 1, 4], "topic": [9, 5, 0]}),
        primary_key="doc_id",
        data_fields="topic",
    )
    try:
        yield project, left, right
    finally:
        project.close()


def test_singleton_and_nullable_coarse_join(artifacts):
    _, left, right = artifacts
    frame = left.query(with_artifacts=right, with_columns="topic", form="table")
    assert frame["doc_id"].tolist() == [1, 1, 2, 3]
    assert frame["topic"].iloc[:3].tolist() == [5, 5, 9]
    assert pd.isna(frame["topic"].iloc[3])
    discovered = left.query_columns(with_artifacts=right)
    assert "topic" in discovered["with"]


def test_attached_relational_where_and_hidden_projection(artifacts):
    _, left, right = artifacts
    frame = left.query(
        with_artifacts=right, with_columns=False,
        where="topic = 5", form="table",
    )
    assert frame["text"].tolist() == ["a", "b"]
    assert "topic" not in frame


def test_attached_relational_filtered_sampling(artifacts):
    _, left, right = artifacts
    frame = left.query(
        with_artifacts=right, with_columns=["topic"],
        where="topic = 5", sample_n=1, random_state=42, form="table",
    )
    assert len(frame) == 1
    assert frame["topic"].iloc[0] == 5


def test_ref_alias_duplicates_and_missing_keys(artifacts):
    project, left, right = artifacts
    frame = left.query(with_artifacts=[right, right.artifact_id], with_columns=True, form="table")
    assert list(frame.columns).count("topic") == 1
    wrong = project.register_external(
        pd.DataFrame({"alien_id": [1], "flag": [True]}),
        primary_key="alien_id", data_fields="flag",
    )
    with pytest.raises(QueryError, match="incompatible primary key"):
        left.query(with_artifacts=wrong)


def test_collision_and_qualification(artifacts):
    project, left, right = artifacts
    other = project.register_external(
        pd.DataFrame({"doc_id": [1, 2, 3], "topic": [8, 7, 6]}),
        primary_key="doc_id", data_fields="topic",
    )
    info = left.query_columns(with_artifacts=[right, other])
    assert len(info["with"]) == 2
    assert len(set(info["with"])) == 2
    qualified = [c["qualified_name"] for c in info["columns"] if c["namespace"] == "with"]
    assert all(q.startswith("with.") and q.endswith(".data.topic") for q in qualified)
    result = left.query(with_artifacts=[right, other], with_columns=qualified, form="table")
    assert len(result) == 4
    assert len([x for x in result.columns if x.endswith("topic")]) == 2


def test_no_with_queries_remain_unchanged(artifacts):
    _, left, right = artifacts
    baseline = left.query(form="table")
    same = left.query(with_artifacts=[], form="table")
    pd.testing.assert_frame_equal(baseline, same)
