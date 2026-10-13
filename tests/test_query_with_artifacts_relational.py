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


def test_actual_alias_lookup_and_invalid_reference_shapes(artifacts):
    project, left, right = artifacts
    project.add_artifact_alias(right, "topic_scores")
    result = left.query(
        with_artifacts=["topic_scores", right.artifact_id, right],
        with_columns="topic", form="table",
    )
    assert result["topic"].iloc[:3].tolist() == [5, 5, 9]
    assert list(result.columns).count("topic") == 1
    for bad in (42, [right, 42], {"topic": right}, [None]):
        with pytest.raises(QueryError, match="with_artifacts"):
            left.query(with_artifacts=bad)
    with pytest.raises(Exception):
        left.query(with_artifacts="missing-topic-alias")


def test_self_attachment_and_local_metadata(tmp_path):
    project = teal.Project.create(tmp_path / "self", name="self_attachment")
    try:
        source = project.register_external(
            [{
                "keys": pd.DataFrame({"doc_id": [0, 1, 2]}),
                "data": pd.DataFrame({"score": [2, 3, 4]}),
                "metadata": pd.DataFrame({"group": ["a", "b", "a"]}),
            }],
            artifact_type="table",
            primary_key="doc_id",
        )
        info = source.query_columns(with_artifacts=source, metadata_mode="none")
        assert info["with"] == ["group"]
        assert info["data"] == ["score"]
        frame = source.query(
            with_artifacts=source, with_columns=True, where='"group" = \'a\'',
            metadata_mode="none", form="table",
        )
        assert frame["doc_id"].tolist() == [0, 2]
        assert frame["score"].tolist() == [2, 4]
        assert frame["group"].tolist() == ["a", "a"]
        full = source.query_columns(with_artifacts=source, metadata_mode="full")
        assert full["with"] == []
        assert full["metadata"] == ["group"]
    finally:
        project.close()


def test_metadata_attached_and_inherited_dedup(tmp_path):
    project = teal.Project.create(tmp_path / "lineage", name="metadata")
    try:
        parent = project.register_external(
            [{
                "keys": pd.DataFrame({"doc_id": [1, 2]}),
                "data": pd.DataFrame({"score": [10, 20]}),
                "metadata": pd.DataFrame({"site": ["north", "south"]}),
            }],
            artifact_type="table",
            primary_key="doc_id",
        )
        child = project.select_keys(parent, [1, 2], output_label="subset")
        info = child.query_columns(with_artifacts=parent, metadata_mode="full")
        assert info["with"] == []
        assert len([c for c in info["columns"] if c["base_name"] == "site"]) == 1
        local = child.query(
            with_artifacts=parent, with_columns="site", metadata_mode="none",
            where="site = 'north'", form="table",
        )
        assert local["doc_id"].tolist() == [1]
    finally:
        project.close()


def test_order_positions_limit_and_nonprefix_key_subset(tmp_path):
    project = teal.Project.create(tmp_path / "key_subset", name="nonprefix")
    try:
        left = project.register_external(
            pd.DataFrame({
                "doc_id": [1, 1, 2, 3], "sentence_id": [0, 1, 0, 0],
                "collection": [7, 8, 7, 8], "body": ["a", "b", "c", "d"],
            }),
            primary_key=["doc_id", "sentence_id", "collection"],
            data_fields="body",
        )
        right = project.register_external(
            pd.DataFrame({"collection": [8, 7], "rank": [1, 9]}),
            primary_key="collection", data_fields="rank",
        )
        result = left.query(
            with_artifacts=right, with_columns="rank",
            where="rank >= 1", order_by="rank DESC",
            positions=[3, 0, 2, 1], limit=2, form="table",
        )
        assert result["body"].tolist() == ["a", "c"]
        assert result["rank"].tolist() == [9, 9]
    finally:
        project.close()


@pytest.mark.parametrize("artifact_type", ["dense_matrix", "sparse_matrix"])
def test_matrix_caller_attaches_relational_data(tmp_path, artifact_type):
    import numpy as np
    from scipy import sparse

    project = teal.Project.create(tmp_path / artifact_type, name="matrix_left")
    try:
        values = np.asarray([[1, 0], [2, 3], [0, 4]], dtype="int64")
        if artifact_type == "sparse_matrix":
            values = sparse.csr_matrix(values)
        left = project.register_external(
            [{
                "keys": pd.DataFrame({"doc_id": [1, 1, 2], "sentence_id": [0, 1, 0]}),
                "data": {"values": values, "columns": ["f0", "f1"]},
            }],
            artifact_type=artifact_type,
            primary_key=["doc_id", "sentence_id"],
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "category": ["alpha", "beta"]}),
            primary_key="doc_id", data_fields="category",
        )
        native = left.query(
            with_artifacts=right, with_columns="category", form="native",
        )
        assert native["info"]["category"].tolist() == ["alpha", "alpha", "beta"]
        stored = native["matrix"]
        if sparse.issparse(stored):
            stored = stored.toarray()
        np.testing.assert_array_equal(stored, [[1, 0], [2, 3], [0, 4]])
        filtered = left.query(
            with_artifacts=right, with_columns=False, where="category = 'alpha'",
            data_columns=["f0"], form="table",
        )
        assert filtered["f0"].tolist() == [1, 2]
        assert "category" not in filtered
    finally:
        project.close()


def test_merge_caller_preserves_unknown_source_origin(tmp_path):
    project = teal.Project.create(tmp_path / "merge", name="merged_origin")
    try:
        branch_a = project.register_external(
            pd.DataFrame({"doc_id": [1], "text": ["a"]}),
            primary_key="doc_id", data_fields="text",
        )
        branch_b = project.register_external(
            pd.DataFrame({"doc_id": [2], "text": ["b"]}),
            primary_key="doc_id", data_fields="text",
        )
        merged = project.merge([branch_a, branch_b], output_label="merged")
        attached = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "topic": [4, 5]}),
            primary_key="doc_id", data_fields="topic",
        )
        info = merged.query_columns(with_artifacts=attached)
        text_field = next(
            c for c in info["columns"]
            if c["namespace"] == "data" and c["base_name"] == "text"
        )
        assert text_field["source_artifact_id"] is None
        result = merged.query(with_artifacts=attached, form="table")
        assert result["text"].tolist() == ["a", "b"]
        assert result["topic"].tolist() == [4, 5]
    finally:
        project.close()
