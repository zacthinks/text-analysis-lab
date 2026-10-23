"""Phase 1C final acceptance: JSONL, sampling, streaming, absence and provenance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import QueryError
from text_analysis_lab.core.artifact_subclasses import SparseMatrixArtifact


@pytest.mark.parametrize("left_type", ["table", "jsonl"])
@pytest.mark.parametrize("right_type", ["table", "jsonl"])
def test_relational_jsonl_combinations_and_native_forms(tmp_path, left_type, right_type):
    project = teal.Project.create(tmp_path / "jsonl", name="jsonl_compose")
    try:
        left_payload = {
            "keys": pd.DataFrame({"doc_id": [1, 1, 2, 3],
                                  "sentence_id": [0, 1, 0, 0]}),
            "data": pd.DataFrame({"text": ["a", "b", "c", "d"]})
                    if left_type == "table"
                    else [{"text": v} for v in ("a", "b", "c", "d")],
        }
        right_payload = {
            "keys": pd.DataFrame({"doc_id": [2, 1]}),
            "data": pd.DataFrame({"label": ["beta", "alpha"]})
                    if right_type == "table"
                    else [{"label": "beta"}, {"label": "alpha"}],
            "metadata": pd.DataFrame({"rank": [20, 10]}),
        }
        left = project.register_external(
            [left_payload], artifact_type=left_type,
            primary_key=["doc_id", "sentence_id"],
        )
        right = project.register_external(
            [right_payload], artifact_type=right_type, primary_key="doc_id",
        )
        base_native = left.query(form="native")
        table = left.query(with_artifacts=right, form="table")
        assert table["label"].iloc[:3].tolist() == ["alpha", "alpha", "beta"]
        assert pd.isna(table["label"].iloc[3])
        assert table["text"].tolist() == ["a", "b", "c", "d"]
        assert table["rank"].iloc[:3].tolist() == [10, 10, 20]
        assert isinstance(left.query(with_artifacts=right, form="native"),
                          type(base_native))
        records = left.query(with_artifacts=right, form="records")
        assert [row["label"] for row in records[:3]] == ["alpha", "alpha", "beta"]
        assert pd.isna(records[3]["label"])
        assert left.query(with_artifacts=right, positions=[2],
                          form="single")["label"] == "beta"
        filtered = left.query(with_artifacts=right, with_columns=False,
                              where="rank = 10", form="table")
        assert filtered["text"].tolist() == ["a", "b"]
        assert "rank" not in filtered
        assert "label" not in filtered
        chunks = list(left.query(with_artifacts=right, form="records",
                                 iter_batches=True, batch_size=2))
        assert [[row["doc_id"] for row in batch] for batch in chunks] == [
            [1, 1], [2, 3]
        ]
    finally:
        project.close()


@pytest.mark.parametrize("left_type", ["table", "jsonl"])
def test_jsonl_caller_with_sparse_attachment_and_local_metadata(tmp_path, left_type):
    project = teal.Project.create(tmp_path / "jsonl_matrix", name="jsonl_matrix")
    try:
        keys = pd.DataFrame({"doc_id": [1, 2, 3]})
        left = project.register_external([{
            "keys": keys,
            "data": pd.DataFrame({"text": ["a", "b", "c"]})
                    if left_type == "table"
                    else [{"text": "a"}, {"text": "b"}, {"text": "c"}],
        }], artifact_type=left_type, primary_key="doc_id")
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "metadata": pd.DataFrame({"rank": [10, 20]}),
            "data": {"values": sparse.csr_matrix([[0], [7]]),
                     "columns": ["topic"]},
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        info = left.query(with_artifacts=right, with_columns="topic", form="table")
        assert info["topic"].iloc[:2].tolist() == [0, 7]
        assert pd.isna(info["topic"].iloc[2])
        filtered = left.query(with_artifacts=right, with_columns="topic",
                              where="rank = 20", form="records")
        assert len(filtered) == 1
        assert filtered[0]["topic"] == 7
        batches = list(left.query(with_artifacts=right, with_columns="topic",
                                  form="table", iter_batches=True,
                                  streaming_mode="arrow", batch_size=1))
        assert len(batches) == 3
        assert [v for batch in batches for v in batch["topic"].iloc[:1]
                if not pd.isna(v)] == [0, 7]
    finally:
        project.close()


def test_filtered_sampling_attached_metadata_matches_final_selection(tmp_path):
    project = teal.Project.create(tmp_path / "sample", name="sample_compose")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3, 4], "text": list("abcd")}),
            primary_key="doc_id", data_fields="text",
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [3, 1, 2]}),
            "metadata": pd.DataFrame({"rank": [30, 10, 20]}),
            "data": {"values": np.asarray([[9], [0], [4]]),
                     "columns": ["topic"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        kwargs = dict(with_artifacts=right, with_columns="topic",
                      where="rank >= 20", sample_n=2, random_state=17,
                      form="table")
        first = left.query(**kwargs)
        again = left.query(**kwargs)
        assert set(first["doc_id"]) == {2, 3}
        assert dict(zip(first["doc_id"], first["topic"])) == {2: 4, 3: 9}
        pd.testing.assert_frame_equal(first, again)
        selected = left.query(with_artifacts=right, with_columns="topic",
                              where="rank >= 20", sample_frac=1.0,
                              random_state=5, form="table")
        assert set(selected["doc_id"]) == {2, 3}
    finally:
        project.close()


def test_entirely_unmatched_stream_batch_and_mixed_match_values(tmp_path):
    project = teal.Project.create(tmp_path / "unmatched", name="unmatched")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3, 4]}), primary_key="doc_id"
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [3, 5]}),
            "data": {"values": sparse.csr_matrix(np.asarray([[0], [8]])),
                     "columns": ["topic"]},
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        batches = list(left.query(with_artifacts=right, with_columns="topic",
                                  form="table", iter_batches=True,
                                  batch_size=2, streaming_mode="arrow"))
        assert [len(batch) for batch in batches] == [2, 2]
        assert all(pd.isna(x) for x in batches[0]["topic"])
        assert batches[1]["topic"].iloc[0] == 0
        assert pd.isna(batches[1]["topic"].iloc[1])
        assert all(not str(c).startswith("_teal_with_position")
                   for batch in batches for c in batch.columns)
        one_missing = left.query(with_artifacts=right, with_columns="topic",
                                 positions=[0], form="single")
        assert pd.isna(one_missing["topic"])
    finally:
        project.close()


def test_failed_query_cleanup_and_ephemeral_reopen(tmp_path):
    path = tmp_path / "durable"
    project = teal.Project.create(path, name="ephemeral")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "text": ["a", "b"]}),
            primary_key="doc_id", data_fields="text",
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1]}),
            "data": {"values": sparse.csr_matrix([[4]]),
                     "columns": ["topic"]},
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        ids_before = {record["artifact_id"] for record in project.list_artifacts()}
        before = left.query(form="table")
        with pytest.raises(QueryError):
            left.query(with_artifacts=right, with_columns="topic",
                       where="this_column_does_not_exist = 0",
                       positions=[0, 1], form="table")
        result = left.query(with_artifacts=right, with_columns="topic",
                            positions=[0, 1], form="table")
        assert result["topic"].iloc[0] == 4
        assert pd.isna(result["topic"].iloc[1])
        pd.testing.assert_frame_equal(before, left.query(form="table"))
        assert {record["artifact_id"] for record in project.list_artifacts()} == ids_before
        left_id, right_id = left.artifact_id, right.artifact_id
    finally:
        project.close()
    reopened = teal.Project.open(path)
    try:
        after = reopened.get_artifact(left_id).query(
            with_artifacts=reopened.get_artifact(right_id),
            with_columns="topic", form="table",
        )
        assert after["topic"].iloc[0] == 4
        assert pd.isna(after["topic"].iloc[1])
        assert {rec["artifact_id"] for rec in reopened.list_artifacts()} == ids_before
    finally:
        reopened.close()


def test_requested_width_is_deferred_and_narrow(monkeypatch, tmp_path):
    """Large feature registries must only load selected right matrix columns."""
    project = teal.Project.create(tmp_path / "wide", name="wide")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2]}), primary_key="doc_id"
        )
        names = [f"f{i}" for i in range(128)]
        values = np.arange(256).reshape(2, 128)
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": sparse.csr_matrix(values),
                     "columns": names},
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        seen = []
        original = SparseMatrixArtifact._data_native_for_positions

        def spy(self, positions, *, data_columns=True):
            if self.artifact_id == right.artifact_id:
                seen.append(data_columns)
            return original(self, positions, data_columns=data_columns)

        monkeypatch.setattr(SparseMatrixArtifact, "_data_native_for_positions", spy)
        frame = left.query(with_artifacts=right, with_columns="f127", form="table")
        assert frame["f127"].tolist() == [127, 255]
        assert "f0" not in frame.columns
        assert seen == [["f127"]]
    finally:
        project.close()


def test_filtered_sampling_can_filter_on_projected_out_collision_field(tmp_path):
    project = teal.Project.create(tmp_path / "sample_hidden", name="sample_hidden")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2, 3, 4]}),
            "data": pd.DataFrame({
                "text": ["A", "B", "C", "D"],
                "data.text": ["skip", "keep", "keep", "skip"],
            }),
        }], artifact_type="table", primary_key="doc_id")
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3, 4],
                          "text": ["right1", "right2", "right3", "right4"]}),
            primary_key="doc_id", data_fields="text",
        )
        inventory = source.query_columns(metadata_mode="none", with_artifacts=right)
        hidden = next(col["output_name"] for col in inventory["columns"]
                      if col["namespace"] == "data"
                      and col["base_name"] == "data.text")
        attached = next(col["output_name"] for col in inventory["columns"]
                        if col["namespace"] == "with"
                        and col["base_name"] == "text")
        arguments = dict(
            metadata_mode="none", data_columns=False,
            with_artifacts=right, with_columns=[attached],
            where=f'"{hidden}" = \'keep\'', form="table",
        )
        full = source.query(**arguments)
        assert full["doc_id"].tolist() == [2, 3]
        assert full[attached].tolist() == ["right2", "right3"]
        for sampling in ({"sample_n": 2}, {"sample_frac": 1.0}):
            sampled = source.query(**arguments, **sampling, random_state=19)
            assert set(sampled["doc_id"]) == {2, 3}
            assert dict(zip(sampled["doc_id"], sampled[attached], strict=True)) == {
                2: "right2", 3: "right3",
            }
            assert hidden not in sampled.columns
    finally:
        project.close()


