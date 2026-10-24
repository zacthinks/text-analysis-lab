"""Persisted key/metadata collision regressions for PR 57 consumers."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal


def _source(project):
    return project.register_external([{
        "keys": pd.DataFrame({
            "doc_id": [1, 1, 2, 2],
            "turn_id": [0, 1, 0, 1],
        }),
        "data": pd.DataFrame({"score": [10, 20, 30, 40]}),
        "metadata": pd.DataFrame({
            "doc_id": [100, 101, 200, 201],
            "speaker": ["A", "A", "B", "B"],
        }),
    }], artifact_type="table", primary_key=["doc_id", "turn_id"])


def _key_label(artifact, name):
    return next(
        item["output_name"]
        for item in artifact.query_columns(metadata_mode="full")["columns"]
        if item["namespace"] == "key" and item["base_name"] == name
    )


def test_aggregate_uses_canonical_key_labels_but_preserves_output_keys(tmp_path):
    project = teal.Project.create(tmp_path / "agg", name="agg_key_collision")
    try:
        source = _source(project)
        assert _key_label(source, "doc_id") != "doc_id"
        result = project.aggregate(
            source, to_key="doc_id", data={"score_sum": teal.agg("score", "sum")}
        )
        frame = result.query(data_columns=True, form="table")
        assert frame["doc_id"].tolist() == [1, 2]
        assert frame["score_sum"].tolist() == [30, 70]
    finally:
        project.close()


def test_collapse_runs_uses_canonical_keys_in_window_and_join(tmp_path):
    project = teal.Project.create(tmp_path / "collapse", name="collapse_key_collision")
    try:
        source = _source(project)
        result = project.collapse_runs(source, by="speaker")
        frame = result.query(
            data_columns=False, metadata_columns=True,
            metadata_mode="local", form="table"
        )
        assert frame["doc_id"].tolist() == [1, 2]
        assert frame["turn_id_start"].tolist() == [0, 0]
        assert frame["turn_id_end"].tolist() == [1, 1]
        assert frame["speaker"].tolist() == ["A", "B"]
        assert frame["n_rows"].tolist() == [2, 2]
    finally:
        project.close()


def test_split_materializes_colliding_key_under_full_scope(tmp_path):
    project = teal.Project.create(tmp_path / "split", name="split_key_collision")
    try:
        source = _source(project)
        children = project.split(
            source, labels=("train", "test"),
            proportions=(0.5, 0.5), random_state=7,
            stratify="speaker",
        )
        keys = []
        for child in children.values():
            frame = child.query(data_columns=False, form="table")
            keys.extend(map(tuple, frame[["doc_id", "turn_id"]].itertuples(index=False, name=None)))
        assert sorted(keys) == [(1, 0), (1, 1), (2, 0), (2, 1)]
    finally:
        project.close()
