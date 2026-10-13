"""Phase 2A: query-time attached fields in scatter color/hover."""
from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("plotly")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError


def test_scatter_attached_topic_color_and_hover_with_key_subset(tmp_path):
    project = teal.Project.create(tmp_path / "scatter", name="attached_scatter")
    try:
        sentences = project.register_external(
            pd.DataFrame({
                "doc_id": [1, 1, 2, 3],
                "sentence_id": [0, 1, 0, 0],
                "text": ["first", "second", "third", "fourth"],
            }),
            primary_key=["doc_id", "sentence_id"],
            data_fields="text",
        )
        coords = project.register_external([{
            "keys": pd.DataFrame({
                "doc_id": [2, 1, 3, 1],
                "sentence_id": [0, 1, 0, 0],
            }),
            "data": {
                "values": np.asarray([[2., 20.], [1., 11.],
                                      [3., 30.], [0., 10.]]),
                "columns": ["u0", "u1"],
            },
        }], artifact_type="dense_matrix",
            primary_key=["doc_id", "sentence_id"])
        topics = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "topic": ["a", "b"],
                          "rank": [10, 20]}),
            primary_key="doc_id", data_fields=["topic", "rank"],
        )
        fig = sentences.visualize.scatter(
            embeddings=coords, with_artifacts=topics,
            with_columns=["topic", "rank"], color="topic",
            hover_fields=["text", "rank"],
            data_columns="text", where="rank >= 10",
            height=400,
        )
        assert sum(len(trace.x) for trace in fig.data) == 3
        actual = sorted(
            (float(x), float(y))
            for trace in fig.data
            for x, y in zip(trace.x, trace.y, strict=True)
        )
        assert actual == [(0., 10.), (1., 11.), (2., 20.)]
        assert set(trace.name for trace in fig.data) == {"a", "b"}
        with pytest.raises(ArtifactError, match="not selected"):
            sentences.visualize.scatter(
                embeddings=coords, with_artifacts=topics,
                with_columns=False, color="topic",
            )
    finally:
        project.close()


def test_scatter_attached_matrix_feature_as_color(tmp_path):
    project = teal.Project.create(tmp_path / "matrix_scatter", name="matrix_scatter")
    try:
        docs = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3], "text": ["a", "b", "c"]}),
            primary_key="doc_id", data_fields="text",
        )
        coords = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2, 3]}),
            "data": {"values": np.asarray([[0., 1.], [2., 3.], [4., 5.]]),
                     "columns": ["x", "y"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        scores = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": np.asarray([[0.0], [0.8]]),
                     "columns": ["topic_0"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        figure = docs.visualize.scatter(
            embeddings=coords, with_artifacts=scores,
            with_columns="topic_0", color="topic_0",
            hover_fields="text", data_columns="text",
        )
        assert sum(len(trace.x) for trace in figure.data) == 3
        assert sorted(float(x) for trace in figure.data for x in trace.x) == [0., 2., 4.]
    finally:
        project.close()


def test_missing_attached_categorical_color_label_keeps_unmatched_points(tmp_path):
    project = teal.Project.create(tmp_path / "missing_color", name="missing_color")
    try:
        docs = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3], "text": ["a", "b", "c"]}),
            primary_key="doc_id", data_fields="text",
        )
        coords = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2, 3]}),
            "data": {"values": np.asarray([[0., 1.], [2., 3.], [4., 5.]]),
                     "columns": ["x", "y"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        groups = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "category": ["A", "B"]}),
            primary_key="doc_id", data_fields="category",
        )
        base = docs.visualize.scatter(
            embeddings=coords, with_artifacts=groups,
            with_columns="category", color="category",
        )
        assert sum(len(trace.x) for trace in base.data) == 2
        filled = docs.visualize.scatter(
            embeddings=coords, with_artifacts=groups,
            with_columns="category", color="category",
            color_na_label="Other",
        )
        assert sum(len(trace.x) for trace in filled.data) == 3
        assert {trace.name for trace in filled.data} == {"A", "B", "Other"}
        with pytest.raises(ValueError, match="collides"):
            docs.visualize.scatter(
                embeddings=coords, with_artifacts=groups,
                with_columns="category", color="category",
                color_na_label="A",
            )
    finally:
        project.close()


def test_missing_color_label_rejects_numeric_color(tmp_path):
    project = teal.Project.create(tmp_path / "numeric_color", name="numeric_color")
    try:
        docs = project.register_external(
            pd.DataFrame({"doc_id": [1, 2]}), primary_key="doc_id",
        )
        coords = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": np.asarray([[0., 1.], [2., 3.]]),
                     "columns": ["x", "y"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        weights = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1]}),
            "data": {"values": np.asarray([[0.25]]),
                     "columns": ["weight"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        with pytest.raises(ValueError, match="categorical"):
            docs.visualize.scatter(
                embeddings=coords, with_artifacts=weights,
                with_columns="weight", color="weight",
                color_na_label="Other",
            )
    finally:
        project.close()
