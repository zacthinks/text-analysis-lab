from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab import analysis
from text_analysis_lab.core.errors import (
    ArtifactError,
    UnsupportedArtifactOperationError,
)
from text_analysis_lab.core.types import ArtifactType


class _MatrixFixture:
    artifact_id = "art_unit4"
    label = "dtm"
    status = "complete"
    n_rows = 4
    primary_key = ("doc_id",)

    def __init__(self, *, sparse_matrix: bool = False) -> None:
        values = np.array(
            [
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.matrix = sparse.csr_matrix(values) if sparse_matrix else values
        self.artifact_type = (
            ArtifactType.SPARSE_MATRIX if sparse_matrix else ArtifactType.DENSE_MATRIX
        )
        self.seen_batch_sizes: list[int] = []

    def get_data_columns(self):
        return ["apple", "banana", "cherry"]

    def position_by_key(self, key):
        if isinstance(key, dict):
            return int(key["doc_id"])
        return int(key)

    def get_matrix(self, *, positions=None, data_columns=True):
        _ = data_columns
        if positions is None:
            return self.matrix
        return self.matrix[[int(value) for value in positions], :]

    def iter_batches(self, *, batch_size, key_columns, include_position, **kwargs):
        _ = kwargs
        self.seen_batch_sizes.append(int(batch_size))
        for start in range(0, self.n_rows, 2):
            stop = min(self.n_rows, start + 2)
            positions = list(range(start, stop))
            info_data: dict[str, object] = {}
            if key_columns:
                info_data["doc_id"] = positions
            if include_position:
                info_data["_position"] = positions
            yield {
                "info": pd.DataFrame(info_data, index=range(len(positions))),
                "matrix": self.matrix[positions, :],
            }

    def query(self, *, positions, **kwargs):
        _ = kwargs
        ordered = sorted(int(value) for value in positions)
        return pd.DataFrame({"doc_id": ordered, "_position": ordered})


def test_cosine_similarity_and_distance_support_dense_and_sparse() -> None:
    for sparse_matrix in (False, True):
        artifact = _MatrixFixture(sparse_matrix=sparse_matrix)
        similarity = analysis.cosine_similarity(artifact, key=0, other_key=2)
        assert similarity == pytest.approx(2.0 / math.sqrt(10.0))
        assert analysis.distance(
            artifact, position=0, other_position=2, metric="cosine"
        ) == pytest.approx(1.0 - similarity)
        assert analysis.distance(
            artifact, position=0, other_position=1, metric="euclidean"
        ) == pytest.approx(math.sqrt(5.0))
        assert analysis.distance(
            artifact, position=0, other_position=1, metric="manhattan"
        ) == pytest.approx(3.0)


def test_distance_locator_contract_is_explicit() -> None:
    artifact = _MatrixFixture()
    with pytest.raises(ValueError):
        analysis.distance(artifact, other_position=1)
    with pytest.raises(ValueError):
        analysis.distance(artifact, key=0, position=0, other_position=1)
    with pytest.raises(ValueError):
        analysis.distance(artifact, position=0)


def test_nearest_neighbors_reuses_generic_metrics_and_stays_bounded() -> None:
    artifact = _MatrixFixture(sparse_matrix=True)
    result = analysis.nearest_neighbors(
        artifact,
        position=0,
        metric="euclidean",
        k=2,
        batch_size=2,
    )
    assert result["doc_id"].tolist() == [2, 1]
    assert result["distance"].tolist() == pytest.approx(
        [math.sqrt(3.0), math.sqrt(5.0)]
    )
    # The method must stream source batches and retain only k candidates.
    assert artifact.seen_batch_sizes == [2]


def test_row_summary_is_keyed_and_batch_bounded() -> None:
    artifact = _MatrixFixture(sparse_matrix=True)
    result = analysis.row_summary(artifact, batch_size=2)
    assert result["doc_id"].tolist() == [0, 1, 2, 3]
    assert result["_position"].tolist() == [0, 1, 2, 3]
    assert result["nonzero_features"].tolist() == [2, 2, 2, 0]
    assert result["sum"].tolist() == pytest.approx([3.0, 2.0, 2.0, 0.0])
    assert result["l2_norm"].tolist() == pytest.approx(
        [math.sqrt(5.0), math.sqrt(2.0), math.sqrt(2.0), 0.0]
    )
    assert result.loc[3, "min"] == 0.0
    assert result.loc[3, "max"] == 0.0
    assert artifact.seen_batch_sizes == [2]


def test_feature_summary_is_representation_agnostic() -> None:
    artifact = _MatrixFixture(sparse_matrix=True)
    result = analysis.feature_summary(artifact, batch_size=2)
    assert result["feature"].tolist() == ["apple", "banana", "cherry"]
    assert result["feature_index"].tolist() == [0, 1, 2]
    assert result["nonzero_rows"].tolist() == [2, 2, 2]
    assert result["sum"].tolist() == pytest.approx([2.0, 2.0, 3.0])
    assert result["mean"].tolist() == pytest.approx([0.5, 0.5, 0.75])
    assert result["min"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result["max"].tolist() == pytest.approx([1.0, 1.0, 2.0])
    assert result["l2_norm"].tolist() == pytest.approx(
        [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(5.0)]
    )
    # On a count DTM these correspond to document frequency and term frequency.
    assert result.set_index("feature").loc["cherry", "nonzero_rows"] == 2
    assert result.set_index("feature").loc["cherry", "sum"] == 3.0


def test_matrix_summary_matches_dense_and_sparse() -> None:
    dense = analysis.matrix_summary(_MatrixFixture(), batch_size=2)
    sparse_result = analysis.matrix_summary(
        _MatrixFixture(sparse_matrix=True), batch_size=2
    )
    for result in (dense, sparse_result):
        assert result.n_rows == 4
        assert result.n_features == 3
        assert result.nonzero_values == 6
        assert result.density == pytest.approx(0.5)
        assert result.zero_rows == 1
        assert result.zero_features == 0
        assert result.sum == pytest.approx(7.0)
        assert result.mean == pytest.approx(7.0 / 12.0)
        assert result.min == 0.0
        assert result.max == 2.0
        assert result.l1_norm == pytest.approx(7.0)
        assert result.frobenius_norm == pytest.approx(3.0)
        assert result.to_frame().loc[0, "artifact_id"] == "art_unit4"


def test_matrix_analytics_reject_non_matrix_artifacts() -> None:
    artifact = _MatrixFixture()
    artifact.artifact_type = ArtifactType.TABLE
    for function, kwargs in [
        (analysis.cosine_similarity, {"position": 0, "other_position": 1}),
        (analysis.distance, {"position": 0, "other_position": 1}),
        (analysis.row_summary, {}),
        (analysis.feature_summary, {}),
        (analysis.matrix_summary, {}),
    ]:
        with pytest.raises(UnsupportedArtifactOperationError):
            function(artifact, **kwargs)


class _ContextFixture:
    primary_key = ("doc_id",)

    def __init__(self):
        self.rows = pd.DataFrame(
            {
                "doc_id": [0, 1, 2, 3],
                "title": ["zero", "one", "two", "three"],
                "group": ["A", "B", "A", "B"],
            }
        )

    def position_by_key(self, key):
        return int(key["doc_id"] if isinstance(key, dict) else key)

    def query_columns(self, *, metadata_mode="none"):
        cols = [
            {
                "namespace": "key",
                "base_name": "doc_id",
                "qualified_name": "key.doc_id",
                "output_name": "doc_id",
            },
            {
                "namespace": "data",
                "base_name": "title",
                "qualified_name": "data.title",
                "output_name": "title",
            },
        ]
        if metadata_mode != "none":
            cols.append(
                {
                    "namespace": "metadata",
                    "base_name": "group",
                    "qualified_name": "metadata.ctx.group",
                    "output_name": "group",
                }
            )
        return {"columns": cols}

    def query(
        self, *, positions, data_columns, metadata_columns, metadata_mode, **kwargs
    ):
        _ = kwargs
        frame = self.rows.iloc[[int(v) for v in positions]].copy()
        keep = ["doc_id"]
        if data_columns is True:
            keep.append("title")
        elif data_columns not in (False, None):
            keep.extend(
                [data_columns] if isinstance(data_columns, str) else list(data_columns)
            )
        if metadata_columns is True and metadata_mode != "none":
            keep.append("group")
        elif metadata_columns not in (False, None) and metadata_mode != "none":
            keep.extend(
                [metadata_columns]
                if isinstance(metadata_columns, str)
                else list(metadata_columns)
            )
        return frame.loc[:, list(dict.fromkeys(keep))].reset_index(drop=True)


class _FakeProject:
    def __init__(self, context):
        self.context = context

    def get_artifact(self, value):
        return self.context if value == "context" else value


def test_nearest_neighbors_can_append_exact_key_context() -> None:
    artifact = _MatrixFixture(sparse_matrix=True)
    context = _ContextFixture()
    artifact.project = _FakeProject(context)
    result = analysis.nearest_neighbors(
        artifact,
        position=0,
        metric="euclidean",
        k=2,
        batch_size=2,
        context="context",
        context_data_columns=["title"],
        context_metadata_columns=["group"],
        context_metadata_mode="local",
    )
    assert result["doc_id"].tolist() == [2, 1]
    assert result["title"].tolist() == ["two", "one"]
    assert result["group"].tolist() == ["A", "B"]


def test_nearest_neighbors_context_requires_leading_key_prefix() -> None:
    artifact = _MatrixFixture()
    artifact.primary_key = ("doc_id", "sentence_id")
    context = _ContextFixture()
    artifact.project = _FakeProject(context)
    # document-level context is valid for sentence-level neighbors
    from text_analysis_lab.analysis.neighbors import _validate_context_keys

    _validate_context_keys(artifact, context)

    context.primary_key = ("sentence_id",)
    with pytest.raises(ArtifactError, match="leading prefix"):
        _validate_context_keys(artifact, context)
