from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from text_analysis_lab import analysis
from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import UnsupportedArtifactOperationError
from text_analysis_lab.core.types import ArtifactType


def test_context_defaults_to_primary_key_columns() -> None:
    parameter = inspect.signature(BaseArtifact.get_context).parameters["key_columns"]
    assert parameter.default is True


class _FakeMatrixArtifact:
    artifact_type = ArtifactType.DENSE_MATRIX
    artifact_id = "art_000001"
    label = "geometry"
    status = "complete"
    n_rows = 4
    primary_key = ("doc_id",)
    operation_id = "run_000001"
    descriptor: ClassVar[dict[str, object]] = {
        "lineage": {"lineage_mode": "new_key", "basis_artifact_ids": []},
        "components": {"keys": {}, "data": {}},
    }

    def __init__(self) -> None:
        self.matrix = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        self.seen_batch_sizes: list[int] = []

    def position_by_key(self, key):
        return int(key)

    def get_matrix(self, *, positions=None, data_columns=True):
        if positions is None:
            return self.matrix
        return self.matrix[[int(value) for value in positions], :]

    def iter_batches(self, *, batch_size, **kwargs):
        self.seen_batch_sizes.append(int(batch_size))
        for start in range(0, self.n_rows, 2):
            stop = min(self.n_rows, start + 2)
            positions = list(range(start, stop))
            yield {
                "info": pd.DataFrame({"_position": positions}),
                "matrix": self.matrix[positions, :],
            }

    def query(self, *, positions, **kwargs):
        # Deliberately return canonical position order rather than caller order;
        # nearest_neighbors() must restore rank order explicitly.
        ordered = sorted(int(value) for value in positions)
        return pd.DataFrame(
            {
                "doc_id": ordered,
                "_position": ordered,
            }
        )


def test_nearest_neighbors_is_ephemeral_bounded_and_keyed() -> None:
    artifact = _FakeMatrixArtifact()
    result = analysis.nearest_neighbors(
        artifact,
        key=0,
        k=2,
        metric="cosine",
        batch_size=2,
    )

    assert result["doc_id"].tolist() == [1, 2]
    assert result["_position"].tolist() == [1, 2]
    assert result["rank"].tolist() == [1, 2]
    assert result["distance"].iloc[0] < result["distance"].iloc[1]
    assert artifact.seen_batch_sizes == [2]


def test_nearest_neighbors_can_include_focus() -> None:
    artifact = _FakeMatrixArtifact()
    result = analysis.nearest_neighbors(
        artifact,
        position=0,
        k=1,
        include_self=True,
    )
    assert result["_position"].tolist() == [0]
    assert result["distance"].tolist() == pytest.approx([0.0])


def test_nearest_neighbors_requires_exactly_one_locator() -> None:
    artifact = _FakeMatrixArtifact()
    with pytest.raises(ValueError):
        analysis.nearest_neighbors(artifact)
    with pytest.raises(ValueError):
        analysis.nearest_neighbors(artifact, key=0, position=0)


def test_nearest_neighbors_rejects_non_matrix_artifact() -> None:
    artifact = _FakeMatrixArtifact()
    artifact.artifact_type = ArtifactType.TABLE
    with pytest.raises(UnsupportedArtifactOperationError):
        analysis.nearest_neighbors(artifact, key=0)


def test_summarize_uses_descriptor_and_catalog_facing_state() -> None:
    artifact = SimpleNamespace(
        artifact_id="art_000007",
        label="sentences",
        artifact_type=ArtifactType.TABLE,
        status="complete",
        n_rows=12,
        primary_key=("document_id", "sentence_id"),
        operation_id="run_000003",
        descriptor={
            "lineage": {
                "lineage_mode": "extended_key",
                "basis_artifact_ids": ["art_000002"],
            },
            "components": {"keys": {}, "data": {}, "metadata": {}},
        },
    )

    result = analysis.summarize(artifact)
    assert result.artifact_id == "art_000007"
    assert result.label == "sentences"
    assert result.primary_key == ("document_id", "sentence_id")
    assert result.lineage_mode == "extended_key"
    assert result.basis_artifact_ids == ("art_000002",)
    assert result.components == ("keys", "data", "metadata")
    assert result.to_frame().loc[0, "n_rows"] == 12


def test_nearest_neighbors_supports_sparse_matrix_batches() -> None:
    from scipy import sparse

    artifact = _FakeMatrixArtifact()
    artifact.artifact_type = ArtifactType.SPARSE_MATRIX
    artifact.matrix = sparse.csr_matrix(artifact.matrix)
    result = analysis.nearest_neighbors(artifact, position=0, k=2, batch_size=2)
    assert result["_position"].tolist() == [1, 2]
