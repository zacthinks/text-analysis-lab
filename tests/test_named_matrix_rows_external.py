from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.writer import create_artifact_writer


def _register(project, artifact_id, *, batches, artifact_type="dense_matrix"):
    writer = create_artifact_writer(
        artifact_type=artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    next_id = 0
    for names, values in batches:
        matrix = np.asarray(values, dtype=np.float32)
        if artifact_type == "sparse_matrix":
            matrix = sparse.csr_matrix(matrix)
        writer.write(
            {
                "keys": pd.DataFrame(
                    {"row_id": np.arange(next_id, next_id + len(names), dtype=np.int64)}
                ),
                "data": {
                    "values": matrix,
                    "columns": ["x", "y"],
                    "row_names": names,
                    "row_name": "term",
                },
            }
        )
        next_id += len(names)
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


@pytest.mark.parametrize("artifact_type", ["dense_matrix", "sparse_matrix"])
def test_named_matrix_rows_round_trip_and_analytics(
    tmp_path: Path, artifact_type: str
) -> None:
    project = teal.Project.create(tmp_path / "named", name="named")
    try:
        matrix = _register(
            project,
            "art_named",
            batches=[(["alpha", "beta"], [[1, 0], [0, 1]]), (["gamma"], [[1, 1]])],
            artifact_type=artifact_type,
        )
        assert matrix.has_row_names
        assert matrix.row_name == "term"
        assert matrix.get_row_names() == ["alpha", "beta", "gamma"]
        assert matrix.get_row_names(positions=[2, 0]) == ["gamma", "alpha"]
        assert matrix.position_by_row_name("beta") == 1
        gamma = matrix.get_row_by_name("gamma")
        rows = matrix.get_rows_by_name(["gamma", "alpha"])
        gamma = gamma.toarray() if sparse.issparse(gamma) else gamma
        rows = rows.toarray() if sparse.issparse(rows) else rows
        assert np.array_equal(gamma, [[1, 1]])
        assert np.array_equal(rows, [[1, 1], [1, 0]])
        assert matrix.analysis.cosine_similarity(
            row_name="alpha", other_row_name="gamma"
        ) == pytest.approx(1 / np.sqrt(2))
        neighbors = matrix.analysis.nearest_neighbors(row_name="alpha", k=2)
        assert neighbors["term"].tolist() == ["gamma", "beta"]
        with pytest.raises(KeyError, match="missing"):
            matrix.get_row_by_name("missing")
    finally:
        project.close()


def test_named_rows_must_be_consistent_and_globally_unique(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "duplicate"
    writer = create_artifact_writer(
        artifact_type="dense_matrix",
        artifact_dir=artifact_dir,
        artifact_id="art_duplicate",
        label="duplicate",
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": [0]}),
            "data": {
                "values": np.asarray([[1, 0]]),
                "columns": ["x", "y"],
                "row_names": ["same"],
                "row_name": "term",
            },
        }
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": [1]}),
            "data": {
                "values": np.asarray([[0, 1]]),
                "columns": ["x", "y"],
                "row_names": ["same"],
                "row_name": "term",
            },
        }
    )
    with pytest.raises(ArtifactError, match="Duplicate matrix row_names"):
        writer.finalize()
    descriptor = json.loads((artifact_dir / "artifact.json").read_text())
    assert descriptor["status"] == "failed"
