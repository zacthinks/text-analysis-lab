from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer

_VALUES = np.array(
    [
        [1.0, 0.0, 2.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
)
_FEATURES = ["apple", "banana", "cherry"]


def _register_matrix(project: teal.Project, *, artifact_type: str):
    artifact_id = f"art_{artifact_type}"
    writer = create_artifact_writer(
        artifact_type=artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_type,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    for start in (0, 2):
        stop = start + 2
        values = _VALUES[start:stop]
        if artifact_type == "sparse_matrix":
            values = sparse.csr_matrix(values)
        writer.write(
            {
                "keys": pd.DataFrame({"doc_id": [start, start + 1]}),
                "data": {"values": values, "columns": _FEATURES},
            }
        )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        label=artifact_type,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


@pytest.mark.parametrize("artifact_type", ["sparse_matrix", "dense_matrix"])
def test_real_matrix_analytics_round_trip(tmp_path: Path, artifact_type: str) -> None:
    project = teal.Project.create(
        tmp_path / artifact_type, name=f"unit4_{artifact_type}"
    )
    try:
        artifact = _register_matrix(project, artifact_type=artifact_type)
        before = len(project.list_artifacts())

        batches = list(
            artifact.iter_batches(
                batch_size=2,
                key_columns=False,
                data_columns=True,
                metadata_columns=False,
                metadata_mode="none",
                form="native",
                include_position=True,
                streaming_mode="arrow",
            )
        )
        assert [
            batch["info"]["_position"].astype(int).tolist() for batch in batches
        ] == [[0, 1], [2, 3]]

        similarity = artifact.analysis.cosine_similarity(key=0, other_key=2)
        assert similarity == pytest.approx(2.0 / math.sqrt(10.0))
        assert artifact.analysis.distance(
            key=0, other_key=2, metric="cosine"
        ) == pytest.approx(1.0 - similarity)
        assert artifact.analysis.distance(
            key=0, other_key=1, metric="euclidean"
        ) == pytest.approx(math.sqrt(5.0))
        assert artifact.analysis.distance(
            key=0, other_key=1, metric="manhattan"
        ) == pytest.approx(3.0)

        neighbors = artifact.analysis.nearest_neighbors(
            key=0, k=2, metric="euclidean", batch_size=2
        )
        assert neighbors["doc_id"].astype(int).tolist() == [2, 1]
        assert neighbors["distance"].tolist() == pytest.approx(
            [math.sqrt(3.0), math.sqrt(5.0)]
        )

        rows = artifact.analysis.row_summary(batch_size=2)
        assert rows["doc_id"].astype(int).tolist() == [0, 1, 2, 3]
        assert rows["nonzero_features"].astype(int).tolist() == [2, 2, 2, 0]

        features = artifact.analysis.feature_summary(batch_size=2)
        cherry = features.set_index("feature").loc["cherry"]
        assert int(cherry["nonzero_rows"]) == 2
        assert float(cherry["sum"]) == pytest.approx(3.0)

        summary = artifact.analysis.matrix_summary(batch_size=2)
        assert summary.n_rows == 4
        assert summary.n_features == 3
        assert summary.nonzero_values == 6
        assert summary.zero_rows == 1
        assert summary.sum == pytest.approx(7.0)
        assert summary.frobenius_norm == pytest.approx(3.0)

        # Analytic Methods must remain ephemeral.
        assert len(project.list_artifacts()) == before

        artifact_id = artifact.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(tmp_path / artifact_type)
    try:
        artifact = reopened.get_artifact(artifact_id)
        assert artifact.analysis.matrix_summary(batch_size=3).sum == pytest.approx(7.0)
        assert artifact.analysis.cosine_similarity(
            position=0, other_position=2
        ) == pytest.approx(2.0 / math.sqrt(10.0))
    finally:
        reopened.close()
