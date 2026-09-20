from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError, LineageError
from text_analysis_lab.core.writer import create_artifact_writer


def _register_table(
    project: teal.Project,
    *,
    artifact_id: str,
    keys: pd.DataFrame,
    data: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
):
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    payload: dict[str, object] = {"keys": keys.reset_index(drop=True)}
    if data is not None:
        payload["data"] = data.reset_index(drop=True)
    if metadata is not None:
        payload["metadata"] = metadata.reset_index(drop=True)
    writer.write(payload)
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_register_external_new_key_data_metadata_and_batching(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_new")
    try:
        frame = pd.DataFrame(
            {
                "pair_id": [3, 1, 2],
                "score": [0.3, 0.1, 0.2],
                "kind": ["c", "a", "b"],
            }
        )
        artifact = project.register_external(
            frame,
            primary_key="pair_id",
            data_fields="score",
            metadata_fields="kind",
            batch_size=2,
            memo="Externally computed pair scores.",
        )
        out = artifact.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="local",
            include_position=True,
            order_by="_position",
            form="table",
        )
        assert out["pair_id"].astype(int).tolist() == [3, 1, 2]
        assert out["score"].tolist() == pytest.approx([0.3, 0.1, 0.2])
        assert out["kind"].tolist() == ["c", "a", "b"]
        assert artifact.descriptor["lineage"] == {
            "lineage_mode": "new_key",
            "basis_artifact_ids": [],
        }
        operation = project.catalog.get_operation(artifact.operation_id)
        assert operation["operation_type"] == "register"
        descriptor = json.loads(
            (project.storage.operation_dir(artifact.operation_id) / "operation.json").read_text()
        )
        assert descriptor["replayable"] is False
        assert descriptor["sources"] == {}
        assert descriptor["basis_labels"] == []
        memo_row = project.catalog.get_memo(
            target_type="operation", target_id=artifact.operation_id
        )
        assert memo_row is not None
        assert memo_row["body"] == "Externally computed pair scores."
    finally:
        project.close()


def test_register_external_span_lineage_and_provenance_sources_are_independent(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_span")
    try:
        turns = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 0, 0],
                    "turn_id": [0, 1, 2, 3],
                }
            ),
            data=pd.DataFrame({"text": ["a", "b", "c", "d"]}),
            metadata=pd.DataFrame({"partner": ["robot"] * 4}),
        )
        lsa = _register_table(
            project,
            artifact_id="lsa",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 0, 0],
                    "turn_id": [0, 1, 2, 3],
                }
            ),
            data=pd.DataFrame({"dim0": [1.0, 0.0, 1.0, 0.0]}),
        )
        embedding = _register_table(
            project,
            artifact_id="embedding",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 0, 0],
                    "turn_id": [0, 1, 2, 3],
                }
            ),
            data=pd.DataFrame({"dim0": [0.1, 0.2, 0.3, 0.4]}),
        )

        pairs = pd.DataFrame(
            {
                "child_id": [0, 0],
                "turn_id_start": [0, 2],
                "turn_id_end": [1, 3],
                "lsa_cosine": [0.5, 0.6],
                "embedding_cosine": [0.7, 0.8],
            }
        )
        registered = project.register_external(
            pairs,
            primary_key=["child_id", "turn_id_start", "turn_id_end"],
            data_fields=["lsa_cosine", "embedding_cosine"],
            sources={
                "turns": turns,
                "lsa": lsa,
                "embedding": embedding,
            },
            lineage_mode="span_key",
            basis_labels="turns",
            memo="Pairwise measures computed externally; turns define row identity.",
        )

        assert registered.primary_key == ["child_id", "turn_id_start", "turn_id_end"]
        assert registered.descriptor["lineage"] == {
            "lineage_mode": "span_key",
            "basis_artifact_ids": [turns.artifact_id],
        }
        descriptor = json.loads(
            (project.storage.operation_dir(registered.operation_id) / "operation.json").read_text()
        )
        assert descriptor["sources"] == {
            "turns": turns.artifact_id,
            "lsa": lsa.artifact_id,
            "embedding": embedding.artifact_id,
        }
        assert descriptor["basis_labels"] == ["turns"]

        source_rows = project.catalog.operation_sources(registered.operation_id)
        assert {
            row["source_label"]: row["source_artifact_id"] for row in source_rows
        } == descriptor["sources"]

        # span_key remains a non-bubbling leaf boundary; context can still map the
        # registered pair span back onto its atomic turn basis.
        focal = registered.query(
            where="turn_id_start = 0 AND turn_id_end = 1",
            key_columns=True,
            data_columns=True,
            metadata_columns=False,
            form="single",
        )
        context = registered.get_context(focal, context_artifact=turns, before=0, after=0)
        assert context["turn_id"].astype(int).tolist() == [0, 1]
    finally:
        project.close()


def test_register_external_validates_declared_lineage_schema(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_invalid")
    try:
        source = _register_table(
            project,
            artifact_id="source",
            keys=pd.DataFrame({"row_id": [0, 1]}),
            data=pd.DataFrame({"text": ["a", "b"]}),
        )
        bad = pd.DataFrame({"row_id": [0], "score": [1.0]})
        with pytest.raises(LineageError, match="span_key output primary key"):
            project.register_external(
                bad,
                primary_key="row_id",
                data_fields="score",
                sources={"source": source},
                lineage_mode="span_key",
                basis_labels="source",
            )

        with pytest.raises(LineageError, match="requires at least one basis"):
            project.register_external(
                bad,
                primary_key="row_id",
                data_fields="score",
                sources={"source": source},
                lineage_mode="preserved_key",
            )

        with pytest.raises(LineageError, match="cannot declare basis_labels"):
            project.register_external(
                bad,
                primary_key="row_id",
                data_fields="score",
                sources={"source": source},
                lineage_mode="new_key",
                basis_labels="source",
            )
    finally:
        project.close()


def test_register_external_validates_fields_and_keys(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_validation")
    try:
        with pytest.raises(ArtifactError, match="Duplicate primary-key"):
            project.register_external(
                pd.DataFrame({"row_id": [0, 0], "score": [1, 2]}),
                primary_key="row_id",
                data_fields="score",
            )

        with pytest.raises(ArtifactError, match="integer"):
            project.register_external(
                pd.DataFrame({"row_id": [0.5], "score": [1]}),
                primary_key="row_id",
                data_fields="score",
            )

        with pytest.raises(ArtifactError, match="overlap"):
            project.register_external(
                pd.DataFrame({"row_id": [0], "score": [1]}),
                primary_key="row_id",
                data_fields="score",
                metadata_fields="score",
            )
    finally:
        project.close()


def test_register_external_empty_keys_only_and_reopen(tmp_path: Path) -> None:
    root = tmp_path / "project"
    project = teal.Project.create(root, name="register_empty")
    try:
        empty = pd.DataFrame({"pair_id": pd.Series(dtype="int64")})
        artifact = project.register_external(empty, primary_key="pair_id")
        artifact_id = artifact.artifact_id
        assert artifact.n_rows == 0
        assert artifact.primary_key == ["pair_id"]
    finally:
        project.close()

    reopened = teal.Project.open(root)
    try:
        artifact = reopened.get_artifact(artifact_id)
        assert artifact.n_rows == 0
        out = artifact.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            form="table",
        )
        assert out.empty
        assert list(out.columns) == ["pair_id"]
    finally:
        reopened.close()


def test_register_external_parquet_file_and_dataset_folder_stream_in_batches(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": [0, 1, 2, 3, 4],
            "score": [0.0, 0.1, 0.2, 0.3, 0.4],
            "kind": ["a", "a", "b", "b", "c"],
        }
    )
    one_file = tmp_path / "pairs.parquet"
    frame.to_parquet(one_file, index=False)
    dataset = tmp_path / "pairs_dataset"
    dataset.mkdir()
    frame.iloc[:2].to_parquet(dataset / "part-000.parquet", index=False)
    frame.iloc[2:].to_parquet(dataset / "part-001.parquet", index=False)

    project = teal.Project.create(tmp_path / "project", name="register_paths")
    try:
        for external in (one_file, dataset):
            artifact = project.register_external(
                external,
                primary_key="pair_id",
                data_fields="score",
                metadata_fields="kind",
                batch_size=2,
            )
            out = artifact.query(
                key_columns=True,
                data_columns=True,
                metadata_columns=True,
                metadata_mode="local",
                order_by="_position",
                form="table",
            )
            assert out["pair_id"].astype(int).tolist() == [0, 1, 2, 3, 4]
            assert out["score"].tolist() == pytest.approx(frame["score"].tolist())
            descriptor = json.loads(
                (project.storage.operation_dir(artifact.operation_id) / "operation.json").read_text()
            )
            assert descriptor["registered_rows"] == 5
            assert descriptor["registered_batches"] == 3
    finally:
        project.close()


def test_register_external_preserves_caller_payload_batches(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_payload_batches")
    try:
        batches = [
            {
                "keys": pd.DataFrame({"row_id": [0, 1]}),
                "data": pd.DataFrame({"score": [1.0, 2.0]}),
            },
            {
                "keys": pd.DataFrame({"row_id": [2]}),
                "data": pd.DataFrame({"score": [3.0]}),
            },
        ]
        artifact = project.register_external(
            batches,
            primary_key="row_id",
            artifact_type="table",
            # batch_size is intentionally irrelevant for caller-supplied batches.
            batch_size=1,
        )
        descriptor = json.loads(
            (project.storage.operation_dir(artifact.operation_id) / "operation.json").read_text()
        )
        assert descriptor["registered_batches"] == 2
        assert descriptor["registered_rows"] == 3
        out = artifact.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        assert out["row_id"].astype(int).tolist() == [0, 1, 2]
        assert out["score"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    finally:
        project.close()


@pytest.mark.parametrize("artifact_type", ["dense_matrix", "sparse_matrix"])
def test_register_external_matrix_batches(tmp_path: Path, artifact_type: str) -> None:
    import numpy as np
    from scipy import sparse

    project = teal.Project.create(tmp_path / artifact_type, name="register_matrix")
    try:
        first = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        second = np.asarray([[1.0, 1.0]], dtype=np.float32)
        if artifact_type == "sparse_matrix":
            first = sparse.csr_matrix(first)
            second = sparse.csr_matrix(second)
        batches = [
            {
                "keys": pd.DataFrame({"row_id": [0, 1]}),
                "data": {"values": first, "columns": ["x", "y"]},
            },
            {
                "keys": pd.DataFrame({"row_id": [2]}),
                "data": {"values": second, "columns": ["x", "y"]},
            },
        ]
        artifact = project.register_external(
            batches,
            artifact_type=artifact_type,
            primary_key="row_id",
        )
        assert artifact.get_feature_frame()["column"].tolist() == ["x", "y"]
        values = artifact.get_matrix(positions=[0, 1, 2])
        values = values.toarray() if sparse.issparse(values) else values
        assert np.array_equal(values, np.asarray([[1, 0], [0, 1], [1, 1]]))
    finally:
        project.close()


def test_register_external_detects_cross_batch_duplicate_keys(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="register_dupe_batches")
    try:
        batches = [
            {"keys": pd.DataFrame({"row_id": [0, 1]})},
            {"keys": pd.DataFrame({"row_id": [1, 2]})},
        ]
        with pytest.raises(ArtifactError, match="Duplicate primary-key"):
            project.register_external(batches, primary_key="row_id")
    finally:
        project.close()
