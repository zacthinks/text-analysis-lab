from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.writer import create_artifact_writer


def _register_table(
    project: teal.Project,
    *,
    artifact_id: str,
    label: str,
    keys: pd.DataFrame,
    data: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    lineage_mode: str = "new_key",
    basis_artifact_ids: tuple[str, ...] = (),
):
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=basis_artifact_ids,
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
        label=label,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=basis_artifact_ids,
    )
    return project.get_artifact(artifact_id)


def _full(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=True,
        metadata_mode="full",
        include_position=True,
        order_by="_position",
        form="table",
    )


def test_join_is_lazy_basis_first_and_survives_reopen(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="join")
    try:
        basis = _register_table(
            project,
            artifact_id="art_basis",
            label="basis",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            data=pd.DataFrame({"text": ["zero", "one", "two"]}),
            metadata=pd.DataFrame({"group": ["a", "a", "b"]}),
        )
        scores = _register_table(
            project,
            artifact_id="art_scores",
            label="scores",
            keys=pd.DataFrame({"row_id": [1, 2, 3]}),
            data=pd.DataFrame({"score": [0.25, 0.75, 1.0]}),
        )
        annotations = _register_table(
            project,
            artifact_id="art_annotations",
            label="annotations",
            keys=pd.DataFrame({"row_id": [0, 2]}),
            metadata=pd.DataFrame({"reviewed": [True, False]}),
        )

        joined = project.join(
            basis, scores, annotations, output_label="joined", batch_size=2
        )
        assert joined.descriptor["lineage"] == {
            "lineage_mode": "joined_key",
            "basis_artifact_ids": [
                basis.artifact_id,
                scores.artifact_id,
                annotations.artifact_id,
            ],
        }
        assert set(joined.components) == {"keys"}
        assert joined.data_artifact is joined

        frame = _full(joined)
        assert frame["_position"].tolist() == [0, 1, 2]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["text"].tolist() == ["zero", "one", "two"]
        assert pd.isna(frame.loc[0, "score"])
        assert frame.loc[1:, "score"].tolist() == pytest.approx([0.25, 0.75])
        assert frame["group"].tolist() == ["a", "a", "b"]
        assert frame["reviewed"].tolist()[0] == True
        assert pd.isna(frame.loc[1, "reviewed"])
        assert frame["reviewed"].tolist()[2] == False
        assert 3 not in set(frame["row_id"].astype(int))

        columns = joined.query_columns(metadata_mode="full")
        assert columns["data"] == ["text", "score"]
        assert columns["metadata"] == ["group", "reviewed"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        joined = reopened.get_artifact("art_000001")
        frame = _full(joined)
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["text"].tolist() == ["zero", "one", "two"]
        assert pd.isna(frame.loc[0, "score"])
        assert frame.loc[2, "score"] == pytest.approx(0.75)
    finally:
        reopened.close()


def test_join_rejects_schema_mismatch_and_new_field_collisions(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="join_validation")
    try:
        basis = _register_table(
            project,
            artifact_id="art_basis",
            label="basis",
            keys=pd.DataFrame({"row_id": [0, 1]}),
            data=pd.DataFrame({"text": ["a", "b"]}),
        )
        mismatch = _register_table(
            project,
            artifact_id="art_mismatch",
            label="mismatch",
            keys=pd.DataFrame({"doc_id": [0, 1]}),
            data=pd.DataFrame({"score": [1, 2]}),
        )
        with pytest.raises(ArtifactError, match="same primary-key fields"):
            project.join(basis, mismatch)

        collision = _register_table(
            project,
            artifact_id="art_collision",
            label="collision",
            keys=pd.DataFrame({"row_id": [0, 1]}),
            data=pd.DataFrame({"text": ["x", "y"]}),
        )
        with pytest.raises(ArtifactError, match="field-name collision"):
            project.join(basis, collision)
    finally:
        project.close()


def test_select_keys_uses_source_order_and_inherits_data_after_reopen(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="select_keys")
    try:
        source = _register_table(
            project,
            artifact_id="art_source",
            label="source",
            keys=pd.DataFrame({"row_id": [0, 1, 2, 3, 4]}),
            data=pd.DataFrame({"text": ["a", "b", "c", "d", "e"]}),
            metadata=pd.DataFrame({"group": [0, 0, 1, 1, 2]}),
        )
        selected = project.select_keys(source, [4, 1], output_label="chosen")
        assert selected.descriptor["lineage"] == {
            "lineage_mode": "preserved_key",
            "basis_artifact_ids": [source.artifact_id],
        }
        assert set(selected.components) == {"keys"}
        frame = _full(selected)
        assert frame["row_id"].astype(int).tolist() == [1, 4]
        assert frame["text"].tolist() == ["b", "e"]
        assert frame["group"].astype(int).tolist() == [0, 2]

        with pytest.raises(ArtifactError, match="duplicate"):
            project.select_keys(source, [1, 1])
        with pytest.raises(ArtifactError, match="not present"):
            project.select_keys(source, [999])
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        selected = reopened.get_artifact("art_000001")
        frame = _full(selected)
        assert frame["row_id"].astype(int).tolist() == [1, 4]
        assert frame["text"].tolist() == ["b", "e"]
    finally:
        reopened.close()
