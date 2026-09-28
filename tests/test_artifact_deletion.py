from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_analysis_lab.core.errors import (
    ArtifactDeletionBlockedError,
    ArtifactRestoreBlockedError,
    InvalidAliasError,
)
from text_analysis_lab.core.project import Project


def _make_artifact(
    project: Project,
    artifact_id: str,
    *,
    basis: tuple[str, ...] = (),
    alias: str | None = None,
):
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label="output",
        lineage_mode="preserved_key" if basis else "new_key",
        status="complete",
        basis_artifact_ids=basis,
    )

    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for component in ("keys", "data", "metadata"):
        (artifact_dir / component).mkdir(exist_ok=True)

    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": "output",
        "status": "complete",
        "primary_key": ["doc_id"],
        "n_rows": 1,
        "components": {
            "keys": {"format": "parquet_dataset", "path": "keys/"},
            "data": {"format": "parquet_dataset", "path": "data/"},
            "metadata": {"format": "parquet_dataset", "path": "metadata/"},
        },
        "lineage": {
            "mode": "preserved_key" if basis else "new_key",
            "basis_artifact_ids": list(basis),
        },
        "operation_id": None,
        "write": {"mode": "batch", "parts": 1},
    }
    (artifact_dir / "artifact.json").write_text(
        json.dumps(descriptor),
        encoding="utf-8",
    )

    if alias is not None:
        project.catalog.add_artifact_alias(artifact_id, alias)

    return project.get_artifact(artifact_id)


def test_delete_blocks_live_dependents_without_mutating_artifact(
    tmp_path: Path,
) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        root = _make_artifact(project, "art_000001", alias="root")
        _make_artifact(
            project,
            "art_000002",
            basis=("art_000001",),
            alias="child",
        )

        with pytest.raises(ArtifactDeletionBlockedError):
            root.delete()

        assert not root.is_deleted
        assert project.catalog.aliases_for_artifact(root.artifact_id) == ["root"]
    finally:
        project.close()


def test_recursive_delete_restore_and_new_alias(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        root = _make_artifact(project, "art_000001", alias="root")
        child = _make_artifact(
            project,
            "art_000002",
            basis=("art_000001",),
            alias="child",
        )
        grandchild = _make_artifact(
            project,
            "art_000003",
            basis=("art_000002",),
            alias="grandchild",
        )

        root.delete(recursive=True, memo="Superseded by a rebuilt corpus.")

        assert root.is_deleted
        assert child.is_deleted
        assert grandchild.is_deleted
        assert not root.is_purged
        assert project.catalog.aliases_for_artifact(root.artifact_id) == []

        memo = project.catalog.get_memo(
            target_type="artifact",
            target_id=root.artifact_id,
        )
        assert memo is not None
        assert "Superseded by a rebuilt corpus." in memo["body"]

        with pytest.raises(ArtifactRestoreBlockedError):
            grandchild.restore(new_alias="grandchild_restored")

        restored_root = root.restore(new_alias="root_restored")
        restored_child = child.restore(new_alias="child_restored")
        restored_grandchild = grandchild.restore(new_alias="grandchild_restored")

        assert not restored_root.is_deleted
        assert not restored_child.is_deleted
        assert not restored_grandchild.is_deleted
        assert (
            project.get_artifact("grandchild_restored").artifact_id
            == grandchild.artifact_id
        )
    finally:
        project.close()


def test_purge_payload_preserves_descriptor_and_memo_and_blocks_restore(
    tmp_path: Path,
) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        artifact = _make_artifact(project, "art_000001", alias="docs")

        artifact.delete(
            purge_payload=True,
            memo="Payload no longer needed.",
        )

        assert artifact.is_deleted
        assert artifact.is_purged

        artifact_dir = project.storage.artifact_dir(artifact.artifact_id)
        assert (artifact_dir / "artifact.json").exists()
        assert not (artifact_dir / "keys").exists()
        assert not (artifact_dir / "data").exists()
        assert not (artifact_dir / "metadata").exists()

        memo = project.catalog.get_memo(
            target_type="artifact",
            target_id=artifact.artifact_id,
        )
        assert memo is not None
        assert "Payload no longer needed." in memo["body"]

        with pytest.raises(ArtifactRestoreBlockedError):
            artifact.restore(new_alias="docs_restored")
    finally:
        project.close()


def test_restore_alias_conflict_is_atomic(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        first = _make_artifact(project, "art_000001", alias="first")
        second = _make_artifact(project, "art_000002", alias="second")

        second.delete()
        project.catalog.add_artifact_alias(first.artifact_id, "taken")

        with pytest.raises(InvalidAliasError):
            second.restore(new_alias="taken")

        assert second.is_deleted
    finally:
        project.close()


def test_operation_source_dependency_blocks_delete_and_recursive_delete_cascades(
    tmp_path: Path,
) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        source = _make_artifact(project, "art_000001")
        output = _make_artifact(project, "art_000002")

        project.catalog.register_operator(
            operator_id="optr_000001",
            operation_type="translate",
        )
        project.catalog.register_operation(
            operation_id="oper_000001",
            operation_type="translate",
            operator_id="optr_000001",
            status="complete",
        )
        project.catalog.add_operation_source(
            "oper_000001",
            "source",
            source.artifact_id,
        )
        project.catalog.add_operation_output(
            "oper_000001",
            "output",
            output.artifact_id,
            ordinal=0,
        )

        with pytest.raises(ArtifactDeletionBlockedError):
            source.delete()

        source.delete(recursive=True)
        assert source.is_deleted
        assert output.is_deleted
    finally:
        project.close()


def test_delete_traverses_through_legacy_deleted_intermediate(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", "project")
    try:
        root = _make_artifact(project, "art_000001")
        intermediate = _make_artifact(
            project,
            "art_000002",
            basis=("art_000001",),
        )
        grandchild = _make_artifact(
            project,
            "art_000003",
            basis=("art_000002",),
        )

        with project.catalog.con as connection:
            connection.execute(
                "UPDATE artifacts SET deleted = 1 WHERE artifact_id = ?",
                (intermediate.artifact_id,),
            )

        with pytest.raises(ArtifactDeletionBlockedError):
            root.delete()

        root.delete(recursive=True)
        assert root.is_deleted
        assert grandchild.is_deleted
    finally:
        project.close()
