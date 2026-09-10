from __future__ import annotations

import json
from pathlib import Path

import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.errors import AliasBundleError, AliasOverwriteBlockedError, InvalidAliasError
from text_analysis_lab.core.idempotence import finalize_alias_plan, prepare_alias_plan, reused_outputs


def _seed_artifact(project: teal.Project, artifact_id: str, *, label: str = "output", basis=()):
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=label,
        lineage_mode="new_key" if not basis else "preserved_key",
        status="complete",
        basis_artifact_ids=basis,
    )
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "keys").mkdir(exist_ok=True)
    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": label,
        "status": "complete",
        "primary_key": ["row_id"],
        "n_rows": 0,
        "components": {"keys": {"format": "parquet_dataset", "path": "keys/"}},
        "lineage": {
            "lineage_mode": "new_key" if not basis else "preserved_key",
            "basis_artifact_ids": list(basis),
        },
        "operation_id": None,
        "write": {"mode": "batch", "parts": 0},
    }
    project.storage.artifact_descriptor_path(artifact_id).write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    return project.get_artifact(artifact_id)


def test_single_alias_shape_and_multi_alias_shape_are_strict(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="alias_shapes")
    try:
        with pytest.raises(AliasBundleError, match="Single-output"):
            prepare_alias_plan(project, ("output",), {"output": "x"})
        with pytest.raises(AliasBundleError, match="Multi-output"):
            prepare_alias_plan(project, ("a", "b"), "x")
        with pytest.raises(AliasBundleError, match="complete output bundle"):
            prepare_alias_plan(project, ("a", "b"), {"a": "x"})
        with pytest.raises(AliasBundleError, match="distinct"):
            prepare_alias_plan(project, ("a", "b"), {"a": "x", "b": "x"})
        with pytest.raises(AliasBundleError, match="requires alias"):
            prepare_alias_plan(project, ("output",), None, overwrite=True)
    finally:
        project.close()


def test_reuse_returns_existing_artifact_without_mutation(tmp_path: Path, capsys) -> None:
    project = teal.Project.create(tmp_path / "project", name="alias_reuse_core")
    try:
        old = _seed_artifact(project, "art_900001")
        old.add_alias("stable")
        plan = prepare_alias_plan(project, ("output",), "stable")
        assert plan is not None and plan.reuse
        assert reused_outputs(project, plan)["output"].artifact_id == old.artifact_id
        assert "current settings were ignored" in capsys.readouterr().out
    finally:
        project.close()


def test_finalize_new_alias_bundle_is_atomic(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="alias_bind_core")
    try:
        a = _seed_artifact(project, "art_900001", label="a")
        b = _seed_artifact(project, "art_900002", label="b")
        plan = prepare_alias_plan(project, ("a", "b"), {"a": "alias_a", "b": "alias_b"})
        assert plan is not None and not plan.reuse
        finalize_alias_plan(project, plan, {"a": a, "b": b})
        assert project.get_artifact("alias_a").artifact_id == a.artifact_id
        assert project.get_artifact("alias_b").artifact_id == b.artifact_id
    finally:
        project.close()


def test_overwrite_ignores_dependencies_inside_bundle_but_blocks_external_dependents(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="alias_dependency_core")
    try:
        old_a = _seed_artifact(project, "art_900001", label="a")
        old_b = _seed_artifact(project, "art_900002", label="b", basis=(old_a.artifact_id,))
        old_a.add_alias("alias_a")
        old_b.add_alias("alias_b")
        plan = prepare_alias_plan(
            project,
            ("a", "b"),
            {"a": "alias_a", "b": "alias_b"},
            overwrite=True,
        )
        assert plan is not None and not plan.reuse

        external = _seed_artifact(project, "art_900003", label="external", basis=(old_a.artifact_id,))
        assert external.artifact_id
        with pytest.raises(AliasOverwriteBlockedError, match="live dependents"):
            prepare_alias_plan(
                project,
                ("a", "b"),
                {"a": "alias_a", "b": "alias_b"},
                overwrite=True,
            )
    finally:
        project.close()


def test_catalog_bundle_swap_checks_expected_alias_target(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="alias_race_core")
    try:
        old = _seed_artifact(project, "art_900001")
        new = _seed_artifact(project, "art_900002")
        other = _seed_artifact(project, "art_900003")
        old.add_alias("stable")
        project.remove_artifact_alias("stable")
        other.add_alias("stable")
        with pytest.raises(InvalidAliasError, match="changed while the operation was running"):
            project.catalog.replace_artifact_alias_bundle(
                {"stable": new.artifact_id},
                expected_existing={"stable": old.artifact_id},
                retire_artifact_ids={old.artifact_id},
            )
        assert project.get_artifact("stable").artifact_id == other.artifact_id
        assert project.get_artifact(old.artifact_id).artifact_id == old.artifact_id
    finally:
        project.close()
