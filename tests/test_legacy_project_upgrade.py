from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import text_analysis_lab as teal

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.translators import TextLength


def _make_source(project: teal.Project, tmp_path: Path):
    path = tmp_path / "documents.csv"
    pd.DataFrame({"text": ["alpha beta", "gamma"]}).to_csv(path, index=False)
    return project.read_csv(path, text_fields="text", metadata_fields=None)


def _downgrade_operator_to_v1(project: teal.Project, operator_id: str) -> None:
    descriptor_path = project.storage.operator_descriptor_path(operator_id)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["schema_version"] = 1
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    project.catalog._set_operator_reuse_status(operator_id, "legacy_unknown")


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_project_legacy_upgrade_dry_run_does_not_mutate(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade_dry")
    try:
        source = _make_source(project, tmp_path)
        translator = TextLength({"text": "words"})
        output = project.translate(translator, source)["output"]
        assert translator.operator_id is not None
        legacy_id = translator.operator_id
        operation = project.operation_for_artifact(output)
        assert operation is not None
        operation_id = str(operation["operation_id"])
        project.add_operator_alias(legacy_id, "lengths")
        _downgrade_operator_to_v1(project, legacy_id)

        descriptor_before = project.storage.operation_descriptor_path(
            operation_id
        ).read_bytes()
        operator_ids_before = {
            str(row["operator_id"]) for row in project.list_operators()
        }

        report = project.upgrade_legacy_operators(dry_run=True)

        assert report["dry_run"] is True
        assert report["items"][0]["classification"] == "exactly_migratable"
        assert report["items"][0]["complete_operations"] == [operation_id]
        assert project.get_operation(operation_id)["operator_id"] == legacy_id
        assert project.resolve_operator_id("lengths") == legacy_id
        assert project.storage.operation_descriptor_path(
            operation_id
        ).read_bytes() == descriptor_before
        assert {
            str(row["operator_id"]) for row in project.list_operators()
        } == operator_ids_before
        assert project.catalog.legacy_operator_upgrade(legacy_id) is None
    finally:
        project.close()


def test_project_legacy_upgrade_rebinds_exact_completed_history(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade")
    try:
        source = _make_source(project, tmp_path)
        translator = TextLength({"text": "words"})
        output = project.translate(translator, source)["output"]
        assert translator.operator_id is not None
        legacy_id = translator.operator_id
        operation = project.operation_for_artifact(output)
        assert operation is not None
        operation_id = str(operation["operation_id"])
        project.add_operator_alias(legacy_id, "lengths")

        artifact_dir = project.storage.artifact_dir(output.artifact_id)
        artifact_before = _tree_bytes(artifact_dir)
        legacy_descriptor_path = project.storage.operator_descriptor_path(legacy_id)
        _downgrade_operator_to_v1(project, legacy_id)
        legacy_descriptor_before = legacy_descriptor_path.read_bytes()

        with pytest.raises(Exception, match="pre-strict legacy snapshot"):
            project.pipeline(start=source, end=output)

        report = project.upgrade_legacy_operators()
        item = report["items"][0]
        assert item["status"] == "upgraded"
        strict_id = str(item["strict_operator_id"])
        assert strict_id != legacy_id

        assert project.get_operation(operation_id)["operator_id"] == strict_id
        assert project.resolve_operator_id("lengths") == strict_id
        assert project.catalog.operator_reuse_status(strict_id) == "reusable"

        operation_descriptor = json.loads(
            project.storage.operation_descriptor_path(operation_id).read_text(
                encoding="utf-8"
            )
        )
        assert operation_descriptor["operator_id"] == strict_id
        assert operation_descriptor["legacy_operator_upgrade"] == {
            "legacy_operator_id": legacy_id,
            "strict_operator_id": strict_id,
        }
        assert project.catalog.legacy_operation_rebinds(
            descriptor_synced=True
        )[0]["operation_id"] == operation_id

        # Existing scientific outputs and the historical v1 snapshot are untouched.
        assert _tree_bytes(artifact_dir) == artifact_before
        assert legacy_descriptor_path.read_bytes() == legacy_descriptor_before

        recovered = project.pipeline(start=source, end=output)
        assert [stage.translator.operator_id for stage in recovered.stages] == [
            strict_id
        ]

        operator_ids_after_first = {
            str(row["operator_id"]) for row in project.list_operators()
        }
        second = project.upgrade_legacy_operators()
        assert {
            str(row["operator_id"]) for row in project.list_operators()
        } == operator_ids_after_first
        assert project.get_operation(operation_id)["operator_id"] == strict_id
        assert second["items"][0]["strict_operator_id"] == strict_id
        assert second["items"][0]["status"] == "already_upgraded"
    finally:
        project.close()


def test_project_legacy_upgrade_skips_historical_only_without_blocking_exact(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade_mixed")
    try:
        source = _make_source(project, tmp_path)
        translator = TextLength({"text": "words"})
        output = project.translate(translator, source)["output"]
        assert translator.operator_id is not None
        exact_id = translator.operator_id
        _downgrade_operator_to_v1(project, exact_id)

        historical_id = "optr_999999"
        historical_dir = project.storage.operator_dir(historical_id)
        (historical_dir / "assets").mkdir(parents=True)
        (historical_dir / "operator.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "operator_id": historical_id,
                    "operation_type": "translate",
                    "class": {
                        "module": "text_analysis_lab.translators.spacy_translator",
                        "qualname": "SpacyTranslator",
                    },
                    "json_state": {"model": "en_core_web_sm"},
                    "assets": {},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        project.catalog.register_operator(
            operator_id=historical_id,
            operation_type="translate",
            snapshot_status="serialized",
            reuse_status="legacy_unknown",
        )

        report = project.upgrade_legacy_operators()
        by_id = {item["legacy_operator_id"]: item for item in report["items"]}

        assert by_id[exact_id]["status"] == "upgraded"
        assert by_id[historical_id]["status"] == "historical_only"
        with pytest.raises(OperatorError, match="pre-strict legacy snapshot"):
            project.get_operator(historical_id)

        recovered = project.pipeline(start=source, end=output)
        assert len(recovered.stages) == 1
    finally:
        project.close()


def test_project_legacy_upgrade_recovers_unsynced_operation_descriptor(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade_recovery")
    try:
        source = _make_source(project, tmp_path)
        translator = TextLength({"text": "words"})
        output = project.translate(translator, source)["output"]
        assert translator.operator_id is not None
        legacy_id = translator.operator_id
        operation = project.operation_for_artifact(output)
        assert operation is not None
        operation_id = str(operation["operation_id"])
        _downgrade_operator_to_v1(project, legacy_id)

        strict = project.migrate_legacy_operator(legacy_id)
        assert strict.operator_id is not None
        strict_id = strict.operator_id
        project.catalog.apply_legacy_operator_upgrade(
            legacy_operator_id=legacy_id,
            strict_operator_id=strict_id,
            operation_ids=[operation_id],
        )

        # Simulate interruption after the catalog transaction but before descriptor sync.
        descriptor = json.loads(
            project.storage.operation_descriptor_path(operation_id).read_text(
                encoding="utf-8"
            )
        )
        assert descriptor["operator_id"] == legacy_id
        assert project.get_operation(operation_id)["operator_id"] == strict_id

        report = project.upgrade_legacy_operators()
        assert report["recovered_descriptor_operations"] == [operation_id]

        descriptor = json.loads(
            project.storage.operation_descriptor_path(operation_id).read_text(
                encoding="utf-8"
            )
        )
        assert descriptor["operator_id"] == strict_id
        assert project.catalog.legacy_operation_rebinds(
            descriptor_synced=False
        ) == []
    finally:
        project.close()


def test_project_legacy_upgrade_does_not_guess_between_prior_migrations(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade_ambiguous")
    try:
        source = _make_source(project, tmp_path)
        translator = TextLength({"text": "words"})
        output = project.translate(translator, source)["output"]
        assert translator.operator_id is not None
        legacy_id = translator.operator_id
        operation = project.operation_for_artifact(output)
        assert operation is not None
        operation_id = str(operation["operation_id"])
        _downgrade_operator_to_v1(project, legacy_id)

        first = project.migrate_legacy_operator(legacy_id)
        second = project.migrate_legacy_operator(legacy_id)
        assert first.operator_id is not None
        assert second.operator_id is not None
        assert first.operator_id != second.operator_id

        report = project.upgrade_legacy_operators()
        item = report["items"][0]
        assert item["status"] == "ambiguous_prior_migrations"
        assert first.operator_id in str(item["reason"])
        assert second.operator_id in str(item["reason"])
        assert project.get_operation(operation_id)["operator_id"] == legacy_id
        assert project.catalog.legacy_operator_upgrade(legacy_id) is None
    finally:
        project.close()


def test_project_legacy_upgrade_leaves_incomplete_operations_on_legacy_operator(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_upgrade_incomplete")
    try:
        legacy_id = "optr_999998"
        legacy_dir = project.storage.operator_dir(legacy_id)
        (legacy_dir / "assets").mkdir(parents=True)

        legacy = TextLength({"text": "words"})
        legacy.assign_operator_id(legacy_id)
        descriptor = legacy.to_descriptor()
        descriptor["schema_version"] = 1
        descriptor["assets"] = {}
        (legacy_dir / "operator.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        project.catalog.register_operator(
            operator_id=legacy_id,
            operation_type="translate",
            snapshot_status="serialized",
            reuse_status="legacy_unknown",
        )
        operation_id = "oper_incomplete"
        project.catalog.register_operation(
            operation_id=operation_id,
            operation_type="translate",
            operator_id=legacy_id,
            status="incomplete",
        )

        report = project.upgrade_legacy_operators()
        item = report["items"][0]
        assert item["status"] == "upgraded"
        assert item["complete_operations"] == []
        assert item["incomplete_operations"] == [operation_id]
        assert project.get_operation(operation_id)["operator_id"] == legacy_id
    finally:
        project.close()
