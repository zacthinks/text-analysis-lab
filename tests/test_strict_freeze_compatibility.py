from __future__ import annotations

import json

import numpy as np
import pandas as pd

import text_analysis_lab as teal
from text_analysis_lab.translators import FittedPredictor, FunctionMapper, TextLength


class _Predictor:
    def predict(self, values):
        return np.zeros(len(values), dtype=int)


def test_generic_wrappers_require_explicit_persistence_for_reuse() -> None:
    mapper = FunctionMapper(lambda packet: packet)
    mapper_caps = mapper.execution_capabilities()
    assert not mapper_caps.reusable
    assert not mapper.supports_resume(mode="translate", route="sequential")

    saved_mapper = FunctionMapper(lambda packet: packet, save_function=True)
    saved_mapper_caps = saved_mapper.execution_capabilities()
    assert saved_mapper_caps.reusable
    assert any("user-asserted" in reason for reason in saved_mapper_caps.reasons)
    assert saved_mapper.supports_resume(mode="translate", route="sequential")

    predictor = FittedPredictor(_Predictor())
    predictor_caps = predictor.execution_capabilities()
    assert not predictor_caps.reusable
    assert not predictor.supports_resume(mode="translate", route="sequential")

    saved_predictor = FittedPredictor(_Predictor(), save_model=True)
    saved_predictor_caps = saved_predictor.execution_capabilities()
    assert saved_predictor_caps.reusable
    assert any("user-asserted" in reason for reason in saved_predictor_caps.reasons)
    assert saved_predictor.supports_resume(mode="translate", route="sequential")


def test_v1_text_length_migration_creates_new_v2_operator(tmp_path) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_migration")
    try:
        legacy_id = "optr_legacy"
        legacy_dir = project.storage.operator_dir(legacy_id)
        (legacy_dir / "assets").mkdir(parents=True, exist_ok=True)

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

        assert project.legacy_operator_status(legacy_id) == "exactly_migratable"

        migrated = project.migrate_legacy_operator(legacy_id)
        assert isinstance(migrated, TextLength)
        assert migrated.operator_id is not None
        assert migrated.operator_id != legacy_id
        assert migrated.lengths == legacy.lengths

        legacy_descriptor = json.loads(
            (legacy_dir / "operator.json").read_text(encoding="utf-8")
        )
        assert legacy_descriptor["schema_version"] == 1

        migrated_dir = project.storage.operator_dir(migrated.operator_id)
        migrated_descriptor = json.loads(
            (migrated_dir / "operator.json").read_text(encoding="utf-8")
        )
        assert migrated_descriptor["schema_version"] == 2
        migration = json.loads(
            (migrated_dir / "legacy_migration.json").read_text(encoding="utf-8")
        )
        assert migration["migrated_from_operator_id"] == legacy_id
        assert project.catalog.operator_reuse_status(migrated.operator_id) == "reusable"
    finally:
        project.close()


def test_unknown_v1_operator_remains_historical_only(tmp_path) -> None:
    project = teal.Project.create(tmp_path / "project", name="legacy_historical")
    try:
        legacy_id = "optr_legacy"
        legacy_dir = project.storage.operator_dir(legacy_id)
        (legacy_dir / "assets").mkdir(parents=True, exist_ok=True)

        descriptor = {
            "schema_version": 1,
            "operator_id": legacy_id,
            "operation_type": "translate",
            "class": {
                "module": "text_analysis_lab.translators.spacy_translator",
                "qualname": "SpacyTranslator",
            },
            "json_state": {"model": "en_core_web_sm"},
            "assets": {},
        }
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

        assert project.legacy_operator_status(legacy_id) == "historical_only"
    finally:
        project.close()
