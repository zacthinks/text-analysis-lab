"""R0b: executing-interpreter provenance and guarded environment selection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from text_analysis_lab import Pipeline, Project
from text_analysis_lab.core import environments as env_module
from text_analysis_lab.core.environments import (
    EnvironmentNotFoundError,
    EnvironmentRegistryError,
    fingerprint_observation,
)
from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.translators import FeatureMatrix, TextLength


def _seed(project: Project, tmp_path: Path):
    source_path = tmp_path / "docs.csv"
    pd.DataFrame({"text": ["alpha beta", "gamma delta", "epsilon"]}).to_csv(
        source_path, index=False
    )
    return project.read_csv(source_path, text_fields="text", metadata_fields=None)


def _alternate_environment(project: Project) -> str:
    record = copy.deepcopy(project.environments.default["observation"])
    record["distributions"].append({"name": "separate-r0b-python-package", "version": "1.0"})
    fingerprint = fingerprint_observation(record)
    return project.catalog.register_environment(
        fingerprint=fingerprint,
        schema_version=record["schema_version"],
        observation_json=json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _verify(project: Project, artifact, expected: str) -> dict:
    operation = project.operation_for_artifact(artifact)
    assert operation is not None
    assert operation["environment_id"] == expected
    assert operation["platform_id"] == project.platforms.opening_id
    assert project.get_operation(operation["operation_id"])["environment_id"] == expected
    assert project.get_operation(operation["operation_id"])["platform_id"] == project.platforms.opening_id
    assert project.list_operations()[-1]["environment_id"] == expected
    descriptor = json.loads(
        project.storage.operation_descriptor_path(operation["operation_id"]).read_text(
            encoding="utf-8"
        )
    )
    assert descriptor["environment_id"] == expected
    assert descriptor["platform_id"] == project.platforms.opening_id
    return descriptor


def test_import_translation_and_structural_operations_record_actual_executor(tmp_path):
    with Project.create(tmp_path / "project", "execution-history") as project:
        local = project.environments.opening_id
        source = _seed(project, tmp_path)
        _verify(project, source, local)

        output = project.translate(
            TextLength({"text": "words"}), source, environment=local,
        )["output"]
        metadata = _verify(project, output, local)
        assert "environment" not in metadata["request"]["params"]

        sampled = project.sample(source, n=2, random_state=5)
        _verify(project, sampled, local)
        restricted = project.restrict(source, to=sampled)
        _verify(project, restricted, local)

        # The active default is not the actual executor of local operations.
        other = _alternate_environment(project)
        project.environments.set_default(other)
        another = project.sample(source, n=1, random_state=1)
        _verify(project, another, local)
        assert project.environments.default["environment_id"] == other

        # Explicit local override does not mutate active default.
        explicit = project.translate(
            TextLength({"text": "characters"}), source, environment=local
        )["output"]
        _verify(project, explicit, local)
        assert project.environments.default["environment_id"] == other


def test_unknown_or_noncurrent_selection_fails_without_persistent_mutation(tmp_path):
    with Project.create(tmp_path / "project", "guard") as project:
        source = _seed(project, tmp_path)
        other = _alternate_environment(project)
        baseline = project.storage.manifest_path.read_bytes()
        before_ops = project.list_operations()
        before_operators = project.list_operators()
        before_artifacts = project.list_artifacts()

        for selector, expected_error, message in (
            ("env_not_registered", EnvironmentNotFoundError, "registered"),
            (other, EnvironmentRegistryError, "no bound Python interpreter"),
        ):
            with pytest.raises(expected_error, match=message):
                project.translate(
                    TextLength({"text": "words"}), source,
                    environment=selector, alias="forbidden",
                )
            assert project.storage.manifest_path.read_bytes() == baseline
            assert project.list_operations() == before_ops
            assert project.list_operators() == before_operators
            assert project.list_artifacts() == before_artifacts
            assert not any("forbidden" in str(row.get("aliases", ())) for row in project.list_artifacts())

        project.environments.set_default(other)
        with pytest.raises(EnvironmentRegistryError, match="no bound Python interpreter"):
            project.translate(TextLength({"text": "words"}), source)
        assert project.storage.manifest_path.read_bytes() == baseline
        assert project.list_operations() == before_ops
        assert project.environments.default["environment_id"] == other


def test_feature_matrix_alternate_selection_fails_before_fit_or_ids(tmp_path):
    with Project.create(tmp_path / "project", "feature-guard") as project:
        source = _seed(project, tmp_path)
        other = _alternate_environment(project)
        translator = FeatureMatrix(data_columns=False, metadata_columns=False)
        before = project.storage.manifest_path.read_bytes()
        with pytest.raises(OperatorError, match="only a new built-in TextLength"):
            project.translate(translator, source, environment=other)
        assert translator.operator_id is None
        assert project.storage.manifest_path.read_bytes() == before
        assert len(project.list_operations()) == 1


def test_pipeline_default_guard_precedes_stage_preparation(tmp_path):
    with Project.create(tmp_path / "project", "pipeline-guard") as project:
        source = _seed(project, tmp_path)
        other = _alternate_environment(project)
        pipeline = Pipeline()
        stage = pipeline.add(
            "lengths", TextLength({"text": "words"}), source=pipeline.input
        )
        pipeline.output("output", stage["output"])
        before = project.storage.manifest_path.read_bytes()
        project.environments.set_default(other)
        with pytest.raises(EnvironmentRegistryError, match="Cross-interpreter"):
            project.run_pipeline(pipeline, source)
        assert project.storage.manifest_path.read_bytes() == before
        assert len(project.list_operations()) == 1


def test_failed_and_incomplete_operation_preserve_actual_executor(tmp_path, monkeypatch):
    from text_analysis_lab.core import translate as translation

    with Project.create(tmp_path / "project", "failure") as project:
        source = _seed(project, tmp_path)
        local = project.environments.opening_id

        def fail_run(**kwargs):
            raise RuntimeError("deliberate execution failure")

        monkeypatch.setattr(translation, "_run_repeated_plan", fail_run)
        with pytest.raises(RuntimeError, match="deliberate execution failure"):
            project.translate(TextLength({"text": "words"}), source)

        failed = project.list_operations(status="failed")
        assert len(failed) == 1
        operation = failed[0]
        assert operation["environment_id"] == local
        metadata = json.loads(
            project.storage.operation_descriptor_path(
                operation["operation_id"]
            ).read_text(encoding="utf-8")
        )
        assert metadata["environment_id"] == local


def test_resume_rejects_mismatched_registered_executor_and_preserves_history(tmp_path):
    with Project.create(tmp_path / "project", "resume") as project:
        a = project.environments.opening_id
        b = _alternate_environment(project)
        project.catalog.register_operator(
            operator_id="optr_test_resume", operation_type="translate"
        )
        project.catalog.register_operation(
            operation_id="oper_test_mismatch", operation_type="translate",
            operator_id="optr_test_resume", environment_id=b,
        )
        before = project.get_operation("oper_test_mismatch")
        with pytest.raises(OperatorError, match="requires an existing operation descriptor"):
            project.resume_operation("oper_test_mismatch")
        assert project.get_operation("oper_test_mismatch") == before

        # Historical unassigned provenance remains unassigned, not inferred as A.
        project.catalog.register_operation(
            operation_id="oper_legacy_null", operation_type="translate",
            operator_id="optr_test_resume",
        )
        with pytest.raises(OperatorError, match="no operation descriptor"):
            project.resume_operation("oper_legacy_null")
        assert project.get_operation("oper_legacy_null")["environment_id"] is None
        assert project.environments.opening_id == a


def test_historical_registry_record_verification_not_tied_to_current_producer(tmp_path, monkeypatch):
    with Project.create(tmp_path / "project", "historic-schema") as project:
        old_id = project.environments.opening_id
        old_obs = project.environments.get(old_id)["observation"]
        monkeypatch.setattr(env_module, "OBSERVATION_SCHEMA_VERSION", 3)
        assert project.environments.get(old_id)["observation"] == old_obs
        with pytest.raises(EnvironmentRegistryError, match="Unsupported"):
            fingerprint_observation(old_obs)
