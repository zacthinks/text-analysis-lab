"""Real independent-interpreter acceptance for the R1 TextLength-only pilot.

The secondary is an actual isolated venv with a distinct installed package
inventory. On POSIX CI it shares test dependencies via PYTHONPATH solely to
avoid reinstalling unrelated packages; TeAL never provisions environments.
Both interpreters use the same Python minor unless separately tested.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

import pandas as pd
import pytest

from text_analysis_lab import Project
from text_analysis_lab.core.environments import EnvironmentRegistryError
from text_analysis_lab.core.secondary_coordinator import (
    PROTOCOL_VERSION,
    SecondaryCoordinatorError,
    SecondaryOperationError,
    _check_probe,
    _json,
    _read_json,
    _run,
)
from text_analysis_lab.translators import FeatureMatrix, TextLength


@pytest.fixture
def two_interpreters(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX dependency-sharing fixture; Windows tested by Acceptance suite")
    directory = tmp_path / "other-venv"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    executable = directory / "bin" / "python"
    result = subprocess.run(
        [str(executable), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    site_dir = Path(result.stdout.strip())
    marker = site_dir / "r1_environment_probe-1.0.dist-info"
    marker.mkdir()
    (marker / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: r1-environment-probe\nVersion: 1.0\n",
        encoding="utf-8",
    )

    # Share the ordinary test dependencies without pretending the two venv
    # inventories are identical. Site-package discovery still sees the probe.
    source_root = Path(__file__).resolve().parents[1] / "src"
    dependencies = [
        path for path in sys.path if path and Path(path).is_dir()
        and ("site-packages" in Path(path).parts or "dist-packages" in Path(path).parts)
    ]
    assert dependencies
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(source_root), *dependencies]),
    )
    project_path = tmp_path / "project"
    with Project.create(project_path, "r1-test") as project:
        csv_path = tmp_path / "source.csv"
        pd.DataFrame({
            "text": ["alpha beta", "one two three", "four", "five six", "seven"],
        }).to_csv(csv_path, index=False)
        source = project.read_csv(csv_path, text_fields="text", metadata_fields=None)
        a = project.environments.opening_id
        source_id = source.artifact_id

    # Open the project once under the separate interpreter to register the
    # genuine observed B environment through the normal R0a path.
    child = subprocess.run(
        [str(executable), "-c",
         "import sys; from text_analysis_lab import Project; "
         "p=Project.open(sys.argv[1]); "
         "print(p.environments.opening_id); p.close()",
         str(project_path)],
        text=True, capture_output=True, timeout=90, check=False,
    )
    assert child.returncode == 0, child.stderr
    b = child.stdout.strip().splitlines()[-1]
    assert b != a
    yield project_path, source_id, executable, a, b, site_dir


def _open_bound(two_interpreters):
    project_path, _, executable, a, b, _ = two_interpreters
    project = Project.open(project_path)
    assert project.environments.opening_id == a
    project.environments.bind_interpreter(b, executable)
    return project


def test_secondary_sequential_translation_correct_artifact_and_provenance(two_interpreters):
    _, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        result = project.translate(
            TextLength({"text": "words"}), source_id,
            environment=b, batch_size=2, workers=1, alias="r1_lengths",
        )
        output = result["output"]
        assert project.environments.default["environment_id"] == a
        assert project.resolve_artifact_id("r1_lengths") == output.artifact_id
        frame = output.query(
            data_columns=False, metadata_columns=["text_words"],
            metadata_mode="local", form="table",
        )
        assert frame["text_words"].tolist() == [2, 3, 1, 2, 1]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2, 3, 4]
        assert output.descriptor["lineage"]["basis_artifact_ids"] == [source_id]
        operation = project.operation_for_artifact(output)
        assert operation is not None
        assert operation["environment_id"] == b
        assert operation["coordinator_request_id"]
        descriptor = json.loads(
            project.storage.operation_descriptor_path(operation["operation_id"]).read_text()
        )
        assert descriptor["environment_id"] == b
        assert descriptor["coordinator_request_id"] == operation["coordinator_request_id"]
        assert project.catalog.operation_outputs(operation["operation_id"])[0][
            "artifact_id"
        ] == output.artifact_id


def test_unbound_missing_fingerprint_drift_and_unsupported_requests(two_interpreters):
    _, source_id, executable, a, b, site_dir = two_interpreters
    with Project.open(two_interpreters[0]) as project:
        prior = project.list_operations()
        with pytest.raises(EnvironmentRegistryError, match="not registered"):
            project.environments.bind_interpreter("env_unregistered", executable)
        with pytest.raises(EnvironmentRegistryError, match="no bound"):
            project.translate(TextLength({"text": "words"}), source_id, environment=b)
        with pytest.raises(EnvironmentRegistryError, match="not executable"):
            project.environments.bind_interpreter(b, site_dir / "missing_python")
        project.environments.bind_interpreter(b, executable)
        with pytest.raises(SecondaryCoordinatorError, match="only a new built-in TextLength"):
            project.translate(
                FeatureMatrix(data_columns=False, metadata_columns=False),
                source_id, environment=b,
            )
        with pytest.raises(SecondaryCoordinatorError, match="does not support"):
            project.translate(
                TextLength({"text": "words"}), source_id,
                environment=b, batch_by="document",
            )
        assert project.list_operations() == prior

        # Change an installed dist-info package version in the bound B venv.
        (site_dir / "r1_environment_probe-1.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: r1-environment-probe\nVersion: 2.0\n",
            encoding="utf-8",
        )
        with pytest.raises(EnvironmentRegistryError, match="Installed packages"):
            project.translate(TextLength({"text": "words"}), source_id, environment=b)
        assert project.list_operations() == prior
        assert project.environments.default["environment_id"] == a


def test_protocol_and_teal_version_mismatch_rejected_before_execution(two_interpreters):
    _, source_id, _, _, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        from text_analysis_lab.core.secondary_coordinator import _verified_executable
        _verified_executable(project, b)
        before = project.list_operations()
        fake = {
            "protocol_version": PROTOCOL_VERSION + 1, "kind": "probe",
            "environment_id": b, "teal_version": "0.12.2",
            "observation_schema_version": 1, "python_version": [3, 11, 0],
        }
        with pytest.raises(SecondaryCoordinatorError, match="protocol version"):
            _check_probe(project, b, fake)
        fake["protocol_version"] = PROTOCOL_VERSION
        fake["teal_version"] = "0.0.incompatible"
        with pytest.raises(SecondaryCoordinatorError, match="TeAL distribution version"):
            _check_probe(project, b, fake)
        assert project.list_operations() == before


def test_pre_operation_failure_and_child_registered_failure(two_interpreters):
    project_path, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        count = len(project.list_operations())
        # Invalid TextLength field fails before the child's operation allocation.
        with pytest.raises(SecondaryOperationError) as error:
            project.translate(TextLength({"missing_field": "words"}), source_id, environment=b)
        assert error.value.operation_id is None
        assert len(project.list_operations()) == count

        # Delete a source payload part while preserving its valid descriptor;
        # input planning/reading fails within the owning secondary operation.
        source = project.get_artifact(source_id)
        part = next(source.keys_dir.glob("part-*.parquet"))
        part.unlink()
        with pytest.raises(SecondaryOperationError) as error:
            project.translate(TextLength({"text": "words"}), source_id, environment=b)
        assert error.value.request_id
        failed = project.catalog.operation_for_coordinator_request(error.value.request_id)
        assert failed is not None
        assert failed["environment_id"] == b
        assert failed["status"] in {"failed", "incomplete"}
        assert len(project.list_operations()) == count + 1


@pytest.mark.acceptance
def test_secondary_dask_parallel_keeps_child_ownership(two_interpreters):
    _, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        output = project.translate(
            TextLength({"text": "characters"}), source_id,
            environment=b, workers=2, batch_size=1,
            max_outstanding_units=2,
        )["output"]
        operation = project.operation_for_artifact(output)
        assert operation["environment_id"] == b
        descriptor = json.loads(
            project.storage.operation_descriptor_path(operation["operation_id"]).read_text()
        )
        assert descriptor["route"] == "parallel"
        assert descriptor["effective_workers"] == 2
        frame = output.query(
            data_columns=False, metadata_columns=["text_characters"],
            metadata_mode="local", form="table",
        )
        assert frame["text_characters"].tolist() == [10, 13, 4, 8, 5]
        assert project.environments.default["environment_id"] == a


@pytest.mark.acceptance
def test_two_independent_parent_sessions_do_not_duplicate_operations(two_interpreters):
    _, source_id, _, _, b, _ = two_interpreters

    def execute(i):
        with _open_bound(two_interpreters) as project:
            result = project.translate(
                TextLength({"text": "words"}), source_id, environment=b,
                batch_size=2,
            )["output"]
            return result.artifact_id, result.descriptor["operation_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(execute, [0, 1]))
    assert len({artifact for artifact, _ in completed}) == 2
    assert len({operation for _, operation in completed}) == 2
    with Project.open(two_interpreters[0]) as project:
        for artifact_id, operation_id in completed:
            assert project.get_operation(operation_id)["environment_id"] == b
            assert project.get_artifact(artifact_id).status == "complete"


@pytest.mark.acceptance
def test_secondary_abrupt_termination_is_correlated_and_does_not_auto_retry(
    two_interpreters, tmp_path
):
    project_path, source_id, executable, _, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        # This is the public request contract exercised directly so the test
        # can deliberately terminate the process after it registers an op.
        from text_analysis_lab.core.secondary_coordinator import _validate_request
        request_id = "interrupted_r1_test"
        request = {
            "protocol_version": PROTOCOL_VERSION, "kind": "translate",
            "request_id": request_id,
            "project_path": str(project_path.resolve()), "environment_id": b,
            "translator": "TextLength",
            "scientific_state": TextLength({"text": "words"}).to_json_state(),
            "source_ids": {"source": source_id},
            "settings": {
                "workers": 1, "batch_size": 1,
                "max_outstanding_units": None, "memo": None,
                "alias": None, "overwrite": False,
            },
            "operation_id": None,
        }
        _validate_request(request)
        # Start after a legitimate source exists; poll the durable operation
        # correlation, not a guessed ID or a background process manager.
        child = subprocess.Popen(
            [str(executable), "-m", "text_analysis_lab.core.secondary_coordinator", "--run"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            child.stdin.write(_json(request))
            child.stdin.close()
            operation = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and child.poll() is None:
                operation = project.catalog.operation_for_coordinator_request(request_id)
                if operation is not None:
                    break
                time.sleep(0.005)
            if operation is None:
                pytest.skip("Child completed before interrupt could be exercised")
            child.kill()
            child.wait(timeout=15)
            operation_id = operation["operation_id"]
            observed = project.get_operation(operation_id)
            assert observed["coordinator_request_id"] == request_id
            assert observed["environment_id"] == b
            assert observed["status"] != "complete"
            # No automatic retry. If incomplete state has a plan/checkpoint,
            # researchers can resume explicitly under recorded B.
            if project.storage.operation_plan_path(operation_id).exists():
                with pytest.raises(EnvironmentRegistryError, match="no bound"):
                    with Project.open(project_path) as different:
                        different.resume_operation(operation_id)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
