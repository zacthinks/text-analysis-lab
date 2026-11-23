"""Real independent-interpreter acceptance for the R1 TextLength-only pilot.

The secondary is an actual isolated venv with a distinct installed package
inventory. Test dependencies are shared via PYTHONPATH solely to avoid
reinstalling unrelated packages; TeAL never provisions environments.
Both interpreters use the same Python minor unless separately tested.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
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
from text_analysis_lab.core import environments, platforms
from text_analysis_lab.core.environments import EnvironmentRegistryError
from text_analysis_lab.core.secondary_coordinator import (
    PROTOCOL_VERSION,
    SecondaryCoordinatorError,
    SecondaryOperationError,
    _check_probe,
    _pilot_code_digest,
    _json,
    _read_json,
    _run,
)
from text_analysis_lab.translators import FeatureMatrix, TextLength


@pytest.fixture
def two_interpreters(tmp_path, monkeypatch):
    directory = tmp_path / "other-venv"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    executable = (
        directory / "Scripts" / "python.exe"
        if sys.platform == "win32" else directory / "bin" / "python"
    )
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


@pytest.mark.acceptance
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
        assert operation["platform_id"] in {
            row["platform_id"] for row in project.platforms.list()
        }
        assert len(project.execution_attempts(operation["operation_id"])) == 1
        attempt = project.execution_attempts(operation["operation_id"])[0]
        assert attempt["environment_id"] == b
        assert attempt["platform_id"] == operation["platform_id"]
        assert attempt["outcome"] == "complete"
        assert operation["coordinator_request_id"]
        descriptor = json.loads(
            project.storage.operation_descriptor_path(operation["operation_id"]).read_text()
        )
        assert descriptor["environment_id"] == b
        assert descriptor["platform_id"] == operation["platform_id"]
        assert descriptor["coordinator_request_id"] == operation["coordinator_request_id"]
        assert project.catalog.operation_outputs(operation["operation_id"])[0][
            "artifact_id"
        ] == output.artifact_id



def test_remote_alias_reuse_and_overwrite_preserve_idempotence(two_interpreters):
    """A reused alias is not a new coordinator request or scientific operation."""
    _, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        before = len(project.list_operations())
        first = project.translate(
            TextLength({"text": "words"}), source_id,
            environment=b, alias="lengths",
        )["output"]
        original_operation = project.operation_for_artifact(first)
        assert original_operation["environment_id"] == b
        assert original_operation["coordinator_request_id"]
        assert len(project.list_operations()) == before + 1
        original_attempts = project.execution_attempts(original_operation["operation_id"])

        # Existing TeAL semantics ignore new scientific parameters on reuse.
        # The new request must NOT be misidentified as a new operation.
        reused = project.translate(
            TextLength({"text": "characters"}), source_id,
            environment=b, alias="lengths", overwrite=False,
        )["output"]
        assert reused.artifact_id == first.artifact_id
        assert project.resolve_artifact_id("lengths") == first.artifact_id
        assert len(project.list_operations()) == before + 1
        assert project.get_operation(original_operation["operation_id"]) == original_operation
        assert project.execution_attempts(original_operation["operation_id"]) == original_attempts
        assert reused.query(
            data_columns=False, metadata_columns=["text_words"],
            metadata_mode="local", form="table",
        )["text_words"].tolist() == [2, 3, 1, 2, 1]

        replacement = project.translate(
            TextLength({"text": "characters"}), source_id,
            environment=b, alias="lengths", overwrite=True,
        )["output"]
        assert replacement.artifact_id != first.artifact_id
        assert project.resolve_artifact_id("lengths") == replacement.artifact_id
        assert len(project.list_operations()) == before + 2
        replacement_operation = project.operation_for_artifact(replacement)
        assert replacement_operation["operation_id"] != original_operation["operation_id"]
        assert replacement_operation["coordinator_request_id"] != (
            original_operation["coordinator_request_id"]
        )
        assert replacement_operation["environment_id"] == b
        assert project.get_artifact(first.artifact_id, include_deleted=True).artifact_id == first.artifact_id
        assert any(
            row["artifact_id"] == first.artifact_id and row["deleted"]
            for row in project.list_artifacts(include_deleted=True)
        )
        assert replacement.query(
            data_columns=False, metadata_columns=["text_characters"],
            metadata_mode="local", form="table",
        )["text_characters"].tolist() == [10, 13, 4, 8, 5]
        assert project.environments.default["environment_id"] == a


def test_remote_alias_reuse_may_return_existing_local_artifact(two_interpreters):
    """Alias reuse is about existing artifacts, not the prior executor identity."""
    _, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        local = project.translate(
            TextLength({"text": "words"}), source_id,
            environment=a, alias="local_lengths",
        )["output"]
        before = len(project.list_operations())
        remote_reuse = project.translate(
            TextLength({"text": "characters"}), source_id,
            environment=b, alias="local_lengths",
        )["output"]
        assert remote_reuse.artifact_id == local.artifact_id
        assert len(project.list_operations()) == before
        assert project.operation_for_artifact(remote_reuse)["environment_id"] == a


def test_remote_alias_mapping_is_rejected_before_dispatch(two_interpreters, monkeypatch):
    from text_analysis_lab.core import secondary_coordinator as coordinator

    _, source_id, _, _, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        before = project.list_operations()
        def no_process(*args, **kwargs):
            raise AssertionError("Invalid alias must not launch a subprocess.")
        monkeypatch.setattr(coordinator, "_verified_executable", no_process)
        with pytest.raises(SecondaryCoordinatorError, match="nonempty string alias"):
            project.translate(
                TextLength({"text": "words"}), source_id,
                environment=b, alias={"output": "lengths"},
            )
        assert project.list_operations() == before


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
        exit_code, fake, stderr = _run(
            project.environments.bound_interpreter(b), "--probe", timeout=60,
        )
        assert exit_code == 0, stderr
        assert fake is not None
        fake["protocol_version"] = PROTOCOL_VERSION + 1
        with pytest.raises(SecondaryCoordinatorError, match="protocol version"):
            _check_probe(project, b, fake)
        fake["protocol_version"] = PROTOCOL_VERSION
        fake["teal_version"] = "0.0.incompatible"
        with pytest.raises(SecondaryCoordinatorError, match="TeAL distribution version"):
            _check_probe(project, b, fake)
        assert project.list_operations() == before


def test_pre_operation_failure_and_child_registered_failure(two_interpreters, tmp_path, monkeypatch):
    project_path, source_id, _, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        count = len(project.list_operations())
        # Invalid TextLength field fails before the child's operation allocation.
        with pytest.raises(SecondaryOperationError) as error:
            project.translate(TextLength({"missing_field": "words"}), source_id, environment=b)
        assert error.value.operation_id is None
        assert len(project.list_operations()) == count

        # Fault injection lives *only* in the independent child's test
        # interpreter startup. It fails inside the ordinary translation runner,
        # after catalog registration and initial durable writer checkpoint.
        # Unlike corrupting source files, this cannot fail during preflight.
        hook_dir = tmp_path / "fault-injection"
        hook_dir.mkdir()
        (hook_dir / "sitecustomize.py").write_text(
            "import os\n"
            "if os.environ.get('TEAL_R1_TEST_FAIL_AFTER_REGISTRATION') == '1':\n"
            "    import text_analysis_lab.core.translate as execution\n"
            "    def fail_registered_operation(**kwargs):\n"
            "        raise RuntimeError('R1 injected post-registration failure')\n"
            "    execution._run_repeated_plan = fail_registered_operation\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join([str(hook_dir), os.environ["PYTHONPATH"]]),
        )
        monkeypatch.setenv("TEAL_R1_TEST_FAIL_AFTER_REGISTRATION", "1")
        try:
            with pytest.raises(SecondaryOperationError) as error:
                project.translate(
                    TextLength({"text": "words"}), source_id, environment=b,
                )
        finally:
            monkeypatch.delenv("TEAL_R1_TEST_FAIL_AFTER_REGISTRATION")
        assert error.value.request_id
        failed = project.catalog.operation_for_coordinator_request(error.value.request_id)
        assert failed is not None
        assert failed["environment_id"] == b
        assert failed["status"] in {"failed", "incomplete"}
        assert len(project.list_operations()) == count + 1

        # Explicit recovery goes to B, based on recorded provenance.
        restored = project.resume_operation(failed["operation_id"])["output"]
        assert restored.status == "complete"
        assert project.operation_for_artifact(restored)["environment_id"] == b
        history = project.execution_attempts(failed["operation_id"])
        assert len(history) == 2
        assert [attempt["outcome"] for attempt in history] == ["failed", "complete"]
        assert all(attempt["environment_id"] == b for attempt in history)
        assert history[-1]["platform_id"] == failed["platform_id"]
        with Project.open(project_path) as unbound:
            with pytest.raises(EnvironmentRegistryError, match="no bound"):
                unbound.resume_operation(failed["operation_id"])


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
        # Enough small batches to reliably interrupt *after* the first
        # checkpoint, not just during secondary process startup.
        csv_path = tmp_path / "interrupt_source.csv"
        pd.DataFrame({"text": [f"sample row {i}" for i in range(150)]}).to_csv(
            csv_path, index=False
        )
        source_id = project.read_csv(
            csv_path, text_fields="text", metadata_fields=None,
        ).artifact_id
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
            ready = False
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and child.poll() is None:
                operation = project.catalog.operation_for_coordinator_request(request_id)
                if operation is not None:
                    checkpoint = (
                        project.storage.operation_temp_dir(operation["operation_id"])
                        / "pass.json"
                    )
                    if checkpoint.exists():
                        ready = True
                        break
                time.sleep(0.005)
            assert ready, "Child never exposed a recoverable operation checkpoint."
            child.kill()
            child.wait(timeout=15)
            operation_id = operation["operation_id"]
            observed = project.get_operation(operation_id)
            assert observed["coordinator_request_id"] == request_id
            assert observed["environment_id"] == b
            assert observed["status"] != "complete"
            with Project.open(project_path) as different:
                with pytest.raises(EnvironmentRegistryError, match="no bound"):
                    different.resume_operation(operation_id)
            output = project.resume_operation(operation_id)["output"]
            assert output.status == "complete"
            assert output.n_rows == 150
            assert project.get_operation(operation_id)["environment_id"] == b
            history = project.execution_attempts(operation_id)
            assert len(history) == 2
            assert [attempt["outcome"] for attempt in history] == ["uncertain", "complete"]
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)


def test_secondary_probe_evidence_and_code_compatibility(two_interpreters):
    """An ID string, TeAL version, or protocol number alone never authorizes B."""
    _, source_id, executable, a, b, _ = two_interpreters
    with _open_bound(two_interpreters) as project:
        code, payload, stderr = _run(executable, "--probe", timeout=60)
        assert code == 0, stderr
        assert payload is not None
        _check_probe(project, b, payload)
        spoofed = copy.deepcopy(payload)
        spoofed["environment_id"] = a
        with pytest.raises(EnvironmentRegistryError, match="claimed"):
            _check_probe(project, b, spoofed)
        changed = copy.deepcopy(payload)
        changed["observation"]["distributions"].append(
            {"name": "unrecorded-b-package", "version": "1.0"}
        )
        with pytest.raises(EnvironmentRegistryError, match="claimed"):
            _check_probe(project, b, changed)
        incompatible = copy.deepcopy(payload)
        incompatible["pilot_code_sha256"] = "0" * 64
        with pytest.raises(SecondaryCoordinatorError, match="implementation/source"):
            _check_probe(project, b, incompatible)
        assert project.catalog.operation_for_coordinator_request("missing") is None


@pytest.mark.acceptance
def test_portable_child_adopts_verified_realization_with_distinct_platform(
    two_interpreters, monkeypatch, tmp_path,
):
    """Real B subprocess and installed inventory; source/host metadata controlled.

    Test environments share editable TeAL dependencies via PYTHONPATH. The
    fixture supplies a *synthetic* immutable source proof, so this exercises
    coordination logic, NOT independent artifact/source-byte verification or
    genuine different-hardware execution.
    """
    project_path, source_id, executable, a, b, _ = two_interpreters
    hook_dir = tmp_path / "controlled-probe"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(
        "import copy\n"
        "from text_analysis_lab.core import environments, platforms\n"
        "_original_software = environments.observe_environment\n"
        "_original_platform = platforms.observe_platform\n"
        "def controlled_software():\n"
        "    result = copy.deepcopy(_original_software())\n"
        "    for dist in result['distributions']:\n"
        "        dist.pop('editable', None)\n"
        "        dist.pop('local_install', None)\n"
        "        if dist['name'] == 'text-analysis-lab':\n"
        "            dist['vcs_commit'] = 'a' * 40\n"
        "    return result\n"
        "def controlled_platform():\n"
        "    result = copy.deepcopy(_original_platform())\n"
        "    result['os']['release'] += '-simulated-secondary'\n"
        "    return result\n"
        "environments.observe_environment = controlled_software\n"
        "platforms.observe_platform = controlled_platform\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(hook_dir), os.environ["PYTHONPATH"]]),
    )
    exit_code, probe, stderr = _run(executable, "--probe", timeout=60)
    assert exit_code == 0, stderr
    assert probe is not None
    observed = probe["observation"]
    assert observed["kind"] == "observed_unlocked"
    assert any(
        dist.get("vcs_commit") == "a" * 40
        for dist in observed["distributions"]
        if dist["name"] == "text-analysis-lab"
    )
    manifest = {
        "realizations": [{"distributions": copy.deepcopy(observed["distributions"])}],
    }
    canonical = environments._normalize_portable_manifest(manifest)
    manifest_digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    spec = {
        "schema_version": 2,
        "kind": "portable_locked",
        "python": observed["python"],
        "teal": {
            "version": observed["teal"]["version"],
            "source": {"vcs": "git", "commit": "a" * 40},
        },
        "extras": [],
        "lock": {"kind": "uv", "manifest": manifest, "sha256": manifest_digest},
    }
    with Project.open(project_path) as project:
        portable = project.environments.register_portable_specification(spec)
        project.environments.bind_interpreter(portable, executable)
        assert project.environments.opening_id == a
        assert project.environments.get(portable)["readiness"] == "unverified"
        output = project.translate(
            TextLength({"text": "words"}), source_id, environment=portable,
            workers=1, batch_size=2,
        )["output"]
        operation = project.operation_for_artifact(output)
        assert operation["environment_id"] == portable
        assert operation["platform_id"] != project.platforms.opening_id
        descriptor = json.loads(
            project.storage.operation_descriptor_path(
                operation["operation_id"]
            ).read_text(encoding="utf-8")
        )
        assert descriptor["platform_id"] == operation["platform_id"]
        history = project.execution_attempts(operation["operation_id"])
        assert len(history) == 1
        assert history[0]["platform_id"] == operation["platform_id"]
        assert history[0]["environment_id"] == portable
        assert history[0]["outcome"] == "complete"

    # The verified association is durable, and can be reopened by the child.
    reopened = subprocess.run(
        [str(executable), "-c",
         "import sys; from text_analysis_lab import Project; "
         "p=Project.open(sys.argv[1]); print(p.environments.opening_id); p.close()",
         str(project_path)],
        text=True, capture_output=True, timeout=90, check=False,
    )
    assert reopened.returncode == 0, reopened.stderr
    assert reopened.stdout.strip().splitlines()[-1] == portable
