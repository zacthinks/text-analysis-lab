"""R2 managed uv environment planning, lifecycle, and child binding regressions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from text_analysis_lab import Project
from text_analysis_lab.core import managed_environments as managed
from text_analysis_lab.core.environments import EnvironmentRegistryError, fingerprint_observation
from text_analysis_lab.translators.text_length import TextLength


@pytest.fixture
def managed_source(monkeypatch, tmp_path):
    monkeypatch.setenv("TEAL_MANAGED_ENV_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(managed, "_source_commit", lambda requested: "a" * 40)
    return tmp_path


def test_plan_exact_python_extras_and_gpu_policy(managed_source):
    first = managed.plan(python="3.11.15", teal_extras=["parallel", "parallel", "transformers"])
    second = managed.plan(python="3.11.15", teal_extras=["transformers", "parallel"])
    assert first == second
    assert first.python == "3.11.15"
    assert first.teal_commit == "a" * 40
    assert first.torch_required
    assert first.gpu == "auto"
    assert managed._request_key(first) == managed._request_key(second)
    assert managed.plan(teal_extras=[], gpu="auto").torch_required is False
    assert managed.plan(teal_extras=[], gpu="auto").portability.startswith("observed_unlocked")
    assert managed.plan(teal_extras=[], gpu=True).torch_required
    assert managed.plan(teal_extras=[], gpu=False).gpu == "cpu"


@pytest.mark.parametrize("python", ["3.11", "3.11.15.1", "3.10.9", "invalid"])
def test_plan_rejects_unpinned_or_incompatible_python(managed_source, python):
    with pytest.raises(EnvironmentRegistryError):
        managed.plan(python=python)


@pytest.mark.parametrize("extras", [["not-a-real-extra"], ["../../temp"], "parallel"])
def test_plan_rejects_invalid_extras(managed_source, extras):
    with pytest.raises(EnvironmentRegistryError):
        managed.plan(teal_extras=extras)


def test_plan_does_not_install_or_write_cache(managed_source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not invoke uv")
    monkeypatch.setattr(managed, "_invoke", forbidden)
    with Project.create(managed_source / "project", "project") as project:
        result = project.environments.plan(teal_extras=["parallel"])
        assert result.teal_extras == ("parallel",)
        assert not (managed_source / "cache").exists()


def test_failed_creation_does_not_register_or_publish(managed_source, monkeypatch):
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    def fail(*args, **kwargs):
        raise EnvironmentRegistryError("injected resolution failure")
    monkeypatch.setattr(managed, "_build", fail)
    with Project.create(managed_source / "project", "project") as project:
        before = len(project.environments.list())
        with pytest.raises(EnvironmentRegistryError, match="injected resolution"):
            project.environments.create(teal_extras=["parallel"], gpu=False)
        assert len(project.environments.list()) == before
        assert not any((managed_source / "cache").glob("*/managed.json"))


def test_managed_request_is_not_recorded_as_fake_portable_lock(managed_source):
    with Project.create(managed_source / "project", "project") as project:
        result = project.environments.plan()
        assert result.portability.startswith("observed_unlocked")
        assert project.environments.default["kind"] == "observed_unlocked"


@pytest.mark.acceptance
def test_real_uv_managed_base_and_r1_textlength(tmp_path, monkeypatch):
    """Opt-in network-heavy acceptance: a real immutable Git source and child.

    Set TEAL_RUN_MANAGED_INSTALL_ACCEPTANCE=1 on an online CI agent with uv.
    Never share the parent checkout through PYTHONPATH for this test.
    """
    if os.environ.get("TEAL_RUN_MANAGED_INSTALL_ACCEPTANCE") != "1":
        pytest.skip("Real managed installation requires explicit online acceptance opt-in")
    if not (Path(__file__).resolve().parents[1] / ".git").exists():
        pytest.skip("Immutable Git checkout required for source verification")
    monkeypatch.setenv("TEAL_MANAGED_ENV_ROOT", str(tmp_path / "managed"))
    with Project.create(tmp_path / "project", "project") as project:
        import pandas as pd

        csv = tmp_path / "source.csv"
        pd.DataFrame({"text": ["hello world", "another sentence"]}).to_csv(csv, index=False)
        source = project.read_csv(csv, text_fields="text", metadata_fields=None)
        result = project.environments.create(teal_extras=["parallel"], gpu="auto")
        assert result.backend == "not_applicable"
        assert result.portability.startswith("observed_unlocked")
        assert project.environments.create(teal_extras=["parallel"], gpu="auto").reused
        project.environments.set_default(result.environment_id)
        output = project.translate(TextLength({"text": "words"}), source)["output"]
        assert output.n_rows == 2
        operation = project.operation_for_artifact(output)
        assert operation["environment_id"] == result.environment_id
        assert project.execution_attempts(operation["operation_id"])[-1]["outcome"] == "complete"
        executable = project.environments.bound_interpreter(result.environment_id)
        assert executable.is_file()
        # uv entrypoints embed an absolute interpreter path; publishing the
        # cache index must not relocate that interpreter after installation.
        entrypoint = executable.parent / ("teal.exe" if os.name == "nt" else "teal")
        cli = subprocess.run(
            [str(entrypoint), "--help"], capture_output=True, text=True,
            timeout=60, check=False,
        )
        assert cli.returncode == 0, cli.stderr
        assert "Text Analysis Lab" in cli.stdout

    with Project.open(tmp_path / "project") as reopened:
        reopened.environments.set_default(result.environment_id)
        assert reopened.environments.bound_interpreter(result.environment_id) == executable
        repeat = reopened.translate(
            TextLength({"text": "words"}), source.artifact_id,
            alias="managed_lengths",
        )["output"]
        assert repeat.n_rows == 2
        reused_alias = reopened.translate(
            TextLength({"text": "words"}), source.artifact_id,
            alias="managed_lengths",
        )["output"]
        assert reused_alias.artifact_id == repeat.artifact_id
