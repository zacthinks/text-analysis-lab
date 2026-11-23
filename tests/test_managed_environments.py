"""R2 managed-environment policy and local-cache lifecycle regressions.

These tests do not download Python, Torch, or any external package.
"""
from __future__ import annotations

import json
import os
import shutil
import copy
from pathlib import Path
import sys

import pytest

from text_analysis_lab.core import managed_environments as managed
from text_analysis_lab.core.managed_environments import ManagedEnvironmentError


COMMIT = "a" * 40


@pytest.fixture
def parent_source(monkeypatch):
    monkeypatch.setattr(managed, "_local_source_commit", lambda: COMMIT)
    monkeypatch.setattr(
        managed.metadata, "metadata",
        lambda name: {"Provides-Extra": "parallel"} if False else _ExtrasMetadata(),
    )
    monkeypatch.setattr(managed.metadata, "version", lambda name: "0.12.2")


class _ExtrasMetadata:
    def get_all(self, name):
        assert name == "Provides-Extra"
        return ["parallel", "transformers", "linguistics", "geco", "all"]


def test_plan_has_no_install_side_effects(parent_source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan invoked an installation or modified the cache")
    monkeypatch.setattr(managed, "_uv", forbidden)
    first = managed.plan(teal_extras=["parallel", "parallel", "transformers"])
    second = managed.plan(teal_extras=["transformers", "parallel"])
    assert first == second
    assert first["teal_commit"] == COMMIT
    assert first["portability"].startswith("observed_unlocked")
    assert first["torch_expected"]
    assert managed.plan()["torch_expected"] is False
    assert managed.plan(gpu=True)["torch_expected"] is True
    assert first["requires_explicit_create"] is True


@pytest.mark.parametrize("kwargs", [
    {"python": "3.11"},
    {"python": "3.10.16"},
    {"python": "3.11.15.1"},
    {"teal": "v0.12.2"},
    {"teal": "b" * 40},
    {"teal_extras": ["unknown"]},
    {"gpu": "yes"},
])
def test_invalid_or_incompatible_request_rejected(parent_source, kwargs):
    with pytest.raises(ManagedEnvironmentError):
        managed.plan(**kwargs)


def test_actual_cache_reuse_and_registration_without_real_download(
    parent_source, tmp_path, monkeypatch,
):
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    invocations = []

    def fake_command(args, **kwargs):
        invocations.append(args)
        if args[:2] == ["uv", "venv"]:
            path = Path(args[2])
            exe = managed._interpreter(path)
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
        return ""
    monkeypatch.setattr(managed, "_run_command", fake_command)

    observation = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [{"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT}],
    }
    def fake_verify(directory, request, uv):
        assert managed._interpreter(directory).is_file()
        return observation, {"status": "not_applicable", "backend": None, "reason": None}

    monkeypatch.setattr(managed, "_verify", fake_verify)

    class Catalog:
        def __init__(self):
            self.rows = {}
        def register_environment(self, *, fingerprint, schema_version, observation_json):
            key = "env_" + fingerprint
            self.rows[key] = json.loads(observation_json)
            return key

    class Registry:
        def __init__(self):
            self._catalog = Catalog()
            self.bindings = {}
        def bind_interpreter(self, env, exe):
            self.bindings[env] = Path(exe)

    registry = Registry()
    first = managed.create(registry)
    second = managed.create(registry)
    assert first["environment_id"] == second["environment_id"]
    assert first["reused"] is False
    assert second["reused"] is True
    assert len([arg for arg in invocations if arg[:2] == ["uv", "pip"]]) == 1
    assert registry.bindings[first["environment_id"]].is_file()
    assert not list((tmp_path / "cache").glob("**/*.pending"))


def test_failed_install_leaves_no_ready_cache(parent_source, tmp_path, monkeypatch):
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    def fail_command(args, **kwargs):
        if args[:2] == ["uv", "pip"]:
            raise managed.ManagedEnvironmentError("Resolution failed")
        if args[:2] == ["uv", "venv"]:
            Path(args[2]).mkdir(parents=True, exist_ok=True)
        return ""
    monkeypatch.setattr(managed, "_run_command", fail_command)
    class Registry:
        def __init__(self):
            self._catalog = None
    with pytest.raises(managed.ManagedEnvironmentError, match="Resolution failed"):
        managed.create(Registry())
    assert not list((tmp_path / "cache").glob("*/managed.json"))



def test_cached_package_drift_is_rejected_without_touching_live_install(parent_source, tmp_path, monkeypatch):
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    invocations = []
    def command(args, **kwargs):
        invocations.append(args)
        if args[:2] == ["uv", "venv"]:
            executable = managed._interpreter(Path(args[2]))
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
        return ""
    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
        ],
    }
    drifted = copy.deepcopy(observation)
    drifted["distributions"].append({"name": "unplanned-package", "version": "1.0"})
    calls = 0
    def verify(*args):
        nonlocal calls
        calls += 1
        # First creation is clean; cache probe detects drift; replacement is clean.
        return (drifted if calls == 2 else observation), {
            "status": "not_applicable", "backend": None, "reason": None,
        }
    monkeypatch.setattr(managed, "_verify", verify)
    class Catalog:
        def register_environment(self, *, fingerprint, **kwargs):
            return "env_" + fingerprint
    class Registry:
        _catalog = Catalog()
        def bind_interpreter(self, env, exe):
            assert Path(exe).is_file()
    first = managed.create(Registry())
    with pytest.raises(managed.ManagedEnvironmentError, match="left untouched"):
        managed.create(Registry())
    assert len([a for a in invocations if a[:2] == ["uv", "venv"]]) == 1
    request = managed._normalized_request(
        python=None, teal=None, teal_extras=None, gpu="auto",
    )
    import hashlib
    cache_key = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert (tmp_path / "managed" / cache_key / "managed.json").is_file()




def test_install_failure_redacts_authenticated_urls(monkeypatch):
    import subprocess
    monkeypatch.setattr(
        managed.subprocess, "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, returncode=1, stdout="",
            stderr="Could not fetch https://secret-token@example.com/simple?token=private-value",
        ),
    )
    with pytest.raises(managed.ManagedEnvironmentError) as caught:
        managed._run_command(["uv", "pip", "install", "test"])
    message = str(caught.value)
    assert "secret-token" not in message
    assert "private-value" not in message
    assert "[redacted]" in message


def test_strict_gpu_never_reports_cpu_ready(monkeypatch, tmp_path):
    executable = tmp_path / "python"
    executable.touch()
    monkeypatch.setattr(
        managed, "_run_command",
        lambda *a, **kw: '{"status":"cpu","backend":"cpu","reason":"No GPU"}',
    )
    with pytest.raises(managed.ManagedEnvironmentError, match="Strict GPU"):
        managed._gpu_probe(executable, True)
    assert managed._gpu_probe(executable, "auto")["status"] == "cpu"


def test_auto_gpu_install_failure_retries_cpu(parent_source, tmp_path, monkeypatch):
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installed_backends = []

    def command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
        elif args[:2] == ["uv", "pip"]:
            backend = args[args.index("--torch-backend") + 1]
            installed_backends.append(backend)
            if backend == "auto":
                raise managed.ManagedEnvironmentError("GPU wheel not installable")
        return ""
    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2, "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
            {"name": "torch", "version": "2.14.1+cpu"},
        ],
    }
    monkeypatch.setattr(managed, "_verify", lambda *args: (
        observation, {"status": "cpu", "backend": "cpu", "reason": "No verified accelerator"},
    ))
    class Catalog:
        def register_environment(self, *, fingerprint, **kwargs):
            return "env_" + fingerprint
    class Registry:
        _catalog = Catalog()
        def bind_interpreter(self, env, exe):
            assert Path(exe).is_file()
    result = managed.create(Registry(), teal_extras=["transformers"], gpu="auto")
    assert installed_backends == ["auto", "cpu"]
    assert result["gpu"]["status"] == "cpu"
    assert "CPU-only retry succeeded" in result["gpu"]["reason"]


@pytest.mark.acceptance
def test_concurrent_managed_create_reuses_one_install(
    parent_source, tmp_path, monkeypatch,
):
    """Threads sharing one cache request cannot publish duplicate installs."""
    from concurrent.futures import ThreadPoolExecutor
    import time

    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installs = []
    def command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            installs.append(args)
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
            time.sleep(0.1)
        return ""
    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2, "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
        ],
    }
    monkeypatch.setattr(
        managed, "_verify", lambda *args: (
            observation, {"status": "not_applicable", "backend": None, "reason": None},
        ),
    )
    class Registry:
        class Catalog:
            def register_environment(self, *, fingerprint, **kwargs):
                return "env_" + fingerprint
        _catalog = Catalog()
        def bind_interpreter(self, env, exe):
            assert Path(exe).is_file()

    registry = Registry()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(managed.create, registry) for _ in range(2)]
        results = [f.result(timeout=30) for f in futures]
    assert len(installs) == 1
    assert sorted(result["reused"] for result in results) == [False, True]
    assert results[0]["environment_id"] == results[1]["environment_id"]

@pytest.mark.acceptance
def test_real_uv_managed_child_textlength(tmp_path, monkeypatch):
    """End-to-end isolated Git-sourced child, no PYTHONPATH dependency sharing."""
    if os.environ.get("TEAL_REAL_MANAGED_ACCEPTANCE") != "1":
        pytest.skip("Opt in with TEAL_REAL_MANAGED_ACCEPTANCE=1")
    if shutil.which("uv") is None or shutil.which("git") is None:
        pytest.skip("uv and git are required for managed-child acceptance")
    from text_analysis_lab import Project
    from text_analysis_lab.translators import TextLength
    import pandas as pd

    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    # A parent notebook may have an unrelated or stale package first on its
    # PYTHONPATH. A managed child must ignore this rather than import it.
    shadow = tmp_path / "shadow"
    shadow_package = shadow / "text_analysis_lab"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        "raise RuntimeError('managed child imported parent PYTHONPATH')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    csv = tmp_path / "source.csv"
    pd.DataFrame({"text": ["hello world", "one two three", "four"]}).to_csv(
        csv, index=False,
    )
    with Project.create(tmp_path / "project", "managed-r2") as project:
        source = project.read_csv(csv, text_fields="text", metadata_fields=None)
        parent = project.environments.opening_id
        result = project.environments.create(teal_extras=["parallel"], gpu=False)
        # Distinct installation requests may resolve to exactly the same
        # observed software. Both published caches must remain reusable.
        same_software = project.environments.create(
            teal_extras=["parallel"], gpu="auto",
        )
        assert not same_software["reused"]
        assert same_software["environment_id"] == result["environment_id"]
        assert not result["reused"]
        assert result["environment_id"] != parent
        assert result["portability"].startswith("observed_unlocked")
        project.environments.set_default(result["environment_id"])
        output = project.translate(
            TextLength({"text": "words"}), source, workers=1, batch_size=2,
        )["output"]
        values = output.query(
            data_columns=False, metadata_columns=["text_words"],
            metadata_mode="local", form="table",
        )
        assert values["text_words"].tolist() == [2, 3, 1]
        operation = project.operation_for_artifact(output)
        assert operation["environment_id"] == result["environment_id"]
        assert operation["platform_id"] in {
            p["platform_id"] for p in project.platforms.list()
        }
        attempts = project.execution_attempts(operation["operation_id"])
        assert len(attempts) == 1
        assert attempts[0]["outcome"] == "complete"
        assert attempts[0]["environment_id"] == result["environment_id"]
        # Deliberately DO NOT call create() again: reopening must discover
        # this published installation by identity rather than by request.

    # A fresh Project instance has no interpreter binding. Reuse must
    # rediscover the already-published installation solely by E1 identity,
    # without calling create() or re-resolving uv packages.
    with Project.open(tmp_path / "project") as reopened:
        assert result["environment_id"] not in reopened.environments._interpreter_bindings
        # Deterministic selection among two fully verified request caches.
        chosen = reopened.environments.bound_interpreter(result["environment_id"])
        assert chosen.is_file()
        assert chosen == managed.find_cached_interpreter(result["environment_id"])
        reopened.environments.set_default(result["environment_id"])
        restored = reopened.translate(
            TextLength({"text": "characters"}), source.artifact_id,
            workers=1, batch_size=2,
        )["output"]
        assert restored.n_rows == 3
        history = reopened.execution_attempts(
            reopened.operation_for_artifact(restored)["operation_id"]
        )
        assert len(history) == 1
        assert history[0]["environment_id"] == result["environment_id"]
        assert result["environment_id"] in reopened.environments._interpreter_bindings



def test_gpu_auto_retries_failed_torch_runtime_and_preserves_cache_reason(
    parent_source, tmp_path, monkeypatch,
):
    """A successful install but failed Torch import must retry as CPU."""
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installed = []

    def fake_command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
        elif args[:3] == ["uv", "pip", "install"]:
            installed.append(args[args.index("--torch-backend") + 1])
        return ""

    monkeypatch.setattr(managed, "_run_command", fake_command)
    observed = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
            {"name": "torch", "version": "2.14.1+cpu"},
        ],
    }
    checked = []

    def verify(*args):
        checked.append(1)
        if len(checked) == 1:
            raise managed.ManagedAcceleratorError("Torch native import failed")
        return observed, {
            "status": "cpu", "backend": "cpu", "reason": "No verified accelerator",
        }

    monkeypatch.setattr(managed, "_verify", verify)

    class Registry:
        class Catalog:
            def register_environment(self, *, fingerprint, **kwargs):
                return "env_" + fingerprint
        _catalog = Catalog()

        def bind_interpreter(self, env, exe):
            assert Path(exe).is_file()

    first = managed.create(Registry(), gpu="auto", teal_extras=["transformers"])
    assert installed == ["auto", "cpu"]
    assert first["gpu"]["status"] == "cpu"
    assert "Torch runtime verification failed" in first["gpu"]["reason"]
    assert first["reused"] is False
    second = managed.create(Registry(), gpu="auto", teal_extras=["transformers"])
    assert second["reused"] is True
    assert second["gpu"]["reason"] == first["gpu"]["reason"]
    assert installed == ["auto", "cpu"]  # No reinstallation.

    markers = list((tmp_path / "managed").glob("*/managed.json"))
    assert len(markers) == 1
    payload = json.loads(markers[0].read_text(encoding="utf-8"))
    assert payload["fallback_reason"] == first["gpu"]["reason"]


@pytest.mark.parametrize("failure", [
    "TeAL source mismatch",
    "R1 protocol mismatch",
    "uv pip check failed",
])
def test_gpu_auto_does_not_retry_unrelated_verification_errors(
    parent_source, tmp_path, monkeypatch, failure,
):
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installs = []

    def fake_command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
        if args[:3] == ["uv", "pip", "install"]:
            installs.append(args[args.index("--torch-backend") + 1])
        return ""

    monkeypatch.setattr(managed, "_run_command", fake_command)
    monkeypatch.setattr(
        managed, "_verify",
        lambda *args: (_ for _ in ()).throw(ManagedEnvironmentError(failure)),
    )
    with pytest.raises(ManagedEnvironmentError, match=failure):
        managed.create(object(), gpu="auto", teal_extras=["transformers"])
    assert installs == ["auto"]
    assert not list((tmp_path / "managed").glob("*/managed.json"))


def test_managed_cache_discovery_rejects_drift_without_binding(
    parent_source, tmp_path, monkeypatch,
):
    from text_analysis_lab import Project

    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    def fake_command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
            exe.chmod(0o755)
        return ""
    monkeypatch.setattr(managed, "_run_command", fake_command)
    observed = {
        "schema_version": 2, "kind": "observed_unlocked",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
        },
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
        ],
    }
    changed = copy.deepcopy(observed)
    changed["distributions"].append({"name": "changed-after-install", "version": "1"})
    state = {"observed": observed}
    monkeypatch.setattr(
        managed, "_verify",
        lambda *args: (state["observed"], {
            "status": "not_applicable", "backend": None, "reason": None,
        }),
    )
    with Project.create(tmp_path / "project", "managed-discovery") as project:
        result = managed.create(project.environments)
        managed_id = result["environment_id"]
    with Project.open(tmp_path / "project") as reopened:
        assert managed_id not in reopened.environments._interpreter_bindings
        assert reopened.environments.bound_interpreter(managed_id).is_file()
    state["observed"] = changed
    with Project.open(tmp_path / "project") as reopened:
        with pytest.raises(managed.ManagedEnvironmentError, match="failed re-verification"):
            reopened.environments.bound_interpreter(managed_id)
        assert managed_id not in reopened.environments._interpreter_bindings


def test_torch_probe_classifies_import_crash_as_accelerator_failure(tmp_path, monkeypatch):
    executable = tmp_path / "python"
    executable.touch()

    def failed_torch_command(*args, **kwargs):
        raise ManagedEnvironmentError("Torch failed to load a CUDA native library")

    monkeypatch.setattr(managed, "_run_command", failed_torch_command)
    with pytest.raises(managed.ManagedAcceleratorError, match="Torch runtime"):
        managed._gpu_probe(executable, "auto")


def test_auto_gpu_does_not_retry_unrelated_install_failure(
    parent_source, tmp_path, monkeypatch,
):
    """A broken TeAL source/index is not evidence of a GPU problem."""
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installed = []

    def fake_command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            executable = managed._interpreter(Path(args[2]))
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
        elif args[:3] == ["uv", "pip", "install"]:
            installed.append(args[args.index("--torch-backend") + 1])
            raise ManagedEnvironmentError("Invalid immutable TeAL Git source")
        return ""

    monkeypatch.setattr(managed, "_run_command", fake_command)
    with pytest.raises(ManagedEnvironmentError, match="TeAL Git source"):
        managed.create(object(), gpu="auto", teal_extras=["transformers"])
    assert installed == ["auto"]
    assert not list((tmp_path / "managed").glob("*/managed.json"))



@pytest.mark.parametrize("all_drifted", [False, True])
def test_duplicate_cache_rediscovery_skips_drift_and_rejects_all_invalid(
    parent_source, tmp_path, monkeypatch, all_drifted,
):
    """A bad cache cannot hide a separate valid installation with the same ID."""
    from text_analysis_lab import Project

    root = tmp_path / "managed"
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(root))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installs = []

    def command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            installs.append(args)
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
            exe.chmod(0o755)
        return ""

    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2, "kind": "observed_unlocked",
        "python": {"implementation": sys.implementation.name,
                   "version": ".".join(map(str, sys.version_info[:3]))},
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
        ],
    }
    drifted = copy.deepcopy(observation)
    drifted["distributions"].append({"name": "unexpected-package", "version": "1"})
    bad_keys = set()
    seen = []

    def verify(directory, request, uv):
        seen.append(directory.name)
        return (drifted if directory.name in bad_keys else observation), {
            "status": "not_applicable", "backend": None, "reason": None,
        }

    monkeypatch.setattr(managed, "_verify", verify)
    with Project.create(tmp_path / "project", "duplicate-uv") as project:
        first = project.environments.create(gpu=False)
        second = project.environments.create(gpu="auto")
        assert first["environment_id"] == second["environment_id"]
    keys = sorted(p.name for p in root.iterdir() if (p / "managed.json").is_file())
    assert len(keys) == 2
    bad_keys.update(keys if all_drifted else keys[:1])
    seen.clear()

    with Project.open(tmp_path / "project") as project:
        if all_drifted:
            with pytest.raises(ManagedEnvironmentError, match="failed re-verification"):
                project.environments.bound_interpreter(first["environment_id"])
            assert first["environment_id"] not in project.environments._interpreter_bindings
        else:
            chosen = project.environments.bound_interpreter(first["environment_id"])
            assert chosen == managed._interpreter(root / keys[1])
    assert seen == keys
    assert len(installs) == 2
    assert all((root / key / "managed.json").exists() for key in keys)


def test_cpu_fallback_stays_cpu_only_on_create_reuse_and_rediscovery(
    parent_source, tmp_path, monkeypatch,
):
    """An advertised but broken accelerator must not be retried after fallback."""
    root = tmp_path / "managed"
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(root))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")
    installs, policies = [], []

    def command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
            exe.chmod(0o755)
        elif args[:3] == ["uv", "pip", "install"]:
            installs.append(args[args.index("--torch-backend") + 1])
        return ""

    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2, "kind": "observed_unlocked",
        "python": {"implementation": sys.implementation.name,
                   "version": ".".join(map(str, sys.version_info[:3]))},
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
            {"name": "torch", "version": "2.14.1+cpu"},
        ],
    }
    monkeypatch.setattr(managed, "_probe", lambda *args: observation)

    def gpu_probe(exe, policy):
        policies.append(policy)
        if policy is not False:
            raise managed.ManagedAcceleratorError("Advertised MPS failed tensor smoke")
        return {"status": "cpu", "backend": "cpu", "reason": "CPU-only policy selected"}

    monkeypatch.setattr(managed, "_gpu_probe", gpu_probe)

    class Registry:
        class Catalog:
            def register_environment(self, *, fingerprint, **kwargs):
                return "env_" + fingerprint
        _catalog = Catalog()
        def bind_interpreter(self, env, exe):
            assert Path(exe).is_file()

    registry = Registry()
    first = managed.create(registry, teal_extras=["transformers"], gpu="auto")
    assert first["gpu"]["status"] == "cpu"
    assert "Torch runtime verification failed" in first["gpu"]["reason"]
    assert policies == ["auto", False]
    reused = managed.create(registry, teal_extras=["transformers"], gpu="auto")
    assert reused["reused"] and reused["gpu"] == first["gpu"]
    assert policies == ["auto", False, False]
    assert managed.find_cached_interpreter(first["environment_id"]).is_file()
    assert policies == ["auto", False, False, False]
    assert installs == ["auto", "cpu"]
    assert len(list(root.glob("*/managed.json"))) == 1


def test_torch_accelerator_tensor_failure_triggers_error_not_silent_cpu(
    monkeypatch, tmp_path,
):
    """Execute the actual Torch smoke script with an advertised broken CUDA."""
    import contextlib
    import importlib.util
    import io
    import types

    executable = tmp_path / "python"
    executable.touch()
    devices: list[str] = []
    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch.version = types.SimpleNamespace(hip=None)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)

    class FakeTensor:
        def sum(self):
            return self
        def item(self):
            return 5.0

    def tensor(values, *, device):
        devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA driver/runtime cannot allocate tensor")
        return FakeTensor()

    torch.tensor = tensor
    import sys as runtime_sys
    monkeypatch.setitem(runtime_sys.modules, "torch", torch)
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: torch.__spec__ if name == "torch" else None,
    )

    def execute_actual_smoke(args, **kwargs):
        assert args[1:3] == ["-I", "-c"]
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(args[3], {"__name__": "__main__"})
        except Exception as exc:
            # _run_command raises ManagedEnvironmentError on nonzero child exit.
            raise managed.ManagedEnvironmentError("Torch runtime subprocess failed") from exc
        return captured.getvalue()

    monkeypatch.setattr(managed, "_run_command", execute_actual_smoke)
    for policy in ("auto", True):
        devices.clear()
        with pytest.raises(managed.ManagedAcceleratorError, match="Torch runtime"):
            managed._gpu_probe(executable, policy)
        assert devices == ["cpu", "cuda"]
    devices.clear()
    assert managed._gpu_probe(executable, False)["status"] == "cpu"
    assert devices == ["cpu"]


def test_discovery_rejects_old_parent_source_commit_even_if_r1_digest_matches(
    parent_source, tmp_path, monkeypatch,
):
    """The parent commit check is independent of R1's narrow module digest."""
    from text_analysis_lab import Project

    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "managed"))
    monkeypatch.setattr(managed, "_uv", lambda: "uv")

    def command(args, **kwargs):
        if args[:2] == ["uv", "venv"]:
            exe = managed._interpreter(Path(args[2]))
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.touch()
            exe.chmod(0o755)
        return ""

    monkeypatch.setattr(managed, "_run_command", command)
    observation = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {"implementation": sys.implementation.name,
                   "version": ".".join(map(str, sys.version_info[:3]))},
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
        ],
    }
    probes = []

    def unchanged_r1_probe(*args):
        probes.append(True)
        return observation, {"status": "not_applicable", "backend": None, "reason": None}

    monkeypatch.setattr(managed, "_verify", unchanged_r1_probe)
    with Project.create(tmp_path / "project", "managed-mismatched-parent") as project:
        first = project.environments.create()
        identity = first["environment_id"]
    monkeypatch.setattr(managed, "_local_source_commit", lambda: "b" * 40)
    with Project.open(tmp_path / "project") as reopened:
        assert identity not in reopened.environments._interpreter_bindings
        with pytest.raises(
            ManagedEnvironmentError, match="failed re-verification",
        ) as exc:
            reopened.environments.bound_interpreter(identity)
        assert "source commit" in str(exc.value.__cause__)
        assert identity not in reopened.environments._interpreter_bindings
    # The child's same-version software and narrow R1 source digest still
    # match the stored installation, but parent revision is now different.
    assert len(probes) == 1
