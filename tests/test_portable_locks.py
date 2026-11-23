"""R2b portable software-lock identity, source safety, and local conformance tests.

Synthetic locks exercise resolver-format validation, not cross-OS installations.
The real uv integration still requires separate CI acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import pytest

from text_analysis_lab.core import portable_locks as locks
from text_analysis_lab.core.environments import (
    EnvironmentRegistryError, fingerprint_observation, normalize_environment,
)

COMMIT = "a" * 40
SOURCE = "https://github.com/zacthinks/text-analysis-lab.git"


def _lock(*, commit=COMMIT, extras=(), python="3.11.15"):
    return {
        "version": 1, "revision": 3, "requires-python": "==" + python,
        "package": [
            {
                "name": "teal-portable-contract", "version": "0.0.0",
                "source": {"virtual": "."},
                "dependencies": [{"name": "text-analysis-lab"}],
                "metadata": {"requires-dist": [{
                    "name": "text-analysis-lab", "extras": list(extras),
                    "git": SOURCE + "?rev=" + commit,
                }]},
            },
            {
                "name": "text-analysis-lab", "version": "0.12.2",
                "source": {"git": SOURCE + "?rev=" + commit + "#" + commit},
                "metadata": {"provides-extras": ["parallel"]},
            },
        ],
    }


def _spec(**kwargs):
    lock = _lock(**kwargs)
    manifest = locks.validate_manifest(
        {"schema": 1, "uv_lock": lock},
        python=kwargs.get("python", "3.11.15"),
        commit=kwargs.get("commit", COMMIT),
        extras=list(kwargs.get("extras", ())), teal_version="0.12.2",
    )
    spec = {
        "schema_version": 2, "kind": "portable_locked",
        "python": {"implementation": "cpython", "version": kwargs.get("python", "3.11.15")},
        "teal": {"version": "0.12.2", "source": {"vcs": "git", "commit": kwargs.get("commit", COMMIT)}},
        "extras": list(kwargs.get("extras", ())),
        "lock": {"kind": locks.LOCK_KIND, "manifest": manifest,
                 "sha256": hashlib.sha256(json.dumps(
                     manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
    }
    return normalize_environment(spec)


class _Registry:
    def __init__(self):
        self.data = {}

    def register_portable_specification(self, spec):
        spec = normalize_environment(spec)
        key = "env_" + fingerprint_observation(spec)
        self.data[key] = spec
        return key

    def get(self, key):
        return {"observation": self.data[key]}


def test_lock_identity_ignores_package_order_but_not_semantics():
    first = _spec()
    rearranged = copy.deepcopy(first)
    rearranged["lock"]["manifest"]["uv_lock"]["package"].reverse()
    assert fingerprint_observation(first) == fingerprint_observation(rearranged)
    assert fingerprint_observation(first) != fingerprint_observation(_spec(commit="b" * 40))
    assert fingerprint_observation(first) != fingerprint_observation(_spec(python="3.11.16"))
    assert fingerprint_observation(first) != fingerprint_observation(_spec(extras=("parallel",)))


def test_export_import_idempotent_and_rejects_tampering(tmp_path):
    left = _Registry()
    spec = _spec()
    key = left.register_portable_specification(spec)
    location = locks.export_specification(left, key, tmp_path / "lock.json")
    data = json.loads(location.read_text())
    assert "cache" not in location.read_text()
    right = _Registry()
    assert locks.import_specification(right, location) == key
    assert locks.import_specification(right, location) == key
    data["environment_id"] = "env_" + "0" * 64
    location.write_text(json.dumps(data))
    with pytest.raises(EnvironmentRegistryError):
        locks.import_specification(right, location)


@pytest.mark.parametrize("mutation", [
    lambda lock: lock["package"][1].update(source={"git": "file:///tmp/evil#" + COMMIT}),
    lambda lock: lock["package"][1].update(source={"git": SOURCE + "?rev=main#" + COMMIT}),
    lambda lock: lock["package"][1].update(source={"git": SOURCE + "?rev=" + COMMIT + "#" + "b" * 40}),
    lambda lock: lock["package"][0]["metadata"]["requires-dist"][0].update(extras=["all"]),
    lambda lock: lock.update(**{"requires-python": ">=3.11"}),
])
def test_invalid_source_or_contract_is_rejected(mutation):
    lock = _lock()
    mutation(lock)
    with pytest.raises(EnvironmentRegistryError):
        locks.validate_manifest({"schema": 1, "uv_lock": lock}, python="3.11.15",
                                commit=COMMIT, extras=[], teal_version="0.12.2")


def test_artifact_hash_required():
    lock = _lock()
    lock["package"].append({"name": "sample", "version": "1.0",
                            "source": {"registry": "https://pypi.org/simple"},
                            "wheels": [{"url": "https://files.pythonhosted.org/sample.whl"}]})
    with pytest.raises(EnvironmentRegistryError, match="hash"):
        locks.validate_manifest({"schema": 1, "uv_lock": lock}, python="3.11.15",
                                commit=COMMIT, extras=[], teal_version="0.12.2")


def test_toml_roundtrip_preserves_nested_uv_lock():
    lock = _lock(extras=("parallel",))
    import tomllib
    assert tomllib.loads(locks._lock_toml(lock)) == lock


def test_profile_rejects_unsupported_gpu_and_extras(monkeypatch):
    with pytest.raises(Exception, match="CPU profiles only"):
        locks.create_specification(_Registry(), gpu="auto")
    with pytest.raises(EnvironmentRegistryError, match="not yet supported"):
        _spec(extras=("transformers",))


def test_matching_installed_exact_version_and_git_origin(monkeypatch):
    spec = _spec()
    from packaging.requirements import Requirement
    monkeypatch.setattr(locks, "_export_requirements", lambda s: [
        Requirement("text-analysis-lab @ git+" + SOURCE + "@" + COMMIT),
        Requirement("numpy==2.2.0"),
    ])
    obs = {"python": {"implementation": "cpython", "version": "3.11.15"},
           "teal": {"version": "0.12.2"}, "distributions": [
               {"name": "text-analysis-lab", "version": "0.12.2", "vcs": "git", "vcs_commit": COMMIT},
               {"name": "numpy", "version": "2.2.0"},
           ]}
    assert locks.matches_installed(spec, obs)
    obs["distributions"][1]["version"] = "2.3.0"
    assert not locks.matches_installed(spec, obs)
    obs["distributions"][1]["version"] = "2.2.0"
    obs["distributions"].append({"name": "extra", "version": "1.0"})
    assert not locks.matches_installed(spec, obs)


@pytest.mark.acceptance
def test_real_uv_portable_lock_reconstruct_and_delegate(tmp_path, monkeypatch):
    """Actual Git/uv resolution, frozen install, TextLength, and restart on host OS.

    CI runs this test independently on Ubuntu and Windows. It does not
    assert macOS or real GPU compatibility.
    """
    import os
    import shutil
    if os.environ.get("TEAL_REAL_MANAGED_ACCEPTANCE") != "1":
        pytest.skip("Requires TEAL_REAL_MANAGED_ACCEPTANCE=1")
    if shutil.which("uv") is None or shutil.which("git") is None:
        pytest.skip("Requires uv and git")
    import pandas as pd
    from text_analysis_lab import Project
    from text_analysis_lab.translators import TextLength

    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "portable-cache"))
    csv = tmp_path / "source.csv"
    pd.DataFrame({"text": ["hello world", "one two three"]}).to_csv(csv, index=False)
    with Project.create(tmp_path / "project", "portable-r2b") as project:
        source = project.read_csv(csv, text_fields="text", metadata_fields=None)
        resolved = project.environments.create_specification(teal_extras=["parallel"], gpu=False)
        logical = resolved["environment_id"]
        assert project.environments.get(logical)["kind"] == "portable_locked"
        portable_file = project.environments.export_specification(logical, tmp_path / "portable.json")
        assert logical == project.environments.import_specification(portable_file)
        first = project.environments.create_from_specification(logical)
        assert first["environment_id"] == logical
        assert first["observed_environment_id"] != logical
        assert first["reused"] is False
        second = project.environments.create_from_specification(logical)
        assert second["reused"] is True
        project.environments.set_default(logical)
        result = project.translate(TextLength({"text": "words"}), source, workers=1, batch_size=2)["output"]
        observed = project.operation_for_artifact(result)
        assert observed["environment_id"] == logical
        assert observed["platform_id"] in {p["platform_id"] for p in project.platforms.list()}
        attempts = project.execution_attempts(observed["operation_id"])
        assert attempts[-1]["environment_id"] == logical
    with Project.open(tmp_path / "project") as reopened:
        assert logical not in reopened.environments._interpreter_bindings
        assert reopened.environments.bound_interpreter(logical).is_file()
        reopened.environments.set_default(logical)
        result = reopened.translate(
            TextLength({"text": "characters"}), source.artifact_id,
            workers=1, batch_size=2,
        )["output"]
        assert result.n_rows == 2
        attempt = reopened.execution_attempts(
            reopened.operation_for_artifact(result)["operation_id"]
        )
        assert attempt[-1]["environment_id"] == logical


def test_platform_marker_selects_only_applicable_locked_inventory(monkeypatch):
    """Synthetic platform-marker evaluation; not an actual Windows install."""
    from packaging.requirements import Requirement
    spec = _spec()
    monkeypatch.setattr(locks, "_export_requirements", lambda s: [
        Requirement("text-analysis-lab @ git+" + SOURCE + "@" + COMMIT),
        Requirement('numpy==1.26.4; sys_platform == "win32"'),
        Requirement('numpy==2.2.0; sys_platform == "linux"'),
    ])
    env = {"python_version": "3.11", "python_full_version": "3.11.15",
           "sys_platform": "win32", "extra": ""}
    monkeypatch.setattr(locks, "default_environment", lambda: dict(env))
    observation = {"python": {"implementation": "cpython", "version": "3.11.15"},
                   "teal": {"version": "0.12.2"}, "distributions": [
                       {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": COMMIT},
                       {"name": "numpy", "version": "1.26.4"},
                   ]}
    assert locks.matches_installed(spec, observation)
    env["sys_platform"] = "linux"
    observation["distributions"][1]["version"] = "2.2.0"
    assert locks.matches_installed(spec, observation)
    observation["distributions"][1]["version"] = "1.26.4"
    assert not locks.matches_installed(spec, observation)


def test_changing_artifact_hash_changes_portable_id():
    base = _spec()
    def with_hash(hexchar):
        value = copy.deepcopy(base)
        value["lock"]["manifest"]["uv_lock"]["package"].append({
            "name": "sample", "version": "1.0",
            "source": {"registry": "https://pypi.org/simple"},
            "wheels": [{"url": "https://files.pythonhosted.org/sample.whl",
                        "hash": "sha256:" + hexchar * 64}],
        })
        from text_analysis_lab.core.environments import _canonical_json, _normalize_lock_manifest
        value["lock"]["manifest"]["uv_lock"] = _normalize_lock_manifest(
            value["lock"]["manifest"]["uv_lock"])
        value["lock"]["sha256"] = hashlib.sha256(
            _canonical_json(value["lock"]["manifest"]).encode()).hexdigest()
        return normalize_environment(value)
    assert fingerprint_observation(with_hash("a")) != fingerprint_observation(with_hash("b"))