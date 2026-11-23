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
    rearranged["lock"]["sha256"] = hashlib.sha256(json.dumps(
        rearranged["lock"]["manifest"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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