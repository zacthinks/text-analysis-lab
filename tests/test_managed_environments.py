"""R2 managed-environment policy and local-cache lifecycle regressions.

These tests do not download Python, Torch, or any external package.
"""
from __future__ import annotations

import json
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
