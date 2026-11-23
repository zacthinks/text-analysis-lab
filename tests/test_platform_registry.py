"""R0 separate software and platform registry acceptance tests."""

from __future__ import annotations

import copy
import json
import sqlite3

import pandas as pd
import pytest

from text_analysis_lab import Project
from text_analysis_lab.core import environments, platforms
from text_analysis_lab.core.environments import (
    fingerprint_observation,
    normalize_environment,
)
from text_analysis_lab.core.platforms import (
    fingerprint_platform,
    normalize_platform,
)


def software(version="3.11.15", torch="2.14.1", teal_source="a" * 40):
    return {
        "schema_version": 2,
        "kind": "portable_locked",
        "python": {"implementation": "cpython", "version": version},
        "teal": {
            "version": "0.12.2",
            "source": {"vcs": "git", "commit": teal_source},
        },
        "distributions": [
            {"name": "torch", "version": torch},
            {"name": "Text_Analysis.Lab", "version": "0.12.2"},
        ],
        "extras": ["transformers", "parallel", "transformers"],
        "lock": {"sha256": "b" * 64, "kind": "uv"},
    }


def machine(os_name="Windows", driver="551.23"):
    return {
        "schema_version": 1,
        "os": {"system": os_name, "release": "11", "kernel": "test"},
        "architecture": "AMD64",
        "python_architecture": "64bit",
        "cpu": {"processor": "Generic processor"},
        "accelerators": [
            {"backend": "nvidia", "model": "test accelerator", "driver_version": driver},
        ],
    }


def test_portable_software_id_ignores_host_and_package_order():
    a = software()
    b = copy.deepcopy(a)
    b["distributions"].reverse()
    b["extras"].reverse()
    b["extras"].append("PARALLEL")
    assert fingerprint_observation(a) == fingerprint_observation(b)
    assert normalize_environment(a)["extras"] == ["parallel", "transformers"]

    c = copy.deepcopy(a)
    c["python"]["version"] = "3.11.16"
    assert fingerprint_observation(c) != fingerprint_observation(a)
    c = copy.deepcopy(a)
    c["teal"]["source"]["commit"] = "c" * 40
    assert fingerprint_observation(c) != fingerprint_observation(a)
    c = copy.deepcopy(a)
    c["distributions"][0]["version"] = "2.14.2"
    assert fingerprint_observation(c) != fingerprint_observation(a)

    # Platform data cannot accidentally join the software specification.
    c = copy.deepcopy(a)
    c["platform"] = {"system": "Windows"}
    with pytest.raises(environments.EnvironmentRegistryError):
        fingerprint_observation(c)


def test_platform_identity_is_independent_and_privacy_limited():
    a = machine()
    same = copy.deepcopy(a)
    same["accelerators"].reverse()
    assert fingerprint_platform(a) == fingerprint_platform(same)
    assert fingerprint_platform(a) != fingerprint_platform(machine("Darwin"))
    assert fingerprint_platform(a) != fingerprint_platform(machine(driver="552.12"))

    bad = copy.deepcopy(a)
    bad["hostname"] = "personally-identifying-host"
    with pytest.raises(platforms.PlatformRegistryError):
        normalize_platform(bad)


def test_absent_gpu_firmware_or_nvidia_tools_are_normal(monkeypatch):
    monkeypatch.setattr(platforms, "_nvidia_devices", lambda: [])
    captured = platforms.observe_platform()
    assert captured["accelerators"] == []
    assert len(fingerprint_platform(captured)) == 64


def test_verified_portable_spec_is_registered_but_not_claimed_executable(tmp_path, monkeypatch):
    location = tmp_path / "study"
    state = {"platform": machine()}
    monkeypatch.setattr(platforms, "observe_platform", lambda: state["platform"])
    with Project.create(location, "study") as project:
        a = project.environments.register_portable_specification(software())
        b = project.environments.register_portable_specification(software())
        assert a == b
        assert project.environments.get(a)["kind"] == "portable_locked"
        assert project.environments.get(a)["readiness"] == "unverified"
        original = project.platforms.opening_id
        assert project.platforms.get(original)["observation"] == normalize_platform(machine())

    state["platform"] = machine("Darwin")
    with Project.open(location) as project:
        assert project.environments.register_portable_specification(software()) == a
        assert project.platforms.opening_id != original
        assert len(project.platforms.list()) == 2

    state["platform"] = machine("Darwin", "999")
    with Project.open(location) as project:
        assert project.environments.register_portable_specification(software()) == a
        assert len(project.platforms.list()) == 3


def test_observed_software_stable_across_os_only_changes(monkeypatch):
    obs = environments.observe_environment()
    assert obs["kind"] == "observed_unlocked"
    assert "platform" not in obs
    assert "abi" not in obs["python"]
    first = fingerprint_observation(obs)
    monkeypatch.setattr(environments.platform, "system", lambda: "DifferentSystem")
    assert fingerprint_observation(environments.observe_environment()) == first


def test_local_operations_write_both_ids_catalog_and_descriptors(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "observe_platform", lambda: machine())
    project_dir = tmp_path / "study"
    csv = tmp_path / "texts.csv"
    pd.DataFrame({"text": ["alpha beta", "gamma"]}).to_csv(csv, index=False)
    with Project.create(project_dir, "study") as project:
        source = project.read_csv(csv, text_fields="text", metadata_fields=None)
        translated = project.translate(
            __import__("text_analysis_lab.translators", fromlist=["TextLength"]).TextLength(
                {"text": "words"}
            ),
            source,
        )["output"]
        for artifact in [source, translated]:
            row = project.operation_for_artifact(artifact)
            assert row["environment_id"] == project.environments.opening_id
            assert row["platform_id"] == project.platforms.opening_id
            metadata = json.loads(
                project.storage.operation_descriptor_path(row["operation_id"]).read_text()
            )
            assert metadata["environment_id"] == row["environment_id"]
            assert metadata["platform_id"] == row["platform_id"]
        original_id = project.platforms.opening_id
        import_operation = project.operation_for_artifact(source)["operation_id"]

    monkeypatch.setattr(platforms, "observe_platform", lambda: machine("Darwin"))
    with Project.open(project_dir) as project:
        row = project.get_operation(import_operation)
        assert row["platform_id"] == original_id
        assert project.platforms.opening_id != original_id
        assert project.environments.opening_id == row["environment_id"]
