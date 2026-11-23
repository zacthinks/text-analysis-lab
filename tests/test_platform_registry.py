"""R0 separate software and platform registry acceptance tests."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
        "extras": ["transformers", "parallel", "transformers"],
        "lock": {
            "kind": "uv",
            "manifest": {"requirements": {"torch": torch, "teal": "0.12.2"}},
            "sha256": hashlib.sha256(
                json.dumps(
                    {"requirements": {"teal": "0.12.2", "torch": torch}},
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
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
    c["lock"]["manifest"]["requirements"]["torch"] = "2.14.2"
    c["lock"]["sha256"] = hashlib.sha256(
        json.dumps(c["lock"]["manifest"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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


def test_verified_local_realization_is_same_logical_environment_across_platforms(
    tmp_path, monkeypatch,
):
    """A verified, pinned software recipe is the execution ID on both hosts."""
    raw = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {"implementation": "cpython", "version": "3.11.15"},
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "Text-Analysis-Lab", "version": "0.12.2", "vcs_commit": "a" * 40},
            {"name": "torch", "version": "2.14.1+cu130"},
        ],
    }
    spec = software(torch="2.14.1+cu130")
    manifest = {
        "realizations": [{"distributions": copy.deepcopy(raw["distributions"])}],
    }
    canonical_manifest = environments._normalize_lock_manifest(manifest)
    spec["lock"] = {
        "kind": "uv",
        "manifest": manifest,
        "sha256": hashlib.sha256(
            json.dumps(
                canonical_manifest, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }

    monkeypatch.setattr(environments, "observe_environment", lambda: copy.deepcopy(raw))
    current = {"value": machine("Windows")}
    monkeypatch.setattr(platforms, "observe_platform", lambda: current["value"])
    directory = tmp_path / "study"
    with Project.create(directory, "study") as project:
        observed_id = project.environments.opening_id
        portable_id = project.environments.register_portable_specification(spec)
        assert portable_id != observed_id
        assert project.environments.get(portable_id)["readiness"] == "unverified"
        project.environments.verify_current_portable(portable_id)
        assert project.environments.default["environment_id"] == portable_id

        source = tmp_path / "texts.csv"
        pd.DataFrame({"text": ["one two"]}).to_csv(source, index=False)
        artifact = project.read_csv(source, text_fields="text", metadata_fields=None)
        operation = project.operation_for_artifact(artifact)
        assert operation["environment_id"] == portable_id
        assert operation["platform_id"] == project.platforms.opening_id
        windows_platform_id = project.platforms.opening_id

    current["value"] = machine("Darwin")
    with Project.open(directory) as project:
        assert project.environments.opening_id == portable_id
        assert project.environments.default["environment_id"] == portable_id
        assert project.platforms.opening_id != windows_platform_id
        assert project.environments.get(observed_id)["kind"] == "observed_unlocked"
        assert project.environments.get(portable_id)["kind"] == "portable_locked"
        assert len(project.environments.list()) == 2

    raw["distributions"][1]["version"] = "2.15.0"
    with Project.open(directory) as project:
        assert project.environments.opening_id != portable_id
        assert project.environments.default["kind"] == "observed_unlocked"
        with pytest.raises(environments.EnvironmentRegistryError, match="does not match"):
            project.environments.verify_current_portable(portable_id)


def test_portable_manifest_requires_real_matching_digest():
    spec = software()
    spec["lock"]["sha256"] = "0" * 64
    with pytest.raises(environments.EnvironmentRegistryError, match="does not match"):
        fingerprint_observation(spec)


def test_concurrent_platform_registration_reuses_one_row(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "observe_platform", lambda: machine())
    directory = tmp_path / "parallel"
    with Project.create(directory, "parallel") as project:
        expected = project.platforms.opening_id

    def open_once(_):
        with Project.open(directory) as project:
            return project.platforms.opening_id

    with ThreadPoolExecutor(max_workers=6) as workers:
        values = list(workers.map(open_once, range(6)))
    assert set(values) == {expected}
    with Project.open(directory) as project:
        assert len(project.platforms.list()) == 1


def test_portable_index_priority_is_not_order_independent():
    """Package sets are unordered, but index selection priority may not be."""
    template = software()
    template["lock"]["manifest"] = {
        "packages": [
            {"name": "torch", "version": "2.14.1"},
            {"name": "teal", "version": "0.12.2"},
        ],
        "index_priority": ["trusted-first", "trusted-second"],
    }

    def frozen(request):
        normalized = environments._normalize_lock_manifest(request["lock"]["manifest"])
        request["lock"]["sha256"] = hashlib.sha256(json.dumps(
            normalized, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return fingerprint_observation(request)

    first = frozen(copy.deepcopy(template))
    reversed_packages = copy.deepcopy(template)
    reversed_packages["lock"]["manifest"]["packages"].reverse()
    assert frozen(reversed_packages) == first
    reversed_priority = copy.deepcopy(template)
    reversed_priority["lock"]["manifest"]["index_priority"].reverse()
    assert frozen(reversed_priority) != first


def test_portable_source_cannot_embed_original_hostname():
    spec = software()
    spec["teal"]["source"]["hostname"] = "original-workstation"
    with pytest.raises(environments.EnvironmentRegistryError, match="immutable"):
        fingerprint_observation(spec)


def test_replay_allowed_but_cross_platform_checkpoint_resume_rejected(
    tmp_path, monkeypatch,
):
    host = {"value": machine("Windows")}
    monkeypatch.setattr(platforms, "observe_platform", lambda: host["value"])
    directory = tmp_path / "resume-check"
    with Project.create(directory, "resume-check") as project:
        project.catalog.register_operator(
            operator_id="optr_r0_resume", operation_type="translate",
        )
        project.catalog.register_operation(
            operation_id="run_r0_resume",
            operation_type="translate",
            operator_id="optr_r0_resume",
            environment_id=project.environments.opening_id,
        )
        historical = project.get_operation("run_r0_resume")
        assert historical["platform_id"] == project.platforms.opening_id

    host["value"] = machine("Darwin")
    with Project.open(directory) as project:
        from text_analysis_lab.core.errors import OperatorError

        assert project.environments.opening_id == historical["environment_id"]
        assert project.platforms.opening_id != historical["platform_id"]
        with pytest.raises(OperatorError, match="Cross-platform checkpoint resume"):
            project.resume_operation("run_r0_resume")
        assert project.get_operation("run_r0_resume") == historical


def test_fast_platform_probe_avoids_bitness_subprocess_and_duplicate_gpu_calls(monkeypatch):
    """Opening many Projects must not launch 'file' or nvidia-smi for each one."""
    def forbidden_architecture():
        raise AssertionError("platform.architecture() may launch an external process")

    calls = []
    def fake_gpu():
        calls.append(1)
        return [{"backend": "nvidia", "model": "Test GPU", "driver_version": "1"}]

    monkeypatch.setattr(platforms.platform, "architecture", forbidden_architecture)
    monkeypatch.setattr(platforms, "_GPU_PROBE_CACHE", None)
    monkeypatch.setattr(platforms, "_query_nvidia_devices", fake_gpu)

    first = platforms.observe_platform()
    for _ in range(8):
        assert platforms.observe_platform() == first
    assert first["python_architecture"] in {"32bit", "64bit"}
    assert len(calls) == 1

    # Driver details may change. The bounded cache must eventually refresh.
    monkeypatch.setattr(platforms, "_GPU_PROBE_CACHE", (
        platforms.time.monotonic() - platforms._GPU_PROBE_TTL_SECONDS - 1,
        first["accelerators"],
    ))
    assert platforms.observe_platform()["accelerators"] == first["accelerators"]
    assert len(calls) == 2


def test_locked_identity_ignores_platform_specific_top_level_distributions():
    """One lock cannot fork into different software IDs for different wheels."""
    a = software()
    b = copy.deepcopy(a)
    a["distributions"] = [{"name": "torch", "version": "2.14.1+cu130"}]
    b["distributions"] = [{"name": "torch", "version": "2.14.1"}]
    # Redundant fields are rejected, rather than independently fingerprinted.
    for candidate in (a, b):
        with pytest.raises(environments.EnvironmentRegistryError, match="Unexpected"):
            fingerprint_observation(candidate)
    clean = software()
    assert "distributions" not in normalize_environment(clean)
    assert fingerprint_observation(clean) == fingerprint_observation(software())


@pytest.mark.parametrize("location,value", [
    (("teal", "hostname"), "researcher-laptop"),
    (("teal", "source", "origin"), "file:///home/researcher/pkg.whl"),
    (("lock", "manifest", "home"), "/home/researcher"),
    (("lock", "manifest", "realizations_extra"), ["hostname:my-pc"]),
    (("lock", "manifest", "index_priority"), ["https://user:secret@example.org/simple"]),
    (("lock", "manifest", "index_priority"), ["file:///home/researcher/simple"]),
    (("lock", "manifest", "index_priority"), ["https://192.168.1.10/simple"]),
    (("lock", "manifest", "index_priority"), ["https://[::1]/simple"]),
    (("lock", "manifest", "index_priority"), ["C:\\Users\\researcher\\index"]),
    (("lock", "manifest", "requirements"), {"torch": "/home/researcher/torch.whl"}),
    (("lock", "manifest", "packages"), [{"name": "torch", "version": "2", "hostname": "researcher-pc"}]),
    (("lock", "manifest", "realizations"), [{"distributions": [
        {"name": "torch", "version": "2", "url": "https://user:secret@example.org"}
    ]}]),
])
def test_portable_registration_rejects_untrusted_nested_metadata(tmp_path, location, value):
    """Unsafe metadata must fail before any project catalog row is persisted."""
    spec = software()
    current = spec
    for key in location[:-1]:
        if key == "packages" and key not in current:
            current[key] = []
        current = current[key]
    current[location[-1]] = value
    with Project.create(tmp_path / "study", "study") as project:
        before = len(project.environments.list())
        with pytest.raises(environments.EnvironmentRegistryError):
            project.environments.register_portable_specification(spec)
        assert len(project.environments.list()) == before


def test_portable_realizations_allow_different_platform_wheels_under_one_id():
    spec = software()
    manifest = {"realizations": [
        {"distributions": [{"name": "torch", "version": "2.14.1+cu130"}]},
        {"distributions": [{"name": "torch", "version": "2.14.1"}]},
    ]}
    canonical = environments._normalize_portable_manifest(manifest)
    spec["lock"]["manifest"] = manifest
    spec["lock"]["sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    first = fingerprint_observation(spec)
    reversed_spec = copy.deepcopy(spec)
    reversed_spec["lock"]["manifest"]["realizations"].reverse()
    assert fingerprint_observation(reversed_spec) == first
    assert len(normalize_environment(spec)["lock"]["manifest"]["realizations"]) == 2


def test_metadata_verification_is_not_artifact_verification(tmp_path, monkeypatch):
    raw = {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {"implementation": "cpython", "version": "3.11.15"},
        "teal": {"version": "0.12.2"},
        "distributions": [
            {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": "a" * 40},
        ],
    }
    spec = software()
    manifest = {"realizations": [{"distributions": copy.deepcopy(raw["distributions"])}]}
    spec["lock"]["manifest"] = manifest
    spec["lock"]["sha256"] = hashlib.sha256(
        json.dumps(environments._normalize_portable_manifest(manifest),
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(environments, "observe_environment", lambda: copy.deepcopy(raw))
    with Project.create(tmp_path / "study", "study") as project:
        env_id = project.environments.register_portable_specification(spec)
        assert project.environments.get(env_id)["verification_level"] == "not_verified"
        record = project.environments.verify_current_portable(env_id)
        assert record["verification_level"] == "installed_metadata_match"
        assert record["verification_level"] != "artifact_verified"


def test_same_lock_verifies_two_different_platform_specific_installations(
    tmp_path, monkeypatch,
):
    """CUDA on Windows and CPU on Linux adopt exactly one portable ID."""
    state = {"torch": "2.14.1+cu130", "os": "Windows"}

    def observed():
        return {
            "schema_version": 2,
            "kind": "observed_unlocked",
            "python": {"implementation": "cpython", "version": "3.11.15"},
            "teal": {"version": "0.12.2"},
            "distributions": [
                {"name": "text-analysis-lab", "version": "0.12.2", "vcs_commit": "a" * 40},
                {"name": "torch", "version": state["torch"]},
            ],
        }

    monkeypatch.setattr(environments, "observe_environment", observed)
    monkeypatch.setattr(platforms, "observe_platform", lambda: machine(state["os"]))
    spec = software()
    manifest = {
        "realizations": [
            {"distributions": [{**observed()["distributions"][0]},
                               {"name": "torch", "version": "2.14.1+cu130"}]},
            {"distributions": [{**observed()["distributions"][0]},
                               {"name": "torch", "version": "2.14.1"}]},
        ],
    }
    spec["lock"]["manifest"] = manifest
    spec["lock"]["sha256"] = hashlib.sha256(
        json.dumps(environments._normalize_portable_manifest(manifest),
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    directory = tmp_path / "portable-project"
    with Project.create(directory, "portable-project") as project:
        portable_id = project.environments.register_portable_specification(spec)
        project.environments.verify_current_portable(portable_id)
        original_platform = project.platforms.opening_id
        original_observed = project.environments._observed_id

    state.update(torch="2.14.1", os="Linux")
    with Project.open(directory) as project:
        assert project.platforms.opening_id != original_platform
        assert project.environments.opening_id != portable_id  # new, unverified inventory
        project.environments.verify_current_portable(portable_id)
        assert project.environments.opening_id == portable_id
        assert project.environments._observed_id != original_observed
        assert project.environments.get(portable_id)["verification_level"] == "installed_metadata_match"
        assert len([row for row in project.environments.list()
                    if row["kind"] == "portable_locked"]) == 1
