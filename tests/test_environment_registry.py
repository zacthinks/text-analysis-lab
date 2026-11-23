"""R0a environment history: registry identity is not execution readiness."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import threading
import venv
from pathlib import Path

import pytest

from text_analysis_lab import Project
from text_analysis_lab.core import environments as env_module
from text_analysis_lab.core.environments import (
    EnvironmentNotFoundError,
    EnvironmentRegistryError,
    fingerprint_observation,
)


def observation(label: str, *, package_version: str = "1", name: str = "sample") -> dict:
    return {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {"implementation": "cpython", "version": "3.11.0"},
        "teal": {"version": "0.12.2"},
        "distributions": [{"name": name, "version": package_version}, {"name": "teal", "version": label}],
    }


@pytest.fixture
def simulated(monkeypatch):
    current = {"value": observation("A")}
    monkeypatch.setattr(env_module, "observe_environment", lambda: current["value"])
    return current


def test_create_reopen_a_b_a_and_session_defaults(tmp_path, simulated):
    path = tmp_path / "study"
    a = Project.create(path, "study")
    try:
        first = a.environments.default
        assert first["environment_id"].startswith("env_")
        assert first["readiness"] == "current_interpreter"
        assert len(a.environments.list()) == 1
    finally:
        a.close()

    again = Project.open(path)
    assert again.environments.default["environment_id"] == first["environment_id"]
    assert len(again.environments.list()) == 1
    again.close()

    simulated["value"] = observation("B")
    b = Project.open(path)
    second = b.environments.default
    assert second["environment_id"] != first["environment_id"]
    assert b.environments.list()[0]["environment_id"] == first["environment_id"]
    assert len(b.environments.list()) == 2
    b.close()

    simulated["value"] = observation("A")
    a = Project.open(path)
    try:
        assert a.environments.default["environment_id"] == first["environment_id"]
        assert a.environments.get(second["environment_id"])["readiness"] == "unverified"
        assert a.environments.set_default(second["environment_id"])["environment_id"] == second["environment_id"]
        assert a.environments.default["environment_id"] == second["environment_id"]

        independent = Project.open(path)
        try:
            assert independent.environments.default["environment_id"] == first["environment_id"]
            assert len(independent.environments.list()) == 2
        finally:
            independent.close()
        with pytest.raises(EnvironmentNotFoundError):
            a.environments.set_default("not-registered")
        assert a.environments.default["environment_id"] == second["environment_id"]
    finally:
        a.close()

    reopened = Project.open(path)
    try:
        assert reopened.environments.default["environment_id"] == first["environment_id"]
        assert len(reopened.environments.list()) == 2
    finally:
        reopened.close()


def test_drift_and_same_name_never_mutate_history(tmp_path, simulated):
    path = tmp_path / "project"
    with Project.create(path, "project") as project:
        first = project.environments.default
        original = json.dumps(first["observation"], sort_keys=True)
    simulated["value"] = observation("A", package_version="2")
    with Project.create(path, "project") as project:
        assert len(project.environments.list()) == 2
        assert project.environments.get(first["environment_id"])["observation"] == json.loads(original)
        assert project.environments.default["environment_id"] != first["environment_id"]
    simulated["value"] = observation("A", name="sample", package_version="3")
    with Project.open(path) as project:
        assert len(project.environments.list()) == 3
    simulated["value"] = observation("A", package_version="2")
    with Project.open(path) as project:
        assert len(project.environments.list()) == 3


def test_legacy_project_operations_remain_unassigned(tmp_path, simulated):
    path = tmp_path / "old"
    with Project.create(path, "old") as project:
        project.catalog.register_operator(operator_id="optr_000001", operation_type="translate")
        project.catalog.register_operation(
            operation_id="run_000001",
            operation_type="translate",
            operator_id="optr_000001",
            status="complete",
        )
    db = path / ".teal" / "catalog" / "catalog.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("DROP TABLE project_environments")
        before = con.execute("SELECT * FROM operations").fetchall()
    with Project.open(path) as project:
        assert len(project.environments.list()) == 1
        assert project.get_operation("run_000001") is not None
        assert project.get_operation("run_000001")["environment_id"] is None
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT * FROM operations").fetchall() == before


def test_concurrent_open_reuses_single_registry_identity(tmp_path, simulated):
    path = tmp_path / "project"
    with Project.create(path, "project"):
        pass
    simulated["value"] = observation("C")
    barrier = threading.Barrier(8)

    def open_once(_):
        barrier.wait(timeout=20)
        with Project.open(path) as project:
            return project.environments.default["environment_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(open_once, range(8)))
    assert len(set(ids)) == 1
    with Project.open(path) as project:
        assert len(project.environments.list()) == 2


def test_registration_failure_rolls_back_without_changing_existing_records(
    tmp_path, simulated
):
    path = tmp_path / "project"
    with Project.create(path, "project") as project:
        first = project.environments.default
    simulated["value"] = observation("D")
    # SQL trigger simulates a write failure before INSERT completion.
    db = path / ".teal" / "catalog" / "catalog.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TRIGGER reject_new_environment BEFORE INSERT ON project_environments "
            "BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"
        )
    with pytest.raises(sqlite3.DatabaseError, match="simulated interruption"):
        Project.open(path)
    with sqlite3.connect(db) as con:
        con.execute("DROP TRIGGER reject_new_environment")
    simulated["value"] = observation("A")
    with Project.open(path) as project:
        assert project.environments.get(first["environment_id"])["observation"] == first["observation"]
        assert len(project.environments.list()) == 1


def test_corrupt_registry_is_explicit_not_silently_rewritten(tmp_path, simulated):
    path = tmp_path / "project"
    with Project.create(path, "project") as project:
        first = project.environments.default["environment_id"]
    db = path / ".teal" / "catalog" / "catalog.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE project_environments SET observation_json = ? WHERE environment_id = ?",
            ('{"schema_version":1}', first),
        )
    with pytest.raises(ValueError, match="conflicts"):
        Project.open(path)


def test_versioned_fingerprint_changes_with_relevant_details():
    a = observation("A")
    digest = fingerprint_observation(a)
    assert len(digest) == 64
    assert digest == fingerprint_observation(json.loads(json.dumps(a, sort_keys=True)))
    assert digest != fingerprint_observation(observation("A", package_version="2"))
    with pytest.raises(EnvironmentRegistryError):
        fingerprint_observation({"schema_version": 99})


def test_direct_urls_are_sanitized_without_reading_local_sources(tmp_path):
    class FakeDistribution:
        version = "1.0"
        metadata = {"Name": "local-test"}

        def __init__(self, url):
            self.url = url

        def read_text(self, name):
            assert name == "direct_url.json"
            return self.url

    source = tmp_path / "source"
    source.mkdir()
    local = FakeDistribution(json.dumps({
        "url": source.as_uri(), "dir_info": {"editable": True},
    }))
    first = env_module._distribution_record(local)
    assert first == {
        "name": "local-test",
        "version": "1.0",
        "direct_install": True,
        "editable": True,
        "local_install": True,
    }
    (source / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert env_module._distribution_record(local) == first
    remote = env_module._distribution_record(FakeDistribution(json.dumps({
        "url": "https://user:secret@example.test/archive.whl?token=private",
        "archive_info": {"hashes": {"sha256": "b" * 64}},
        "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
    })))
    assert "secret" not in json.dumps(remote)
    assert "token" not in json.dumps(remote)
    assert remote["archive_sha256"] == "b" * 64
    assert remote["vcs_commit"] == "a" * 40


def test_observe_installed_inventory_and_ignore_model_and_json_files(
    tmp_path, monkeypatch
):
    """A package-installed spaCy model is ordinary distribution metadata.

    Downloaded model assets and scientific rules are not part of the registry.
    """
    class FakeDistribution:
        metadata = {"Name": "en_core_web_sm"}

        def __init__(self, version):
            self.version = version

        def read_text(self, name):
            return None

    versions = {"model": "3.8.0", "extra": False}
    calls = {"count": 0}

    def installed():
        calls["count"] += 1
        entries = [FakeDistribution(versions["model"])]
        if versions["extra"]:
            entries.append(type("OtherDistribution", (), {
                "metadata": {"Name": "New-Package"},
                "version": "2.0",
                "read_text": lambda self, name: None,
            })())
        return entries

    signature = {"value": "initial"}
    monkeypatch.setattr(env_module, "_DISTRIBUTION_CACHE", None)
    monkeypatch.setattr(
        env_module, "_distribution_metadata_signature",
        lambda: (signature["value"],),
    )
    monkeypatch.setattr(env_module.metadata, "distributions", installed)
    first = env_module.observe_environment()
    packages = first["distributions"]
    assert {"name": "en-core-web-sm", "version": "3.8.0"} in packages
    assert "source_sha256" not in first["teal"]
    assert env_module.observe_environment() == first
    assert calls["count"] == 1, "unchanged metadata should reuse the inventory"

    # Changes to downloaded model assets and scientific JSON files are not
    # changes to the installed Python-distribution inventory.
    model = tmp_path / "model-download" / "weights.bin"
    model.parent.mkdir()
    model.write_bytes(b"old weights")
    rules = tmp_path / "linguistic-rules.json"
    rules.write_text('{"rule": 1}', encoding="utf-8")
    source = tmp_path / "module.py"
    source.write_text("value = 1", encoding="utf-8")
    baseline = env_module.fingerprint_observation(first)
    model.write_bytes(b"new weights")
    rules.write_text('{"rule": 2}', encoding="utf-8")
    source.write_text("value = 2", encoding="utf-8")
    assert env_module.fingerprint_observation(env_module.observe_environment()) == baseline

    versions["model"] = "3.9.0"
    signature["value"] = "model-version-changed"
    changed = env_module.observe_environment()
    assert env_module.fingerprint_observation(changed) != baseline
    assert {"name": "en-core-web-sm", "version": "3.9.0"} in changed["distributions"]

    versions["extra"] = True
    signature["value"] = "package-added"
    added = env_module.observe_environment()
    assert env_module.fingerprint_observation(added) != env_module.fingerprint_observation(changed)
    assert {"name": "new-package", "version": "2.0"} in added["distributions"]

    versions["extra"] = False
    signature["value"] = "package-removed"
    assert env_module.observe_environment() == changed


def test_discovery_sees_import_path_and_current_directory_changes(monkeypatch, tmp_path):
    """Standard metadata discovery, including sys.path's empty current-dir entry."""
    import importlib
    from importlib import metadata as real_metadata

    pkg_dir = tmp_path / "project"
    pkg_dir.mkdir()
    info = pkg_dir / "special-model-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: special-model\nVersion: 1.0\n",
        encoding="utf-8",
    )
    # Do not monkeypatch metadata.distributions here: exercise real discovery.
    with monkeypatch.context() as patch:
        patch.chdir(pkg_dir)
        patch.syspath_prepend(str(pkg_dir))
        patch.syspath_prepend("")
        importlib.invalidate_caches()
        cwd_signature = env_module._distribution_metadata_signature()
        found = [
            {"name": d.metadata.get("Name"), "version": d.version}
            for d in real_metadata.distributions()
            if d.metadata.get("Name") == "special-model"
        ]
        assert {"name": "special-model", "version": "1.0"} in found
        inventory = env_module.observe_environment()["distributions"]
        assert {"name": "special-model", "version": "1.0"} in inventory

        # Moving the cwd changes what an empty sys.path entry means.
        other_dir = tmp_path / "another-project"
        other_dir.mkdir()
        other_info = other_dir / "second-model-2.0.dist-info"
        other_info.mkdir()
        (other_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: second-model\nVersion: 2.0\n",
            encoding="utf-8",
        )
        patch.chdir(other_dir)
        assert env_module._distribution_metadata_signature() != cwd_signature
        patch.syspath_prepend(str(other_dir))
        importlib.invalidate_caches()
        other_inventory = env_module.observe_environment()["distributions"]
        assert {"name": "second-model", "version": "2.0"} in other_inventory
        assert {"name": "special-model", "version": "1.0"} in other_inventory


def test_missing_historical_local_source_does_not_block_registry(tmp_path, simulated):
    path = tmp_path / "project"
    with Project.create(path, "project") as project:
        previous = project.environments.default["environment_id"]
    simulated["value"] = observation("B")
    with Project.open(path) as project:
        assert project.environments.get(previous)["readiness"] == "unverified"
        assert len(project.environments.list()) == 2




@pytest.mark.skipif(sys.platform == "win32", reason="separate-venv smoke is covered on POSIX CI")
def test_real_independent_venv_opening_is_a_distinct_observation(tmp_path, monkeypatch):
    """Actual second interpreter, with inherited dependencies, not a mocked execution route."""
    project_path = tmp_path / "study"
    with Project.create(project_path, "study") as project:
        parent_id = project.environments.default["environment_id"]
    venv_path = tmp_path / "independent"
    # CI's uv-created .venv is not a system site-package directory.
    # Build a distinct interpreter environment and expose only the parent's
    # installed distributions for this import-based integration smoke.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(venv_path)
    python = venv_path / "bin" / "python"
    site_result = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    dist_info = Path(site_result.stdout.strip()) / "r0a_probe-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: r0a-probe\nVersion: 1.0\n",
        encoding="utf-8",
    )
    script = (
        "from text_analysis_lab import Project; import sys; "
        "from importlib import metadata; "
        "assert sys.prefix == sys.argv[2]; "
        "assert metadata.version('r0a-probe') == '1.0'; "
        "p=Project.open(sys.argv[1]); "
        "print(p.environments.default['environment_id']); p.close()"
    )
    child_env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    parent_sites = [
        entry for entry in sys.path
        if entry
        and Path(entry).is_dir()
        and ("site-packages" in Path(entry).parts or "dist-packages" in Path(entry).parts)
    ]
    assert parent_sites, "Cannot locate the test runner's installed dependencies."
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), *parent_sites, child_env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [str(python), "-c", script, str(project_path), str(venv_path)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"Child interpreter failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    child_id = result.stdout.strip().splitlines()[-1]
    # The child really has a distinct distribution inventory.
    assert child_id.startswith("env_")
    assert child_id != parent_id
    with Project.open(project_path) as project:
        assert project.environments.get(child_id)["environment_id"] == child_id
        assert project.environments.default["environment_id"] == parent_id
