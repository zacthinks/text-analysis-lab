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
        "schema_version": 1,
        "python": {"implementation": "cpython", "version": "3.11.0"},
        "platform": {"system": "Test", "machine": "testing"},
        "teal": {"version": "0.12.2", "source_sha256": "a" * 64},
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
        assert "environment_id" not in project.get_operation("run_000001")
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


def test_direct_urls_are_sanitized_and_local_source_drift_is_detected(tmp_path):
    class FakeDistribution:
        version = "1.0"
        metadata = {"Name": "local-test"}

        def __init__(self, url):
            self.url = url

        def read_text(self, name):
            assert name == "direct_url.json"
            return self.url

    url = json.dumps({
        "url": (tmp_path / "source").as_uri(),
        "dir_info": {"editable": True},
    })
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    first = env_module._distribution_record(FakeDistribution(url))
    assert first["editable"] is True
    assert first["local_install"] is True
    assert str(source) not in json.dumps(first)
    (source / "module.py").write_text("x = 2\n", encoding="utf-8")
    second = env_module._distribution_record(FakeDistribution(url))
    assert first["local_source_sha256"] != second["local_source_sha256"]
    remote = env_module._distribution_record(FakeDistribution(json.dumps({
        "url": "https://user:secret@example.test/archive.whl?token=private",
        "archive_info": {"hashes": {"sha256": "b" * 64}},
    })))
    assert "secret" not in json.dumps(remote)
    assert "token" not in json.dumps(remote)
    assert remote["archive_sha256"] == "b" * 64


def test_missing_historical_local_source_does_not_block_registry(tmp_path, simulated):
    path = tmp_path / "project"
    with Project.create(path, "project") as project:
        previous = project.environments.default["environment_id"]
    simulated["value"] = observation("B")
    with Project.open(path) as project:
        assert project.environments.get(previous)["readiness"] == "unverified"
        assert len(project.environments.list()) == 2


def test_source_digest_prunes_venvs_and_caches_safe_source_reads(tmp_path, monkeypatch):
    root = tmp_path / "editable"
    source = root / "src" / "unit.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\\n", encoding="utf-8")
    ignored = root / ".venv" / "lib" / "library.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("giant package should not be touched\\n", encoding="utf-8")

    original_scandir = os.scandir

    def guard_scandir(path):
        if Path(path) == root / ".venv":
            raise AssertionError("source fingerprint traversed a virtual environment")
        return original_scandir(path)

    original_read_bytes = Path.read_bytes
    reads = []

    def tracking_read_bytes(path):
        if path == source:
            reads.append(path)
        if path == ignored:
            raise AssertionError("source fingerprint read an excluded dependency")
        return original_read_bytes(path)

    monkeypatch.setattr(os, "scandir", guard_scandir)
    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)

    first = env_module._source_digest(root)
    assert env_module._source_digest(root) == first
    assert len(reads) == 1, "identical source should reuse the cached digest"

    source.write_text("value = 2\\n", encoding="utf-8")
    second = env_module._source_digest(root)
    assert second != first
    assert len(reads) == 2, "modified source must invalidate cached digest"

    added = root / "src" / "new.py"
    added.write_text("another = 1\\n", encoding="utf-8")
    third = env_module._source_digest(root)
    assert third != second, "adding source must invalidate the digest"
    added.unlink()
    assert env_module._source_digest(root) == second, "removing source must invalidate it"


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
