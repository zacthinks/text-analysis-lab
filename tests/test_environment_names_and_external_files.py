"""Named environments and external dependency inventories (no TeAL injection)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from text_analysis_lab import Project
from text_analysis_lab.core import environments, external_environments, managed_environments
from text_analysis_lab.core.environments import (
    EnvironmentNotFoundError,
    EnvironmentRegistryError,
)


def _observation(version="1", *, teal=True):
    distributions = [{"name": "example", "version": version}]
    if teal:
        distributions.append({"name": "text-analysis-lab", "version": "0.12.2"})
    return {
        "schema_version": 2,
        "kind": "observed_unlocked",
        "python": {"implementation": "cpython", "version": "3.11.15"},
        "teal": {"version": "0.12.2" if teal else "0+unknown"},
        "distributions": distributions,
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(environments, "observe_environment", lambda: _observation())
    instance = Project.create(tmp_path / "project", "Test")
    yield instance
    instance.close()


def test_automatic_name_is_persisted_and_selectable(project):
    original = project.environments.opening_id
    item = project.environments.get(original)
    generated = "env-" + item["fingerprint"][:8]
    assert item["name"] == generated
    assert item["names"] == [generated]
    assert project.environments.default["name"] == generated
    assert project.environments.get(generated)["environment_id"] == original
    assert project.environments.get(generated.upper())["environment_id"] == original
    assert project.catalog.environment_names_for(original) == [generated]
    with pytest.raises(EnvironmentRegistryError, match="reserved"):
        project.environments._assign_name(original, "env-my-choice")
    with pytest.raises(EnvironmentRegistryError, match="reserved"):
        project.environments.rename(generated, "ENV-MY-CHOICE")


def test_generated_name_is_replaced_by_explicit_name_and_rename(project):
    original = project.environments.opening_id
    generated = project.environments.get(original)["name"]
    project.environments._assign_name(original, "CPU")
    assert project.environments.get(original)["names"] == ["CPU"]
    with pytest.raises(EnvironmentNotFoundError):
        project.environments.get(generated)

    renamed = project.environments.rename("cpu", "Primary CPU")
    assert renamed["name"] == "Primary CPU"
    assert renamed["names"] == ["Primary CPU"]
    with Project.open(project.path) as reopened:
        assert reopened.environments.get(original)["name"] == "Primary CPU"
        with pytest.raises(EnvironmentNotFoundError):
            reopened.environments.get(generated)


def test_renaming_generated_name_does_not_regenerate(project):
    original = project.environments.opening_id
    auto = project.environments.get(original)["name"]
    renamed = project.environments.rename(auto, "GPU")
    assert renamed["name"] == "GPU"
    assert renamed["names"] == ["GPU"]
    with Project.open(project.path) as reopened:
        assert reopened.environments.get(original)["name"] == "GPU"
        assert reopened.environments.get(original)["names"] == ["GPU"]
        with pytest.raises(EnvironmentNotFoundError):
            reopened.environments.get(auto)


def test_legacy_unnamed_environments_are_backfilled_safely(project):
    original = project.environments.opening_id
    original_fp = project.environments.get(original)["fingerprint"]
    second = environments.normalize_environment(_observation("15"))
    second_id = project.catalog.register_environment(
        fingerprint=environments.fingerprint_observation(second),
        schema_version=2,
        observation_json=environments._canonical_json(second),
    )
    project.catalog.con.execute(
        "DELETE FROM environment_names WHERE environment_id IN (?, ?)",
        (original, second_id),
    )
    project.catalog.con.commit()
    assert project.catalog.environment_names_for(original) == []
    assert project.catalog.environment_names_for(second_id) == []
    with Project.open(project.path) as reopened:
        rows = reopened.environments.list()
        for identifier in (original, second_id):
            name = reopened.environments.get(identifier)["name"]
            assert name.startswith("env-")
            assert reopened.environments.get(name)["environment_id"] == identifier
        assert reopened.environments.get(original)["name"] == "env-" + original_fp[:8]
    with Project.open(project.path) as reopened:
        assert reopened.environments.get(original)["names"] == ["env-" + original_fp[:8]]
        assert len(reopened.environments.get(second_id)["names"]) == 1


def test_generated_short_hash_extends_on_collision(project):
    first_fingerprint = "a83f12c9" + "0" * 56
    second_fingerprint = "a83f12c9" + "1" * 56
    first = project.catalog.register_environment(
        fingerprint=first_fingerprint,
        schema_version=2,
        observation_json='{"test":1}',
    )
    second = project.catalog.register_environment(
        fingerprint=second_fingerprint,
        schema_version=2,
        observation_json='{"test":2}',
    )
    assert project.catalog.environment_names_for(first) == ["env-a83f12c9"]
    assert project.catalog.environment_names_for(second) == ["env-a83f12c91"]
    assert first != second
    # Subsequent registration does not create extra aliases.
    project.catalog.register_environment(
        fingerprint=second_fingerprint,
        schema_version=2,
        observation_json='{"test":2}',
    )
    assert project.catalog.environment_names_for(second) == ["env-a83f12c91"]


def test_parallel_environment_registration_preserves_unique_names(project):
    from concurrent.futures import ThreadPoolExecutor
    from text_analysis_lab.core.catalog import ProjectCatalog

    catalog_dir = project.storage.catalog_dir
    fingerprints = ["b76c90ab" + str(i) * 56 for i in range(4)]
    # Initialize catalogs sequentially, then let each worker open its own
    # thread-local SQLite connection and race the actual registration.
    catalogs = [ProjectCatalog(catalog_dir) for _ in range(4)]
    for catalog in catalogs:
        catalog.close()

    def register(i):
        other = catalogs[i]
        try:
            return other.register_environment(
                fingerprint=fingerprints[i],
                schema_version=2,
                observation_json='{"concurrent":' + str(i) + '}',
            )
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = list(executor.map(register, range(4)))

    assert len(set(identities)) == 4
    aliases = [
        project.catalog.environment_names_for(identifier)[0]
        for identifier in identities
    ]
    assert len(set(aliases)) == 4
    assert all(alias.startswith("env-b76c90ab") for alias in aliases)
    assert sorted(len(alias) for alias in aliases) == [12, 13, 13, 13]


def test_unnamed_creation_returns_generated_name_for_both_apis(
    project, tmp_path, monkeypatch,
):
    observation = environments.normalize_environment(_observation("30"))
    identity = project.catalog.register_environment(
        fingerprint=environments.fingerprint_observation(observation),
        schema_version=2,
        observation_json=environments._canonical_json(observation),
    )
    monkeypatch.setattr(
        managed_environments, "create",
        lambda registry, **kwargs: {"environment_id": identity},
    )
    monkeypatch.setattr(
        external_environments, "create_from_file",
        lambda registry, source, **kwargs: {"environment_id": identity},
    )
    managed = project.environments.create()
    imported = project.environments.create_from_file(tmp_path / "example.lock")
    name = "env-" + project.environments.get(identity)["fingerprint"][:8]
    assert managed["name"] == name
    assert imported["name"] == name
    assert project.environments.get(name)["environment_id"] == identity

    # Supplying an explicit alias makes it the display name and drops the hash alias.
    explicit = project.environments.create(name="GPU")
    assert explicit["name"] == "GPU"
    assert project.environments.get(identity)["names"] == ["GPU"]
    with pytest.raises(EnvironmentNotFoundError):
        project.environments.get(name)


def test_backfill_preserves_existing_user_assigned_name(project):
    identity = project.environments.opening_id
    project.catalog.con.execute(
        "DELETE FROM environment_names WHERE environment_id = ?", (identity,),
    )
    project.catalog.con.execute(
        "INSERT INTO environment_names (name, name_key, environment_id) "
        "VALUES ('My CPU', 'my cpu', ?)", (identity,),
    )
    project.catalog.con.commit()
    with Project.open(project.path) as reopened:
        assert reopened.environments.get(identity)["name"] == "My CPU"
        assert reopened.environments.get(identity)["names"] == ["My CPU"]
        assert reopened.environments.get("mY cPu")["environment_id"] == identity


def test_names_persist_are_case_insensitive_and_leave_ids_unchanged(project):
    original = project.environments.default["environment_id"]
    project.environments._assign_name(original, "GPU")
    assert project.environments.get("gpu")["environment_id"] == original
    assert project.environments.set_default("gPu")["environment_id"] == original
    assert project.environments.default["name"] == "GPU"
    renamed = project.environments.rename("gpu", "NVIDIA GPU")
    assert renamed["environment_id"] == original
    assert renamed["fingerprint"] == project.environments.get(original)["fingerprint"]
    assert project.environments.get("nvidia gpu")["environment_id"] == original
    with pytest.raises(EnvironmentNotFoundError):
        project.environments.get("GPU")
    with pytest.raises(EnvironmentRegistryError, match="already in use"):
        project.environments._assign_name(
            "env_" + "a" * 64, "nvidia gpu"
        )
    with pytest.raises(EnvironmentRegistryError, match="reserved"):
        project.environments._assign_name(original, "env_not_an_identifier")
    with pytest.raises(EnvironmentRegistryError, match="whitespace"):
        project.environments.rename("NVIDIA GPU", " NVIDIA GPU")
    reopened = Project.open(project.path)
    try:
        assert reopened.environments.get("NVIDIA GPU")["environment_id"] == original
        assert reopened.environments.default["environment_id"] == reopened.environments.opening_id
        assert reopened.environments.get(original)["name"] == "NVIDIA GPU"
    finally:
        reopened.close()


def test_rename_preserves_operation_provenance(project, tmp_path):
    original = project.environments.default["environment_id"]
    project.environments._assign_name(original, "CPU")
    csv = tmp_path / "sample.csv"
    csv.write_text("text\nhello world\n", encoding="utf-8")
    artifact = project.read_csv(csv, text_fields="text", metadata_fields=None)
    operation = project.operation_for_artifact(artifact)
    assert operation["environment_id"] == original
    before = project.catalog.get_operation(operation["operation_id"])
    project.environments.rename("CPU", "Main CPU")
    after = project.catalog.get_operation(operation["operation_id"])
    assert after["environment_id"] == before["environment_id"] == original
    assert after["platform_id"] == before["platform_id"]
    assert project.environments.get("Main CPU")["environment_id"] == original


def test_distinct_names_are_unique_across_identities(project):
    first = project.environments.default["environment_id"]
    second_observation = environments.normalize_environment(_observation("2"))
    second = project.catalog.register_environment(
        fingerprint=environments.fingerprint_observation(second_observation),
        schema_version=2,
        observation_json=environments._canonical_json(second_observation),
    )
    project.environments._assign_name(first, "GPU")
    project.environments._assign_name(second, "Legacy")
    with pytest.raises(EnvironmentRegistryError, match="already in use"):
        project.environments.rename("Legacy", "gpu")
    assert project.environments.get("LEGACY")["environment_id"] == second
    project.environments.set_default("LEGACY")
    assert project.environments.resolve() == second
    assert project.environments.resolve("GPU") == first


def test_named_execution_does_not_silently_use_parent(project, monkeypatch):
    other = environments.normalize_environment(_observation("2"))
    external_id = project.catalog.register_environment(
        fingerprint=environments.fingerprint_observation(other),
        schema_version=2, observation_json=environments._canonical_json(other),
    )
    project.environments._assign_name(external_id, "Legacy")
    called = []

    def delegate(*args, **kwargs):
        called.append(args[1])
        raise EnvironmentRegistryError("Incompatible external environment")

    import text_analysis_lab.core.secondary_coordinator as coordinator
    monkeypatch.setattr(coordinator, "delegate_translate", delegate)
    with pytest.raises(EnvironmentRegistryError, match="Incompatible"):
        project.translate(object(), "not-an-artifact", environment="legacy")
    assert called == [external_id]


@pytest.fixture
def fake_external(tmp_path, monkeypatch):
    root = tmp_path / "user-cache"
    monkeypatch.setattr(managed_environments, "_cache_root", lambda: root)
    monkeypatch.setattr(managed_environments, "_uv", lambda: "uv")
    calls = []

    def command(args, **kwargs):
        calls.append(list(args))
        if args[1] == "venv":
            executable = managed_environments._interpreter(Path(args[2]))
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"fake-python")
            executable.chmod(0o755)
        return ""

    monkeypatch.setattr(managed_environments, "_run_command", command)
    monkeypatch.setattr(
        external_environments, "_probe",
        lambda exe, python: _observation("9", teal=False),
    )
    return root, calls


def test_external_requirements_install_and_reuse_without_teal(project, tmp_path, fake_external):
    root, calls = fake_external
    source = tmp_path / "requirements.txt"
    source.write_text("example==9\n", encoding="utf-8")
    first = project.environments.create_from_file(source, name="Legacy", python="3.11.15")
    with pytest.warns(UserWarning, match="already exists"):
        second = project.environments.create_from_file(
            source, name="Legacy", python="3.11.15",
        )
    assert first["environment_id"] == second["environment_id"]
    assert second == project.environments.get("legacy")
    assert second["name"] == "Legacy"
    assert first["source"]["kind"] == "requirements"
    assert first["portability"].startswith("observed_unlocked")
    assert project.environments.get("legacy")["kind"] == "observed_unlocked"
    assert project.environments.get("Legacy")["observation"]["teal"]["version"] == "0+unknown"
    assert len([c for c in calls if c[:2] == ["uv", "venv"]]) == 1
    assert len([c for c in calls if c[:3] == ["uv", "pip", "install"]]) == 1
    assert any((target / "external.json").exists() for target in root.iterdir() if target.is_dir())
    reopened = Project.open(project.path)
    try:
        assert reopened.environments.get("Legacy")["environment_id"] == first["environment_id"]
        assert reopened.environments.bound_interpreter("LEGACY").is_file()
    finally:
        reopened.close()


def test_uv_lock_install_keeps_external_portability_separate(project, tmp_path, fake_external):
    _, calls = fake_external
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrequires-python = ">=3.11"\n'
        '[[package]]\nname = "example"\nversion = "0.1.0"\n'
        'source = { virtual = "." }\n',
        encoding="utf-8",
    )
    planned = project.environments.plan_from_file(lock, python="3.11.15")
    assert planned["kind"] == "external_uv_lock"
    first = project.environments.create_from_file(
        lock, name="Experiment", python="3.11.15",
    )
    assert first["source"]["sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert any(c[:2] == ["uv", "sync"] and "--frozen" in c for c in calls)
    cache_root = managed_environments._cache_root()
    generated = next(cache_root.glob("*/source/pyproject.toml"))
    assert 'name = "example"' in generated.read_text()
    assert 'requires-python = ">=3.11"' in generated.read_text()
    assert project.environments.get("Experiment")["kind"] == "observed_unlocked"
    assert project.environments.get("Experiment")["sources"][0]["specification_kind"] == "external_uv_lock"


def test_source_failure_has_no_environment_registration(project, tmp_path, fake_external, monkeypatch):
    root, calls = fake_external
    source = tmp_path / "requirements.txt"
    source.write_text("example==9\n")
    before = len(project.environments.list())

    def failure(args, **kwargs):
        if args[1:3] == ["pip", "install"]:
            raise managed_environments.ManagedEnvironmentError("uv resolver failed")
        if args[1] == "venv":
            executable = managed_environments._interpreter(Path(args[2]))
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"fake")
            executable.chmod(0o755)
        return ""

    monkeypatch.setattr(managed_environments, "_run_command", failure)
    with pytest.raises(managed_environments.ManagedEnvironmentError, match="resolver"):
        project.environments.create_from_file(source, python="3.11.15")
    assert len(project.environments.list()) == before
    assert not any((d / "external.json").exists() for d in root.iterdir() if d.is_dir())



def test_unnamed_python_uses_lock_constraint(project, tmp_path, fake_external):
    _, calls = fake_external
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrequires-python = ">=3.11"\n'
        '[[package]]\nname = "example"\nversion = "0.1.0"\n'
        'source = { virtual = "." }\n',
        encoding="utf-8",
    )
    result = project.environments.create_from_file(lock)
    assert result["python"] == "3.11.15"
    assert any(c[:2] == ["uv", "venv"] and c[-2:] == ["--python", ">=3.11"] for c in calls)


def test_explicit_python_conflict_with_lock_fails_before_install(project, tmp_path, fake_external):
    _, calls = fake_external
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\nrequires-python = ">=3.12"\n'
        '[[package]]\nname = "example"\nversion = "0.1.0"\n'
        'source = { virtual = "." }\n',
        encoding="utf-8",
    )
    with pytest.raises(managed_environments.ManagedEnvironmentError, match="conflicts"):
        project.environments.create_from_file(lock, python="3.11.15")
    assert calls == []


def test_unnamed_python_requirements_uses_uv_default(project, tmp_path, fake_external):
    _, calls = fake_external
    req = tmp_path / "requirements.txt"
    req.write_text("example>=1\n")
    project.environments.create_from_file(req)
    assert any(c[:2] == ["uv", "venv"] and "--python" not in c for c in calls)


def test_standalone_lock_without_root_fails_before_registration(project, tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\nrequires-python = ">=3.11"\npackage = []\n')
    before = len(project.environments.list())
    with pytest.raises(managed_environments.ManagedEnvironmentError, match="root project"):
        project.environments.create_from_file(lock)
    assert len(project.environments.list()) == before



def test_existing_name_preflights_before_provisioning_or_reading_file(
    project, tmp_path, monkeypatch,
):
    """Named reuse is a no-op, even if the supplied source no longer exists."""
    identity = project.environments.default["environment_id"]
    project.environments._assign_name(identity, "Notebook")

    def should_not_install(*args, **kwargs):
        raise AssertionError("Provisioning must not run for an existing name")

    monkeypatch.setattr(managed_environments, "create", should_not_install)
    monkeypatch.setattr(external_environments, "create_from_file", should_not_install)

    with pytest.warns(UserWarning, match="already exists"):
        managed_existing = project.environments.create(
            name="notebook", python="99.99.99", teal_extras=["invalid"],
        )
    with pytest.warns(UserWarning, match="already exists"):
        file_existing = project.environments.create_from_file(
            tmp_path / "missing.lock", name="NOTEBOOK", python="not-a-python",
        )

    expected = project.environments.get("Notebook")
    assert managed_existing == expected
    assert file_existing == expected
    assert expected["environment_id"] == identity


def test_existing_name_preflight_after_reopen(project, monkeypatch):
    """Aliases survive Project reopen and remain safe to reuse without uv."""
    identity = project.environments.opening_id
    project.environments._assign_name(identity, "Saved")
    with Project.open(project.path) as reopened:
        monkeypatch.setattr(
            managed_environments, "create",
            lambda *args, **kwargs: pytest.fail("unexpected managed installation"),
        )
        with pytest.warns(UserWarning, match="already exists"):
            reused = reopened.environments.create(name="saved")
        assert reused == reopened.environments.get("Saved")


def test_managed_name_reuse_and_collision(project, monkeypatch):
    first = project.environments.default["environment_id"]
    second_observation = environments.normalize_environment(_observation("2"))
    second = project.catalog.register_environment(
        fingerprint=environments.fingerprint_observation(second_observation),
        schema_version=2, observation_json=environments._canonical_json(second_observation),
    )
    from text_analysis_lab.core import managed_environments as managed
    results = iter([first, second])
    monkeypatch.setattr(
        managed, "create", lambda *args, **kwargs: {"environment_id": next(results)}
    )
    created = project.environments.create(name="Reusable")
    with pytest.warns(UserWarning, match="already exists"):
        again = project.environments.create(name="Reusable")
    assert again == project.environments.get("Reusable")
    assert again["environment_id"] == created["environment_id"]
    # A conflicting request also returns the existing name without provisioning.
    with pytest.warns(UserWarning, match="already exists"):
        conflict = project.environments.create(name="reusable", gpu=True)
    assert conflict["environment_id"] == created["environment_id"]


@pytest.mark.parametrize("bad", ["-r other.txt\n", "-e ./local\n", "../project\n"])
def test_rejects_incomplete_requirements_snapshots(tmp_path, bad):
    source = tmp_path / "requirements.txt"
    source.write_text(bad)
    with pytest.raises(managed_environments.ManagedEnvironmentError):
        external_environments.plan_from_file(source, python="3.11.15")


@pytest.mark.acceptance
def test_real_external_uv_and_freeze_on_current_platform(tmp_path, monkeypatch):
    """Install TeAL-free environments with real uv on Ubuntu and Windows."""
    import os
    import shutil
    import subprocess

    if os.environ.get("TEAL_REAL_MANAGED_ACCEPTANCE") != "1":
        pytest.skip("Real managed environments acceptance is opt-in.")
    if shutil.which("uv") is None:
        pytest.skip("uv is required")
    monkeypatch.setenv("TEAL_MANAGED_ENVIRONMENTS_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "no-such-parent-dependency"))
    python = __import__("platform").python_version()

    req = tmp_path / "requirements.txt"
    req.write_text("packaging==25.0\n", encoding="utf-8")
    external_project = tmp_path / "external"
    external_project.mkdir()
    (external_project / "pyproject.toml").write_text(
        '[project]\nname = "external-spec-test"\nversion = "0.1.0"\n'
        'requires-python = ">=3.11"\ndependencies = ["packaging==25.0"]\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["uv", "lock", "--directory", str(external_project)],
        check=True, capture_output=True, text=True,
    )
    # The recipient has only the emailed lockfile; no original pyproject.toml.
    delivered = tmp_path / "downloads"
    delivered.mkdir()
    copied_lock = delivered / "research-deps.lock"
    copied_lock.write_bytes((external_project / "uv.lock").read_bytes())
    assert not (delivered / "pyproject.toml").exists()
    with Project.create(tmp_path / "research", "external") as project:
        requirement = project.environments.create_from_file(
            req, name="Legacy", python=python,
        )
        uv_lock = project.environments.create_from_file(
            copied_lock, name="Experiment", python=python,
        )
        for item in (requirement, uv_lock):
            row = project.environments.get(item["environment_id"])
            assert row["kind"] == "observed_unlocked"
            assert row["observation"]["teal"]["version"] == "0+unknown"
            assert any(d["name"] == "packaging" for d in row["observation"]["distributions"])
            assert not any(
                d["name"] == "text-analysis-lab"
                for d in row["observation"]["distributions"]
            )
        assert project.environments.get("experiment")["environment_id"] == uv_lock["environment_id"]
        # Identical installed inventories are the same immutable identity,
        # even when the two source specifications differ.
        assert requirement["environment_id"] == uv_lock["environment_id"]
        with pytest.warns(UserWarning, match="already exists"):
            repeat_req = project.environments.create_from_file(
                req, name="Legacy", python=python,
            )
        with pytest.warns(UserWarning, match="already exists"):
            repeat_lock = project.environments.create_from_file(
                copied_lock, name="Experiment", python=python,
            )
        assert repeat_req["environment_id"] == requirement["environment_id"]
        assert repeat_lock["environment_id"] == uv_lock["environment_id"]
        executable = project.environments.bound_interpreter("Legacy")
        assert executable.is_file()

        # Explicitly selecting an incompatible child must not translate in the parent.
        from text_analysis_lab.core.secondary_coordinator import _verified_executable
        from text_analysis_lab.core.secondary_coordinator import SecondaryCoordinatorError
        with pytest.raises(SecondaryCoordinatorError, match="could not import or probe compatible TeAL"):
            _verified_executable(project, requirement["environment_id"])

    with Project.open(tmp_path / "research") as reopened:
        assert reopened.environments.get("LEGACY")["environment_id"] == requirement["environment_id"]
        assert reopened.environments.get("EXPERIMENT")["environment_id"] == uv_lock["environment_id"]
        assert reopened.environments.bound_interpreter("Legacy").is_file()
