"""Portable, resolver-backed uv project locks for TeAL's R2 managed interpreters.

A portable ID identifies a canonical frozen dependency resolution. A managed
installation remains a disposable, verified host-local realization of that ID.
This module intentionally supports CPU-compatible base/lightweight profiles
only; a lock does not establish universal CUDA/MPS/ROCm support.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
from typing import Any
from urllib.parse import parse_qs, urlsplit

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
from packaging.version import Version

from text_analysis_lab.core import managed_environments as managed
from text_analysis_lab.core.environments import (
    OBSERVATION_SCHEMA_VERSION,
    EnvironmentRegistryError,
    _canonical_json,
    _normalize_lock_manifest,
    _portable_index_reference,
    fingerprint_observation,
    normalize_environment,
)

LOCK_KIND = "uv-project-v1"
PROFILE_SCHEMA = 1
_ROOT = "teal-portable-contract"
_SOURCE = "https://github.com/zacthinks/text-analysis-lab.git"
_DIGEST = re.compile(r"^sha256:[a-fA-F0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_PYTHON = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SUPPORTED_EXTRAS = frozenset({"parallel"})


def _project_file(*, python: str, commit: str, extras: list[str]) -> str:
    suffix = "[" + ",".join(extras) + "]" if extras else ""
    requirement = f"text-analysis-lab{suffix} @ git+{_SOURCE}@{commit}"
    return ("[project]\n"
            f"name = {_quote(_ROOT)}\n"
            "version = \"0.0.0\"\n"
            f"requires-python = {_quote('==' + python)}\n"
            f"dependencies = [{_quote(requirement)}]\n\n"
            "[tool.uv]\npackage = false\n")


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return _quote(value)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(x) for x in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{_quote(k)} = {_toml_value(v)}" for k, v in sorted(value.items())
        ) + " }"
    raise EnvironmentRegistryError("Unsupported value in a frozen uv lock.")


def _lock_toml(lock: dict[str, Any]) -> str:
    """Lossless TOML serialization for uv's JSON-compatible lock values.

    All nested uv records are emitted as inline tables; only top-level package
    entries need TOML array-of-table syntax. Parse-roundtrip is mandatory.
    """
    lines = []
    for key, value in sorted(lock.items()):
        if key != "package":
            lines.append(f"{_quote(key)} = {_toml_value(value)}")
    for package in lock.get("package", []):
        lines.append("\n[[package]]")
        for key, value in sorted(package.items()):
            lines.append(f"{_quote(key)} = {_toml_value(value)}")
    result = "\n".join(lines) + "\n"
    if tomllib.loads(result) != lock:
        raise EnvironmentRegistryError("uv lock cannot be serialized without semantic drift.")
    return result


def _trusted_https(url: str) -> str:
    """Protect import against local paths, credentials and private network URLs."""
    return _portable_index_reference(url)


def _validate_source(source: Any, *, name: str) -> None:
    if not isinstance(source, dict) or len(source) != 1:
        raise EnvironmentRegistryError("Portable package must have one immutable source.")
    kind, value = next(iter(source.items()))
    if kind == "virtual" and name == _ROOT and value == ".":
        return
    if kind == "registry" and isinstance(value, str):
        _trusted_https(value)
        return
    if kind == "git" and name == "text-analysis-lab" and isinstance(value, str):
        parsed = urlsplit(value)
        parameters = parse_qs(parsed.query)
        if (parsed.scheme != "https" or parsed.netloc != "github.com"
                or parsed.path.rstrip("/") not in {
                    "/zacthinks/text-analysis-lab", "/zacthinks/text-analysis-lab.git"
                } or not _COMMIT.fullmatch(parsed.fragment)
                or set(parameters) - {"rev"}
                or ("rev" in parameters and parameters["rev"] != [parsed.fragment])):
            raise EnvironmentRegistryError("TeAL Git lock must reference a verified immutable commit.")
        return
    raise EnvironmentRegistryError(
        f"Unsupported portable source for {name!r}; local, editable and unpinned sources are forbidden."
    )


def _validate_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("hash"), str):
        raise EnvironmentRegistryError("Locked artifact is missing its integrity hash.")
    if not _DIGEST.fullmatch(artifact["hash"]):
        raise EnvironmentRegistryError("Locked artifact requires a SHA-256 digest.")
    url = artifact.get("url")
    if not isinstance(url, str):
        raise EnvironmentRegistryError("Locked artifact needs a portable URL.")
    _trusted_https(url)
    if "size" in artifact and (type(artifact["size"]) is not int or artifact["size"] < 0):
        raise EnvironmentRegistryError("Invalid locked artifact size.")


def validate_manifest(manifest: Any, *, python: str, commit: str, extras: list[str],
                      teal_version: str) -> dict[str, Any]:
    """Validate uv's resolver output without network access or external execution."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "uv_lock"}:
        raise EnvironmentRegistryError("Unsupported uv portable manifest fields.")
    if manifest["schema"] != PROFILE_SCHEMA:
        raise EnvironmentRegistryError("Unsupported TeAL portable-lock profile schema.")
    lock = manifest["uv_lock"]
    if (not isinstance(lock, dict) or type(lock.get("version")) is not int
            or lock["version"] != 1 or type(lock.get("revision", 0)) is not int
            or not isinstance(lock.get("requires-python"), str)
            or lock["requires-python"] != "==" + python
            or not isinstance(lock.get("package"), list)):
        raise EnvironmentRegistryError("Unsupported uv lock structure or Python target.")
    seen: set[tuple[str, str, str]] = set()
    root = teal = None
    packages: set[str] = set()
    for package in lock["package"]:
        if not isinstance(package, dict):
            raise EnvironmentRegistryError("Malformed package in uv lock.")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
                or not isinstance(version, str)):
            raise EnvironmentRegistryError("Portable uv lock has an invalid package name/version.")
        try:
            Version(version)
        except ValueError as exc:
            raise EnvironmentRegistryError("Invalid uv package version.") from exc
        _validate_source(source, name=name)
        source_key = _canonical_json(source)
        key = name, version, source_key
        if key in seen:
            raise EnvironmentRegistryError("Duplicate package record in uv lock.")
        seen.add(key)
        packages.add(name)
        if name == _ROOT:
            if root is not None or source != {"virtual": "."}:
                raise EnvironmentRegistryError("Unexpected uv portable root project.")
            root = package
        if name == "text-analysis-lab":
            if teal is not None or version != teal_version:
                raise EnvironmentRegistryError("TeAL version is inconsistent with frozen resolution.")
            git = source.get("git", "")
            if urlsplit(git).fragment != commit:
                raise EnvironmentRegistryError("TeAL immutable commit differs from lock.")
            teal = package
        for kind in ("sdist",):
            if kind in package:
                _validate_artifact(package[kind])
        for wheel in package.get("wheels", []):
            _validate_artifact(wheel)
        if "wheels" in package and not isinstance(package["wheels"], list):
            raise EnvironmentRegistryError("Invalid uv wheel alternatives.")
        for dep in package.get("dependencies", []):
            if not isinstance(dep, dict) or not isinstance(dep.get("name"), str):
                raise EnvironmentRegistryError("Malformed locked dependency edge.")
            if "marker" in dep:
                try:
                    Marker(dep["marker"])
                except (ValueError, TypeError) as exc:
                    raise EnvironmentRegistryError("Invalid locked dependency marker.") from exc
    if root is None or teal is None:
        raise EnvironmentRegistryError("A portable uv lock must contain root and pinned TeAL.")
    for package in lock["package"]:
        for dep in package.get("dependencies", []):
            if dep["name"] not in packages:
                raise EnvironmentRegistryError("A frozen dependency points to an absent package.")
    # The root must actually select the TeAL package and the frozen extras.
    root_deps = [d for d in root.get("dependencies", []) if d.get("name") == "text-analysis-lab"]
    requires_dist = root.get("metadata", {}).get("requires-dist", [])
    root_metadata = [d for d in requires_dist if d.get("name") == "text-analysis-lab"]
    if (len(root_deps) != 1 or len(root_metadata) != 1
            or sorted(root_metadata[0].get("extras", [])) != extras):
        raise EnvironmentRegistryError("The uv root dependency contradicts requested TeAL extras.")
    if any(x not in _SUPPORTED_EXTRAS for x in extras):
        raise EnvironmentRegistryError("Portable Torch/heavy extras are not yet supported.")
    if "resolution-markers" in lock:
        if not isinstance(lock["resolution-markers"], list):
            raise EnvironmentRegistryError("Invalid uv resolution forks.")
        try:
            for marker in lock["resolution-markers"]:
                Marker(marker)
        except (ValueError, TypeError) as exc:
            raise EnvironmentRegistryError("Invalid uv resolution condition.") from exc
    # Normalize the full parsed resolution, preserving branch/source semantics.
    result = {"schema": PROFILE_SCHEMA, "uv_lock": _normalize_lock_manifest(lock)}
    _lock_toml(result["uv_lock"])
    return result


def _specification(request: dict[str, Any], parsed_lock: dict[str, Any]) -> dict[str, Any]:
    manifest = validate_manifest(
        {"schema": PROFILE_SCHEMA, "uv_lock": parsed_lock},
        python=request["python"], commit=request["teal_commit"],
        extras=request["teal_extras"], teal_version=request["teal_version"],
    )
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "kind": "portable_locked",
        "python": {"implementation": "cpython", "version": request["python"]},
        "teal": {"version": request["teal_version"],
                 "source": {"vcs": "git", "commit": request["teal_commit"]}},
        "extras": request["teal_extras"],
        "lock": {"kind": LOCK_KIND, "manifest": manifest,
                 "sha256": hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()},
    }


def _request_for(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] != "portable_locked" or spec["lock"]["kind"] != LOCK_KIND:
        raise EnvironmentRegistryError("Expected an R2b uv project portable specification.")
    return {"python": spec["python"]["version"],
            "teal_commit": spec["teal"]["source"]["commit"],
            "teal_version": spec["teal"]["version"],
            "teal_extras": spec["extras"], "gpu": False}


def _materialize(spec: dict[str, Any], directory: Path) -> None:
    request = _request_for(spec)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(_project_file(
        python=request["python"], commit=request["teal_commit"],
        extras=request["teal_extras"]), encoding="utf-8")
    (directory / "uv.lock").write_text(
        _lock_toml(spec["lock"]["manifest"]["uv_lock"]), encoding="utf-8")


def _export_requirements(spec: dict[str, Any]) -> list[Requirement]:
    with tempfile.TemporaryDirectory(prefix="teal-lock-inspect-") as work:
        _materialize(spec, Path(work))
        output = managed._run_command([
            managed._uv(), "export", "--frozen", "--no-dev", "--no-emit-project",
            "--no-hashes", "--format", "requirements-txt", "--directory", work,
        ], timeout=120)
    parsed: list[Requirement] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "\\":
            continue
        if stripped.startswith(("--hash", "--index-url", "--extra-index-url")):
            continue
        if stripped.endswith("\\"):
            stripped = stripped[:-1].strip()
        if not stripped:
            continue
        try:
            parsed.append(Requirement(stripped))
        except ValueError as exc:
            raise EnvironmentRegistryError("uv exported an unsupported frozen requirement.") from exc
    return parsed


def matches_installed(spec: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Check a local installed distribution set against the applicable lock slice.

    uv export selects the frozen dependency graph. Marker evaluation is done
    for this OS and the *child* Python patch, not the parent's Python patch.
    This establishes metadata/lock conformance, not installed wheel-byte proof.
    """
    if (spec["python"] != observed["python"] or
            spec["teal"]["version"] != observed["teal"]["version"]):
        return False
    teals = [d for d in observed["distributions"] if d["name"] == "text-analysis-lab"]
    if len(teals) != 1 or teals[0].get("vcs_commit") != spec["teal"]["source"]["commit"]:
        return False
    environment = default_environment()
    major, minor, _ = spec["python"]["version"].split(".")
    environment.update({
        "python_version": f"{major}.{minor}",
        "python_full_version": spec["python"]["version"],
        "implementation_name": "cpython", "platform_python_implementation": "CPython",
        "extra": "",
    })
    expected = {}
    for req in _export_requirements(spec):
        if req.marker and not req.marker.evaluate(environment):
            continue
        name = re.sub(r"[-_.]+", "-", req.name).lower()
        if name in expected:
            return False
        if name == "text-analysis-lab":
            if not req.url or spec["teal"]["source"]["commit"] not in req.url:
                return False
            expected[name] = Version(spec["teal"]["version"])
        else:
            if req.url or len(req.specifier) != 1:
                return False
            pin = next(iter(req.specifier))
            if pin.operator != "==":
                return False
            expected[name] = Version(pin.version)
    actual = {}
    for dist in observed["distributions"]:
        if dist["name"] in actual or dist.get("editable") or dist.get("local_install"):
            return False
        if dist["name"] != "text-analysis-lab" and dist.get("vcs"):
            return False
        actual[dist["name"]] = Version(dist["version"])
    return actual == expected


def create_specification(registry: Any, *, python: str | None = None,
                         teal: str | None = None,
                         teal_extras: list[str] | tuple[str, ...] | None = None,
                         gpu: str | bool = False) -> dict[str, Any]:
    """Explicit network-resolution step, with no installation or cache publication."""
    if gpu is not False:
        raise managed.ManagedEnvironmentError(
            "Portable R2b currently supports CPU profiles only; GPU auto/strict "
            "must use observed managed environments until backend alternatives are locked."
        )
    request = managed._normalized_request(
        python=python, teal=teal, teal_extras=teal_extras, gpu=False,
    )
    if set(request["teal_extras"]) - _SUPPORTED_EXTRAS:
        raise managed.ManagedEnvironmentError(
            "Portable R2b supports base TeAL and lightweight 'parallel' extra only."
        )
    with tempfile.TemporaryDirectory(prefix="teal-lock-resolve-") as work:
        root = Path(work)
        (root / "pyproject.toml").write_text(_project_file(
            python=request["python"], commit=request["teal_commit"],
            extras=request["teal_extras"]), encoding="utf-8")
        managed._run_command([managed._uv(), "lock", "--directory", work], timeout=1800)
        parsed = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    specification = normalize_environment(_specification(request, parsed))
    environment_id = registry.register_portable_specification(specification)
    return {"environment_id": environment_id, "specification": specification,
            "readiness": "registered_not_installed", "lock_kind": LOCK_KIND}


def export_specification(registry: Any, environment_id: str, destination: str | Path) -> Path:
    spec = registry.get(environment_id)["observation"]
    _request_for(spec)
    path = Path(destination)
    if path.is_dir():
        raise EnvironmentRegistryError("Specification export destination must be a file.")
    data = {"environment_id": environment_id, "specification": spec}
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def import_specification(registry: Any, source: str | Path) -> str:
    path = Path(source)
    if not path.is_file() or path.stat().st_size > 8_000_000:
        raise EnvironmentRegistryError("Portable specification is missing or too large.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentRegistryError("Cannot read portable specification JSON.") from exc
    if not isinstance(data, dict) or set(data) != {"environment_id", "specification"}:
        raise EnvironmentRegistryError("Unexpected portable specification export schema.")
    spec = normalize_environment(data["specification"])
    _request_for(spec)
    environment_id = "env_" + fingerprint_observation(spec)
    if data["environment_id"] != environment_id:
        raise EnvironmentRegistryError("Portable specification identity was tampered with.")
    return registry.register_portable_specification(spec)


def _cache_request(environment_id: str) -> dict[str, Any]:
    return {"portable_environment_id": environment_id, "gpu": False,
            "schema_version": PROFILE_SCHEMA}


def _cache_target(environment_id: str) -> Path:
    key = managed._request_key(_cache_request(environment_id))
    return managed._cache_root() / key


def _read_marker(target: Path, environment_id: str) -> dict[str, Any]:
    if target.is_symlink() or not target.is_dir():
        raise managed.ManagedEnvironmentError("Unsafe portable cache directory.")
    marker = json.loads((target / "portable.json").read_text(encoding="utf-8"))
    if (not isinstance(marker, dict)
            or set(marker) != {"schema_version", "environment_id", "observed_fingerprint"}
            or marker["schema_version"] != PROFILE_SCHEMA
            or marker["environment_id"] != environment_id
            or not isinstance(marker["observed_fingerprint"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", marker["observed_fingerprint"])):
        raise managed.ManagedEnvironmentError("Malformed portable cache marker.")
    return marker


def _verify_cached(registry: Any, target: Path, environment_id: str,
                   spec: dict[str, Any]) -> dict[str, Any]:
    marker = _read_marker(target, environment_id)
    request = _request_for(spec)
    if managed._local_source_commit() != request["teal_commit"]:
        raise managed.ManagedEnvironmentError("Parent TeAL source commit is incompatible.")
    observed, gpu_state = managed._verify(target, request, managed._uv())
    if fingerprint_observation(observed) != marker["observed_fingerprint"]:
        raise managed.ManagedEnvironmentError("Portable installation inventory drifted.")
    if not matches_installed(spec, observed):
        raise managed.ManagedEnvironmentError("Installation does not match the portable uv lock.")
    if gpu_state["status"] == "available":
        raise managed.ManagedEnvironmentError("CPU-only portable profile verified an unexpected GPU.")
    return observed


def create_from_specification(registry: Any, environment_id: str) -> dict[str, Any]:
    """Explicitly reconstruct one frozen resolution in the existing user cache."""
    spec = registry.get(environment_id)["observation"]
    request = _request_for(spec)
    if request["teal_commit"] != managed._local_source_commit():
        raise managed.ManagedEnvironmentError("R1 requires the current immutable TeAL source commit.")
    uv = managed._uv()
    target = _cache_target(environment_id)
    lock_file = target.parent / "locks" / (target.name + ".lock")
    with managed._request_lock(lock_file):
        reused = False
        if target.is_dir() and (target / "portable.json").is_file():
            observed = _verify_cached(registry, target, environment_id, spec)
            reused = True
        else:
            if target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise managed.ManagedEnvironmentError("Invalid portable cache target.")
                shutil.rmtree(target)
            target.mkdir(parents=True)
            try:
                managed._run_command([uv, "venv", str(target), "--python", request["python"]])
                with tempfile.TemporaryDirectory(prefix="teal-lock-sync-") as work:
                    _materialize(spec, Path(work))
                    # A non-default project environment cannot be moved after
                    # installation (entry points may embed absolute paths).
                    managed._run_command([
                        uv, "sync", "--frozen", "--no-dev", "--no-default-groups",
                        "--no-install-project", "--directory", work,
                    ], timeout=1800, env_overrides={"UV_PROJECT_ENVIRONMENT": str(target)})
                observed, _ = managed._verify(target, request, uv)
                if not matches_installed(spec, observed):
                    raise managed.ManagedEnvironmentError(
                        "Frozen sync completed but installed packages violate the portable lock."
                    )
                (target / "portable.json.pending").write_text(json.dumps({
                    "schema_version": PROFILE_SCHEMA, "environment_id": environment_id,
                    "observed_fingerprint": fingerprint_observation(observed),
                }, sort_keys=True), encoding="utf-8")
                (target / "portable.json.pending").replace(target / "portable.json")
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise
        observed_id = registry._catalog.register_environment(
            fingerprint=fingerprint_observation(observed),
            schema_version=observed["schema_version"],
            observation_json=_canonical_json(observed),
        )
        registry._catalog.record_environment_realization(observed_id, environment_id)
        registry.bind_interpreter(environment_id, managed._interpreter(target))
        return {"environment_id": environment_id, "observed_environment_id": observed_id,
                "readiness": "verified_local", "verification_level":
                "uv_frozen_sync_and_installed_metadata" if not reused else "lock_and_metadata_reverified",
                "reused": reused, "gpu": {"status": "cpu_only"},
                "portability": "portable_resolution; not binary-identical across platforms"}


def find_cached_portable_interpreter(registry: Any, environment_id: str) -> Path | None:
    """Reopen a registered portable identity without installing or saving paths."""
    spec = registry.get(environment_id)["observation"]
    _request_for(spec)
    target = _cache_target(environment_id)
    if not target.exists():
        return None
    with managed._request_lock(target.parent / "locks" / (target.name + ".lock")):
        _verify_cached(registry, target, environment_id, spec)
    return managed._interpreter(target)