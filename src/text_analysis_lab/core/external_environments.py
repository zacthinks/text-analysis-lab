"""Install ordinary dependency files without assuming TeAL is present.

External requirements inventories and standalone third-party uv.lock files are NOT TeAL's
verified portable specifications. All physical installations and source snapshots
live in the user-level cache, never inside the research Project.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tomllib
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from text_analysis_lab.core import managed_environments as managed
from text_analysis_lab.core.environments import (
    OBSERVATION_SCHEMA_VERSION,
    _canonical_json,
    fingerprint_observation,
    normalize_environment,
)

_MAX_SPEC_BYTES = 8_000_000
_PYTHON = re.compile(r"^\d+\.\d+\.\d+$")
# The probe intentionally uses only Python's standard library: TeAL need not
# exist in the child. It mirrors core.environments._distribution_record.
_OBSERVATION_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
import re
import sys
from urllib.parse import urlsplit
records = []
for dist in metadata.distributions():
    name = str(dist.metadata.get('Name') or '').strip()
    version = str(dist.version or '').strip()
    if not name or not version:
        raise RuntimeError('Installed distribution has no name/version')
    row = {'name': re.sub(r'[-_.]+', '-', name).lower(), 'version': version}
    raw = dist.read_text('direct_url.json')
    if raw is not None:
        direct = json.loads(raw)
        if not isinstance(direct, dict):
            raise RuntimeError('Malformed installed distribution origin')
        row['direct_install'] = True
        if isinstance(direct.get('dir_info'), dict):
            row['editable'] = bool(direct['dir_info'].get('editable', False))
        vcs = direct.get('vcs_info')
        if isinstance(vcs, dict):
            if vcs.get('vcs') in ('git', 'hg', 'svn', 'bzr'):
                row['vcs'] = vcs['vcs']
            commit = vcs.get('commit_id')
            if isinstance(commit, str) and re.fullmatch(r'[a-fA-F0-9]{7,64}', commit):
                row['vcs_commit'] = commit.lower()
        archive = direct.get('archive_info')
        if isinstance(archive, dict) and isinstance(archive.get('hashes'), dict):
            digest = archive['hashes'].get('sha256')
            if isinstance(digest, str) and re.fullmatch(r'[a-fA-F0-9]{64}', digest):
                row['archive_sha256'] = digest.lower()
        origin = direct.get('url')
        if isinstance(origin, str) and urlsplit(origin).scheme == 'file':
            row['local_install'] = True
    records.append(row)
try:
    teal = metadata.version('text-analysis-lab')
except metadata.PackageNotFoundError:
    teal = '0+unknown'
print(json.dumps({
    'schema_version': 2, 'kind': 'observed_unlocked',
    'python': {'implementation': sys.implementation.name,
               'version': platform.python_version()},
    'teal': {'version': teal}, 'distributions': records,
}, sort_keys=True))
"""


def _bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise managed.ManagedEnvironmentError(f"Dependency file {path.name!r} is missing or unsafe.")
    if path.stat().st_size > _MAX_SPEC_BYTES:
        raise managed.ManagedEnvironmentError("Dependency specification exceeds the size limit.")
    return path.read_bytes()


def _standalone_lock_project(lock: dict[str, Any]) -> bytes:
    """Synthesize only the metadata uv needs to sync an independent lock.

    The original project specification is not available and must not be
    recreated or used to re-resolve. The lock's root package is authoritative.
    """
    packages = lock.get("package")
    if not isinstance(lock.get("version"), int) or not isinstance(packages, list):
        raise managed.ManagedEnvironmentError("uv.lock has no recognized resolution/package table.")
    roots = [
        package for package in packages
        if isinstance(package, dict)
        and isinstance(package.get("source"), dict)
        and any(package["source"].get(key) == "." for key in ("virtual", "editable"))
    ]
    if len(roots) != 1:
        raise managed.ManagedEnvironmentError(
            "Standalone uv.lock must identify one root project; "
            "workspace or local project metadata may be required."
        )
    root = roots[0]
    name, version = root.get("name"), root.get("version")
    python_range = lock.get("requires-python")
    if (not isinstance(name, str) or not name
            or not isinstance(version, str) or not version
            or not isinstance(python_range, str) or not python_range):
        raise managed.ManagedEnvironmentError(
            "uv.lock is missing a root project name, version, or Python constraint."
        )
    return (
        "[project]\n"
        + "name = " + json.dumps(name) + "\n"
        + "version = " + json.dumps(version) + "\n"
        + "requires-python = " + json.dumps(python_range) + "\n"
    ).encode("utf-8")


def _request(
    source: str | Path, *, python: str | None,
) -> tuple[dict[str, Any], bytes, bytes | None]:
    source_path = Path(source).expanduser().resolve()
    raw = _bytes(source_path)
    if python is not None and (
        not isinstance(python, str) or _PYTHON.fullmatch(python) is None
    ):
        raise managed.ManagedEnvironmentError("python must be an exact major.minor.patch version.")
    if source_path.name == "uv.lock" or source_path.suffix.lower() == ".lock":
        kind = "external_uv_lock"
        try:
            lock = tomllib.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise managed.ManagedEnvironmentError("Invalid uv.lock TOML.") from exc
        metadata_bytes = _standalone_lock_project(lock)
        python_range = lock["requires-python"]
        try:
            constraint = SpecifierSet(python_range)
        except InvalidSpecifier as exc:
            raise managed.ManagedEnvironmentError(
                "uv.lock has an invalid Python compatibility constraint."
            ) from exc
        if python is not None and not constraint.contains(python):
            raise managed.ManagedEnvironmentError(
                f"Python {python} conflicts with uv.lock's requires-python {python_range!r}."
            )
    else:
        kind = "requirements"
        metadata_bytes = None
        python_range = None
        try:
            lines = raw.decode("utf-8-sig").splitlines()
        except UnicodeError as exc:
            raise managed.ManagedEnvironmentError("Requirements file must be UTF-8.") from exc
        # Nested includes/local relative references would require capturing more
        # files than the supplied specification: reject rather than misstate its
        # provenance. Normal package pins, hashes, markers and indexes are kept.
        for line in lines:
            stripped = line.strip()
            if (not stripped or stripped.startswith("#")
                    or stripped == "\\"):
                continue
            if re.match(r"^(?:-r|-c|--requirement|--constraint)(?:\s|=|$)", stripped):
                raise managed.ManagedEnvironmentError(
                    "Recursive requirements/constraints need a self-contained input file."
                )
            if (stripped.startswith(("./", "../", "-e ", "--editable "))
                    or stripped.startswith("file:")):
                raise managed.ManagedEnvironmentError(
                    "Relative or editable local installs cannot be reproduced from one file."
                )
    request = {
        "schema": 2,
        "kind": kind,
        "python": python,
        "requires_python": python_range,
        "specification_sha256": hashlib.sha256(raw).hexdigest(),
        # Generated project metadata is not an independently supplied source.
        "metadata_sha256": None,
        "host_system": sys.platform,
        "host_architecture": platform.machine().lower(),
    }
    return request, raw, metadata_bytes


def plan_from_file(
    source: str | Path, *, python: str | None = None,
) -> dict[str, Any]:
    """Validate inputs without writing to the Project or installing packages."""
    request, _, _ = _request(source, python=python)
    return {
        **request,
        "actions": ["create user-cache uv venv", "install supplied dependencies",
                    "verify installed inventory and uv pip check",
                    "register observed software identity"],
        "portability": "observed_unlocked; external file is not a verified TeAL portable lock",
        "requires_explicit_create": True,
    }


def _probe(executable: Path, python: str | None) -> dict[str, Any]:
    output = managed._run_command(
        [str(executable), "-I", "-c", _OBSERVATION_PROBE], timeout=120,
    )
    try:
        observed = normalize_environment(json.loads(output.splitlines()[-1]))
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise managed.ManagedEnvironmentError("Cannot verify installed dependency inventory.") from exc
    if observed["kind"] != "observed_unlocked" or (python is not None and observed["python"]["version"] != python):
        raise managed.ManagedEnvironmentError("Managed Python does not match requested patch version.")
    return observed


def _verify(
    target: Path, request: dict[str, Any], uv: str,
) -> dict[str, Any]:
    marker_path = target / "external.json"
    if target.is_symlink() or not target.is_dir() or not marker_path.is_file():
        raise managed.ManagedEnvironmentError("Unverified external environment cache.")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise managed.ManagedEnvironmentError("Malformed external cache marker.") from exc
    if (not isinstance(marker, dict)
            or set(marker) != {"request", "observed_fingerprint"}
            or marker["request"] != request
            or not isinstance(marker["observed_fingerprint"], str)):
        raise managed.ManagedEnvironmentError("External environment cache request changed.")
    spec_root = target / "source"
    source_name = "uv.lock" if request["kind"] == "external_uv_lock" else "requirements.txt"
    if hashlib.sha256(_bytes(spec_root / source_name)).hexdigest() != request["specification_sha256"]:
        raise managed.ManagedEnvironmentError("Cached external dependency specification changed.")
    if request["kind"] == "external_uv_lock":
        try:
            lock = tomllib.loads(_bytes(spec_root / "uv.lock").decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise managed.ManagedEnvironmentError("Cached uv.lock is invalid.") from exc
        if _bytes(spec_root / "pyproject.toml") != _standalone_lock_project(lock):
            raise managed.ManagedEnvironmentError("Generated uv project metadata changed.")
    executable = managed._interpreter(target)
    managed._run_command([uv, "pip", "check", "--python", str(executable)], timeout=120)
    observed = _probe(executable, request["python"])
    if fingerprint_observation(observed) != marker["observed_fingerprint"]:
        raise managed.ManagedEnvironmentError("External environment installed inventory drifted.")
    return observed


def create_from_file(
    registry: Any, source: str | Path, *, python: str | None = None,
) -> dict[str, Any]:
    """Install exactly the supplied requirements or uv project dependencies.

    No TeAL version/source check, TeAL injection, or R1 import probe occurs.
    Installed distributions are registered as observed_unlocked, not locked.
    """
    request, raw, metadata_bytes = _request(source, python=python)
    uv = managed._uv()
    root = managed._cache_root()
    key = managed._request_key(request)
    target = root / key
    with managed._request_lock(root / "locks" / (key + ".lock")):
        reused = False
        if (target / "external.json").is_file():
            observed = _verify(target, request, uv)
            reused = True
        else:
            if target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise managed.ManagedEnvironmentError("Invalid external cache target.")
                shutil.rmtree(target)
            target.mkdir(parents=True)
            try:
                python_request = request["python"] or request["requires_python"]
                venv_args = [uv, "venv", str(target)]
                if python_request is not None:
                    venv_args.extend(["--python", python_request])
                managed._run_command(venv_args)
                executable = managed._interpreter(target)
                src_root = target / "source"
                src_root.mkdir()
                filename = "uv.lock" if request["kind"] == "external_uv_lock" else "requirements.txt"
                (src_root / filename).write_bytes(raw)
                if metadata_bytes is not None:
                    (src_root / "pyproject.toml").write_bytes(metadata_bytes)
                if request["kind"] == "external_uv_lock":
                    managed._run_command([
                        uv, "sync", "--frozen", "--no-dev", "--no-default-groups",
                        "--no-install-project", "--directory", str(src_root),
                    ], timeout=1800, env_overrides={"UV_PROJECT_ENVIRONMENT": str(target)})
                else:
                    managed._run_command([
                        uv, "pip", "install", "--python", str(executable),
                        "-r", str(src_root / filename),
                    ], timeout=1800)
                managed._run_command(
                    [uv, "pip", "check", "--python", str(executable)], timeout=120,
                )
                observed = _probe(executable, request["python"])
                marker = {
                    "request": request,
                    "observed_fingerprint": fingerprint_observation(observed),
                }
                pending = target / "external.json.pending"
                pending.write_text(_canonical_json(marker), encoding="utf-8")
                pending.replace(target / "external.json")
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise
        environment_id = registry._catalog.register_environment(
            fingerprint=fingerprint_observation(observed),
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observation_json=_canonical_json(observed),
        )
        registry._catalog.register_environment_source(
            environment_id, request["kind"], request["specification_sha256"],
            request["metadata_sha256"],
        )
        registry.bind_interpreter(environment_id, managed._interpreter(target))
        return {
            "environment_id": environment_id,
            "python": observed["python"]["version"],
            "readiness": "verified_local",
            "verification_level": "installed_inventory_and_uv_pip_check",
            "source": {
                "kind": request["kind"],
                "sha256": request["specification_sha256"],
                "metadata_sha256": request["metadata_sha256"],
            },
            "reused": reused,
            "portability": "observed_unlocked; no cross-platform lock guarantee",
        }


def find_cached_external_interpreter(environment_id: str) -> Path | None:
    """Rediscover only an exact-inventory cache, without installing packages."""
    if re.fullmatch(r"env_[a-f0-9]{64}", environment_id) is None:
        return None
    root = managed._cache_root()
    if not root.is_dir():
        return None
    first_failure: Exception | None = None
    for target in sorted(root.iterdir()):
        if target.is_symlink() or not target.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", target.name):
            continue
        marker_path = target / "external.json"
        if not marker_path.is_file():
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            request = marker["request"]
            if (marker.get("observed_fingerprint") != environment_id[4:]
                    or managed._request_key(request) != target.name):
                continue
            with managed._request_lock(root / "locks" / (target.name + ".lock")):
                observed = _verify(target, request, managed._uv())
                if fingerprint_observation(observed) == environment_id[4:]:
                    return managed._interpreter(target)
        except (OSError, ValueError, TypeError, KeyError, managed.ManagedEnvironmentError) as exc:
            first_failure = exc
    if first_failure is not None:
        raise managed.ManagedEnvironmentError(
            "External managed installation failed cache re-verification."
        ) from first_failure
    return None
