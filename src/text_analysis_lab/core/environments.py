"""Portable observations and session-local defaults for Project Python environments.

Registry membership is historical; it does not imply a runnable interpreter.
No machine-local executable path or installer state is stored in project data.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from threading import RLock
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.errors import TeALError

if TYPE_CHECKING:
    from text_analysis_lab.core.catalog import ProjectCatalog

OBSERVATION_SCHEMA_VERSION = 1
_NAME_SEPARATOR = re.compile(r"[-_.]+")
_SAFE_COMMIT = re.compile(r"^[a-fA-F0-9]{7,64}$")
_SOURCE_EXTENSIONS = {".py", ".toml", ".cfg"}
_SOURCE_FILES = {"uv.lock", "poetry.lock", "pdm.lock", "setup.py"}
_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", "node_modules", "build", "dist",
}


class EnvironmentRegistryError(TeALError):
    """An interpreter observation or persisted registry record is invalid."""


class EnvironmentNotFoundError(EnvironmentRegistryError, LookupError):
    """A selected environment is not registered with this project."""


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# Cache content digests only while all observed files and parent directories
# have unchanged metadata. File stats catch in-place edits; directory stats
# catch added/removed files. No time-based expiry or blind process-wide snapshot.
_SOURCE_DIGEST_CACHE: dict[
    Path, tuple[dict[Path, tuple[int, int]], dict[Path, tuple[int, int, int]], str]
] = {}
_SOURCE_DIGEST_LOCK = RLock()


def _source_digest(root: Path) -> str:
    """Hash relevant source by relative names/bytes, pruning ignored trees.

    A previous digest is reused only after validating directory and file
    metadata. This avoids repeatedly reading source contents for every Project
    open without missing ordinary edits, additions, or deletions.
    """
    root = Path(root).resolve()
    with _SOURCE_DIGEST_LOCK:
        cached = _SOURCE_DIGEST_CACHE.get(root)
        if cached is not None:
            directories, files, previous_digest = cached
            try:
                if all(
                    (st.st_mtime_ns, st.st_ctime_ns) == signature
                    for path, signature in directories.items()
                    for st in (path.stat(),)
                ) and all(
                    (st.st_mtime_ns, st.st_ctime_ns, st.st_size) == signature
                    for path, signature in files.items()
                    for st in (path.stat(),)
                ):
                    return previous_digest
            except OSError:
                pass

        digest = hashlib.sha256()
        directories = {}
        files = {}
        paths: list[Path] = []
        # rglob("*") traverses .venv before rejecting its contents. os.walk
        # allows top-down pruning of dependencies and VCS metadata instead.
        for directory, subdirs, filenames in os.walk(root, topdown=True):
            subdirs[:] = sorted(
                name for name in subdirs
                if name not in _EXCLUDED_DIRS
                and not (Path(directory) / name).is_symlink()
            )
            directory_path = Path(directory)
            directory_stat = directory_path.stat()
            directories[directory_path] = (
                directory_stat.st_mtime_ns,
                directory_stat.st_ctime_ns,
            )
            for filename in filenames:
                candidate = directory_path / filename
                if (
                    not candidate.is_symlink()
                    and (candidate.suffix in _SOURCE_EXTENSIONS or filename in _SOURCE_FILES)
                ):
                    paths.append(candidate)

        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
            st = path.stat()
            files[path] = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            blob = path.read_bytes()
            digest.update(len(blob).to_bytes(8, "big"))
            digest.update(blob)
        value = digest.hexdigest()
        _SOURCE_DIGEST_CACHE[root] = (directories, files, value)
        return value


def _local_source_path(url: str) -> Path | None:
    """Use a local direct_url only for hashing; never persist the URL/path."""
    parsed = urlsplit(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(url2pathname(unquote(parsed.path)))


def _distribution_record(dist: metadata.Distribution) -> dict[str, Any]:
    name = str(dist.metadata.get("Name") or "").strip()
    version = str(dist.version or "").strip()
    if not name or not version:
        raise EnvironmentRegistryError(
            "Cannot fingerprint an installed distribution with missing name/version."
        )
    record: dict[str, Any] = {
        "name": _NAME_SEPARATOR.sub("-", name).lower(),
        "version": version,
    }
    direct_text = dist.read_text("direct_url.json")
    if direct_text is None:
        return record
    try:
        direct = json.loads(direct_text)
        if not isinstance(direct, dict):
            raise ValueError("expected object")
    except (ValueError, TypeError) as exc:
        raise EnvironmentRegistryError(
            f"Invalid direct_url.json for distribution {name!r}."
        ) from exc

    # Never copy direct_url.url: it can contain tokens, credentials, or host paths.
    record["direct_install"] = True
    dir_info = direct.get("dir_info")
    if isinstance(dir_info, dict):
        record["editable"] = bool(dir_info.get("editable", False))
    vcs_info = direct.get("vcs_info")
    if isinstance(vcs_info, dict):
        vcs_type = str(vcs_info.get("vcs", ""))
        if vcs_type in {"git", "hg", "svn", "bzr"}:
            record["vcs"] = vcs_type
        commit = str(vcs_info.get("commit_id", ""))
        if _SAFE_COMMIT.fullmatch(commit):
            record["vcs_commit"] = commit.lower()
    archive_info = direct.get("archive_info")
    if isinstance(archive_info, dict):
        hashes = archive_info.get("hashes")
        if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
            if re.fullmatch(r"[a-fA-F0-9]{64}", hashes["sha256"]):
                record["archive_sha256"] = hashes["sha256"].lower()

    local_path = _local_source_path(str(direct.get("url", "")))
    if local_path is not None:
        record["local_install"] = True
        if local_path.is_dir():
            try:
                record["local_source_sha256"] = _source_digest(local_path)
            except OSError as exc:
                raise EnvironmentRegistryError(
                    f"Cannot fingerprint local source for {name!r}."
                ) from exc
        else:
            # Historical installation metadata can point at a removed source.
            record["local_source_available"] = False
    return record


def observe_environment() -> dict[str, Any]:
    """Observe the current interpreter without reading unrelated environment vars.

    Captures the whole installed distribution inventory so that changing any
    installed package version produces a new historical identity. Version 1
    fingerprints are portable software/platform observations, not lockfiles.
    """
    try:
        distributions = [_distribution_record(dist) for dist in metadata.distributions()]
        distributions.sort(key=lambda item: _canonical_json(item))
        package_root = Path(__file__).resolve().parents[1]
        teal_source_sha = _source_digest(package_root)
        try:
            teal_version = metadata.version("text-analysis-lab")
        except metadata.PackageNotFoundError:
            teal_version = "0+unknown"
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "python": {
                "implementation": sys.implementation.name,
                "version": platform.python_version(),
                "implementation_version": list(sys.implementation.version[:3]),
                "abi": str(sysconfig.get_config_var("SOABI") or ""),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "architecture": platform.architecture()[0],
            },
            "teal": {
                "version": teal_version,
                "source_sha256": teal_source_sha,
            },
            "distributions": distributions,
        }
    except EnvironmentRegistryError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise EnvironmentRegistryError(
            "Unable to observe the opening Python environment."
        ) from exc


def fingerprint_observation(observation: dict[str, Any]) -> str:
    """SHA-256 of canonical, versioned environment observation JSON."""
    if (
        not isinstance(observation, dict)
        or observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or not all(key in observation for key in ("python", "platform", "teal", "distributions"))
    ):
        raise EnvironmentRegistryError("Unsupported or incomplete environment observation.")
    try:
        return hashlib.sha256(_canonical_json(observation).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise EnvironmentRegistryError("Environment observation is not serializable.") from exc


class EnvironmentRegistry:
    """Environment history and active default for one open Project instance.

    Public API:
      list()              -> registered historical observations, oldest first
      get(environment_id) -> a registered observation or EnvironmentNotFoundError
      default             -> the current instance's selected observation
      set_default(id)     -> select a registered observation for this instance only

    An entry with readiness='unverified' is *not* known to be executable.
    Switching the default never switches the running Python interpreter.
    """

    def __init__(self, catalog: ProjectCatalog) -> None:
        self._catalog = catalog
        observed = observe_environment()
        fingerprint = fingerprint_observation(observed)
        self._opening_id = catalog.register_environment(
            fingerprint=fingerprint,
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observation_json=_canonical_json(observed),
        )
        self._default_id = self._opening_id

    def _public_row(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            observation = json.loads(row["observation_json"])
            if fingerprint_observation(observation) != row["fingerprint"]:
                raise EnvironmentRegistryError("Fingerprint mismatch in environment registry.")
            if row["environment_id"] != "env_" + row["fingerprint"]:
                raise EnvironmentRegistryError("Environment identity mismatch in registry.")
            if int(row["schema_version"]) != observation["schema_version"]:
                raise EnvironmentRegistryError("Environment schema mismatch in registry.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EnvironmentRegistryError("Corrupt environment registry record.") from exc
        return {
            "environment_id": row["environment_id"],
            "fingerprint": row["fingerprint"],
            "registered_at": row["registered_at"],
            "observation": observation,
            "readiness": (
                "current_interpreter"
                if row["environment_id"] == self._opening_id
                else "unverified"
            ),
        }

    def list(self) -> list[dict[str, Any]]:
        """Return all historical entries in registration order."""
        return [self._public_row(row) for row in self._catalog.list_environments()]

    def get(self, environment_id: str) -> dict[str, Any]:
        """Look up a content-addressed ID; raise for unknown IDs."""
        if not isinstance(environment_id, str):
            raise EnvironmentNotFoundError("Environment ID must be a string.")
        row = self._catalog.get_environment(environment_id)
        if row is None:
            raise EnvironmentNotFoundError(
                f"Environment {environment_id!r} is not registered."
            )
        return self._public_row(row)

    @property
    def default(self) -> dict[str, Any]:
        """The active default for this Project instance, not a durable setting."""
        return self.get(self._default_id)

    def set_default(self, environment_id: str) -> dict[str, Any]:
        """Select any registered historical entry; no execution is implied."""
        environment = self.get(environment_id)
        self._default_id = environment["environment_id"]
        return environment
