"""Portable observations and session-local defaults for Project Python environments.

Registry membership is historical; it does not imply a runnable interpreter.
No machine-local executable path or installer state is stored in project data.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from text_analysis_lab.core.errors import TeALError

if TYPE_CHECKING:
    from text_analysis_lab.core.catalog import ProjectCatalog

OBSERVATION_SCHEMA_VERSION = 2
_NAME_SEPARATOR = re.compile(r"[-_.]+")
_SAFE_COMMIT = re.compile(r"^[a-fA-F0-9]{7,64}$")


class EnvironmentRegistryError(TeALError):
    """An interpreter observation or persisted registry record is invalid."""


class EnvironmentNotFoundError(EnvironmentRegistryError, LookupError):
    """A selected environment is not registered with this project."""


_ENVIRONMENT_ID = re.compile(r"^env_[a-f0-9]{64}$")
_RESERVED_ENVIRONMENT_NAME = re.compile(r"^env_", re.IGNORECASE)


def _environment_name(value: str) -> tuple[str, str]:
    """Validate a display alias and return its case-folded uniqueness key."""
    if not isinstance(value, str):
        raise EnvironmentRegistryError("Environment name must be a string.")
    if not value or value != value.strip() or len(value) > 100:
        raise EnvironmentRegistryError(
            "Environment name must be 1-100 characters with no outer whitespace."
        )
    if _RESERVED_ENVIRONMENT_NAME.match(value):
        raise EnvironmentRegistryError(
            "Environment names beginning with 'env_' are reserved for immutable IDs."
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise EnvironmentRegistryError("Environment name contains a control character.")
    return value, value.casefold()


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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

    # A file:// origin indicates a local installation; do not store or inspect
    # the host-specific path, which is not portable project identity.
    origin = direct.get("url")
    if isinstance(origin, str) and urlsplit(origin).scheme == "file":
        record["local_install"] = True
    return record


# Installation metadata is much cheaper to stat than to reparse on every
# Project.open(). This cache never contains package source or model contents.
_DISTRIBUTION_LOCK = RLock()
_DISTRIBUTION_CACHE: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None


def _distribution_metadata_signature() -> tuple[Any, ...]:
    """Notice installed package changes, metadata edits and sys.path drift.

    Resolve empty sys.path entries to the *current* working directory. Watch
    the metadata files themselves: updating an existing METADATA/direct_url.json
    need not change its containing directory's timestamp.
    """
    roots: list[tuple[Any, ...]] = []
    for entry in sys.path:
        base = Path(entry or os.getcwd()).absolute()
        try:
            st = base.stat()
        except FileNotFoundError:
            roots.append((str(base), "missing"))
            continue

        if not base.is_dir():
            roots.append((str(base), st.st_mtime_ns, st.st_ctime_ns, st.st_size))
            continue

        distributions = []
        with os.scandir(base) as scan:
            for child in scan:
                if not child.name.endswith((".dist-info", ".egg-info")):
                    continue
                directory = Path(child.path)
                info = directory.stat()
                files = []
                if child.is_dir():
                    for name in ("METADATA", "PKG-INFO", "direct_url.json"):
                        try:
                            part = (directory / name).stat()
                            files.append((name, part.st_mtime_ns, part.st_ctime_ns, part.st_size))
                        except FileNotFoundError:
                            files.append((name, None))
                distributions.append(
                    (child.name, info.st_mtime_ns, info.st_ctime_ns,
                     info.st_size, tuple(files))
                )
        roots.append(
            (str(base), st.st_mtime_ns, st.st_ctime_ns,
             tuple(sorted(distributions)))
        )
    return tuple(roots)


def _installed_distributions() -> list[dict[str, Any]]:
    """Reuse parsed inventory only while installation metadata is unchanged."""
    global _DISTRIBUTION_CACHE
    with _DISTRIBUTION_LOCK:
        signature = _distribution_metadata_signature()
        if _DISTRIBUTION_CACHE is None or _DISTRIBUTION_CACHE[0] != signature:
            records = sorted(
                (_distribution_record(dist) for dist in metadata.distributions()),
                key=_canonical_json,
            )
            _DISTRIBUTION_CACHE = (signature, records)
        return [record.copy() for record in _DISTRIBUTION_CACHE[1]]


def observe_environment() -> dict[str, Any]:
    """Observe the current interpreter without reading unrelated environment vars.

    Captures the whole installed distribution inventory so that changing any
    installed package version produces a new historical identity. Version 1
    fingerprints are software observations, not a portable lockfile.
    """
    try:
        distributions = _installed_distributions()
        try:
            teal_version = metadata.version("text-analysis-lab")
        except metadata.PackageNotFoundError:
            teal_version = "0+unknown"
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "kind": "observed_unlocked",
            "python": {
                "implementation": sys.implementation.name,
                "version": platform.python_version(),
            },
            "teal": {
                "version": teal_version,
            },
            "distributions": distributions,
        }
    except EnvironmentRegistryError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise EnvironmentRegistryError(
            "Unable to observe the opening Python environment."
        ) from exc


def _normalize_lock_manifest(value: Any, *, field: str | None = None) -> Any:
    """Sort only dependency-set arrays; preserve potentially meaningful index order.

    A lock specification may have ordered indexes/priorities. Sorting every
    array would incorrectly identify distinct resolutions as equivalent.
    """
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise EnvironmentRegistryError("Invalid portable lock field name.")
        return {
            key: _normalize_lock_manifest(item, field=key)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        items = [_normalize_lock_manifest(item) for item in value]
        if field in {"packages", "package", "wheels", "requires-dist",
                     "distributions", "realizations", "dependencies", "extras"}:
            return sorted(
                items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        return items
    if value is None or type(value) in {bool, str, int, float}:
        return value
    raise EnvironmentRegistryError("Invalid portable lock manifest value.")


_PORTABLE_PACKAGE_FIELDS = frozenset({
    "name", "version", "direct_install", "editable", "local_install",
    "vcs", "vcs_commit", "archive_sha256",
})
_PORTABLE_MANIFEST_FIELDS = frozenset({
    "requirements", "realizations", "packages", "index_priority", "extras",
})
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_PINNED_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
_SAFE_INDEX_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _portable_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _PACKAGE_NAME.fullmatch(value):
        raise EnvironmentRegistryError(f"Invalid portable {label}; local paths are not allowed.")
    return _NAME_SEPARATOR.sub("-", value).lower()


def _portable_version(value: Any) -> str:
    if not isinstance(value, str) or not _PINNED_VERSION.fullmatch(value):
        raise EnvironmentRegistryError("Portable versions must be pinned package versions, not URLs or paths.")
    try:
        Version(value)
    except InvalidVersion as exc:
        raise EnvironmentRegistryError("Invalid pinned package version.") from exc
    return value


def _exact_python_version(value: Any) -> str:
    """Require Python's explicit major.minor.micro, not a minor-only request."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?", value
    ):
        raise EnvironmentRegistryError("Portable Python version must include major.minor.micro.")
    return value


def _requirement_satisfied(pin: str, installed_version: str) -> bool:
    """PEP 440 == pins accept local build variants when pin has no local tag."""
    return SpecifierSet(f"=={pin}").contains(installed_version)



def _normalize_distribution_records(
    records: list[dict[str, Any]], *, portable: bool = False,
) -> list[dict[str, Any]]:
    """Canonicalize observed inventory or strictly validated locked realizations."""
    if not isinstance(records, list):
        raise EnvironmentRegistryError("Distributions must be a list.")
    distributions = []
    for record in records:
        if not isinstance(record, dict):
            raise EnvironmentRegistryError("Invalid package record.")
        if set(record) - _PORTABLE_PACKAGE_FIELDS:
            raise EnvironmentRegistryError("Unexpected distribution metadata field.")
        name, version = record.get("name"), record.get("version")
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(version, str) or not version):
            raise EnvironmentRegistryError("A package needs a name and version.")
        normalized = dict(record)
        normalized["name"] = (
            _portable_name(name, label="package name") if portable
            else _NAME_SEPARATOR.sub("-", name.strip()).lower()
        )
        if portable:
            normalized["version"] = _portable_version(version)
        for flag in ("direct_install", "editable", "local_install"):
            if flag in record and type(record[flag]) is not bool:
                raise EnvironmentRegistryError(f"Invalid {flag} package flag.")
        if portable and (record.get("editable") or record.get("local_install")):
            raise EnvironmentRegistryError("Portable locks cannot reference editable/local installations.")
        if "vcs" in record and record["vcs"] not in {"git", "hg", "svn", "bzr"}:
            raise EnvironmentRegistryError("Invalid package VCS.")
        if "vcs_commit" in record:
            commit = record["vcs_commit"]
            if not isinstance(commit, str) or not _SAFE_COMMIT.fullmatch(commit):
                raise EnvironmentRegistryError("Invalid package VCS commit.")
            normalized["vcs_commit"] = commit.lower()
        if "archive_sha256" in record:
            digest = record["archive_sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                raise EnvironmentRegistryError("Invalid package archive digest.")
            normalized["archive_sha256"] = digest.lower()
        distributions.append(normalized)
    return sorted(distributions, key=_canonical_json)


def _portable_index_reference(value: Any) -> str:
    """Permit public HTTPS indexes or symbolic labels; never persist credentials."""
    if not isinstance(value, str) or not value:
        raise EnvironmentRegistryError("Invalid portable index reference.")
    if _SAFE_INDEX_LABEL.fullmatch(value):
        if value.lower() in {"localhost", "home", "users"} or value.lower().endswith(".local"):
            raise EnvironmentRegistryError("A local host is not a portable index.")
        return value
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError as exc:
        raise EnvironmentRegistryError("Malformed portable index URL.") from exc
    if (parsed.scheme != "https" or not host or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or host.lower() == "localhost"
            or host.lower().endswith((".local", ".localhost", ".internal", ".lan"))):
        raise EnvironmentRegistryError("Portable index must be a public HTTPS URL without credentials.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass  # Public DNS names remain permitted; no network resolution in R0.
    else:
        if not address.is_global:
            raise EnvironmentRegistryError("Private/local IP addresses are not portable indexes.")
    if "\\" in value or any(part in {"..", "~"} for part in parsed.path.split("/")):
        raise EnvironmentRegistryError("Portable index URL contains a local path.")
    return value


def _normalize_portable_manifest(manifest: Any, *, python: str | None = None,
                                 teal_commit: str | None = None,
                                 teal_version: str | None = None,
                                 extras: list[str] | None = None) -> dict[str, Any]:
    """Validate legacy explicit realizations or a resolver-backed uv project lock."""
    if isinstance(manifest, dict) and "uv_lock" in manifest:
        from text_analysis_lab.core.portable_locks import validate_manifest
        if not (python and teal_commit and teal_version and extras is not None):
            raise EnvironmentRegistryError("Incomplete uv lock validation context.")
        return validate_manifest(manifest, python=python,
                                 commit=teal_commit, extras=extras,
                                 teal_version=teal_version)
    if (not isinstance(manifest, dict) or not manifest
            or set(manifest) - _PORTABLE_MANIFEST_FIELDS):
        raise EnvironmentRegistryError("Unsupported portable lock manifest field.")
    if not set(manifest) & {"requirements", "realizations", "packages"}:
        raise EnvironmentRegistryError("Portable lock needs pinned requirements or realizations.")
    result: dict[str, Any] = {}
    if "requirements" in manifest:
        reqs = manifest["requirements"]
        if not isinstance(reqs, dict):
            raise EnvironmentRegistryError("Portable requirements must be a name/version mapping.")
        normalized_reqs: dict[str, str] = {}
        for name, version in reqs.items():
            canonical = _portable_name(name, label="requirement name")
            if canonical in normalized_reqs:
                raise EnvironmentRegistryError("Duplicate normalized portable requirement.")
            normalized_reqs[canonical] = _portable_version(version)
        result["requirements"] = dict(sorted(normalized_reqs.items()))
    if "packages" in manifest:
        result["packages"] = _normalize_distribution_records(manifest["packages"], portable=True)
    if "realizations" in manifest:
        realizations = manifest["realizations"]
        if not isinstance(realizations, list):
            raise EnvironmentRegistryError("Portable realizations must be a list.")
        result["realizations"] = []
        for realization in realizations:
            if not isinstance(realization, dict) or set(realization) != {"distributions"}:
                raise EnvironmentRegistryError("Unsupported portable realization field.")
            result["realizations"].append({
                "distributions": _normalize_distribution_records(
                    realization["distributions"], portable=True,
                ),
            })
        result["realizations"].sort(key=_canonical_json)
    if "requirements" in result and "realizations" in result:
        pins = result["requirements"]
        for realization in result["realizations"]:
            inventory = {}
            for distribution in realization["distributions"]:
                name = distribution["name"]
                if name in inventory:
                    raise EnvironmentRegistryError(
                        f"Duplicate package {name!r} in portable realization."
                    )
                inventory[name] = distribution["version"]
            for name, pin in pins.items():
                if name not in inventory or not _requirement_satisfied(pin, inventory[name]):
                    raise EnvironmentRegistryError(
                        f"Portable realization contradicts exact requirement {name}=={pin}."
                    )
    if "index_priority" in manifest:
        entries = manifest["index_priority"]
        if not isinstance(entries, list):
            raise EnvironmentRegistryError("Portable index priority must be a list.")
        # Priority is meaningful and must not be sorted.
        result["index_priority"] = [_portable_index_reference(x) for x in entries]
    if "extras" in manifest:
        extras = manifest["extras"]
        if not isinstance(extras, list):
            raise EnvironmentRegistryError("Portable extras must be a list.")
        result["extras"] = sorted({_portable_name(x, label="extra") for x in extras})
    return result


def normalize_environment(observation: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a software observation or a platform-independent lock."""
    version = observation.get("schema_version") if isinstance(observation, dict) else None
    if type(version) is not int or version < 2 or version > OBSERVATION_SCHEMA_VERSION:
        raise EnvironmentRegistryError("Unsupported software observation schema.")
    kind = observation.get("kind")
    if kind not in {"observed_unlocked", "portable_locked"}:
        raise EnvironmentRegistryError("Unknown software environment kind.")
    expected = {"schema_version", "kind", "python", "teal"}
    if kind == "observed_unlocked":
        expected.add("distributions")
    else:
        expected.update({"extras", "lock"})
    if set(observation) != expected:
        raise EnvironmentRegistryError("Unexpected software environment fields.")
    python_info, teal = observation["python"], observation["teal"]
    if (not isinstance(python_info, dict)
        or set(python_info) != {"implementation", "version"}
        or any(not isinstance(v, str) or not v for v in python_info.values())
        or not isinstance(teal, dict)
        or set(teal) != ({"version"} if kind == "observed_unlocked" else {"version", "source"})
        or not isinstance(teal.get("version"), str)
        or not teal["version"]):
        raise EnvironmentRegistryError("Invalid Python or TeAL identity.")
    result = {
        "schema_version": version,
        "kind": kind,
        "python": dict(python_info),
        "teal": dict(teal),
    }
    if kind == "observed_unlocked":
        result["distributions"] = _normalize_distribution_records(observation["distributions"])
        return result

    # A portable lock hashes only its canonical software specification, never
    # a redundant top-level inventory from one particular platform.
    _exact_python_version(python_info["version"])
    _portable_name(python_info["implementation"], label="Python implementation")
    _portable_version(teal["version"])
    source = teal["source"]
    if (not isinstance(source, dict)
            or set(source) - {"vcs", "commit", "sha256"}
            or not any(
                isinstance(source.get(field), str)
                and re.fullmatch(r"[a-fA-F0-9]{40,64}", source[field])
                for field in ("commit", "sha256")
            )):
        raise EnvironmentRegistryError("Portable TeAL requires an immutable commit/hash.")
    normalized_source: dict[str, str] = {}
    if "vcs" in source:
        if source["vcs"] != "git":
            raise EnvironmentRegistryError("Portable TeAL VCS must be git.")
        normalized_source["vcs"] = "git"
    for field in ("commit", "sha256"):
        if field in source:
            value = source[field]
            expected_length = 64 if field == "sha256" else 40
            if (not isinstance(value, str) or len(value) != expected_length
                    or not re.fullmatch(r"[a-fA-F0-9]+", value)):
                raise EnvironmentRegistryError("Invalid immutable TeAL source hash.")
            normalized_source[field] = value.lower()
    result["teal"] = {"version": teal["version"], "source": normalized_source}
    extras, lock = observation.get("extras"), observation.get("lock")
    if (not isinstance(extras, list)
            or not isinstance(lock, dict)
            or set(lock) != {"kind", "manifest", "sha256"}):
        raise EnvironmentRegistryError("Portable specification requires extras and lock metadata.")
    result["extras"] = sorted({_portable_name(x, label="extra") for x in extras})
    if not isinstance(lock["kind"], str) or not _SAFE_INDEX_LABEL.fullmatch(lock["kind"]):
        raise EnvironmentRegistryError("Invalid portable lock kind.")
    manifest = _normalize_portable_manifest(
        lock["manifest"], python=python_info["version"],
        teal_commit=normalized_source.get("commit"),
        teal_version=teal["version"], extras=result["extras"],
    )
    expected_digest = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    digest = lock["sha256"]
    if not isinstance(digest, str) or digest.lower() != expected_digest:
        raise EnvironmentRegistryError("Portable lock content hash does not match manifest.")
    result["lock"] = {
        "kind": lock["kind"],
        "manifest": manifest,
        "sha256": expected_digest,
    }
    return result


def fingerprint_observation(
    observation: dict[str, Any], *, require_current_schema: bool = True
) -> str:
    """Hash canonical, versioned software identity, never an OS/platform."""
    try:
        normalized = normalize_environment(observation)
        if require_current_schema and normalized["schema_version"] != OBSERVATION_SCHEMA_VERSION:
            raise EnvironmentRegistryError("Unsupported current software schema.")
        return hashlib.sha256(
            _canonical_json(normalized).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise EnvironmentRegistryError("Software observation is not serializable.") from exc


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
        observed = normalize_environment(observe_environment())
        fingerprint = fingerprint_observation(observed)
        self._observed_id = catalog.register_environment(
            fingerprint=fingerprint,
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observation_json=_canonical_json(observed),
        )
        self._opening_id = self._observed_id
        matching = [
            candidate for candidate in catalog.portable_realizations_for(self._observed_id)
            if self._matches_portable(candidate, observed)
        ]
        # Do not select one of several ambiguous matched portable specifications.
        if len(matching) == 1:
            self._opening_id = matching[0]
        self._default_id = self._opening_id
        # Machine-local, per-Project references; never written to SQLite.
        self._interpreter_bindings: dict[str, Path] = {}

    def plan(
        self, *, python: str | None = None, teal: str | None = None,
        teal_extras: list[str] | tuple[str, ...] | None = None,
        gpu: str | bool = "auto",
    ) -> dict[str, Any]:
        """Preview a managed interpreter install without modifying anything."""
        from text_analysis_lab.core.managed_environments import plan
        return plan(
            python=python, teal=teal, teal_extras=teal_extras, gpu=gpu,
        )

    def create(
        self, *, name: str | None = None,
        python: str | None = None, teal: str | None = None,
        teal_extras: list[str] | tuple[str, ...] | None = None,
        gpu: str | bool = "auto",
    ) -> dict[str, Any]:
        """Install/reuse a managed interpreter; optionally assign a project alias."""
        if name is not None:
            name, key = _environment_name(name)
            if self._catalog.get_environment_by_name(key) is not None:
                raise EnvironmentRegistryError(f"Environment name {name!r} is already in use.")
        from text_analysis_lab.core.managed_environments import create
        result = create(
            self, python=python, teal=teal, teal_extras=teal_extras, gpu=gpu,
        )
        if name is not None:
            self._assign_name(result["environment_id"], name)
            result["name"] = name
        return result

    def plan_from_file(
        self, source: str | Path, *, project_file: str | Path | None = None,
        python: str | None = None,
    ) -> dict[str, Any]:
        """Read-only validation and preview of a dependency-file installation."""
        from text_analysis_lab.core.external_environments import plan_from_file
        return plan_from_file(source, project_file=project_file, python=python)

    def create_from_file(
        self, source: str | Path, *, project_file: str | Path | None = None,
        name: str | None = None, python: str | None = None,
    ) -> dict[str, Any]:
        """Provision an external dependency specification without adding TeAL."""
        if name is not None:
            name, key = _environment_name(name)
            if self._catalog.get_environment_by_name(key) is not None:
                raise EnvironmentRegistryError(f"Environment name {name!r} is already in use.")
        from text_analysis_lab.core.external_environments import create_from_file
        result = create_from_file(self, source, project_file=project_file, python=python)
        if name is not None:
            self._assign_name(result["environment_id"], name)
            result["name"] = name
        return result

    def _assign_name(self, environment_id: str, name: str) -> None:
        normalized, key = _environment_name(name)
        try:
            self._catalog.register_environment_name(environment_id, normalized, key)
        except ValueError as exc:
            raise EnvironmentRegistryError(str(exc)) from exc

    def rename(self, old_name: str, new_name: str) -> dict[str, Any]:
        """Rename a project alias without changing its environment or provenance."""
        _, old_key = _environment_name(old_name)
        normalized, new_key = _environment_name(new_name)
        try:
            environment_id = self._catalog.rename_environment_name(
                old_key, normalized, new_key,
            )
        except KeyError as exc:
            raise EnvironmentNotFoundError(
                f"Environment name {old_name!r} is not registered."
            ) from exc
        except ValueError as exc:
            raise EnvironmentRegistryError(str(exc)) from exc
        return self.get(environment_id)

    def create_specification(
        self, *, python: str | None = None, teal: str | None = None,
        teal_extras: list[str] | tuple[str, ...] | None = None,
        gpu: str | bool = False,
    ) -> dict[str, Any]:
        """Resolve/register a content-addressed cross-platform uv software lock."""
        from text_analysis_lab.core.portable_locks import create_specification
        return create_specification(self, python=python, teal=teal,
                                    teal_extras=teal_extras, gpu=gpu)

    def export_specification(self, environment_id: str, destination: str | Path) -> Path:
        """Export a registered frozen specification without local cache paths."""
        from text_analysis_lab.core.portable_locks import export_specification
        return export_specification(self, environment_id, destination)

    def import_specification(self, source: str | Path) -> str:
        """Validate and register a frozen specification without installing it."""
        from text_analysis_lab.core.portable_locks import import_specification
        return import_specification(self, source)

    def create_from_specification(self, environment_id: str) -> dict[str, Any]:
        """Explicitly reconstruct and bind a verified realization of a portable lock."""
        from text_analysis_lab.core.portable_locks import create_from_specification
        return create_from_specification(self, environment_id)

    @property
    def opening_id(self) -> str:
        """Registered identity of the interpreter running this Project instance."""
        return self._opening_id

    def same_verified_realization(
        self, first: str, second: str, *, observed_id: str,
    ) -> bool:
        """Recognize ONLY a verified observed/portable pair for one installation.

        Equality never means arbitrary portable specifications or two observed
        inventories are interchangeable. The R0 association must exist and its
        locked realization must still match the registered observed inventory.
        Passing an observed_id makes this usable by a parent validating the
        independently probed *child*, not just its own opening interpreter.
        """
        if first == second:
            return True
        if observed_id not in {first, second}:
            return False
        portable_id = second if first == observed_id else first
        observed = self.get(observed_id)["observation"]
        if observed["kind"] != "observed_unlocked":
            return False
        if portable_id not in self._catalog.portable_realizations_for(observed_id):
            return False
        return self._matches_portable(portable_id, observed)

    def equivalent_executor_identities(self, expected: str, actual: str) -> bool:
        """Check a same-installation observed/portable pair from the R0 registry."""
        if expected == actual:
            return True
        for candidate in (expected, actual):
            if self.get(candidate)["kind"] == "observed_unlocked":
                return self.same_verified_realization(
                    expected, actual, observed_id=candidate,
                )
        return False

    def is_current_interpreter(self, environment_id: str) -> bool:
        """Whether a selector is the verified running interpreter.

        The current portable logical ID and its original observed inventory
        can both refer to this interpreter, but a different portable ID cannot
        silently become the current choice. Re-observe before accepting that
        alias so installed-package drift cannot exploit a historical mapping.
        """
        environment_id = self.resolve(environment_id)
        if environment_id == self._opening_id:
            return True
        if (environment_id != self._observed_id
                or self._opening_id == self._observed_id):
            return False
        actual = normalize_environment(observe_environment())
        if "env_" + fingerprint_observation(actual) != self._observed_id:
            return False
        return self.same_verified_realization(
            environment_id, self._opening_id, observed_id=self._observed_id,
        )

    def require_local_execution(self, environment_id: str | None = None) -> str:
        """Resolve the selector; fail before scientific work when it is not local.

        Setting a default or supplying an override never starts another
        interpreter. That execution route is not yet implemented.
        """
        selected = (
            self._default_id if environment_id is None
            else self.get(environment_id)["environment_id"]
        )
        if not self.is_current_interpreter(selected):
            raise EnvironmentRegistryError(
                f"Selected environment {selected!r} is registered but is not "
                "the current Python interpreter. Cross-interpreter execution "
                "is not yet available; reopen the project under that environment "
                "or select the current interpreter."
            )
        return selected

    def resolve(self, environment_id: str | None = None) -> str:
        """Resolve an explicit registered ID or this instance's active default."""
        if environment_id is None:
            return self._default_id
        return self.get(environment_id)["environment_id"]

    def bind_interpreter(
        self, environment_id: str, executable: str | Path
    ) -> None:
        """Bind a registered observation to a local Python executable for this session.

        Binding is not proof of readiness: each remote execution probes the
        executable's actual observation before any scientific work.
        """
        environment_id = self.resolve(environment_id)
        path = Path(executable).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EnvironmentRegistryError(
                f"Interpreter {str(path)!r} does not exist or is not executable."
            )
        self._interpreter_bindings[environment_id] = path

    def bound_interpreter(self, environment_id: str) -> Path:
        """Return this session's executable, or reject an unbound/missing one."""
        environment_id = self.resolve(environment_id)
        path = self._interpreter_bindings.get(environment_id)
        if path is None:
            # A Project records software identities, never machine-local paths.
            # On reopen, discover a previously published uv interpreter by its
            # *registered observed identity* and fully reverify the installation
            # before installing a new session-only binding. Unmanaged and
            # unverified environments still require an explicit binding.
            from text_analysis_lab.core.managed_environments import find_cached_interpreter

            row = self.get(environment_id)
            if (row["kind"] == "portable_locked"
                    and row["observation"]["lock"]["kind"] == "uv-project-v1"):
                from text_analysis_lab.core.portable_locks import find_cached_portable_interpreter
                path = find_cached_portable_interpreter(self, environment_id)
            else:
                path = find_cached_interpreter(environment_id)
            if path is not None:
                self.bind_interpreter(environment_id, path)
        if path is None:
            raise EnvironmentRegistryError(
                f"Environment {environment_id!r} has no bound Python interpreter. "
                "Call project.environments.bind_interpreter(...) or create a "
                "verified managed installation."
            )
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EnvironmentRegistryError(
                f"Bound interpreter {str(path)!r} is missing or not executable."
            )
        return path

    def register_portable_specification(self, specification: dict[str, Any]) -> str:
        """Record a supplied portable lock; do not assert it can run locally."""
        normalized = normalize_environment(specification)
        if normalized["kind"] != "portable_locked":
            raise EnvironmentRegistryError("Expected a portable locked specification.")
        return self._catalog.register_environment(
            fingerprint=fingerprint_observation(normalized),
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observation_json=_canonical_json(normalized),
        )

    def _matches_portable(self, environment_id: str, observed: dict[str, Any]) -> bool:
        """Verify an installed, local realization against one declared lock variant.

        Portable spec identity alone does not establish executable compatibility;
        all observed distribution records (including local wheel/build variants)
        must match a recorded realization, and TeAL source must be immutable.
        """
        spec = self.get(environment_id)["observation"]
        if (spec["kind"] != "portable_locked"
                or spec["python"] != observed["python"]
                or spec["teal"]["version"] != observed["teal"]["version"]):
            return False
        source = spec["teal"]["source"]
        teal_package = next((
            item for item in observed["distributions"]
            if item["name"] == "text-analysis-lab"
        ), None)
        if teal_package is None:
            return False
        if source.get("commit") and teal_package.get("vcs_commit") != source["commit"].lower():
            return False
        if source.get("sha256") and teal_package.get("archive_sha256") != source["sha256"].lower():
            return False
        if spec["lock"]["kind"] == "uv-project-v1":
            from text_analysis_lab.core.portable_locks import matches_installed
            return matches_installed(spec, observed)
        realizations = spec["lock"]["manifest"].get("realizations")
        if not isinstance(realizations, list):
            return False
        for realization in realizations:
            if not isinstance(realization, dict) or set(realization) != {"distributions"}:
                continue
            try:
                candidate = _normalize_distribution_records(
                    realization["distributions"], portable=True,
                )
            except EnvironmentRegistryError:
                continue
            if candidate == observed["distributions"]:
                return True
        return False

    def verify_current_portable(self, environment_id: str) -> dict[str, Any]:
        """Match installed *metadata* to a portable realization and adopt its ID.

        This checks Python, TeAL source metadata and the package inventory only.
        It does not verify installed wheel bytes against resolver artifact hashes;
        R2 must provide that stronger verification before claiming exact rebuild.
        The physical platform is separately observed, not a required lock match.
        """
        observed = normalize_environment(observe_environment())
        if fingerprint_observation(observed) != self._catalog.get_environment(
            self._observed_id
        )["fingerprint"]:
            raise EnvironmentRegistryError(
                "Current Python installation changed after Project.open; reopen before verifying."
            )
        if not self._matches_portable(environment_id, observed):
            raise EnvironmentRegistryError(
                "Installed software does not match any verified portable lock realization."
            )
        self._catalog.record_environment_realization(self._observed_id, environment_id)
        self._opening_id = environment_id
        self._default_id = environment_id
        current_platform = getattr(self._catalog, "_execution_platform_id", None)
        if current_platform is not None:
            self._catalog.set_execution_context(environment_id, current_platform)
        return self.get(environment_id)

    def _public_row(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            observation = normalize_environment(json.loads(row["observation_json"]))
            if fingerprint_observation(observation, require_current_schema=False) != row["fingerprint"]:
                raise EnvironmentRegistryError("Fingerprint mismatch in environment registry.")
            if row["environment_id"] != "env_" + row["fingerprint"]:
                raise EnvironmentRegistryError("Environment identity mismatch in registry.")
            if int(row["schema_version"]) != observation["schema_version"]:
                raise EnvironmentRegistryError("Environment schema mismatch in registry.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EnvironmentRegistryError("Corrupt environment registry record.") from exc
        names = self._catalog.environment_names_for(row["environment_id"])
        return {
            "environment_id": row["environment_id"],
            "name": names[0] if names else None,
            "names": names,
            "fingerprint": row["fingerprint"],
            "registered_at": row["registered_at"],
            "observation": observation,
            "kind": observation["kind"],
            "verification_level": (
                "installed_metadata_match"
                if observation["kind"] == "portable_locked"
                and row["environment_id"] == self._opening_id
                else "observed_inventory"
                if observation["kind"] == "observed_unlocked"
                and row["environment_id"] == self._observed_id
                else "not_verified"
            ),
            "readiness": (
                "current_interpreter"
                if row["environment_id"] == self._opening_id
                else "current_observed_interpreter"
                if row["environment_id"] == self._observed_id
                else "unverified"
            ),
        }

    def list(self) -> list[dict[str, Any]]:
        """Return all historical entries in registration order."""
        return [self._public_row(row) for row in self._catalog.list_environments()]

    def get(self, environment_id: str) -> dict[str, Any]:
        """Resolve an immutable ID or a case-insensitive project-local name."""
        if not isinstance(environment_id, str):
            raise EnvironmentNotFoundError("Environment selector must be a string.")
        if _ENVIRONMENT_ID.fullmatch(environment_id):
            row = self._catalog.get_environment(environment_id)
        elif _RESERVED_ENVIRONMENT_NAME.match(environment_id):
            row = None
        else:
            row = self._catalog.get_environment_by_name(environment_id.casefold())
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
