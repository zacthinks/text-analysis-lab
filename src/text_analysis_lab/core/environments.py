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
from importlib import metadata
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any

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


def _normalize_lock_manifest(value: Any) -> Any:
    """Canonicalize unordered manifest records without embedding local paths."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise EnvironmentRegistryError("Invalid portable lock field name.")
        return {key: _normalize_lock_manifest(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted(
            (_normalize_lock_manifest(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if value is None or type(value) in {bool, str, int, float}:
        return value
    raise EnvironmentRegistryError("Invalid portable lock manifest value.")


def normalize_environment(observation: dict[str, Any]) -> dict[str, Any]:
    """Canonical identity of software without original OS, GPU, drivers or ABI."""
    version = observation.get("schema_version") if isinstance(observation, dict) else None
    if type(version) is not int or version < 2 or version > OBSERVATION_SCHEMA_VERSION:
        raise EnvironmentRegistryError("Unsupported software observation schema.")
    kind = observation.get("kind")
    if kind not in {"observed_unlocked", "portable_locked"}:
        raise EnvironmentRegistryError("Unknown software environment kind.")
    expected = {"schema_version", "kind", "python", "teal", "distributions"}
    if kind == "portable_locked":
        expected |= {"extras", "lock"}
    if set(observation) != expected:
        raise EnvironmentRegistryError("Unexpected software environment fields.")
    python_info, teal = observation["python"], observation["teal"]
    if (not isinstance(python_info, dict)
        or set(python_info) != {"implementation", "version"}
        or any(not isinstance(v, str) or not v for v in python_info.values())
        or not isinstance(teal, dict)
        or not isinstance(teal.get("version"), str)
        or not teal["version"]):
        raise EnvironmentRegistryError("Invalid Python or TeAL identity.")
    records = observation["distributions"]
    if not isinstance(records, list):
        raise EnvironmentRegistryError("Distributions must be a list.")
    distributions = []
    for record in records:
        if not isinstance(record, dict):
            raise EnvironmentRegistryError("Invalid package record.")
        name, version = record.get("name"), record.get("version")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version:
            raise EnvironmentRegistryError("A package needs a name and version.")
        copy = dict(record)
        copy["name"] = _NAME_SEPARATOR.sub("-", name.strip()).lower()
        distributions.append(copy)
    result = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "kind": kind,
        "python": dict(python_info),
        "teal": dict(teal),
        "distributions": sorted(distributions, key=_canonical_json),
    }
    if kind == "portable_locked":
        extras, lock = observation.get("extras"), observation.get("lock")
        if (not isinstance(extras, list)
            or any(not isinstance(x, str) or not x.strip() for x in extras)
            or not isinstance(lock, dict) or not lock):
            raise EnvironmentRegistryError("Portable specification requires extras and lock.")
        if set(lock) != {"kind", "manifest", "sha256"} or not isinstance(lock.get("kind"), str):
            raise EnvironmentRegistryError("Portable lock requires kind, manifest and SHA-256.")
        manifest = _normalize_lock_manifest(lock["manifest"])
        if not isinstance(manifest, dict) or not manifest:
            raise EnvironmentRegistryError("Portable lock manifest must be a nonempty mapping.")
        digest = lock.get("sha256")
        expected_digest = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        if not isinstance(digest, str) or digest.lower() != expected_digest:
            raise EnvironmentRegistryError("Portable lock content hash does not match manifest.")
        source = teal.get("source")
        if (not isinstance(source, dict)
            or not any(
                isinstance(source.get(field), str)
                and re.fullmatch(r"[a-fA-F0-9]{40,64}", source[field])
                for field in ("commit", "sha256")
            )):
            raise EnvironmentRegistryError("Portable TeAL requires an immutable commit/hash.")
        result["extras"] = sorted(set(x.strip().lower() for x in extras))
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

    @property
    def opening_id(self) -> str:
        """Registered identity of the interpreter running this Project instance."""
        return self._opening_id

    def require_local_execution(self, environment_id: str | None = None) -> str:
        """Resolve the selector; fail before scientific work when it is not local.

        Setting a default or supplying an override never starts another
        interpreter. That execution route is not yet implemented.
        """
        selected = (
            self._default_id if environment_id is None
            else self.get(environment_id)["environment_id"]
        )
        if selected != self._opening_id:
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
        self.get(environment_id)
        path = Path(executable).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EnvironmentRegistryError(
                f"Interpreter {str(path)!r} does not exist or is not executable."
            )
        self._interpreter_bindings[environment_id] = path

    def bound_interpreter(self, environment_id: str) -> Path:
        """Return this session's executable, or reject an unbound/missing one."""
        self.get(environment_id)
        path = self._interpreter_bindings.get(environment_id)
        if path is None:
            raise EnvironmentRegistryError(
                f"Environment {environment_id!r} has no bound Python interpreter. "
                "Call project.environments.bind_interpreter(...)."
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
        realizations = spec["lock"]["manifest"].get("realizations")
        if not isinstance(realizations, list):
            return False
        for realization in realizations:
            if not isinstance(realization, dict) or set(realization) != {"distributions"}:
                continue
            try:
                candidate = _normalize_distribution_records(realization["distributions"])
            except EnvironmentRegistryError:
                continue
            if candidate == observed["distributions"]:
                return True
        return False

    def verify_current_portable(self, environment_id: str) -> dict[str, Any]:
        """Adopt a portable software ID only after checking the actual interpreter.

        The current physical platform remains a separate observation, unchanged.
        The verification is remembered for this exact observed installation.
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
        return {
            "environment_id": row["environment_id"],
            "fingerprint": row["fingerprint"],
            "registered_at": row["registered_at"],
            "observation": observation,
            "kind": observation["kind"],
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
