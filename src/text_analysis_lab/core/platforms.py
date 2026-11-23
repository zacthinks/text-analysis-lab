"""Immutable, privacy-limited observations of the execution platform.

The platform registry is independent of the Python software environment.
It observes a configuration, not a unique computer or a provisioning target.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.errors import TeALError

if TYPE_CHECKING:
    from text_analysis_lab.core.catalog import ProjectCatalog

PLATFORM_SCHEMA_VERSION = 1


class PlatformRegistryError(TeALError):
    """A platform observation or registry entry is invalid."""


class PlatformNotFoundError(PlatformRegistryError, LookupError):
    """The requested observed platform is not registered."""


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_label(value: object, *, length: int = 160) -> str:
    """Keep only bounded printable hardware descriptors, never identifiers."""
    raw = str(value or "").strip()
    return "".join(ch for ch in raw if ch.isprintable() and ch not in "\r\n")[:length]


def _nvidia_devices() -> list[dict[str, str]]:
    """Best-effort GPU/driver query: no UUID, serial, bus ID or utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode:
            return []
        devices = []
        for line in result.stdout.splitlines()[:16]:
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            model, driver = (_safe_label(part) for part in parts)
            if model and driver:
                devices.append({
                    "backend": "nvidia",
                    "model": model,
                    "driver_version": driver,
                })
        return sorted(devices, key=_canonical)
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return []


def observe_platform() -> dict[str, Any]:
    """Collect stable, non-identifying facts; missing GPU data is acceptable.

    This does not identify a physical machine and does not claim no GPU is
    present when a vendor-specific probe returns nothing.
    """
    def optional(call) -> str:
        try:
            return _safe_label(call())
        except (OSError, ValueError, AttributeError):
            return ""

    return {
        "schema_version": PLATFORM_SCHEMA_VERSION,
        "os": {
            "system": optional(platform.system),
            "release": optional(platform.release),
            "kernel": optional(platform.version),
        },
        "architecture": optional(platform.machine),
        "python_architecture": optional(lambda: platform.architecture()[0]),
        "cpu": {"processor": optional(platform.processor)},
        "accelerators": _nvidia_devices(),
    }


def normalize_platform(observation: dict[str, Any]) -> dict[str, Any]:
    """Canonical, intentionally small whitelist; reject private host fields."""
    if not isinstance(observation, dict) or set(observation) != {
        "schema_version", "os", "architecture", "python_architecture",
        "cpu", "accelerators",
    } or observation["schema_version"] != PLATFORM_SCHEMA_VERSION:
        raise PlatformRegistryError("Invalid platform observation schema.")
    os_info = observation["os"]
    cpu = observation["cpu"]
    devices = observation["accelerators"]
    if (
        not isinstance(os_info, dict)
        or set(os_info) != {"system", "release", "kernel"}
        or not isinstance(cpu, dict)
        or set(cpu) != {"processor"}
        or not isinstance(devices, list)
    ):
        raise PlatformRegistryError("Invalid platform observation fields.")
    if any(not isinstance(value, str) or len(value) > 160
           for value in (*os_info.values(), *cpu.values(),
                         observation["architecture"], observation["python_architecture"])):
        raise PlatformRegistryError("Invalid platform observation label.")
    normalized_devices = []
    for device in devices:
        if (not isinstance(device, dict)
                or set(device) != {"backend", "model", "driver_version"}
                or any(not isinstance(value, str) or len(value) > 160
                       for value in device.values())):
            raise PlatformRegistryError("Invalid accelerator observation.")
        normalized_devices.append(dict(device))
    result = dict(observation)
    result["os"] = dict(os_info)
    result["cpu"] = dict(cpu)
    result["accelerators"] = sorted(normalized_devices, key=_canonical)
    return result


def fingerprint_platform(observation: dict[str, Any]) -> str:
    try:
        return hashlib.sha256(
            _canonical(normalize_platform(observation)).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise PlatformRegistryError("Invalid platform observation.") from exc


class PlatformRegistry:
    """Append-only history of observed execution platforms, without a default."""

    def __init__(self, catalog: ProjectCatalog) -> None:
        self._catalog = catalog
        observation = normalize_platform(observe_platform())
        fingerprint = fingerprint_platform(observation)
        self._opening_id = catalog.register_platform(
            fingerprint=fingerprint,
            schema_version=PLATFORM_SCHEMA_VERSION,
            observation_json=_canonical(observation),
        )

    @property
    def opening_id(self) -> str:
        """Actual host observation for this Project instance."""
        return self._opening_id

    def _public_row(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            observation = normalize_platform(json.loads(row["observation_json"]))
            if (fingerprint_platform(observation) != row["fingerprint"]
                    or row["platform_id"] != "platform_" + row["fingerprint"]
                    or row["schema_version"] != observation["schema_version"]):
                raise PlatformRegistryError("Platform registry fingerprint mismatch.")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise PlatformRegistryError("Corrupt platform registry record.") from exc
        return {
            "platform_id": row["platform_id"],
            "fingerprint": row["fingerprint"],
            "registered_at": row["registered_at"],
            "observation": observation,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self._public_row(row) for row in self._catalog.list_platforms()]

    def get(self, platform_id: str) -> dict[str, Any]:
        if not isinstance(platform_id, str):
            raise PlatformNotFoundError("Platform ID must be a string.")
        row = self._catalog.get_platform(platform_id)
        if row is None:
            raise PlatformNotFoundError(f"Platform {platform_id!r} is not registered.")
        return self._public_row(row)
