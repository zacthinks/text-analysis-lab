"""Opt-in uv provisioning for the R0/R1 TeAL environment registries.

This module owns only host-local installations. Project metadata contains
software observations, never cached venv paths. Installed inventories are not
automatically promoted to universal portable locks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from packaging.requirements import Requirement

from text_analysis_lab.core.environments import (
    OBSERVATION_SCHEMA_VERSION,
    EnvironmentRegistryError,
    _canonical_json,
    fingerprint_observation,
    normalize_environment,
)

_SOURCE_URL = "https://github.com/zacthinks/text-analysis-lab.git"
_COMMIT = re.compile(r"^[a-fA-F0-9]{40}$")
_PYTHON = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_EXTRA = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MANIFEST_NAME = "managed.json"


@dataclass(frozen=True)
class ManagedEnvironmentPlan:
    python: str
    teal_version: str
    teal_commit: str
    teal_extras: tuple[str, ...]
    gpu: str
    torch_required: bool
    source: str
    actions: tuple[str, ...]
    portability: str = "observed_unlocked; universal wheel/source lock not established"


@dataclass(frozen=True)
class ManagedEnvironmentResult:
    environment_id: str
    python: str
    teal_version: str
    teal_commit: str
    teal_extras: tuple[str, ...]
    gpu: str
    backend: str
    fallback_reason: str | None
    verification: str
    reused: bool
    portability: str = "observed_unlocked; universal wheel/source lock not established"


def _root() -> Path:
    override = os.environ.get("TEAL_MANAGED_ENV_ROOT")
    if override:
        return Path(override).expanduser().absolute()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "TeAL" / "environments" if base else Path.home() / "AppData" / "Local" / "TeAL" / "environments"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "text-analysis-lab" / "environments"


def _python_executable(directory: Path) -> Path:
    return directory / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)
    env["UV_NO_PROGRESS"] = "1"
    return env


def sanitized_child_env(executable: Path) -> dict[str, str] | None:
    """Use isolated imports only for executables beneath our own managed cache."""
    root = _root().resolve()
    try:
        Path(executable).absolute().relative_to(root)
    except ValueError:
        return None
    return _clean_env()


def _invoke(command: list[str], *, timeout: int = 1800) -> str:
    try:
        result = subprocess.run(
            command, cwd=str(_root()), env=_clean_env(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentRegistryError(f"Managed uv operation could not complete: {type(exc).__name__}.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1800:]
        # Never print index credentials, access tokens, or user-info URLs.
        detail = re.sub(r"https?://[^/\s@]+@", "https://[redacted]@", detail)
        detail = re.sub(r"(?i)(token|password|secret)=\S+", r"\1=[redacted]", detail)
        raise EnvironmentRegistryError(
            f"Managed uv operation failed (exit {result.returncode}): {detail}"
        )
    return result.stdout


def _uv() -> str:
    executable = shutil.which("uv")
    if not executable:
        raise EnvironmentRegistryError("uv is required for managed environment creation; install uv first.")
    return executable


def _source_commit(requested: str | None) -> str:
    from text_analysis_lab.core.secondary_coordinator import _pilot_code_digest

    _pilot_code_digest()  # Reject unverifiable installed source before provisioning.
    current = Path(__file__).resolve()
    root = next(
        (p for p in current.parents if (p / "pyproject.toml").is_file() and (p / ".git").exists()),
        None,
    )
    if root is None:
        raise EnvironmentRegistryError(
            "Managed TeAL requires an immutable source commit from a clean source checkout; "
            "the current installation has no verifiable Git checkout."
        )
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args], text=True, capture_output=True, check=False, timeout=30,
        )
        if result.returncode:
            raise EnvironmentRegistryError("Unable to verify TeAL's Git source revision.")
        return result.stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise EnvironmentRegistryError(
            "The current TeAL checkout has uncommitted tracked changes; commit them before a reproducible managed install."
        )
    head = git("rev-parse", "HEAD").lower()
    if not _COMMIT.fullmatch(head):
        raise EnvironmentRegistryError("TeAL checkout did not resolve to an immutable commit.")
    if requested is None:
        return head
    if not isinstance(requested, str) or not _COMMIT.fullmatch(requested):
        raise EnvironmentRegistryError(
            "teal must be an exact 40-character Git commit, not a version string or moving branch."
        )
    if requested.lower() != head:
        raise EnvironmentRegistryError(
            "R1 requires matching TeAL implementation bytes; selecting another commit "
            "is unsupported until cross-version protocol compatibility is defined."
        )
    return head


def _extras(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if values is None:
        values = []
    if not isinstance(values, (list, tuple)):
        raise EnvironmentRegistryError("teal_extras must be a list of supported extra names.")
    normalized: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not _EXTRA.fullmatch(item.lower()):
            raise EnvironmentRegistryError("Invalid TeAL extra name.")
        normalized.add(item.lower().replace("_", "-"))
    declared = {
        x.lower().replace("_", "-")
        for x in (metadata.distribution("text-analysis-lab").metadata.get_all("Provides-Extra") or [])
    }
    missing = normalized - declared
    if missing:
        raise EnvironmentRegistryError(f"Unknown TeAL extras: {', '.join(sorted(missing))}.")
    return tuple(sorted(normalized))


def _torch_required(extras: tuple[str, ...]) -> bool:
    dist = metadata.distribution("text-analysis-lab")
    for raw in dist.requires or []:
        try:
            requirement = Requirement(raw)
        except ValueError:
            continue
        if requirement.name.lower().replace("_", "-") != "torch":
            continue
        if requirement.marker is None:
            return True
        for extra in extras:
            if requirement.marker.evaluate({"extra": extra}):
                return True
    return False


def plan(*, python: str | None = None, teal: str | None = None,
         teal_extras: list[str] | tuple[str, ...] | None = None,
         gpu: str | bool = "auto") -> ManagedEnvironmentPlan:
    """Resolve an immutable source request without downloading or modifying anything."""
    if python is None:
        python = ".".join(str(x) for x in sys.version_info[:3])
    if not isinstance(python, str) or not _PYTHON.fullmatch(python):
        raise EnvironmentRegistryError("Managed Python must be an exact major.minor.micro version.")
    if python.split(".")[0] != "3" or int(python.split(".")[1]) < 11:
        raise EnvironmentRegistryError("TeAL requires Python >= 3.11.")
    if gpu not in ("auto", True, False) or (gpu is not True and gpu is not False and gpu != "auto"):
        raise EnvironmentRegistryError("gpu must be 'auto', True, or False.")
    chosen_extras = _extras(teal_extras)
    commit = _source_commit(teal)
    return ManagedEnvironmentPlan(
        python=python,
        teal_version=metadata.version("text-analysis-lab"),
        teal_commit=commit,
        teal_extras=chosen_extras,
        gpu="auto" if gpu == "auto" else "required" if gpu is True else "cpu",
        torch_required=_torch_required(chosen_extras) or gpu is True,
        source=f"git+{_SOURCE_URL}@{commit}",
        actions=("create isolated uv venv", "install pinned TeAL and requested extras",
                 "validate exact Python and R1 source/protocol", "check dependencies and compute backend",
                 "publish cache and register observed software"),
    )


def _request_key(chosen: ManagedEnvironmentPlan) -> str:
    return hashlib.sha256(_canonical_json({
        "python": chosen.python, "teal_commit": chosen.teal_commit,
        "teal_extras": list(chosen.teal_extras), "gpu": chosen.gpu,
    }).encode("utf-8")).hexdigest()


@contextmanager
def _installation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            import time
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\x00")
                handle.flush()
            handle.seek(0)
            deadline = time.monotonic() + 1800
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise EnvironmentRegistryError("Timed out waiting for managed installation lock.") from exc
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_child(executable: Path, args: list[str], *, stdin: str | None = None) -> dict[str, Any]:
    result = subprocess.run(
        [str(executable), *args], input=stdin, text=True, capture_output=True,
        env=_clean_env(), cwd=str(_root()), check=False, timeout=120,
    )
    if result.returncode:
        raise EnvironmentRegistryError(
            f"Managed child verification failed (exit {result.returncode}): "
            f"{result.stderr[-1000:]}"
        )
    try:
        value = json.loads(result.stdout.strip())
    except ValueError as exc:
        raise EnvironmentRegistryError("Managed child returned invalid verification JSON.") from exc
    if not isinstance(value, dict):
        raise EnvironmentRegistryError("Managed child returned unexpected verification data.")
    return value


def _probe(executable: Path, chosen: ManagedEnvironmentPlan) -> dict[str, Any]:
    from text_analysis_lab.core.secondary_coordinator import PROTOCOL_VERSION, _pilot_code_digest

    payload = _run_child(
        executable, ["-m", "text_analysis_lab.core.secondary_coordinator", "--probe"],
    )
    observed = normalize_environment(payload.get("observation"))
    if (observed["kind"] != "observed_unlocked"
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("pilot_code_sha256") != _pilot_code_digest()
            or payload.get("teal_version") != chosen.teal_version
            or payload.get("environment_id") != "env_" + fingerprint_observation(observed)
            or observed["python"]["version"] != chosen.python
            or observed["teal"]["version"] != chosen.teal_version):
        raise EnvironmentRegistryError(
            "Managed child has incompatible Python, TeAL source, installed metadata, or R1 protocol."
        )
    teal_records = [d for d in observed["distributions"] if d["name"] == "text-analysis-lab"]
    if len(teal_records) != 1 or teal_records[0].get("vcs_commit") != chosen.teal_commit:
        raise EnvironmentRegistryError(
            "Managed child TeAL installation lacks the requested immutable Git source identity."
        )
    return observed


def _device_check(executable: Path, chosen: ManagedEnvironmentPlan) -> tuple[str, str | None]:
    if not chosen.torch_required:
        return "not_applicable", None
    script = (
        "import json, torch\n"
        "d = 'cpu'\n"
        "if torch.cuda.is_available(): d = 'cuda'\n"
        "elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): d = 'mps'\n"
        "elif hasattr(torch, 'xpu') and torch.xpu.is_available(): d = 'xpu'\n"
        "value = (torch.ones((2,), device=d) + 1).sum().item()\n"
        "assert value == 4.0\n"
        "name = ('rocm' if d == 'cuda' and torch.version.hip else d)\n"
        "print(json.dumps({'device':name}))\n"
    )
    result = _run_child(executable, ["-c", script])
    device = result.get("device")
    if device not in {"cpu", "cuda", "mps", "xpu", "rocm"}:
        raise EnvironmentRegistryError("Invalid child Torch backend verification.")
    if chosen.gpu == "required" and device == "cpu":
        raise EnvironmentRegistryError("A working GPU backend is required, but only CPU passed verification.")
    if chosen.gpu == "cpu" and device != "cpu":
        raise EnvironmentRegistryError("CPU-only request unexpectedly selected an accelerator.")
    return device, ("No supported working accelerator; CPU backend selected." if chosen.gpu == "auto" and device == "cpu" else None)


def _build(executable_uv: str, directory: Path, chosen: ManagedEnvironmentPlan,
           *, torch_backend: str | None) -> tuple[dict[str, Any], str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    venv_dir = directory / "venv"
    _invoke([executable_uv, "--no-config", "venv", "--python", chosen.python, str(venv_dir)])
    python_exe = _python_executable(venv_dir)
    extras = "[" + ",".join(chosen.teal_extras) + "]" if chosen.teal_extras else ""
    requirement = f"text-analysis-lab{extras} @ git+{_SOURCE_URL}@{chosen.teal_commit}"
    cmd = [executable_uv, "--no-config", "pip", "install", "--python", str(python_exe)]
    if torch_backend is not None:
        cmd.extend(["--torch-backend", torch_backend])
    cmd.append(requirement)
    if chosen.gpu == "required" and not _torch_required(chosen.teal_extras):
        cmd.append("torch>=2.0")
    _invoke(cmd)
    _invoke([executable_uv, "--no-config", "pip", "check", "--python", str(python_exe)])
    observation = _probe(python_exe, chosen)
    backend, reason = _device_check(python_exe, chosen)
    return observation, backend, reason


def _valid_cached(directory: Path, chosen: ManagedEnvironmentPlan, uv_path: str) -> tuple[dict[str, Any], str, str | None] | None:
    meta_path = directory / _MANIFEST_NAME
    if not meta_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored.get("request_key") != _request_key(chosen):
            return None
        executable = _python_executable(directory / "venv")
        _invoke([uv_path, "--no-config", "pip", "check", "--python", str(executable)], timeout=120)
        actual = _probe(executable, chosen)
        backend, reason = _device_check(executable, chosen)
        if (stored.get("observed_id") != "env_" + fingerprint_observation(actual)
                or stored.get("backend") != backend):
            return None
        return actual, backend, reason
    except (OSError, ValueError, EnvironmentRegistryError, subprocess.TimeoutExpired):
        return None


def _register(registry: Any, directory: Path, observed: dict[str, Any], chosen: ManagedEnvironmentPlan,
              backend: str, reason: str | None, reused: bool) -> ManagedEnvironmentResult:
    fingerprint = fingerprint_observation(observed)
    environment_id = registry._catalog.register_environment(
        fingerprint=fingerprint, schema_version=OBSERVATION_SCHEMA_VERSION,
        observation_json=_canonical_json(observed),
    )
    registry.bind_interpreter(environment_id, _python_executable(directory / "venv"))
    return ManagedEnvironmentResult(
        environment_id=environment_id, python=chosen.python,
        teal_version=chosen.teal_version, teal_commit=chosen.teal_commit,
        teal_extras=chosen.teal_extras, gpu=chosen.gpu,
        backend=backend, fallback_reason=reason,
        verification="observed_inventory+R1_source_protocol+uv_pip_check+tensor_smoke" if chosen.torch_required
        else "observed_inventory+R1_source_protocol+uv_pip_check",
        reused=reused,
    )


def find_cached_interpreter(environment_id: str) -> Path | None:
    """Resolve only locally published environments; R1 will probe on each use."""
    root = _root()
    if not root.is_dir():
        return None
    for entry in root.iterdir():
        if not entry.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", entry.name):
            continue
        meta = entry / _MANIFEST_NAME
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        executable = _python_executable(entry / "venv")
        if data.get("observed_id") == environment_id and executable.is_file():
            return executable
    return None


def create(registry: Any, *, python: str | None = None, teal: str | None = None,
           teal_extras: list[str] | tuple[str, ...] | None = None,
           gpu: str | bool = "auto") -> ManagedEnvironmentResult:
    """Explicit opt-in: stage, verify, publish, register, and bind one uv installation."""
    chosen = plan(python=python, teal=teal, teal_extras=teal_extras, gpu=gpu)
    uv = _uv()
    cache = _root()
    cache.mkdir(parents=True, exist_ok=True)
    key = _request_key(chosen)
    target = cache / key
    with _installation_lock(cache / "locks" / (key + ".lock")):
        existing = _valid_cached(target, chosen, uv) if target.exists() else None
        if existing is not None:
            return _register(registry, target, existing[0], chosen, existing[1], existing[2], True)

        staged = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(cache)))
        old: Path | None = None
        try:
            selected_backend = None if not chosen.torch_required else (
                "cpu" if chosen.gpu == "cpu" else "auto"
            )
            try:
                observed, backend, reason = _build(uv, staged, chosen, torch_backend=selected_backend)
            except EnvironmentRegistryError as exc:
                if chosen.gpu != "auto" or not chosen.torch_required:
                    raise
                # Retry in a new staging venv; never mutate an existing install.
                shutil.rmtree(staged)
                staged.mkdir()
                observed, backend, reason = _build(uv, staged, chosen, torch_backend="cpu")
                reason = f"Automatic accelerator setup failed; a clean CPU installation passed: {type(exc).__name__}."
            (staged / _MANIFEST_NAME).write_text(
                _canonical_json({
                    "request_key": key, "observed_id": "env_" + fingerprint_observation(observed),
                    "backend": backend, "python": chosen.python, "teal_commit": chosen.teal_commit,
                    "extras": list(chosen.teal_extras), "gpu": chosen.gpu,
                    "verification": "installed_metadata_and_runtime_only",
                }),
                encoding="utf-8",
            )
            if target.exists():
                old = cache / (".retired-" + uuid4().hex)
                target.rename(old)
            try:
                staged.rename(target)
            except OSError:
                if old is not None:
                    old.rename(target)
                    old = None
                raise
            staged = None
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
            return _register(registry, target, observed, chosen, backend, reason, False)
        finally:
            if staged is not None:
                shutil.rmtree(staged, ignore_errors=True)
