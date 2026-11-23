"""Opt-in uv managed child interpreters for the TextLength R1 pilot.

Installed virtual environments are host-local caches, never portable Project
artifacts. A successful local install is recorded as observed/unlocked unless
a separately defensible portable lock has been resolved and verified.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from typing import Any, Iterator

from text_analysis_lab.core.environments import (
    EnvironmentRegistryError,
    fingerprint_observation,
    normalize_environment,
)
from text_analysis_lab.core.secondary_coordinator import (
    PROTOCOL_VERSION,
    _pilot_code_digest,
    _run,
)

_SOURCE = "https://github.com/zacthinks/text-analysis-lab.git"
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_PYTHON = re.compile(r"^\d+\.\d+\.\d+$")
_EXTRA = re.compile(r"^[a-z][a-z0-9-]*$")
_TORCH_EXTRAS = {"transformers", "linguistics", "all"}
_CACHE_SCHEMA = 1


class ManagedEnvironmentError(EnvironmentRegistryError):
    """Unable to create or verify a managed local Python environment."""


def _cache_root() -> Path:
    override = os.environ.get("TEAL_MANAGED_ENVIRONMENTS_HOME")
    if override:
        return Path(override).expanduser().absolute()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "TextAnalysisLab" / "environments"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "text-analysis-lab" / "environments"


def _run_command(args: list[str], *, timeout: int = 1800) -> str:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedEnvironmentError(f"Managed environment command could not finish: {args[0]}") from exc
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        # Never expose credential-bearing source URLs from resolver output.
        message = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", message)
        raise ManagedEnvironmentError(
            f"Managed environment command failed (exit {completed.returncode}): "
            + message[-1800:]
        )
    return completed.stdout.strip()


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise ManagedEnvironmentError("uv is not installed or is not on PATH.")
    return executable


def _local_source_commit() -> str:
    source_root = Path(__file__).resolve().parents[3]
    try:
        root = _run_command(["git", "-C", str(source_root), "rev-parse", "--show-toplevel"], timeout=15)
        if Path(root).resolve() != source_root:
            raise ManagedEnvironmentError("The current TeAL source root is not a Git checkout.")
        changed = _run_command(["git", "-C", str(source_root), "status", "--porcelain"], timeout=15)
        if changed:
            raise ManagedEnvironmentError(
                "Managed TeAL installation requires a clean committed parent checkout."
            )
        commit = _run_command(["git", "-C", str(source_root), "rev-parse", "HEAD"], timeout=15)
    except ManagedEnvironmentError:
        raise
    except Exception as exc:
        raise ManagedEnvironmentError("Cannot identify the parent's immutable TeAL source.") from exc
    if not _COMMIT.fullmatch(commit):
        raise ManagedEnvironmentError("TeAL source did not resolve to a full immutable Git commit.")
    return commit


def _normalized_request(
    *, python: str | None, teal: str | None,
    teal_extras: list[str] | tuple[str, ...] | None, gpu: str | bool,
) -> dict[str, Any]:
    if python is None:
        python = ".".join(map(str, sys.version_info[:3]))
    if not isinstance(python, str) or not _PYTHON.fullmatch(python):
        raise ManagedEnvironmentError("python must specify exact major.minor.patch.")
    if type(gpu) is not bool and gpu != "auto":
        raise ManagedEnvironmentError("gpu must be 'auto', True, or False.")
    if not isinstance(teal_extras, (tuple, list)):
        raise ManagedEnvironmentError("teal_extras must be a sequence of extra names.")
    extras: list[str] = []
    for raw in teal_extras:
        if not isinstance(raw, str):
            raise ManagedEnvironmentError("TeAL extra names must be strings.")
        name = re.sub(r"[-_.]+", "-", raw.strip().lower())
        if not _EXTRA.fullmatch(name):
            raise ManagedEnvironmentError(f"Invalid TeAL extra name {raw!r}.")
        extras.append(name)
    extras = sorted(set(extras))
    available = {
        re.sub(r"[-_.]+", "-", name.lower())
        for name in (metadata.metadata("text-analysis-lab").get_all("Provides-Extra") or [])
    }
    unknown = sorted(set(extras) - available)
    if unknown:
        raise ManagedEnvironmentError(f"Unknown TeAL extras: {', '.join(unknown)}.")
    if teal is None:
        teal = _local_source_commit()
    if not isinstance(teal, str) or not _COMMIT.fullmatch(teal):
        raise ManagedEnvironmentError(
            "teal must be a 40-character Git commit; version-only references "
            "and moving branches are not reproducible."
        )
    # A mismatched child will be rejected by the actual R1 source probe.
    return {
        "python": python,
        "teal_commit": teal,
        "teal_version": metadata.version("text-analysis-lab"),
        "teal_extras": extras,
        "gpu": gpu,
    }


def plan(
    *, python: str | None = None, teal: str | None = None,
    teal_extras: list[str] | tuple[str, ...] | None = None,
    gpu: str | bool = "auto",
) -> dict[str, Any]:
    """Describe the managed installation without changing a cache or project."""
    request = _normalized_request(
        python=python, teal=teal, teal_extras=teal_extras, gpu=gpu,
    )
    torch_requested = bool(_TORCH_EXTRAS.intersection(request["teal_extras"])) or gpu is True
    return {
        **request,
        "source": f"git+{_SOURCE}@{request['teal_commit']}",
        "gpu_policy": (
            "strict_accelerator" if gpu is True
            else "cpu_only" if gpu is False else "prefer_accelerator"
        ),
        "torch_expected": torch_requested,
        "actions": ["create isolated uv virtual environment", "resolve and install TeAL",
                    "verify exact interpreter, TeAL, dependencies and R1 source",
                    "verify Torch device if installed", "publish user-local cache"],
        "portability": "observed_unlocked; universal cross-platform lock not established",
        "requires_explicit_create": True,
    }


@contextlib.contextmanager
def _request_lock(path: Path) -> Iterator[None]:
    """Cross-process advisory exclusion; the OS releases the lock on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _interpreter(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _probe(executable: Path, request: dict[str, Any]) -> dict[str, Any]:
    if not executable.is_file():
        raise ManagedEnvironmentError("Managed interpreter is missing.")
    exit_code, payload, stderr = _run(executable, "--probe", timeout=120)
    if exit_code or payload is None:
        raise ManagedEnvironmentError(f"Managed TeAL child probe failed: {stderr[-500:]}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ManagedEnvironmentError("Managed TeAL has an incompatible R1 protocol.")
    if payload.get("pilot_code_sha256") != _pilot_code_digest():
        raise ManagedEnvironmentError("Managed TeAL source differs from the parent R1 implementation.")
    observed = normalize_environment(payload["observation"])
    if observed["kind"] != "observed_unlocked":
        raise ManagedEnvironmentError("Managed TeAL did not report installed software.")
    if observed["python"]["version"] != request["python"]:
        raise ManagedEnvironmentError("Managed Python patch differs from the requested version.")
    if observed["teal"]["version"] != request["teal_version"]:
        raise ManagedEnvironmentError("Managed TeAL distribution version differs from the parent.")
    teal_dist = next(
        (row for row in observed["distributions"] if row["name"] == "text-analysis-lab"),
        None,
    )
    if teal_dist is None or teal_dist.get("vcs_commit") != request["teal_commit"]:
        raise ManagedEnvironmentError("Managed TeAL immutable Git revision was not verified.")
    return observed


def _gpu_probe(executable: Path, policy: str | bool) -> dict[str, Any]:
    script = (
        "import json,importlib.util\n"
        "if importlib.util.find_spec('torch') is None:\n"
        " print(json.dumps({'status':'not_applicable','backend':None,'reason':None}))\n"
        "else:\n"
        " import torch\n"
        " device='cpu'\n"
        " if torch.cuda.is_available(): device='cuda'\n"
        " elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): device='mps'\n"
        " elif hasattr(torch,'xpu') and torch.xpu.is_available(): device='xpu'\n"
        " elif hasattr(torch.version,'hip') and torch.version.hip: device='cuda'\n"
        " if device != 'cpu':\n"
        "  try:\n"
        "   x=torch.tensor([2.,3.],device=device); assert float(x.sum().item()) == 5.\n"
        "  except Exception: device='cpu'\n"
        " print(json.dumps({'status':'available' if device!='cpu' else 'cpu',"
        "'backend':device,'reason':None if device!='cpu' else 'No verified accelerator'}))\n"
    )
    output = _run_command([str(executable), "-c", script], timeout=120)
    try:
        state = json.loads(output.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ManagedEnvironmentError("Invalid Torch device verification result.") from exc
    if policy is True and state["status"] != "available":
        raise ManagedEnvironmentError(
            "Strict GPU environment requested but no working accelerator was verified."
        )
    if policy is False and state["status"] == "available":
        raise ManagedEnvironmentError("CPU-only policy unexpectedly selected an accelerator.")
    return state


def _verify(directory: Path, request: dict[str, Any], uv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    executable = _interpreter(directory)
    observed = _probe(executable, request)
    _run_command([uv, "pip", "check", "--python", str(executable)], timeout=120)
    gpu_state = _gpu_probe(executable, request["gpu"])
    return observed, gpu_state


def _source_requirement(request: dict[str, Any]) -> str:
    extras = request["teal_extras"]
    suffix = "[" + ",".join(extras) + "]" if extras else ""
    return f"text-analysis-lab{suffix} @ git+{_SOURCE}@{request['teal_commit']}"


def create(
    registry: Any, *, python: str | None = None, teal: str | None = None,
    teal_extras: list[str] | tuple[str, ...] | None = None,
    gpu: str | bool = "auto",
) -> dict[str, Any]:
    """Build or verify a cached environment and bind it for R1 delegation."""
    request = _normalized_request(
        python=python, teal=teal, teal_extras=teal_extras, gpu=gpu,
    )
    uv = _uv()
    root = _cache_root()
    # Request hashes select a local candidate, never a portable environment ID.
    request_key = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target = root / request_key
    with _request_lock(root / "locks" / (request_key + ".lock")):
        reused = False
        verified: tuple[dict[str, Any], dict[str, Any]] | None = None
        if target.is_dir():
            try:
                manifest = json.loads((target / "managed.json").read_text(encoding="utf-8"))
                if manifest == {"schema_version": _CACHE_SCHEMA, "request": request}:
                    verified = _verify(target, request, uv)
                    reused = True
            except (ManagedEnvironmentError, OSError, ValueError, KeyError):
                verified = None
        if verified is None:
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
            try:
                _run_command([uv, "venv", str(staging), "--python", request["python"]])
                executable = _interpreter(staging)
                args = [uv, "pip", "install", "--python", str(executable)]
                args += ["--torch-backend", "cpu" if gpu is False else "auto"]
                args.append(_source_requirement(request))
                if gpu is True and not _TORCH_EXTRAS.intersection(request["teal_extras"]):
                    args.append("torch")
                _run_command(args)
                verified = _verify(staging, request, uv)
                (staging / "managed.json").write_text(
                    json.dumps({"schema_version": _CACHE_SCHEMA, "request": request},
                               sort_keys=True), encoding="utf-8",
                )
                if target.exists():
                    # No live environment may be modified. A corrupt/stale cache
                    # is quarantined, then replaced while holding the request lock.
                    old = root / (".retired-" + request_key)
                    if old.exists():
                        shutil.rmtree(old)
                    target.rename(old)
                staging.rename(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        assert verified is not None
        observed, gpu_state = verified
        fingerprint = fingerprint_observation(observed)
        environment_id = registry._catalog.register_environment(
            fingerprint=fingerprint,
            schema_version=observed["schema_version"],
            observation_json=json.dumps(observed, sort_keys=True, separators=(",", ":")),
        )
        registry.bind_interpreter(environment_id, _interpreter(target))
        return {
            "environment_id": environment_id,
            "python": observed["python"]["version"],
            "teal_version": observed["teal"]["version"],
            "teal_source": {"vcs": "git", "commit": request["teal_commit"]},
            "teal_extras": request["teal_extras"],
            "verification_level": "observed_inventory_and_r1_source_match",
            "readiness": "verified_local",
            "gpu": gpu_state,
            "reused": reused,
            "portability": "observed_unlocked; universal lock not verified",
        }
