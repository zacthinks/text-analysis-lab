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
import time
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
_TORCH_EXTRAS = {"transformers", "linguistics", "geco", "all"}
_CACHE_SCHEMA = 1


class ManagedEnvironmentError(EnvironmentRegistryError):
    """Unable to create or verify a managed local Python environment."""


class ManagedAcceleratorError(ManagedEnvironmentError):
    """Torch import/tensor/backend failed after software and source verification."""


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
            # Installation builds must not import the parent's notebook paths.
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedEnvironmentError(f"Managed environment command could not finish: {args[0]}") from exc
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()
        # Never expose credential-bearing source URLs from resolver output.
        message = re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[redacted]@", message)
        message = re.sub(r"(?i)(https?://[^?\s]+)\?[^\s]+", r"\1?[redacted]", message)
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
    """Resolve a verifiable parent source from Git or trusted install metadata."""
    source_root = Path(__file__).resolve().parents[3]
    try:
        root = _run_command(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            timeout=15,
        )
        if Path(root).resolve() == source_root:
            changes = _run_command(
                ["git", "-C", str(source_root), "status", "--porcelain", "--",
                 "src/text_analysis_lab", "pyproject.toml"], timeout=15,
            )
            if changes:
                raise ManagedEnvironmentError(
                    "Managed TeAL installation requires committed parent source files."
                )
            commit = _run_command(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], timeout=15,
            )
            if _COMMIT.fullmatch(commit):
                return commit
    except ManagedEnvironmentError as exc:
        # A missing Git CLI or non-checkout installation may still have a
        # recorded immutable Git origin; dirty source files are never accepted.
        if "requires committed parent source files" in str(exc):
            raise

    try:
        installed = metadata.distribution("text-analysis-lab")
        raw = installed.read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
        origin = direct.get("url")
        vcs = direct.get("vcs_info", {})
        commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
        if (isinstance(origin, str)
                and re.fullmatch(
                    r"https://github\.com/zacthinks/text-analysis-lab(?:\.git)?/?",
                    origin,
                )
                and isinstance(vcs, dict)
                and vcs.get("vcs") == "git"
                and isinstance(commit, str)
                and _COMMIT.fullmatch(commit)):
            return commit
    except (metadata.PackageNotFoundError, ValueError, TypeError, OSError):
        pass
    raise ManagedEnvironmentError(
        "Cannot identify an immutable TeAL source. Run from a clean committed "
        "TeAL checkout or install TeAL from its pinned Git commit."
    )


def _normalized_request(
    *, python: str | None, teal: str | None,
    teal_extras: list[str] | tuple[str, ...] | None, gpu: str | bool,
) -> dict[str, Any]:
    if python is None:
        python = ".".join(map(str, sys.version_info[:3]))
    if not isinstance(python, str) or not _PYTHON.fullmatch(python):
        raise ManagedEnvironmentError("python must specify exact major.minor.patch.")
    if tuple(map(int, python.split("."))) < (3, 11, 0):
        raise ManagedEnvironmentError("This TeAL source requires Python 3.11 or newer.")
    if type(gpu) is not bool and gpu != "auto":
        raise ManagedEnvironmentError("gpu must be 'auto', True, or False.")
    if teal_extras is None:
        teal_extras = []
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
    if teal != _local_source_commit():
        raise ManagedEnvironmentError(
            "The R2 pilot requires the current parent TeAL source commit; "
            "different immutable revisions are not yet supported."
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
            # Only seed an empty lock file. Writing the locked byte on
            # every contender would fail before waiting on Windows.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            deadline = time.monotonic() + 1800
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise ManagedEnvironmentError(
                            "Timed out waiting for another managed installation."
                        ) from exc
                    time.sleep(0.2)
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
    exit_code, payload, stderr = _run(executable, "--probe", timeout=120, isolated=True)
    if exit_code or payload is None:
        raise ManagedEnvironmentError(f"Managed TeAL child probe failed: {stderr[-500:]}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ManagedEnvironmentError("Managed TeAL has an incompatible R1 protocol.")
    if payload.get("pilot_code_sha256") != _pilot_code_digest():
        raise ManagedEnvironmentError("Managed TeAL source differs from the parent R1 implementation.")
    observed = normalize_environment(payload["observation"])
    if observed["kind"] != "observed_unlocked":
        raise ManagedEnvironmentError("Managed TeAL did not report installed software.")
    if payload.get("environment_id") != "env_" + fingerprint_observation(observed):
        raise ManagedEnvironmentError("Managed TeAL claimed identity differs from observed inventory.")
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
    allowed_accelerators = policy is not False
    script = (
        "import json,importlib.util\n"
        "if importlib.util.find_spec('torch') is None:\n"
        " print(json.dumps({'status':'not_applicable','backend':None,'reason':None}))\n"
        "else:\n"
        " import torch\n"
        " device='cpu'\n"
        " x=torch.tensor([2.,3.],device='cpu'); assert float(x.sum().item()) == 5.\n"
        " if " + str(allowed_accelerators) + " and torch.cuda.is_available(): device='cuda'\n"
        " elif " + str(allowed_accelerators) + " and hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): device='mps'\n"
        " elif " + str(allowed_accelerators) + " and hasattr(torch,'xpu') and torch.xpu.is_available(): device='xpu'\n"
        " if device != 'cpu':\n"
        "  try:\n"
        "   x=torch.tensor([2.,3.],device=device); assert float(x.sum().item()) == 5.\n"
        "  except Exception: device='cpu'\n"
        " print(json.dumps({'status':'available' if device!='cpu' else 'cpu',"
        "'backend':('rocm' if device=='cuda' and torch.version.hip else device),"
        "'reason':None if device!='cpu' else 'No verified accelerator'}))\n"
    )
    try:
        output = _run_command([str(executable), "-I", "-c", script], timeout=120)
    except ManagedEnvironmentError as exc:
        # This child command does ONLY Torch import and tensor/backend checks.
        # TeAL source and protocol verification and uv pip check occur outside
        # this boundary and must never trigger an automatic CPU fallback.
        raise ManagedAcceleratorError("Torch runtime verification failed.") from exc
    try:
        state = json.loads(output.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ManagedAcceleratorError("Invalid Torch runtime verification result.") from exc
    if (not isinstance(state, dict)
            or state.get("status") not in {"not_applicable", "available", "cpu"}
            or state.get("backend") not in {None, "cpu", "cuda", "mps", "xpu", "rocm"}
            or state.get("reason") is not None
            and not isinstance(state["reason"], str)):
        raise ManagedAcceleratorError("Malformed Torch runtime verification result.")
    if policy is False and state["status"] == "cpu":
        state["reason"] = "CPU-only policy selected"
    if policy is True and state["status"] != "available":
        raise ManagedEnvironmentError(
            "Strict GPU environment requested but no working accelerator was verified."
        )
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


_FALLBACK_REASONS = {
    "GPU-preferred installation failed; CPU-only retry succeeded",
    "GPU-preferred Torch runtime verification failed; CPU-only retry succeeded",
}


def _request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_cache_manifest(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a local ready marker without trusting paths or its request."""
    if target.is_symlink() or not re.fullmatch(r"[a-f0-9]{64}", target.name):
        raise ManagedEnvironmentError("Invalid managed installation directory.")
    marker = json.loads((target / "managed.json").read_text(encoding="utf-8"))
    required = {"schema_version", "request", "observed_fingerprint"}
    # Existing R2 cache entries predate fallback_reason. They remain usable
    # after full re-verification; only newly published entries retain a reason.
    if (not isinstance(marker, dict)
            or not required.issubset(marker)
            or set(marker) - required - {"fallback_reason"}):
        raise ManagedEnvironmentError("Invalid managed cache metadata.")
    request = marker["request"]
    if (marker["schema_version"] != _CACHE_SCHEMA
            or not isinstance(request, dict)
            or set(request) != {"python", "teal_commit", "teal_version", "teal_extras", "gpu"}
            or not isinstance(request["python"], str)
            or _PYTHON.fullmatch(request["python"]) is None
            or not isinstance(request["teal_commit"], str)
            or _COMMIT.fullmatch(request["teal_commit"]) is None
            or not isinstance(request["teal_version"], str)
            or not request["teal_version"]
            or not isinstance(request["teal_extras"], list)
            or any(not isinstance(x, str) or _EXTRA.fullmatch(x) is None
                   for x in request["teal_extras"])
            or request["teal_extras"] != sorted(set(request["teal_extras"]))
            or not (type(request["gpu"]) is bool or request["gpu"] == "auto")
            or _request_key(request) != target.name
            or not isinstance(marker["observed_fingerprint"], str)
            or re.fullmatch(r"[a-f0-9]{64}", marker["observed_fingerprint"]) is None
            or marker.get("fallback_reason") is not None
            and (request["gpu"] != "auto"
                 or not isinstance(marker["fallback_reason"], str)
                 or marker["fallback_reason"] not in _FALLBACK_REASONS)):
        raise ManagedEnvironmentError("Invalid managed cache metadata.")
    return request, marker


def _verified_cache(target: Path, request: dict[str, Any], uv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stored_request, marker = _read_cache_manifest(target)
    if stored_request != request:
        raise ManagedEnvironmentError("Managed cache request does not match.")
    observation, gpu_state = _verify(target, request, uv)
    if fingerprint_observation(observation) != marker["observed_fingerprint"]:
        raise ManagedEnvironmentError("Managed environment software inventory drifted.")
    fallback = marker.get("fallback_reason")
    if fallback is not None:
        if gpu_state["status"] != "cpu":
            raise ManagedEnvironmentError("Managed CPU fallback metadata disagrees with runtime.")
        gpu_state = {**gpu_state, "reason": fallback}
    return observation, gpu_state


def find_cached_interpreter(environment_id: str) -> Path | None:
    """Rediscover a previously published installation by its verified E1 ID.

    Never trust the cache marker alone, install packages, or persist local
    executable paths in project history. All R1 source/protocol, uv dependency
    and actual inventory checks are repeated under the request lock.
    """
    if not isinstance(environment_id, str) or re.fullmatch(
        r"env_[a-f0-9]{64}", environment_id,
    ) is None:
        return None
    root = _cache_root()
    if not root.is_dir():
        return None
    candidates = []
    for target in root.iterdir():
        if target.is_symlink() or not target.is_dir() or re.fullmatch(
            r"[a-f0-9]{64}", target.name,
        ) is None:
            continue
        marker_path = target / "managed.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(marker, dict) or (
            marker.get("observed_fingerprint") != environment_id[4:]
        ):
            continue
        with _request_lock(root / "locks" / (target.name + ".lock")):
            try:
                request, _ = _read_cache_manifest(target)
                _verified_cache(target, request, _uv())
            except (OSError, ValueError, KeyError, ManagedEnvironmentError) as exc:
                raise ManagedEnvironmentError(
                    "Previously published managed interpreter failed re-verification; "
                    "the installation was left untouched."
                ) from exc
            candidates.append(_interpreter(target))
    if len(candidates) > 1:
        raise ManagedEnvironmentError(
            "Several verified managed installations claim the same observed environment."
        )
    return candidates[0] if candidates else None


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
    request_key = _request_key(request)
    target = root / request_key
    with _request_lock(root / "locks" / (request_key + ".lock")):
        reused = False
        verified: tuple[dict[str, Any], dict[str, Any]] | None = None
        if target.is_dir() and (target / "managed.json").exists():
            try:
                verified = _verified_cache(target, request, uv)
                reused = True
            except (ManagedEnvironmentError, OSError, ValueError, KeyError) as exc:
                # A published environment might still have running children.
                # Never remove or rebuild it in place on a cache mismatch.
                raise ManagedEnvironmentError(
                    "The existing managed interpreter is no longer verified; "
                    "it was left untouched to protect running operations. "
                    "Use a different request or repair this cache when idle."
                ) from exc
        if verified is None:
            # Virtual environments are not relocatable: installed console-script
            # shebangs may embed the interpreter's absolute path. Construct at
            # the final cache path, but do not publish the ready marker until
            # every verification passes. The lock prevents a concurrent reader.
            if target.exists():
                # No readiness marker: interrupted installation under a
                # previously released request lock is safe to clean up.
                if target.is_symlink() or not target.is_dir():
                    raise ManagedEnvironmentError("Managed cache target is not a directory.")
                shutil.rmtree(target)
            target.mkdir(parents=True)
            try:
                _run_command([uv, "venv", str(target), "--python", request["python"]])
                executable = _interpreter(target)
                args = [uv, "pip", "install", "--python", str(executable)]
                args += ["--torch-backend", "cpu" if gpu is False else "auto"]
                args.append(_source_requirement(request))
                if gpu is True and not _TORCH_EXTRAS.intersection(request["teal_extras"]):
                    args.append("torch")
                fallback_reason = None
                stage = "install"
                try:
                    _run_command(args)
                    stage = "verify"
                    verified = _verify(target, request, uv)
                except ManagedEnvironmentError as primary_error:
                    # uv resolution errors may be retried under the existing
                    # auto policy. AFTER installation only an actual Torch
                    # import/device/tensor failure can trigger CPU rebuilding:
                    # missing TeAL source, mismatched R1 protocol, uv pip check
                    # and other general verification problems remain fatal.
                    if (gpu != "auto"
                            or not _TORCH_EXTRAS.intersection(request["teal_extras"])
                            or not (stage == "install"
                                    or isinstance(primary_error, ManagedAcceleratorError))):
                        raise
                    shutil.rmtree(target)
                    target.mkdir()
                    _run_command([uv, "venv", str(target), "--python", request["python"]])
                    executable = _interpreter(target)
                    cpu_args = [uv, "pip", "install", "--python", str(executable),
                                "--torch-backend", "cpu", _source_requirement(request)]
                    try:
                        _run_command(cpu_args)
                        verified = _verify(target, request, uv)
                    except ManagedEnvironmentError as cpu_error:
                        raise ManagedEnvironmentError(
                            "GPU-preferred setup and CPU fallback both failed; "
                            f"CPU error: {cpu_error}"
                        ) from primary_error
                    fallback_reason = (
                        "GPU-preferred Torch runtime verification failed; CPU-only retry succeeded"
                        if stage == "verify" else
                        "GPU-preferred installation failed; CPU-only retry succeeded"
                    )
                if fallback_reason is not None:
                    observed, gpu_state = verified
                    if gpu_state["status"] != "available":
                        gpu_state = {**gpu_state, "reason": fallback_reason}
                    verified = (observed, gpu_state)
                temp_marker = target / "managed.json.pending"
                temp_marker.write_text(
                    json.dumps({
                        "schema_version": _CACHE_SCHEMA, "request": request,
                        "observed_fingerprint": fingerprint_observation(verified[0]),
                        "fallback_reason": fallback_reason,
                    }, sort_keys=True), encoding="utf-8",
                )
                temp_marker.replace(target / "managed.json")
            except BaseException:
                shutil.rmtree(target, ignore_errors=True)
                raise
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
