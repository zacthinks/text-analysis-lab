"""Supervised, interpreter-selected launches of TeAL-linked GeCo workspaces.

Only JSON, names and local filesystem paths cross the child boundary. The child
opens TeAL and GeCo anew, and owns its linked provider and HTTP listener.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from text_analysis_lab.core.secondary_coordinator import (
    _check_probe, _probe_payload, _verified_executable,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


class GeCoLaunchError(RuntimeError):
    """The selected Python environment cannot launch its linked GeCo server."""


_PROTOCOL = 1
_TIMEOUT = 120.0


def _source_digest() -> str:
    """Check the R3 boundary itself as well as R1's TeAL source probe."""
    from text_analysis_lab.integrations import geco
    digest = hashlib.sha256()
    for module in (sys.modules[__name__], geco):
        source = getattr(module, "__file__", None)
        if source is None or not source.endswith(".py"):
            raise GeCoLaunchError("Cannot verify TeAL GeCo launcher source.")
        digest.update(Path(source).read_bytes())
    return digest.hexdigest()


def _write_status(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _log_tail(path: Path) -> str:
    with suppress(OSError):
        return path.read_text(encoding="utf-8", errors="replace")[-3000:]
    return ""


def _http_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return 200 <= response.status < 400
    except (OSError, URLError, ValueError):
        return False


class ChildGeCoServer:
    """One project-owned child GeCo server, not a general process supervisor."""

    def __init__(
        self, process: subprocess.Popen[str], *, temporary: tempfile.TemporaryDirectory[str],
        log_path: Path, url: str, port: int, device: str | None,
    ) -> None:
        self._process = process
        self._temporary = temporary
        self._log_path = log_path
        self._url = url
        self._port = port
        self._device = device
        self._closing = False
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return self._url

    @property
    def port(self) -> int:
        return self._port

    @property
    def execution_device(self) -> str | None:
        return self._device

    @property
    def closed(self) -> bool:
        return self._process.poll() is not None

    def wait(self, timeout: float | None = None) -> bool:
        try:
            code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        if code != 0 and not self._closing:
            raise GeCoLaunchError(
                f"GeCo child exited unexpectedly (exit={code}): {_log_tail(self._log_path)}"
            )
        return True

    def close(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        try:
            if self._process.poll() is None:
                try:
                    assert self._process.stdin is not None
                    self._process.stdin.write("stop\n")
                    self._process.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
        finally:
            if self._process.stdin is not None:
                with suppress(OSError, ValueError):
                    self._process.stdin.close()
            self._temporary.cleanup()

    def __enter__(self) -> ChildGeCoServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def launch(
    project: Project, name: str, *, environment: str,
    focus: bool, host: str, port: int | None, warmup: bool, device: str | None,
    debug: bool,
) -> ChildGeCoServer:
    """Start, verify, and own exactly one child server."""
    # This verifies actual installed inventory and compatible R1 TeAL source.
    executable = _verified_executable(project, environment)
    temp = tempfile.TemporaryDirectory(prefix="teal-geco-")
    directory = Path(temp.name)
    status = directory / "status.json"
    log_path = directory / "child.log"
    request_id = uuid4().hex
    request = {
        "protocol": _PROTOCOL, "request_id": request_id,
        "project_path": str(project.path.resolve()), "workspace": name,
        "environment": environment, "focus": focus, "host": host,
        "port": port, "warmup": warmup, "device": device, "debug": debug,
        "status_path": str(status),
    }
    process = None
    try:
        # -I ignores parent PYTHONPATH/user site, even for manually bound venvs.
        # Missing installed TeAL/GeCo is an error, never a parent fallback.
        env = {k: v for k, v in os.environ.items() if k not in {
            "PYTHONPATH", "PYTHONHOME",
        }}
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [str(executable), "-I", "-m",
                 "text_analysis_lab.integrations.geco_process", "--child"],
                stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                text=True, env=env,
            )
        assert process.stdin is not None
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.flush()

        deadline = time.monotonic() + _TIMEOUT
        record: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if status.is_file():
                try:
                    candidate = json.loads(status.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    candidate = None
                if isinstance(candidate, dict):
                    record = candidate
                    break
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if record is None:
            raise GeCoLaunchError(
                f"GeCo child did not become ready (exit={process.poll()}). "
                f"{_log_tail(log_path)}"
            )
        if (record.get("protocol") != _PROTOCOL
                or record.get("request_id") != request_id
                or record.get("workspace") != name):
            raise GeCoLaunchError("GeCo child returned a mismatched readiness response.")
        if record.get("status") == "error":
            raise GeCoLaunchError(
                f"GeCo launch failed in selected environment {environment}: "
                f"{record.get('error')}. {_log_tail(log_path)}"
            )
        if record.get("status") != "ready":
            raise GeCoLaunchError("Unexpected GeCo child status.")
        if record.get("source_digest") != _source_digest():
            raise GeCoLaunchError("GeCo child has incompatible TeAL launch source.")
        _check_probe(project, environment, record["probe"])
        url = record.get("url")
        bound_port = record.get("port")
        resolved_device = record.get("execution_device")
        if (not isinstance(url, str) or not url.startswith(("http://", "https://"))
                or type(bound_port) is not int or not 0 < bound_port <= 65535
                or (resolved_device is not None and not isinstance(resolved_device, str))):
            raise GeCoLaunchError("GeCo child provided invalid server readiness fields.")
        # Do not trust process creation or a file alone as HTTP readiness.
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise GeCoLaunchError("GeCo child exited before HTTP readiness.")
            if _http_ready(url):
                return ChildGeCoServer(
                    process, temporary=temp, log_path=log_path, url=url,
                    port=bound_port, device=resolved_device,
                )
            time.sleep(0.15)
        raise GeCoLaunchError("GeCo child started but HTTP endpoint was not ready.")
    except Exception:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin is not None:
                with suppress(OSError, ValueError):
                    process.stdin.close()
        temp.cleanup()
        raise


def _child_main() -> int:
    """Child entry point: the only permitted command is a single launch request."""
    from text_analysis_lab.core.project import Project
    from text_analysis_lab.integrations.geco import _load_geometric_coder

    record: dict[str, Any] | None = None
    status_path: Path | None = None
    project: Project | None = None
    server: Any = None
    try:
        raw = sys.stdin.readline()
        record = json.loads(raw)
        expected = {
            "protocol", "request_id", "project_path", "workspace",
            "environment", "focus", "host", "port", "warmup", "device",
            "debug", "status_path",
        }
        if not isinstance(record, dict) or set(record) != expected:
            raise GeCoLaunchError("Invalid R3 child launch request.")
        if record["protocol"] != _PROTOCOL or not isinstance(record["request_id"], str):
            raise GeCoLaunchError("Incompatible R3 child launch protocol.")
        if not isinstance(record["workspace"], str) or not isinstance(record["environment"], str):
            raise GeCoLaunchError("Invalid requested workspace/environment.")
        status_path = Path(record["status_path"])
        project = Project.open(record["project_path"])
        selected = record["environment"]
        # R1's probe is returned and independently checked by the parent.
        probe = _probe_payload()
        _load_geometric_coder()  # Check current public GeCo integration capabilities.
        linked = project.geco.open(record["workspace"])
        if record["focus"]:
            if not linked.is_focus_coder:
                raise GeCoLaunchError("Requested Focus Coding for an exploratory workspace.")
            server = linked.launch_focus(
                host=record["host"], port=record["port"],
                debug=record["debug"], use_reloader=False,
            )
        else:
            if linked.is_focus_coder:
                raise GeCoLaunchError("Focus Coding requires launch_focus(...).")
            server = linked.launch(
                host=record["host"], port=record["port"],
                debug=record["debug"], use_reloader=False,
                warmup=record["warmup"], device=record["device"],
            )
        _write_status(status_path, {
            "protocol": _PROTOCOL, "request_id": record["request_id"],
            "workspace": record["workspace"], "status": "ready",
            "url": server.url, "port": server.port,
            "execution_device": server.execution_device,
            "source_digest": _source_digest(), "probe": probe,
        })
        stopped = threading.Event()

        def read_stop() -> None:
            try:
                # EOF on parent death also closes the child server.
                sys.stdin.readline()
            finally:
                stopped.set()

        threading.Thread(target=read_stop, daemon=True, name="teal-geco-control").start()
        while not stopped.is_set():
            if server.wait(timeout=0.25):
                break
        return 0
    except Exception as exc:
        if status_path is not None and record is not None:
            with suppress(OSError, TypeError, ValueError):
                _write_status(status_path, {
                    "protocol": _PROTOCOL, "request_id": record["request_id"],
                    "workspace": record["workspace"], "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        print(f"TeAL linked GeCo child error: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1
    finally:
        if server is not None:
            with suppress(Exception):
                server.close()
        if project is not None:
            with suppress(Exception):
                project.close()


def main() -> int:
    if sys.argv[1:] != ["--child"]:
        print("GeCo process launcher accepts --child only.", file=sys.stderr)
        return 2
    return _child_main()


if __name__ == "__main__":
    raise SystemExit(main())
