"""Launch a persisted TeAL-linked GeCo workspace in a verified child Python.

This is a deliberately narrow local server protocol, not an operation executor.
All GeCo objects and provider callbacks remain inside the selected interpreter.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import ProxyHandler, build_opener

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project

_PROTOCOL = 1
_STARTUP_TIMEOUT = 180.0
_MAX_MESSAGE = 2_000_000


def _wire(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _recv_line(connection: socket.socket, *, limit: int = _MAX_MESSAGE) -> dict[str, Any]:
    raw = bytearray()
    while b"\n" not in raw:
        block = connection.recv(min(65536, limit + 1 - len(raw)))
        if not block:
            raise ConnectionError("GeCo child closed its control channel.")
        raw.extend(block)
        if len(raw) > limit:
            raise ValueError("Oversized GeCo control message.")
    parsed = json.loads(bytes(raw).split(b"\n", 1)[0])
    if not isinstance(parsed, dict):
        raise ValueError("GeCo control message must be an object.")
    return parsed


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _tail(log: Any) -> str:
    with contextlib.suppress(Exception):
        log.flush()
        log.seek(0)
        return log.read()[-3000:]
    return ""


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class GeCoChildServer:
    """Project-owned handle to exactly one verified GeCo child HTTP server."""

    def __init__(
        self, process: subprocess.Popen[Any], channel: socket.socket, log: Any,
        *, request_id: str, token: str, url: str, port: int,
        execution_device: str | None,
    ) -> None:
        self._process = process
        self._channel = channel
        self._log = log
        self._request_id = request_id
        self._token = token
        self._url = url
        self._port = port
        self._execution_device = execution_device
        self._close_requested = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def port(self) -> int:
        return self._port

    @property
    def execution_device(self) -> str | None:
        return self._execution_device

    @property
    def closed(self) -> bool:
        return self._process.poll() is not None

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for actual child termination; expose unexpected failure."""
        try:
            code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        if code != 0 and not self._close_requested:
            raise RuntimeError(
                f"GeCo child exited unexpectedly (exit={code}). {_tail(self._log)}"
            )
        return True

    def close(self) -> None:
        """Stop only this child's GeCo listener and owned interpreter."""
        if self._close_requested:
            return
        self._close_requested = True
        try:
            if not self.closed:
                with contextlib.suppress(OSError, ConnectionError):
                    self._channel.sendall(_wire({
                        "protocol_version": _PROTOCOL, "request_id": self._request_id,
                        "token": self._token, "command": "stop",
                    }))
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _stop_process(self._process)
        finally:
            with contextlib.suppress(OSError):
                self._channel.close()
            with contextlib.suppress(OSError):
                self._log.close()

    def __enter__(self) -> GeCoChildServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _http_ready(url: str, process: subprocess.Popen[Any]) -> None:
    # Bypass notebook proxy settings when probing the local GeCo HTTP listener.
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("GeCo child exited before HTTP readiness.")
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(f"HTTP status {response.status}")
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"GeCo did not establish HTTP readiness at {url}: {last_error}")


def launch_child(
    project: Project, name: str, *, environment: str, mode: str,
    host: str, port: int | None, device: str | None,
    warmup: bool, debug: bool,
) -> GeCoChildServer:
    """Launch with a separately verified installed TeAL/GeCo interpreter."""
    from text_analysis_lab.core.secondary_coordinator import (
        _check_probe, _run,
    )

    executable = project.environments.bound_interpreter(environment)
    # R3 requires the child's own installation; an editable parent PYTHONPATH
    # must never satisfy or override GeCo/TeAL imports in this subprocess.
    code, probe, stderr = _run(executable, "--probe", isolated=True, timeout=60)
    if code != 0 or probe is None:
        raise RuntimeError(
            "Selected interpreter cannot load an isolated, compatible TeAL "
            f"installation ({executable}; exit={code}). {stderr}"
        )
    _check_probe(project, environment, probe)

    request_id = uuid.uuid4().hex
    token = uuid.uuid4().hex + uuid.uuid4().hex
    request = {
        "protocol_version": _PROTOCOL, "request_id": request_id, "token": token,
        "project_path": str(project.path.resolve()), "name": name,
        "environment": environment, "mode": mode, "host": host,
        "port": port, "device": device, "warmup": warmup, "debug": debug,
        "source_sha256": _source_digest(),
    }
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    process: subprocess.Popen[Any] | None = None
    channel: socket.socket | None = None
    log: Any = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.5)
        control_port = listener.getsockname()[1]
        request["control_port"] = control_port
        log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        child_env = {
            key: value for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
        }
        process = subprocess.Popen(
            [str(executable), "-I", "-m", "text_analysis_lab.integrations.geco_launch", "--child"],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            text=True, env=child_env,
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.close()
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"GeCo child exited during startup (exit={process.returncode}). "
                    f"{_tail(log)}"
                )
            try:
                candidate, peer = listener.accept()
            except socket.timeout:
                continue
            candidate.settimeout(3)
            try:
                if peer[0] != "127.0.0.1":
                    raise ValueError("Unexpected GeCo control peer.")
                message = _recv_line(candidate)
                if (
                    message.get("protocol_version") != _PROTOCOL
                    or message.get("token") != token
                    or message.get("request_id") != request_id
                    or message.get("pid") != process.pid
                ):
                    raise ValueError("Uncorrelated GeCo child readiness response.")
            except (ValueError, OSError, ConnectionError, json.JSONDecodeError):
                candidate.close()
                continue
            channel = candidate
            if message.get("status") == "error":
                raise RuntimeError(
                    f"GeCo child could not launch {name!r}: {message.get('error')}"
                )
            if message.get("status") != "ready":
                raise RuntimeError("GeCo child sent an unexpected startup status.")
            required = {
                "protocol_version", "request_id", "token", "pid", "status",
                "url", "port", "execution_device", "mode", "probe",
            }
            if set(message) != required or message["mode"] != mode:
                raise RuntimeError("GeCo child returned malformed readiness metadata.")
            if (
                type(message["port"]) is not int
                or not 0 < message["port"] <= 65535
                or not isinstance(message["url"], str)
                or not isinstance(message["execution_device"], (str, type(None)))
            ):
                raise RuntimeError("GeCo child returned invalid HTTP/device information.")
            from urllib.parse import urlsplit

            parsed = urlsplit(message["url"])
            if parsed.scheme != "http" or parsed.port != message["port"] or not parsed.hostname:
                raise RuntimeError("GeCo child returned an invalid server URL.")
            _check_probe(project, environment, message["probe"])
            _http_ready(message["url"], process)
            server = GeCoChildServer(
                process, channel, log, request_id=request_id, token=token,
                url=message["url"], port=message["port"],
                execution_device=message["execution_device"],
            )
            channel = None
            process = None
            log = None
            return server
        raise TimeoutError(f"Timed out launching linked GeCo workspace {name!r}.")
    except Exception as exc:
        details = _tail(log) if log is not None else ""
        raise RuntimeError(f"Unable to launch linked GeCo workspace {name!r}: {exc}. {details}") from exc
    finally:
        listener.close()
        if channel is not None:
            with contextlib.suppress(OSError):
                channel.close()
        if process is not None:
            _stop_process(process)
        if log is not None:
            log.close()


def _child_main() -> int:
    """Executed ONLY by the child interpreter, before importing GeCo."""
    try:
        request = json.loads(sys.stdin.readline())
        required = {
            "protocol_version", "request_id", "token", "project_path",
            "name", "environment", "mode", "host", "port", "device",
            "warmup", "debug", "source_sha256", "control_port",
        }
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError("Invalid GeCo child launch request.")
        if (
            request["protocol_version"] != _PROTOCOL
            or not isinstance(request["request_id"], str)
            or not isinstance(request["token"], str)
            or not isinstance(request["project_path"], str)
            or not isinstance(request["name"], str)
            or not isinstance(request["environment"], str)
            or request["mode"] not in {"explore", "focus_coder"}
            or not isinstance(request["host"], str)
            or type(request["control_port"]) is not int
            or type(request["warmup"]) is not bool
            or type(request["debug"]) is not bool
            or request["source_sha256"] != _source_digest()
        ):
            raise ValueError("Incompatible or malformed GeCo launch protocol/source.")
    except Exception as exc:
        print(f"GeCo child request error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    channel: socket.socket | None = None
    server: Any = None
    try:
        channel = socket.create_connection(("127.0.0.1", request["control_port"]), timeout=15)
        channel.settimeout(None)
        from text_analysis_lab.core.project import Project
        from text_analysis_lab.core.secondary_coordinator import _check_probe, _probe_payload
        with Project.open(request["project_path"]) as project:
            probe = _probe_payload()
            _check_probe(project, request["environment"], probe)
            linked = project.geco.open(request["name"])
            if linked.mode != request["mode"]:
                raise ValueError(
                    f"Linked workspace {request['name']!r} is {linked.mode!r}, "
                    f"not {request['mode']!r}."
                )
            if linked.is_focus_coder:
                if not callable(getattr(linked.coder, "launch_focus_coder", None)):
                    raise RuntimeError(
                        "This GeCo installation lacks the supported finite-task "
                        "Focus Coding launcher; update geometric-coder."
                    )
                server = linked.launch_focus(
                    host=request["host"], port=request["port"],
                    debug=request["debug"], use_reloader=False,
                )
            else:
                server = linked.launch(
                    host=request["host"], port=request["port"],
                    device=request["device"], warmup=request["warmup"],
                    debug=request["debug"], use_reloader=False,
                )
            if not all(hasattr(server, key) for key in (
                "url", "port", "execution_device", "closed", "close", "wait",
            )):
                raise RuntimeError("Installed GeCo launcher lacks the required server lifecycle.")
            channel.sendall(_wire({
                "protocol_version": _PROTOCOL, "request_id": request["request_id"],
                "token": request["token"], "pid": os.getpid(), "status": "ready",
                "mode": linked.mode, "url": server.url, "port": server.port,
                "execution_device": server.execution_device, "probe": probe,
            }))
            channel.settimeout(0.25)
            pending = bytearray()
            while not server.closed:
                try:
                    data = channel.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break  # owning notebook has exited
                pending.extend(data)
                if len(pending) > 4096:
                    raise ValueError("Oversized GeCo stop message.")
                if b"\n" not in pending:
                    continue
                stop = json.loads(bytes(pending).split(b"\n", 1)[0])
                if stop != {
                    "protocol_version": _PROTOCOL, "request_id": request["request_id"],
                    "token": request["token"], "command": "stop",
                }:
                    raise ValueError("Invalid GeCo stop request.")
                break
            server.close()
            linked.close()
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        if channel is not None and server is None:
            with contextlib.suppress(OSError):
                channel.sendall(_wire({
                    "protocol_version": _PROTOCOL, "request_id": request["request_id"],
                    "token": request["token"], "pid": os.getpid(),
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                }))
        return 1
    finally:
        if server is not None:
            with contextlib.suppress(Exception):
                server.close()
        if channel is not None:
            channel.close()


if __name__ == "__main__":
    raise SystemExit(_child_main() if sys.argv[1:] == ["--child"] else 2)
