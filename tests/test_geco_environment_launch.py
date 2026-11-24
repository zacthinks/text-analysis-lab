"""R3 name-based routing and owned child-server lifecycle tests.

Pure routing tests deliberately do not import geometric_coder.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from text_analysis_lab import Project
from text_analysis_lab.integrations.geco_process import (
    ChildGeCoServer,
    GeCoLaunchError,
)


class _Server:
    def __init__(self):
        self.closed = False
        self.url = "http://127.0.0.1:4000/"
        self.port = 4000
        self.execution_device = "cpu"

    def close(self):
        self.closed = True


@pytest.fixture
def routing(tmp_path, monkeypatch):
    project = Project.create(tmp_path / "teal", "r3-routing")
    manager = project.geco
    manifests = {
        "coding": {"mode": "explore", "workspace_path": "workspaces/coding.geco"},
        "audit": {"mode": "focus_coder", "workspace_path": "workspaces/audit.geco"},
    }
    monkeypatch.setattr(manager, "_read_manifest", lambda name: manifests[name])
    monkeypatch.setattr(
        manager, "_workspace_path_from_manifest",
        lambda manifest: tmp_path / manifest["workspace_path"],
    )
    active = ["child-A"]
    monkeypatch.setattr(
        project.environments, "resolve",
        lambda env=None: active[0] if env is None else env,
    )
    monkeypatch.setattr(
        project.environments, "is_current_interpreter",
        lambda env: env == "local",
    )
    calls = []
    servers = []

    def child_launch(*args, **kwargs):
        server = _Server()
        calls.append(kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(
        "text_analysis_lab.integrations.geco_process.launch", child_launch,
    )
    try:
        yield project, manager, active, calls, servers, monkeypatch
    finally:
        project.close()


def test_default_override_and_change_affect_new_launch_only(routing):
    project, manager, active, calls, servers, _ = routing
    first = manager.launch("coding", device="auto", warmup=False)
    assert calls[-1]["environment"] == "child-A"
    assert calls[-1]["device"] == "auto"
    second = manager.launch("coding", environment="child-B", device="cpu")
    assert calls[-1]["environment"] == "child-B"
    assert active == ["child-A"]
    active[0] = "child-C"
    third = manager.launch("coding")
    assert calls[-1]["environment"] == "child-C"
    assert [s.closed for s in (first, second, third)] == [False] * 3
    project.close()
    assert all(server.closed for server in servers)


def test_focus_uses_distinct_launch_contract(routing):
    _, manager, _, calls, _, _ = routing
    manager.launch_focus("audit", environment="child-B")
    assert calls[-1]["focus"] is True
    assert calls[-1]["device"] is None
    assert calls[-1]["warmup"] is False
    with pytest.raises(Exception, match="launch_focus"):
        manager.launch("audit")
    with pytest.raises(Exception, match="launch"):
        manager.launch_focus("coding")


def test_child_launch_does_not_import_geco_in_parent(routing, monkeypatch):
    import builtins

    _, manager, _, calls, _, _ = routing
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "geometric_coder" or name.startswith("geometric_coder."):
            raise AssertionError("Parent must not import GeCo for child launch.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    manager.launch("coding", environment="child-B")
    assert calls[-1]["environment"] == "child-B"


def test_child_unbound_selector_failure_is_actionable(routing, monkeypatch):
    from text_analysis_lab.core.environments import EnvironmentRegistryError

    _, manager, _, _, _, _ = routing

    def reject(*args, **kwargs):
        raise EnvironmentRegistryError("no bound Python interpreter")

    monkeypatch.setattr(
        "text_analysis_lab.integrations.geco_process.launch", reject,
    )
    with pytest.raises(EnvironmentRegistryError, match="bound Python"):
        manager.launch("coding", environment="unbound")


def test_current_interpreter_preserves_native_server(routing):
    _, manager, _, _, _, monkeypatch = routing
    expected = _Server()
    native_calls = []
    linked = SimpleNamespace(
        _launch_native=lambda **kwargs: (native_calls.append(kwargs), expected)[1],
        _launch_focus_native=lambda **kwargs: expected,
    )
    monkeypatch.setattr(manager, "open", lambda name: linked)
    assert manager.launch("coding", environment="local", device="cpu") is expected
    assert native_calls[0]["device"] == "cpu"
    assert native_calls[0]["use_reloader"] is False
    assert manager.launch_focus("audit", environment="local") is expected


def _child_process(code: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _handle(proc: subprocess.Popen[str]) -> ChildGeCoServer:
    temp = tempfile.TemporaryDirectory(prefix="test-geco-handle-")
    log_path = Path(temp.name) / "log.txt"
    log_path.write_text("", encoding="utf-8")
    return ChildGeCoServer(
        proc, temporary=temp, log_path=log_path,
        url="http://127.0.0.1:12345", port=12345, device="cpu",
    )


def test_child_handle_stops_only_its_own_process():
    first = _child_process("import sys; sys.stdin.readline()")
    second = _child_process("import sys; sys.stdin.readline()")
    a = _handle(first)
    b = _handle(second)
    try:
        assert a.port == 12345 and a.execution_device == "cpu"
        assert not a.closed and not b.closed
        a.close()
        assert a.wait(timeout=1)
        assert a.closed and not b.closed
    finally:
        a.close()
        b.close()
    assert b.closed



def test_windows_log_cleanup_error_does_not_mask_normal_close(monkeypatch):
    process = _child_process("import sys; sys.stdin.readline()")
    handle = _handle(process)
    original_cleanup = handle._temporary.cleanup

    def locked():
        raise PermissionError("Windows child.log is temporarily locked")

    monkeypatch.setattr(handle._temporary, "cleanup", locked)
    try:
        handle.close()
        assert handle.wait(timeout=1)
        assert handle.closed
    finally:
        original_cleanup()


def test_child_handle_detects_abrupt_exit():
    process = _child_process("import sys; sys.exit(7)")
    handle = _handle(process)
    try:
        with pytest.raises(GeCoLaunchError, match="unexpectedly"):
            handle.wait(timeout=10)
        assert handle.closed
    finally:
        handle.close()



def _linked(manager, name, mode, coder):
    from text_analysis_lab.integrations.geco import LinkedGeCoWorkspace

    return LinkedGeCoWorkspace(
        manager, {"name": name, "mode": mode, "workspace_path": f"workspaces/{name}.geco"},
        coder, None,
    )


def test_linked_local_launch_preserves_native_arguments_and_manager_cleanup(routing):
    project, manager, active, _, _, monkeypatch = routing
    active[0] = "local"
    native_calls = []
    handles = []

    def native(**kwargs):
        server = _Server()
        native_calls.append(kwargs)
        handles.append(server)
        return server

    linked = _linked(manager, "coding", "explore", SimpleNamespace(launch=native))
    monkeypatch.setattr(manager, "open", lambda name: linked)
    first = linked.launch(device="cpu", warmup=False, use_reloader=True)
    assert native_calls[-1] == {
        "device": "cpu", "warmup": False, "use_reloader": True,
    }
    assert first.url.startswith("http://") and first.execution_device == "cpu"
    assert not first.closed
    second = manager.launch("coding", environment="local", device="cpu", warmup=False)
    assert second is handles[-1] and second is not first
    assert native_calls[-1]["use_reloader"] is False
    assert native_calls[-1]["device"] == "cpu"
    project.close()
    assert first.closed and second.closed


def test_linked_child_routes_explicit_default_and_manager_equivalently(routing):
    project, manager, active, calls, servers, monkeypatch = routing
    linked = _linked(
        manager, "coding", "explore",
        SimpleNamespace(launch=lambda **kwargs: pytest.fail("native GeCo was invoked")),
    )
    monkeypatch.setattr(manager, "open", lambda name: linked)
    first = linked.launch(environment="child-B", device="cpu", warmup=False)
    assert calls[-1]["environment"] == "child-B"
    assert calls[-1]["device"] == "cpu"
    assert first.url and first.execution_device == "cpu"
    second = manager.launch("coding", environment="child-B", device="cpu", warmup=False)
    assert calls[-1]["environment"] == "child-B"
    assert calls[-1]["device"] == "cpu"
    active[0] = "child-C"
    third = linked.launch(device="auto", warmup=False)
    assert calls[-1]["environment"] == "child-C"
    assert calls[-1]["device"] == "auto"
    assert all(not server.closed for server in (first, second, third))
    project.close()
    assert all(server.closed for server in servers)


def test_linked_child_forwards_gpu_request_without_claiming_cuda_execution(routing):
    project, manager, _, calls, _, monkeypatch = routing
    linked = _linked(manager, "coding", "explore", object())
    monkeypatch.setattr(manager, "open", lambda name: linked)

    def simulated_child(*args, **kwargs):
        server = _Server()
        server.execution_device = "cuda:0"
        calls.append(kwargs)
        return server

    monkeypatch.setattr(
        "text_analysis_lab.integrations.geco_process.launch", simulated_child,
    )
    server = linked.launch(environment="child-B", device="cuda")
    assert calls[-1]["device"] == "cuda"
    assert server.execution_device == "cuda:0"
    assert server.closed is False
    project.close()
    assert server.closed


def test_linked_focus_routes_environment_and_preserves_native_focus_options(routing):
    project, manager, _, calls, servers, monkeypatch = routing
    native_calls = []

    def native_focus(**kwargs):
        native_calls.append(kwargs)
        return _Server()

    linked = _linked(
        manager, "audit", "focus_coder",
        SimpleNamespace(launch_focus_coder=native_focus),
    )
    monkeypatch.setattr(manager, "open", lambda name: linked)
    linked._manifest["focus_session_id"] = 7
    local = linked.launch_focus(
        environment="local", session_id=7, allow_unsure=False,
        use_reloader=False,
    )
    assert native_calls[-1] == {
        "session_id": 7, "allow_unsure": False, "use_reloader": False,
    }
    child = linked.launch_focus(environment="child-B", use_reloader=False)
    assert calls[-1]["focus"] is True
    assert calls[-1]["warmup"] is False and calls[-1]["device"] is None
    with pytest.raises(Exception, match="native.*options|Cross-interpreter"):
        linked.launch_focus(environment="child-B", session_id=7)
    with pytest.raises(Exception, match="native.*options|Cross-interpreter"):
        linked.launch_focus(environment="child-B", use_reloader=True)
    project.close()
    assert local.closed and child.closed and servers[-1].closed


def test_geco_dependency_metadata_has_no_removed_extras():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["optional-dependencies"]["geco"] == ["geometric-coder"]
    all_dependencies = (
        metadata["project"]["optional-dependencies"]["all"]
        + metadata["dependency-groups"]["dev"]
    )
    assert any("geometric-coder" in item for item in all_dependencies)
    assert all("geometric-coder[" not in item for item in all_dependencies)
    assert not any(
        item == "geometric-coder" for item in metadata["project"]["dependencies"]
    )
