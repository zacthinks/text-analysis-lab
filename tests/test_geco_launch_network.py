"""Network regressions for R3 linked-GeCo readiness and strict port checks."""
from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from text_analysis_lab.integrations.geco_process import (
    GeCoLaunchError,
    _http_ready,
    assert_port_available,
)


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ready")

    def log_message(self, *args: object) -> None:
        pass


def test_local_http_readiness_ignores_unreachable_proxy(monkeypatch):
    with ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("no_proxy", "")
            monkeypatch.setenv("NO_PROXY", "")
            assert _http_ready(f"http://127.0.0.1:{server.server_port}/")
        finally:
            server.shutdown()
            worker.join(timeout=5)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_explicit_occupied_ipv4_port_is_rejected(host):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            occupied.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        with pytest.raises(GeCoLaunchError, match="port unavailable or in use"):
            assert_port_available(host, occupied.getsockname()[1])


def test_localhost_preflight_uses_werkzeug_ipv4_address_family(monkeypatch):
    original = socket.getaddrinfo
    calls = []

    def spy(host, port, family=0, type=0, proto=0, flags=0):
        calls.append(family)
        return original(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", spy)
    assert_port_available("localhost", 0)  # Automatic port skips the preflight.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        with pytest.raises(GeCoLaunchError, match="port unavailable or in use"):
            assert_port_available("localhost", occupied.getsockname()[1])
    assert calls and all(family == socket.AF_INET for family in calls)
