"""Opt-in actual separate-installed-interpreter TeAL-linked GeCo acceptance.

Requires uv and network/package caches to install TeAL[geco] in an independent
venv. This is deliberately not a mock GeCo server or standalone CLI launch.
Run on Ubuntu/Windows with TEAL_REAL_GECO_R3=1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener
from pathlib import Path

import pandas as pd
import pytest

from text_analysis_lab import Project
from text_analysis_lab.translators import CountVectorizer

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.skipif(
        os.environ.get("TEAL_REAL_GECO_R3") != "1",
        reason="Set TEAL_REAL_GECO_R3=1 for independent GeCo install/launch.",
    ),
]


def _run(argv: list[str], *, timeout: int = 900, env=None) -> str:
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, env=env,
    )
    assert result.returncode == 0, (
        f"Command failed: {argv!r}\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}"
    )
    return result.stdout


@pytest.fixture(scope="module")
def second_python(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3-installed-python")
    venv = root / "child"
    _run(["uv", "venv", str(venv), "--python", sys.executable])
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    checkout = Path(__file__).resolve().parents[1]
    _run(
        ["uv", "pip", "install", "--python", str(python), "-e", f"{checkout}[geco]"],
        timeout=1200,
    )
    # A deliberate hostile parent PYTHONPATH cannot leak into the isolated child.
    _run(
        [str(python), "-I", "-c",
         "import text_analysis_lab, geometric_coder; "
         "print(text_analysis_lab.__file__, geometric_coder.__file__)"],
    )
    return python


def _register_second(project_path: Path, executable: Path) -> str:
    script = (
        "import sys; from text_analysis_lab import Project; "
        "p=Project.open(sys.argv[1]); print(p.environments.opening_id); p.close()"
    )
    return _run(
        [str(executable), "-I", "-c", script, str(project_path)],
    ).strip().splitlines()[-1]


def test_actual_child_exploratory_geometry_and_server(tmp_path, second_python, monkeypatch):
    root = tmp_path / "teal-project"
    with Project.create(root, "r3-real-geco") as project:
        csv_path = tmp_path / "docs.csv"
        pd.DataFrame({
            "text": ["a cat runs", "a dog jumps", "birds swim", "fish fly"],
        }).to_csv(csv_path, index=False)
        documents = project.read_csv(
            csv_path, text_fields="text", metadata_fields=None,
        )
        count = project.translate(
            CountVectorizer(text_field="text", min_df=1), documents,
        )["output"]
        linked = project.geco.create(
            "coding", documents=documents, text_field="text", geometry=count,
        )
        geometry = linked.coder.geometries()[0]
        geometry_name = geometry["name"]
        geometry_id = int(geometry["geometry_id"])
        linked.coder.saved_semantic_searches.register("cat runs")
        linked.close()
        default_id = project.environments.default["environment_id"]
        child_id = _register_second(root, second_python)
        assert child_id != default_id
        project.environments.bind_interpreter(child_id, second_python)
        project.environments.set_default(child_id)

        # Verify the actual provider callback using the installed child TeAL
        # and child GeCo, not an object serialized from the parent.
        probe = (
            "import sys, json; from text_analysis_lab import Project; "
            "p=Project.open(sys.argv[1]); w=p.geco.open('coding'); "
            "g=w.coder.geometry_matrix(sys.argv[2]); "
            "print(json.dumps({'shape': list(g.shape), "
            "'provider': type(w.external_provider).__name__})); "
            "p.close()"
        )
        result = _run([
            str(second_python), "-I", "-c", probe,
            str(root), geometry_name,
        ]).strip().splitlines()[-1]
        assert json.loads(result)["shape"][0] == 4
        assert json.loads(result)["provider"] == "TeALGeCoProvider"

        hostile = tmp_path / "hostile"
        (hostile / "text_analysis_lab").mkdir(parents=True)
        (hostile / "text_analysis_lab" / "__init__.py").write_text(
            "raise RuntimeError('hostile parent PYTHONPATH')",
            encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(hostile))
        # Readiness is local even if the notebook routes all HTTP through
        # an unreachable proxy and explicitly disables proxy bypass.
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        server = project.geco.launch("coding", device="cpu", warmup=False)
        try:
            # The opened linked handle must use the same verified child route.
            reopened = project.geco.open("coding")
            direct = reopened.launch(
                environment=child_id, device="cpu", warmup=False,
            )
            try:
                assert direct.port > 0 and direct.url.startswith("http://")
                assert direct.execution_device == "cpu" and not direct.closed
                assert direct.port != server.port
            finally:
                direct.close()
            assert direct.wait(timeout=10)

            # Native selection remains available even when the active default
            # points at the child interpreter.
            native = reopened.launch(
                environment=default_id, device="cpu", warmup=False,
            )
            try:
                assert native.execution_device == "cpu"
                assert native.port > 0 and native.url.startswith("http://")
                assert not native.closed
            finally:
                native.close()
            assert native.wait(timeout=10)
            assert server.port > 0
            assert server.url.startswith("http://")
            assert server.execution_device == "cpu"
            assert not server.closed
            assert project.environments.default["environment_id"] == child_id

            # Trigger GeCo's actual semantic-search Dash callback *over HTTP*.
            # This loads the external TeAL geometry and executes the linked
            # frozen text-to-vector provider in the running server child.
            callback = {
                "output": "semantic-result-candidate.data",
                "outputs": {"id": "semantic-result-candidate", "property": "data"},
                "inputs": [
                    {"id": "semantic-active-query", "property": "data",
                     "value": "cat runs"},
                    {"id": "geometry-dropdown", "property": "value",
                     "value": geometry_id},
                ],
                "state": [],
                "changedPropIds": ["semantic-active-query.data"],
            }
            request = Request(
                server.url.rstrip("/") + "/_dash-update-component",
                data=json.dumps(callback).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
                assert response.status == 200
                payload = json.loads(response.read())
            candidate = payload["response"]["semantic-result-candidate"]["data"]
            assert candidate["query"] == "cat runs"
            assert candidate["geometry_id"] == geometry_id
            assert candidate["status"] == ""
            assert candidate["token"]
            assert len(candidate["bounds"]) == 2

            # One strict port collision must fail without killing the first
            # live child or the parent interpreter.
            other = project.geco.launch(
                "coding", environment=child_id, device="auto", warmup=False,
            )
            try:
                assert other.execution_device in {"cpu", "cuda:0", "mps"}
                assert other.port != server.port
                assert not other.closed and not server.closed
            finally:
                other.close()
            assert other.wait(timeout=10)

            with pytest.raises(Exception, match="port|use|start|launch"):
                project.geco.launch(
                    "coding", environment=child_id, port=server.port,
                    device="cpu", warmup=False,
                )
            assert not server.closed

            # "localhost" can resolve to IPv6 first, while Werkzeug binds
            # IPv4. The preflight must reject the occupied actual IPv4 port.
            with pytest.raises(Exception, match="port|use|start|launch"):
                project.geco.launch(
                    "coding", environment=child_id, host="localhost",
                    port=server.port, device="cpu", warmup=False,
                )
            assert not server.closed

            # This cannot be a valid CUDA index on the runner, even when the
            # host has an accelerator. No CPU fallback is permitted.
            with pytest.raises(Exception, match="cuda|device|CUDA|launch"):
                project.geco.launch(
                    "coding", environment=child_id,
                    device="cuda:9999", warmup=False,
                )
            assert not server.closed
        finally:
            server.close()
        assert server.wait(timeout=10)


def test_actual_child_focus_finite_task(tmp_path, second_python):
    root = tmp_path / "focus"
    with Project.create(root, "r3-real-focus") as project:
        csv_path = tmp_path / "docs.csv"
        pd.DataFrame({"text": ["one", "two", "three"]}).to_csv(csv_path, index=False)
        documents = project.read_csv(
            csv_path, text_fields="text", metadata_fields=None,
        )
        linked = project.geco.create_focus(
            "audit", documents=documents, text_source=documents,
            text_field="text", codes={"a": "A", "b": "B"},
            allow_unsure=False,
        )
        session = linked.manifest["focus_session_id"]
        linked.close()
        child_id = _register_second(root, second_python)
        project.environments.bind_interpreter(child_id, second_python)
        server = project.geco.launch_focus("audit", environment=child_id)
        try:
            reopened = project.geco.open("audit")
            direct = reopened.launch_focus(environment=child_id)
            try:
                assert direct.execution_device is None
                assert direct.port > 0 and not direct.closed
                assert direct.port != server.port
            finally:
                direct.close()
            assert direct.wait(timeout=10)
            assert server.execution_device is None
            assert server.port > 0
            assert not server.closed
        finally:
            server.close()
        assert server.wait(timeout=10)
        reopened = project.geco.open("audit")
        assert reopened.manifest["focus_session_id"] == session
        assert reopened.coder.focus_task(session)["session_id"] == session
