from __future__ import annotations

import pytest

from text_analysis_lab import cli


def test_gui_cli_waits_for_project_center_and_closes_project(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    calls: list[str] = []

    class FakeCenter:
        def wait(self, timeout=None):
            assert timeout is None
            calls.append("wait")
            return True

        def close(self):
            calls.append("center_close")

    class FakeProject:
        def launch_project_center(self):
            calls.append("launch")
            return FakeCenter()

        def close(self):
            calls.append("project_close")

    monkeypatch.setattr(
        cli.Project,
        "open",
        classmethod(lambda cls, path: FakeProject()),
    )

    assert cli.main(["gui", str(project_path)]) == 0
    assert calls == ["launch", "wait", "project_close"]


def test_gui_cli_ctrl_c_closes_center_and_project(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    calls: list[str] = []

    class FakeCenter:
        def wait(self, timeout=None):
            calls.append("wait")
            raise KeyboardInterrupt

        def close(self):
            calls.append("center_close")

    class FakeProject:
        def launch_project_center(self):
            calls.append("launch")
            return FakeCenter()

        def close(self):
            calls.append("project_close")

    monkeypatch.setattr(
        cli.Project,
        "open",
        classmethod(lambda cls, path: FakeProject()),
    )

    assert cli.main(["gui", str(project_path)]) == 0
    assert calls == ["launch", "wait", "center_close", "project_close"]


def _patch_linked_geco_project(monkeypatch, names, *, focus=False, interrupt=False):
    calls: list[str] = []

    class FakeServer:
        def wait(self):
            calls.append("wait")
            if interrupt:
                raise KeyboardInterrupt

        def close(self):
            calls.append("server_close")

    class FakeWorkspace:
        is_focus_coder = focus

        def launch(self):
            calls.append("launch")
            return FakeServer()

        def launch_focus(self):
            calls.append("launch_focus")
            return FakeServer()

    class FakeGeCo:
        def list(self):
            return [{"name": name} for name in names]

        def open(self, name):
            calls.append(f"open:{name}")
            return FakeWorkspace()

    class FakeProject:
        def __init__(self):
            self.geco = FakeGeCo()

        def close(self):
            calls.append("project_close")

    monkeypatch.setattr(
        cli.Project, "open", classmethod(lambda cls, path: FakeProject())
    )
    return calls


@pytest.mark.parametrize(
    ("names", "selected", "focus", "interrupt", "expected_name"),
    [
        (["only"], None, False, False, "only"),
        (["first", "second"], "second", False, False, "second"),
        (["focus"], None, True, False, "focus"),
        (["first", "focus"], "focus", True, True, "focus"),
    ],
)
def test_geco_cli_launches_linked_workspace(
    tmp_path, monkeypatch, names, selected, focus, interrupt, expected_name
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    calls = _patch_linked_geco_project(
        monkeypatch, names, focus=focus, interrupt=interrupt
    )

    argv = ["geco", str(project_path)]
    if selected is not None:
        argv.append(selected)

    assert cli.main(argv) == 0
    launch = "launch_focus" if focus else "launch"
    assert calls == [
        f"open:{expected_name}", launch, "wait", "server_close", "project_close"
    ]


@pytest.mark.parametrize(
    ("names", "selected", "message"),
    [
        ([], None, "No linked GeCo workspaces"),
        (["first", "second"], None, "Specify a name: first, second"),
        (["first"], "wrong", "Unknown linked GeCo workspace 'wrong'"),
    ],
)
def test_geco_cli_reports_selection_errors_and_closes_project(
    tmp_path, monkeypatch, capsys, names, selected, message
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    calls = _patch_linked_geco_project(monkeypatch, names)
    argv = ["geco", str(project_path)]
    if selected is not None:
        argv.append(selected)

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert calls == ["project_close"]
