from __future__ import annotations

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
