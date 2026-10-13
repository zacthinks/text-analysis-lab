"""Gate C: deterministic recovery windows for repeated translation plans.

Failure points are injected into real Project execution; each recovery opens a
fresh Project. No pass-specific source plan is regenerated on resume.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.core.translate import _ExecutionPlan
from test_repeated_translation_plan import _ReplayProbe, _source


def _assert_complete(path, operation_id):
    project = Project.open(path)
    try:
        outputs = project.resume_operation(operation_id)
        output = outputs["output"]
        assert output.n_rows == 5
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == list("abcde")
        assert project.get_operation(operation_id)["status"] == "complete"
        return output.artifact_id
    finally:
        project.close()


@pytest.mark.parametrize("point", [
    "before_first_checkpoint",
    "after_last_first_pass_unit",
    "during_boundary_hook",
    "before_boundary_promotion",
    "after_boundary_checkpoint",
    "after_second_pass_unit_checkpoint",
    "after_terminal_checkpoint",
])
def test_repeated_translation_recovers_each_checkpoint_window(
    tmp_path, monkeypatch, point
):
    import text_analysis_lab.core.translate as runner

    path = tmp_path / "recovery"
    project = Project.create(path, name="recovery")
    source = _source(project)
    probe = _ReplayProbe(2)
    original_save = runner._save_intermediate_state
    original_mark = _ExecutionPlan.mark_complete
    original_activate = _ExecutionPlan.activate_pass
    original_boundary = _ReplayProbe.complete_translation_pass
    original_complete = runner._complete_runtime
    triggered = {"value": False}

    def fail_once():
        if not triggered["value"]:
            triggered["value"] = True
            raise RuntimeError("crash-injection")

    def save(*args, **kwargs):
        if point == "before_first_checkpoint" and (
            kwargs.get("pass_index") == 0
            and kwargs.get("completed_unit_index") == 0
        ):
            fail_once()
        if point == "before_boundary_promotion" and (
            kwargs.get("pass_index") == 1
            and kwargs.get("completed_unit_index") is None
        ):
            fail_once()
        return original_save(*args, **kwargs)

    def mark(self, unit_index):
        if (
            point == "after_second_pass_unit_checkpoint"
            and self.operation_dir == project.storage.operation_dir(
                str(project.list_operations()[-1]["operation_id"])
            )
            and unit_index == 0
            and self.con.execute(
                "SELECT current_pass FROM plan_pass_state WHERE id=1"
            ).fetchone()[0] == 1
        ):
            fail_once()
        return original_mark(self, unit_index)

    def activate(self, pass_index):
        if point == "after_boundary_checkpoint" and pass_index == 1:
            fail_once()
        return original_activate(self, pass_index)

    def boundary(self, *, mode, request):
        if point == "during_boundary_hook":
            fail_once()
        return original_boundary(self, mode=mode, request=request)

    def complete(proj, runtime):
        if point == "after_terminal_checkpoint":
            fail_once()
        return original_complete(proj, runtime)

    if point in {"before_first_checkpoint", "before_boundary_promotion"}:
        monkeypatch.setattr(runner, "_save_intermediate_state", save)
    elif point == "after_last_first_pass_unit":
        # Last pass-1 unit has checkpointed its outputs, but the SQL status
        # transition is interrupted before the nonterminal hook can run.
        def mark_last(self, unit_index):
            if unit_index == 2:
                fail_once()
            return original_mark(self, unit_index)
        monkeypatch.setattr(_ExecutionPlan, "mark_complete", mark_last)
    elif point == "after_second_pass_unit_checkpoint":
        monkeypatch.setattr(_ExecutionPlan, "mark_complete", mark)
    elif point == "after_boundary_checkpoint":
        monkeypatch.setattr(_ExecutionPlan, "activate_pass", activate)
    elif point == "during_boundary_hook":
        monkeypatch.setattr(_ReplayProbe, "complete_translation_pass", boundary)
    elif point == "after_terminal_checkpoint":
        monkeypatch.setattr(runner, "_complete_runtime", complete)

    try:
        with pytest.raises(RuntimeError, match="crash-injection"):
            project.translate(probe, source)
        operation_id = str(project.list_operations()[-1]["operation_id"])
        assert triggered["value"]
        # The saved materialization plan remains present and has three units.
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            assert plan.total_count() == 3
            assert [plan.get_unit(i).sources["source"].start_position
                    for i in range(3)] == [0, 2, 4]
        finally:
            plan.close()
    finally:
        project.close()

    # Remove the injected failure before reopening in a separate Project.
    monkeypatch.undo()
    _assert_complete(path, operation_id)


def test_repeat_recovery_preserves_checkpointed_terminal_callback_once(
    tmp_path, monkeypatch
):
    import text_analysis_lab.core.translate as runner

    path = tmp_path / "terminal"
    project = Project.create(path, name="terminal")
    source = _source(project)

    original_complete = runner._complete_runtime
    marker = {"attempts": 0}

    def fail_after_terminal(proj, runtime):
        marker["attempts"] += 1
        if marker["attempts"] == 1:
            raise RuntimeError("after terminal callback")
        return original_complete(proj, runtime)

    monkeypatch.setattr(runner, "_complete_runtime", fail_after_terminal)
    try:
        with pytest.raises(RuntimeError, match="after terminal callback"):
            project.translate(_ReplayProbe(2), source)
        operation_id = str(project.list_operations()[-1]["operation_id"])
        temp = project.storage.operation_temp_dir(operation_id)
        assert json.loads((temp / "pass.json").read_text())["terminal_finalized"] is True
        saved = list(temp.rglob("probe.json"))
        assert len(saved) == 1
        state = json.loads(saved[0].read_text())
        assert len([e for e in state["events"] if e[0] == "final"]) == 1
    finally:
        project.close()
    monkeypatch.undo()
    _assert_complete(path, operation_id)


def test_checkpointed_pass_one_unit_does_not_repeat_fit_contributions(
    tmp_path, monkeypatch
):
    import text_analysis_lab.core.translate as runner

    path = tmp_path / "partial"
    project = Project.create(path, name="partial")
    source = _source(project)
    original_mark = _ExecutionPlan.mark_complete
    fired = {"value": False}

    def fail_mark(self, unit_index):
        if unit_index == 0 and not fired["value"]:
            fired["value"] = True
            raise RuntimeError("after first unit checkpoint")
        return original_mark(self, unit_index)

    monkeypatch.setattr(_ExecutionPlan, "mark_complete", fail_mark)
    try:
        with pytest.raises(RuntimeError, match="after first unit checkpoint"):
            project.translate(_ReplayProbe(2), source)
        operation_id = str(project.list_operations()[-1]["operation_id"])
        checkpoint = project.storage.operation_temp_dir(operation_id)
        state_paths = list(checkpoint.rglob("probe.json"))
        assert len(state_paths) == 1
        state = json.loads(state_paths[0].read_text())
        assert state["seen"]["0"] == [["a", "b"]]
    finally:
        project.close()
    monkeypatch.undo()
    _assert_complete(path, operation_id)
