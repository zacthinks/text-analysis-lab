"""Gate B: sequential replay of a single persisted translation plan."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator, BatchResult, ColumnRequest, OutputSpec, SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.translate import _ExecutionPlan
from text_analysis_lab.core.writer import create_artifact_writer


def _source(project):
    rows = pd.DataFrame({
        "id": list(range(5)),
        "value": ["a", "b", "c", "d", "e"],
    })
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_replay"),
        artifact_id="art_replay", label="source",
        lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["id"]], "data": rows[["value"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_replay", artifact_type="table", label="source",
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact("art_replay")


class _ReplayProbe(BaseTranslator):
    def __init__(self, passes=2, *, fail_at=None, signal=None, operator_id=None):
        super().__init__(operator_id=operator_id)
        self.passes = passes
        self.fail_at = fail_at
        self.signal = None if signal is None else str(signal)
        self.events = []
        self.finished = set()
        self.seen = {}

    def supports_resume(self, *, mode, route):
        return mode == "translate" and route == "sequential"

    def translation_passes(self, *, sources, mode, request, input_request):
        return self.passes

    def output_specs(self, *, sources, request):
        return OutputSpec(artifact_type="table", lineage_mode="preserved_key")

    def input_request(self, *, sources, mode, request):
        return SourceRequest(
            artifact_type="table", mode="batches", batch_size=2, form="table",
            columns=ColumnRequest(keys=True, data="value", metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        batch = inputs["source"]
        frame = batch.data
        values = frame["value"].tolist()
        key = (request.pass_index, batch.batch_index)
        if self.fail_at == key and self.signal and not Path(self.signal).exists():
            Path(self.signal).write_text("tripped")
            raise RuntimeError("interrupted replay")
        self.events.append(("batch", request.pass_index, batch.batch_index, values))
        self.seen.setdefault(request.pass_index, []).append(tuple(values))
        if request.pass_index != request.pass_count - 1:
            return BatchResult()
        return BatchResult(outputs={
            "output": {
                "keys": frame[["id"]].reset_index(drop=True),
                "data": frame[["value"]].reset_index(drop=True),
            }
        })

    def handle_batch_result(self, result, *, batch_index, mode, request):
        self.events.append(("handle", request.pass_index, batch_index))
        return result.outputs

    def complete_translation_pass(self, *, mode, request):
        self.events.append(("barrier", request.pass_index))
        self.finished.add(request.pass_index)
        return None

    def finalize_translation(self, *, mode, request):
        self.events.append(("final", request.pass_count))
        return None

    def to_json_state(self):
        return {"passes": self.passes, "fail_at": self.fail_at, "signal": self.signal}

    @classmethod
    def from_json_state(cls, state):
        failure = state.get("fail_at")
        return cls(
            state["passes"],
            fail_at=None if failure is None else tuple(failure),
            signal=state.get("signal"),
        )

    def save_intermediate_state(self, intermediate_dir, *, operator_id, mode, route):
        target = Path(intermediate_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "probe.json").write_text(json.dumps({
            "config": self.to_json_state(), "events": self.events,
            "finished": sorted(self.finished),
            "seen": {str(k): v for k, v in self.seen.items()},
        }))

    @classmethod
    def load_intermediate_state(cls, intermediate_dir, *, operator_id, mode, route):
        state = json.loads((Path(intermediate_dir) / "probe.json").read_text())
        obj = cls.from_json_state(state["config"])
        obj.operator_id = operator_id
        obj.events = [tuple(x) for x in state["events"]]
        obj.finished = set(state["finished"])
        obj.seen = {int(k): [tuple(v) for v in xs] for k, xs in state["seen"].items()}
        return obj


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_replays_saved_plan_and_only_last_pass_writes(tmp_path, passes):
    project = Project.create(tmp_path / "project", name="replay")
    try:
        source = _source(project)
        translator = _ReplayProbe(passes)
        output = project.translate(translator, source)["output"]
        assert output.n_rows == 5
        expected = [("a", "b"), ("c", "d"), ("e",)]
        assert [translator.seen[k] for k in range(passes)] == [expected] * passes
        assert [e for e in translator.events if e[0] == "barrier"] == [
            ("barrier", k) for k in range(passes - 1)
        ]
        assert [e for e in translator.events if e[0] == "final"] == [("final", passes)]
        for k in range(passes):
            assert [e[2] for e in translator.events if e[0] == "handle" and e[1] == k] == [0, 1, 2]
        op = project.list_operations()[-1]
        operation_id = str(op["operation_id"])
        descriptor = json.loads(
            (project.storage.operation_dir(operation_id) / "operation.json").read_text()
        )
        assert descriptor["request"]["pass_count"] == passes
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            assert plan.total_count() == plan.complete_count() == 3
        finally:
            plan.close()
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == list("abcde")
    finally:
        project.close()


def test_resume_second_pass_reuses_saved_assignments_and_fit_checkpoint(tmp_path):
    path = tmp_path / "project"
    project = Project.create(path, name="replay_resume")
    try:
        source = _source(project)
        probe = _ReplayProbe(2, fail_at=(1, 1), signal=tmp_path / "fail_once")
        with pytest.raises(RuntimeError, match="interrupted replay"):
            project.translate(probe, source)
        op_id = str(project.list_operations()[-1]["operation_id"])
        saved = json.loads(
            (project.storage.operation_temp_dir(op_id) / "pass.json").read_text()
        )
        assert saved["pass_index"] == 1
        plan = _ExecutionPlan(project.storage.operation_dir(op_id))
        try:
            assert plan.total_count() == 3
            assert plan.complete_count() == 1
        finally:
            plan.close()
    finally:
        project.close()

    reopened = Project.open(path)
    try:
        output = reopened.resume_operation(op_id)["output"]
        assert output.n_rows == 5
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == list("abcde")
        assert reopened.get_operation(op_id)["status"] == "complete"
    finally:
        reopened.close()


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_invalid_pass_counts_rejected_before_operation_registration(tmp_path, invalid):
    project = Project.create(tmp_path / "project", name="invalid_passes")
    try:
        source = _source(project)
        with pytest.raises(OperatorError, match="pass count"):
            project.translate(_ReplayProbe(invalid), source)
        assert project.list_operations() == []
    finally:
        project.close()


def test_legacy_request_defaults_to_single_pass():
    request = TranslationRequest(2, 5, 3, {"flag": 1})
    assert request.pass_count == 1
    assert request.pass_index == 0
    assert request.workers == 2
    assert request.batch_size == 5
    assert request.max_outstanding_units == 3
    assert request.params == {"flag": 1}
