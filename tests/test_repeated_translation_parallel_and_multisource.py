"""Gate D: repeated parallel barriers and unmodified mixed source plan replay."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.core.operator import (
    BaseTranslator, BatchResult, ColumnRequest, OutputSpec, SourceRequest,
)
from text_analysis_lab.core.translate import _ExecutionPlan
from text_analysis_lab.core.writer import create_artifact_writer


def _source(project, name, keys, values):
    writer = create_artifact_writer(
        artifact_type="table", artifact_dir=project.storage.artifact_dir(name),
        artifact_id=name, label=name, lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({"keys": pd.DataFrame({"id": keys}), "data": pd.DataFrame({"value": values})})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=name, artifact_type="table", label=name, lineage_mode="new_key",
        status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(name)


class _MixedProbe(BaseTranslator):
    def __init__(self, *, operator_id=None):
        super().__init__(operator_id=operator_id)
        self.events = []
        self.fit = []

    @property
    def requires_fit(self):
        return True

    @property
    def is_fitted(self):
        return len(self.fit) > 0

    @property
    def supports_fit_translate(self):
        return True

    def supports_resume(self, *, mode, route):
        return mode == "fit_translate" and route == "sequential"

    def translation_passes(self, *, sources, mode, request, input_request):
        return 2

    def output_specs(self, *, sources, request):
        return {
            "first": OutputSpec(artifact_type="table", lineage_mode="preserved_key", basis_labels="left"),
            "second": OutputSpec(artifact_type="table", lineage_mode="preserved_key", basis_labels="left"),
        }

    def input_request(self, *, sources, mode, request):
        spec = ColumnRequest(keys=True, data="value", metadata=False)
        return {
            "left": SourceRequest(artifact_type="table", mode="batches", batch_size=2, columns=spec, form="table"),
            "right": SourceRequest(artifact_type="table", mode="full_artifact", columns=spec, form="table"),
        }

    def translate_batch(self, inputs, *, mode, request):
        labels = tuple(inputs)
        batch = inputs["left"]
        values = tuple(batch.data["value"].tolist())
        self.events.append(("batch", request.pass_index, batch.batch_index, labels, values))
        if request.pass_index == 0:
            self.fit.extend(values)
            return BatchResult()
        assert self.fit == list("abcde")
        frame = batch.data
        payload = {"keys": frame[["id"]].reset_index(drop=True), "data": frame[["value"]].reset_index(drop=True)}
        return BatchResult(outputs={"first": payload, "second": payload})

    def handle_batch_result(self, result, *, batch_index, mode, request):
        self.events.append(("handle", request.pass_index, batch_index))
        return result.outputs

    def complete_translation_pass(self, *, mode, request):
        self.events.append(("barrier", request.pass_index))
        return None

    def finalize_translation(self, *, mode, request):
        self.events.append(("final", request.pass_index))
        return None

    def to_json_state(self):
        # Keep scientific fitted state in the frozen operator, not just a
        # transient operation checkpoint.
        return {"fit": list(self.fit)}

    @classmethod
    def from_json_state(cls, state):
        obj = cls()
        obj.fit = list(state.get("fit", []))
        return obj

    def save_intermediate_state(self, intermediate_dir, *, operator_id, mode, route):
        target = Path(intermediate_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "state.json").write_text(json.dumps({
            "fit": self.fit,
            "events": self.events,
        }), encoding="utf-8")

    @classmethod
    def load_intermediate_state(cls, intermediate_dir, *, operator_id, mode, route):
        data = json.loads(
            (Path(intermediate_dir) / "state.json").read_text(encoding="utf-8")
        )
        obj = cls(operator_id=operator_id)
        obj.fit = list(data["fit"])
        obj.events = [
            ("batch", event[1], event[2], tuple(event[3]), tuple(event[4]))
            if event[0] == "batch" else tuple(event)
            for event in data["events"]
        ]
        return obj


def test_mixed_multisource_plan_repeats_identically_and_writes_two_outputs(tmp_path):
    path = tmp_path / "mixed"
    project = Project.create(path, name="repeat_mixed")
    try:
        left = _source(project, "art_left", list(range(5)), list("abcde"))
        right = _source(project, "art_right", [7, 8], ["x", "y"])
        probe = _MixedProbe()
        outputs = project.translate(probe, {"left": left, "right": right})
        batches = [event for event in probe.events if event[0] == "batch"]
        assert [(e[1], e[2], e[3], e[4]) for e in batches] == [
            (p, i, ("left", "right") if i == 0 else ("left",), vals)
            for p in (0, 1)
            for i, vals in enumerate([("a", "b"), ("c", "d"), ("e",)])
        ]
        assert [e for e in probe.events if e[0] == "barrier"] == [("barrier", 0)]
        assert probe.events[-1] == ("final", 1)
        assert probe.fit == list("abcde")
        assert probe.is_fitted
        assert set(outputs) == {"first", "second"}
        assert all(a.n_rows == 5 for a in outputs.values())
        assert all(a.query(key_columns=False, data_columns="value", form="table")["value"].tolist() == list("abcde") for a in outputs.values())
        operation_id = str(project.list_operations()[-1]["operation_id"])
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            assert plan.total_count() == plan.complete_count() == 3
            assert [list(plan.get_unit(i).sources) for i in range(3)] == [
                ["left", "right"], ["left"], ["left"]
            ]
        finally:
            plan.close()
        ids = [out.artifact_id for out in outputs.values()]
    finally:
        project.close()
    reopened = Project.open(path)
    try:
        assert [reopened.get_artifact(a).n_rows for a in ids] == [5, 5]
    finally:
        reopened.close()


class _ParallelProbe(BaseTranslator):
    def __init__(self, *, fitted=False, operator_id=None):
        super().__init__(operator_id=operator_id)
        self.fitted = fitted
        self.events = []
        self.worker_states = []

    @property
    def requires_fit(self):
        return True

    @property
    def is_fitted(self):
        return self.fitted

    @property
    def supports_fit_translate(self):
        return True

    @property
    def supports_parallel_fit_translate(self):
        return True

    @property
    def supports_parallel_translate(self):
        return True

    def supports_resume(self, *, mode, route):
        return mode in {"fit_translate", "translate"} and route in {"sequential", "parallel"}

    def translation_passes(self, *, sources, mode, request, input_request):
        return 2

    def output_specs(self, *, sources, request):
        return OutputSpec(artifact_type="table", lineage_mode="preserved_key")

    def input_request(self, *, sources, mode, request):
        return SourceRequest(
            artifact_type="table", mode="batches", batch_size=2, form="table",
            columns=ColumnRequest(keys=True, data="value", metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        packet = inputs["source"]
        if request.pass_index == 1:
            assert self.fitted, "Pass 2 received worker created before pass-1 barrier"
        frame = packet.data
        result = BatchResult()
        if request.pass_index == 1:
            result = BatchResult(outputs={"output": {
                "keys": frame[["id"]].reset_index(drop=True),
                "data": frame[["value"]].reset_index(drop=True),
            }})
        return result

    def handle_batch_result(self, result, *, batch_index, mode, request):
        self.events.append(("handle", request.pass_index, batch_index))
        return result.outputs

    def complete_translation_pass(self, *, mode, request):
        assert [e[2] for e in self.events if e[0] == "handle" and e[1] == 0] == [0, 1, 2]
        self.events.append(("barrier", request.pass_index))
        self.fitted = True

    def finalize_translation(self, *, mode, request):
        self.events.append(("final", request.pass_index))

    def make_translate_worker(self, *, mode, request):
        self.worker_states.append((request.pass_index, self.fitted))
        return _ParallelProbe(fitted=self.fitted)

    def to_json_state(self):
        return {"fitted": self.fitted}

    @classmethod
    def from_json_state(cls, state):
        return cls(fitted=state["fitted"])

    def save_intermediate_state(self, intermediate_dir, *, operator_id, mode, route):
        target = Path(intermediate_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "state.json").write_text(json.dumps({
            "fitted": self.fitted, "events": self.events,
            "worker_states": self.worker_states,
        }))

    @classmethod
    def load_intermediate_state(cls, intermediate_dir, *, operator_id, mode, route):
        data = json.loads((Path(intermediate_dir) / "state.json").read_text())
        result = cls(fitted=data["fitted"], operator_id=operator_id)
        result.events = [tuple(e) for e in data["events"]]
        result.worker_states = [tuple(e) for e in data["worker_states"]]
        return result


def test_parallel_replay_creates_fresh_workers_only_after_complete_barrier(tmp_path):
    pytest.importorskip("dask.distributed")
    project = Project.create(tmp_path / "parallel", name="parallel_replay")
    try:
        source = _source(project, "art_parallel", list(range(5)), list("abcde"))
        probe = _ParallelProbe()
        output = project.translate(probe, source, workers=2, max_outstanding_units=2)["output"]
        # The procedure begins unfitted, so the existing fit_translate path
        # does not freeze its scientific state before the pass-one barrier.
        assert probe.is_fitted
        assert output.n_rows == 5
        assert probe.worker_states == [(0, False), (1, True)]
        assert [e for e in probe.events if e[0] == "handle"] == [
            ("handle", p, i) for p in (0, 1) for i in range(3)
        ]
        assert [e for e in probe.events if e[0] == "barrier"] == [("barrier", 0)]
        assert probe.events[-1] == ("final", 1)
        assert output.query(key_columns=False, data_columns="value", form="table")["value"].tolist() == list("abcde")
        operation_id = str(project.list_operations()[-1]["operation_id"])
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            assert plan.total_count() == plan.complete_count() == 3
        finally:
            plan.close()
    finally:
        project.close()
