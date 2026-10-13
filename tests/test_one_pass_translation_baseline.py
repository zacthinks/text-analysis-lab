"""Golden one-pass Project.translate contracts before optional plan replay.

Gate A is characterization only: no repeated-pass API or executor changes.
The assertions deliberately target the existing public packet/callback lifecycle,
durable source-unit assignments, writer output and fitted reuse.
"""
from __future__ import annotations

import json

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


def _source(project, artifact_id: str, frame: pd.DataFrame, keys: list[str]):
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({
        "keys": frame[keys].reset_index(drop=True),
        "data": frame[["value"]].reset_index(drop=True),
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


class _OnePassProbe(BaseTranslator):
    """Trace canonical callbacks while passing source values into real writers."""

    def __init__(self, shape="rows", *, two_outputs=False, operator_id=None):
        super().__init__(operator_id=operator_id)
        self.shape = shape
        self.two_outputs = two_outputs
        self.events = []

    @property
    def supports_parallel_translate(self):
        return True

    def output_specs(self, *, sources, request):
        _ = sources, request
        basis = "left" if self.shape == "mixed" else "source"
        spec = OutputSpec(
            artifact_type="table", lineage_mode="preserved_key",
            basis_labels=basis,
        )
        return {"first": spec, "second": spec} if self.two_outputs else {"output": spec}

    def input_request(self, *, sources, mode, request):
        _ = sources, mode
        column_request = ColumnRequest(keys=True, data="value", metadata=False)
        if self.shape == "mixed":
            return {
                "left": SourceRequest(
                    artifact_type="table", mode="batches",
                    batch_size=request.batch_size or 2, form="table",
                    columns=column_request,
                ),
                "right": SourceRequest(
                    artifact_type="table", mode="full_artifact",
                    form="table", columns=column_request,
                ),
            }
        return SourceRequest(
            artifact_type="table",
            mode="full_artifact" if self.shape == "full" else "batches",
            batch_size=(
                None if self.shape == "full"
                else (request.batch_size or (1 if self.shape == "single" else 2))
            ),
            form="single" if self.shape == "single" else "table",
            columns=column_request,
        )

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        received = {}
        for label, packet in inputs.items():
            data = packet.data
            values = (data["value"],) if isinstance(data, dict) else tuple(data["value"])
            received[label] = (
                packet.batch_index, packet.batch_count,
                packet.is_first, packet.is_last, values,
            )
        self.events.append(("batch", received))
        output_input = inputs.get("left", inputs.get("source"))
        if output_input is None:
            return BatchResult()
        frame = output_input.data
        if isinstance(frame, dict):
            frame = pd.DataFrame([frame])
        payload = {
            "keys": frame[list(output_input.primary_key)].reset_index(drop=True),
            "data": frame[["value"]].reset_index(drop=True),
        }
        labels = ("first", "second") if self.two_outputs else ("output",)
        return BatchResult(outputs={label: payload for label in labels})

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = mode, request
        self.events.append(("handle", batch_index))
        return result.outputs

    def finalize_translation(self, *, mode, request):
        _ = mode, request
        self.events.append(("final", None))
        return None

    def make_translate_worker(self, *, mode, request):
        _ = mode, request
        return type(self)(self.shape, two_outputs=self.two_outputs)

    def to_json_state(self):
        return {"shape": self.shape, "two_outputs": self.two_outputs}

    @classmethod
    def from_json_state(cls, state):
        return cls(state["shape"], two_outputs=state["two_outputs"])


def _plan(project, operation_id):
    plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
    try:
        return [plan.get_unit(i) for i in range(plan.total_count())], plan.complete_count()
    finally:
        plan.close()


def _operation(project):
    operation_id = str(project.list_operations()[-1]["operation_id"])
    metadata = json.loads(
        (project.storage.operation_dir(operation_id) / "operation.json").read_text()
    )
    return operation_id, metadata


@pytest.mark.parametrize(
    ("shape", "batch_size", "expected"),
    [
        ("rows", 2, [("a", "b"), ("c",)]),
        ("single", 1, [("a",), ("b",), ("c",)]),
        ("full", None, [("a", "b", "c")]),
    ],
)
def test_default_one_pass_packets_and_finalization(
    tmp_path, shape, batch_size, expected
):
    project = Project.create(tmp_path / shape, name="one_pass_" + shape)
    try:
        source = _source(
            project, "art_values",
            pd.DataFrame({"id": [0, 1, 2], "value": ["a", "b", "c"]}),
            ["id"],
        )
        probe = _OnePassProbe(shape)
        result = project.translate(probe, source, batch_size=batch_size)["output"]
        assert result.n_rows == 3
        events = probe.events
        received = [event[1]["source"] for event in events if event[0] == "batch"]
        assert [packet[4] for packet in received] == expected
        assert [packet[:4] for packet in received] == [
            (i, len(expected), i == 0, i == len(expected) - 1)
            for i in range(len(expected))
        ]
        assert [event for event in events if event[0] == "handle"] == [
            ("handle", i) for i in range(len(expected))
        ]
        assert events[-1] == ("final", None)
        assert sum(kind == "final" for kind, _ in events) == 1

        operation_id, metadata = _operation(project)
        units, completed = _plan(project, operation_id)
        assert len(units) == completed == len(expected)
        assert metadata["mode"] == "translate"
        assert metadata["route"] == "sequential"
        assert metadata["input_request"]["source"]["mode"] == (
            "full_artifact" if shape == "full" else "batches"
        )
        assert metadata["status"] == "complete"
        assert result.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == ["a", "b", "c"]
    finally:
        project.close()


def test_one_pass_uneven_mixed_sources_multioutput_and_reopen(tmp_path):
    path = tmp_path / "mixed"
    project = Project.create(path, name="one_pass_mixed")
    try:
        left = _source(
            project, "art_left",
            pd.DataFrame({"id": list(range(5)), "value": list("abcde")}),
            ["id"],
        )
        right = _source(
            project, "art_right",
            pd.DataFrame({"ref": [7, 8], "value": ["x", "y"]}),
            ["ref"],
        )
        probe = _OnePassProbe("mixed", two_outputs=True)
        outputs = project.translate(probe, {"left": left, "right": right}, batch_size=2)
        assert set(outputs) == {"first", "second"}
        seen = [e[1] for e in probe.events if e[0] == "batch"]
        assert [list(unit) for unit in seen] == [
            ["left", "right"], ["left"], ["left"]
        ]
        assert [unit["left"][4] for unit in seen] == [
            ("a", "b"), ("c", "d"), ("e",)
        ]
        assert seen[0]["right"] == (0, 1, True, True, ("x", "y"))
        assert [e for e in probe.events if e[0] == "handle"] == [
            ("handle", 0), ("handle", 1), ("handle", 2)
        ]
        assert probe.events[-1] == ("final", None)

        operation_id, metadata = _operation(project)
        units, completed = _plan(project, operation_id)
        assert completed == len(units) == 3
        assert [list(unit.sources) for unit in units] == [
            ["left", "right"], ["left"], ["left"]
        ]
        assert len(project.operation_sources(operation_id)) == 2
        assert len(project.operation_outputs(operation_id)) == 2
        assert metadata["status"] == "complete"
        ids = {label: output.artifact_id for label, output in outputs.items()}
        for output in outputs.values():
            assert output.status == "complete"
            assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
            assert output.query(
                key_columns=False, data_columns="value", form="table"
            )["value"].tolist() == list("abcde")
    finally:
        project.close()

    reopened = Project.open(path)
    try:
        assert [reopened.get_artifact(aid).n_rows for aid in ids.values()] == [5, 5]
        assert reopened.get_operation(operation_id)["status"] == "complete"
    finally:
        reopened.close()


def test_one_pass_grouped_plan_preserves_complete_groups_and_order(tmp_path):
    project = Project.create(tmp_path / "grouped", name="one_pass_grouped")
    try:
        source = _source(
            project, "art_grouped",
            pd.DataFrame({
                "corpus": [1, 2, 1, 2, 1],
                "document": [1, 1, 1, 1, 2],
                "token": [0, 0, 1, 1, 0],
                "value": list("abcde"),
            }),
            ["corpus", "document", "token"],
        )
        probe = _OnePassProbe("rows")
        output = project.translate(
            probe, source, batch_by="document", batch_size=1
        )["output"]
        seen = [e[1]["source"] for e in probe.events if e[0] == "batch"]
        assert [packet[4] for packet in seen] == [
            ("a", "c"), ("b", "d"), ("e",)
        ]
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == list("acbde")

        operation_id, metadata = _operation(project)
        units, completed = _plan(project, operation_id)
        assert completed == len(units) == 3
        assert [
            json.loads(unit.sources["source"].group_keys_json) for unit in units
        ] == [[[1, 1]], [[2, 1]], [[1, 2]]]
        assert metadata["request"]["batch_by"] == "document"
        assert probe.events[-1] == ("final", None)
    finally:
        project.close()


def test_existing_count_vectorizer_fit_and_frozen_reuse_remain_one_pass(tmp_path):
    from text_analysis_lab.translators import CountVectorizer

    project = Project.create(tmp_path / "vectorizer", name="one_pass_vectorizer")
    try:
        source = _source(
            project, "art_text",
            pd.DataFrame({
                "id": [1, 2, 3],
                "value": ["red green", "green blue", "red blue red"],
            }),
            ["id"],
        )
        vectorizer = CountVectorizer(text_field="value")
        fitted = project.translate(vectorizer, source)["output"]
        fit_id, fit_metadata = _operation(project)
        fit_units, fit_complete = _plan(project, fit_id)
        assert fit_metadata["mode"] == "fit_translate"
        assert fit_metadata["input_request"]["source"]["mode"] == "full_artifact"
        assert len(fit_units) == fit_complete == 1
        assert vectorizer.is_fitted

        reused = project.translate(vectorizer, source, batch_size=2)["output"]
        transform_id, transform_metadata = _operation(project)
        transform_units, transform_complete = _plan(project, transform_id)
        assert transform_metadata["mode"] == "translate"
        assert transform_metadata["input_request"]["source"]["mode"] == "batches"
        assert len(transform_units) == transform_complete == 2
        assert fitted.get_data_columns() == reused.get_data_columns()
        assert (fitted.get_matrix() != reused.get_matrix()).nnz == 0
    finally:
        project.close()


def test_one_pass_parallel_canonical_handler_and_output(tmp_path):
    pytest.importorskip("dask.distributed")
    project = Project.create(tmp_path / "parallel", name="one_pass_parallel")
    try:
        source = _source(
            project, "art_parallel",
            pd.DataFrame({"id": [0, 1, 2, 3, 4], "value": list("abcde")}),
            ["id"],
        )
        probe = _OnePassProbe("rows")
        output = project.translate(
            probe, source, workers=2, batch_size=2, max_outstanding_units=2
        )["output"]
        assert [e for e in probe.events if e[0] == "handle"] == [
            ("handle", 0), ("handle", 1), ("handle", 2)
        ]
        assert probe.events[-1] == ("final", None)
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == list("abcde")
        operation_id, metadata = _operation(project)
        units, completed = _plan(project, operation_id)
        assert metadata["route"] == "parallel"
        assert len(units) == completed == 3
    finally:
        project.close()
