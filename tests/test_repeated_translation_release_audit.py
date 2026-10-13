"""Gate E release audit for opt-in repeated translation plans.

These tests protect compatibility, real persisted artifact semantics and bounded
plan storage; they do not alter or exercise deferred vectorizer algorithms.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.core.operator import (
    BaseTranslator, BatchResult, ColumnRequest, OutputSpec, SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.translate import (
    _ExecutionPlan, _build_plan_units,
)
from text_analysis_lab.core.writer import create_artifact_writer
from test_repeated_translation_plan import _ReplayProbe, _source
from test_hierarchy_aware_batching import _Echo, _source as _hierarchical_source
from test_repeated_translation_parallel_and_multisource import _ParallelProbe


def _operation(project):
    op_id = str(project.list_operations()[-1]["operation_id"])
    return op_id, json.loads(
        (project.storage.operation_dir(op_id) / "operation.json").read_text()
    )


def test_prior_one_pass_descriptor_without_pass_metadata_resumes(tmp_path):
    """Real incomplete operations without new metadata retain old semantics."""
    path = tmp_path / "legacy"
    project = Project.create(path, name="legacy_pass_metadata")
    try:
        source = _source(project)
        translator = _ReplayProbe(
            1, fail_at=(0, 1), signal=tmp_path / "fail_once"
        )
        with pytest.raises(RuntimeError, match="interrupted replay"):
            project.translate(translator, source)
        op_id, descriptor = _operation(project)
        assert descriptor["request"].pop("pass_count") == 1
        operation_path = project.storage.operation_dir(op_id) / "operation.json"
        operation_path.write_text(json.dumps(descriptor))
        # Previous one-pass engines had no pass-marker file. The existing
        # canonical translator and writer intermediate checkpoints remain.
        (project.storage.operation_temp_dir(op_id) / "pass.json").unlink()
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
        plan = _ExecutionPlan(reopened.storage.operation_dir(op_id))
        try:
            assert plan.total_count() == plan.complete_count() == 3
        finally:
            plan.close()
    finally:
        reopened.close()


def test_three_pass_plan_is_resolved_once_and_has_only_three_sqlite_units(
    tmp_path, monkeypatch
):
    import text_analysis_lab.core.translate as runner

    project = Project.create(tmp_path / "linear", name="stable_plan")
    plan_builds = []
    original_build = runner._build_plan_units

    def watch(*args, **kwargs):
        plan_builds.append(1)
        yield from original_build(*args, **kwargs)

    monkeypatch.setattr(runner, "_build_plan_units", watch)
    try:
        source = _source(project)
        output = project.translate(_ReplayProbe(3), source)["output"]
        op_id, descriptor = _operation(project)
        assert descriptor["request"]["pass_count"] == 3
        assert len(plan_builds) == 1
        plan = _ExecutionPlan(project.storage.operation_dir(op_id))
        try:
            assert plan.total_count() == plan.complete_count() == 3
            assert plan.con.execute("SELECT COUNT(*) FROM plan_sources").fetchone()[0] == 3
            assert plan.con.execute(
                "SELECT current_pass FROM plan_pass_state WHERE id=1"
            ).fetchone()[0] == 2
            assert [(plan.get_unit(i).sources["source"].start_position,
                     plan.get_unit(i).sources["source"].stop_position)
                    for i in range(3)] == [(0, 2), (2, 4), (4, 5)]
        finally:
            plan.close()
        assert output.n_rows == 5
        assert len(project.operation_sources(op_id)) == 1
        assert len(project.operation_outputs(op_id)) == 1
    finally:
        project.close()


class _GroupedReplay(_Echo):
    def __init__(self, *, operator_id=None):
        super().__init__(min_level="document", operator_id=operator_id)

    def translation_passes(self, *, sources, mode, request, input_request):
        return 2

    def translate_batch(self, inputs, *, mode, request):
        if request.pass_index == 0:
            return BatchResult()
        return super().translate_batch(inputs, mode=mode, request=request)


def test_repeated_grouped_batches_retain_original_source_group_assignments(tmp_path):
    project = Project.create(tmp_path / "groups", name="repeated_group")
    try:
        source, rows = _hierarchical_source(project)
        output = project.translate(
            _GroupedReplay(), source, batch_by="document", batch_size=1
        )["output"]
        assert output.n_rows == len(rows)
        actual = output.query(key_columns=True, data_columns="value", form="table")
        cols = ["collection", "document", "token", "value"]
        assert sorted(map(tuple, actual[cols].to_numpy())) == sorted(
            map(tuple, rows[cols].to_numpy())
        )
        op_id, descriptor = _operation(project)
        assert descriptor["request"]["batch_by"] == "document"
        assert descriptor["request"]["pass_count"] == 2
        plan = _ExecutionPlan(project.storage.operation_dir(op_id))
        try:
            assert plan.complete_count() == plan.total_count() == 4
            assert [
                json.loads(plan.get_unit(i).sources["source"].group_keys_json)
                for i in range(4)
            ] == [[[1, 1]], [[2, 1]], [[1, 2]], [[2, 2]]]
            assert plan.con.execute(
                "SELECT COUNT(*) FROM plan_sources"
            ).fetchone()[0] == 4
        finally:
            plan.close()
    finally:
        project.close()


class _SparseReplay(BaseTranslator):
    def translation_passes(self, *, sources, mode, request, input_request):
        return 2

    def output_specs(self, *, sources, request):
        return OutputSpec(
            artifact_type="sparse_matrix", lineage_mode="preserved_key",
            basis_labels="source", feature_metadata_mode="inherit",
        )

    def input_request(self, *, sources, mode, request):
        return SourceRequest(
            artifact_type="sparse_matrix", mode="batches", batch_size=2,
            form="native", columns=ColumnRequest(keys=True, data=True, metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        if request.pass_index == 0:
            return BatchResult()
        from text_analysis_lab.translators._matrix_transform_utils import (
            native_matrix_packet, key_frame,
        )
        from scipy import sparse as sp
        info, matrix, key_columns = native_matrix_packet(
            inputs["source"], name="sparse_replay"
        )
        values = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
        return BatchResult(outputs={"output": {
            "keys": key_frame(info, key_columns),
            "data": {"values": values, "columns": ["alpha", "beta", "gamma"]},
        }})

    def handle_batch_result(self, result, *, batch_index, mode, request):
        return result.outputs

    def finalize_translation(self, *, mode, request):
        return None


def test_sparse_repeated_plan_preserves_fixed_features_and_reopens(tmp_path):
    path = tmp_path / "sparse"
    project = Project.create(path, name="sparse_repeat")
    expected = sparse.csr_matrix([
        [1, 0, 0], [0, 2, 1], [3, 0, 0], [0, 1, 2], [1, 1, 1]
    ])
    try:
        writer = create_artifact_writer(
            artifact_type="sparse_matrix",
            artifact_dir=project.storage.artifact_dir("art_sparse_repeat"),
            artifact_id="art_sparse_repeat", label="source",
            lineage_mode="new_key", basis_artifact_ids=(),
        )
        for start, end in ((0, 2), (2, 5)):
            writer.write({
                "keys": pd.DataFrame({"doc_id": list(range(start, end))}),
                "data": {
                    "values": expected[start:end],
                    "columns": ["alpha", "beta", "gamma"],
                },
            })
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id="art_sparse_repeat", artifact_type="sparse_matrix",
            label="source", lineage_mode="new_key", status="complete",
            basis_artifact_ids=(),
        )
        source = project.get_artifact("art_sparse_repeat")
        output = project.translate(_SparseReplay(), source)["output"]
        assert output.get_data_columns() == ["alpha", "beta", "gamma"]
        assert output.n_rows == expected.shape[0]
        assert (output.get_matrix() != expected).nnz == 0
        output_id = output.artifact_id
        op_id, metadata = _operation(project)
        assert metadata["request"]["pass_count"] == 2
        assert project.operation_for_artifact(output_id)["operation_id"] == op_id
    finally:
        project.close()
    reopened = Project.open(path)
    try:
        output = reopened.get_artifact(output_id)
        assert output.get_data_columns() == ["alpha", "beta", "gamma"]
        assert (output.get_matrix() != expected).nnz == 0
    finally:
        reopened.close()


def test_zero_unit_plan_advance_does_not_invent_sources_or_batches(tmp_path):
    """Zero-row batched source plans stay empty for each scheduled traversal."""
    class Empty:
        artifact_id = "art_empty"
        n_rows = 0

    specs = {"source": SourceRequest(
        artifact_type="table", mode="batches", batch_size=5,
        form="table",
    )}
    units = list(_build_plan_units(
        {"source": Empty()}, specs, TranslationRequest(batch_size=5, pass_count=3)
    ))
    assert units == []
    plan = _ExecutionPlan(tmp_path / "empty_plan")
    try:
        plan.initialize(units)
        assert plan.total_count() == 0
        for index in range(3):
            plan.activate_pass(index)
            assert plan.complete_count() == plan.total_count() == 0
            assert list(plan.iter_pending()) == []
        assert plan.con.execute("SELECT COUNT(*) FROM plan_sources").fetchone()[0] == 0
    finally:
        plan.close()


def test_parallel_pass_two_resume_uses_checkpointed_fitted_state(
    tmp_path, monkeypatch
):
    pytest.importorskip("dask.distributed")
    import text_analysis_lab.core.translate as runner
    from test_repeated_translation_parallel_and_multisource import _source as make_source

    path = tmp_path / "parallel"
    project = Project.create(path, name="repeated_parallel_resume")
    original = runner._parallel_results
    failed = {"once": False}

    def interrupted(**kwargs):
        for item in original(**kwargs):
            yield item
            if kwargs["request"].pass_index == 1 and item[0] == 0 and not failed["once"]:
                failed["once"] = True
                raise RuntimeError("injected pass-two parallel failure")

    monkeypatch.setattr(runner, "_parallel_results", interrupted)
    try:
        source = make_source(
            project, "art_parallel_resume", list(range(5)), list("abcde")
        )
        with pytest.raises(RuntimeError, match="injected pass-two parallel failure"):
            project.translate(
                _ParallelProbe(), source, workers=2, max_outstanding_units=2
            )
        op_id, metadata = _operation(project)
        assert failed["once"] is True
        assert metadata["route"] == "parallel"
        assert metadata["request"]["pass_count"] == 2
    finally:
        project.close()
    monkeypatch.undo()

    reopened = Project.open(path)
    try:
        output = reopened.resume_operation(op_id)["output"]
        assert output.n_rows == 5
        assert output.query(key_columns=False, data_columns="value", form="table")[
            "value"
        ].tolist() == list("abcde")
        assert reopened.get_operation(op_id)["status"] == "complete"
    finally:
        reopened.close()
