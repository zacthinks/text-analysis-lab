"""End-to-end contracts for optional primary-key-level translation batching."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator, BatchResult, ColumnRequest, OutputSpec, SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.translate import (
    _ExecutionPlan, _PlanUnit, _SourcePlan, _batch_prefix, _validate_batch_hierarchy,
)


class _Echo(BaseTranslator):
    def __init__(self, min_level=None, *, operator_id=None):
        super().__init__(operator_id=operator_id)
        self.min_level = min_level

    @property
    def supports_parallel_translate(self):
        return True

    def output_specs(self, *, sources, request):
        return OutputSpec(artifact_type="table", lineage_mode="preserved_key")

    def input_request(self, *, sources, mode, request):
        return SourceRequest(
            artifact_type="table", mode="batches", batch_size=request.batch_size or 3,
            min_batch_by=self.min_level, form="table",
            columns=ColumnRequest(keys=True, data="value", metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        source = inputs["source"]
        frame = source.data
        return BatchResult(outputs={
            "output": {
                "keys": frame[list(source.primary_key)].reset_index(drop=True),
                "data": frame[["value"]].reset_index(drop=True),
            }
        })

    def handle_batch_result(self, result, *, batch_index, mode, request):
        return result.outputs

    def finalize_translation(self, *, mode, request):
        return None

    def make_translate_worker(self, *, mode, request):
        return _Echo(self.min_level)

    def to_json_state(self):
        return {"min_level": self.min_level}

    @classmethod
    def from_json_state(cls, state):
        return cls(state["min_level"])


def _source(project):
    from text_analysis_lab.core.writer import create_artifact_writer

    # Deliberately out of key order and interleaved by document; repeated document
    # IDs under distinct collections must remain different groups.
    rows = pd.DataFrame({
        "collection": [1, 2, 1, 1, 2, 1, 2],
        "document":   [1, 1, 2, 1, 1, 2, 2],
        "token":      [0, 0, 0, 1, 1, 1, 0],
        "value":      ["a", "b", "c", "d", "e", "f", "g"],
    })
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_hierarchy_test"),
        artifact_id="art_hierarchy_test",
        label="source", lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({
        "keys": rows[["collection", "document", "token"]],
        "data": rows[["value"]],
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_hierarchy_test", artifact_type="table", label="source",
        lineage_mode="new_key", status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact("art_hierarchy_test"), rows


def test_prefix_resolution_and_required_boundary():
    class Fake:
        primary_key = ("a", "b", "c", "d")
    source = Fake()
    assert _batch_prefix(source, "c") == ("a", "b", "c")
    with pytest.raises(OperatorError, match="not in source primary key"):
        _batch_prefix(source, "document")
    spec = SourceRequest(artifact_type="table", mode="batches", batch_size=2, min_batch_by="c")
    _validate_batch_hierarchy({"source": source}, {"source": spec}, TranslationRequest(batch_by="b"))
    _validate_batch_hierarchy({"source": source}, {"source": spec}, TranslationRequest(batch_by="c"))
    with pytest.raises(OperatorError, match="finer"):
        _validate_batch_hierarchy({"source": source}, {"source": spec}, TranslationRequest(batch_by="d"))
    with pytest.raises(OperatorError, match="requires batch_by"):
        _validate_batch_hierarchy({"source": source}, {"source": spec}, TranslationRequest())
    leaf = SourceRequest(
        artifact_type="table", mode="batches", batch_size=2, min_batch_by="d"
    )
    _validate_batch_hierarchy({"source": source}, {"source": leaf}, TranslationRequest())


def test_plan_round_trip_and_restart(tmp_path):
    plan = _ExecutionPlan(tmp_path / "run")
    serialized = json.dumps([[1, 1], [1, 2]])
    try:
        plan.initialize([_PlanUnit(0, {"source": _SourcePlan(
            "source", "artifact", "batches", 0, 1, group_keys_json=serialized
        )})])
        assert plan.get_unit(0).sources["source"].group_keys_json == serialized
        plan.mark_running(0)
        plan.reset_running_to_pending()
        assert [u.unit_index for u in plan.iter_pending()] == [0]
    finally:
        plan.close()


def test_project_grouped_batches_preserve_rows_and_are_reopenable(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from text_analysis_lab import Project

    path = tmp_path / "project"
    project = Project.create(path, name="grouped")
    try:
        source, original = _source(project)
        assert _batch_prefix(source, "document") == ("collection", "document")

        # Every complete document group is one batch; source's physical rows
        # deliberately interleave groups, so contiguous slices are not enough.
        output = project.translate(
            _Echo(min_level="document"), source,
            batch_size=1, batch_by="document",
        )["output"]
        assert output.n_rows == len(original)
        # Group batches follow first occurrence of each group, then stable
        # artifact position within that group. This differs from interleaved
        # physical source order and is an explicit batching policy.
        assert output.query(
            key_columns=False, data_columns="value", form="table"
        )["value"].tolist() == ["a", "d", "b", "e", "c", "f", "g"]
        observed = output.query(
            key_columns=True, data_columns="value", form="table"
        )
        fields = ["collection", "document", "token", "value"]
        assert sorted(map(tuple, observed[fields].to_numpy())) == sorted(
            map(tuple, original[fields].to_numpy())
        )
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        output_id = output.artifact_id

        with pytest.raises(OperatorError, match="finer"):
            project.translate(_Echo(min_level="document"), source, batch_size=2, batch_by="token")
        with pytest.raises(OperatorError, match="requires batch_by"):
            project.translate(_Echo(min_level="document"), source, batch_size=2)
        with pytest.raises(OperatorError, match="full_artifact"):
            from text_analysis_lab.translators import ArtifactCountVectorizer
            project.translate(
                ArtifactCountVectorizer(field="value", group_by=("collection",)),
                source, batch_size=1, batch_by="document",
            )
    finally:
        project.close()

    reopened = Project.open(path)
    try:
        assert reopened.get_artifact(output_id).n_rows == len(original)
    finally:
        reopened.close()


class _FailOnceEcho(_Echo):
    def __init__(self, sentinel_path, *, operator_id=None):
        super().__init__(min_level="document", operator_id=operator_id)
        self.sentinel_path = str(sentinel_path)

    def supports_resume(self, *, mode, route):
        return mode == "translate" and route == "sequential"

    def translate_batch(self, inputs, *, mode, request):
        batch = inputs["source"]
        if batch.batch_index == 1 and not __import__("pathlib").Path(self.sentinel_path).exists():
            __import__("pathlib").Path(self.sentinel_path).write_text("failed")
            raise RuntimeError("interrupted grouped unit")
        return super().translate_batch(inputs, mode=mode, request=request)

    def to_json_state(self):
        return {"sentinel_path": self.sentinel_path}

    @classmethod
    def from_json_state(cls, state):
        return cls(state["sentinel_path"])

    def save_intermediate_state(self, intermediate_dir, *, operator_id, mode, route):
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "state.json").write_text(json.dumps(self.to_json_state()))

    @classmethod
    def load_intermediate_state(cls, intermediate_dir, *, operator_id, mode, route):
        state = json.loads((intermediate_dir / "state.json").read_text())
        value = cls.from_json_state(state)
        value.operator_id = operator_id
        return value


def test_grouped_translation_resumes_without_replaying_completed_units(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from text_analysis_lab import Project

    project = Project.create(tmp_path / "resume_project", name="grouped_resume")
    try:
        source, rows = _source(project)
        with pytest.raises(RuntimeError, match="interrupted grouped unit"):
            project.translate(
                _FailOnceEcho(tmp_path / "fail-once"),
                source,
                batch_size=1,
                batch_by="document",
                workers=1,
            )
        operation_id = str(project.catalog.list_operations()[-1]["operation_id"])
        assert project.catalog.get_operation(operation_id)["status"] == "failed"

        from text_analysis_lab.core.translate import _ExecutionPlan
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            assert plan.complete_count() == 1
            first = plan.get_unit(0).sources["source"]
            assert json.loads(first.group_keys_json) == [[1, 1]]
        finally:
            plan.close()

        output = project.resume_operation(operation_id)["output"]
        assert output.status == "complete"
        assert output.n_rows == len(rows)
        actual = output.query(key_columns=True, data_columns="value", form="table")
        fields = ["collection", "document", "token", "value"]
        assert sorted(map(tuple, actual[fields].to_numpy())) == sorted(
            map(tuple, rows[fields].to_numpy())
        )
        assert len(actual.drop_duplicates(["collection", "document", "token"])) == len(rows)
        assert project.catalog.get_operation(operation_id)["status"] == "complete"
    finally:
        project.close()


def test_legacy_positional_requests_are_unchanged():
    # Existing user-defined translators may construct these positionally.
    from text_analysis_lab.core.operator import ColumnRequest
    columns = ColumnRequest(keys=True, data=False, metadata=False)
    old = SourceRequest("table", "batches", columns, 17, "table", "none", True)
    assert old.batch_size == 17
    assert old.form == "table"
    assert old.metadata_mode == "none"
    assert old.include_position is True
    assert old.min_batch_by is None
    old_translation = TranslationRequest(2, 5, 3, {"flag": 1})
    assert old_translation.workers == 2
    assert old_translation.batch_size == 5
    assert old_translation.max_outstanding_units == 3
    assert old_translation.params == {"flag": 1}
    assert old_translation.batch_by is None


def test_legacy_plan_sqlite_migration(tmp_path):
    import sqlite3

    path = tmp_path / "old_operation"
    path.mkdir()
    connection = sqlite3.connect(path / "plan.sqlite")
    connection.executescript("""
        CREATE TABLE plan_units (
            unit_index INTEGER PRIMARY KEY, status TEXT NOT NULL,
            assigned_worker INTEGER, started_at TEXT, completed_at TEXT, error TEXT
        );
        CREATE TABLE plan_sources (
            unit_index INTEGER NOT NULL, source_label TEXT NOT NULL,
            artifact_id TEXT NOT NULL, mode TEXT NOT NULL,
            source_batch_index INTEGER NOT NULL, source_batch_count INTEGER NOT NULL,
            start_position INTEGER, stop_position INTEGER,
            PRIMARY KEY (unit_index, source_label)
        );
        INSERT INTO plan_units(unit_index, status) VALUES (0, 'pending');
        INSERT INTO plan_sources VALUES (0, 'source', 'old', 'batches', 0, 1, 0, 2);
    """)
    connection.close()
    plan = _ExecutionPlan(path)
    try:
        old = plan.get_unit(0).sources["source"]
        assert old.start_position == 0 and old.stop_position == 2
        assert old.group_keys_json is None
        plan.mark_complete(0)
        assert plan.complete_count() == 1
    finally:
        plan.close()


def test_user_key_names_overlap_internal_group_aliases(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from text_analysis_lab import Project
    from text_analysis_lab.core.writer import create_artifact_writer

    project = Project.create(tmp_path / "collision", name="aliases")
    try:
        # TeAL rejects primary-key labels beginning with "_". Exercise the
        # original unqualified first_position SQL alias collision using valid
        # opaque labels, rather than weakening the primary-key contract.
        keys = ["first_position", "batch_order", "teal_group_first_position", "token"]
        rows = pd.DataFrame({
            "first_position": [1, 2, 1, 2],
            "batch_order": [1, 1, 1, 1],
            "teal_group_first_position": [1, 1, 1, 1],
            "token": [0, 0, 1, 1],
            "value": ["one", "two", "three", "four"],
        })
        writer = create_artifact_writer(
            artifact_type="table", artifact_dir=project.storage.artifact_dir("art_aliases"),
            artifact_id="art_aliases", label="source", lineage_mode="new_key",
            basis_artifact_ids=(),
        )
        writer.write({"keys": rows[keys], "data": rows[["value"]]})
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id="art_aliases", artifact_type="table", label="source",
            lineage_mode="new_key", status="complete", basis_artifact_ids=(),
        )
        source = project.get_artifact("art_aliases")
        output = project.translate(
            _Echo(min_level="teal_group_first_position"), source,
            batch_size=1, batch_by="teal_group_first_position",
        )["output"]
        assert output.n_rows == 4
        actual = output.query(key_columns=True, data_columns="value", form="table")
        assert sorted(actual["value"].tolist()) == sorted(rows["value"].tolist())
    finally:
        project.close()


class _TwoOutputEcho(_Echo):
    def output_specs(self, *, sources, request):
        return {
            "left": OutputSpec(artifact_type="table", lineage_mode="preserved_key"),
            "right": OutputSpec(artifact_type="table", lineage_mode="preserved_key"),
        }

    def translate_batch(self, inputs, *, mode, request):
        one = super().translate_batch(inputs, mode=mode, request=request).outputs["output"]
        return BatchResult(outputs={"left": one, "right": {
            "keys": one["keys"].copy(),
            "data": one["data"].copy(),
        }})

    def make_translate_worker(self, *, mode, request):
        return _TwoOutputEcho(self.min_level)


def test_grouped_multi_output_writer_and_parallel_workers(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    pytest.importorskip("dask.distributed")
    from text_analysis_lab import Project

    project = Project.create(tmp_path / "parallel", name="grouped_parallel")
    try:
        source, rows = _source(project)
        results = project.translate(
            _TwoOutputEcho(min_level="document"), source,
            batch_size=1, batch_by="document", workers=2,
        )
        assert set(results) == {"left", "right"}
        for output in results.values():
            assert output.n_rows == len(rows)
            actual = output.query(key_columns=True, data_columns="value", form="table")
            fields = ["collection", "document", "token", "value"]
            assert sorted(map(tuple, actual[fields].to_numpy())) == sorted(
                map(tuple, rows[fields].to_numpy())
            )
    finally:
        project.close()
