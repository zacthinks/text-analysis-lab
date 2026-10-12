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
