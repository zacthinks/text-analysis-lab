from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    InputBatch,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.translate import (
    _ExecutionPlan,
    _parallel_results,
    _PlanUnit,
    _replace_directory_with_staging,
    _SourcePlan,
)


class _ProcessAsCompleted:
    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()

    def add(self, future: Any) -> None:
        future.add_done_callback(self._queue.put)

    def __iter__(self):
        return self

    def __next__(self):
        return self._queue.get(timeout=30)


class _ProcessClient:
    def __init__(self, workers: int) -> None:
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self.submitted = 0
        self.finished = 0
        self.max_in_flight = 0

    def scatter(self, value: Any, *, broadcast: bool = False) -> Any:
        _ = broadcast
        return value

    def submit(self, fn, *args, pure: bool = False):
        _ = pure
        future = self.executor.submit(fn, *args)
        self.submitted += 1
        self.max_in_flight = max(self.max_in_flight, self.submitted - self.finished)

        def done(_future):
            self.finished += 1

        future.add_done_callback(done)
        return future

    def cancel(self, futures) -> None:
        for future in futures:
            future.cancel()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


class _ProcessCluster:
    def close(self) -> None:
        return None


class _DelayTranslator(BaseTranslator):
    operation_type = "translate"

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def output_specs(self, *, sources, request):
        return {
            "output": OutputSpec(artifact_type="table", lineage_mode="preserved_key")
        }

    def input_request(self, *, sources, mode, request):
        return SourceRequest(
            artifact_type="table", mode="batches", batch_size=1, form="table"
        )

    def translate_batch(
        self, inputs: Mapping[str, InputBatch], *, mode, request
    ) -> BatchResult:
        _ = mode, request
        batch = next(iter(inputs.values()))
        unit = int(batch.data["unit"].iloc[0])
        # Force completion order to differ from plan order.
        time.sleep({0: 0.20, 1: 0.02, 2: 0.01, 3: 0.03, 4: 0.01}.get(unit, 0.01))
        return BatchResult(value=unit)

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = result, batch_index, mode, request

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def make_translate_worker(self, *, mode, request):
        _ = mode, request
        return _DelayTranslator()


@dataclass
class _FakeArtifact:
    artifact_id: str = "art_source"
    primary_key: tuple[str, ...] = ("id",)


def _plan_units(n: int) -> list[_PlanUnit]:
    return [
        _PlanUnit(
            unit_index=index,
            sources={
                "source": _SourcePlan(
                    source_label="source",
                    artifact_id="art_source",
                    mode="batches",
                    source_batch_index=index,
                    source_batch_count=n,
                    start_position=index,
                    stop_position=index + 1,
                )
            },
        )
        for index in range(n)
    ]


def _fake_materialize(unit: _PlanUnit, *, sources, input_request):
    _ = sources, input_request
    index = unit.unit_index
    source_plan = unit.sources["source"]
    return {
        "source": InputBatch(
            source_label="source",
            artifact_id="art_source",
            primary_key=("id",),
            data=pd.DataFrame({"id": [index], "unit": [index]}),
            batch_index=index,
            batch_count=source_plan.source_batch_count,
            is_first=index == 0,
            is_last=index == source_plan.source_batch_count - 1,
        )
    }


def test_parallel_results_are_restored_to_plan_order_with_real_processes(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab.core.translate as translate_module

    plan = _ExecutionPlan(tmp_path / "operation")
    plan.initialize(_plan_units(5))
    client_holder: dict[str, _ProcessClient] = {}

    def open_client(*, workers: int):
        client = _ProcessClient(workers)
        client_holder["client"] = client
        return client, _ProcessCluster(), _ProcessAsCompleted

    monkeypatch.setattr(translate_module, "_open_dask_client", open_client)
    monkeypatch.setattr(translate_module, "_materialize_plan_unit", _fake_materialize)

    request = TranslationRequest(workers=3, batch_size=1, max_outstanding_units=3)
    try:
        results = list(
            _parallel_results(
                translate_worker=_DelayTranslator(),
                plan=plan,
                sources={"source": _FakeArtifact()},
                input_request={
                    "source": SourceRequest(artifact_type="table", batch_size=1)
                },
                mode="translate",
                request=request,
                workers=3,
            )
        )
    finally:
        plan.close()

    assert [index for index, _ in results] == [0, 1, 2, 3, 4]
    assert [result.value for _, result in results] == [0, 1, 2, 3, 4]
    assert client_holder["client"].max_in_flight <= 3


def test_resume_retries_failed_and_interrupted_units(tmp_path: Path) -> None:
    plan = _ExecutionPlan(tmp_path / "operation")
    plan.initialize(_plan_units(5))
    try:
        plan.mark_complete(0)
        plan.mark_failed(1, RuntimeError("transient"))
        plan.mark_running(2)
        # units 3 and 4 remain pending

        plan.reset_running_to_pending()
        pending = [unit.unit_index for unit in plan.iter_pending()]

        # A resume must retry the failed unit itself, reset any interrupted running
        # work, leave already-complete units alone, and retain untouched pending work.
        assert pending == [1, 2, 3, 4]
        assert plan.complete_count() == 1
    finally:
        plan.close()


# ---------------------------------------------------------------------------
# Near-end-to-end translation/resume tests.  These replace only unavailable
# parquet/DuckDB I/O; the real Project, Catalog, operation runner, plan,
# checkpoints, ArtifactWriter, finalization, and lineage validation are used.
# ---------------------------------------------------------------------------


def _install_pickle_parquet(monkeypatch) -> None:
    def fake_to_parquet(self, path, index=False, **kwargs):
        _ = index, kwargs
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_pickle(path)

    def fake_read_parquet(path, columns=None, **kwargs):
        _ = kwargs
        frame = pd.read_pickle(path)
        if columns is not None:
            frame = frame.loc[:, list(columns)]
        return frame.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet, raising=True)


def _seed_source_artifact(project, rows: pd.DataFrame):
    artifact_id = "art_source"
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label="source",
        lineage_mode="new_key",
        status="complete",
    )
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": "source",
        "status": "complete",
        "primary_key": ["id"],
        "n_rows": len(rows),
        "components": {"keys": {"format": "parquet_dataset", "path": "keys/"}},
        "lineage": {"lineage_mode": "new_key", "basis_artifact_ids": []},
        "operation_id": None,
        "write": {"mode": "batch", "parts": 1},
    }
    project.storage.artifact_descriptor_path(artifact_id).write_text(
        __import__("json").dumps(descriptor, indent=2), encoding="utf-8"
    )
    artifact = project.get_artifact(artifact_id)
    artifact._test_rows = rows.copy()  # type: ignore[attr-defined]
    return artifact


def _install_fake_source_queries(monkeypatch, rows: pd.DataFrame) -> None:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    def fake_query(self, **kwargs):
        frame = rows.copy()
        where = kwargs.get("where")
        if where:
            import re

            match = re.fullmatch(
                r"_position >= (\d+) AND _position < (\d+)", str(where)
            )
            if not match:
                raise AssertionError(f"unexpected where clause: {where}")
            start, stop = map(int, match.groups())
            frame = frame.iloc[start:stop].copy()
        frame = frame.reset_index(drop=True)
        frame["_position"] = range(
            0 if not where else int(str(where).split()[2]),
            (0 if not where else int(str(where).split()[2])) + len(frame),
        )
        return frame

    monkeypatch.setattr(BaseArtifact, "query", fake_query, raising=True)


class _TransientResumableTranslator(BaseTranslator):
    operation_type = "translate"

    def __init__(self, sentinel_path: str, fail_unit: int, *, operator_id=None) -> None:
        super().__init__(operator_id=operator_id)
        self.sentinel_path = str(sentinel_path)
        self.fail_unit = int(fail_unit)

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode, route) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(self, *, sources, request):
        return {
            "output": OutputSpec(
                artifact_type="table",
                lineage_mode="preserved_key",
                basis_labels="source",
            )
        }

    def input_request(self, *, sources, mode, request):
        _ = sources, mode
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            batch_size=request.batch_size or 2,
            form="table",
            include_position=True,
            columns=__import__(
                "text_analysis_lab.core.operator", fromlist=["ColumnRequest"]
            ).ColumnRequest(keys=True, data=False, metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        batch = inputs["source"]
        if (
            batch.batch_index == self.fail_unit
            and not Path(self.sentinel_path).exists()
        ):
            Path(self.sentinel_path).write_text("failed-once", encoding="utf-8")
            raise RuntimeError(f"transient unit {self.fail_unit}")
        return BatchResult(
            outputs={"output": {"keys": batch.data.loc[:, ["id"]].copy()}}
        )

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def make_translate_worker(self, *, mode, request):
        _ = mode, request
        return self.__class__(self.sentinel_path, self.fail_unit)

    def to_json_state(self):
        return {"sentinel_path": self.sentinel_path, "fail_unit": self.fail_unit}

    @classmethod
    def from_json_state(cls, state):
        return cls(str(state["sentinel_path"]), int(state["fail_unit"]))

    def save_intermediate_state(self, intermediate_dir, *, operator_id, mode, route):
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "state.json").write_text(
            __import__("json").dumps(self.to_json_state()), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(cls, intermediate_dir, *, operator_id, mode, route):
        _ = mode, route
        state = __import__("json").loads(
            (Path(intermediate_dir) / "state.json").read_text()
        )
        obj = cls.from_json_state(state)
        obj.operator_id = operator_id
        return obj


def test_successful_resumable_operation_removes_checkpoint_state(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(4))})
    _install_fake_source_queries(monkeypatch, rows)
    project = teal.Project.create(tmp_path / "project", name="cleanup_project")
    try:
        source = _seed_source_artifact(project, rows)
        translator = _TransientResumableTranslator(
            str(tmp_path / "never-fail"), fail_unit=999
        )
        outputs = project.translate(translator, source, batch_size=2, workers=1)
        assert outputs["output"].status == "complete"

        operation_id = str(project.catalog.list_operations()[-1]["operation_id"])
        assert project.catalog.get_operation(operation_id)["status"] == "complete"
        temp = project.storage.operation_temp_dir(operation_id)
        assert not temp.exists()
        assert not temp.with_name("temp.__staging__").exists()
        assert not temp.with_name("temp.__previous__").exists()
    finally:
        project.close()


def test_sequential_resume_retries_first_failed_unit_without_redoing_completed_work(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(6))})
    _install_fake_source_queries(monkeypatch, rows)
    project = teal.Project.create(tmp_path / "project", name="resume_project")
    try:
        source = _seed_source_artifact(project, rows)
        translator = _TransientResumableTranslator(
            str(tmp_path / "fail-once"), fail_unit=0
        )
        with pytest.raises(RuntimeError, match="transient unit 0"):
            project.translate(translator, source, batch_size=2, workers=1)

        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["status"] == "failed"
        assert project.storage.operation_temp_dir(operation_id).exists()

        outputs = project.resume_operation(operation_id)
        output = outputs["output"]
        assert output.status == "complete"
        assert output.n_rows == 6
        assert project.catalog.get_operation(operation_id)["status"] == "complete"
        assert not project.storage.operation_temp_dir(operation_id).exists()
        assert (
            not project.storage.operation_temp_dir(operation_id)
            .with_name("temp.__staging__")
            .exists()
        )
        assert (
            not project.storage.operation_temp_dir(operation_id)
            .with_name("temp.__previous__")
            .exists()
        )
    finally:
        project.close()


def test_parallel_resume_preserves_canonical_output_order_and_completed_units(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal
    import text_analysis_lab.core.translate as translate_module

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(10))})
    _install_fake_source_queries(monkeypatch, rows)

    def open_client(*, workers: int):
        return _ProcessClient(workers), _ProcessCluster(), _ProcessAsCompleted

    monkeypatch.setattr(translate_module, "_open_dask_client", open_client)

    project = teal.Project.create(tmp_path / "project", name="parallel_resume_project")
    try:
        source = _seed_source_artifact(project, rows)
        translator = _TransientResumableTranslator(
            str(tmp_path / "parallel-fail-once"), fail_unit=2
        )
        with pytest.raises(RuntimeError, match="transient unit 2"):
            project.translate(
                translator,
                source,
                batch_size=2,
                workers=3,
                max_outstanding_units=3,
            )

        operation_id = str(project.catalog.list_operations()[-1]["operation_id"])
        plan = _ExecutionPlan(project.storage.operation_dir(operation_id))
        try:
            completed_before_resume = plan.complete_count()
            assert completed_before_resume == 2
        finally:
            plan.close()

        outputs = project.resume_operation(operation_id)
        output = outputs["output"]
        assert output.status == "complete"
        assert output.n_rows == 10

        # Writer part order must reflect canonical unit order despite worker completion order.
        key_frames = [
            pd.read_parquet(output.keys_dir / f"part-{index:06d}.parquet")
            for index in range(5)
        ]
        assert pd.concat(key_frames, ignore_index=True)["id"].tolist() == list(
            range(10)
        )
    finally:
        project.close()


def keep_even_rows(packet):
    return packet["id"] % 2 == 0


def _install_fake_query_columns(monkeypatch) -> None:
    from text_analysis_lab.core.query import QueryEngine

    def fake_query_columns(self, artifact, *, metadata_mode="full"):
        _ = self, artifact, metadata_mode
        return {
            "output": ("id", "group"),
            "ambiguous": {},
            "columns": (
                {"output_name": "id", "namespace": "key"},
                {"output_name": "group", "namespace": "data"},
            ),
        }

    monkeypatch.setattr(QueryEngine, "query_columns", fake_query_columns, raising=True)


def _read_all_key_ids(artifact) -> list[int]:
    parts = int(artifact.descriptor["write"]["parts"])
    frames = [
        pd.read_parquet(artifact.keys_dir / f"part-{index:06d}.parquet")
        for index in range(parts)
    ]
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True)["id"].astype(int).tolist()


def test_function_subset_parallel_acceptance_preserves_canonical_selected_keys(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal
    import text_analysis_lab.core.translate as translate_module

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(12)), "group": [x % 2 for x in range(12)]})
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)

    def open_client(*, workers: int):
        return _ProcessClient(workers), _ProcessCluster(), _ProcessAsCompleted

    monkeypatch.setattr(translate_module, "_open_dask_client", open_client)

    project = teal.Project.create(tmp_path / "project", name="subset_parallel_project")
    try:
        source = _seed_source_artifact(project, rows)
        outputs = project.subset(
            source,
            keep_even_rows,
            data_columns=False,
            batch_size=3,
            workers=3,
            output_label="evens",
        )
        output = outputs["evens"]
        assert output.status == "complete"
        assert output.label == "evens"
        assert output.primary_key == ["id"]
        assert _read_all_key_ids(output) == [0, 2, 4, 6, 8, 10]
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.descriptor["lineage"]["basis_artifact_ids"] == [
            source.artifact_id
        ]
    finally:
        project.close()


def test_random_split_acceptance_is_reproducible_stratified_and_exhaustive(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame(
        {
            "id": list(range(12)),
            "group": [0] * 6 + [1] * 6,
        }
    )
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)

    project = teal.Project.create(tmp_path / "project", name="split_project")
    try:
        source = _seed_source_artifact(project, rows)
        outputs = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.5, 0.5),
            random_state=19,
            stratify="group",
            workers=1,
        )
        learn = _read_all_key_ids(outputs["learn"])
        audit = _read_all_key_ids(outputs["audit"])

        assert len(learn) == 6
        assert len(audit) == 6
        assert set(learn).isdisjoint(audit)
        assert set(learn) | set(audit) == set(range(12))
        assert sum(value < 6 for value in learn) == 3
        assert sum(value >= 6 for value in learn) == 3
        assert sum(value < 6 for value in audit) == 3
        assert sum(value >= 6 for value in audit) == 3

        # Same seed and source should produce the same key memberships.
        outputs2 = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.5, 0.5),
            random_state=19,
            stratify="group",
            workers=1,
        )
        assert _read_all_key_ids(outputs2["learn"]) == learn
        assert _read_all_key_ids(outputs2["audit"]) == audit
    finally:
        project.close()


def transient_keep_even_rows(packet):
    import os

    sentinel = Path(os.environ["TEAL_SUBSET_SENTINEL"])
    # Fail once on the second planned batch.
    if int(packet["id"].iloc[0]) == 3 and not sentinel.exists():
        sentinel.write_text("failed-once", encoding="utf-8")
        raise RuntimeError("transient subset failure")
    return packet["id"] % 2 == 0


def test_parallel_subset_can_resume_from_last_completed_unit(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal
    import text_analysis_lab.core.translate as translate_module

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(12)), "group": [x % 2 for x in range(12)]})
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)
    sentinel = tmp_path / "subset-fail-once"
    monkeypatch.setenv("TEAL_SUBSET_SENTINEL", str(sentinel))

    def open_client(*, workers: int):
        return _ProcessClient(workers), _ProcessCluster(), _ProcessAsCompleted

    monkeypatch.setattr(translate_module, "_open_dask_client", open_client)

    project = teal.Project.create(tmp_path / "project", name="subset_resume_project")
    try:
        source = _seed_source_artifact(project, rows)
        with pytest.raises(RuntimeError, match="transient subset failure"):
            project.subset(
                source,
                transient_keep_even_rows,
                data_columns=False,
                batch_size=3,
                workers=3,
                output_label="evens",
            )

        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["operation_type"] == "subset"
        assert operation["status"] == "failed"

        outputs = project.resume_operation(operation_id)
        assert outputs["evens"].status == "complete"
        assert _read_all_key_ids(outputs["evens"]) == [0, 2, 4, 6, 8, 10]
    finally:
        project.close()


def test_file_backed_subset_operator_uses_frozen_asset_after_source_is_deleted(
    tmp_path: Path,
) -> None:
    from text_analysis_lab.core.operator import BaseOperator
    from text_analysis_lab.core.subset import FunctionSubsetTranslator

    source_file = tmp_path / "subset_rule.py"
    source_file.write_text(
        "def keep_even(packet):\n    return packet['id'] % 2 == 0\n",
        encoding="utf-8",
    )
    operator_dir = tmp_path / "operator"
    translator = FunctionSubsetTranslator((source_file, "keep_even"))
    translator.save_to_dir(operator_dir, operator_id="op_000001")

    source_file.unlink()
    loaded = BaseOperator.load_from_dir(operator_dir)
    assert isinstance(loaded, FunctionSubsetTranslator)
    mask = loaded.function(pd.DataFrame({"id": [0, 1, 2, 3]}))
    assert list(mask) == [True, False, True, False]


def test_file_backed_subset_resume_state_is_self_contained(tmp_path: Path) -> None:
    from text_analysis_lab.core.subset import FunctionSubsetTranslator

    source_file = tmp_path / "subset_rule.py"
    source_file.write_text(
        "def keep_even(packet):\n    return packet['id'] % 2 == 0\n",
        encoding="utf-8",
    )
    intermediate = tmp_path / "intermediate"
    translator = FunctionSubsetTranslator((source_file, "keep_even"))
    translator.save_intermediate_state(
        intermediate,
        operator_id="op_000001",
        mode="translate",
        route="parallel",
    )

    source_file.unlink()
    loaded = FunctionSubsetTranslator.load_intermediate_state(
        intermediate,
        operator_id="op_000001",
        mode="translate",
        route="parallel",
    )
    mask = loaded.function(pd.DataFrame({"id": [0, 1, 2, 3]}))
    assert list(mask) == [True, False, True, False]


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_parallel_backpressure_limit_is_respected_under_out_of_order_completion(
    tmp_path: Path, monkeypatch, limit: int
) -> None:
    import text_analysis_lab.core.translate as translate_module

    n_units = 20
    plan = _ExecutionPlan(tmp_path / f"operation-{limit}")
    plan.initialize(_plan_units(n_units))
    client_holder: dict[str, _ProcessClient] = {}

    def open_client(*, workers: int):
        client = _ProcessClient(workers)
        client_holder["client"] = client
        return client, _ProcessCluster(), _ProcessAsCompleted

    monkeypatch.setattr(translate_module, "_open_dask_client", open_client)
    monkeypatch.setattr(translate_module, "_materialize_plan_unit", _fake_materialize)
    request = TranslationRequest(
        workers=4,
        batch_size=1,
        max_outstanding_units=limit,
    )
    try:
        results = list(
            _parallel_results(
                translate_worker=_DelayTranslator(),
                plan=plan,
                sources={"source": _FakeArtifact()},
                input_request={
                    "source": SourceRequest(artifact_type="table", batch_size=1)
                },
                mode="translate",
                request=request,
                workers=4,
            )
        )
    finally:
        plan.close()

    assert [index for index, _ in results] == list(range(n_units))
    assert client_holder["client"].max_in_flight <= limit


def test_execution_plan_recovery_state_survives_close_and_reopen(
    tmp_path: Path,
) -> None:
    operation_dir = tmp_path / "operation"
    plan = _ExecutionPlan(operation_dir)
    plan.initialize(_plan_units(4))
    plan.mark_complete(0)
    plan.mark_failed(1, "transient")
    plan.mark_running(2)
    plan.close()

    reopened = _ExecutionPlan(operation_dir)
    try:
        reopened.reset_running_to_pending()
        assert reopened.complete_count() == 1
        assert [unit.unit_index for unit in reopened.iter_pending()] == [1, 2, 3]
    finally:
        reopened.close()


def test_split_ignores_parallel_workers_for_full_artifact_source(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(8)), "group": [0, 0, 0, 0, 1, 1, 1, 1]})
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)

    project = teal.Project.create(tmp_path / "project", name="split_workers_project")
    try:
        source = _seed_source_artifact(project, rows)
        with pytest.warns(UserWarning, match="workers was ignored"):
            outputs = project.split(
                source,
                proportions=(0.5, 0.5),
                random_state=4,
                workers=4,
            )
        assert sum(output.n_rows for output in outputs.values()) == len(rows)
    finally:
        project.close()


def test_file_backed_subset_parallel_worker_uses_frozen_copy(tmp_path: Path) -> None:
    from text_analysis_lab.core.operator import TranslationRequest
    from text_analysis_lab.core.subset import FunctionSubsetTranslator

    source_file = tmp_path / "subset_rule.py"
    source_file.write_text(
        "def select(packet):\n    return packet['id'] % 2 == 0\n",
        encoding="utf-8",
    )
    translator = FunctionSubsetTranslator((source_file, "select"))
    translator.save_to_dir(tmp_path / "operator", operator_id="op_000001")

    # Mutate the external source after the operator has been frozen.
    source_file.write_text(
        "def select(packet):\n    return packet['id'] % 2 == 1\n",
        encoding="utf-8",
    )
    worker = translator.make_translate_worker(
        mode="translate",
        request=TranslationRequest(),
    )
    mask = worker.function(pd.DataFrame({"id": [0, 1, 2, 3]}))
    assert list(mask) == [True, False, True, False]


class _MultiTransientTranslator(_TransientResumableTranslator):
    def __init__(self, sentinel_dir: str, fail_units, *, operator_id=None) -> None:
        BaseTranslator.__init__(self, operator_id=operator_id)
        self.sentinel_dir = str(sentinel_dir)
        self.fail_units = tuple(int(value) for value in fail_units)
        Path(self.sentinel_dir).mkdir(parents=True, exist_ok=True)

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        batch = inputs["source"]
        unit = int(batch.batch_index)
        sentinel = Path(self.sentinel_dir) / f"unit-{unit}.failed"
        if unit in self.fail_units and not sentinel.exists():
            sentinel.write_text("failed-once", encoding="utf-8")
            raise RuntimeError(f"transient unit {unit}")
        return BatchResult(
            outputs={"output": {"keys": batch.data.loc[:, ["id"]].copy()}}
        )

    def make_translate_worker(self, *, mode, request):
        _ = mode, request
        return self.__class__(self.sentinel_dir, self.fail_units)

    def to_json_state(self):
        return {"sentinel_dir": self.sentinel_dir, "fail_units": list(self.fail_units)}

    @classmethod
    def from_json_state(cls, state):
        return cls(str(state["sentinel_dir"]), state["fail_units"])


def test_operation_can_fail_resume_fail_again_and_resume_again(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(10))})
    _install_fake_source_queries(monkeypatch, rows)
    project = teal.Project.create(tmp_path / "project", name="multi_resume_project")
    try:
        source = _seed_source_artifact(project, rows)
        translator = _MultiTransientTranslator(
            str(tmp_path / "sentinels"), fail_units=(1, 3)
        )
        with pytest.raises(RuntimeError, match="transient unit 1"):
            project.translate(translator, source, batch_size=2, workers=1)
        operation_id = str(project.catalog.list_operations()[-1]["operation_id"])

        with pytest.raises(RuntimeError, match="transient unit 3"):
            project.resume_operation(operation_id)
        assert project.catalog.get_operation(operation_id)["status"] == "failed"

        outputs = project.resume_operation(operation_id)
        assert outputs["output"].status == "complete"
        assert outputs["output"].n_rows == 10
        assert _read_all_key_ids(outputs["output"]) == list(range(10))
    finally:
        project.close()


def keep_ids_at_least_three(packet):
    return packet["id"] >= 3


def test_subset_skips_empty_selected_batches_without_failing_writer(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": list(range(9)), "group": [0] * 9})
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)
    project = teal.Project.create(
        tmp_path / "project", name="subset_empty_batch_project"
    )
    try:
        source = _seed_source_artifact(project, rows)
        outputs = project.subset(
            source,
            keep_ids_at_least_three,
            data_columns=False,
            batch_size=3,
            workers=1,
            output_label="kept",
        )
        assert _read_all_key_ids(outputs["kept"]) == [3, 4, 5, 6, 7, 8]
    finally:
        project.close()


def test_subset_key_columns_false_hides_keys_from_predicate_but_preserves_output(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    _install_pickle_parquet(monkeypatch)
    rows = pd.DataFrame({"id": [0, 1, 2, 3], "group": [0, 1, 0, 1]})
    _install_fake_source_queries(monkeypatch, rows)
    _install_fake_query_columns(monkeypatch)
    project = teal.Project.create(tmp_path / "project", name="subset_hidden_keys")
    try:
        source = _seed_source_artifact(project, rows)

        def keep_group_one(frame):
            assert "id" not in frame.columns
            assert "group" in frame.columns
            return frame["group"] == 1

        output = project.subset(
            source,
            keep_group_one,
            key_columns=False,
            data_columns=["group"],
            include_position=False,
            output_label="kept",
        )["kept"]

        assert _read_all_key_ids(output) == [1, 3]
    finally:
        project.close()


def test_subset_cloudpickle_freezes_captured_state(tmp_path: Path) -> None:
    from text_analysis_lab.core.operator import BaseOperator
    from text_analysis_lab.core.subset import FunctionSubsetTranslator

    threshold = 10
    rule = lambda frame: frame["value"] >= threshold
    translator = FunctionSubsetTranslator(rule)
    operator_dir = tmp_path / "operator"
    translator.save_to_dir(operator_dir, operator_id="op_000001")

    # Mutating the surrounding closure after freeze changes the original callable,
    # but neither the live frozen operator nor a reopened operator may change.
    threshold = 100
    frame = pd.DataFrame({"value": [5, 50, 150]})
    assert list(rule(frame)) == [False, False, True]
    assert list(translator.function(frame)) == [False, True, True]

    loaded = BaseOperator.load_from_dir(operator_dir)
    assert isinstance(loaded, FunctionSubsetTranslator)
    assert list(loaded.function(frame)) == [False, True, True]


def test_function_mapper_cloudpickle_freezes_captured_state(tmp_path: Path) -> None:
    from text_analysis_lab.core.operator import BaseOperator
    from text_analysis_lab.translators import FunctionMapper

    offset = 10
    mapper_fn = lambda packet: {
        "data": pd.DataFrame({"value": packet["data"]["value"] + offset})
    }
    mapper = FunctionMapper(mapper_fn)
    operator_dir = tmp_path / "mapper"
    mapper.save_to_dir(operator_dir, operator_id="op_000002")

    offset = 100
    packet = {"data": pd.DataFrame({"value": [1, 2]})}
    assert mapper_fn(packet)["data"]["value"].tolist() == [101, 102]
    assert mapper.function(packet)["data"]["value"].tolist() == [11, 12]

    loaded = BaseOperator.load_from_dir(operator_dir)
    assert isinstance(loaded, FunctionMapper)
    assert loaded.function(packet)["data"]["value"].tolist() == [11, 12]


def test_checkpoint_swap_retries_transient_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    from text_analysis_lab.core import translate

    target = tmp_path / "temp"
    target.mkdir()
    (target / "state.txt").write_text("old", encoding="utf-8")
    real_rename = Path.rename
    failures = {"remaining": 1}

    def flaky_rename(self: Path, destination: Path):
        if self.name.endswith(".__staging__") and failures["remaining"]:
            failures["remaining"] -= 1
            raise PermissionError("transient Windows sharing violation")
        return real_rename(self, destination)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr(translate.time, "sleep", lambda _seconds: None)

    _replace_directory_with_staging(
        target,
        lambda staging: (staging / "state.txt").write_text("new", encoding="utf-8"),
    )

    assert (target / "state.txt").read_text(encoding="utf-8") == "new"
    assert not target.with_name("temp.__staging__").exists()
    assert not target.with_name("temp.__previous__").exists()


def test_checkpoint_swap_persistent_promotion_failure_restores_previous(
    tmp_path: Path, monkeypatch
) -> None:
    from text_analysis_lab.core import translate

    target = tmp_path / "temp"
    target.mkdir()
    (target / "state.txt").write_text("old", encoding="utf-8")
    real_rename = Path.rename

    def blocked_promotion(self: Path, destination: Path):
        if self.name.endswith(".__staging__"):
            raise PermissionError("persistent Windows sharing violation")
        return real_rename(self, destination)

    monkeypatch.setattr(Path, "rename", blocked_promotion)
    monkeypatch.setattr(translate.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="persistent Windows"):
        _replace_directory_with_staging(
            target,
            lambda staging: (staging / "state.txt").write_text("new", encoding="utf-8"),
        )

    assert (target / "state.txt").read_text(encoding="utf-8") == "old"
    assert not target.with_name("temp.__staging__").exists()
    assert not target.with_name("temp.__previous__").exists()


def test_checkpoint_swap_recovers_interrupted_previous_generation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "temp"
    previous = tmp_path / "temp.__previous__"
    staging = tmp_path / "temp.__staging__"
    previous.mkdir()
    (previous / "state.txt").write_text("old", encoding="utf-8")
    staging.mkdir()
    (staging / "state.txt").write_text("partial", encoding="utf-8")

    _replace_directory_with_staging(
        target,
        lambda fresh: (fresh / "state.txt").write_text("new", encoding="utf-8"),
    )

    assert (target / "state.txt").read_text(encoding="utf-8") == "new"
    assert not previous.exists()
    assert not staging.exists()


def test_checkpoint_swap_discards_stale_previous_when_target_is_complete(
    tmp_path: Path,
) -> None:
    target = tmp_path / "temp"
    previous = tmp_path / "temp.__previous__"
    target.mkdir()
    previous.mkdir()
    (target / "state.txt").write_text("current", encoding="utf-8")
    (previous / "state.txt").write_text("older", encoding="utf-8")

    _replace_directory_with_staging(
        target,
        lambda fresh: (fresh / "state.txt").write_text("new", encoding="utf-8"),
    )

    assert (target / "state.txt").read_text(encoding="utf-8") == "new"
    assert not previous.exists()
