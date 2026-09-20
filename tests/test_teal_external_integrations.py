from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# These tests are intentionally for an environment with the real optional/runtime
# integrations installed. They should skip rather than fail if copied into a
# lightweight environment.
pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("dask.distributed")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer


def _seed_real_table(project: teal.Project, rows: pd.DataFrame):
    """Create one real multi-part Parquet table artifact and register it."""
    artifact_id = "art_source"
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="source",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )

    # Multiple writes matter: this exercises true partitioned Parquet output and
    # the artifact-wide final seal rather than a single in-memory frame.
    for start in range(0, len(rows), 5):
        batch = rows.iloc[start : start + 5].reset_index(drop=True)
        writer.write(
            {
                "keys": batch[["id"]],
                "data": batch[["group", "text"]],
            }
        )
    writer.finalize()

    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label="source",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def _ids(artifact) -> list[int]:
    frame = artifact.query(
        key_columns=True,
        data_columns=False,
        form="table",
        include_position=True,
        order_by="_position",
    )
    return frame["id"].astype(int).tolist()


def _write_rule(path: Path, body: str) -> tuple[Path, str]:
    path.write_text(body, encoding="utf-8")
    return path, "keep"


def test_real_pyarrow_duckdb_query_and_arrow_streaming(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "id": list(range(23)),
            "group": [x % 3 for x in range(23)],
            "text": [f"row-{x}" for x in range(23)],
        }
    )
    project = teal.Project.create(tmp_path / "project", name="external_query_test")
    try:
        source = _seed_real_table(project, rows)

        materialized = source.query(
            key_columns=True,
            data_columns=True,
            form="table",
            include_position=True,
            order_by="_position",
        )
        assert materialized["id"].astype(int).tolist() == list(range(23))
        assert materialized["group"].astype(int).tolist() == rows["group"].tolist()
        assert materialized["text"].tolist() == rows["text"].tolist()
        assert materialized["_position"].astype(int).tolist() == list(range(23))

        # Force the Arrow reader path rather than allowing the paged fallback.
        batches = list(
            source.iter_table_batches(
                batch_size=7,
                key_columns=True,
                data_columns=True,
                include_position=True,
                order_by="_position",
                streaming_mode="arrow",
            )
        )
        assert [len(batch) for batch in batches] == [7, 7, 7, 2]
        streamed = pd.concat(batches, ignore_index=True)
        assert streamed["id"].astype(int).tolist() == list(range(23))
        assert streamed["_position"].astype(int).tolist() == list(range(23))

        # Also force the bounded paged fallback so both external query paths are
        # checked in the same environment.
        paged = list(
            source.iter_table_batches(
                batch_size=6,
                key_columns=True,
                data_columns=True,
                include_position=True,
                order_by="_position",
                streaming_mode="paged",
            )
        )
        assert [len(batch) for batch in paged] == [6, 6, 6, 5]
        assert pd.concat(paged, ignore_index=True)["id"].astype(int).tolist() == list(
            range(23)
        )
    finally:
        project.close()


def test_real_dask_parallel_subset_preserves_canonical_order(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "id": list(range(24)),
            "group": [x % 2 for x in range(24)],
            "text": [f"row-{x}" for x in range(24)],
        }
    )
    project = teal.Project.create(tmp_path / "project", name="external_dask_subset")
    try:
        source = _seed_real_table(project, rows)
        rule = _write_rule(
            tmp_path / "keep_even.py",
            "def keep(packet):\n    return packet['id'] % 2 == 0\n",
        )

        output = project.subset(
            source,
            rule,
            data_columns=False,
            batch_size=4,
            workers=3,
            output_label="evens",
        )["evens"]

        assert output.status == "complete"
        assert _ids(output) == list(range(0, 24, 2))
    finally:
        project.close()


def test_real_dask_parallel_subset_fail_then_resume(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "id": list(range(24)),
            "group": [x % 2 for x in range(24)],
            "text": [f"row-{x}" for x in range(24)],
        }
    )
    sentinel = tmp_path / "failed-once.txt"
    rule_path = tmp_path / "fail_once.py"
    # Use a literal shared-filesystem path so the worker process and the driver
    # agree on the same fail-once state without relying on inherited env vars.
    rule_path.write_text(
        "from pathlib import Path\n"
        f"SENTINEL = Path({str(sentinel)!r})\n"
        "def keep(packet):\n"
        "    first = int(packet['id'].iloc[0])\n"
        "    if first == 8 and not SENTINEL.exists():\n"
        "        SENTINEL.write_text('failed-once', encoding='utf-8')\n"
        "        raise RuntimeError('intentional external integration failure')\n"
        "    return packet['id'] % 2 == 0\n",
        encoding="utf-8",
    )

    project = teal.Project.create(tmp_path / "project", name="external_dask_resume")
    try:
        source = _seed_real_table(project, rows)

        with pytest.raises(
            RuntimeError, match="intentional external integration failure"
        ):
            project.subset(
                source,
                (rule_path, "keep"),
                data_columns=False,
                batch_size=4,
                workers=3,
                output_label="evens",
            )

        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["operation_type"] == "subset"
        assert operation["status"] == "failed"
        assert sentinel.exists()

        # Close/reopen before resume so this exercises durable recovery, not just
        # same-process object state.
        project_path = project.path
        project.close()
        project = teal.Project.open(project_path)

        output = project.resume_operation(operation_id)["evens"]
        assert output.status == "complete"
        assert _ids(output) == list(range(0, 24, 2))

        # Completed plan units must remain complete after recovery; nothing should
        # be left failed/running/pending once the operation is sealed.
        operation_dir = project.storage.operation_dir(operation_id)
        plan_path = operation_dir / "plan.sqlite"
        assert plan_path.exists(), f"missing durable plan: {plan_path}"
        import sqlite3

        with sqlite3.connect(plan_path) as con:
            statuses = [
                row[0]
                for row in con.execute(
                    "SELECT status FROM plan_units ORDER BY unit_index"
                ).fetchall()
            ]
        assert statuses and set(statuses) == {"complete"}
    finally:
        project.close()


def test_external_dependency_versions_are_visible() -> None:
    # This is intentionally informational in pytest -s output. It makes returned
    # logs much more useful when diagnosing environment-specific failures.
    from dask import __version__ as dask_version

    print(f"pyarrow={pyarrow.__version__}")
    print(f"duckdb={duckdb.__version__}")
    print(f"dask={dask_version}")
