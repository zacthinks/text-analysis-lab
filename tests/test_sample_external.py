from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal


def _documents(project: teal.Project, tmp_path: Path):
    path = tmp_path / "docs.csv"
    pd.DataFrame(
        {
            "text": [f"document {i}" for i in range(12)],
            "group": [i % 3 for i in range(12)],
        }
    ).to_csv(path, index=False)
    return project.read_csv(
        path,
        text_fields="text",
        metadata_fields="group",
        alias="docs",
    )


def _rows(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=True,
        metadata_mode="full",
        order_by="_position",
        form="table",
    )


def test_sample_is_exact_reproducible_preserved_key_and_source_order(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="sample")
    try:
        docs = _documents(project, tmp_path)
        first = project.sample(docs, n=5, random_state=17, output_label="sample_a")
        second = project.sample(docs, n=5, random_state=17, output_label="sample_b")

        first_artifact_id = first.artifact_id
        first_rows = _rows(first)
        second_rows = _rows(second)
        first_keys = first_rows["row_id"].astype(int).tolist()
        second_keys = second_rows["row_id"].astype(int).tolist()

        assert len(first_rows) == 5
        assert first_keys == second_keys
        assert first_keys == sorted(first_keys)
        assert first.descriptor["lineage"] == {
            "lineage_mode": "preserved_key",
            "basis_artifact_ids": [docs.artifact_id],
        }
        assert set(first.components) == {"keys"}
        assert first_rows["text"].tolist() == [f"document {i}" for i in first_keys]
        assert first_rows["group"].astype(int).tolist() == [i % 3 for i in first_keys]

        operation = project.catalog.get_operation(first.operation_id)
        op_path = project.storage.operation_dir(first.operation_id) / "operation.json"
        descriptor = json.loads(op_path.read_text(encoding="utf-8"))
        assert operation["operation_type"] == "subset"
        assert descriptor["kind"] == "sample"
        assert descriptor["request"]["n"] == 5
        assert descriptor["request"]["random_state"] == 17
        assert descriptor["request"]["replace"] is False
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        first = reopened.get_artifact(first_artifact_id)
        rows = _rows(first)
        assert len(rows) == 5
        assert rows["row_id"].astype(int).tolist() == sorted(
            rows["row_id"].astype(int).tolist()
        )
    finally:
        reopened.close()


def test_sample_validates_size_and_alias_reuse(tmp_path: Path, capsys) -> None:
    project = teal.Project.create(tmp_path / "project", name="sample_alias")
    try:
        docs = _documents(project, tmp_path)
        with pytest.raises(ValueError, match="positive"):
            project.sample(docs, n=0)
        with pytest.raises(ValueError, match="exceeds source row count"):
            project.sample(docs, n=13)

        first = project.sample(docs, n=3, random_state=1, alias="demo")
        before = len(project.list_operations())
        reused = project.sample(docs, n=5, random_state=999, alias="demo")
        assert reused.artifact_id == first.artifact_id
        assert len(project.list_operations()) == before
        assert "current settings were ignored" in capsys.readouterr().out

        rebuilt = project.sample(
            docs,
            n=4,
            random_state=2,
            alias="demo",
            overwrite=True,
        )
        assert rebuilt.artifact_id != first.artifact_id
        assert len(_rows(rebuilt)) == 4
        assert project.get_artifact("demo").artifact_id == rebuilt.artifact_id
    finally:
        project.close()
