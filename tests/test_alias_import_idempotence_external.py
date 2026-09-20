from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal


def test_read_csv_alias_makes_import_idempotent_and_ignores_changed_source_file(
    tmp_path, capsys
):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(first_path, index=False)
    pd.DataFrame({"text": ["CHANGED", "SOURCE", "ROWS"]}).to_csv(
        second_path, index=False
    )

    project = teal.Project.create(tmp_path / "project", name="import_alias")
    try:
        first = project.read_csv(
            first_path,
            text_fields="text",
            metadata_fields=None,
            alias="corpus",
        )
        before = len(project.list_operations())
        second = project.read_csv(
            second_path,
            text_fields="text",
            metadata_fields=None,
            alias="corpus",
        )
        assert second.artifact_id == first.artifact_id
        assert len(project.list_operations()) == before
        message = capsys.readouterr().out
        assert "operation was not executed" in message
        assert "current settings were ignored" in message
    finally:
        project.close()
