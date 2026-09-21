from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
openpyxl = pytest.importorskip("openpyxl")

import text_analysis_lab as teal


def test_project_read_excel_folder_dtype_string_materializes_time_cells(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    source = source_dir / "one.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["start", "end", "speaker", "content", "code"])
    ws.append([time(0, 0, 5), time(0, 0, 7), "child", "hello", 7])
    ws.append([time(0, 1, 2), time(0, 1, 4), "parent", "hi", None])
    wb.save(source)
    wb.close()

    project = teal.Project.create(tmp_path / "project", name="excel_dtype")
    try:
        artifact = project.read_excel_folder(
            source_dir,
            text_fields="content",
            metadata_fields=["start", "end", "speaker", "code"],
            dtype="string",
            recursive=False,
        )

        frame = artifact.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="full",
            order_by="_position",
            form="table",
        )

        assert frame["content"].tolist() == ["hello", "hi"]
        assert frame["start"].tolist() == ["00:00:05", "00:01:02"]
        assert frame["end"].tolist() == ["00:00:07", "00:01:04"]
        assert frame["speaker"].tolist() == ["child", "parent"]
        assert frame["code"].tolist()[0] == "7"
        assert pd.isna(frame["code"].tolist()[1])
        assert frame["source_file"].tolist() == ["one.xlsx", "one.xlsx"]

        operation = project.operation_for_artifact(artifact)
        assert operation is not None
        payload = project.get_operation(operation["operation_id"])
        assert payload["status"] == "complete"
    finally:
        project.close()
