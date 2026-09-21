from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from text_analysis_lab.core import importers

openpyxl = pytest.importorskip("openpyxl")


def _workbook(path: Path) -> None:
    wb = openpyxl.Workbook()
    first = wb.active
    first.title = "First"
    first.append(["speaker", "content", "notes"])
    first.append(["child", "one", "n1"])
    first.append([None, None, None])
    first.append(["parent", "two", None])
    second = wb.create_sheet("Hidden")
    second.append(["speaker", "content"])
    second.append(["child", "three"])
    second.sheet_state = "hidden"
    wb.save(path)
    wb.close()


def test_read_excel_streams_selected_sheets_to_writer_shaped_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.xlsx"
    _workbook(source)
    captured: dict[str, object] = {}

    def fake_execute(project, **kwargs):
        captured.update(kwargs)
        captured["payloads"] = list(kwargs["payloads"])
        return "artifact"

    monkeypatch.setattr(importers, "_execute_import", fake_execute)
    result = importers.read_excel(
        object(),
        source,
        sheets=None,
        text_fields="content",
        metadata_fields=["speaker", "notes"],
        missing_fields=None,
        batch_size=2,
    )
    assert result == "artifact"
    assert captured["kind"] == "read_excel"
    payloads = captured["payloads"]
    assert len(payloads) == 2
    keys = pd.concat([payload["keys"] for payload in payloads], ignore_index=True)
    data = pd.concat([payload["data"] for payload in payloads], ignore_index=True)
    metadata = pd.concat(
        [payload["metadata"] for payload in payloads], ignore_index=True
    )
    assert keys["row_id"].tolist() == [0, 1, 2]
    assert data["content"].tolist() == ["one", "two", "three"]
    assert metadata["source_sheet"].tolist() == ["First", "First", "Hidden"]
    assert metadata["source_sheet_index"].tolist() == [0, 0, 1]
    assert metadata["source_row"].tolist() == [0, 1, 0]
    assert pd.isna(metadata.loc[2, "notes"])
    assert captured["request"]["formula_policy"] == "cached_values"


def test_read_excel_missing_field_is_preflight_error_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.xlsx"
    _workbook(source)
    called = False

    def fake_execute(project, **kwargs):
        nonlocal called
        called = True
        return "artifact"

    monkeypatch.setattr(importers, "_execute_import", fake_execute)
    with pytest.raises(Exception, match="missing_fields=None"):
        importers.read_excel(
            object(),
            source,
            sheets=None,
            text_fields="content",
            metadata_fields=["speaker", "notes"],
        )
    assert called is False


def test_read_excel_dtype_string_coerces_excel_time_cells_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "times.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["start", "speaker", "content", "code"])
    ws.append([time(0, 0, 5), "child", "hello", 7])
    ws.append([time(0, 1, 2), "parent", "hi", None])
    wb.save(source)
    wb.close()

    captured: dict[str, object] = {}

    def fake_execute(project, **kwargs):
        captured.update(kwargs)
        captured["payloads"] = list(kwargs["payloads"])
        return "artifact"

    monkeypatch.setattr(importers, "_execute_import", fake_execute)
    result = importers.read_excel(
        object(),
        source,
        text_fields="content",
        metadata_fields=["start", "speaker", "code"],
        dtype="string",
    )

    assert result == "artifact"
    payload = captured["payloads"][0]
    data = payload["data"]
    metadata = payload["metadata"]

    assert data["content"].tolist() == ["hello", "hi"]
    assert metadata["start"].tolist() == ["00:00:05", "00:01:02"]
    assert metadata["speaker"].tolist() == ["child", "parent"]
    assert metadata["code"].tolist()[0] == "7"
    assert pd.isna(metadata["code"].tolist()[1])
    assert captured["request"]["dtype"] == {
        "content": "string",
        "start": "string",
        "speaker": "string",
        "code": "string",
    }


def test_read_excel_dtype_mapping_only_coerces_named_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mixed.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["content", "year", "score"])
    ws.append(["one", "2020", "1.5"])
    ws.append(["two", None, "2.0"])
    wb.save(source)
    wb.close()

    captured: dict[str, object] = {}

    def fake_execute(project, **kwargs):
        captured.update(kwargs)
        captured["payloads"] = list(kwargs["payloads"])
        return "artifact"

    monkeypatch.setattr(importers, "_execute_import", fake_execute)
    importers.read_excel(
        object(),
        source,
        text_fields="content",
        metadata_fields=["year", "score"],
        dtype={"year": "Int64", "score": "Float64"},
    )

    payload = captured["payloads"][0]
    metadata = payload["metadata"]
    assert str(metadata["year"].dtype) == "Int64"
    assert str(metadata["score"].dtype) == "Float64"
    assert metadata["year"].tolist()[0] == 2020
    assert pd.isna(metadata["year"].tolist()[1])
    assert metadata["score"].tolist() == [1.5, 2.0]
    assert captured["request"]["dtype"] == {
        "year": "Int64",
        "score": "Float64",
    }


def test_read_excel_folder_dtype_string_handles_time_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "one.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["start", "speaker", "content"])
    ws.append([time(0, 0, 5), "child", "hello"])
    wb.save(source)
    wb.close()

    captured: dict[str, object] = {}

    def fake_execute(project, **kwargs):
        captured.update(kwargs)
        captured["payloads"] = list(kwargs["payloads"])
        return "artifact"

    monkeypatch.setattr(importers, "_execute_import", fake_execute)
    result = importers.read_excel_folder(
        object(),
        tmp_path,
        text_fields="content",
        metadata_fields=["start", "speaker"],
        dtype="string",
        recursive=False,
    )

    assert result == "artifact"
    assert captured["kind"] == "read_excel_folder"
    payload = captured["payloads"][0]
    assert payload["metadata"]["start"].tolist() == ["00:00:05"]
    assert payload["metadata"]["source_file"].tolist() == ["one.xlsx"]
