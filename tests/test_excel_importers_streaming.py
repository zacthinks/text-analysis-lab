from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import text_analysis_lab.core.importers as importers

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
    metadata = pd.concat([payload["metadata"] for payload in payloads], ignore_index=True)
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
