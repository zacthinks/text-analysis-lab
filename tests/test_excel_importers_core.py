from __future__ import annotations

from pathlib import Path

import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.importers import (
    ImportOperator,
    _normalize_sheet_selectors,
    _resolve_excel_sheets,
    _validate_missing_fields_policy,
)


def test_excel_import_kinds_round_trip() -> None:
    for kind in ("read_excel", "read_excel_folder"):
        operator = ImportOperator(kind)
        state = operator.to_json_state()
        assert state == {"kind": kind}
        assert ImportOperator.from_json_state(state).kind == kind


def test_excel_sheet_selector_normalization_and_resolution() -> None:
    assert _normalize_sheet_selectors(None) is None
    assert _normalize_sheet_selectors("Beta") == ("Beta",)
    assert _normalize_sheet_selectors(1) == (1,)
    assert _normalize_sheet_selectors(["Beta", 0]) == ("Beta", 0)

    resolved = _resolve_excel_sheets(
        ["Alpha", "Beta", "Hidden"],
        ("Beta", 0),
        source_path=Path("book.xlsx"),
    )
    assert resolved == ((1, "Beta"), (0, "Alpha"))

    with pytest.raises(ArtifactError, match="duplicate sheet"):
        _resolve_excel_sheets(
            ["Alpha", "Beta"],
            (0, "Alpha"),
            source_path=Path("book.xlsx"),
        )


def test_missing_fields_policy_distinguishes_error_from_null_fill() -> None:
    import text_analysis_lab.core.importers as importers

    assert _validate_missing_fields_policy(importers._MISSING_FIELDS_ERROR) == {
        "mode": "error"
    }
    assert _validate_missing_fields_policy(None) == {"mode": "fill", "value": None}
    assert _validate_missing_fields_policy(0) == {"mode": "fill", "value": 0}
    with pytest.raises(TypeError, match="scalar"):
        _validate_missing_fields_policy({"not": "scalar"})
