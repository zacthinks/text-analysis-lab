from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.importers import (
    ImportOperator,
    _coerce_import_dtypes,
    _duckdb_source_sql,
    _inventory_paths,
    _normalize_import_dtype,
    _plan_tabular_columns,
    _quote_duckdb_identifier,
    _validate_batch_size,
    _validate_duckdb_options,
    _validate_patterns,
)
from text_analysis_lab.core.operator import TranslationRequest


def test_import_operator_is_source_free_new_key_table() -> None:
    operator = ImportOperator("read_csv")
    assert operator.requires_source is False
    specs = operator.validated_output_specs(
        sources={},
        request=TranslationRequest(params={"output_label": "corpus"}),
    )
    assert set(specs) == {"corpus"}
    assert specs["corpus"].artifact_type.value == "table"
    assert specs["corpus"].lineage_mode == "new_key"
    assert specs["corpus"].basis_labels == ()
    state = operator.to_json_state()
    assert ImportOperator.from_json_state(state).kind == "read_csv"


def test_tabular_column_plan_requires_explicit_roles_and_discards_everything_else() -> (
    None
):
    plan = _plan_tabular_columns(
        ["document_id", "title", "text", "speaker", "score", "_position"],
        text_fields=["title", "text"],
        metadata_fields=["document_id", "speaker"],
    )
    assert plan == {
        "primary_key": ("row_id",),
        "text_fields": ("title", "text"),
        "metadata_fields": ("document_id", "speaker"),
        "discarded_fields": ("score", "_position"),
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text_fields": [], "metadata_fields": None}, "at least one"),
        ({"text_fields": "missing", "metadata_fields": None}, "Unknown text_fields"),
        (
            {"text_fields": "text", "metadata_fields": "missing"},
            "Unknown metadata_fields",
        ),
        (
            {"text_fields": ["text", "group"], "metadata_fields": "group"},
            "both text_fields and metadata_fields",
        ),
        (
            {"text_fields": "text", "metadata_fields": "row_id"},
            "reserves row_id",
        ),
    ],
)
def test_tabular_column_plan_rejects_invalid_role_assignments(kwargs, match) -> None:
    with pytest.raises((ArtifactError, ValueError), match=match):
        _plan_tabular_columns(
            ["row_id", "text", "group"],
            **kwargs,
        )


def test_tabular_column_plan_rejects_teal_structural_columns() -> None:
    with pytest.raises(ArtifactError, match="reserved structural"):
        _plan_tabular_columns(
            ["text", "_position"],
            text_fields="_position",
            metadata_fields=None,
        )


def test_import_dtype_normalization_supports_scalar_and_mapping() -> None:
    selected = ("text", "year", "score")

    assert _normalize_import_dtype(None, selected_fields=selected) == {}
    assert _normalize_import_dtype("string", selected_fields=selected) == {
        "text": "string",
        "year": "string",
        "score": "string",
    }
    assert _normalize_import_dtype(
        {"year": "Int64", "score": "Float64"},
        selected_fields=selected,
    ) == {"year": "Int64", "score": "Float64"}

    with pytest.raises(ArtifactError, match="unselected field"):
        _normalize_import_dtype(
            {"unused": "string"},
            selected_fields=selected,
        )
    with pytest.raises(ValueError, match="Unsupported import dtype"):
        _normalize_import_dtype(
            {"year": "definitely-not-a-dtype"},
            selected_fields=selected,
        )


def test_import_dtype_coercion_is_nullable_and_strict() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "text": ["one", None],
            "year": ["2020", None],
            "score": ["1.5", "2.0"],
        }
    )
    coerced = _coerce_import_dtypes(
        frame,
        {"text": "string", "year": "Int64", "score": "Float64"},
        context="test import",
    )

    assert str(coerced["text"].dtype) == "string"
    assert str(coerced["year"].dtype) == "Int64"
    assert str(coerced["score"].dtype) == "Float64"
    assert pd.isna(coerced.loc[1, "text"])
    assert pd.isna(coerced.loc[1, "year"])

    with pytest.raises(ArtifactError, match="Could not coerce imported field 'year'"):
        _coerce_import_dtypes(
            pd.DataFrame({"year": ["not-an-integer"]}),
            {"year": "Int64"},
            context="test import",
        )


def test_duckdb_source_sql_quotes_path_options_and_projection_identifiers() -> None:
    sql = _duckdb_source_sql(
        "csv",
        Path("/tmp/O'Brien/data.csv"),
        {"header": True, "delim": "|", "nullstr": ["NA", "NULL"]},
    )
    assert sql == (
        "read_csv('/tmp/O''Brien/data.csv', header = TRUE, delim = '|', "
        "nullstr = ['NA', 'NULL'])"
    )

    json_sql = _duckdb_source_sql("jsonl", Path("/tmp/data.jsonl"), {})
    assert "read_json_auto" in json_sql
    assert "format = 'newline_delimited'" in json_sql
    assert _quote_duckdb_identifier('odd"field') == '"odd""field"'


def test_duckdb_source_sql_normalizes_windows_paths_to_forward_slashes() -> None:
    sql = _duckdb_source_sql(
        "parquet",
        PureWindowsPath(r"C:\Users\Zac\O'Brien\data.parquet"),
        {},
    )
    assert sql == "read_parquet('C:/Users/Zac/O''Brien/data.parquet')"


def test_duckdb_option_validation_rejects_sqlish_names_and_objects() -> None:
    with pytest.raises(ValueError, match="Invalid DuckDB reader option name"):
        _validate_duckdb_options({"header); DROP TABLE x; --": True})
    assert _validate_duckdb_options({"types": {"id": "BIGINT"}}) == {
        "types": {"id": "BIGINT"}
    }
    with pytest.raises(TypeError, match="JSON-like"):
        _validate_duckdb_options({"bad": {1, 2}})


def test_folder_inventory_paths_are_sorted_and_deduplicated(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "nested" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "nested" / "ignore.md").write_text("x", encoding="utf-8")

    paths = _inventory_paths(
        tmp_path.resolve(),
        ("*.txt", "*.pdf", "b.*"),
        recursive=True,
    )
    assert [path.relative_to(tmp_path.resolve()).as_posix() for path in paths] == [
        "a.pdf",
        "b.txt",
        "nested/c.txt",
    ]


def test_import_validation_helpers() -> None:
    assert _validate_patterns("*.txt") == ("*.txt",)
    assert _validate_patterns(["*.txt", "*.pdf"]) == ("*.txt", "*.pdf")
    assert _validate_batch_size(5) == 5
    with pytest.raises(TypeError):
        _validate_batch_size(True)
    for invalid in (0, -1, 1.5):
        with pytest.raises(ValueError):
            _validate_batch_size(invalid)


def test_duckdb_source_sql_many_accepts_explicit_csv_file_list() -> None:
    from text_analysis_lab.core.importers import _duckdb_source_sql_many

    sql = _duckdb_source_sql_many(
        "csv",
        ["/tmp/a.csv", "/tmp/O'Brien/b.csv"],
        {"header": True, "filename": True},
    )
    assert sql == (
        "read_csv(['/tmp/a.csv', '/tmp/O''Brien/b.csv'], "
        "header = TRUE, filename = TRUE)"
    )
