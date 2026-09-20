from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.translators import RegexCleaner, RegexReplaceRule


def _query_full(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=True,
        metadata_mode="full",
        include_position=True,
        order_by="_position",
        form="table",
    )


def test_read_csv_assigns_teal_keys_selects_fields_and_supports_translation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus.csv"
    pd.DataFrame(
        {
            # This looks like a perfectly usable source ID. TeAL deliberately does
            # not adopt it as a primary key; it can be retained only as metadata.
            "doc_id": [10, 11, 12, 13, 14],
            "text": ["A-one", "B-two", "C-three", "D-four", "E-five"],
            "group": ["a", "a", "b", "b", "b"],
            # Not selected below: it must never enter the artifact.
            "score": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    ).to_csv(source, index=False)

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="csv_import")
    try:
        artifact = project.read_csv(
            source,
            text_fields="text",
            metadata_fields=["doc_id", "group"],
            batch_size=2,
            output_label="documents",
            duckdb_options={"header": True},
            memo="course corpus import",
        )
        assert artifact.primary_key == ["row_id"]
        assert artifact.descriptor["lineage"] == {
            "lineage_mode": "new_key",
            "basis_artifact_ids": [],
        }
        assert set(artifact.components) == {"keys", "data", "metadata"}
        assert len(list((artifact.artifact_dir / "keys").glob("part-*.parquet"))) == 3
        assert len(list((artifact.artifact_dir / "data").glob("part-*.parquet"))) == 3
        assert (
            len(list((artifact.artifact_dir / "metadata").glob("part-*.parquet"))) == 3
        )

        frame = _query_full(artifact)
        assert frame["_position"].tolist() == [0, 1, 2, 3, 4]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2, 3, 4]
        assert frame["doc_id"].astype(int).tolist() == [10, 11, 12, 13, 14]
        assert frame["text"].tolist() == [
            "A-one",
            "B-two",
            "C-three",
            "D-four",
            "E-five",
        ]
        assert frame["group"].tolist() == ["a", "a", "b", "b", "b"]
        assert "score" not in frame.columns
        assert artifact.query_columns(metadata_mode="local")["data"] == ["text"]
        assert artifact.query_columns(metadata_mode="local")["metadata"] == [
            "doc_id",
            "group",
        ]

        operation = project.operation_for_artifact(artifact)
        assert operation is not None
        assert operation["operation_type"] == "import"
        assert operation["status"] == "complete"
        operation_payload = json.loads(
            (
                project.storage.operation_dir(operation["operation_id"])
                / "operation.json"
            ).read_text(encoding="utf-8")
        )
        request = operation_payload["request"]
        assert operation_payload["import_kind"] == "read_csv"
        assert request["primary_key"] == ["row_id"]
        assert request["generated_primary_key"] is True
        assert request["text_fields"] == ["text"]
        assert request["metadata_fields"] == ["doc_id", "group"]
        assert request["discarded_fields"] == ["score"]
        assert request["batch_size"] == 2
        assert operation_payload["external_source"]["path"] == str(source.resolve())
        assert project.catalog.operation_sources(operation["operation_id"]) == []

        operator = project.get_operator(operation["operator_id"])
        assert operator.operation_type == "import"
        assert operator.kind == "read_csv"

        cleaned = project.translate(
            RegexCleaner(
                text_field="text",
                output_field="clean_text",
                rules=[RegexReplaceRule(r"-", " ")],
            ),
            artifact,
            batch_size=2,
        )["output"]
        cleaned_frame = _query_full(cleaned)
        assert cleaned_frame["row_id"].astype(int).tolist() == [0, 1, 2, 3, 4]
        assert cleaned_frame["clean_text"].tolist() == [
            "A one",
            "B two",
            "C three",
            "D four",
            "E five",
        ]
        assert cleaned_frame["doc_id"].astype(int).tolist() == [10, 11, 12, 13, 14]
        assert cleaned_frame["group"].tolist() == ["a", "a", "b", "b", "b"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        artifact = reopened.get_artifact("art_000001")
        frame = _query_full(artifact)
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2, 3, 4]
        assert frame["doc_id"].astype(int).tolist() == [10, 11, 12, 13, 14]
        assert frame["text"].tolist()[0] == "A-one"
        assert "score" not in frame.columns
    finally:
        reopened.close()


def test_read_jsonl_supports_multiple_text_fields_and_discards_unselected_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    records = [
        {"title": "one", "text": "first", "site": "x", "year": 2020, "unused": 100},
        {"title": "two", "text": "second", "site": "y", "year": 2021, "unused": 200},
        {"title": "three", "text": "third", "site": "x", "year": 2022, "unused": 300},
        {"title": "four", "text": "fourth", "site": "z", "year": 2023, "unused": 400},
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    project = teal.Project.create(tmp_path / "project", name="jsonl_import")
    try:
        artifact = project.read_jsonl(
            source,
            text_fields=["title", "text"],
            metadata_fields=["site", "year"],
            batch_size=2,
        )
        frame = _query_full(artifact)
        assert artifact.primary_key == ["row_id"]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2, 3]
        assert frame["title"].tolist() == ["one", "two", "three", "four"]
        assert frame["text"].tolist() == ["first", "second", "third", "fourth"]
        assert frame["site"].tolist() == ["x", "y", "x", "z"]
        assert frame["year"].astype(int).tolist() == [2020, 2021, 2022, 2023]
        assert "unused" not in frame.columns
        assert artifact.query_columns(metadata_mode="local")["data"] == [
            "title",
            "text",
        ]
        assert artifact.query_columns(metadata_mode="local")["metadata"] == [
            "site",
            "year",
        ]
    finally:
        project.close()


def test_read_parquet_always_rekeys_and_can_retain_source_ids_as_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paragraphs.parquet"
    pd.DataFrame(
        {
            "doc_id": [1, 1, 2, 2],
            "paragraph_id": [0, 1, 0, 1],
            "text": ["a", "b", "c", "d"],
            "source": ["s1", "s1", "s2", "s2"],
            "drop_me": [9, 8, 7, 6],
        }
    ).to_parquet(source, index=False)

    project = teal.Project.create(tmp_path / "project", name="parquet_import")
    try:
        artifact = project.read_parquet(
            source,
            text_fields="text",
            metadata_fields=["doc_id", "paragraph_id", "source"],
            batch_size=3,
        )
        frame = _query_full(artifact)
        assert artifact.primary_key == ["row_id"]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2, 3]
        assert list(
            zip(frame["doc_id"].astype(int), frame["paragraph_id"].astype(int))
        ) == [
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        assert frame["text"].tolist() == ["a", "b", "c", "d"]
        assert frame["source"].tolist() == ["s1", "s1", "s2", "s2"]
        assert "drop_me" not in frame.columns
    finally:
        project.close()


def test_duplicate_source_identifiers_do_not_affect_generated_teal_keys(
    tmp_path: Path,
) -> None:
    source = tmp_path / "duplicates.csv"
    pd.DataFrame(
        {
            "source_id": [1, 2, 1],
            "text": ["a", "b", "c"],
        }
    ).to_csv(source, index=False)

    project = teal.Project.create(tmp_path / "project", name="duplicate_source_ids")
    try:
        artifact = project.read_csv(
            source,
            text_fields="text",
            metadata_fields="source_id",
            batch_size=2,
        )
        frame = _query_full(artifact)
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["source_id"].astype(int).tolist() == [1, 2, 1]
        assert (
            project.list_operations(operation_type="import")[0]["status"] == "complete"
        )
    finally:
        project.close()


def test_tabular_import_preflight_validation_does_not_create_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus.csv"
    pd.DataFrame({"id": [1], "text": ["x"]}).to_csv(source, index=False)
    project = teal.Project.create(tmp_path / "project", name="preflight")
    try:
        with pytest.raises(ArtifactError, match="Unknown metadata_fields"):
            project.read_csv(
                source,
                text_fields="text",
                metadata_fields="missing",
            )
        assert project.list_operations(operation_type="import") == []
        assert project.list_artifacts() == []
    finally:
        project.close()


def test_folder_inventory_is_path_only_deterministic_and_reusable(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "nested").mkdir(parents=True)
    (corpus / "b.txt").write_text("B body", encoding="utf-8")
    (corpus / "a.pdf").write_bytes(b"%PDF fake")
    (corpus / "nested" / "c.txt").write_text("C body", encoding="utf-8")
    (corpus / "nested" / "ignore.md").write_text("ignore", encoding="utf-8")

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="folder_import")
    try:
        inventory = project.folder_inventory(
            corpus,
            patterns=["*.txt", "*.pdf", "b.*"],
            recursive=True,
            batch_size=2,
            output_label="files",
        )
        assert inventory.primary_key == ["file_id"]
        assert set(inventory.components) == {"keys", "data"}
        frame = inventory.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=False,
            include_position=True,
            order_by="_position",
            form="table",
        )
        assert frame["file_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["_position"].tolist() == [0, 1, 2]
        assert frame["relative_path"].tolist() == ["a.pdf", "b.txt", "nested/c.txt"]
        assert frame["extension"].tolist() == [".pdf", ".txt", ".txt"]
        assert all(Path(value).is_absolute() for value in frame["path"])
        assert frame["file_name"].tolist() == ["a.pdf", "b.txt", "c.txt"]

        cleaned = project.translate(
            RegexCleaner(
                text_field="file_name",
                output_field="normalized_name",
                rules=[RegexReplaceRule(r"\.(txt|pdf)$", "")],
            ),
            inventory,
            batch_size=2,
        )["output"]
        cleaned_frame = cleaned.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        assert cleaned_frame["file_id"].astype(int).tolist() == [0, 1, 2]
        assert cleaned_frame["normalized_name"].tolist() == ["a", "b", "c"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        inventory = reopened.get_artifact("art_000001")
        frame = inventory.query(
            key_columns=True,
            data_columns="relative_path",
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        assert frame["relative_path"].tolist() == ["a.pdf", "b.txt", "nested/c.txt"]
    finally:
        reopened.close()


def test_folder_inventory_no_matches_is_preflight_error(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="empty_folder")
    try:
        with pytest.raises(ArtifactError, match="matched no files"):
            project.folder_inventory(tmp_path, patterns="*.pdf", recursive=False)
        assert project.list_operations(operation_type="import") == []
    finally:
        project.close()


def test_read_csv_folder_combines_files_in_sorted_order_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aera_2026"
    nested = root / "nested"
    nested.mkdir(parents=True)
    pd.DataFrame(
        {
            "record_type": ["paper", "session"],
            "title": ["B paper", "B session"],
            "abstract": ["B abstract", "B session abstract"],
            "authors": ["One, U", "Two, U; Three, V"],
        }
    ).to_csv(root / "B_Type.csv", index=False)
    pd.DataFrame(
        {
            "record_type": ["paper"],
            "title": ["A paper"],
            "abstract": ["A abstract"],
            "authors": ["Author, U"],
        }
    ).to_csv(nested / "A_Type.csv", index=False)

    project = teal.Project.create(tmp_path / "project", name="csv_folder")
    try:
        artifact = project.read_csv_folder(
            root,
            text_fields=["title", "abstract"],
            metadata_fields=["record_type", "authors"],
            recursive=True,
            output_label="documents",
        )
        frame = artifact.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="local",
            include_position=True,
            order_by="_position",
            form="table",
        )
        # Deterministic ordering is by relative path: B_Type.csv precedes nested/A_Type.csv.
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["title"].tolist() == ["B paper", "B session", "A paper"]
        assert frame["source_file"].tolist() == [
            "B_Type.csv",
            "B_Type.csv",
            "nested/A_Type.csv",
        ]
        assert frame["source_row"].astype(int).tolist() == [0, 1, 0]
        assert "source_relative_path" not in frame.columns
        assert "session_type" not in frame.columns

        operation = project.operation_for_artifact(artifact)
        assert operation is not None
        payload = json.loads(
            (
                project.storage.operation_dir(operation["operation_id"])
                / "operation.json"
            ).read_text(encoding="utf-8")
        )
        assert payload["import_kind"] == "read_csv_folder"
        assert payload["request"]["matched_files"] == [
            "B_Type.csv",
            "nested/A_Type.csv",
        ]
        assert payload["external_source"]["matched_file_count"] == 2
    finally:
        project.close()


def _write_excel_workbook(
    path: Path, sheets: list[tuple[str, list[list[object]], bool]]
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows, hidden in sheets:
        worksheet = workbook.create_sheet(name)
        for row in rows:
            worksheet.append(row)
        if hidden:
            worksheet.sheet_state = "hidden"
    workbook.save(path)
    workbook.close()


def test_read_excel_multiple_sheets_by_name_and_index_preserves_sheet_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "student.xlsx"
    _write_excel_workbook(
        source,
        [
            (
                "Conversation A",
                [
                    ["title row"],
                    ["speaker", "content", "notes"],
                    ["child", "A1", "n1"],
                    ["parent", "A2", None],
                ],
                False,
            ),
            (
                "Conversation B",
                [
                    ["title row"],
                    ["speaker", "content", "notes"],
                    ["child", "B1", "n2"],
                ],
                False,
            ),
        ],
    )

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="excel_import")
    try:
        artifact = project.read_excel(
            source,
            sheets=["Conversation B", 0],
            header_row=1,
            text_fields="content",
            metadata_fields=["speaker", "notes"],
            batch_size=2,
            output_label="turn_rows",
        )
        frame = _query_full(artifact)
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["content"].tolist() == ["B1", "A1", "A2"]
        assert frame["source_sheet"].tolist() == [
            "Conversation B",
            "Conversation A",
            "Conversation A",
        ]
        assert frame["source_sheet_index"].astype(int).tolist() == [1, 0, 0]
        assert frame["source_row"].astype(int).tolist() == [0, 0, 1]

        operation = project.operation_for_artifact(artifact)
        payload = json.loads(
            (
                project.storage.operation_dir(operation["operation_id"])
                / "operation.json"
            ).read_text(encoding="utf-8")
        )
        assert payload["import_kind"] == "read_excel"
        assert payload["request"]["sheets"] == ["Conversation B", 0]
        assert payload["request"]["header_row"] == 1
        assert payload["request"]["formula_policy"] == "cached_values"
        assert payload["request"]["reader_engine"]["data_only"] is True
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        artifact = reopened.get_artifact("art_000001")
        frame = _query_full(artifact)
        assert frame["content"].tolist() == ["B1", "A1", "A2"]
    finally:
        reopened.close()


def test_read_excel_all_sheets_includes_hidden_and_missing_fields_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "student.xlsx"
    _write_excel_workbook(
        source,
        [
            (
                "Visible",
                [
                    ["speaker", "content", "notes"],
                    ["child", "visible", "note"],
                ],
                False,
            ),
            (
                "Hidden",
                [
                    ["speaker", "content"],
                    ["child", "hidden"],
                ],
                True,
            ),
        ],
    )

    project = teal.Project.create(tmp_path / "strict_project", name="excel_strict")
    try:
        with pytest.raises(ArtifactError, match="missing requested field") as exc_info:
            project.read_excel(
                source,
                sheets=None,
                text_fields="content",
                metadata_fields=["speaker", "notes"],
            )
        assert "missing_fields=None" in str(exc_info.value)
        assert project.list_operations(operation_type="import") == []
    finally:
        project.close()

    project = teal.Project.create(tmp_path / "fill_project", name="excel_fill")
    try:
        artifact = project.read_excel(
            source,
            sheets=None,
            text_fields="content",
            metadata_fields=["speaker", "notes"],
            missing_fields=None,
        )
        frame = _query_full(artifact)
        assert frame["content"].tolist() == ["visible", "hidden"]
        assert frame["source_sheet"].tolist() == ["Visible", "Hidden"]
        assert frame["notes"].iloc[0] == "note"
        assert pd.isna(frame["notes"].iloc[1])
    finally:
        project.close()


def test_read_excel_folder_orders_files_then_sheets_and_can_fill_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workbooks"
    nested = root / "nested"
    nested.mkdir(parents=True)
    _write_excel_workbook(
        root / "B.xlsx",
        [
            ("First", [["speaker", "content", "rating"], ["child", "B1", 4]], False),
            ("Second", [["speaker", "content"], ["parent", "B2"]], False),
        ],
    )
    _write_excel_workbook(
        nested / "A.xlsx",
        [
            ("First", [["speaker", "content", "rating"], ["child", "A1", 5]], False),
            ("Second", [["speaker", "content"], ["parent", "A2"]], False),
        ],
    )

    project = teal.Project.create(tmp_path / "project", name="excel_folder")
    try:
        artifact = project.read_excel_folder(
            root,
            sheets=None,
            text_fields="content",
            metadata_fields=["speaker", "rating"],
            missing_fields=0,
            recursive=True,
            batch_size=1,
        )
        frame = _query_full(artifact)
        assert frame["content"].tolist() == ["B1", "B2", "A1", "A2"]
        assert frame["source_file"].tolist() == [
            "B.xlsx",
            "B.xlsx",
            "nested/A.xlsx",
            "nested/A.xlsx",
        ]
        assert frame["source_sheet"].tolist() == ["First", "Second", "First", "Second"]
        assert frame["source_sheet_index"].astype(int).tolist() == [0, 1, 0, 1]
        assert frame["source_row"].astype(int).tolist() == [0, 0, 0, 0]
        assert frame["rating"].tolist() == [4, 0, 5, 0]

        operation = project.operation_for_artifact(artifact)
        payload = json.loads(
            (
                project.storage.operation_dir(operation["operation_id"])
                / "operation.json"
            ).read_text(encoding="utf-8")
        )
        assert payload["import_kind"] == "read_excel_folder"
        assert payload["request"]["matched_files"] == ["B.xlsx", "nested/A.xlsx"]
        assert payload["request"]["missing_fields"] == {"mode": "fill", "value": 0}
    finally:
        project.close()
