from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pdfplumber = pytest.importorskip("pdfplumber")

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    PdfTextExtractor,
    RegexCleaner,
    RegexReplaceRule,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdf_extractor"


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


def test_pdf_extractor_folder_inventory_statuses_downstream_and_reopen(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ["sample_two_page.pdf", "blank.pdf", "corrupt.pdf"]:
        shutil.copy2(FIXTURES / name, corpus / name)

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="pdf_extract")
    try:
        inventory = project.folder_inventory(corpus, patterns="*.pdf", batch_size=2)
        inventory_frame = inventory.query(
            key_columns=True,
            data_columns=["path", "relative_path"],
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        # folder_inventory sorts by relative path: blank, corrupt, sample.
        assert inventory_frame["relative_path"].tolist() == [
            "blank.pdf",
            "corrupt.pdf",
            "sample_two_page.pdf",
        ]

        extractor = PdfTextExtractor()
        extracted = project.translate(extractor, inventory, batch_size=2)["output"]
        frame = _query_full(extracted)
        assert extracted.primary_key == ["file_id"]
        assert frame["file_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["_position"].tolist() == [0, 1, 2]
        assert frame["extraction_status"].tolist() == ["no_text", "failed", "ok"]
        assert frame["page_count"].tolist()[0] == 1
        assert frame["pages_extracted"].astype(int).tolist() == [1, 0, 2]
        assert frame["text"].tolist()[0] == ""
        assert frame["text"].tolist()[1] == ""
        assert pd.isna(frame["extraction_error"].tolist()[0])
        assert (
            "PDF" in str(frame["extraction_error"].tolist()[1])
            or frame["extraction_error"].tolist()[1]
        )

        sample_text = frame.loc[frame["file_id"].astype(int) == 2, "text"].iloc[0]
        assert "HEADER PAGE 1" in sample_text
        assert "Body page one alpha" in sample_text
        assert "FOOTER PAGE 1" in sample_text
        assert "\f" in sample_text
        assert "HEADER PAGE 2" in sample_text
        assert "Body page two beta" in sample_text
        assert "FOOTER PAGE 2" in sample_text

        # A normal downstream Translator must be able to consume the extracted artifact.
        cleaned = project.translate(
            RegexCleaner(
                text_field="text",
                output_field="clean_text",
                rules=[RegexReplaceRule(r"Body", "CONTENT")],
            ),
            extracted,
            batch_size=2,
        )["output"]
        cleaned_frame = _query_full(cleaned)
        assert cleaned_frame["file_id"].astype(int).tolist() == [0, 1, 2]
        assert (
            "CONTENT page one alpha"
            in cleaned_frame.loc[
                cleaned_frame["file_id"].astype(int) == 2, "clean_text"
            ].iloc[0]
        )

        operation = project.operation_for_artifact(extracted)
        assert operation is not None
        operator = project.get_operator(operation["operator_id"])
        assert isinstance(operator, PdfTextExtractor)
        assert operator.backend == "pdfplumber"
        assert operator.backend_version == str(pdfplumber.__version__)
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        extracted = reopened.get_artifact("art_000002")
        frame = _query_full(extracted)
        assert frame["extraction_status"].tolist() == ["no_text", "failed", "ok"]
        assert (
            "Body page two beta"
            in frame.loc[frame["file_id"].astype(int) == 2, "text"].iloc[0]
        )

        # Reopened frozen extractor remains executable against the same source contract.
        operation = reopened.operation_for_artifact(extracted)
        assert operation is not None
        operator = reopened.get_operator(operation["operator_id"])
        inventory = reopened.get_artifact("art_000001")
        rerun = reopened.translate(operator, inventory, batch_size=1)["output"]
        rerun_frame = _query_full(rerun)
        assert rerun_frame["extraction_status"].tolist() == ["no_text", "failed", "ok"]
    finally:
        reopened.close()


def test_pdf_extractor_normalized_margins_and_page_range(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy2(FIXTURES / "sample_two_page.pdf", corpus / "sample.pdf")

    project = teal.Project.create(tmp_path / "project", name="pdf_crop")
    try:
        inventory = project.folder_inventory(corpus, patterns="*.pdf")
        extracted = project.translate(
            PdfTextExtractor(
                margins={"top": 0.1, "bottom": 0.1},
                start_page=2,
                end_page=2,
            ),
            inventory,
        )["output"]
        frame = _query_full(extracted)
        assert frame["extraction_status"].tolist() == ["ok"]
        assert frame["page_count"].astype(int).tolist() == [2]
        assert frame["pages_extracted"].astype(int).tolist() == [1]
        text = frame["text"].iloc[0]
        assert text == "Body page two beta"
        assert "HEADER" not in text
        assert "FOOTER" not in text
        assert "\f" not in text
    finally:
        project.close()
