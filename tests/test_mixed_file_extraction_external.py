from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("pdfplumber")

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    PdfTextExtractor,
    RegexCleaner,
    RegexReplaceRule,
    TextFileExtractor,
)

PDF_FIXTURE = (
    Path(__file__).parent / "fixtures" / "pdf_extractor" / "sample_two_page.pdf"
)


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


def _write_extension_rule(path: Path, extension: str) -> Path:
    path.write_text(
        "def keep(packet):\n"
        f"    return packet['extension'].str.lower() == {extension!r}\n",
        encoding="utf-8",
    )
    return path


def test_mixed_pdf_txt_inventory_extract_merge_downstream_and_reopen(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy2(PDF_FIXTURE, corpus / "b.pdf")
    (corpus / "a.txt").write_bytes(b"Plain\r\ntext alpha")
    (corpus / "c.txt").write_text("Plain text gamma", encoding="utf-8")

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="mixed_file_extract")
    try:
        inventory = project.folder_inventory(
            corpus, patterns=["*.pdf", "*.txt"], batch_size=2
        )
        inventory_frame = inventory.query(
            key_columns=True,
            data_columns=["relative_path", "extension"],
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        assert inventory_frame["relative_path"].tolist() == ["a.txt", "b.pdf", "c.txt"]
        assert inventory_frame["file_id"].astype(int).tolist() == [0, 1, 2]

        pdf_branch = project.subset(
            inventory,
            (_write_extension_rule(tmp_path / "keep_pdf.py", ".pdf"), "keep"),
            data_columns="extension",
            batch_size=2,
            output_label="pdf_paths",
        )["pdf_paths"]
        txt_branch = project.subset(
            inventory,
            (_write_extension_rule(tmp_path / "keep_txt.py", ".txt"), "keep"),
            data_columns="extension",
            batch_size=2,
            output_label="txt_paths",
        )["txt_paths"]

        pdf_text = project.translate(PdfTextExtractor(), pdf_branch, batch_size=1)[
            "output"
        ]
        txt_text = project.translate(TextFileExtractor(), txt_branch, batch_size=1)[
            "output"
        ]

        # The two extractors deliberately expose the same logical output schema,
        # allowing strict structural merge after format-specific processing.
        assert (
            pdf_text.query_columns(metadata_mode="full")["data"]
            == txt_text.query_columns(metadata_mode="full")["data"]
        )

        merged = project.merge([pdf_text, txt_text], batch_size=1)
        assert set(merged.components) == {"keys"}
        frame = _query_full(merged)
        # Merge source order is durable: PDF branch first, then TXT branch.
        assert frame["file_id"].astype(int).tolist() == [1, 0, 2]
        assert frame["extraction_status"].tolist() == ["ok", "ok", "ok"]
        assert (
            "Body page one alpha"
            in frame.loc[frame["file_id"].astype(int) == 1, "text"].iloc[0]
        )
        assert (
            frame.loc[frame["file_id"].astype(int) == 0, "text"].iloc[0]
            == "Plain\ntext alpha"
        )
        assert (
            frame.loc[frame["file_id"].astype(int) == 2, "text"].iloc[0]
            == "Plain text gamma"
        )
        assert pd.isna(
            frame.loc[frame["file_id"].astype(int) == 0, "page_count"].iloc[0]
        )
        assert (
            int(frame.loc[frame["file_id"].astype(int) == 1, "page_count"].iloc[0]) == 2
        )

        cleaned = project.translate(
            RegexCleaner(
                text_field="text",
                output_field="clean_text",
                rules=[RegexReplaceRule(r"Plain", "CLEAN")],
            ),
            merged,
            batch_size=1,
        )["output"]
        cleaned_frame = _query_full(cleaned)
        assert cleaned_frame["file_id"].astype(int).tolist() == [1, 0, 2]
        assert (
            cleaned_frame.loc[cleaned_frame["file_id"].astype(int) == 0, "clean_text"]
            .iloc[0]
            .startswith("CLEAN")
        )
        merged_id = merged.artifact_id
        cleaned_id = cleaned.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        merged = reopened.get_artifact(merged_id)
        frame = _query_full(merged)
        assert frame["file_id"].astype(int).tolist() == [1, 0, 2]
        assert (
            frame.loc[frame["file_id"].astype(int) == 2, "text"].iloc[0]
            == "Plain text gamma"
        )

        cleaned = reopened.get_artifact(cleaned_id)
        cleaned_frame = _query_full(cleaned)
        assert (
            cleaned_frame.loc[cleaned_frame["file_id"].astype(int) == 0, "clean_text"]
            .iloc[0]
            .startswith("CLEAN")
        )
    finally:
        reopened.close()
