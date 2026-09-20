from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip(
    "pdfplumber",
    reason="PdfTextExtractor tests require the optional pdf extra.",
)

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import PdfTextExtractor


def _packet(frame: pd.DataFrame) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_source",
        primary_key=("file_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_pdf_text_extractor_validates_configuration_and_round_trips_state() -> None:
    with pytest.raises(ValueError, match="positive 1-based"):
        PdfTextExtractor(start_page=0)
    with pytest.raises(ValueError, match="greater than or equal"):
        PdfTextExtractor(start_page=3, end_page=2)
    with pytest.raises(ValueError, match="Unsupported margin names"):
        PdfTextExtractor(margins={"header": 0.1})
    with pytest.raises(ValueError, match=r"left \+ right"):
        PdfTextExtractor(margins={"left": 0.6, "right": 0.4})
    with pytest.raises(ValueError, match="page_separator"):
        PdfTextExtractor(page_separator="")

    extractor = PdfTextExtractor(
        path_field="pdf_path",
        text_field="document_text",
        margins={"top": 0.1, "bottom": 0.05},
        start_page=2,
        end_page=4,
        page_separator="<PAGE>",
    )
    state = json.loads(json.dumps(extractor.to_json_state()))
    restored = PdfTextExtractor.from_json_state(state)
    assert restored.to_json_state() == extractor.to_json_state()
    assert restored.backend == "pdfplumber"
    assert restored.backend_version


def test_pdf_text_extractor_declares_preserved_key_batch_contract() -> None:
    source = SimpleNamespace(artifact_type=ArtifactType.TABLE, primary_key=["file_id"])
    sources = {"source": source}
    extractor = PdfTextExtractor(path_field="path")
    request = TranslationRequest(batch_size=7)

    spec = extractor.output_specs(sources=sources, request=request)
    assert spec.lineage_mode == "preserved_key"
    assert spec.basis_labels == ("source",)

    source_request = extractor.input_request(
        sources=sources,
        mode="translate",
        request=request,
    )
    assert source_request.mode == "batches"
    assert source_request.batch_size == 7
    assert source_request.columns.keys is True
    assert source_request.columns.data == "path"
    assert source_request.columns.metadata is False

    colliding = SimpleNamespace(
        artifact_type=ArtifactType.TABLE, primary_key=["page_count"]
    )
    with pytest.raises(OperatorError, match="collide with source primary-key"):
        extractor.input_request(
            sources={"source": colliding},
            mode="translate",
            request=TranslationRequest(),
        )


def test_pdf_text_extractor_missing_path_is_row_aligned_failure() -> None:
    extractor = PdfTextExtractor()
    result = extractor.translate_batch(
        {"source": _packet(pd.DataFrame({"file_id": [10, 11], "path": [None, None]}))},
        mode="translate",
        request=TranslationRequest(),
    )
    payload = result.outputs["output"]
    assert payload["keys"]["file_id"].tolist() == [10, 11]
    assert payload["data"]["text"].tolist() == ["", ""]
    assert payload["data"]["extraction_status"].tolist() == ["failed", "failed"]
    assert payload["data"]["pages_extracted"].astype(int).tolist() == [0, 0]
    assert all(
        "Missing PDF path" in value for value in payload["data"]["extraction_error"]
    )


def test_pdf_text_extractor_real_backend_statuses_and_page_separator() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "pdf_extractor"
    frame = pd.DataFrame(
        {
            "file_id": [0, 1, 2],
            "path": [
                str(fixtures / "blank.pdf"),
                str(fixtures / "corrupt.pdf"),
                str(fixtures / "sample_two_page.pdf"),
            ],
        }
    )
    extractor = PdfTextExtractor()
    payload = extractor.translate_batch(
        {"source": _packet(frame)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    data = payload["data"]
    assert payload["keys"]["file_id"].tolist() == [0, 1, 2]
    assert data["extraction_status"].tolist() == ["no_text", "failed", "ok"]
    assert str(data["text"].dtype) == "string"
    assert str(data["extraction_status"].dtype) == "string"
    assert str(data["extraction_error"].dtype) == "string"
    assert data["pages_extracted"].astype(int).tolist() == [1, 0, 2]
    text = data.loc[2, "text"]
    assert "HEADER PAGE 1" in text
    assert "Body page one alpha" in text
    assert "FOOTER PAGE 1" in text
    assert "\f" in text
    assert "Body page two beta" in text


def test_pdf_text_extractor_real_backend_normalized_crop_and_page_range() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "pdf_extractor" / "sample_two_page.pdf"
    )
    extractor = PdfTextExtractor(
        margins={"top": 0.1, "bottom": 0.1},
        start_page=2,
        end_page=2,
    )
    payload = extractor.translate_batch(
        {"source": _packet(pd.DataFrame({"file_id": [7], "path": [str(fixture)]}))},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    data = payload["data"]
    assert data["extraction_status"].tolist() == ["ok"]
    assert data["page_count"].astype(int).tolist() == [2]
    assert data["pages_extracted"].astype(int).tolist() == [1]
    assert data["text"].tolist() == ["Body page two beta"]


def test_pdf_text_extractor_all_null_error_batch_keeps_string_dtype() -> None:
    fixture = Path(__file__).parent / "fixtures" / "pdf_extractor" / "blank.pdf"
    extractor = PdfTextExtractor()
    payload = extractor.translate_batch(
        {"source": _packet(pd.DataFrame({"file_id": [0], "path": [str(fixture)]}))},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    data = payload["data"]
    assert data["extraction_status"].tolist() == ["no_text"]
    assert str(data["extraction_error"].dtype) == "string"
    assert pd.isna(data["extraction_error"].iloc[0])
