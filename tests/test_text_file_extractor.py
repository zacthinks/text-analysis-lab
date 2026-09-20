from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import TextFileExtractor


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


def test_text_file_extractor_validates_configuration_and_round_trips_state() -> None:
    with pytest.raises(ValueError, match="Unknown text encoding"):
        TextFileExtractor(encoding="definitely-not-a-codec")
    with pytest.raises(ValueError, match="Unknown codec error handler"):
        TextFileExtractor(errors="definitely-not-an-error-handler")
    with pytest.raises(ValueError, match="collides with a reserved extraction field"):
        TextFileExtractor(text_field="extraction_status")

    extractor = TextFileExtractor(
        path_field="source_path",
        text_field="document_text",
        encoding="utf-8",
        errors="strict",
    )
    restored = TextFileExtractor.from_json_state(
        json.loads(json.dumps(extractor.to_json_state()))
    )
    assert restored.to_json_state() == extractor.to_json_state()


def test_text_file_extractor_declares_preserved_key_batch_contract() -> None:
    source = SimpleNamespace(artifact_type=ArtifactType.TABLE, primary_key=["file_id"])
    extractor = TextFileExtractor(path_field="path")
    request = TranslationRequest(batch_size=7)

    spec = extractor.output_specs(sources={"source": source}, request=request)
    assert spec.lineage_mode == "preserved_key"
    assert spec.basis_labels == ("source",)

    source_request = extractor.input_request(
        sources={"source": source}, mode="translate", request=request
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


def test_text_file_extractor_reads_normalizes_and_keeps_pdf_compatible_schema(
    tmp_path,
) -> None:
    text_path = tmp_path / "document.txt"
    text_path.write_bytes(b"alpha\r\nbeta\rgamma\n")
    empty_path = tmp_path / "empty.txt"
    empty_path.write_bytes(b"")

    extractor = TextFileExtractor()
    payload = extractor.translate_batch(
        {
            "source": _packet(
                pd.DataFrame(
                    {
                        "file_id": [3, 4, 5],
                        "path": [
                            str(text_path),
                            str(empty_path),
                            str(tmp_path / "missing.txt"),
                        ],
                    }
                )
            )
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]

    assert payload["keys"]["file_id"].tolist() == [3, 4, 5]
    data = payload["data"]
    assert list(data.columns) == [
        "text",
        "extraction_status",
        "extraction_error",
        "page_count",
        "pages_extracted",
    ]
    assert data["text"].tolist() == ["alpha\nbeta\ngamma\n", "", ""]
    assert data["extraction_status"].tolist() == ["ok", "no_text", "failed"]
    assert pd.isna(data.loc[0, "extraction_error"])
    assert data.loc[2, "extraction_error"]
    assert data["page_count"].isna().all()
    assert data["pages_extracted"].isna().all()
    assert str(data["text"].dtype) == "string"
    assert str(data["extraction_error"].dtype) == "string"
    assert str(data["page_count"].dtype) == "Int64"


def test_text_file_extractor_missing_path_is_row_aligned_failure() -> None:
    extractor = TextFileExtractor()
    payload = extractor.translate_batch(
        {"source": _packet(pd.DataFrame({"file_id": [10], "path": [None]}))},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert payload["keys"]["file_id"].tolist() == [10]
    assert payload["data"]["text"].tolist() == [""]
    assert payload["data"]["extraction_status"].tolist() == ["failed"]
    assert "Missing text-file path" in payload["data"]["extraction_error"].iloc[0]


def test_text_file_extractor_supports_explicit_non_utf8_encoding(tmp_path) -> None:
    text_path = tmp_path / "latin1.txt"
    text_path.write_bytes("caf\xe9".encode("latin-1"))
    extractor = TextFileExtractor(encoding="latin-1", errors="strict")
    payload = extractor.translate_batch(
        {"source": _packet(pd.DataFrame({"file_id": [1], "path": [str(text_path)]}))},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert payload["data"]["text"].tolist() == ["caf\xe9"]
    assert payload["data"]["extraction_status"].tolist() == ["ok"]
