"""PDF text-extraction translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    RunRoute,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_EXTRACTION_STATUS_FIELD = "extraction_status"
_EXTRACTION_ERROR_FIELD = "extraction_error"
_PAGE_COUNT_FIELD = "page_count"
_PAGES_EXTRACTED_FIELD = "pages_extracted"
_MARGIN_NAMES = ("top", "bottom", "left", "right")


class PdfTextExtractor(BaseTranslator):
    """Extract machine-readable PDF text while preserving source keys.

    One input row remains one output row. Page boundaries are retained inside the
    extracted string via ``page_separator`` (form-feed by default), so page-level
    decomposition remains a separate TeAL operation. Normalized ``margins`` are
    fractions of each page dimension and are applied independently per page.

    This v1 intentionally does not perform OCR, automatic header/footer detection,
    dehyphenation, table extraction, or page decomposition.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        path_field: str = "path",
        text_field: str = "text",
        margins: Mapping[str, float] | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        page_separator: str = "\f",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(path_field, str) or not path_field:
            raise ValueError("path_field must be a non-empty string.")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        if text_field in {
            _EXTRACTION_STATUS_FIELD,
            _EXTRACTION_ERROR_FIELD,
            _PAGE_COUNT_FIELD,
            _PAGES_EXTRACTED_FIELD,
        }:
            raise ValueError(
                f"text_field={text_field!r} collides with a reserved PDF extraction field."
            )
        if not isinstance(page_separator, str) or not page_separator:
            raise ValueError("page_separator must be a non-empty string.")

        self.path_field = path_field
        self.text_field = text_field
        self.margins = _normalize_margins(margins)
        self.start_page = _normalize_page_number(start_page, name="start_page")
        self.end_page = _normalize_page_number(end_page, name="end_page")
        if (
            self.start_page is not None
            and self.end_page is not None
            and self.end_page < self.start_page
        ):
            raise ValueError("end_page must be greater than or equal to start_page.")
        self.page_separator = page_separator
        self.backend = "pdfplumber"
        self.backend_version = _pdfplumber_version()

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        _single_source(sources)
        return OutputSpec(
            artifact_type="table",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                f"PdfTextExtractor does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("PdfTextExtractor requires a table artifact source.")
        output_fields = {
            self.text_field,
            _EXTRACTION_STATUS_FIELD,
            _EXTRACTION_ERROR_FIELD,
            _PAGE_COUNT_FIELD,
            _PAGES_EXTRACTED_FIELD,
        }
        key_collisions = sorted(
            output_fields.intersection(str(name) for name in source.primary_key)
        )
        if key_collisions:
            raise OperatorError(
                "PdfTextExtractor output fields collide with source primary-key fields: "
                f"{key_collisions}."
            )
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.path_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 100,
            form="table",
            metadata_mode="none",
            include_position=False,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = mode, request
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        key_columns = [str(name) for name in packet.primary_key]
        missing = [
            name
            for name in [*key_columns, self.path_field]
            if name not in frame.columns
        ]
        if missing:
            raise ArtifactError(
                f"PdfTextExtractor source batch is missing columns {missing}."
            )

        rows = [self._extract_row(value) for value in frame[self.path_field].tolist()]
        keys = frame.loc[:, key_columns].reset_index(drop=True)
        data = pd.DataFrame(rows)
        if not data.empty:
            # Keep nullable output types stable across translation batches. In
            # particular, an all-success/all-no-text batch has only null
            # extraction_error values; without an explicit string dtype pandas /
            # Arrow may serialize that part as physical NULL while a later failed
            # batch serializes VARCHAR.
            for field in (
                self.text_field,
                _EXTRACTION_STATUS_FIELD,
                _EXTRACTION_ERROR_FIELD,
            ):
                data[field] = pd.array(data[field], dtype="string")
            data[_PAGE_COUNT_FIELD] = pd.array(data[_PAGE_COUNT_FIELD], dtype="Int64")
            data[_PAGES_EXTRACTED_FIELD] = pd.array(
                data[_PAGES_EXTRACTED_FIELD], dtype="Int64"
            )
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, "data": data}})

    def _extract_row(self, path_value: Any) -> dict[str, Any]:
        if path_value is None or pd.isna(path_value):
            return _failed_row(
                text_field=self.text_field,
                error="ValueError: Missing PDF path.",
            )

        path = Path(str(path_value))
        pages_extracted = 0
        page_count: int | None = None
        try:
            import pdfplumber

            page_texts: list[str] = []
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
                first = 1 if self.start_page is None else self.start_page
                last = (
                    page_count
                    if self.end_page is None
                    else min(self.end_page, page_count)
                )
                if first <= page_count and first <= last:
                    for page_number in range(first, last + 1):
                        page = pdf.pages[page_number - 1]
                        target = _crop_page(page, self.margins)
                        try:
                            extracted = target.extract_text() or ""
                            page_texts.append(_normalize_newlines(extracted))
                            pages_extracted += 1
                        finally:
                            # pdfplumber caches page layouts/objects. Explicitly release
                            # both derived crops and the source page, even if extraction
                            # raises, to bound memory on large PDFs.
                            if target is not page:
                                close_target = getattr(target, "close", None)
                                if callable(close_target):
                                    close_target()
                            page.close()

            text = self.page_separator.join(page_texts)
            status = "ok" if text.strip() else "no_text"
            return {
                self.text_field: text,
                _EXTRACTION_STATUS_FIELD: status,
                _EXTRACTION_ERROR_FIELD: None,
                _PAGE_COUNT_FIELD: page_count,
                _PAGES_EXTRACTED_FIELD: pages_extracted,
            }
        except Exception as exc:  # row-aligned failure is part of the contract
            return _failed_row(
                text_field=self.text_field,
                error=f"{exc.__class__.__name__}: {exc}",
                page_count=page_count,
                pages_extracted=pages_extracted,
            )

    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> PdfTextExtractor:
        _ = mode, request
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "path_field": self.path_field,
            "text_field": self.text_field,
            "margins": dict(self.margins),
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_separator": self.page_separator,
            "backend": self.backend,
            "backend_version": self.backend_version,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> PdfTextExtractor:
        backend = str(state.get("backend", "pdfplumber"))
        if backend != "pdfplumber":
            raise OperatorError(
                f"Unsupported PdfTextExtractor backend {backend!r}; expected 'pdfplumber'."
            )
        obj = cls(
            path_field=str(state.get("path_field", "path")),
            text_field=str(state.get("text_field", "text")),
            margins=cast(Mapping[str, float] | None, state.get("margins")),
            start_page=cast(int | None, state.get("start_page")),
            end_page=cast(int | None, state.get("end_page")),
            page_separator=str(state.get("page_separator", "\f")),
        )
        saved_version = state.get("backend_version")
        if saved_version is not None:
            obj.backend_version = str(saved_version)
        return obj

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_state()
        payload["operator_id"] = operator_id
        (intermediate_dir / "state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> PdfTextExtractor:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        return obj


def _normalize_page_number(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive 1-based integer or None.")
    return int(value)


def _normalize_margins(margins: Mapping[str, float] | None) -> dict[str, float]:
    normalized = {name: 0.0 for name in _MARGIN_NAMES}
    if margins is None:
        return normalized
    if not isinstance(margins, Mapping):
        raise TypeError("margins must be a mapping of top/bottom/left/right fractions.")
    unknown = sorted(set(margins) - set(_MARGIN_NAMES))
    if unknown:
        raise ValueError(
            f"Unsupported margin names {unknown}; expected only {list(_MARGIN_NAMES)}."
        )
    for name, raw_value in margins.items():
        if isinstance(raw_value, bool):
            raise ValueError(f"margins[{name!r}] must be a number in [0, 1).")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"margins[{name!r}] must be a number in [0, 1).") from exc
        if not 0.0 <= value < 1.0:
            raise ValueError(f"margins[{name!r}] must be in [0, 1).")
        normalized[str(name)] = value

    if normalized["left"] + normalized["right"] >= 1.0:
        raise ValueError("left + right margins must be less than 1.")
    if normalized["top"] + normalized["bottom"] >= 1.0:
        raise ValueError("top + bottom margins must be less than 1.")
    return normalized


def _crop_page(page: Any, margins: Mapping[str, float]) -> Any:
    if not any(float(value) for value in margins.values()):
        return page
    x0, top, x1, bottom = (float(value) for value in page.bbox)
    width = x1 - x0
    height = bottom - top
    bbox = (
        x0 + margins["left"] * width,
        top + margins["top"] * height,
        x1 - margins["right"] * width,
        bottom - margins["bottom"] * height,
    )
    return page.crop(bbox, strict=True)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _failed_row(
    *,
    text_field: str,
    error: str,
    page_count: int | None = None,
    pages_extracted: int = 0,
) -> dict[str, Any]:
    return {
        text_field: "",
        _EXTRACTION_STATUS_FIELD: "failed",
        _EXTRACTION_ERROR_FIELD: error,
        _PAGE_COUNT_FIELD: page_count,
        _PAGES_EXTRACTED_FIELD: pages_extracted,
    }


def _pdfplumber_version() -> str:
    try:
        return str(importlib_metadata.version("pdfplumber"))
    except importlib_metadata.PackageNotFoundError as exc:
        raise OperatorError(
            "PdfTextExtractor requires pdfplumber. Install TeAL's PDF extra "
            "(for example: uv sync --extra pdf)."
        ) from exc


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"PdfTextExtractor requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"PdfTextExtractor expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"PdfTextExtractor expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
