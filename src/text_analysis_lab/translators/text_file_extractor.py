"""Plain-text file extraction translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import codecs
import json
from collections.abc import Mapping
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
# These PDF-oriented diagnostics are deliberately retained as nullable fields in
# the TXT output schema so TXT and PDF extraction branches can be recombined by
# TeAL's strict structural merge without copying or schema-rewriting data.
_PAGE_COUNT_FIELD = "page_count"
_PAGES_EXTRACTED_FIELD = "pages_extracted"


class TextFileExtractor(BaseTranslator):
    """Read plain-text files while preserving source keys.

    One input path remains one output row. The translator reads only the selected
    path field, normalizes platform line endings to ``\n``, and records row-level
    extraction failures instead of dropping observations or failing the batch.

    ``page_count`` and ``pages_extracted`` are nullable/not-applicable for TXT
    rows. They are present only so TXT output has the same logical schema as
    :class:`PdfTextExtractor`, allowing heterogeneous folder branches to be
    structurally merged after extraction.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        path_field: str = "path",
        text_field: str = "text",
        encoding: str = "utf-8",
        errors: str = "replace",
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
                f"text_field={text_field!r} collides with a reserved extraction field."
            )
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty codec name.")
        if not isinstance(errors, str) or not errors:
            raise ValueError("errors must be a non-empty codec error-handler name.")
        try:
            codecs.lookup(encoding)
        except LookupError as exc:
            raise ValueError(f"Unknown text encoding {encoding!r}.") from exc
        try:
            codecs.lookup_error(errors)
        except LookupError as exc:
            raise ValueError(f"Unknown codec error handler {errors!r}.") from exc

        self.path_field = path_field
        self.text_field = text_field
        self.encoding = encoding
        self.errors = errors

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
                f"TextFileExtractor does not accept operation parameters; got {sorted(params)}."
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
            raise OperatorError("TextFileExtractor requires a table artifact source.")
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
                "TextFileExtractor output fields collide with source primary-key fields: "
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
                f"TextFileExtractor source batch is missing columns {missing}."
            )

        rows = [self._extract_row(value) for value in frame[self.path_field].tolist()]
        keys = frame.loc[:, key_columns].reset_index(drop=True)
        data = pd.DataFrame(rows)
        if not data.empty:
            for field in (
                self.text_field,
                _EXTRACTION_STATUS_FIELD,
                _EXTRACTION_ERROR_FIELD,
            ):
                data[field] = pd.array(data[field], dtype="string")
            # Not applicable to TXT, but keep a stable nullable integer physical
            # type so the schema can be unioned with PdfTextExtractor output.
            data[_PAGE_COUNT_FIELD] = pd.array(data[_PAGE_COUNT_FIELD], dtype="Int64")
            data[_PAGES_EXTRACTED_FIELD] = pd.array(
                data[_PAGES_EXTRACTED_FIELD], dtype="Int64"
            )
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, "data": data}})

    def _extract_row(self, path_value: Any) -> dict[str, Any]:
        if path_value is None or pd.isna(path_value):
            return _failed_row(
                text_field=self.text_field,
                error="ValueError: Missing text-file path.",
            )

        path = Path(str(path_value))
        try:
            text = path.read_text(encoding=self.encoding, errors=self.errors)
            text = _normalize_newlines(text)
            return {
                self.text_field: text,
                _EXTRACTION_STATUS_FIELD: "ok" if text.strip() else "no_text",
                _EXTRACTION_ERROR_FIELD: None,
                _PAGE_COUNT_FIELD: None,
                _PAGES_EXTRACTED_FIELD: None,
            }
        except Exception as exc:  # row-aligned failure is part of the contract
            return _failed_row(
                text_field=self.text_field,
                error=f"{exc.__class__.__name__}: {exc}",
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
    ) -> TextFileExtractor:
        _ = mode, request
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "path_field": self.path_field,
            "text_field": self.text_field,
            "encoding": self.encoding,
            "errors": self.errors,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> TextFileExtractor:
        return cls(
            path_field=str(state.get("path_field", "path")),
            text_field=str(state.get("text_field", "text")),
            encoding=str(state.get("encoding", "utf-8")),
            errors=str(state.get("errors", "replace")),
        )

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
    ) -> TextFileExtractor:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        return obj


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _failed_row(*, text_field: str, error: str) -> dict[str, Any]:
    return {
        text_field: "",
        _EXTRACTION_STATUS_FIELD: "failed",
        _EXTRACTION_ERROR_FIELD: error,
        _PAGE_COUNT_FIELD: None,
        _PAGES_EXTRACTED_FIELD: None,
    }


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"TextFileExtractor requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"TextFileExtractor expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"TextFileExtractor expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
