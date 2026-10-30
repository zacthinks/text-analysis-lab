"""Declarative preserved-key regex extraction into TeAL data and metadata."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.query_scope import QueryScope
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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL, MetadataMode
from text_analysis_lab.translators.regex_cleaner import _compile_flags, _normalize_flags

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


UnmatchedPolicy = Literal["fill", "warn", "fail"]
_RESERVED = frozenset({"_position", "_batch", "_row_offset"})


class PatternExtractor(BaseTranslator):
    """Extract named regex groups into data/metadata without copying source fields.

    Group names must start with "data_" or "metadata_"; the remainder is the
    output field name. Unnamed capture groups are allowed but are not emitted.
    Every observation is evaluated with re.fullmatch: at most one match per row.

    Standalone translate accepts a Series, a DataFrame containing field, or
    a {"data": DataFrame, "metadata": DataFrame} preview packet (either component optional). It returns
    component-separated DataFrames. Persisted execution adds TeAL-managed keys,
    and the source's other columns remain available through lineage rather than
    being duplicated.
    """

    operation_type = "translate"
    frozen_runtime_fields = frozenset({"_source_field"})

    def __init__(
        self,
        pattern: str,
        *,
        field: str = "text",
        metadata_mode: MetadataMode = "none",
        on_unmatched: UnmatchedPolicy = "fill",
        fill_value: str | None = "",
        flags: Sequence[str] | str | int = (),
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("PatternExtractor pattern must be a non-empty string.")
        if not isinstance(field, str) or not field:
            raise ValueError("field must be a non-empty string.")
        if metadata_mode not in {"none", "local", "full"}:
            raise ValueError("metadata_mode must be 'none', 'local', or 'full'.")
        if on_unmatched not in {"fill", "warn", "fail"}:
            raise ValueError("on_unmatched must be 'fill', 'warn', or 'fail'.")
        if fill_value is not None and not isinstance(fill_value, str):
            raise TypeError("fill_value must be a string or None.")

        self.pattern = pattern
        self.field = field
        self.metadata_mode = metadata_mode
        self._source_field: str | None = None
        self.on_unmatched: UnmatchedPolicy = on_unmatched
        self.fill_value = fill_value
        self.flags = _normalize_flags(flags)
        # The standard library handles escapes, classes, comments, duplicate
        # names, backreferences and group numbering. No regex rewriting needed.
        self._compiled = re.compile(self.pattern, _compile_flags(self.flags))
        if not self._compiled.groupindex:
            raise ValueError("PatternExtractor requires at least one named capture group.")

        destinations: list[tuple[str, str, str]] = []
        for group_name in self._compiled.groupindex:
            if group_name.startswith("metadata_"):
                domain, field = "metadata", group_name[len("metadata_"):]
            elif group_name.startswith("data_"):
                domain, field = "data", group_name[len("data_"):]
            else:
                raise ValueError(
                    f"Invalid named capture group {group_name!r}: expected "
                    "'data_<field>' or 'metadata_<field>'."
                )
            if not field or not field.isidentifier():
                raise ValueError(
                    f"Invalid named capture group {group_name!r}: the field "
                    "after its namespace must be a non-empty identifier."
                )
            if field in _RESERVED:
                raise ValueError(
                    f"Named capture group {group_name!r} targets a reserved "
                    f"TeAL structural field {field!r}."
                )
            destinations.append((group_name, domain, field))
        self._destinations = tuple(destinations)

    def translate(
        self,
        values: str | None | Sequence[str | None] | pd.Series | pd.DataFrame | Mapping[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Extract standalone values into only the specified output components."""
        if isinstance(values, pd.Series):
            text = values
        elif isinstance(values, str) or values is None:
            text = pd.Series([values], dtype="string")
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            text = pd.Series(list(values), dtype="string")
        else:
            if isinstance(values, Mapping):
                data = values.get("data")
                metadata = values.get("metadata")
                if data is not None and not isinstance(data, pd.DataFrame):
                    raise TypeError("PatternExtractor preview data must be a DataFrame.")
                if metadata is not None and not isinstance(metadata, pd.DataFrame):
                    raise TypeError("PatternExtractor preview metadata must be a DataFrame.")
                if data is None and metadata is None:
                    raise ValueError("PatternExtractor preview packet requires data or metadata.")
                if data is not None and metadata is not None and len(metadata) != len(data):
                    raise ValueError(
                        "PatternExtractor preview metadata must be a row-aligned DataFrame."
                    )
            elif isinstance(values, pd.DataFrame):
                data = values
            else:
                raise TypeError(
                    "PatternExtractor.translate expects text, a pandas Series, "
                    "a sequence, a DataFrame, or a data/metadata DataFrame packet."
                )
            component = "data"
            if isinstance(values, Mapping):
                in_data = isinstance(data, pd.DataFrame) and self.field in data.columns
                in_metadata = isinstance(metadata, pd.DataFrame) and self.field in metadata.columns
                if in_data and in_metadata:
                    raise ValueError(f"PatternExtractor preview field {self.field!r} is ambiguous across components.")
                if in_metadata and not in_data:
                    component = "metadata"
            frame = metadata if component == "metadata" else data
            if not isinstance(frame, pd.DataFrame) or self.field not in frame.columns:
                raise ValueError(f"PatternExtractor input is missing field {self.field!r}.")
            if not frame.columns.is_unique:
                raise ValueError("PatternExtractor input DataFrame columns must be unique.")
            text = frame[self.field]

        columns: dict[str, list[str | None]] = {
            name: [] for name, _domain, _field in self._destinations
        }
        unmatched = 0
        for index, value in enumerate(text):
            if isinstance(value, str):
                match = self._compiled.fullmatch(value)
            elif pd.api.types.is_scalar(value) and bool(pd.isna(value)):
                match = None
            else:
                raise TypeError(
                    f"PatternExtractor source value at row {index} must be "
                    "a string or a missing value."
                )
            if match is None:
                unmatched += 1
                if self.on_unmatched == "fail":
                    raise OperatorError(
                        f"PatternExtractor fullmatch failed at input row {index}."
                    )
            for group_name in columns:
                capture = None if match is None else match.group(group_name)
                columns[group_name].append(
                    self.fill_value if capture is None else capture
                )

        if unmatched and self.on_unmatched == "warn":
            warnings.warn(
                f"PatternExtractor: {unmatched} of {len(text)} input rows "
                "did not fully match; applied fill_value.",
                UserWarning,
                stacklevel=2,
            )

        output: dict[str, pd.DataFrame] = {}
        for domain in ("data", "metadata"):
            fields = {
                field: pd.Series(columns[group_name], index=text.index, dtype="string")
                for group_name, destination, field in self._destinations
                if destination == domain
            }
            if fields:
                output[domain] = pd.DataFrame(fields, index=text.index)
        return output

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

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

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        if params:
            raise OperatorError(
                "PatternExtractor does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        source = _single_source(sources)
        # Structural key labels are immutable across query scopes.
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
        if source.artifact_type.value not in {"table", "jsonl"}:
            raise OperatorError("PatternExtractor requires a table or jsonl source.")

        scope = QueryScope(source, metadata_mode=self.metadata_mode)
        available = scope.columns()
        matches = [
            column for column in available.get("columns", [])
            if column.get("namespace") in {"data", "metadata"}
            and column.get("output_name") == self.field
        ]
        if len(matches) != 1:
            choices = [
                str(column["output_name"])
                for column in available.get("columns", [])
                if column.get("namespace") in {"data", "metadata"}
            ]
            raise OperatorError(
                f"PatternExtractor field {self.field!r} is unavailable or ambiguous "
                f"with metadata_mode={self.metadata_mode!r}; "
                f"available fields: {sorted(choices)}."
            )
        selected = matches[0]
        self._source_field = str(selected["output_name"])
        namespace = str(selected["namespace"])

        # A data-only extraction must not require a broader metadata scope
        # that can be invalid even when this operation's narrower scope works.
        # Metadata destinations alone need the inherited-metadata overlap check.
        metadata_outputs = {
            field for _name, domain, field in self._destinations
            if domain == "metadata"
        }
        if metadata_outputs:
            full_columns = QueryScope(
                source, include_data=False, metadata_mode="full"
            ).columns()
            existing_metadata = {
                str(col["base_name"])
                for col in full_columns.get("columns", [])
                if col.get("namespace") == "metadata"
            }
            metadata_overlap = sorted(metadata_outputs.intersection(existing_metadata))
            if metadata_overlap:
                raise OperatorError(
                    "PatternExtractor metadata output conflicts with existing source "
                    f"metadata: {metadata_overlap}."
                )

        return scope.source_request(
            artifact_type=("table", "jsonl"),
            mode="batches",
            columns=ColumnRequest(
                keys=True,
                data=self._source_field if namespace == "data" else False,
                metadata=self._source_field if namespace == "metadata" else False,
            ),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
            form="table",
            include_position=False,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported PatternExtractor mode {mode!r}.")
        if set(inputs) != {DEFAULT_SOURCE_LABEL}:
            raise OperatorError("PatternExtractor expects one 'source' batch.")
        packet = inputs[DEFAULT_SOURCE_LABEL]
        if not isinstance(packet.data, pd.DataFrame):
            raise ArtifactError("PatternExtractor expected a table-form DataFrame batch.")
        frame = packet.data
        # Structural key labels are never qualified in QueryScope.
        key_names = list(packet.primary_key)
        missing_keys = [name for name in key_names if name not in frame.columns]
        if missing_keys:
            raise ArtifactError(
                f"PatternExtractor source batch is missing key column(s) {missing_keys}."
            )
        keys = frame.loc[:, key_names].copy().reset_index(drop=True)
        source_field = self._source_field or self.field
        if source_field not in frame:
            raise ArtifactError(f"PatternExtractor source batch is missing {source_field!r}.")
        extracted = self.translate(frame[source_field])
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    **{
                        domain: values.reset_index(drop=True)
                        for domain, values in extracted.items()
                    },
                }
            }
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
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def make_translate_worker(
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> PatternExtractor:
        _ = request
        if mode != "translate":
            raise OperatorError("PatternExtractor workers support translate mode only.")
        worker = self.from_json_state(self.to_json_state())
        worker._source_field = self._source_field
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "field": self.field,
            "metadata_mode": self.metadata_mode,
            "on_unmatched": self.on_unmatched,
            "fill_value": self.fill_value,
            "flags": list(self.flags),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> PatternExtractor:
        return cls(
            pattern=cast(str, state["pattern"]),
            field=cast(str, state.get("field", state.get("text_field", "text"))),
            metadata_mode=cast(MetadataMode, state.get("metadata_mode", "none")),
            on_unmatched=cast(UnmatchedPolicy, state.get("on_unmatched", "fill")),
            fill_value=cast(str | None, state.get("fill_value", "")),
            flags=cast(Sequence[str], state.get("flags", ())),
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
        state = self.to_json_state()
        state["operator_id"] = operator_id
        state["source_field"] = self._source_field
        (intermediate_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> PatternExtractor:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(state)
        obj._source_field = state.get("source_field")
        obj.operator_id = operator_id
        return obj


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("PatternExtractor requires exactly one 'source' artifact.")
    return sources[DEFAULT_SOURCE_LABEL]
