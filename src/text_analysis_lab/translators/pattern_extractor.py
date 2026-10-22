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

    Standalone translate accepts a Series, a DataFrame containing text_field, or
    a {"data": DataFrame, "metadata": DataFrame} preview packet. It returns
    component-separated DataFrames. Persisted execution adds TeAL-managed keys,
    and the source's other columns remain available through lineage rather than
    being duplicated.
    """

    operation_type = "translate"

    def __init__(
        self,
        pattern: str,
        *,
        text_field: str = "text",
        on_unmatched: UnmatchedPolicy = "fill",
        fill_value: str | None = "",
        flags: Sequence[str] | str | int = (),
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("PatternExtractor pattern must be a non-empty string.")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        if on_unmatched not in {"fill", "warn", "fail"}:
            raise ValueError("on_unmatched must be 'fill', 'warn', or 'fail'.")
        if fill_value is not None and not isinstance(fill_value, str):
            raise TypeError("fill_value must be a string or None.")

        self.pattern = pattern
        self.text_field = text_field
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
                if not isinstance(data, pd.DataFrame):
                    raise TypeError(
                        "PatternExtractor preview packets require a 'data' DataFrame."
                    )
                if metadata is not None and (
                    not isinstance(metadata, pd.DataFrame) or len(metadata) != len(data)
                ):
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
            if self.text_field not in data.columns:
                raise ValueError(
                    f"PatternExtractor input is missing data field {self.text_field!r}."
                )
            if not data.columns.is_unique:
                raise ValueError("PatternExtractor input DataFrame columns must be unique.")
            text = data[self.text_field]

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
        _ = sources, mode
        if params:
            raise OperatorError(
                "PatternExtractor does not accept operation parameters; "
                f"got {sorted(params)}."
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
        if source.artifact_type.value not in {"table", "jsonl"}:
            raise OperatorError("PatternExtractor requires a table or jsonl source.")
        self._validate_key_collisions(source.primary_key)

        available = source.query_columns(metadata_mode="full")
        available_data = {str(value) for value in available["data"]}
        if self.text_field not in available_data:
            raise OperatorError(
                f"PatternExtractor source data field {self.text_field!r} is unavailable; "
                f"available fields: {sorted(available_data)}."
            )

        # Creating a fresh data representation is allowed (e.g. data_text).
        # Creating same-named local metadata that shadows inherited metadata is
        # not: full-lineage queries would then be ambiguous.
        existing_metadata = {
            str(col["base_name"])
            for col in available.get("columns", [])
            if col.get("namespace") == "metadata"
        }
        metadata_overlap = sorted(
            field for _name, domain, field in self._destinations
            if domain == "metadata" and field in existing_metadata
        )
        if metadata_overlap:
            raise OperatorError(
                "PatternExtractor metadata output conflicts with existing source "
                f"metadata: {metadata_overlap}."
            )

        return SourceRequest(
            artifact_type=("table", "jsonl"),
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
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
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported PatternExtractor mode {mode!r}.")
        if set(inputs) != {DEFAULT_SOURCE_LABEL}:
            raise OperatorError("PatternExtractor expects one 'source' batch.")
        packet = inputs[DEFAULT_SOURCE_LABEL]
        if not isinstance(packet.data, pd.DataFrame):
            raise ArtifactError("PatternExtractor expected a table-form DataFrame batch.")
        frame = packet.data
        keys = [str(name) for name in packet.primary_key]
        self._validate_key_collisions(keys)
        missing = [name for name in [*keys, self.text_field] if name not in frame]
        if missing:
            raise ArtifactError(f"PatternExtractor source batch is missing {missing}.")
        extracted = self.translate(frame[self.text_field])
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": frame.loc[:, keys].reset_index(drop=True),
                    **{
                        domain: values.reset_index(drop=True)
                        for domain, values in extracted.items()
                    },
                }
            }
        )

    def _validate_key_collisions(self, primary_key: Sequence[str]) -> None:
        overlap = sorted(
            {
                field
                for _name, _domain, field in self._destinations
                if field in set(primary_key)
            }
        )
        if overlap:
            raise OperatorError(
                f"PatternExtractor output fields collide with primary keys: {overlap}."
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
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "text_field": self.text_field,
            "on_unmatched": self.on_unmatched,
            "fill_value": self.fill_value,
            "flags": list(self.flags),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> PatternExtractor:
        return cls(
            pattern=cast(str, state["pattern"]),
            text_field=cast(str, state.get("text_field", "text")),
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
        obj.operator_id = operator_id
        return obj


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("PatternExtractor requires exactly one 'source' artifact.")
    return sources[DEFAULT_SOURCE_LABEL]
