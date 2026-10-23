"""Preserved-key text-length metadata translation for TeAL artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL, MetadataMode
from text_analysis_lab.translators._field_selection import resolve_text_fields

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


LengthUnit = Literal["characters", "words", "log_characters", "log_words"]
LengthRequest = Mapping[str, LengthUnit | Sequence[LengthUnit]]
_SUPPORTED_UNITS = ("characters", "words", "log_characters", "log_words")


class TextLength(BaseTranslator):
    """Attach one or more text-length measures as local metadata.

    ``lengths`` maps each source text field to one unit or a sequence of units.
    Output metadata names are deterministic: ``<field>_<unit>``. The output owns
    keys and the requested metadata only; source representation data and earlier
    metadata remain available lazily through preserved-key lineage.
    """

    operation_type = "translate"
    frozen_runtime_fields = frozenset({"_field_namespaces"})

    def __init__(
        self,
        lengths: LengthRequest | None = None,
        *,
        fields: Sequence[str] | None = None,
        units: LengthUnit | Sequence[LengthUnit] = "characters",
        output_field: str | None = None,
        delimiter: str = " ",
        metadata_mode: MetadataMode = "none",
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if metadata_mode not in {"none", "local", "full"}:
            raise ValueError("metadata_mode must be 'none', 'local', or 'full'.")
        if (lengths is None) == (fields is None):
            raise ValueError("Provide exactly one of lengths or fields.")
        if not isinstance(delimiter, str):
            raise TypeError("delimiter must be a string.")
        self.combined_fields: tuple[str, ...] | None = None
        self.output_field = output_field
        self.delimiter = delimiter
        self.metadata_mode = metadata_mode
        self._field_namespaces: dict[str, str] = {}
        if fields is not None:
            if isinstance(fields, str) or not fields or any(not isinstance(v, str) or not v for v in fields):
                raise ValueError("fields must be a nonempty sequence of field labels.")
            if len(set(fields)) != len(fields):
                raise ValueError("fields cannot contain repeated labels.")
            if not isinstance(output_field, str) or not output_field:
                raise ValueError("Combined TextLength needs an explicit output_field.")
            self.combined_fields = tuple(fields)
            self.lengths = _normalize_lengths({output_field: units})
        else:
            if output_field is not None or delimiter != " ":
                raise ValueError("output_field and delimiter require combined fields.")
            self.lengths = _normalize_lengths(cast(LengthRequest, lengths))

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
                f"TextLength does not accept operation parameters; got {sorted(params)}."
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
            raise OperatorError("TextLength requires a table or jsonl artifact source.")

        selected = list(self.combined_fields) if self.combined_fields else list(self.lengths)
        namespaces = resolve_text_fields(source, selected, metadata_mode=self.metadata_mode,
                                         translator="TextLength")
        self._field_namespaces = namespaces
        available = source.query_columns(metadata_mode="full")
        existing = {str(name) for name in available["output"]}
        collisions = sorted(existing.intersection(self.output_columns))
        if collisions:
            raise OperatorError(
                "TextLength output metadata would collide with existing source "
                f"columns: {collisions}. These columns may be inherited from an "
                "earlier TextLength artifact; reuse the existing metadata or request "
                "only genuinely missing length units."
            )

        return SourceRequest(
            artifact_type=("table", "jsonl"),
            mode="batches",
            columns=ColumnRequest(
                keys=True,
                data=[n for n, kind in namespaces.items() if kind == "data"] or False,
                metadata=[n for n, kind in namespaces.items() if kind == "metadata"] or False,
            ),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
            form="table",
            metadata_mode=self.metadata_mode,
            include_position=False,
        )

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{field}_{unit}" for field, units in self.lengths.items() for unit in units
        )

    def translate(
        self,
        data: str | None | Mapping[str, Any] | pd.DataFrame,
    ) -> dict[str, Any] | pd.DataFrame:
        """Measure ordinary text records without creating TeAL artifacts."""

        return_record = not isinstance(data, pd.DataFrame)
        if isinstance(data, (str, type(None))):
            if len(self.lengths) != 1:
                raise ValueError(
                    "Scalar TextLength translation requires exactly one configured text field."
                )
            field = next(iter(self.lengths))
            frame = pd.DataFrame({field: [data]})
        elif isinstance(data, Mapping):
            frame = pd.DataFrame([dict(data)])
        elif isinstance(data, pd.DataFrame):
            frame = data
        else:
            raise TypeError(
                "TextLength.translate(...) expects text, a mapping record, or a pandas DataFrame."
            )

        measured = self._measure_frame(frame)
        if return_record:
            return measured.iloc[0].to_dict()
        return measured

    def _measure_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = self.combined_fields if self.combined_fields is not None else self.lengths
        missing = [field for field in selected if field not in frame.columns]
        if missing:
            raise ValueError(f"TextLength input is missing field(s) {missing}.")

        metadata: dict[str, Any] = {}
        if self.combined_fields is not None:
            assert self.output_field is not None
            pieces = [frame[n].fillna("").astype("string") for n in self.combined_fields]
            joined = pieces[0]
            for piece in pieces[1:]:
                joined = joined + self.delimiter + piece
            sources = {self.output_field: joined}
        else:
            sources = {n: frame[n].fillna("").astype("string") for n in self.lengths}
        for field, units in self.lengths.items():
            source = sources[field]
            for unit in units:
                base_unit = unit.removeprefix("log_")
                if base_unit == "characters":
                    values = source.str.len()
                else:
                    values = source.str.split().str.len()
                integers = values.astype("int64").to_numpy()
                metadata[f"{field}_{unit}"] = (
                    np.log1p(integers.astype("float64"))
                    if unit.startswith("log_")
                    else integers
                )
        return pd.DataFrame(metadata, index=frame.index)

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported TextLength mode {mode!r}.")
        packet = _single_input(inputs)
        if not isinstance(packet.data, pd.DataFrame):
            raise ArtifactError("TextLength expected a table-form pandas batch.")
        frame = packet.data
        key_columns = [str(name) for name in packet.primary_key]
        selected = self.combined_fields if self.combined_fields is not None else self.lengths
        missing = [name for name in [*key_columns, *selected] if name not in frame]
        if missing:
            raise ArtifactError(
                f"TextLength source batch is missing columns {missing}."
            )

        metadata = self._measure_frame(frame).reset_index(drop=True)

        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": frame.loc[:, key_columns].reset_index(drop=True),
                    "metadata": metadata,
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
    ) -> TextLength:
        _ = request
        if mode != "translate":
            raise OperatorError("TextLength workers support translate mode only.")
        worker = self.from_json_state(self.to_json_state())
        worker._field_namespaces = dict(self._field_namespaces)
        return worker

    def to_json_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "lengths": {field: list(units) for field, units in self.lengths.items()},
            "metadata_mode": self.metadata_mode,
        }
        if self.combined_fields is not None:
            state["fields"] = list(self.combined_fields)
            state["output_field"] = self.output_field
            state["delimiter"] = self.delimiter
        return state

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> TextLength:
        raw = state.get("lengths")
        if not isinstance(raw, Mapping):
            raise OperatorError("TextLength state is missing its lengths mapping.")
        metadata_mode = cast(MetadataMode, state.get("metadata_mode", "none"))
        if "fields" in state:
            stem = str(state["output_field"])
            return cls(
                fields=cast(Sequence[str], state["fields"]),
                units=cast(Sequence[LengthUnit], raw[stem]),
                output_field=stem,
                delimiter=str(state.get("delimiter", " ")),
                metadata_mode=metadata_mode,
            )
        return cls(cast(LengthRequest, raw), metadata_mode=metadata_mode)

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        """Persist the stateless configuration needed to resume safely."""

        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_state()
        payload["operator_id"] = operator_id
        payload["field_namespaces"] = self._field_namespaces
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
    ) -> TextLength:
        """Restore a resumable translator from its operation-local state."""

        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        obj._field_namespaces = dict(state.get("field_namespaces", {}))
        return obj


def _normalize_lengths(lengths: LengthRequest) -> dict[str, tuple[LengthUnit, ...]]:
    if not isinstance(lengths, Mapping) or not lengths:
        raise ValueError("lengths must be a non-empty mapping of text fields to units.")
    normalized: dict[str, tuple[LengthUnit, ...]] = {}
    for raw_field, raw_units in lengths.items():
        field = str(raw_field)
        if not field:
            raise ValueError("TextLength field names must be non-empty strings.")
        values = [raw_units] if isinstance(raw_units, str) else list(raw_units)
        if not values:
            raise ValueError(
                f"TextLength field {field!r} must request at least one unit."
            )
        units: list[LengthUnit] = []
        for raw_unit in values:
            unit = str(raw_unit)
            if unit not in _SUPPORTED_UNITS:
                raise ValueError(
                    f"Unsupported TextLength unit {unit!r}; expected one of "
                    f"{list(_SUPPORTED_UNITS)}."
                )
            typed = cast(LengthUnit, unit)
            if typed in units:
                raise ValueError(
                    f"TextLength field {field!r} requests duplicate unit {unit!r}."
                )
            units.append(typed)
        normalized[field] = tuple(units)
    return normalized


def _single_source(sources: Mapping[str, Any]) -> Any:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("TextLength requires exactly one source under 'source'.")
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("TextLength expected exactly one input under 'source'.")
    return inputs[DEFAULT_SOURCE_LABEL]
