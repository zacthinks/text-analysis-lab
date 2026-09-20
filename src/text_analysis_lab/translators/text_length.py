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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL

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

    def __init__(
        self,
        lengths: LengthRequest,
        *,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.lengths = _normalize_lengths(lengths)

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

        available = source.query_columns(metadata_mode="full")
        available_data = set(str(name) for name in available["data"])
        missing = [field for field in self.lengths if field not in available_data]
        if missing:
            raise OperatorError(
                f"TextLength source data field(s) are unavailable: {missing}. "
                f"Available data fields: {sorted(available_data)}."
            )
        existing = set(str(name) for name in available["output"])
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
            columns=ColumnRequest(keys=True, data=list(self.lengths), metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
            form="table",
            metadata_mode="none",
            include_position=False,
        )

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{field}_{unit}" for field, units in self.lengths.items() for unit in units
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
            raise OperatorError(f"Unsupported TextLength mode {mode!r}.")
        packet = _single_input(inputs)
        if not isinstance(packet.data, pd.DataFrame):
            raise ArtifactError("TextLength expected a table-form pandas batch.")
        frame = packet.data
        key_columns = [str(name) for name in packet.primary_key]
        missing = [name for name in [*key_columns, *self.lengths] if name not in frame]
        if missing:
            raise ArtifactError(
                f"TextLength source batch is missing columns {missing}."
            )

        metadata: dict[str, Any] = {}
        for field, units in self.lengths.items():
            source = frame[field].fillna("").astype("string")
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

        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": frame.loc[:, key_columns].reset_index(drop=True),
                    "metadata": pd.DataFrame(metadata),
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
        return TextLength(self.lengths)

    def to_json_state(self) -> dict[str, Any]:
        return {
            "lengths": {field: list(units) for field, units in self.lengths.items()}
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> TextLength:
        raw = state.get("lengths")
        if not isinstance(raw, Mapping):
            raise OperatorError("TextLength state is missing its lengths mapping.")
        return cls(cast(LengthRequest, raw))

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
