"""Delimiter-based decomposition translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BatchResult,
    BaseTranslator,
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


class DelimiterDecomposer(BaseTranslator):
    """Split one text row into ordered child rows using a literal delimiter.

    The source key is extended with ``new_key``.  Splitting, optional stripping,
    empty-value filtering, and child numbering are implemented with pandas
    vectorized operations rather than Python row loops.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        delimiter: str = "\n",
        new_key: str = "segment_id",
        text_field: str = "text",
        output_text_field: str | None = None,
        drop_empty: bool = True,
        strip: bool = True,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(delimiter, str) or delimiter == "":
            raise ValueError("delimiter must be a non-empty string.")
        if not isinstance(new_key, str) or not new_key or new_key.startswith("_"):
            raise ValueError("new_key must be a non-empty non-structural column name.")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        output_field = text_field if output_text_field is None else output_text_field
        if not isinstance(output_field, str) or not output_field:
            raise ValueError("output_text_field must be a non-empty string.")

        self.delimiter = delimiter
        self.new_key = new_key
        self.text_field = text_field
        self.output_text_field = output_field
        self.drop_empty = bool(drop_empty)
        self.strip = bool(strip)

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = _single_source(sources)
        if self.new_key in source.primary_key:
            raise OperatorError(
                f"DelimiterDecomposer new_key {self.new_key!r} already exists in "
                f"source primary key {tuple(source.primary_key)!r}."
            )
        return OutputSpec(
            artifact_type="table",
            lineage_mode="extended_key",
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
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                f"DelimiterDecomposer does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("DelimiterDecomposer requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
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
        _ = mode, request
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        key_columns = [str(name) for name in packet.primary_key]
        missing = [name for name in [*key_columns, self.text_field] if name not in frame.columns]
        if missing:
            raise ArtifactError(
                f"DelimiterDecomposer source batch is missing columns {missing}."
            )
        if self.new_key in frame.columns:
            raise ArtifactError(
                f"DelimiterDecomposer new_key {self.new_key!r} collides with a source column."
            )

        working = frame.loc[:, [*key_columns, self.text_field]].copy()
        working["__teal_source_row"] = np.arange(len(working), dtype="int64")
        text = working[self.text_field].fillna("").astype("string")
        working["__teal_part"] = text.str.split(self.delimiter, regex=False)
        exploded = working.explode("__teal_part", ignore_index=True)

        values = exploded["__teal_part"].astype("string")
        if self.strip:
            values = values.str.strip()
        exploded["__teal_part"] = values
        if self.drop_empty:
            exploded = exploded.loc[exploded["__teal_part"].fillna("").ne("")].copy()

        if exploded.empty:
            return BatchResult(outputs={})

        exploded[self.new_key] = (
            exploded.groupby("__teal_source_row", sort=False, dropna=False)
            .cumcount()
            .astype("int64")
        )
        keys = exploded.loc[:, [*key_columns, self.new_key]].reset_index(drop=True)
        data = exploded.loc[:, ["__teal_part"]].rename(
            columns={"__teal_part": self.output_text_field}
        ).reset_index(drop=True)
        return BatchResult(
            outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, "data": data}}
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
        return result.outputs or None

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
    ) -> "DelimiterDecomposer":
        _ = mode, request
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "delimiter": self.delimiter,
            "new_key": self.new_key,
            "text_field": self.text_field,
            "output_text_field": self.output_text_field,
            "drop_empty": self.drop_empty,
            "strip": self.strip,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "DelimiterDecomposer":
        return cls(
            delimiter=str(state.get("delimiter", "\n")),
            new_key=str(state.get("new_key", "segment_id")),
            text_field=str(state.get("text_field", "text")),
            output_text_field=str(state.get("output_text_field", state.get("text_field", "text"))),
            drop_empty=bool(state.get("drop_empty", True)),
            strip=bool(state.get("strip", True)),
        )

    def save_intermediate_state(
        self,
        intermediate_dir,
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
        intermediate_dir,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> "DelimiterDecomposer":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        return obj


def _single_source(sources: Mapping[str, "BaseArtifact"]) -> "BaseArtifact":
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"DelimiterDecomposer requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"DelimiterDecomposer expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"DelimiterDecomposer expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
