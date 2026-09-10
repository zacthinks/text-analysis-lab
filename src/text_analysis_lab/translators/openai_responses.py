"""Thin OpenAI Responses API translator for row-wise LLM analysis.

The translator keeps credentials outside TeAL state, freezes the prompt/model
configuration as an ordinary operator snapshot, and deliberately processes one
source row per TeAL batch so completed API calls are durably checkpointed before
another billable request begins.
"""

from __future__ import annotations

import json
import os
import string
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

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


class OpenAIResponsesTranslator(BaseTranslator):
    """Apply one OpenAI Responses API request to each source observation.

    ``prompt`` uses simple ``str.format`` placeholders naming TeAL primary keys,
    data fields, or inherited metadata fields, for example
    ``"Classify this text:\n\n{text}"``. Credentials are read at execution time
    from ``api_key_env`` and are never serialized into the project.

    When ``json_schema`` is supplied, TeAL requests strict Structured Outputs
    and materializes the schema's top-level properties as data columns.
    Otherwise one plain-text ``output_field`` is produced.
    """

    operation_type = "translate"

    def __init__(
        self,
        model: str,
        prompt: str,
        *,
        instructions: str | None = None,
        json_schema: Mapping[str, Any] | None = None,
        schema_name: str = "teal_response",
        output_field: str = "response",
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        store: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty OpenAI model id.")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty template string.")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None.")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("api_key_env must be a non-empty environment-variable name.")
        if max_output_tokens is not None and int(max_output_tokens) <= 0:
            raise ValueError("max_output_tokens must be positive or None.")
        if temperature is not None and not (0.0 <= float(temperature) <= 2.0):
            raise ValueError("temperature must satisfy 0 <= temperature <= 2 or be None.")
        if not isinstance(output_field, str) or not output_field:
            raise ValueError("output_field must be a non-empty string.")
        if not isinstance(schema_name, str) or not schema_name:
            raise ValueError("schema_name must be a non-empty string.")

        self.model = model.strip()
        self.prompt = prompt
        self.instructions = instructions
        self.json_schema = _normalize_json_schema(json_schema)
        self.schema_name = schema_name
        self.output_field = output_field
        self.api_key_env = api_key_env
        self.max_output_tokens = None if max_output_tokens is None else int(max_output_tokens)
        self.reasoning_effort = None if reasoning_effort is None else str(reasoning_effort)
        self.temperature = None if temperature is None else float(temperature)
        self.store = bool(store)
        self.template_fields = _template_fields(prompt)
        self._data_fields: tuple[str, ...] = ()
        self._metadata_fields: tuple[str, ...] = ()

    @property
    def supports_parallel_translate(self) -> bool:
        # Keep the MVP deterministic and cost-transparent. TeAL can add bounded
        # request concurrency later without changing the operator contract.
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route == "sequential"

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
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
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        if params:
            raise OperatorError(
                f"OpenAIResponsesTranslator does not accept operation parameters; got {sorted(params)}."
            )
        source = _single_source(sources)
        if source.artifact_type.value not in {"table", "jsonl"}:
            raise OperatorError("OpenAIResponsesTranslator requires a table or jsonl source.")
        self._bind_template_fields(source)
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode, request
        source = _single_source(sources)
        self._bind_template_fields(source)
        return SourceRequest(
            artifact_type=("table", "jsonl"),
            mode="batches",
            columns=ColumnRequest(
                keys=True,
                data=list(self._data_fields) if self._data_fields else False,
                metadata=list(self._metadata_fields) if self._metadata_fields else False,
            ),
            # One billable request per durable TeAL batch gives true per-row resume.
            batch_size=1,
            form="table",
            metadata_mode="full" if self._metadata_fields else "none",
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
            raise OperatorError(f"Unsupported OpenAIResponsesTranslator mode {mode!r}.")
        packet = _single_input(inputs)
        if not isinstance(packet.data, pd.DataFrame):
            raise ArtifactError("OpenAIResponsesTranslator expected a table input batch.")
        frame = packet.data.reset_index(drop=True)
        if len(frame) != 1:
            raise ArtifactError(
                "OpenAIResponsesTranslator requires exactly one input row per execution batch "
                "so completed API calls can be resumed without duplicate billing."
            )
        row = frame.iloc[0].to_dict()
        try:
            rendered = self.prompt.format_map(_StrictFormatMap(row))
        except KeyError as exc:
            raise ArtifactError(f"Prompt template field {exc.args[0]!r} is unavailable in the input row.") from exc
        response = self._request(rendered)
        data = self._response_data(response)
        metadata = pd.DataFrame([_response_metadata(response, requested_model=self.model)])
        keys = frame.loc[:, list(packet.primary_key)].reset_index(drop=True)
        return BatchResult(
            outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, "data": data, "metadata": metadata}}
        )

    def _request(self, rendered_prompt: str) -> Any:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise OperatorError(
                f"OpenAIResponsesTranslator requires API credentials in environment variable "
                f"{self.api_key_env!r}; credentials are intentionally not stored in TeAL."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OperatorError(
                "OpenAIResponsesTranslator requires the optional OpenAI client. Install with "
                "`uv sync --extra llm` or `pip install text-analysis-lab[llm]`."
            ) from exc
        client = OpenAI(api_key=api_key)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": rendered_prompt,
            "store": self.store,
        }
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "schema": self.json_schema,
                    "strict": True,
                }
            }
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:
            raise OperatorError(
                f"OpenAI Responses API request failed for model {self.model!r}: {exc}"
            ) from exc

    def _response_data(self, response: Any) -> pd.DataFrame:
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            raise OperatorError("OpenAI response did not expose a text output via response.output_text.")
        if self.json_schema is None:
            return pd.DataFrame([{self.output_field: text}])
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OperatorError("OpenAI Structured Output was not valid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise OperatorError("OpenAI Structured Output must be a top-level JSON object.")
        fields = self._structured_fields()
        missing = [field for field in fields if field not in payload]
        if missing:
            raise OperatorError(f"OpenAI Structured Output omitted schema field(s) {missing}.")
        return pd.DataFrame([{field: payload[field] for field in fields}])

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

    def to_json_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "instructions": self.instructions,
            "json_schema": self.json_schema,
            "schema_name": self.schema_name,
            "output_field": self.output_field,
            "api_key_env": self.api_key_env,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "store": self.store,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "OpenAIResponsesTranslator":
        return cls(
            model=str(state.get("model", "")),
            prompt=str(state.get("prompt", "")),
            instructions=cast(str | None, state.get("instructions")),
            json_schema=cast(Mapping[str, Any] | None, state.get("json_schema")),
            schema_name=str(state.get("schema_name", "teal_response")),
            output_field=str(state.get("output_field", "response")),
            api_key_env=str(state.get("api_key_env", "OPENAI_API_KEY")),
            max_output_tokens=cast(int | None, state.get("max_output_tokens")),
            reasoning_effort=cast(str | None, state.get("reasoning_effort")),
            temperature=cast(float | None, state.get("temperature")),
            store=bool(state.get("store", False)),
        )

    def _bind_template_fields(self, source: "BaseArtifact") -> None:
        keys = {str(value) for value in source.primary_key}
        data = {str(value) for value in source.get_data_columns()}
        metadata = {str(value) for value in source.get_full_metadata_columns()}
        unknown = sorted(self.template_fields - keys - data - metadata)
        if unknown:
            raise OperatorError(
                f"Prompt template references unknown source field(s) {unknown}; "
                f"keys={sorted(keys)}, data={sorted(data)}, metadata={sorted(metadata)}."
            )
        self._data_fields = tuple(sorted(self.template_fields & data))
        self._metadata_fields = tuple(sorted((self.template_fields - data - keys) & metadata))

    def _structured_fields(self) -> tuple[str, ...]:
        assert self.json_schema is not None
        props = self.json_schema.get("properties", {})
        return tuple(str(value) for value in props)


class _StrictFormatMap(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def _single_source(sources: Mapping[str, Any]) -> Any:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("OpenAIResponsesTranslator requires exactly one source under 'source'.")
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError("OpenAIResponsesTranslator expected exactly one input under 'source'.")
    return inputs[DEFAULT_SOURCE_LABEL]


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("Prompt template contains an empty replacement field.")
        if any(token in field_name for token in (".", "[", "]")):
            raise ValueError(
                "Prompt placeholders must be simple TeAL field names; attribute/index access is not supported."
            )
        if format_spec or conversion:
            raise ValueError("Prompt placeholders do not support format specs or conversions.")
        fields.add(field_name)
    return fields


def _normalize_json_schema(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = json.loads(json.dumps(dict(value)))
    if payload.get("type") != "object":
        raise ValueError("json_schema must describe a top-level object.")
    props = payload.get("properties")
    if not isinstance(props, Mapping) or not props:
        raise ValueError("json_schema must contain a non-empty top-level properties mapping.")
    return payload


def _response_metadata(response: Any, *, requested_model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "openai_response_id": getattr(response, "id", None),
        "openai_requested_model": requested_model,
        "openai_response_model": getattr(response, "model", None),
        "openai_input_tokens": getattr(usage, "input_tokens", None) if usage is not None else None,
        "openai_output_tokens": getattr(usage, "output_tokens", None) if usage is not None else None,
        "openai_total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
    }
