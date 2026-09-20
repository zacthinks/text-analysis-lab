from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import OpenAIResponsesTranslator


class _Source:
    artifact_type = SimpleNamespace(value="table")
    primary_key = ["row_id"]

    def get_data_columns(self):
        return ["text"]

    def get_full_metadata_columns(self):
        return ["party"]


class _FakeResponses:
    calls = []
    output_text = "ok"

    def create(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            model="gpt-test-resolved",
            output_text=self.__class__.output_text,
            usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
        )


class _FakeOpenAI:
    def __init__(self, *, api_key):
        assert api_key == "secret"
        self.responses = _FakeResponses()


def _install_fake_openai(monkeypatch):
    _FakeResponses.calls.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))


def _batch():
    return InputBatch(
        source_label="source",
        artifact_id="source1",
        primary_key=("row_id",),
        data=pd.DataFrame({"row_id": [7], "text": ["hello"], "party": ["D"]}),
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_openai_translator_plain_text_templates_fields_and_checkpoints_per_row(
    monkeypatch,
):
    _install_fake_openai(monkeypatch)
    translator = OpenAIResponsesTranslator(
        "gpt-test",
        "Party={party}; text={text}; id={row_id}",
        instructions="Return a short answer.",
        max_output_tokens=20,
        reasoning_effort="low",
        store=False,
    )
    source = _Source()
    translator.validate_operation_params(
        {}, sources={"source": source}, mode="translate"
    )
    request = translator.input_request(
        sources={"source": source}, mode="translate", request=TranslationRequest()
    )
    assert request.batch_size == 1
    assert request.metadata_mode == "full"
    assert translator.supports_resume(mode="translate", route="sequential")

    result = translator.translate_batch(
        {"source": _batch()}, mode="translate", request=TranslationRequest()
    ).outputs["output"]
    assert result["data"].iloc[0]["response"] == "ok"
    assert result["metadata"].iloc[0]["openai_response_id"] == "resp_test"
    call = _FakeResponses.calls[0]
    assert call["input"] == "Party=D; text=hello; id=7"
    assert call["instructions"] == "Return a short answer."
    assert call["reasoning"] == {"effort": "low"}
    assert call["store"] is False


def test_openai_translator_structured_outputs_uses_json_schema(monkeypatch):
    _install_fake_openai(monkeypatch)
    _FakeResponses.output_text = json.dumps({"label": 1, "reason": "clear"})
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    }
    translator = OpenAIResponsesTranslator(
        "gpt-test",
        "Classify {text}",
        json_schema=schema,
        schema_name="binary_code",
    )
    translator.validate_operation_params(
        {}, sources={"source": _Source()}, mode="translate"
    )
    result = translator.translate_batch(
        {"source": _batch()}, mode="translate", request=TranslationRequest()
    ).outputs["output"]
    assert result["data"].iloc[0].to_dict() == {"label": 1, "reason": "clear"}
    fmt = _FakeResponses.calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == schema


def test_openai_translator_does_not_store_credentials_and_requires_env(monkeypatch):
    translator = OpenAIResponsesTranslator("gpt-test", "{text}", api_key_env="MY_KEY")
    state = translator.to_json_state()
    assert "api_key" not in state
    assert state["api_key_env"] == "MY_KEY"
    restored = OpenAIResponsesTranslator.from_json_state(state)
    assert restored.prompt == "{text}"
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(Exception, match="credentials"):
        restored._request("hello")
