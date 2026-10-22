"""Tests for declarative PatternExtractor capture routing and TeAL batch semantics."""

import re
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import PatternExtractor


PATTERN = r"(?P<metadata_speaker>[^:]+):\s*(?P<data_utterance>.*)"


def test_standalone_named_groups_and_unnamed_captures():
    translator = PatternExtractor(r"(Speaker\s+)(?P<metadata_speaker>\d+):\s*(?P<data_utterance>.*)")
    data = pd.Series(["Speaker 1: Hello", "Speaker 2: Bonjour"], index=[9, 12])
    output = translator.translate(data)
    assert set(output) == {"data", "metadata"}
    assert list(output["data"].columns) == ["utterance"]
    assert output["data"]["utterance"].tolist() == ["Hello", "Bonjour"]
    assert output["metadata"]["speaker"].tolist() == ["1", "2"]
    assert output["data"].index.tolist() == [9, 12]


def test_fullmatch_only_fill_and_nulls():
    source = pd.Series(["Ada: Hello", "prefix Ada: Hello trailing", None, "Bob: "])
    output = PatternExtractor(PATTERN).translate(source)
    assert output["metadata"]["speaker"].tolist() == ["Ada", "", "", "Bob"]
    assert output["data"]["utterance"].tolist() == ["Hello", "", "", ""]
    nulls = PatternExtractor(PATTERN, fill_value=None).translate(source)
    assert pd.isna(nulls["metadata"]["speaker"].iloc[1])
    assert pd.isna(nulls["data"]["utterance"].iloc[2])


def test_optional_groups_and_custom_fill():
    p = r"(?P<data_word>\w+)(?:-(?P<metadata_suffix>\w+))?"
    out = PatternExtractor(p, fill_value="UNKNOWN").translate(pd.Series(["abc", "abc-xyz"]))
    assert out["metadata"]["suffix"].tolist() == ["UNKNOWN", "xyz"]


def test_warn_and_fail():
    source = pd.Series(["A: good", "no delimiter", None])
    with pytest.warns(UserWarning, match="2 of 3"):
        out = PatternExtractor(PATTERN, on_unmatched="warn").translate(source)
    assert out["data"]["utterance"].tolist() == ["good", "", ""]
    with pytest.raises(OperatorError, match="row 1"):
        PatternExtractor(PATTERN, on_unmatched="fail").translate(source)


@pytest.mark.parametrize(
    "pattern,problem",
    [
        (r"(?P<speaker>\w+)", "Invalid named capture"),
        (r"(?P<other_field>\w+)", "Invalid named capture"),
        (r"(?P<data_>\w+)", "field"),
        (r"(?P<data__position>\w+)", "reserved"),
        (r"(\w+)", "at least one named"),
        (r"(?P<data_name>\w+)(?P<data_name>\w+)", "redefinition"),
        (r"(?P<data_name>\w+)(?P=missing)", "unknown group"),
    ],
)
def test_compile_validation(pattern, problem):
    with pytest.raises((ValueError, re.error), match=problem):
        PatternExtractor(pattern)


def test_backreferences_character_classes_and_flags():
    p = r"(?P<data_word>\w+)\s+(?P=data_word)"
    t = PatternExtractor(p, flags=("IGNORECASE",))
    assert t.translate(pd.Series(["Hi hi"]))["data"]["word"].iloc[0] == "Hi"
    t = PatternExtractor(r"[(?P<data_word>](?P<metadata_tag>\w+)")
    assert t.translate(pd.Series(["(ok"]))["metadata"]["tag"].iloc[0] == "ok"


def test_state_roundtrip_and_packet():
    t = PatternExtractor(PATTERN, on_unmatched="warn", fill_value="?", flags=["IGNORECASE"])
    restored = PatternExtractor.from_json_state(t.to_json_state())
    assert restored.to_json_state() == t.to_json_state()
    values = pd.DataFrame({"text": ["A: yes", "bad"], "other": [10, 20]})
    with pytest.warns(UserWarning):
        preview = restored.translate({"data": values, "metadata": pd.DataFrame({"old": [1, 2]})})
    assert set(preview) == {"data", "metadata"}
    assert preview["data"].columns.tolist() == ["utterance"]
    batch = InputBatch(
        source_label="source", artifact_id="art_source", primary_key=("row_id",),
        data=values.assign(row_id=[5, 6]), batch_index=0, batch_count=1,
        is_first=True, is_last=True,
    )
    with pytest.warns(UserWarning):
        payload = restored.translate_batch(
            {"source": batch}, mode="translate", request=TranslationRequest()
        ).outputs["output"]
    assert payload["keys"]["row_id"].tolist() == [5, 6]
    assert payload["data"]["utterance"].tolist() == ["yes", "?"]
    assert payload["metadata"]["speaker"].tolist() == ["A", "?"]


def test_source_request_does_not_copy_data_or_metadata():
    t = PatternExtractor(PATTERN)
    src = SimpleNamespace(
        artifact_type=ArtifactType.TABLE,
        primary_key=("row_id",),
        query_columns=lambda **kwargs: {
            "data": ["text", "other"], "metadata": ["old"],
            "columns": [{"namespace": "metadata", "base_name": "old"}],
        },
    )
    req = t.input_request(sources={"source": src}, mode="translate", request=TranslationRequest())
    assert req.columns.data == "text"
    assert req.columns.metadata is False
    assert t.output_specs(sources={"source": src}, request=TranslationRequest()).lineage_mode == "preserved_key"
    collision_src = SimpleNamespace(
        artifact_type=ArtifactType.TABLE,
        primary_key=("row_id",),
        query_columns=lambda **kwargs: {
            "data": ["text"], "metadata": ["speaker"],
            "columns": [{"namespace": "metadata", "base_name": "speaker"}],
        },
    )
    with pytest.raises(OperatorError, match="conflicts"):
        t.input_request(sources={"source": collision_src}, mode="translate", request=TranslationRequest())
