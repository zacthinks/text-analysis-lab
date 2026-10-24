"""Tests for declarative PatternExtractor capture routing and TeAL batch semantics."""

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import BaseOperator, InputBatch, TranslationRequest
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
    source = pd.Series(["Ada: Hello", "not a speaker line", None, "Bob: "])
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
            "columns": [{"namespace": "data", "output_name": "text", "base_name": "text"}, {"namespace": "metadata", "output_name": "old", "base_name": "old"}],
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
            "columns": [{"namespace": "data", "output_name": "text", "base_name": "text"}, {"namespace": "metadata", "output_name": "speaker", "base_name": "speaker"}],
        },
    )
    with pytest.raises(OperatorError, match="conflicts"):
        t.input_request(sources={"source": collision_src}, mode="translate", request=TranslationRequest())


def test_fullmatch_rejects_partial_match():
    # "word:word" may occur inside the input, but fullmatch must consume all.
    t = PatternExtractor(r"(?P<metadata_label>\w+):(?P<data_text>\w+)")
    out = t.translate(["pre KEY:VALUE post", "KEY:VALUE"])
    assert out["data"]["text"].tolist() == ["", "VALUE"]


def test_snapshot_reuse_parallel_and_intermediate_state(tmp_path):
    original = PatternExtractor(
        PATTERN, field="text", on_unmatched="fill",
        fill_value=None, flags=("IGNORECASE",),
    )
    assert original.requires_fit is False
    assert original.is_fitted is True
    caps = original.execution_capabilities()
    assert caps.reusable and caps.artifact and caps.native and caps.portable
    assert original.supports_parallel_translate
    assert original.supports_resume(mode="translate", route="sequential")
    assert original.supports_resume(mode="translate", route="parallel")
    assert not original.supports_resume(mode="fit_translate", route="sequential")

    snapshot_dir = tmp_path / "snapshot"
    original.save_to_dir(snapshot_dir, operator_id="op_pattern")
    loaded = BaseOperator.load_from_dir(snapshot_dir)
    assert isinstance(loaded, PatternExtractor)
    assert loaded.is_frozen
    assert loaded.operator_id == "op_pattern"
    assert loaded.to_json_state() == original.to_json_state()
    worker = loaded.make_translate_worker(mode="translate", request=TranslationRequest())
    assert isinstance(worker, PatternExtractor)
    assert worker.to_json_state() == loaded.to_json_state()
    assert worker.translate("Ada: Hello")["data"]["utterance"].iloc[0] == "Hello"
    with pytest.raises(OperatorError, match="workers support translate"):
        loaded.make_translate_worker(mode="fit_translate", request=TranslationRequest())

    checkpoint_dir = tmp_path / "checkpoint"
    loaded.save_intermediate_state(
        checkpoint_dir, operator_id=loaded.operator_id,
        mode="translate", route="sequential",
    )
    recovered = PatternExtractor.load_intermediate_state(
        checkpoint_dir, operator_id=loaded.operator_id,
        mode="translate", route="sequential",
    )
    assert recovered.operator_id == loaded.operator_id
    assert recovered.to_json_state() == loaded.to_json_state()
    assert pd.isna(recovered.translate("bad")["data"]["utterance"].iloc[0])


def test_metadata_only_and_data_only_packets():
    md = PatternExtractor(r"(?P<metadata_speaker>\w+)")
    assert list(md.translate(["Ada"])) == ["metadata"]
    data = PatternExtractor(r"(?P<data_text>\w+)")
    assert list(data.translate(["Ada"])) == ["data"]
    empty = md.translate(pd.Series([], dtype="string"))
    assert len(empty["metadata"]) == 0
    assert empty["metadata"].columns.tolist() == ["speaker"]


@pytest.mark.parametrize("policy", ["fill", "warn", "fail"])
def test_invalid_on_unmatched_value_type_is_rejected(policy):
    assert PatternExtractor(PATTERN, on_unmatched=policy).on_unmatched == policy
    with pytest.raises(ValueError, match="on_unmatched"):
        PatternExtractor(PATTERN, on_unmatched="skip")
    with pytest.raises(TypeError, match="fill_value"):
        PatternExtractor(PATTERN, fill_value=0)

def test_source_selection_scopes_data_and_inherited_metadata():
    def query_columns(*, metadata_mode):
        cols = [{"namespace": "data", "output_name": "data.text" if metadata_mode == "full" else "text", "base_name": "text"}]
        if metadata_mode == "full":
            cols.append({"namespace": "metadata", "output_name": "metadata.parent.text", "base_name": "text"})
        if metadata_mode == "local":
            cols.append({"namespace": "metadata", "output_name": "local_note", "base_name": "local_note"})
        return {"columns": cols, "data": [c["output_name"] for c in cols if c["namespace"] == "data"]}

    src = SimpleNamespace(artifact_type=ArtifactType.TABLE, primary_key=("row_id",), query_columns=query_columns)
    t = PatternExtractor(r"(?P<data_result>.+)", field="metadata.parent.text", metadata_mode="full")
    req = t.input_request(sources={"source": src}, mode="translate", request=TranslationRequest())
    assert req.columns.data is False
    assert req.columns.metadata == "metadata.parent.text"
    assert req.metadata_mode == "full"
    packet = InputBatch(source_label="source", artifact_id="src", primary_key=("row_id",),
        data=pd.DataFrame({"row_id": [0], "metadata.parent.text": ["Hello"]}),
        batch_index=0, batch_count=1, is_first=True, is_last=True)
    output = t.translate_batch({"source": packet}, mode="translate", request=TranslationRequest()).outputs["output"]
    assert output["data"]["result"].tolist() == ["Hello"]
    worker = t.make_translate_worker(mode="translate", request=TranslationRequest())
    assert worker.translate_batch({"source": packet}, mode="translate", request=TranslationRequest()).outputs["output"]["data"]["result"].tolist() == ["Hello"]
    for metadata_mode in ("none", "local"):
        missing = PatternExtractor(r"(?P<data_result>.+)", field="metadata.parent.text", metadata_mode=metadata_mode)
        with pytest.raises(OperatorError, match="available fields"):
            missing.input_request(sources={"source": src}, mode="translate", request=TranslationRequest())


def test_standalone_metadata_packet_field():
    t = PatternExtractor(r"(?P<data_value>.+)", field="source_note", metadata_mode="local")
    packet = {"data": pd.DataFrame({"text": ["ignored"]}), "metadata": pd.DataFrame({"source_note": ["chosen"]})}
    assert t.translate(packet)["data"]["value"].tolist() == ["chosen"]
    assert t.translate({"metadata": packet["metadata"]})["data"]["value"].tolist() == ["chosen"]



def test_persisted_metadata_source_with_colliding_data_field(tmp_path):
    import text_analysis_lab as teal
    project = teal.Project.create(tmp_path / "pattern_metadata", name="pattern_metadata")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"text": ["not chosen", "not chosen"]}),
            "metadata": pd.DataFrame({"text": ["Ada: Hello", "Bob: Bye"]}),
        }], artifact_type="table", primary_key="doc_id")
        choices = source.query_columns(metadata_mode="full")["columns"]
        label = next(item["output_name"] for item in choices
                     if item["namespace"] == "metadata" and item["base_name"] == "text")
        t = PatternExtractor(PATTERN, field=label, metadata_mode="full")
        output = project.translate(t, source)["output"]
        result = output.query(
            data_columns=True, metadata_columns=True,
            metadata_mode="local", form="table")
        assert result["utterance"].tolist() == ["Hello", "Bye"]
        assert result["speaker"].tolist() == ["Ada", "Bob"]
    finally:
        project.close()
