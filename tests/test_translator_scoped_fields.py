"""Regression tests for scoped field selection in text translators."""

import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.translators import RegexCleaner, TextLength, DelimiterDecomposer


def _source(project):
    return project.register_external([{
        "keys": pd.DataFrame({"doc_id": [1, 2]}),
        "data": pd.DataFrame({"text": ["A | B", "C | D"], "title": ["One", "Two"]}),
        "metadata": pd.DataFrame({"text": ["Alice|Bob", "Carol|Dave"]}),
    }], artifact_type="table", primary_key="doc_id")


def _label(artifact, namespace, base):
    return next(c["output_name"] for c in artifact.query_columns(metadata_mode="full")["columns"]
                if c["namespace"] == namespace and c["base_name"] == base)


def test_regex_cleaner_legacy_and_new_api():
    with pytest.warns(FutureWarning, match="deprecated"):
        old = RegexCleaner(text_field="text", output_field="clean")
    assert old.fields == {"text": "clean"}
    assert RegexCleaner.from_json_state({
        "text_field": "text", "output_field": "clean"
    }).fields == old.fields
    with pytest.raises(ValueError, match="cannot be combined"):
        RegexCleaner(fields={"text": "clean"}, text_field="text")
    with pytest.raises(ValueError, match="nonempty"):
        RegexCleaner(fields={})


def test_mixed_regex_fields_and_metadata_only(tmp_path):
    project = teal.Project.create(tmp_path / "regex", name="regex")
    try:
        source = _source(project)
        data = _label(source, "data", "text")
        meta = _label(source, "metadata", "text")
        cleaner = RegexCleaner(
            fields={data: "clean_data", meta: "clean_metadata"},
            metadata_mode="full",
            rules=[{"pattern": r"\s*\|\s*", "replacement": "/"}],
        )
        result = project.translate(cleaner, source)["output"]
        frame = result.query(metadata_mode="local", metadata_columns=True, form="table")
        assert frame["clean_data"].tolist() == ["A/B", "C/D"]
        assert frame["clean_metadata"].tolist() == ["Alice/Bob", "Carol/Dave"]
        metadata_only = RegexCleaner(
            fields={meta: "only_clean"}, metadata_mode="full",
            rules=[{"pattern": r"\|", "replacement": "/"}],
        )
        target = project.translate(metadata_only, source)["output"]
        values = target.query(data_columns=False, metadata_mode="local",
                              metadata_columns=True, form="table")
        assert values["only_clean"].tolist() == ["Alice/Bob", "Carol/Dave"]
        saved = cleaner.to_json_state()
        assert RegexCleaner.from_json_state(saved).fields == cleaner.fields
    finally:
        project.close()


def test_text_length_combined_mode_and_legacy(tmp_path):
    project = teal.Project.create(tmp_path / "length", name="length")
    try:
        source = _source(project)
        title = _label(source, "data", "title")
        text = _label(source, "metadata", "text")
        length = TextLength(
            fields=[title, text],
            units=["characters", "words"],
            output_field="combined", delimiter=" / ", metadata_mode="full",
        )
        assert length.translate({title: "Hi", text: "there"}) == {
            "combined_characters": 10, "combined_words": 3,
        }
        out = project.translate(length, source)["output"]
        rows = out.query(data_columns=False, metadata_mode="local",
                         metadata_columns=True, form="table")
        assert rows["combined_characters"].tolist() == [15, 16]
        assert TextLength.from_json_state(length.to_json_state()).to_json_state() == length.to_json_state()
        legacy = TextLength({"title": "characters"})
        assert legacy.translate({"title": "Hi"}) == {"title_characters": 2}
        assert TextLength.from_json_state(legacy.to_json_state()).lengths == legacy.lengths
    finally:
        project.close()


def test_delimiter_decomposer_metadata_source(tmp_path):
    project = teal.Project.create(tmp_path / "decompose", name="decompose")
    try:
        source = _source(project)
        meta = _label(source, "metadata", "text")
        t = DelimiterDecomposer(
            text_field=meta, metadata_mode="full", delimiter="|",
            output_text_field="part",
        )
        target = project.translate(t, source)["output"]
        rows = target.query(metadata_mode="none", form="table")
        assert rows["part"].tolist() == ["Alice", "Bob", "Carol", "Dave"]
        assert rows["segment_id"].tolist() == [0, 1, 0, 1]
        assert DelimiterDecomposer.from_json_state(t.to_json_state()).metadata_mode == "full"
        with pytest.raises(OperatorError, match="unavailable"):
            project.translate(DelimiterDecomposer(text_field=meta, delimiter="|"), source)
    finally:
        project.close()


def test_regex_standalone_mixed_preview_matches_persisted(tmp_path):
    project = teal.Project.create(tmp_path / "preview", name="preview")
    try:
        source = _source(project)
        data_label = _label(source, "data", "text")
        metadata_label = _label(source, "metadata", "text")
        cleaner = RegexCleaner(
            fields={data_label: "clean_data", metadata_label: "clean_metadata"},
            metadata_mode="full",
            rules=[{"pattern": r"\s*\|\s*", "replacement": "/"}],
        )
        values = source.query(
            metadata_mode="full", data_columns=[data_label],
            metadata_columns=[metadata_label], key_columns=False, form="table",
        )
        preview = cleaner.translate({
            "data": values[[data_label]],
            "metadata": values[[metadata_label]],
        })
        output = project.translate(cleaner, source)["output"].query(
            metadata_mode="local", metadata_columns=True, form="table",
        )
        assert preview["data"]["clean_data"].tolist() == output["clean_data"].tolist()
        assert preview["metadata"]["clean_metadata"].tolist() == output["clean_metadata"].tolist()
        # Parity alone could pass if both paths silently failed to replace the pipe.
        assert preview["data"]["clean_data"].tolist() == ["A/B", "C/D"]
        assert preview["metadata"]["clean_metadata"].tolist() == ["Alice/Bob", "Carol/Dave"]
        assert output["clean_data"].tolist() == ["A/B", "C/D"]
        assert output["clean_metadata"].tolist() == ["Alice/Bob", "Carol/Dave"]
        only = RegexCleaner(fields={metadata_label: "clean_metadata"}, metadata_mode="full")
        assert only.translate({"metadata": values[[metadata_label]]})["metadata"]["clean_metadata"].tolist() == ["Alice|Bob", "Carol|Dave"]
    finally:
        project.close()


def test_scoped_translators_preserve_key_when_data_has_same_name(tmp_path):
    project_path = tmp_path / "key_data_names"
    project = teal.Project.create(project_path, name="key_data_names")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"text": [10, 11]}),
            "data": pd.DataFrame({"text": ["Ada|Hello", "Bob|Bye"]}),
            "metadata": pd.DataFrame({"note": ["first", "second"]}),
        }], artifact_type="table", primary_key="text")
        assert source.query_columns(metadata_mode="full")["data"] == ["data.text"]

        cleaner = RegexCleaner(
            fields={"data.text": "cleaned"},
            rules=[{"pattern": r"\|", "replacement": " "}],
        )
        cleaned = project.translate(cleaner, source)["output"].query(form="table")
        assert cleaned["text"].tolist() == [10, 11]
        assert cleaned["cleaned"].tolist() == ["Ada Hello", "Bob Bye"]

        measured = project.translate(
            TextLength(
                fields=["data.text", "note"], units="words",
                output_field="combined", metadata_mode="full", delimiter=" ",
            ), source,
        )["output"].query(
            data_columns=False, metadata_columns=True,
            metadata_mode="local", form="table",
        )
        assert measured["text"].tolist() == [10, 11]
        assert measured["combined_words"].tolist() == [2, 2]

        split = project.translate(
            DelimiterDecomposer(
                text_field="data.text", delimiter="|",
                output_text_field="part",
            ), source,
        )["output"].query(form="table")
        assert split["text"].tolist() == [10, 10, 11, 11]
        assert split["segment_id"].tolist() == [0, 1, 0, 1]
        assert split["part"].tolist() == ["Ada", "Hello", "Bob", "Bye"]
    finally:
        project.close()


def test_regex_cleaner_output_can_share_structural_key_name_and_reopen(tmp_path):
    project_path = tmp_path / "regex_key_output_collision"
    project = teal.Project.create(project_path, name="regex_key_output_collision")
    output_id = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"text": [1, 2]}),
            "data": pd.DataFrame({"text": ["A B", "C D"]}),
        }], artifact_type="table", primary_key="text")
        result = project.translate(
            RegexCleaner(
                fields={"data.text": "text"},
                rules=[{"pattern": " ", "replacement": "_"}],
            ), source,
        )["output"]
        output_id = result.artifact_id
        frame = result.query(form="table")
        assert frame["text"].tolist() == [1, 2]
        assert frame["data.text"].tolist() == ["A_B", "C_D"]
    finally:
        project.close()
    reopened = teal.Project.open(project_path)
    try:
        frame = reopened.get_artifact(output_id).query(form="table")
        assert frame["text"].tolist() == [1, 2]
        assert frame["data.text"].tolist() == ["A_B", "C_D"]
    finally:
        reopened.close()


def test_pattern_extractor_data_output_can_share_key_name_and_reopen(tmp_path):
    from text_analysis_lab.translators import PatternExtractor

    project_path = tmp_path / "pattern_key_output_collision"
    project = teal.Project.create(project_path, name="pattern_key_output_collision")
    output_id = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"text": [3, 4]}),
            "data": pd.DataFrame({"utterance": ["Ada", "Bob"]}),
        }], artifact_type="table", primary_key="text")
        result = project.translate(
            PatternExtractor(r"(?P<data_text>\w+)", field="utterance"), source
        )["output"]
        output_id = result.artifact_id
        frame = result.query(form="table")
        assert frame["text"].tolist() == [3, 4]
        assert frame["data.text"].tolist() == ["Ada", "Bob"]
    finally:
        project.close()
    reopened = teal.Project.open(project_path)
    try:
        frame = reopened.get_artifact(output_id).query(form="table")
        assert frame["text"].tolist() == [3, 4]
        assert frame["data.text"].tolist() == ["Ada", "Bob"]
    finally:
        reopened.close()


def test_text_length_metadata_name_can_match_key_and_data_and_reopen(tmp_path):
    project_path = tmp_path / "length_output_namespace_collision"
    project = teal.Project.create(project_path, name="length_output_namespace_collision")
    output_id = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"text_characters": [1, 2]}),
            "data": pd.DataFrame({
                "text": ["abc", "hello"],
                "text_characters": ["original-A", "original-B"],
            }),
        }], artifact_type="table", primary_key="text_characters")
        result = project.translate(
            TextLength({"text": "characters"}), source
        )["output"]
        output_id = result.artifact_id
        for artifact in (result,):
            names = artifact.query_columns(metadata_mode="local")["columns"]
            key_label = next(x["output_name"] for x in names if x["namespace"] == "key")
            metadata_label = next(x["output_name"] for x in names if x["namespace"] == "metadata")
            data_label = next(x["output_name"] for x in names if x["namespace"] == "data"
                              and x["base_name"] == "text_characters")
            frame = artifact.query(metadata_mode="local", metadata_columns=True, form="table")
            assert key_label == "text_characters"
            assert metadata_label not in {key_label, data_label}
            assert frame[key_label].tolist() == [1, 2]
            assert frame[metadata_label].tolist() == [3, 5]
            assert frame[data_label].tolist() == ["original-A", "original-B"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        artifact = reopened.get_artifact(output_id)
        names = artifact.query_columns(metadata_mode="local")["columns"]
        meta = next(x["output_name"] for x in names if x["namespace"] == "metadata")
        data = next(x["output_name"] for x in names if x["namespace"] == "data"
                    and x["base_name"] == "text_characters")
        frame = artifact.query(metadata_mode="local", metadata_columns=True, form="table")
        assert frame["text_characters"].tolist() == [1, 2]
        assert frame[meta].tolist() == [3, 5]
        assert frame[data].tolist() == ["original-A", "original-B"]
    finally:
        reopened.close()


def test_delimiter_extended_key_can_match_input_data_field_and_reopen(tmp_path):
    project_path = tmp_path / "delimiter_key_data_collision"
    project = teal.Project.create(project_path, name="delimiter_key_data_collision")
    output_id = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [8, 9]}),
            "data": pd.DataFrame({"segment_id": ["one|two", "red|blue"]}),
        }], artifact_type="table", primary_key="doc_id")
        target = project.translate(
            DelimiterDecomposer(
                text_field="segment_id", new_key="segment_id",
                output_text_field="part", delimiter="|",
            ), source
        )["output"]
        output_id = target.artifact_id
        frame = target.query(form="table")
        assert frame["doc_id"].tolist() == [8, 8, 9, 9]
        assert frame["segment_id"].tolist() == [0, 1, 0, 1]
        assert frame["part"].tolist() == ["one", "two", "red", "blue"]
    finally:
        project.close()
    reopened = teal.Project.open(project_path)
    try:
        frame = reopened.get_artifact(output_id).query(form="table")
        assert frame["doc_id"].tolist() == [8, 8, 9, 9]
        assert frame["segment_id"].tolist() == [0, 1, 0, 1]
        assert frame["part"].tolist() == ["one", "two", "red", "blue"]
    finally:
        reopened.close()


def test_pattern_data_only_works_when_full_metadata_scope_cannot_resolve(tmp_path):
    from text_analysis_lab.core.errors import QueryError
    from text_analysis_lab.translators import PatternExtractor

    project_path = tmp_path / "pattern_narrow_scope"
    project = teal.Project.create(project_path, name="pattern_narrow_scope")
    output_id = None
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"data.text": [21, 22]}),
            "data": pd.DataFrame({"text": ["Ada", "Bob"]}),
            "metadata": pd.DataFrame({"text": ["noise-a", "noise-b"]}),
        }], artifact_type="table", primary_key="data.text")
        assert source.query_columns(metadata_mode="none")["data"] == ["text"]
        with pytest.raises(QueryError, match="Could not disambiguate"):
            source.query_columns(metadata_mode="full")
        output = project.translate(
            PatternExtractor(r"(?P<data_name>\w+)", field="text", metadata_mode="none"),
            source,
        )["output"]
        output_id = output.artifact_id
        frame = output.query(form="table")
        assert frame["data.text"].tolist() == [21, 22]
        assert frame["name"].tolist() == ["Ada", "Bob"]
    finally:
        project.close()
    reopened = teal.Project.open(project_path)
    try:
        frame = reopened.get_artifact(output_id).query(form="table")
        assert frame["data.text"].tolist() == [21, 22]
        assert frame["name"].tolist() == ["Ada", "Bob"]
    finally:
        reopened.close()
