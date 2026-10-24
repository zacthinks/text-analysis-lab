"""Full-metadata key collisions must not break scoped text translators."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    DelimiterDecomposer,
    PatternExtractor,
    RegexCleaner,
    TextLength,
)


@pytest.mark.parametrize("translator_name", [
    "pattern", "regex", "length", "delimiter",
])
def test_text_translators_restore_structural_keys_from_canonical_scope(
    tmp_path, translator_name
):
    project = teal.Project.create(
        tmp_path / translator_name, name="text_key_" + translator_name
    )
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"text": ["Ada|Hello", "Bob|Bye"]}),
            "metadata": pd.DataFrame({
                "doc_id": [99, 98],
                "speaker": ["Alice", "Robert"],
            }),
        }], artifact_type="table", primary_key="doc_id")
        columns = source.query_columns(metadata_mode="full")["columns"]
        key_label = next(
            entry["output_name"] for entry in columns
            if entry["namespace"] == "key" and entry["base_name"] == "doc_id"
        )
        assert key_label != "doc_id"
        if translator_name == "pattern":
            translator = PatternExtractor(
                r"(?P<data_word>\w+)\|(?P<metadata_suffix>\w+)",
                field="text", metadata_mode="full",
            )
        elif translator_name == "regex":
            translator = RegexCleaner(
                fields={"text": "clean_text"}, metadata_mode="full",
                rules=[{"pattern": r"\|", "replacement": " "}],
            )
        elif translator_name == "length":
            translator = TextLength(
                {"text": "characters"}, metadata_mode="full",
            )
        else:
            translator = DelimiterDecomposer(
                text_field="text", delimiter="|",
                output_text_field="part", metadata_mode="full",
            )
        output = project.translate(translator, source)["output"]
        frame = output.query(
            key_columns=True, data_columns=True,
            metadata_columns=True, metadata_mode="local", form="table",
        )
        if translator_name == "pattern":
            assert frame["doc_id"].tolist() == [1, 2]
            assert frame["word"].tolist() == ["Ada", "Bob"]
            assert frame["suffix"].tolist() == ["Hello", "Bye"]
        elif translator_name == "regex":
            assert frame["doc_id"].tolist() == [1, 2]
            assert frame["clean_text"].tolist() == ["Ada Hello", "Bob Bye"]
        elif translator_name == "length":
            assert frame["doc_id"].tolist() == [1, 2]
            assert frame["text_characters"].tolist() == [9, 7]
        else:
            assert frame["doc_id"].tolist() == [1, 1, 2, 2]
            assert frame["segment_id"].tolist() == [0, 1, 0, 1]
            assert frame["part"].tolist() == ["Ada", "Hello", "Bob", "Bye"]
    finally:
        project.close()
