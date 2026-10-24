"""Focused contract tests for operation-local QueryScope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.operator import ColumnRequest
from text_analysis_lab.core.query_scope import QueryScope


@pytest.fixture
def source(tmp_path):
    project = teal.Project.create(tmp_path / "scope_project", name="scope_project")
    try:
        artifact = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"score": [11, 22], "text": ["a", "b"]}),
            "metadata": pd.DataFrame({"score": [101, 202], "group": ["x", "y"]}),
        }], artifact_type="table", primary_key="doc_id")
        yield artifact
    finally:
        project.close()


def test_scope_discovery_and_multiple_projections_share_labels(source):
    scope = QueryScope(source, include_data=True, metadata_mode="full")
    inventory = scope.columns()
    data_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "data" and c["base_name"] == "score"
    )
    metadata_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "metadata" and c["base_name"] == "score"
    )
    assert data_score != metadata_score
    data = scope.query(
        key_columns=False, data_columns=[data_score], metadata_columns=False,
        form="table",
    )
    metadata = scope.query(
        key_columns=False, data_columns=False,
        metadata_columns=[metadata_score], form="table",
    )
    assert data[data_score].tolist() == [11, 22]
    assert metadata[metadata_score].tolist() == [101, 202]


def test_scope_generates_source_request_with_same_naming_scope(source):
    scope = QueryScope(source, include_data=True, metadata_mode="full")
    inventory = scope.columns()
    data_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "data" and c["base_name"] == "score"
    )
    request = scope.source_request(
        columns=ColumnRequest(keys=True, data=[data_score], metadata=False),
        mode="batches", batch_size=2, form="table",
    )
    assert request.metadata_mode == "full"
    assert request.include_data is True
    assert request.columns.data == [data_score]


def test_scope_rejects_scope_overrides(source):
    scope = QueryScope(source, metadata_mode="full")
    with pytest.raises(TypeError, match="naming-scope"):
        scope.query(metadata_mode="none", data_columns=False)
    with pytest.raises(TypeError, match="naming-scope"):
        scope.source_request(
            columns=ColumnRequest(data=False),
            mode="full_artifact",
            metadata_mode="none",
        )


def test_scope_can_exclude_data_without_changing_key_names(source):
    scope = QueryScope(source, include_data=False, metadata_mode="none")
    assert scope.columns()["key"] == ["doc_id"]
    frame = scope.query(
        key_columns=True, data_columns=False, metadata_columns=False,
        form="table",
    )
    assert frame["doc_id"].tolist() == [1, 2]


def test_classifier_search_uses_full_scope_for_data_only_labels(source):
    from text_analysis_lab.analysis.classifier_search import _load_development_labels

    inventory = QueryScope(source, metadata_mode="full").columns()
    data_score = next(
        c["output_name"] for c in inventory["columns"]
        if c["namespace"] == "data" and c["base_name"] == "score"
    )
    keys, labels, groups = _load_development_labels(
        source, target=data_score, group=None, batch_size=2,
    )
    assert keys == [{"doc_id": 1}, {"doc_id": 2}]
    assert labels.tolist() == [11, 22]
    assert groups is None


def test_scope_with_two_attached_artifacts_selects_exact_outputs(tmp_path):
    project = teal.Project.create(tmp_path / "attached_scope", name="attached_scope")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3], "score": [100, 200, 300]}),
            primary_key="doc_id", data_fields="score",
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "score": [10, 20]}),
            primary_key="doc_id", data_fields="score",
        )
        other = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3], "score": [1, 2, 3]}),
            primary_key="doc_id", data_fields="score",
        )
        scope = QueryScope(
            left, include_data=True, metadata_mode="none",
            with_artifacts=[right, other],
        )
        fields = scope.columns()["columns"]
        caller_name = next(
            c["output_name"] for c in fields
            if c["namespace"] == "data" and c["base_name"] == "score"
        )
        right_name = next(
            c["output_name"] for c in fields
            if c["namespace"] == "with" and c["base_name"] == "score"
            and c["source_artifact_id"] == right.artifact_id
        )
        other_name = next(
            c["output_name"] for c in fields
            if c["namespace"] == "with" and c["base_name"] == "score"
            and c["source_artifact_id"] == other.artifact_id
        )
        assert len({caller_name, right_name, other_name}) == 3
        first = scope.query(
            key_columns=True, data_columns=[caller_name],
            metadata_columns=False, with_columns=[right_name], form="table",
        )
        assert first["doc_id"].tolist() == [1, 2, 3]
        assert first[caller_name].tolist() == [100, 200, 300]
        assert first[right_name].iloc[:2].tolist() == [10, 20]
        assert pd.isna(first[right_name].iloc[2])
        second = list(scope.query(
            key_columns=False, data_columns=False, metadata_columns=False,
            with_columns=[other_name], form="table", iter_batches=True,
            batch_size=1,
        ))
        assert [v for batch in second for v in batch[other_name].tolist()] == [1, 2, 3]
        with pytest.raises(TypeError, match="naming-scope"):
            scope.query(with_artifacts=right, data_columns=False)
        with pytest.raises(ValueError, match="with_artifacts"):
            scope.source_request(
                columns=ColumnRequest(data=False), mode="full_artifact",
            )
    finally:
        project.close()


def test_scope_with_attached_sparse_matrix_materializes_values(tmp_path):
    from scipy import sparse

    project = teal.Project.create(tmp_path / "attached_sparse_scope", name="attached_sparse_scope")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3]}), primary_key="doc_id",
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {
                "values": sparse.csr_matrix([[0, 9], [5, 8]]),
                "columns": ["feature", "other"],
            },
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        scope = QueryScope(left, with_artifacts=right)
        features = [
            c for c in scope.columns()["columns"]
            if c["namespace"] == "with" and c["base_name"] == "feature"
        ]
        assert len(features) == 1
        label = features[0]["output_name"]
        result = scope.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            with_columns=[label], form="table",
        )
        assert result["doc_id"].tolist() == [1, 2, 3]
        assert result[label].iloc[:2].tolist() == [0, 5]
        assert pd.isna(result[label].iloc[2])
    finally:
        project.close()


def test_function_mapper_prefers_literal_dotted_data_output(tmp_path):
    from text_analysis_lab.translators import FunctionMapper

    project = teal.Project.create(tmp_path / "mapper_dotted", name="mapper_dotted")
    try:
        source = project.register_external(
            pd.DataFrame({
                "doc_id": [1, 2],
                "text": ["wrong_a", "wrong_b"],
                "data.text": ["correct_a", "correct_b"],
            }),
            primary_key="doc_id", data_fields=["text", "data.text"],
        )
        inventory = QueryScope(source).columns()
        names = {
            c["base_name"]: c["output_name"]
            for c in inventory["columns"] if c["namespace"] == "data"
        }
        assert names == {"text": "text", "data.text": "data.text"}
        mapper = FunctionMapper(
            lambda packet: {
                "data": pd.DataFrame({
                    "picked": packet["data"]["data.text"].tolist(),
                }),
            },
        )
        result = project.translate(
            mapper, source, data_columns=["data.text"],
        )["output"]
        assert result.query(data_columns=["picked"], form="table")["picked"].tolist() == [
            "correct_a", "correct_b",
        ]
    finally:
        project.close()


def test_text_length_uses_data_only_scope_when_metadata_collides(tmp_path):
    from text_analysis_lab.translators import TextLength

    project = teal.Project.create(tmp_path / "length_scope", name="length_scope")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"text": ["a b", "three words here"]}),
            "metadata": pd.DataFrame({"text": ["metadata_a", "metadata_b"]}),
        }], artifact_type="table", primary_key="doc_id")
        result = project.translate(
            TextLength({"text": "words"}), source, batch_size=1,
        )["output"]
        info = result.query(
            key_columns=True, data_columns=False, metadata_columns=True,
            metadata_mode="local", form="table",
        )
        assert info["text_words"].tolist() == [2, 3]
    finally:
        project.close()


def test_nearest_neighbor_context_with_colliding_data_metadata(tmp_path):
    import numpy as np

    project = teal.Project.create(tmp_path / "neighbors_scope", name="neighbors_scope")
    try:
        matrix = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": {"values": np.asarray([[1., 0.], [0., 1.]]),
                     "columns": ["x", "y"]},
        }], artifact_type="dense_matrix", primary_key="doc_id")
        context = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"score": [11, 22]}),
            "metadata": pd.DataFrame({"score": [101, 202]}),
        }], artifact_type="table", primary_key="doc_id")
        result = matrix.analysis.nearest_neighbors(
            position=0, k=1, include_self=True,
            context=context, context_data_columns=["data.score"],
            context_metadata_columns=["metadata." + context.artifact_id + ".score"],
            context_metadata_mode="full",
        )
        assert result.iloc[0]["data.score"] == 11
        assert result.iloc[0]["metadata." + context.artifact_id + ".score"] == 101
    finally:
        project.close()


@pytest.mark.parametrize("matrix_type", ["dense_matrix", "sparse_matrix"])
def test_matrix_transpose_promotes_key_and_colliding_metadata(matrix_type, tmp_path):
    from scipy import sparse
    from text_analysis_lab.translators import MatrixTranspose

    project = teal.Project.create(tmp_path / matrix_type, name="transpose_collisions")
    try:
        values = np.asarray([[1., 2.], [3., 4.]])
        if matrix_type == "sparse_matrix":
            values = sparse.csr_matrix(values)
        source = project.register_external([{
            "keys": pd.DataFrame({"row_id": [10, 20]}),
            "data": {"values": values, "columns": ["f0", "f1"]},
            "metadata": pd.DataFrame({"row_id": ["alpha", "beta"]}),
        }], artifact_type=matrix_type, primary_key="row_id")
        scope = source.query_columns(metadata_mode="full")
        key_name = scope["key"][0]
        meta_name = next(
            c["output_name"] for c in scope["columns"]
            if c["namespace"] == "metadata" and c["base_name"] == "row_id"
        )
        assert key_name == "row_id"
        assert meta_name != key_name
        transposed = project.translate(
            MatrixTranspose(), source,
            feature_metadata_columns=[key_name, meta_name],
        )["output"]
        fm = transposed.get_feature_metadata()
        assert fm[key_name].tolist() == [10, 20]
        assert fm[meta_name].tolist() == ["alpha", "beta"]
        dense = transposed.get_matrix()
        if sparse.issparse(dense):
            dense = dense.toarray()
        np.testing.assert_array_equal(dense, [[1, 3], [2, 4]])
    finally:
        project.close()


@pytest.mark.parametrize("kind", ["RegexCleaner", "DelimiterDecomposer", "TextLength"])
def test_existing_text_translators_bind_data_name_when_key_is_text(tmp_path, kind):
    from text_analysis_lab.translators import RegexCleaner, DelimiterDecomposer, TextLength
    project = teal.Project.create(tmp_path / kind, name="text_key_collision")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"text": [10, 20]}),
            "data": pd.DataFrame({"text": ["one two", "three four"]}),
        }], artifact_type="table", primary_key="text")
        if kind == "RegexCleaner":
            operator = RegexCleaner(text_field="text", output_field="cleaned",
                                    rules=[{"pattern": " ", "replacement": "_"}])
        elif kind == "DelimiterDecomposer":
            operator = DelimiterDecomposer(text_field="text", delimiter=" ",
                                           new_key="part_id", output_text_field="word")
        else:
            operator = TextLength({"text": "words"})
        result = project.translate(operator, source)["output"]
        assert result.primary_key[0] == "text"
        if kind == "RegexCleaner":
            frame = result.query(data_columns=["cleaned"], form="table")
            assert frame["cleaned"].tolist() == ["one_two", "three_four"]
            assert frame["text"].tolist() == [10, 20]
        elif kind == "DelimiterDecomposer":
            frame = result.query(data_columns=["word"], form="table")
            assert frame["word"].tolist() == ["one", "two", "three", "four"]
            assert frame["text"].tolist() == [10, 10, 20, 20]
        else:
            frame = result.query(data_columns=False, metadata_columns=True,
                                 metadata_mode="local", form="table")
            assert frame["text_words"].tolist() == [2, 2]
            assert frame["text"].tolist() == [10, 20]
    finally:
        project.close()


def test_openai_template_raw_placeholders_bind_scoped_data_and_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from text_analysis_lab.translators import OpenAIResponsesTranslator

    project = teal.Project.create(tmp_path / "prompt_scoped", name="prompt_scoped")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"row_id": [1]}),
            "data": pd.DataFrame({"text": ["DATA"]}),
            "metadata": pd.DataFrame({"text": ["META"], "party": ["P"]}),
        }], artifact_type="table", primary_key="row_id")
        calls = []
        operator = OpenAIResponsesTranslator("test-model", "Text: {text}; Party: {party}")
        def fake_request(self, prompt):
            calls.append(prompt)
            return SimpleNamespace(
                id="fake", model="test-model", output_text="ok", usage=None
            )
        monkeypatch.setattr(OpenAIResponsesTranslator, "_request", fake_request)
        result = project.translate(operator, source)["output"]
        assert calls == ["Text: DATA; Party: P"]
        from text_analysis_lab.translators.openai_responses import OpenAIResponsesTranslator as OpenAITranslator
        saved = tmp_path / "saved_prompt"
        operator.save_intermediate_state(
            saved, operator_id="optr_prompt",
            mode="translate", route="sequential",
        )
        restored = OpenAITranslator.load_intermediate_state(
            saved, operator_id="optr_prompt",
            mode="translate", route="sequential",
        )
        assert restored._prompt_input_names == operator._prompt_input_names

        assert result.query(data_columns=["response"], form="table")["response"].tolist() == ["ok"]
    finally:
        project.close()
