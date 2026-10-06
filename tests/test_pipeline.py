from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.errors import PipelineError
from text_analysis_lab.core.operator import ExecutionCapabilities
from text_analysis_lab.translators import (
    CountVectorizer,
    EmbeddingLookup,
    FeatureTrimmer,
    FunctionMapper,
    MatrixTranspose,
    RegexCleaner,
    SVD,
    TfidfTransformer,
    UMAP,
    Word2Vec,
)


def _source(project: teal.Project, tmp_path: Path):
    path = tmp_path / "documents.csv"
    pd.DataFrame(
        {
            "text": ["  alpha   beta  ", " gamma   alpha "],
        }
    ).to_csv(path, index=False)
    return project.read_csv(
        path,
        text_fields="text",
        metadata_fields=None,
        batch_size=1,
    )






class _KeywordOnlyTranslator(teal.BaseTranslator):
    def translate(self, *, source):
        return pd.Series(source).str.upper()

    def output_specs(self, *, sources, request):
        _ = sources, request
        raise NotImplementedError

    def input_request(self, *, sources, mode, request):
        _ = sources, mode, request
        raise NotImplementedError

    def translate_batch(self, inputs, *, mode, request):
        _ = inputs, mode, request
        raise NotImplementedError

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = result, batch_index, mode, request
        raise NotImplementedError

    def finalize_translation(self, *, mode, request):
        _ = mode, request
        raise NotImplementedError


class _DistinctTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        table: pd.DataFrame,
        payload: dict[str, object],
    ) -> tuple[int, str]:
        return len(table), str(payload["name"])


class _AmbiguousTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
    ) -> tuple[int, int]:
        return len(left), len(right)


class _MixedKeywordTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        table: pd.DataFrame,
        payload: dict[str, object],
    ) -> tuple[int, str]:
        return len(table), str(payload["name"])


class _IncompleteTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        table: pd.DataFrame,
        payload: dict[str, object],
        required: str,
    ) -> tuple[int, str, str]:
        return len(table), str(payload["name"]), required


class _UnionTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        table: pd.DataFrame,
        payload: dict[str, object] | str,
    ) -> tuple[int, str]:
        value = payload["name"] if isinstance(payload, dict) else payload
        return len(table), str(value)


class _PositionalOnlyTypedInputs(_KeywordOnlyTranslator):
    def translate(
        self,
        prefix: str = "default",
        table: pd.DataFrame | None = None,
        /,
        *,
        payload: dict[str, object],
    ) -> tuple[str, int, str]:
        assert table is not None
        return prefix, len(table), str(payload["name"])


class _StandaloneMultiOutput(teal.BaseTranslator):
    def translate(self, values):
        values = pd.Series(values)
        return {
            "left": values.str.lower(),
            "right": values.str.upper(),
        }

    def execution_capabilities(self, *, project=None):
        _ = project
        return ExecutionCapabilities(
            reusable=True,
            artifact=False,
            native=True,
            portable=True,
        )

    def output_specs(self, *, sources, request):
        _ = sources, request
        raise NotImplementedError

    def input_request(self, *, sources, mode, request):
        _ = sources, mode, request
        raise NotImplementedError

    def translate_batch(self, inputs, *, mode, request):
        _ = inputs, mode, request
        raise NotImplementedError

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = result, batch_index, mode, request
        raise NotImplementedError

    def finalize_translation(self, *, mode, request):
        _ = mode, request
        raise NotImplementedError




def test_pipeline_native_single_source_uses_matching_keyword_only_parameter() -> None:
    pipeline = teal.Pipeline()
    stage = pipeline.add(
        "upper",
        _KeywordOnlyTranslator(),
        source=pipeline.input,
    )
    pipeline.output("output", stage["output"])

    result = pipeline.translate(pd.Series(["alpha", "beta"]))

    assert result["output"].tolist() == ["ALPHA", "BETA"]


def test_pipeline_rejects_ports_from_different_pipeline_instances() -> None:
    pipeline_a = teal.Pipeline()
    pipeline_b = teal.Pipeline()

    with pytest.raises(PipelineError, match="different Pipeline"):
        pipeline_b.add(
            "clean",
            RegexCleaner(),
            source=pipeline_a.input,
        )

    stage_a = pipeline_a.add(
        "shared_name",
        RegexCleaner(),
        source=pipeline_a.input,
    )
    pipeline_b.add(
        "shared_name",
        RegexCleaner(),
        source=pipeline_b.input,
    )

    with pytest.raises(PipelineError, match="different Pipeline"):
        pipeline_b.output("wrong", stage_a["output"])


def test_pipeline_native_multi_input_preserves_structured_default_output() -> None:
    pipeline = teal.Pipeline(inputs=["tokens", "embeddings"])
    lookup = pipeline.add(
        "lookup",
        EmbeddingLookup(field="word"),
        sources={
            "tokens": pipeline.input["tokens"],
            "embeddings": pipeline.input["embeddings"],
        },
    )
    pipeline.output("vectors", lookup["output"])

    embeddings = {
        "values": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "metadata": pd.DataFrame({"word": ["alpha", "beta"]}),
        "row_name_column": "word",
        "feature_metadata": pd.DataFrame({"column": ["x", "y"]}),
    }
    result = pipeline.translate(
        inputs={
            "tokens": pd.DataFrame({"word": ["beta", "alpha"]}),
            "embeddings": embeddings,
        }
    )

    vectors = result["vectors"]
    np.testing.assert_allclose(vectors["values"], [[3.0, 4.0], [1.0, 2.0]])
    assert vectors["feature_metadata"]["column"].tolist() == ["x", "y"]


def test_pipeline_multi_source_uses_unique_runtime_type_binding() -> None:
    pipeline = teal.Pipeline(inputs=["payload_input", "table_input"])
    combined = pipeline.add(
        "combined",
        _DistinctTypedInputs(),
        sources={
            "recorded_payload": pipeline.input["payload_input"],
            "recorded_table": pipeline.input["table_input"],
        },
    )
    pipeline.output("output", combined["output"])

    result = pipeline.translate(
        inputs={
            "payload_input": {"name": "matched"},
            "table_input": pd.DataFrame({"x": [1, 2, 3]}),
        }
    )

    assert result["output"] == (3, "matched")


def test_pipeline_multi_source_rejects_ambiguous_runtime_type_binding() -> None:
    pipeline = teal.Pipeline(inputs=["first", "second"])
    combined = pipeline.add(
        "combined",
        _AmbiguousTypedInputs(),
        sources={
            "recorded_first": pipeline.input["first"],
            "recorded_second": pipeline.input["second"],
        },
    )
    pipeline.output("output", combined["output"])

    with pytest.raises(PipelineError, match="multiple type-compatible bindings"):
        pipeline.translate(
            inputs={
                "first": pd.DataFrame({"x": [1]}),
                "second": pd.DataFrame({"x": [2]}),
            }
        )


def test_pipeline_multi_source_combines_keyword_and_type_binding() -> None:
    pipeline = teal.Pipeline(inputs=["table_input", "payload_input"])
    combined = pipeline.add(
        "combined",
        _MixedKeywordTypedInputs(),
        sources={
            "table": pipeline.input["table_input"],
            "recorded_payload": pipeline.input["payload_input"],
        },
    )
    pipeline.output("output", combined["output"])

    result = pipeline.translate(
        inputs={
            "table_input": pd.DataFrame({"x": [1, 2]}),
            "payload_input": {"name": "mixed"},
        }
    )

    assert result["output"] == (2, "mixed")


def test_pipeline_multi_source_requires_all_required_parameters() -> None:
    pipeline = teal.Pipeline(inputs=["table_input", "payload_input"])
    combined = pipeline.add(
        "combined",
        _IncompleteTypedInputs(),
        sources={
            "recorded_table": pipeline.input["table_input"],
            "recorded_payload": pipeline.input["payload_input"],
        },
    )
    pipeline.output("output", combined["output"])

    with pytest.raises(PipelineError, match="no complete type match"):
        pipeline.translate(
            inputs={
                "table_input": pd.DataFrame({"x": [1]}),
                "payload_input": {"name": "missing-required"},
            }
        )


def test_pipeline_multi_source_type_binding_supports_union_annotations() -> None:
    pipeline = teal.Pipeline(inputs=["payload_input", "table_input"])
    combined = pipeline.add(
        "combined",
        _UnionTypedInputs(),
        sources={
            "recorded_payload": pipeline.input["payload_input"],
            "recorded_table": pipeline.input["table_input"],
        },
    )
    pipeline.output("output", combined["output"])

    result = pipeline.translate(
        inputs={
            "payload_input": {"name": "union"},
            "table_input": pd.DataFrame({"x": [1, 2, 3]}),
        }
    )

    assert result["output"] == (3, "union")


def test_pipeline_multi_source_type_binding_handles_positional_only_defaults() -> None:
    pipeline = teal.Pipeline(inputs=["table_input", "payload_input"])
    combined = pipeline.add(
        "combined",
        _PositionalOnlyTypedInputs(),
        sources={
            "recorded_table": pipeline.input["table_input"],
            "payload": pipeline.input["payload_input"],
        },
    )
    pipeline.output("output", combined["output"])

    result = pipeline.translate(
        inputs={
            "table_input": pd.DataFrame({"x": [1, 2]}),
            "payload_input": {"name": "positional"},
        }
    )

    assert result["output"] == ("default", 2, "positional")


def test_pipeline_native_multi_output_routes_only_referenced_labels() -> None:
    pipeline = teal.Pipeline()
    split = pipeline.add(
        "split",
        _StandaloneMultiOutput(),
        source=pipeline.input,
    )
    cleaned = pipeline.add(
        "cleaned",
        RegexCleaner(rules=({"pattern": "a", "replacement": "@"},)),
        source=split["left"],
    )
    pipeline.output("cleaned", cleaned["output"])
    pipeline.output("upper", split["right"])

    result = pipeline.translate(
        pd.Series(["Alpha", "Beta"]),
        outputs=["cleaned", "upper"],
    )

    assert result["cleaned"].tolist() == ["@lph@", "bet@"]
    assert result["upper"].tolist() == ["ALPHA", "BETA"]


def test_pipeline_native_linear_execution_uses_translator_contracts() -> None:
    pipeline = teal.Pipeline()
    compact = pipeline.add(
        "compact",
        RegexCleaner(rules=({"pattern": r"\s+", "replacement": " "},)),
        source=pipeline.input,
    )
    renamed = pipeline.add(
        "renamed",
        RegexCleaner(rules=({"pattern": "alpha", "replacement": "A"},)),
        source=compact["output"],
    )
    pipeline.output("clean", renamed["output"])

    result = pipeline.translate(
        pd.Series(["  alpha   beta  ", " gamma   alpha "]),
        outputs=["clean"],
    )

    assert result["clean"].tolist() == ["A beta", "gamma A"]
    caps = pipeline.execution_capabilities(outputs=["clean"])
    assert caps.reusable
    assert caps.native
    assert caps.artifact
    assert caps.portable


def test_requested_output_closure_ignores_unrelated_nonreusable_branch() -> None:
    pipeline = teal.Pipeline()
    clean = pipeline.add(
        "clean",
        RegexCleaner(),
        source=pipeline.input,
    )
    training = pipeline.add(
        "training",
        Word2Vec(min_count=1),
        source=pipeline.input,
    )
    pipeline.output("clean", clean["output"])
    pipeline.output("training", training["output"])

    clean_caps = pipeline.execution_capabilities(outputs=["clean"])
    all_caps = pipeline.execution_capabilities()
    assert clean_caps.native
    assert clean_caps.reusable
    assert not all_caps.native
    assert not all_caps.reusable

    result = pipeline.translate(pd.Series([" alpha "]), outputs=["clean"])
    assert result["clean"].tolist() == ["alpha"]


def test_pipeline_blocks_unfitted_translator_before_project_operation(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project",
        name="pipeline_unfitted",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        pipeline = teal.Pipeline()
        count = pipeline.add(
            "count",
            CountVectorizer(text_field="text"),
            source=pipeline.input,
        )
        pipeline.output("counts", count["output"])
        before = len(project.list_operations())

        with pytest.raises(PipelineError, match="requires fitted state|cannot execute"):
            project.run_pipeline(pipeline, source)

        assert len(project.list_operations()) == before
    finally:
        project.close()


def test_project_pipeline_executes_each_stage_as_normal_translation(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project",
        name="pipeline_artifacts",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        pipeline = teal.Pipeline()
        compact = pipeline.add(
            "compact",
            RegexCleaner(
                text_field="text",
                output_field="compact_text",
                rules=({"pattern": r"\s+", "replacement": " "},),
            ),
            source=pipeline.input,
        )
        renamed = pipeline.add(
            "renamed",
            RegexCleaner(
                text_field="compact_text",
                output_field="clean_text",
                rules=({"pattern": "alpha", "replacement": "A"},),
            ),
            source=compact["output"],
        )
        pipeline.output("clean", renamed["output"])

        before = len(project.list_operations())
        outputs = project.run_pipeline(pipeline, source)
        after = len(project.list_operations())

        assert after - before == 2
        frame = outputs["clean"].query(
            key_columns=False,
            data_columns=True,
            metadata_columns=False,
            include_position=False,
            form="table",
        )
        assert frame["clean_text"].tolist() == ["A beta", "gamma A"]
    finally:
        project.close()


def test_project_pipeline_reconstructs_recorded_representation_chain(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project-reconstruct"
    project = teal.Project.create(
        project_path,
        name="pipeline_reconstruct",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        count = CountVectorizer(text_field="text", min_df=1)
        counts = project.translate(count, source)["output"]
        trimmer = FeatureTrimmer(min_df=1)
        trimmed = project.translate(trimmer, counts)["output"]
        tfidf = TfidfTransformer(norm=None)
        weighted = project.translate(tfidf, trimmed)["output"]
        svd = SVD(n_components=1, random_state=7)
        reduced = project.translate(svd, weighted)["output"]

        held_out = pd.Series(["alpha gamma", "beta alpha"])
        expected = svd.translate(
            tfidf.translate(
                trimmer.translate(
                    count.translate(held_out)
                )
            )
        )
        expected_values = np.asarray(expected["values"]).copy()
        expected_feature_metadata = expected["feature_metadata"].copy()
        source_id = str(source.artifact_id)
        reduced_id = str(reduced.artifact_id)
        operator_ids = [
            count.operator_id,
            trimmer.operator_id,
            tfidf.operator_id,
            svd.operator_id,
        ]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        recovered = reopened.pipeline(start=source_id, end=reduced_id)
        assert [stage.translator.operator_id for stage in recovered.stages] == operator_ids

        before = len(reopened.list_operations())
        replayed = recovered.translate(held_out, project=reopened)["output"]
        assert len(reopened.list_operations()) == before

        np.testing.assert_allclose(replayed["values"], expected_values)
        pd.testing.assert_frame_equal(
            replayed["feature_metadata"],
            expected_feature_metadata,
        )
    finally:
        reopened.close()


def test_project_pipeline_reconstructs_recomputable_umap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeUMAP:
        def __init__(self, *, n_components=2, **kwargs):
            self.n_components = n_components
            self._fit_offset = None

        @staticmethod
        def _dense(matrix):
            return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

        def fit_transform(self, matrix):
            self.fit(matrix)
            return self.transform(matrix)

        def fit(self, matrix):
            dense = self._dense(matrix)
            self._fit_offset = dense.mean(axis=0)
            return self

        def transform(self, matrix):
            if self._fit_offset is None:
                raise AssertionError("FakeUMAP.transform(...) called before fit(...).")
            dense = self._dense(matrix)
            return (
                dense[:, : self.n_components]
                + self._fit_offset[: self.n_components]
            )

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    project_path = tmp_path / "project-reconstruct-umap"
    training_texts = ["alpha alpha alpha", "beta"]
    source_path = tmp_path / "umap-documents.csv"
    pd.DataFrame({"text": training_texts}).to_csv(source_path, index=False)

    project = teal.Project.create(
        project_path,
        name="pipeline_reconstruct_umap",
        delete_existing=True,
    )
    try:
        source = project.read_csv(
            source_path,
            text_fields="text",
            metadata_fields=None,
        )
        count = CountVectorizer(text_field="text", min_df=1)
        counts = project.translate(count, source)["output"]
        reducer = UMAP(
            n_components=1,
            n_neighbors=2,
            random_state=7,
            reuse="recompute",
        )
        reduced = project.translate(reducer, counts)["output"]

        assert reducer.is_fitted
        assert reducer._estimator is None
        source_id = str(source.artifact_id)
        reduced_id = str(reduced.artifact_id)
        count_operator_id = count.operator_id
        reducer_operator_id = reducer.operator_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        recovered = reopened.pipeline(start=source_id, end=reduced_id)
        assert [stage.translator.operator_id for stage in recovered.stages] == [
            count_operator_id,
            reducer_operator_id,
        ]
        recovered_count = recovered.stages[0].translator
        recovered_umap = recovered.stages[-1].translator
        assert recovered_umap.is_fitted
        assert recovered_umap._estimator is None

        held_out = pd.Series(["alpha", "beta beta beta"])
        fit_payload = recovered_count.translate(pd.Series(training_texts))
        held_out_payload = recovered_count.translate(held_out)
        fit_values = fit_payload["values"]
        held_out_values = held_out_payload["values"]
        if hasattr(fit_values, "toarray"):
            fit_values = fit_values.toarray()
        if hasattr(held_out_values, "toarray"):
            held_out_values = held_out_values.toarray()
        expected = (
            np.asarray(held_out_values)[:, :1]
            + np.asarray(fit_values).mean(axis=0)[:1]
        )

        replayed = recovered.translate(
            held_out,
            project=reopened,
        )["output"]

        np.testing.assert_allclose(replayed["values"], expected)
        assert recovered_umap._estimator is not None
    finally:
        reopened.close()


def test_project_pipeline_reconstructs_unambiguous_multi_source_dag(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project-reconstruct-dag",
        name="pipeline_reconstruct_dag",
        delete_existing=True,
    )
    try:
        source_path = tmp_path / "dag-documents.csv"
        pd.DataFrame({"text": ["alpha", "beta", "gamma"]}).to_csv(
            source_path,
            index=False,
        )
        source = project.read_csv(
            source_path,
            text_fields="text",
            metadata_fields=None,
        )

        count = CountVectorizer(text_field="text", min_df=1)
        counts = project.translate(count, source)["output"]
        transpose = MatrixTranspose()
        embeddings = project.translate(transpose, counts)["output"]
        lookup = EmbeddingLookup(field="text")
        looked_up = project.translate(
            lookup,
            {"tokens": source, "embeddings": embeddings},
        )["output"]

        recovered = project.pipeline(start=source, end=looked_up)
        assert [stage.translator.operator_id for stage in recovered.stages] == [
            count.operator_id,
            transpose.operator_id,
            lookup.operator_id,
        ]

        held_out = pd.Series(["gamma", "alpha"])
        manual = lookup.translate(
            held_out,
            transpose.translate(count.translate(held_out)),
        )
        replayed = recovered.translate(held_out, project=project)["output"]

        np.testing.assert_allclose(replayed["values"], manual["values"])
        pd.testing.assert_frame_equal(
            replayed["feature_metadata"],
            manual["feature_metadata"],
        )
    finally:
        project.close()


def test_project_pipeline_rejects_external_dependency_inside_multi_source_dag(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project-reconstruct-external-dag",
        name="pipeline_reconstruct_external_dag",
        delete_existing=True,
    )
    try:
        source_path = tmp_path / "dag-primary.csv"
        pd.DataFrame({"text": ["alpha", "beta"]}).to_csv(source_path, index=False)
        source = project.read_csv(
            source_path,
            text_fields="text",
            metadata_fields=None,
        )

        external_path = tmp_path / "dag-external.csv"
        pd.DataFrame({"text": ["alpha", "beta"]}).to_csv(external_path, index=False)
        external = project.read_csv(
            external_path,
            text_fields="text",
            metadata_fields=None,
        )
        external_count = CountVectorizer(text_field="text", min_df=1)
        external_counts = project.translate(external_count, external)["output"]
        external_embeddings = project.translate(
            MatrixTranspose(),
            external_counts,
        )["output"]
        target = project.translate(
            EmbeddingLookup(field="text"),
            {"tokens": source, "embeddings": external_embeddings},
        )["output"]

        with pytest.raises(PipelineError, match="not reachable from declared start"):
            project.pipeline(start=source, end=target)
    finally:
        project.close()


def test_project_pipeline_rejects_nonreusable_recorded_stage(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project-reconstruct-nonreusable",
        name="pipeline_reconstruct_nonreusable",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        mapper = FunctionMapper(lambda packet: packet, save_function=False)
        mapped = project.translate(mapper, source)["output"]

        with pytest.raises(
            PipelineError,
            match="cannot be reconstructed.*reusable",
        ):
            project.pipeline(start=source, end=mapped)
    finally:
        project.close()


def test_project_pipeline_rejects_target_outside_declared_start_provenance(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project-unreachable",
        name="pipeline_unreachable",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        target = project.translate(
            RegexCleaner(text_field="text"),
            source,
        )["output"]

        other_path = tmp_path / "other-documents.csv"
        pd.DataFrame({"text": ["unrelated text"]}).to_csv(other_path, index=False)
        other = project.read_csv(
            other_path,
            text_fields="text",
            metadata_fields=None,
        )

        with pytest.raises(PipelineError, match="not reachable from declared start"):
            project.pipeline(start=other, end=target)
    finally:
        project.close()
