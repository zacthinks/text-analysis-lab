"""Phase 2B: persisted-project batchwise aggregation across attached artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab import agg
from text_analysis_lab.translators import FunctionMapper
from text_analysis_lab.core.errors import ArtifactError, QueryError


def test_attached_histogram_and_crosstab_across_batch_boundaries(tmp_path):
    project = teal.Project.create(tmp_path / "aggregations", name="aggregations")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 1, 2, 3, 4],
                          "sentence_id": [0, 1, 0, 0, 0],
                          "kind": ["A", "A", "B", "B", "A"]}),
            primary_key=["doc_id", "sentence_id"], data_fields="kind",
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [2, 1, 3],
                          "score": [3.0, 1.0, 5.0],
                          "group": ["Y", "X", "Y"]}),
            primary_key="doc_id", data_fields=["score", "group"],
        )
        result = left.visualize.histogram(
            "score", by="kind", with_artifacts=right,
            bins=[0, 2, 4, 6], batch_size=2, output="result",
        )
        assert result.observed_count == 4
        assert result.missing_count == 1
        counts = result.to_frame()
        assert counts.loc[counts["kind"] == "A", "count"].tolist() == [2, 0, 0]
        assert counts.loc[counts["kind"] == "B", "count"].tolist() == [0, 1, 1]
        dynamic = left.visualize.histogram(
            "score", by="group", with_artifacts=right,
            bins=3, batch_size=2, output="result",
        )
        assert dynamic.observed_count == 4
        assert sum(int(values.sum()) for values in dynamic.counts.values()) == 4
        tab = left.analysis.crosstab(
            "kind", "group", with_artifacts=right, margins=True, batch_size=2,
        )
        assert tab.loc["A", "X"] == 2
        assert tab.loc["B", "Y"] == 2
        assert tab.loc["All", "All"] == 4
        missing = left.analysis.crosstab(
            "kind", "group", with_artifacts=right,
            dropna=False, batch_size=2,
        )
        assert missing.loc["A", "<NA>"] == 1
        with pytest.raises(ArtifactError, match="excluded"):
            left.analysis.crosstab(
                "kind", "group", with_artifacts=right,
                with_columns=False,
            )
    finally:
        project.close()


def test_attached_sparse_numeric_histogram_preserves_zeros_and_null(tmp_path):
    project = teal.Project.create(tmp_path / "sparse", name="sparse")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3, 4]}), primary_key="doc_id",
        )
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2, 3]}),
            "data": {"values": sparse.csr_matrix([[0, 9], [2, 8], [0, 7]]),
                     "columns": ["score", "other"]},
        }], artifact_type="sparse_matrix", primary_key="doc_id")
        result = left.visualize.histogram(
            "score", with_artifacts=right,
            bins=[0, 1, 3], batch_size=2, output="result",
        )
        assert result.observed_count == 3
        assert result.missing_count == 1
        assert result.to_frame()["count"].tolist() == [2, 1]
        tab = left.analysis.crosstab(
            "score", "score", with_artifacts=right, batch_size=2,
        )
        assert tab.loc[0, 0] == 2
        assert tab.loc[2, 2] == 1
        with pytest.raises(ArtifactError, match="excluded"):
            left.visualize.histogram(
                "score", with_artifacts=right,
                with_columns=["other"], output="result",
            )
    finally:
        project.close()


def test_attached_histogram_sql_filter_on_relational_right(tmp_path):
    project = teal.Project.create(tmp_path / "filter", name="filter")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3]}), primary_key="doc_id"
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3],
                          "group": ["A", "B", "A"], "weight": [1, 2, 3]}),
            primary_key="doc_id", data_fields=["group", "weight"],
        )
        result = left.visualize.histogram(
            "weight", with_artifacts=right,
            where='"group" = \'A\'', bins=[0, 2, 4], output="result", batch_size=1,
        )
        assert result.observed_count == 2
        assert result.to_frame()["count"].tolist() == [1, 1]
    finally:
        project.close()


def test_exact_attached_labels_use_narrow_output_names(tmp_path):
    project = teal.Project.create(tmp_path / "collision", name="collision")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2, 3],
                          "score": [100, 200, 300],
                          "kind": ["A", "B", "A"]}),
            primary_key="doc_id", data_fields=["score", "kind"],
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2],
                          "score": [1, 2],
                          "kind": ["X", "Y"]}),
            primary_key="doc_id", data_fields=["score", "kind"],
        )
        fields = left.query_columns(with_artifacts=right)["columns"]
        caller_kind = next(x["output_name"] for x in fields
                           if x["namespace"] == "data" and x["base_name"] == "kind")
        score = next(x["output_name"] for x in fields
                     if x["namespace"] == "with" and x["base_name"] == "score")
        kind = next(x["output_name"] for x in fields
                    if x["namespace"] == "with" and x["base_name"] == "kind")
        hist = left.visualize.histogram(
            score, with_artifacts=right, bins=[0, 1.5, 3],
            batch_size=1, output="result",
        )
        assert hist.observed_count == 2
        assert hist.missing_count == 1
        assert hist.to_frame()["count"].tolist() == [1, 1]
        ct = left.analysis.crosstab(
            caller_kind, kind, with_artifacts=right, batch_size=1,
        )
        assert ct.loc["A", "X"] == 1
        assert ct.loc["B", "Y"] == 1
        ct_both = left.analysis.crosstab(
            kind, score, with_artifacts=right, batch_size=1,
        )
        assert ct_both.loc["X", 1] == 1
        assert ct_both.loc["Y", 2] == 1
        another = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "score": [11, 12]}),
            primary_key="doc_id", data_fields="score",
        )
        with pytest.raises(QueryError, match="ambiguous"):
            left.visualize.histogram(
                score, with_artifacts=[right, another],
                with_columns=["score"], output="result",
            )
    finally:
        project.close()


def test_attached_metadata_collision_uses_projected_output_names(tmp_path):
    project = teal.Project.create(tmp_path / "metadata_collision", name="metadata_collision")
    try:
        left = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2, 3]}),
            "metadata": pd.DataFrame({"cohort": ["caller_A", "caller_B", "caller_C"]}),
            "data": pd.DataFrame({"value": [100, 200, 300]}),
        }], artifact_type="table", primary_key="doc_id")
        right = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "metadata": pd.DataFrame({"cohort": ["right_A", "right_B"]}),
            "data": pd.DataFrame({"value": [1, 2]}),
        }], artifact_type="table", primary_key="doc_id")
        fields = left.query_columns(metadata_mode="full", with_artifacts=right)["columns"]
        rhs_cohort = next(col["output_name"] for col in fields
                          if col["namespace"] == "with"
                          and col["base_name"] == "cohort")
        rhs_value = next(col["output_name"] for col in fields
                         if col["namespace"] == "with"
                         and col["base_name"] == "value")
        hist = left.visualize.histogram(
            rhs_value, by=rhs_cohort, with_artifacts=right,
            bins=[0, 1.5, 3], batch_size=1, output="result",
        )
        assert hist.observed_count == 2
        assert hist.missing_count == 1
        out = hist.to_frame()
        assert out.loc[out[rhs_cohort] == "right_A", "count"].tolist() == [1, 0]
        assert out.loc[out[rhs_cohort] == "right_B", "count"].tolist() == [0, 1]
        # Both analytics queries must retain the full discovery naming
        # universe even when caller metadata is not in the selected fields.
        value_only = left.visualize.histogram(
            rhs_value, with_artifacts=right,
            bins=[0, 1.5, 3], batch_size=1, output="result",
        )
        assert value_only.observed_count == 2
        table = left.analysis.crosstab(
            rhs_cohort, rhs_value, with_artifacts=right, batch_size=1,
        )
        assert table.loc["right_A", 1] == 1
        assert table.loc["right_B", 2] == 1
    finally:
        project.close()



def test_nonvisible_qualified_selectors_are_rejected(tmp_path):
    project = teal.Project.create(tmp_path / "strict_selectors", name="strict_selectors")
    try:
        left = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "kind": ["A", "B"]}),
            primary_key="doc_id", data_fields="kind",
        )
        right = project.register_external(
            pd.DataFrame({"doc_id": [1, 2], "score": [1, 2]}),
            primary_key="doc_id", data_fields="score",
        )
        visible = left.query_columns(with_artifacts=right)["columns"]
        data = next(c for c in visible if c["namespace"] == "data"
                    and c["base_name"] == "kind")
        assert data["output_name"] == "kind"
        assert data["qualified_name"] == "data.kind"
        with pytest.raises(QueryError, match="Available data columns"):
            left.query(
                data_columns=["data.kind"], key_columns=False,
                with_artifacts=right, with_columns=False,
            )
        with pytest.raises(ArtifactError, match="exact output label"):
            left.visualize.histogram(
                "data.kind", by="score", with_artifacts=right,
                output="result", bins=[0, 2, 4],
            )
        # Correct public selectors still work and select the right values.
        result = left.visualize.histogram(
            "score", by="kind", with_artifacts=right,
            output="result", bins=[0, 2, 4],
        )
        assert result.observed_count == 2
    finally:
        project.close()


def test_query_labels_stay_fixed_when_selecting_data_metadata_and_keys(tmp_path):
    project = teal.Project.create(tmp_path / "stable_labels", name="stable_labels")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"row_id": [1, 2]}),
            "data": pd.DataFrame({
                "text": ["data_a", "data_b"],
                "data.text": ["literal_a", "literal_b"],
            }),
            "metadata": pd.DataFrame({
                "text": ["meta_a", "meta_b"],
                "row_id": ["meta_key_a", "meta_key_b"],
            }),
        }], artifact_type="table", primary_key="row_id")
        schema = source.query_columns(metadata_mode="full")
        data_text = next(c["output_name"] for c in schema["columns"]
                         if c["namespace"] == "data" and c["base_name"] == "text")
        literal = next(c["output_name"] for c in schema["columns"]
                       if c["namespace"] == "data" and c["base_name"] == "data.text")
        meta_text = next(c["output_name"] for c in schema["columns"]
                         if c["namespace"] == "metadata" and c["base_name"] == "text")
        meta_key = next(c["output_name"] for c in schema["columns"]
                        if c["namespace"] == "metadata" and c["base_name"] == "row_id")
        full = source.query(metadata_mode="full", metadata_columns=True, form="table")
        assert full[data_text].tolist() == ["data_a", "data_b"]
        assert full[literal].tolist() == ["literal_a", "literal_b"]
        assert full[meta_text].tolist() == ["meta_a", "meta_b"]
        assert full[meta_key].tolist() == ["meta_key_a", "meta_key_b"]
        only_metadata = source.query(
            key_columns=False, data_columns=False, metadata_columns=[meta_text, meta_key],
            metadata_mode="full", form="table")
        assert list(only_metadata.columns) == [meta_text, meta_key]
        assert only_metadata[meta_text].tolist() == ["meta_a", "meta_b"]
        assert only_metadata[meta_key].tolist() == ["meta_key_a", "meta_key_b"]
        only_data = source.query(
            key_columns=False, data_columns=[literal, data_text],
            metadata_mode="full", form="table")
        assert list(only_data.columns) == [literal, data_text]
        only_keys = source.query(
            key_columns=True, data_columns=False,
            metadata_mode="full", form="table")
        assert list(only_keys.columns) == schema["key"]
        batches = list(source.query(
            key_columns=False, data_columns=False, metadata_columns=[meta_text],
            metadata_mode="full", form="table", iter_batches=True, batch_size=1))
        assert [batch[meta_text].iloc[0] for batch in batches] == ["meta_a", "meta_b"]
    finally:
        project.close()


def test_attached_names_do_not_change_after_projection(tmp_path):
    project = teal.Project.create(tmp_path / "stable_attached", name="stable_attached")
    try:
        left = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 2]}),
            "data": pd.DataFrame({"score": [10, 20]}),
            "metadata": pd.DataFrame({"group": ["left_a", "left_b"]}),
        }], artifact_type="table", primary_key="doc_id")
        right = project.register_external(pd.DataFrame({
            "doc_id": [1, 2], "score": [3, 4], "group": ["right_a", "right_b"]}),
            primary_key="doc_id", data_fields=["score", "group"])
        schema = left.query_columns(metadata_mode="full", with_artifacts=right)
        cols = schema["columns"]
        attached_score = next(c["output_name"] for c in cols
                              if c["namespace"] == "with" and c["base_name"] == "score")
        attached_group = next(c["output_name"] for c in cols
                              if c["namespace"] == "with" and c["base_name"] == "group")
        full = left.query(metadata_mode="full", metadata_columns=True,
                          with_artifacts=right, form="table")
        narrow = left.query(
            key_columns=False, data_columns=False, metadata_columns=False,
            metadata_mode="full", with_artifacts=right,
            with_columns=[attached_score, attached_group], form="table")
        assert narrow.columns.tolist() == [attached_score, attached_group]
        assert narrow[attached_score].tolist() == full[attached_score].tolist()
        assert narrow[attached_group].tolist() == full[attached_group].tolist()
    finally:
        project.close()


def test_literal_dotted_field_identity_in_consumers(tmp_path):
    project = teal.Project.create(tmp_path / "consumer_identity", name="consumer_identity")
    try:
        source = project.register_external([{
            "keys": pd.DataFrame({"doc_id": [1, 1, 2, 2], "turn_id": [0, 1, 0, 1]}),
            "data": pd.DataFrame({
                "text": ["base_a", "base_a", "base_b", "base_b"],
                "data.text": ["literal_x", "literal_x", "literal_y", "literal_y"],
            }),
            "metadata": pd.DataFrame({
                "text": ["meta_a", "meta_a", "meta_b", "meta_b"],
            }),
        }], artifact_type="table", primary_key=["doc_id", "turn_id"])
        info = source.query_columns(metadata_mode="full")
        labels = {
            (item["namespace"], item["base_name"]): item["output_name"]
            for item in info["columns"]
        }
        short = labels["data", "text"]
        dotted = labels["data", "data.text"]
        assert short != dotted
        assert source.query(metadata_mode="full", data_columns=[short, dotted],
                            form="table")[dotted].tolist() == [
                                "literal_x", "literal_x", "literal_y", "literal_y",
                            ]

        reduced = project.aggregate(
            source, to_key="doc_id",
            data={
                "from_short": agg(short, "first"),
                "from_dotted": agg(dotted, "first"),
            },
        )
        result = reduced.query(form="table")
        assert result["from_short"].tolist() == ["base_a", "base_b"]
        assert result["from_dotted"].tolist() == ["literal_x", "literal_y"]

        collapsed = project.collapse_runs(
            source, by=dotted, data={"picked": agg(short, "first")},
        )
        spans = collapsed.query(metadata_mode="local", metadata_columns=True,
                                form="table")
        assert spans["picked"].tolist() == ["base_a", "base_b"]
        assert spans[dotted].tolist() == ["literal_x", "literal_y"]

        mapper = FunctionMapper(
            lambda packet: {
                "data": pd.DataFrame({
                    "picked": packet["data"][dotted].reset_index(drop=True),
                })
            }
        )
        mapped = project.translate(
            mapper, source, data_columns=[dotted], metadata_columns=False,
            metadata_mode="full", batch_size=2,
        )["output"]
        assert mapped.query(form="table")["picked"].tolist() == [
            "literal_x", "literal_x", "literal_y", "literal_y",
        ]
        # A qualified identity that was not advertised as an output must not
        # be silently interpreted as the other source field.
        nonvisible = next(
            item["qualified_name"] for item in info["columns"]
            if item["namespace"] == "data" and item["base_name"] == "text"
        )
        if nonvisible not in {short, dotted}:
            with pytest.raises(ArtifactError, match="unknown"):
                project.translate(
                    FunctionMapper(lambda packet: {"data": packet["data"].copy()}),
                    source, data_columns=[nonvisible], metadata_mode="full",
                )
    finally:
        project.close()


