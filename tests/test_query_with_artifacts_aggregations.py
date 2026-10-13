"""Phase 2B: persisted-project batchwise aggregation across attached artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
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


def test_qualified_attached_collisions_use_narrow_output_names(tmp_path):
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
        caller_kind = next(x["qualified_name"] for x in fields
                           if x["namespace"] == "data" and x["base_name"] == "kind")
        score = next(x["qualified_name"] for x in fields
                     if x["namespace"] == "with" and x["base_name"] == "score")
        kind = next(x["qualified_name"] for x in fields
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
        with pytest.raises(QueryError, match="ambiguous"):
            left.visualize.histogram(
                score, with_artifacts=right,
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
        rhs_cohort = next(col["qualified_name"] for col in fields
                          if col["namespace"] == "with"
                          and col["base_name"] == "cohort")
        rhs_value = next(col["qualified_name"] for col in fields
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
        table = left.analysis.crosstab(
            rhs_cohort, rhs_value, with_artifacts=right, batch_size=1,
        )
        assert table.loc["right_A", 1] == 1
        assert table.loc["right_B", 2] == 1
    finally:
        project.close()
