from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.translators import DelimiterDecomposer


def _documents(project: teal.Project, tmp_path: Path):
    path = tmp_path / "docs.csv"
    pd.DataFrame({"text": ["a|b", "c|d|e", "f|g"]}).to_csv(path, index=False)
    return project.read_csv(path, text_fields="text", metadata_fields=None, alias="docs")


def test_restrict_alias_reuses_even_when_new_call_arguments_change(tmp_path, capsys):
    project = teal.Project.create(tmp_path / "project", name="restrict_alias")
    try:
        docs = _documents(project, tmp_path)
        segments = project.translate(
            DelimiterDecomposer(delimiter="|", new_key="segment_id"), docs, alias="segments"
        )["output"]
        domain_a = project.select_keys(docs, [0, 1])
        domain_b = project.select_keys(docs, [2])
        first = project.restrict(segments, to=domain_a, alias="analysis_segments")
        before = len(project.list_operations())
        second = project.restrict(segments, to=domain_b, alias="analysis_segments")
        assert second.artifact_id == first.artifact_id
        assert len(project.list_operations()) == before
        assert "current settings were ignored" in capsys.readouterr().out
    finally:
        project.close()


def test_aggregate_alias_reuses_then_overwrites_when_requested(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="aggregate_alias")
    try:
        docs = _documents(project, tmp_path)
        segments = project.translate(
            DelimiterDecomposer(delimiter="|", new_key="segment_id"), docs
        )["output"]
        first = project.aggregate(
            segments,
            to_key="row_id",
            data={"text": "first"},
            alias="document_text_rollup",
        )
        reused = project.aggregate(
            segments,
            to_key="row_id",
            data={"text": "last"},
            alias="document_text_rollup",
        )
        assert reused.artifact_id == first.artifact_id
        rebuilt = project.aggregate(
            segments,
            to_key="row_id",
            data={"text": "last"},
            alias="document_text_rollup",
            overwrite=True,
        )
        assert rebuilt.artifact_id != first.artifact_id
        assert project.get_artifact("document_text_rollup").artifact_id == rebuilt.artifact_id
    finally:
        project.close()
