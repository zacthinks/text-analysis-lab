from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactNotFoundError
from text_analysis_lab.translators import TextLength


def _source(project: teal.Project, tmp_path: Path):
    path = tmp_path / "docs.csv"
    pd.DataFrame({"text": ["one two", "three four five", "six"]}).to_csv(
        path, index=False
    )
    return project.read_csv(
        path, text_fields="text", metadata_fields=None, alias="docs"
    )


def test_single_output_alias_reuses_without_executing_and_reports_ignored_settings(
    tmp_path, capsys
):
    project = teal.Project.create(tmp_path / "project", name="alias_reuse")
    try:
        source = _source(project, tmp_path)
        first = project.translate(
            TextLength({"text": "words"}), source, alias="lengths"
        )["output"]
        before = len(project.list_operations())

        second = project.translate(
            TextLength({"text": "characters"}),
            source,
            alias="lengths",
        )["output"]

        assert second.artifact_id == first.artifact_id
        assert len(project.list_operations()) == before
        message = capsys.readouterr().out
        assert "already exists" in message
        assert "operation was not executed" in message
        assert "current settings were ignored" in message
        assert "overwrite=True" in message
    finally:
        project.close()


def test_single_output_overwrite_rebuilds_rebinds_and_soft_deletes_old_artifact(
    tmp_path,
):
    project = teal.Project.create(tmp_path / "project", name="alias_overwrite")
    try:
        source = _source(project, tmp_path)
        old = project.translate(TextLength({"text": "words"}), source, alias="lengths")[
            "output"
        ]
        old_id = old.artifact_id

        new = project.translate(
            TextLength({"text": "characters"}),
            source,
            alias="lengths",
            overwrite=True,
        )["output"]

        assert new.artifact_id != old_id
        assert project.get_artifact("lengths").artifact_id == new.artifact_id
        with pytest.raises(ArtifactNotFoundError):
            project.get_artifact(old_id)
        tombstone = project.get_artifact(old_id, include_deleted=True)
        assert tombstone.artifact_id == old_id
        assert project.catalog.aliases_for_artifact(old_id) == []
        row = next(
            r
            for r in project.list_artifacts(include_deleted=True)
            if r["artifact_id"] == old_id
        )
        assert row["deleted"] is True
    finally:
        project.close()


def test_overwrite_true_on_missing_alias_creates_normally(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="alias_create_overwrite")
    try:
        source = _source(project, tmp_path)
        artifact = project.translate(
            TextLength({"text": "words"}), source, alias="new_lengths", overwrite=True
        )["output"]
        assert project.get_artifact("new_lengths").artifact_id == artifact.artifact_id
    finally:
        project.close()
