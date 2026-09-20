from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import AliasOverwriteBlockedError
from text_analysis_lab.translators import TextLength


def _source(project: teal.Project, tmp_path: Path):
    path = tmp_path / "docs.csv"
    pd.DataFrame({"text": ["one two", "three four", "five six"]}).to_csv(
        path, index=False
    )
    return project.read_csv(
        path, text_fields="text", metadata_fields=None, alias="docs"
    )


def test_overwrite_refuses_artifact_with_live_downstream_dependency(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="dependency_block")
    try:
        source = _source(project, tmp_path)
        current = project.translate(
            TextLength({"text": "words"}), source, alias="lengths"
        )["output"]
        project.translate(TextLength({"text": "characters"}), current)
        before = len(project.list_operations())
        with pytest.raises(AliasOverwriteBlockedError, match="live dependents"):
            project.translate(
                TextLength({"text": "characters"}),
                source,
                alias="lengths",
                overwrite=True,
            )
        assert len(project.list_operations()) == before
        assert project.get_artifact("lengths").artifact_id == current.artifact_id
    finally:
        project.close()


def test_overwrite_refuses_artifact_with_another_alias(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="alias_block")
    try:
        source = _source(project, tmp_path)
        current = project.translate(
            TextLength({"text": "words"}), source, alias="lengths"
        )["output"]
        current.add_alias("important_secondary_name")
        with pytest.raises(AliasOverwriteBlockedError, match="also has aliases"):
            project.translate(
                TextLength({"text": "characters"}),
                source,
                alias="lengths",
                overwrite=True,
            )
        assert project.get_artifact("lengths").artifact_id == current.artifact_id
        assert (
            project.get_artifact("important_secondary_name").artifact_id
            == current.artifact_id
        )
    finally:
        project.close()


def test_probability_split_can_overwrite_internal_dependency_bundle(tmp_path):
    project = teal.Project.create(
        tmp_path / "project", name="internal_bundle_dependency"
    )
    try:
        source = _source(project, tmp_path)
        aliases = {
            "remainder": "audit_remainder",
            "sample": "audit_sample",
            "pi": "audit_pi",
        }
        old = project.probability_split(source, n=1, random_state=1, alias=aliases)
        # pi depends on sample inside the same operation bundle. That must not block
        # replacing the whole bundle together.
        new = project.probability_split(
            source,
            n=2,
            random_state=2,
            alias=aliases,
            overwrite=True,
        )
        assert all(
            new[label].artifact_id != old[label].artifact_id for label in aliases
        )
    finally:
        project.close()
