from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import AliasBundleError, ArtifactNotFoundError


def _source(project: teal.Project, tmp_path: Path, n: int = 10):
    path = tmp_path / "docs.csv"
    pd.DataFrame({"text": [f"document {i}" for i in range(n)]}).to_csv(path, index=False)
    return project.read_csv(path, text_fields="text", metadata_fields=None, alias="docs")


def test_multioutput_split_reuses_complete_alias_bundle(tmp_path, capsys):
    project = teal.Project.create(tmp_path / "project", name="split_reuse")
    try:
        source = _source(project, tmp_path)
        aliases = {"learn": "learn_docs", "audit": "audit_docs"}
        first = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.7, 0.3),
            random_state=1,
            alias=aliases,
        )
        before = len(project.list_operations())
        second = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.5, 0.5),
            random_state=999,
            alias=aliases,
        )
        assert {k: v.artifact_id for k, v in second.items()} == {
            k: v.artifact_id for k, v in first.items()
        }
        assert len(project.list_operations()) == before
        message = capsys.readouterr().out
        assert "existing output bundle" in message
        assert "current settings were ignored" in message
    finally:
        project.close()


def test_multioutput_alias_mapping_must_be_complete_and_cannot_be_string(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="split_shape")
    try:
        source = _source(project, tmp_path)
        with pytest.raises(AliasBundleError, match="require alias"):
            project.split(
                source,
                labels=("learn", "audit"),
                proportions=(0.7, 0.3),
                alias="split",
            )
        with pytest.raises(AliasBundleError, match="complete output bundle"):
            project.split(
                source,
                labels=("learn", "audit"),
                proportions=(0.7, 0.3),
                alias={"learn": "learn_docs"},
            )
    finally:
        project.close()


def test_partial_existing_multioutput_bundle_fails_without_new_operation(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="split_partial")
    try:
        source = _source(project, tmp_path)
        raw = project.split(source, labels=("learn", "audit"), proportions=(0.7, 0.3), random_state=3)
        raw["learn"].add_alias("learn_docs")
        before = len(project.list_operations())
        with pytest.raises(AliasBundleError, match="partially present"):
            project.split(
                source,
                labels=("learn", "audit"),
                proportions=(0.7, 0.3),
                random_state=4,
                alias={"learn": "learn_docs", "audit": "audit_docs"},
            )
        assert len(project.list_operations()) == before
    finally:
        project.close()


def test_multioutput_overwrite_retires_whole_bundle_atomically(tmp_path):
    project = teal.Project.create(tmp_path / "project", name="split_overwrite")
    try:
        source = _source(project, tmp_path)
        aliases = {"learn": "learn_docs", "audit": "audit_docs"}
        old = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.7, 0.3),
            random_state=1,
            alias=aliases,
        )
        old_ids = {k: v.artifact_id for k, v in old.items()}
        new = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.5, 0.5),
            random_state=2,
            alias=aliases,
            overwrite=True,
        )
        assert all(new[k].artifact_id != old_ids[k] for k in aliases)
        for label, alias in aliases.items():
            assert project.get_artifact(alias).artifact_id == new[label].artifact_id
            with pytest.raises(ArtifactNotFoundError):
                project.get_artifact(old_ids[label])
    finally:
        project.close()
