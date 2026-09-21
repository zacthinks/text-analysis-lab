from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError, QueryError


def _install_pickle_parquet(monkeypatch) -> None:
    def fake_to_parquet(self, path, index=False, **kwargs):
        _ = index, kwargs
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_pickle(path)

    def fake_read_parquet(path, columns=None, **kwargs):
        _ = kwargs
        frame = pd.read_pickle(path)
        if columns is not None:
            frame = frame.loc[:, list(columns)]
        return frame.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet, raising=True)


def _seed_source(project, rows: pd.DataFrame, *, primary_key: list[str]):
    artifact_id = "art_source"
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label="source",
        lineage_mode="new_key",
        status="complete",
    )
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": "source",
        "status": "complete",
        "primary_key": primary_key,
        "n_rows": len(rows),
        "components": {"keys": {"format": "parquet_dataset", "path": "keys/"}},
        "lineage": {"lineage_mode": "new_key", "basis_artifact_ids": []},
        "operation_id": None,
        "write": {"mode": "batch", "parts": 1},
    }
    project.storage.artifact_descriptor_path(artifact_id).write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return project.get_artifact(artifact_id)


def _install_source_query(monkeypatch, rows: pd.DataFrame) -> None:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    def fake_query(self, **kwargs):
        _ = self
        frame = rows.copy().reset_index(drop=True)
        frame["_position"] = range(len(frame))
        return frame

    monkeypatch.setattr(BaseArtifact, "query", fake_query, raising=True)


def _install_query_columns(monkeypatch, columns: list[tuple[str, str]]) -> None:
    from text_analysis_lab.core.query import QueryEngine

    def fake_query_columns(self, artifact, *, metadata_mode="full"):
        _ = self, artifact, metadata_mode
        return {
            "output": tuple(name for name, _ in columns),
            "ambiguous": {},
            "columns": tuple(
                {"output_name": name, "namespace": namespace}
                for name, namespace in columns
            ),
        }

    monkeypatch.setattr(QueryEngine, "query_columns", fake_query_columns, raising=True)


def _read_keys(artifact) -> pd.DataFrame:
    parts = int(artifact.descriptor["write"]["parts"])
    frames = [
        pd.read_parquet(artifact.keys_dir / f"part-{index:06d}.parquet")
        for index in range(parts)
    ]
    if not frames:
        return pd.DataFrame(columns=artifact.primary_key)
    return pd.concat(frames, ignore_index=True).loc[:, list(artifact.primary_key)]


def test_split_by_primary_key_field_keeps_higher_level_units_intact(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    rows = pd.DataFrame(
        {
            "doc_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "sent_id": [0, 1, 2] * 4,
        }
    )
    _install_pickle_parquet(monkeypatch)
    _install_source_query(monkeypatch, rows)
    _install_query_columns(monkeypatch, [("doc_id", "key"), ("sent_id", "key")])

    project = teal.Project.create(tmp_path / "project", name="split_by_key")
    try:
        source = _seed_source(project, rows, primary_key=["doc_id", "sent_id"])
        outputs = project.split(
            source,
            labels=("train", "test"),
            proportions=(0.5, 0.5),
            random_state=7,
            by="doc_id",
        )

        train = _read_keys(outputs["train"])
        test = _read_keys(outputs["test"])
        train_docs = set(train["doc_id"].astype(int))
        test_docs = set(test["doc_id"].astype(int))

        assert len(train_docs) == 2
        assert len(test_docs) == 2
        assert train_docs.isdisjoint(test_docs)
        assert train_docs | test_docs == {0, 1, 2, 3}
        assert len(train) == 6
        assert len(test) == 6
        assert set(map(tuple, pd.concat([train, test]).to_numpy())) == set(
            map(tuple, rows[["doc_id", "sent_id"]].to_numpy())
        )

        repeated = project.split(
            source,
            labels=("train", "test"),
            proportions=(0.5, 0.5),
            random_state=7,
            by="doc_id",
        )
        assert set(_read_keys(repeated["train"])["doc_id"].astype(int)) == train_docs
        assert set(_read_keys(repeated["test"])["doc_id"].astype(int)) == test_docs
    finally:
        project.close()


def test_split_by_metadata_assigns_whole_metadata_groups(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    rows = pd.DataFrame(
        {
            "row_id": range(12),
            "school": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
        }
    )
    _install_pickle_parquet(monkeypatch)
    _install_source_query(monkeypatch, rows)
    _install_query_columns(monkeypatch, [("row_id", "key"), ("school", "metadata")])

    project = teal.Project.create(tmp_path / "project", name="split_by_metadata")
    try:
        source = _seed_source(project, rows, primary_key=["row_id"])
        outputs = project.split(
            source,
            labels=("a", "b"),
            proportions=(0.5, 0.5),
            random_state=11,
            by="school",
        )

        assignment_by_row: dict[int, str] = {}
        for label, artifact in outputs.items():
            for row_id in _read_keys(artifact)["row_id"].astype(int):
                assignment_by_row[row_id] = label

        for school_rows in rows.groupby("school")["row_id"]:
            _, row_ids = school_rows
            assert len({assignment_by_row[int(row_id)] for row_id in row_ids}) == 1
    finally:
        project.close()


def test_split_by_and_stratify_operate_over_assignment_units(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    rows = pd.DataFrame(
        {
            "doc_id": [doc for doc in range(8) for _ in range(2)],
            "sent_id": [0, 1] * 8,
            "topic": ["x"] * 8 + ["y"] * 8,
        }
    )
    _install_pickle_parquet(monkeypatch)
    _install_source_query(monkeypatch, rows)
    _install_query_columns(
        monkeypatch,
        [("doc_id", "key"), ("sent_id", "key"), ("topic", "metadata")],
    )

    project = teal.Project.create(tmp_path / "project", name="split_by_stratified")
    try:
        source = _seed_source(project, rows, primary_key=["doc_id", "sent_id"])
        outputs = project.split(
            source,
            labels=("learn", "audit"),
            proportions=(0.5, 0.5),
            random_state=19,
            by="doc_id",
            stratify="topic",
        )

        for artifact in outputs.values():
            assigned_docs = set(_read_keys(artifact)["doc_id"].astype(int))
            assert len(assigned_docs & {0, 1, 2, 3}) == 2
            assert len(assigned_docs & {4, 5, 6, 7}) == 2
    finally:
        project.close()


def test_split_by_rejects_unit_that_spans_multiple_strata(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab as teal

    rows = pd.DataFrame(
        {
            "doc_id": [0, 0, 1, 1],
            "sent_id": [0, 1, 0, 1],
            "topic": ["x", "y", "x", "x"],
        }
    )
    _install_pickle_parquet(monkeypatch)
    _install_source_query(monkeypatch, rows)
    _install_query_columns(
        monkeypatch,
        [("doc_id", "key"), ("sent_id", "key"), ("topic", "metadata")],
    )

    project = teal.Project.create(tmp_path / "project", name="split_bad_strata")
    try:
        source = _seed_source(project, rows, primary_key=["doc_id", "sent_id"])
        with pytest.raises(ArtifactError, match="spans multiple strata"):
            project.split(
                source,
                proportions=(0.5, 0.5),
                random_state=3,
                by="doc_id",
                stratify="topic",
            )
    finally:
        project.close()


def test_split_by_rejects_data_columns(tmp_path: Path, monkeypatch) -> None:
    import text_analysis_lab as teal

    rows = pd.DataFrame({"row_id": range(4), "category": ["a", "a", "b", "b"]})
    _install_pickle_parquet(monkeypatch)
    _install_source_query(monkeypatch, rows)
    _install_query_columns(monkeypatch, [("row_id", "key"), ("category", "data")])

    project = teal.Project.create(tmp_path / "project", name="split_by_data")
    try:
        source = _seed_source(project, rows, primary_key=["row_id"])
        with pytest.raises(QueryError, match="only supports: key, metadata"):
            project.split(source, proportions=(0.5, 0.5), by="category")
    finally:
        project.close()
