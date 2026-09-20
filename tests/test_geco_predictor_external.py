from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import GeCoPredictor


class _ColumnProbability:
    def __init__(self, column: int):
        self.column = int(column)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if sparse.issparse(X):
            values = np.asarray(X[:, self.column].toarray()).reshape(-1)
        else:
            values = np.asarray(X)[:, self.column]
        p = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _seed_matrix(
    project: teal.Project,
    artifact_id: str,
    values,
    *,
    keys=None,
    columns=None,
):
    values = values if sparse.issparse(values) else np.asarray(values)
    n_rows, n_cols = values.shape
    keys = list(range(n_rows)) if keys is None else list(keys)
    columns = [f"x{i}" for i in range(n_cols)] if columns is None else list(columns)
    writer = create_artifact_writer(
        artifact_type="sparse_matrix" if sparse.issparse(values) else "dense_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": keys}),
            "data": {"values": values, "columns": columns},
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix" if sparse.issparse(values) else "dense_matrix",
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def _frame(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=False,
        order_by="_position",
        include_position=True,
        form="table",
    )


def _single_predictor(model, *, threshold=0.5):
    return GeCoPredictor(
        [model],
        source_specs=[{"source_index": 0, "geometry_name": "g0", "n_features": 2}],
        member_specs=[{"source_index": 0, "positive_class": 1, "name": "m0"}],
        aggregation="single",
        threshold=threshold,
        provenance={"geco_name": "single"},
    )


def test_single_sparse_source_list_batch_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "project"
    project = teal.Project.create(path, name="geco_predictor_single")
    try:
        values = sparse.csr_matrix(
            np.array(
                [
                    [0.10, 0.90],
                    [0.25, 0.75],
                    [0.60, 0.40],
                    [0.80, 0.20],
                    [0.45, 0.55],
                ]
            )
        )
        source = _seed_matrix(project, "sparse", values)
        predictor = _single_predictor(_ColumnProbability(0))
        output = project.translate(predictor, [source], batch_size=2)["output"]
        frame = _frame(output)
        np.testing.assert_allclose(frame["probability"], [0.10, 0.25, 0.60, 0.80, 0.45])
        assert frame["prediction"].astype(int).tolist() == [0, 0, 1, 1, 0]
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.descriptor["lineage"]["basis_artifact_ids"] == [
            source.artifact_id
        ]
        operator_id = predictor.operator_id
        source_id = source.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(path)
    try:
        frozen = reopened.get_operator(operator_id)
        assert isinstance(frozen, GeCoPredictor)
        reused = reopened.translate(
            frozen, [reopened.get_artifact(source_id)], batch_size=3
        )["output"]
        np.testing.assert_allclose(
            _frame(reused)["probability"], [0.10, 0.25, 0.60, 0.80, 0.45]
        )
    finally:
        reopened.close()


def test_single_source_bare_artifact_remains_compatible(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="single_compat")
    try:
        source = _seed_matrix(project, "dense", np.array([[0.2, 0.8], [0.7, 0.3]]))
        predictor = _single_predictor(_ColumnProbability(0))
        output = project.translate(predictor, source)["output"]
        np.testing.assert_allclose(_frame(output)["probability"], [0.2, 0.7])
    finally:
        project.close()


def test_fixed_committee_shared_source_is_supplied_once(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="shared_source")
    try:
        values = sparse.csr_matrix(np.array([[0.2, 0.8], [0.6, 0.1], [0.4, 0.9]]))
        source = _seed_matrix(project, "shared", values)
        predictor = GeCoPredictor(
            [_ColumnProbability(0), _ColumnProbability(1)],
            source_specs=[{"source_index": 0, "geometry_name": "shared"}],
            member_specs=[
                {"source_index": 0, "positive_class": 1, "name": "a"},
                {"source_index": 0, "positive_class": 1, "name": "b"},
            ],
            aggregation="mean",
        )
        output = project.translate(predictor, [source])["output"]
        np.testing.assert_allclose(_frame(output)["probability"], [0.5, 0.35, 0.65])
    finally:
        project.close()


def test_fixed_committee_mixed_sparse_dense_sources_and_preserved_output_lineage(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="mixed_sources")
    try:
        sparse_source = _seed_matrix(
            project,
            "sparse",
            sparse.csr_matrix(np.array([[0.2, 9.0], [0.6, 8.0], [0.4, 7.0]])),
        )
        dense_source = _seed_matrix(
            project,
            "dense",
            np.array([[0.9, 2.0], [0.1, 3.0], [0.8, 4.0]]),
        )
        predictor = GeCoPredictor(
            [_ColumnProbability(0), _ColumnProbability(0)],
            source_specs=[
                {"source_index": 0, "geometry_name": "lexical"},
                {"source_index": 1, "geometry_name": "semantic"},
            ],
            member_specs=[
                {"source_index": 0, "positive_class": 1},
                {"source_index": 1, "positive_class": 1},
            ],
            aggregation="maximum",
        )
        output = project.translate(
            predictor, [sparse_source, dense_source], batch_size=2
        )["output"]
        np.testing.assert_allclose(_frame(output)["probability"], [0.9, 0.6, 0.8])
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert output.descriptor["lineage"]["basis_artifact_ids"] == [
            sparse_source.artifact_id,
        ]
        op_sources = project.catalog.operation_sources(output.operation_id)
        assert {
            row["source_label"]: row["source_artifact_id"] for row in op_sources
        } == {
            "source_0": sparse_source.artifact_id,
            "source_1": dense_source.artifact_id,
        }
    finally:
        project.close()


def test_logistic_stacker_preserves_member_order(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="stacker")
    try:
        source0 = _seed_matrix(
            project,
            "g0",
            sparse.csr_matrix(
                np.array([[0.1, 0], [0.8, 0], [0.3, 0], [0.9, 0], [0.6, 0]])
            ),
        )
        source1 = _seed_matrix(
            project,
            "g1",
            np.array([[0.7, 0], [0.2, 0], [0.4, 0], [0.9, 0], [0.1, 0]]),
        )
        p0 = np.array([0.1, 0.8, 0.3, 0.9, 0.6])
        p1 = np.array([0.7, 0.2, 0.4, 0.9, 0.1])
        y = np.array([0, 1, 0, 1, 1])
        stacker = LogisticRegression(random_state=0).fit(np.column_stack([p0, p1]), y)
        expected = stacker.predict_proba(np.column_stack([p0, p1]))[:, 1]
        predictor = GeCoPredictor(
            [_ColumnProbability(0), _ColumnProbability(0)],
            source_specs=[
                {"source_index": 0, "geometry_name": "g0"},
                {"source_index": 1, "geometry_name": "g1"},
            ],
            member_specs=[
                {"source_index": 0, "positive_class": 1, "name": "first"},
                {"source_index": 1, "positive_class": 1, "name": "second"},
            ],
            aggregation="logistic_stack",
            stacker=stacker,
            stacker_positive_class=1,
        )
        output = project.translate(predictor, [source0, source1], batch_size=2)[
            "output"
        ]
        np.testing.assert_allclose(_frame(output)["probability"], expected)
    finally:
        project.close()


def test_same_length_reordered_keys_fail_before_prediction(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="misaligned")
    try:
        source0 = _seed_matrix(
            project, "g0", np.array([[0.1], [0.2], [0.3]]), keys=[0, 1, 2]
        )
        source1 = _seed_matrix(
            project, "g1", np.array([[0.4], [0.5], [0.6]]), keys=[1, 0, 2]
        )
        predictor = GeCoPredictor(
            [_ColumnProbability(0), _ColumnProbability(0)],
            source_specs=[{"source_index": 0}, {"source_index": 1}],
            member_specs=[
                {"source_index": 0, "positive_class": 1},
                {"source_index": 1, "positive_class": 1},
            ],
            aggregation="mean",
        )
        with pytest.raises(ArtifactError, match="not row-aligned"):
            project.translate(predictor, [source0, source1], batch_size=3)
    finally:
        project.close()


def test_row_count_mismatch_fails_before_execution(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="row_mismatch")
    try:
        source0 = _seed_matrix(project, "g0", np.array([[0.1], [0.2], [0.3]]))
        source1 = _seed_matrix(project, "g1", np.array([[0.4], [0.5]]))
        predictor = GeCoPredictor(
            [_ColumnProbability(0), _ColumnProbability(0)],
            source_specs=[{"source_index": 0}, {"source_index": 1}],
            member_specs=[
                {"source_index": 0, "positive_class": 1},
                {"source_index": 1, "positive_class": 1},
            ],
            aggregation="mean",
        )
        with pytest.raises(OperatorError, match="same number of rows"):
            project.translate(predictor, [source0, source1])
    finally:
        project.close()


def test_batch_size_equivalence(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="batch_equivalence")
    try:
        source = _seed_matrix(
            project,
            "g0",
            np.array(
                [
                    [0.05, 0.0],
                    [0.15, 0.0],
                    [0.35, 0.0],
                    [0.55, 0.0],
                    [0.75, 0.0],
                    [0.95, 0.0],
                ]
            ),
        )
        p1 = _single_predictor(_ColumnProbability(0))
        first = project.translate(p1, [source], batch_size=1)["output"]
        p2 = _single_predictor(_ColumnProbability(0))
        second = project.translate(p2, [source], batch_size=4)["output"]
        np.testing.assert_allclose(
            _frame(first)["probability"], _frame(second)["probability"]
        )
    finally:
        project.close()
