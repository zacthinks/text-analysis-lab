"""Metadata transforms retain frozen FeatureMatrix recipe semantics."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.translators._feature_matrix_encoding import FeatureMatrixEncoding
from text_analysis_lab.core.errors import ArtifactError


def test_decade_anchor_lower_edges_and_frozen_other_rows():
    rows = pd.DataFrame({"year": [1990, 1999, 2000, 2009, 2010, np.nan]})
    encoder = FeatureMatrixEncoding({"year": {
        "kind": "bin", "strategy": "fixed_width", "width": 10, "anchor": 2000,
    }}).fit(rows)
    assert encoder.columns == ("year_bin_0", "year_bin_1", "year_bin_2")
    assert encoder._specs[0]["fitted"]["edges"] == [1990, 2000, 2010, 2020]
    np.testing.assert_array_equal(encoder.transform(rows), [
        [1, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 1, 0], [0, 0, 1], [0, 0, 0],
    ])
    shifted = FeatureMatrixEncoding({"year": {
        "kind": "bin", "strategy": "fixed_width", "width": 10, "anchor": 2005,
    }}).fit(rows)
    assert shifted._specs[0]["fitted"]["edges"] == [1985, 1995, 2005, 2015]
    frozen = FeatureMatrixEncoding.from_state(json.loads(json.dumps(encoder.to_state())))
    np.testing.assert_array_equal(
        frozen.transform(pd.DataFrame({"year": [1980, 2000, 2020, np.nan]})),
        [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
    )


def test_zscore_minmax_raw_onehot_and_selected_columns():
    frame = pd.DataFrame({
        "z": [10., 20., 30., np.nan],
        "mm": [10., 20., 30., np.nan],
        "raw": [5., 6., 7., 8.],
        "category": ["x", "y", "x", None],
    })
    op = FeatureMatrixEncoding({
        "z": {"kind": "zscore"}, "mm": {"kind": "minmax"},
        "raw": {"kind": "raw"}, "category": "onehot",
    }).fit_batches((frame.iloc[:2], frame.iloc[2:]), columns=frame.columns)
    actual = op.transform(frame)
    np.testing.assert_allclose(actual[:3, 0], [-np.sqrt(1.5), 0., np.sqrt(1.5)])
    np.testing.assert_allclose(actual[:, 1], [0., .5, 1., 0.])
    np.testing.assert_allclose(actual[:, 2], [5., 6., 7., 8.])
    assert op.required_columns([4]) == ["category"]
    np.testing.assert_array_equal(
        op.transform(frame[["category"]], indices=[4]), [[0], [1], [0], [0]],
    )
    np.testing.assert_array_equal(
        op.transform(pd.DataFrame({
            "z": [100.], "mm": [40.], "raw": [9.], "category": ["unseen"],
        }))[:, 3:],
        [[0., 0.]],
    )


@pytest.mark.parametrize("strategy", ["quantile", "equal_width"])
def test_bins_ties_missing_and_outliers(strategy):
    cfg = {"kind": "bin", "strategy": strategy, "n_bins": 4}
    frame = pd.DataFrame({"n": [1., 1., 1., 1., 2., 2., 2., 2., np.nan]})
    fitted = FeatureMatrixEncoding({"n": cfg}).fit(frame)
    assert 1 <= len(fitted.columns) <= 4
    output = fitted.transform(frame)
    assert np.all(output[:8].sum(axis=1) == 1)
    assert output[-1].sum() == 0
    new = fitted.transform(pd.DataFrame({"n": [-100., 100.]}))
    assert np.all(new.sum(axis=1) == 1)
    assert sparse.isspmatrix_csr(fitted.transform(frame, sparse_output=True))


def test_explicit_edges_right_side_and_zero_variance():
    encoder = FeatureMatrixEncoding({
        "edge": {"kind": "bin", "strategy": "boundaries", "boundaries": [0, 10]},
        "constant": "zscore",
    }).fit(pd.DataFrame({"edge": [-1., 0., 10., 11.], "constant": [5.] * 4}))
    np.testing.assert_array_equal(encoder.transform(pd.DataFrame({
        "edge": [-1., 0., 10., 11., np.nan],
        "constant": [5., 5., 5., 5., np.nan],
    })), [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [0, 0, 1, 0], [0, 0, 0, 0],
    ])
    assert json.loads(json.dumps(encoder.to_state()))


def test_quantile_deterministic_bounded_fit():
    frame = pd.DataFrame({"x": np.arange(300_000, dtype=float)})
    a = FeatureMatrixEncoding({"x": {
        "kind": "bin", "strategy": "quantile", "n_bins": 10,
    }}).fit_batches((frame.iloc[i:i+3000] for i in range(0, len(frame), 3000)),
                   columns=["x"])
    b = FeatureMatrixEncoding({"x": {
        "kind": "bin", "strategy": "quantile", "n_bins": 10,
    }}).fit_batches((frame.iloc[i:i+7500] for i in range(0, len(frame), 7500)),
                   columns=["x"])
    assert a.to_state() == b.to_state()
    assert len(a.columns) == 10


def test_numerical_validation():
    for cfg in (
        {"kind": "bin", "strategy": "fixed_width", "width": 0},
        {"kind": "bin", "strategy": "quantile", "n_bins": 0},
        {"kind": "bin", "strategy": "boundaries", "boundaries": [10, 0]},
        {"kind": "bin", "strategy": "equal_width", "n_bins": 2, "weird": True},
    ):
        with pytest.raises(ArtifactError):
            FeatureMatrixEncoding({"v": cfg}).fit(pd.DataFrame({"v": [1, 2]}))
    with pytest.raises(ArtifactError, match="finite"):
        FeatureMatrixEncoding({"v": "zscore"}).fit(pd.DataFrame({"v": [np.inf]}))


def test_real_project_sparse_inherited_metadata_reopen_and_pushdown(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    from text_analysis_lab import Project
    from text_analysis_lab.translators import CountVectorizer, FeatureMatrix

    input_csv = tmp_path / "dataset.csv"
    pd.DataFrame({
        "text": ["alpha beta", "beta beta", "alpha gamma", "gamma beta"],
        "year": [1990, 1999, 2000, 2010],
        "source": ["a", "b", "a", "c"],
        "length": [100, 200, 300, 400],
    }).to_csv(input_csv, index=False)
    project_path = tmp_path / "lab"
    project = Project.create(project_path, name="augmented-metadata")
    try:
        source = project.read_csv(
            input_csv, text_fields="text",
            metadata_fields=["year", "source", "length"],
        )
        dtm = project.translate(CountVectorizer(text_field="text"), source)["output"]
        op = FeatureMatrix(
            metadata_columns=["year", "source", "length"],
            encodings={
                "year": {"kind": "bin", "strategy": "fixed_width",
                         "width": 10, "anchor": 2000},
                "source": "onehot",
                "length": {"kind": "zscore"},
            },
        )
        augmented = project.translate(op, dtm)["output"]
        assert augmented.artifact_type.value == "sparse_matrix"
        assert sparse.issparse(augmented.get_matrix())
        assert augmented.get_matrix().shape[1] == dtm.get_matrix().shape[1] + 3 + 3 + 1
        requested = []
        original_query = type(dtm).query

        def spy(artifact, **kwargs):
            if artifact.artifact_id == dtm.artifact_id:
                requested.append(kwargs)
                if kwargs.get("form") == "native" and kwargs.get("data_columns") is not False:
                    pytest.fail("Metadata-only projection fetched DTM")
            return original_query(artifact, **kwargs)

        monkeypatch.setattr(type(dtm), "query", spy)
        narrow = augmented.get_matrix(data_columns=["length"])
        np.testing.assert_allclose(
            narrow.toarray().ravel(),
            [-np.sqrt(1.8), -np.sqrt(.2), np.sqrt(.2), np.sqrt(1.8)],
        )
        assert requested
        artifact_id, source_id, op_id = augmented.artifact_id, dtm.artifact_id, op.operator_id
        expected = augmented.get_matrix().toarray() if False else None
    finally:
        project.close()
    reopened = Project.open(project_path)
    try:
        restored = reopened.get_artifact(artifact_id)
        assert sparse.issparse(restored.get_matrix())
        frozen = reopened.get_operator(op_id)
        replay = reopened.translate(frozen, reopened.get_artifact(source_id))["output"]
        np.testing.assert_array_equal(replay.get_matrix().toarray(), restored.get_matrix().toarray())
        assert replay.get_data_columns() == restored.get_data_columns()
    finally:
        reopened.close()
