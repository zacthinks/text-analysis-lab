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

        # Spy only on the metadata-only projection. Full matrix reads below,
        # including after reopening the project, must remain legitimate.
        with monkeypatch.context() as narrow_patch:
            narrow_patch.setattr(type(dtm), "query", spy)
            narrow = augmented.get_matrix(data_columns=["length"])
            np.testing.assert_allclose(
                narrow.toarray().ravel(),
                [-np.sqrt(1.8), -np.sqrt(.2), np.sqrt(.2), np.sqrt(1.8)],
            )
            assert requested
        artifact_id, source_id, op_id = augmented.artifact_id, dtm.artifact_id, op.operator_id
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


@pytest.mark.parametrize(
    "mapping", [
        {"kind": 7, "other": 3},
        {"kind": np.int64(7), "other": np.float64(3.0)},
        {"kind": np.float32(7.0), "other": np.int32(3)},
    ],
)
def test_legacy_mapping_can_have_kind_as_category(mapping):
    frame = pd.DataFrame({"name": ["kind", "other"]})
    encoder = FeatureMatrixEncoding({"name": mapping}).fit(frame)
    np.testing.assert_array_equal(encoder.transform(frame), [[7], [3]])
    frozen = FeatureMatrixEncoding.from_state(json.loads(json.dumps(encoder.to_state())))
    np.testing.assert_array_equal(frozen.transform(frame), [[7], [3]])


@pytest.mark.parametrize(
    ("width", "anchor", "observations", "expected_edges", "expected_bins"),
    [
        (0.1, 0.0, [0.3, 0.4], [0.3, 0.4, 0.5], [0, 1]),
        (0.1, 0.0, [-0.4, -0.3, -0.2], [-0.4, -0.3, -0.2, -0.1], [0, 1, 2]),
        (0.1, 0.05, [0.25, 0.35], [0.25, 0.35, 0.45], [0, 1]),
        (0.1, -0.05, [-0.35, -0.25], [-0.35, -0.25, -0.15], [0, 1]),
    ],
)
def test_decimal_fixed_width_edges_and_lower_inclusion(
    width, anchor, observations, expected_edges, expected_bins,
):
    config = {
        "kind": "bin", "strategy": "fixed_width",
        "width": width, "anchor": anchor,
    }
    frame = pd.DataFrame({"value": observations})
    encoder = FeatureMatrixEncoding({"value": config}).fit(frame)
    assert len(encoder.columns) == len(expected_edges) - 1
    np.testing.assert_allclose(
        encoder._specs[0]["fitted"]["edges"], expected_edges,
        rtol=0, atol=1e-14,
    )
    actual = encoder.transform(frame, sparse_output=True)
    assert sparse.isspmatrix_csr(actual)
    np.testing.assert_array_equal(actual.argmax(axis=1), np.array(expected_bins).reshape(-1, 1))
    np.testing.assert_array_equal(actual.toarray().sum(axis=1), np.ones(len(frame)))
    reopened = FeatureMatrixEncoding.from_state(json.loads(json.dumps(encoder.to_state())))
    np.testing.assert_array_equal(reopened.transform(frame), actual.toarray())


def test_decimal_fixed_width_half_open_and_frozen_out_of_range():
    encoder = FeatureMatrixEncoding({"value": {
        "kind": "bin", "strategy": "fixed_width", "width": 0.1, "anchor": 0.,
    }}).fit(pd.DataFrame({"value": [0.3, 0.4]}))
    np.testing.assert_array_equal(
        encoder.transform(pd.DataFrame({
            "value": [0.29999999999999993, 0.3, 0.35, 0.4, 0.5, np.nan],
        })),
        [
            [0, 0],  # immediately below the first lower edge
            [1, 0],  # inclusive lower edge
            [1, 0],
            [0, 1],  # exclusive upper edge of preceding bin
            [0, 0],  # upper edge of the final bin
            [0, 0],
        ],
    )


def test_fixed_width_unrepresentable_float_edges_fail_explicitly():
    # Large anchors cannot distinguish widths below one float64 ULP.
    with pytest.raises(ArtifactError, match="represented distinctly"):
        FeatureMatrixEncoding({"value": {
            "kind": "bin", "strategy": "fixed_width",
            "width": 0.1, "anchor": 1e20,
        }}).fit(pd.DataFrame({"value": [1e20]}))


def test_sparse_bin_projection_order_duplicates_and_nonbin_interleaving():
    frame = pd.DataFrame({
        "value": [0.0, 1.0, 2.0, 3.0, np.nan],
        "category": ["a", "b", "a", "b", None],
        "other": [10., 20., 30., 40., 50.],
    })
    encoder = FeatureMatrixEncoding({
        "value": {"kind": "bin", "strategy": "fixed_width", "width": 1},
        "category": "onehot",
        "other": "zscore",
    }).fit(frame)
    all_dense = encoder.transform(frame)
    # Deliberately interleave binned, scalar and categorical features.
    indices = [2, 4, 2, 1, 6, 5, 0, 3]
    selected_columns = encoder.required_columns(indices)
    result = encoder.transform(
        frame.loc[:, selected_columns], sparse_output=True, indices=indices,
    )
    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_equal(result.toarray(), all_dense[:, indices])


def test_sparse_wide_bins_300k_rows_bounded_nnz():
    n_rows, width = 300_000, 4096
    frame = pd.DataFrame({"value": (np.arange(n_rows) % width).astype(float)})
    encoder = FeatureMatrixEncoding({
        "value": {"kind": "bin", "strategy": "fixed_width",
                  "width": 1., "anchor": 0.},
    }).fit_batches(
        (frame.iloc[i:i + 10000] for i in range(0, n_rows, 10000)),
        columns=["value"],
    )
    assert len(encoder.columns) == width
    result = encoder.transform(frame, sparse_output=True)
    assert sparse.isspmatrix_csr(result)
    assert result.shape == (n_rows, width)
    assert result.nnz == n_rows
    np.testing.assert_array_equal(
        result[::29999].argmax(axis=1),
        np.asarray((np.arange(n_rows) % width)[::29999]).reshape(-1, 1),
    )
    # A narrow lazy projection remains one sparse output column.
    narrow = encoder.transform(frame[["value"]], sparse_output=True, indices=[0])
    assert narrow.shape == (n_rows, 1)
    assert narrow.nnz == int(np.count_nonzero((np.arange(n_rows) % width) == 0))


@pytest.mark.parametrize("strategy", ["fixed_width", "quantile", "equal_width"])
@pytest.mark.parametrize("sparse_output", [False, True])
def test_all_missing_fitted_bins_stay_inactive_after_reopen(strategy, sparse_output):
    config = {
        "kind": "bin", "strategy": strategy,
        **({"width": 10, "anchor": 2000} if strategy == "fixed_width"
           else {"n_bins": 4}),
    }
    encoder = FeatureMatrixEncoding({"year": config}).fit(
        pd.DataFrame({"year": [np.nan, np.nan]})
    )
    assert encoder.columns == ("year_bin_0",)
    frozen = encoder.to_state()
    assert frozen["specs"][0]["fitted"]["inactive"] is True
    reopened = FeatureMatrixEncoding.from_state(json.loads(json.dumps(frozen)))
    incoming = pd.DataFrame({"year": [2000., 2010., -100., np.nan]})
    result = reopened.transform(incoming, sparse_output=sparse_output)
    if sparse_output:
        assert sparse.isspmatrix_csr(result)
        assert result.nnz == 0
        result = result.toarray()
    np.testing.assert_array_equal(result, np.zeros((4, 1)))
    # Preserve replay of an older state that omitted the explicit inactive flag.
    legacy = json.loads(json.dumps(frozen))
    del legacy["specs"][0]["fitted"]["inactive"]
    legacy_replay = FeatureMatrixEncoding.from_state(legacy).transform(
        incoming, sparse_output=sparse_output,
    )
    if sparse.issparse(legacy_replay):
        legacy_replay = legacy_replay.toarray()
    np.testing.assert_array_equal(legacy_replay, np.zeros((4, 1)))


@pytest.mark.parametrize("sparse_output", [False, True])
def test_all_missing_explicit_edges_remain_active_after_reopen(sparse_output):
    encoder = FeatureMatrixEncoding({"year": {
        "kind": "bin", "strategy": "boundaries", "boundaries": [2000, 2010],
    }}).fit(pd.DataFrame({"year": [np.nan, np.nan]}))
    assert encoder.to_state()["specs"][0]["fitted"]["inactive"] is False
    frozen = FeatureMatrixEncoding.from_state(
        json.loads(json.dumps(encoder.to_state()))
    )
    result = frozen.transform(
        pd.DataFrame({"year": [1999., 2000., 2010., np.nan]}),
        sparse_output=sparse_output,
    )
    if sparse.issparse(result):
        result = result.toarray()
    np.testing.assert_array_equal(result, [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
    ])


def test_equal_width_nonrepresentable_boundaries_rejected():
    # The 4 requested bins cannot occupy distinct float64 intervals.
    frame = pd.DataFrame({"value": [1e16, 1e16 + 2]})
    with pytest.raises(ArtifactError, match="represented distinctly"):
        FeatureMatrixEncoding({"value": {
            "kind": "bin", "strategy": "equal_width", "n_bins": 4,
        }}).fit(frame)
    valid = FeatureMatrixEncoding({"value": {
        "kind": "bin", "strategy": "equal_width", "n_bins": 1,
    }}).fit(frame)
    assert valid.columns == ("value_bin_0",)
    np.testing.assert_array_equal(
        valid.transform(pd.DataFrame({"value": [1e16, 1e16 + 2]})),
        [[1], [1]],
    )


@pytest.mark.parametrize("sparse_output", [False, True])
def test_large_offset_zscore_frozen_batches_and_replay(sparse_output):
    frame = pd.DataFrame({"score": [1e16, 1e16 + 2]})
    model = FeatureMatrixEncoding({"score": "zscore"}).fit_batches(
        (frame.iloc[:1], frame.iloc[1:]), columns=["score"],
    )
    state = json.loads(json.dumps(model.to_state(), allow_nan=False))
    assert state["specs"][0]["fitted"]["offset"] == 1e16
    assert state["specs"][0]["fitted"]["mean_shift"] == 1.0
    assert state["specs"][0]["fitted"]["sd"] == 1.0
    reopened = FeatureMatrixEncoding.from_state(state)
    actual = reopened.transform(frame, sparse_output=sparse_output)
    if sparse.issparse(actual):
        actual = actual.toarray()
    np.testing.assert_allclose(actual.ravel(), [-1., 1.], rtol=0, atol=1e-14)
    # The old fitted schema with only mean/sd must still be readable.
    legacy = FeatureMatrixEncoding.from_state({
        "specs": [{
            "name": "score", "kind": "zscore", "outputs": ["score"],
            "fitted": {"mean": 20., "sd": 10.},
        }],
    })
    np.testing.assert_allclose(
        legacy.transform(pd.DataFrame({"score": [10., 30.]})).ravel(),
        [-1., 1.],
    )


@pytest.mark.parametrize("sparse_output", [False, True])
def test_extreme_finite_minmax_fit_json_and_replay(sparse_output):
    frame = pd.DataFrame({"score": [-1e308, 1e308, np.nan]})
    fitted = FeatureMatrixEncoding({"score": "minmax"}).fit(frame)
    state = json.loads(json.dumps(fitted.to_state(), allow_nan=False))
    reloaded = FeatureMatrixEncoding.from_state(state)
    transformed = reloaded.transform(frame, sparse_output=sparse_output)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.ravel(), [0., 1., 0.])


@pytest.mark.parametrize("sparse_output", [False, True])
def test_extreme_finite_quantiles_json_frozen_replay(sparse_output):
    frame = pd.DataFrame({"score": [-1e308, 1e308]})
    fitted = FeatureMatrixEncoding({"score": {
        "kind": "bin", "strategy": "quantile", "n_bins": 4,
    }}).fit(frame)
    state = json.loads(json.dumps(fitted.to_state(), allow_nan=False))
    edges = state["specs"][0]["fitted"]["edges"]
    assert edges == [None, -5e307, 0., 5e307, None]
    assert all(np.isfinite(edge) for edge in edges[1:-1])
    reloaded = FeatureMatrixEncoding.from_state(state)
    result = reloaded.transform(frame, sparse_output=sparse_output)
    if sparse.issparse(result):
        result = result.toarray()
    np.testing.assert_array_equal(result, [
        [1, 0, 0, 0], [0, 0, 0, 1],
    ])


def test_unrepresentable_zscore_moments_raise_instead_of_silent_corruption():
    with pytest.raises(ArtifactError, match="stable float64 precision"):
        FeatureMatrixEncoding({"score": "zscore"}).fit(
            pd.DataFrame({"score": [-1e308, 1e308]})
        )


def test_nonfinite_zscore_replay_is_rejected():
    trained = FeatureMatrixEncoding({"score": "zscore"}).fit(
        pd.DataFrame({"score": [1., 1. + 1e-12]})
    )
    with pytest.raises(ArtifactError, match="cannot be represented"):
        trained.transform(pd.DataFrame({"score": [1e308]}))


def test_stable_zscore_batch_independence():
    frame = pd.DataFrame({
        "score": [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6],
    })
    first = FeatureMatrixEncoding({"score": "zscore"}).fit(frame)
    second = FeatureMatrixEncoding({"score": "zscore"}).fit_batches(
        (frame.iloc[:2], frame.iloc[2:]), columns=["score"],
    )
    np.testing.assert_allclose(
        first.transform(frame), second.transform(frame), atol=1e-14,
    )
    np.testing.assert_allclose(
        first.transform(frame).ravel(),
        [-3 / np.sqrt(5), -1 / np.sqrt(5), 1 / np.sqrt(5), 3 / np.sqrt(5)],
        atol=1e-14,
    )


def test_tiny_unrepresentable_zscore_variance_rejected():
    # Distinct finite subnormal observations must not freeze zero variance.
    with pytest.raises(ArtifactError, match="stable float64 precision|underflowed"):
        FeatureMatrixEncoding({"score": "zscore"}).fit(
            pd.DataFrame({"score": [1e-300, 2e-300]})
        )


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int64", [2**53, 2**53 + 1]),
        ("uint64", [2**64 - 2, 2**64 - 1]),
        ("int64", [-2**63, -2**63 + 1]),
        ("Int64", [2**53, 2**53 + 1]),
        ("UInt64", [2**64 - 2, 2**64 - 1]),
    ],
)
@pytest.mark.parametrize("sparse_output", [False, True])
def test_large_integer_scalers_preserve_adjacent_values_after_reopen(
    dtype, values, sparse_output,
):
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    for kind, expected in (("zscore", [-1., 1.]), ("minmax", [0., 1.])):
        fitted = FeatureMatrixEncoding({"value": {"kind": kind}}).fit_batches(
            (frame.iloc[:1], frame.iloc[1:]), columns=["value"],
        )
        state = json.loads(json.dumps(fitted.to_state(), allow_nan=False))
        assert state["specs"][0]["fitted"]["integer_source"] is True
        assert state["specs"][0]["fitted"]["minimum"] == values[0]
        assert state["specs"][0]["fitted"]["maximum"] == values[1]
        reopened = FeatureMatrixEncoding.from_state(state)
        result = reopened.transform(frame, sparse_output=sparse_output)
        if sparse.issparse(result):
            result = result.toarray()
        np.testing.assert_allclose(result.ravel(), expected, atol=1e-14)
        # Freeze must not inadvertently accept rounded float replay values.
        with pytest.raises(ArtifactError, match="requires integer values"):
            reopened.transform(
                pd.DataFrame({"value": np.asarray(values, dtype=float)})
            )


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("strategy", ["fixed_width", "quantile", "equal_width", "boundaries"])
def test_large_integer_binning_rejected_before_lossy_float_conversion(dtype, strategy):
    values = ([2**53, 2**53 + 1] if dtype == "int64"
              else [2**64 - 2, 2**64 - 1])
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    config = {"kind": "bin", "strategy": strategy}
    if strategy == "fixed_width":
        config["width"] = 1
    elif strategy == "boundaries":
        config["boundaries"] = [0]
    else:
        config["n_bins"] = 2
    with pytest.raises(ArtifactError, match="lossless float64 binning range"):
        FeatureMatrixEncoding({"value": config}).fit(frame)
    # Even an ordinary fitted schema must not silently quantize huge
    # integers at replay time; both paths use the same guard.
    ordinary = FeatureMatrixEncoding({"value": {
        "kind": "bin", "strategy": "equal_width", "n_bins": 2,
    }}).fit(pd.DataFrame({"value": [1., 3.]}))
    for sparse_output in (False, True):
        with pytest.raises(ArtifactError, match="lossless float64 binning range"):
            ordinary.transform(frame, sparse_output=sparse_output)


@pytest.mark.parametrize("n_bins", [1, 2, 4])
@pytest.mark.parametrize("sparse_output", [False, True])
def test_extreme_equal_width_bins_are_finite_and_replay(
    n_bins, sparse_output,
):
    frame = pd.DataFrame({"value": [-1e308, 1e308]})
    fitted = FeatureMatrixEncoding({"value": {
        "kind": "bin", "strategy": "equal_width", "n_bins": n_bins,
    }}).fit(frame)
    state = json.loads(json.dumps(fitted.to_state(), allow_nan=False))
    edges = state["specs"][0]["fitted"]["edges"]
    assert len(edges) == n_bins + 1
    if n_bins == 1:
        assert edges == [None, None]
    elif n_bins == 2:
        assert edges == [None, 0.0, None]
    else:
        assert edges == [None, -5e307, 0., 5e307, None]
    frozen = FeatureMatrixEncoding.from_state(state)
    result = frozen.transform(frame, sparse_output=sparse_output)
    if sparse.issparse(result):
        result = result.toarray()
    expected = np.zeros((2, n_bins))
    expected[0, 0] = 1
    expected[1, -1] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
@pytest.mark.parametrize("encoding", [None, "numeric", {"kind": "raw"}])
def test_raw_large_integer_metadata_rejects_lossy_float_conversion(
    dtype, encoding,
):
    values = ([2**53, 2**53 + 1] if dtype == "int64"
              else [2**64 - 2, 2**64 - 1])
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    fitted = FeatureMatrixEncoding(
        {"value": encoding} if encoding is not None else None
    ).fit(frame)
    with pytest.raises(ArtifactError, match="lossless float64 raw numeric-feature"):
        fitted.transform(frame)
