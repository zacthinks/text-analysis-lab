"""Spectrum and artifact-independent scree plot contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab import analysis, visualization
from text_analysis_lab.analysis.accessor import ArtifactAnalysis
from text_analysis_lab.core.types import ArtifactType


class Matrix:
    artifact_type = ArtifactType.DENSE_MATRIX

    def __init__(self, x, *, is_sparse=False):
        self.x = sparse.csr_matrix(x) if is_sparse else np.asarray(x, dtype=float)
        self.artifact_type = ArtifactType.SPARSE_MATRIX if is_sparse else ArtifactType.DENSE_MATRIX
        self.analysis = ArtifactAnalysis(self)
        self.accesses = 0

    def get_matrix(self):
        self.accesses += 1
        return self.x


@pytest.mark.parametrize("is_sparse", [False, True])
def test_spectrum_matches_exact_diagonal_energy(is_sparse):
    x = np.diag([5.0, 3.0, 1.0])
    source = Matrix(x, is_sparse=is_sparse)
    result = source.analysis.spectrum(n_components=2, random_state=0)
    assert result.component.tolist() == [1, 2]
    np.testing.assert_allclose(result.singular_value, [5, 3], atol=1e-6)
    np.testing.assert_allclose(result.energy_ratio, [25 / 35, 9 / 35], atol=1e-6)
    assert result.cumulative_energy_ratio.iloc[-1] == pytest.approx(34 / 35)
    assert result.cumulative_energy_ratio.iloc[-1] < 1
    assert source.accesses == 1
    pd.testing.assert_frame_equal(
        result, analysis.spectrum(source, n_components=2, random_state=0)
    )


@pytest.mark.parametrize("is_sparse", [False, True])
def test_energy_differs_from_explained_variance_on_uncentered_matrix(is_sparse):
    matrix = Matrix([[10, 0, 0], [10, 1, 0], [10, 0, 1], [10, 0, 0]], is_sparse=is_sparse)
    result = analysis.spectrum(matrix, n_components=2)
    assert result.energy_ratio.iloc[0] > result.explained_variance_ratio.iloc[0]
    assert result.cumulative_energy_ratio.iloc[-1] <= 1
    assert result.cumulative_explained_variance_ratio.iloc[-1] <= 1


@pytest.mark.parametrize("is_sparse", [False, True])
def test_zero_matrix_has_finite_zero_ratios(is_sparse):
    result = analysis.spectrum(Matrix(np.zeros((4, 3)), is_sparse=is_sparse), n_components=2)
    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()
    assert result.energy_ratio.tolist() == [0, 0]
    assert result.explained_variance_ratio.tolist() == [0, 0]


@pytest.mark.parametrize("value", [0, -1, 1.1, True])
def test_reject_bad_component_counts(value):
    with pytest.raises((TypeError, ValueError)):
        analysis.spectrum(Matrix(np.eye(3)), n_components=value)


def test_reject_invalid_input_shapes_and_values():
    with pytest.raises(ValueError, match="exceeds"):
        analysis.spectrum(Matrix(np.eye(3)), n_components=4)
    with pytest.raises(ValueError, match="finite"):
        analysis.spectrum(Matrix([[1, 0, 0], [np.nan, 1, 0], [0, 0, 2]]), n_components=2)
    with pytest.raises(ValueError, match="two rows"):
        analysis.spectrum(Matrix([[1, 2, 3]]), n_components=1)


def test_scree_combines_individual_bars_and_red_cumulative_line():
    pytest.importorskip("plotly")
    source = pd.DataFrame({
        "component": [1, 2, 3],
        "energy_ratio": [0.6, 0.2, 0.1],
        "cumulative_energy_ratio": [0.6, 0.8, 0.9],
        "explained_variance_ratio": [0.5, 0.3, 0.1],
        "cumulative_explained_variance_ratio": [0.5, 0.8, 0.9],
    })
    original = source.copy(deep=True)
    fig = visualization.scree(source)
    assert [trace.type for trace in fig.data] == ["bar", "scatter"]
    np.testing.assert_allclose(fig.data[0].y, [.6, .2, .1])
    np.testing.assert_allclose(fig.data[1].y, [.6, .8, .9])
    assert fig.data[1].line.color == "#c43b43"
    assert fig.data[1].yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    variance = visualization.scree(source, measure="explained_variance", width=800, height=500)
    np.testing.assert_allclose(variance.data[0].y, [.5, .3, .1])
    assert variance.layout.width == 800
    pd.testing.assert_frame_equal(source, original)


def test_scree_rejects_missing_or_misleading_data():
    data = pd.DataFrame({"component": [1, 2], "energy_ratio": [.4, .3], "cumulative_energy_ratio": [.4, .7]})
    with pytest.raises(ValueError, match="missing"):
        visualization.scree(data, measure="explained_variance")
    with pytest.raises(ValueError, match="consistent"):
        visualization.scree(data.assign(cumulative_energy_ratio=[.4, 1.0]))
    with pytest.raises(ValueError, match="consecutive"):
        visualization.scree(data.assign(component=[0, 1]))
