from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from text_analysis_lab.translators._umap_persistence import (
    dump_compact_umap,
    load_compact_umap,
)


def _sparse_matrix(*, rows: int, columns: int, density: float):
    rng = np.random.default_rng(42)

    def data_rvs(n):
        return rng.random(n, dtype=np.float32)

    return sparse.random(
        rows,
        columns,
        density=density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
        data_rvs=data_rvs,
    )


def test_compact_sparse_cosine_umap_round_trip_exact(tmp_path: Path) -> None:
    umap = pytest.importorskip("umap")
    pytest.importorskip("pynndescent")

    # >4096 rows exercises UMAP's ordinary PyNNDescent search-index path rather
    # than the small-data exact-neighbor path that would miss the storage bug.
    train = _sparse_matrix(rows=4200, columns=1000, density=0.005)
    test = _sparse_matrix(rows=64, columns=1000, density=0.005)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        transform_seed=42,
        low_memory=True,
    ).fit(train)
    expected = np.asarray(model.transform(test))

    path = tmp_path / "umap.compact.pkl"
    dump_compact_umap(path, model)
    restored = load_compact_umap(path)
    observed = np.asarray(restored.transform(test))

    assert observed.shape == expected.shape
    assert np.array_equal(observed, expected)


def test_compact_umap_is_smaller_than_native_on_pynndescent_path(
    tmp_path: Path,
) -> None:
    umap = pytest.importorskip("umap")
    pytest.importorskip("pynndescent")
    import cloudpickle

    train = _sparse_matrix(rows=4200, columns=1000, density=0.005)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        metric="cosine",
        random_state=42,
        transform_seed=42,
        low_memory=True,
    ).fit(train)

    native = tmp_path / "native.pkl"
    with native.open("wb") as handle:
        cloudpickle.dump(model, handle)
    compact = tmp_path / "compact.pkl"
    dump_compact_umap(compact, model)

    assert compact.stat().st_size < native.stat().st_size
