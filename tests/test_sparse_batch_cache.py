"""Regression tests for bounded sparse batch reuse and invalidation."""

import numpy as np
from scipy import sparse

from text_analysis_lab.core import _sparse_batch_cache as cache


def test_sparse_batch_reuses_decompressed_matrix(tmp_path, monkeypatch):
    path = tmp_path / "batch.npz"
    sparse.save_npz(path, sparse.csr_matrix(np.eye(12)))
    cache._ROWS.clear()
    cache._SIZE_BYTES = 0
    loads = []
    original = sparse.load_npz

    def tracked(file):
        loads.append(str(file))
        return original(file)

    monkeypatch.setattr(sparse, "load_npz", tracked)
    one = cache.sparse_batch(path)
    two = cache.sparse_batch(path)
    assert one is two
    assert len(loads) == 1


def test_sparse_batch_invalidates_file_replacement(tmp_path):
    path = tmp_path / "batch.npz"
    cache._ROWS.clear()
    cache._SIZE_BYTES = 0
    sparse.save_npz(path, sparse.csr_matrix([[1.0]]))
    first = cache.sparse_batch(path)
    sparse.save_npz(path, sparse.csr_matrix([[2.0], [3.0]]))
    second = cache.sparse_batch(path)
    assert first.shape == (1, 1)
    assert second.shape == (2, 1)
    assert second[0, 0] == 2.0


def test_sparse_cache_retains_bounded_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_MAX_BYTES", 256)
    cache._ROWS.clear()
    cache._SIZE_BYTES = 0
    for i in range(8):
        path = tmp_path / f"{i}.npz"
        sparse.save_npz(path, sparse.csr_matrix(np.ones((8, 8))))
        cache.sparse_batch(path)
    assert cache._SIZE_BYTES <= 256
