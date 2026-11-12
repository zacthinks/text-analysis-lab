"""Process-local, byte-bounded cache for immutable sparse artifact batches.

Sparse NPZ parts cannot be memory-mapped. A scattered GeCo page often reads
the same batches repeatedly, and scipy.load_npz decompresses the whole part.
Keep a small shared LRU of CSR arrays while bounding retained memory.

Key files by path and filesystem fingerprint so replacements are invalidated.
The cache is an optimization only: oversized matrices remain uncached.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

_MAX_BYTES = 128 * 1024 * 1024
_LOCK = RLock()
_ROWS: OrderedDict[tuple[str, int, int], tuple[Any, int]] = OrderedDict()
_SIZE_BYTES = 0


def _csr_nbytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def sparse_batch(path: Path) -> Any:
    """Return a batch, retaining at most 128 MiB of CSR data across requests."""
    global _SIZE_BYTES
    from scipy import sparse

    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        cached = _ROWS.get(key)
        if cached is not None:
            _ROWS.move_to_end(key)
            return cached[0]

    matrix = sparse.load_npz(path).tocsr()
    bytes_used = _csr_nbytes(matrix)
    if bytes_used > _MAX_BYTES:
        return matrix

    with _LOCK:
        # An overlapping callback may have inserted the same batch already.
        existing = _ROWS.get(key)
        if existing is not None:
            _ROWS.move_to_end(key)
            return existing[0]
        # Drop prior versions of this path.
        for old_key in list(_ROWS):
            if old_key[0] == key[0]:
                _, weight = _ROWS.pop(old_key)
                _SIZE_BYTES -= weight
        while _ROWS and _SIZE_BYTES + bytes_used > _MAX_BYTES:
            _, (_, weight) = _ROWS.popitem(last=False)
            _SIZE_BYTES -= weight
        _ROWS[key] = matrix, bytes_used
        _SIZE_BYTES += bytes_used
    return matrix
