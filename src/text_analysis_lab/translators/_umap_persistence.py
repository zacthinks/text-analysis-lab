"""TeAL-specific compact persistence for fitted ``umap.UMAP`` estimators.

Upstream PyNNDescent serializes compiled/runtime search callables that it rebuilds
on load. For large sparse text matrices those callables can dominate the pickle.
TeAL's compact format stores the fitted/search data but omits rebuildable runtime
machinery, then reconstructs it when loading.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cloudpickle

from text_analysis_lab.core.errors import OperatorError

_COMPACT_SCHEMA_VERSION = 1
_RUNTIME_NNDESCENT_ATTRS = (
    "_search_function",
    "_tree_search",
    "_deheap_function",
    "_rerank_function",
    "_distance_func",
    "_true_distance_func",
    "_quantized_distance_func",
    "_quantized_distance_func_base",
)


def dump_compact_umap(path: Path, estimator: Any) -> None:
    """Write exact transform-capable UMAP state without rebuildable callables."""
    state = dict(getattr(estimator, "__dict__", {}))
    if not state:
        raise OperatorError(
            "Compact UMAP persistence requires estimator __dict__ state."
        )

    index = state.get("_knn_search_index")
    index_state = None
    if index is not None:
        getter = getattr(index, "__getstate__", None)
        if not callable(getter):
            raise OperatorError(
                "Compact UMAP persistence requires a PyNNDescent search index "
                "with __getstate__(). Use storage='native' only if upstream "
                "serialization is explicitly required."
            )
        raw_index_state = getter()
        if not isinstance(raw_index_state, Mapping):
            raise OperatorError("PyNNDescent __getstate__() did not return a mapping.")
        index_state = dict(raw_index_state)
        for name in _RUNTIME_NNDESCENT_ATTRS:
            index_state.pop(name, None)
        # The index is represented separately so cloudpickle never sees the
        # original object graph and its compiled callables.
        state["_knn_search_index"] = None

    payload = {
        "schema_version": _COMPACT_SCHEMA_VERSION,
        "umap_state": state,
        "nndescent_state": index_state,
    }
    with Path(path).open("wb") as handle:
        cloudpickle.dump(payload, handle)


def load_compact_umap(path: Path) -> Any:
    """Restore a UMAP estimator from TeAL's compact exact-state format."""
    path = Path(path)
    if not path.exists():
        raise OperatorError(f"Missing fitted UMAP asset: {path}.")
    with path.open("rb") as handle:
        payload = cloudpickle.load(handle)
    if not isinstance(payload, Mapping):
        raise OperatorError("Compact UMAP asset must contain a mapping payload.")
    if int(payload.get("schema_version", -1)) != _COMPACT_SCHEMA_VERSION:
        raise OperatorError(
            "Unsupported compact UMAP persistence schema "
            f"{payload.get('schema_version')!r}."
        )

    raw_state = payload.get("umap_state")
    if not isinstance(raw_state, Mapping):
        raise OperatorError("Compact UMAP asset is missing umap_state.")

    try:
        import umap
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise OperatorError(
            "Loading a compact UMAP asset requires umap-learn."
        ) from exc

    estimator = umap.UMAP.__new__(umap.UMAP)
    estimator.__dict__ = dict(raw_state)

    raw_index_state = payload.get("nndescent_state")
    if raw_index_state is not None:
        if not isinstance(raw_index_state, Mapping):
            raise OperatorError("Compact UMAP nndescent_state must be a mapping.")
        estimator._knn_search_index = _restore_nndescent(dict(raw_index_state))
    return estimator


def _restore_nndescent(state: dict[str, Any]) -> Any:
    """Restore NNDescent while rebuilding runtime functions instead of loading them."""
    try:
        import numba
        import pynndescent.pynndescent_ as pynn
        import pynndescent.sparse as pynn_sparse
        from pynndescent.rp_trees import renumbaify_tree
    except ImportError as exc:  # pragma: no cover - transitive UMAP dependency
        raise OperatorError(
            "Compact UMAP persistence requires pynndescent and numba."
        ) from exc

    index = pynn.NNDescent.__new__(pynn.NNDescent)
    index.__dict__ = state

    raw_forest = state.get("_search_forest", ())
    index._search_forest = tuple(renumbaify_tree(tree) for tree in raw_forest)

    if bool(getattr(index, "_is_sparse", False)):
        _restore_sparse_distance(index, pynn_sparse=pynn_sparse, numba=numba)
        init_sparse = getattr(index, "_init_sparse_search_function", None)
        if not callable(init_sparse):
            raise OperatorError(
                "Installed pynndescent cannot rebuild sparse UMAP search state."
            )
        init_sparse()
    else:
        set_distance = getattr(index, "_set_distance_func", None)
        init_search = getattr(index, "_init_search_function", None)
        if not callable(set_distance) or not callable(init_search):
            raise OperatorError(
                "Installed pynndescent cannot rebuild dense UMAP search state."
            )
        set_distance()
        init_search()
    return index


def _restore_sparse_distance(index: Any, *, pynn_sparse: Any, numba: Any) -> None:
    """Recreate the sparse distance function used by NNDescent construction.

    PyNNDescent 0.6.0's generic ``__setstate__`` selects the dense metric table
    before building sparse search, which breaks sparse cosine models. Mirror the
    sparse constructor path here instead.
    """
    metric = index.metric
    index._is_proxy_distance = False

    if callable(metric):
        distance_func = metric
    elif metric in pynn_sparse.sparse_fast_distance_alternatives:
        alternative = pynn_sparse.sparse_fast_distance_alternatives[metric]
        distance_func = alternative["dist"]
        index._distance_correction = alternative["correction"]
    elif metric in pynn_sparse.sparse_named_distances:
        distance_func = pynn_sparse.sparse_named_distances[metric]
    else:
        raise OperatorError(
            f"Metric {metric!r} is not supported for compact sparse UMAP restore."
        )

    dist_args = tuple(getattr(index, "_dist_args", ()))
    if dist_args:
        base_distance_func = distance_func

        @numba.njit()
        def partial_sparse_dist(ind1, data1, ind2, data2):
            return base_distance_func(ind1, data1, ind2, data2, *dist_args)

        index._distance_func = partial_sparse_dist
    else:
        index._distance_func = distance_func
