"""Nearest-neighbor analytic methods for matrix artifacts."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._matrix_utils import require_matrix_artifact, resolve_position
from text_analysis_lab.core.errors import ArtifactError, UnsupportedArtifactOperationError
from text_analysis_lab.core.lineage import is_prefix

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


def nearest_neighbors(
    artifact: "BaseArtifact",
    *,
    key: Any | None = None,
    position: int | None = None,
    row_name: str | None = None,
    k: int = 10,
    metric: str = "cosine",
    include_self: bool = False,
    batch_size: int = 10_000,
    context: "BaseArtifact | str | None" = None,
    context_data_columns: Any = False,
    context_metadata_columns: Any = False,
    context_metadata_mode: str = "none",
    **metric_kwargs: Any,
) -> pd.DataFrame:
    """Return the nearest rows to one focal row of a matrix artifact.

    The search is ephemeral and bounded by ``batch_size``: matrix rows are read
    batch-wise and only the best ``k`` candidates are retained in memory.  The
    returned DataFrame contains the artifact primary key, optional row name,
    ``_position``, one-based ``rank``, and ``distance``. When ``context`` is
    supplied, selected context data/metadata columns are appended by exact-key
    or leading-prefix-key alignment.
    """
    require_matrix_artifact(artifact, "nearest_neighbors()")
    if int(k) < 0:
        raise ValueError("k must be non-negative.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    focus_position = resolve_position(
        artifact,
        key=key,
        position=position,
        row_name=row_name,
        argument_name="row",
    )
    primary_key = [str(value) for value in artifact.primary_key]
    row_name_column = getattr(artifact, "row_name", None)
    output_columns = [*primary_key]
    if row_name_column is not None and row_name_column not in output_columns:
        output_columns.append(str(row_name_column))
    output_columns.extend(["_position", "rank", "distance"])
    context_artifact = None if context is None else artifact.project.get_artifact(context)
    context_columns: list[str] = []
    if context_artifact is not None:
        _validate_context_keys(artifact, context_artifact)
        context_columns = _selected_context_output_columns(
            context_artifact,
            data_columns=context_data_columns,
            metadata_columns=context_metadata_columns,
            metadata_mode=context_metadata_mode,
        )
        collisions = sorted(set(output_columns).intersection(context_columns))
        if collisions:
            raise ArtifactError(
                f"nearest-neighbor context columns collide with result columns: {collisions}."
            )
    if int(k) == 0:
        return pd.DataFrame(columns=[*output_columns, *context_columns])

    try:
        from sklearn.metrics import pairwise_distances
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise UnsupportedArtifactOperationError(
            "nearest_neighbors() requires scikit-learn."
        ) from exc

    focus_matrix = artifact.get_matrix(positions=[focus_position])
    if getattr(focus_matrix, "shape", (0,))[0] != 1:
        raise RuntimeError(
            f"Could not materialize focal matrix row at position {focus_position}."
        )

    # Min-heap over (-distance, -position, position, distance).  The root is the
    # worst retained candidate, so memory remains O(k) regardless of row count.
    retained: list[tuple[float, int, int, float]] = []

    for batch in artifact.iter_batches(
        batch_size=int(batch_size),
        key_columns=False,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        form="native",
        include_position=True,
    ):
        info = batch["info"]
        matrix = batch["matrix"]
        if info.empty:
            continue

        distances = np.asarray(
            pairwise_distances(
                matrix, focus_matrix, metric=metric, **metric_kwargs
            )
        ).reshape(-1)
        positions = info["_position"].astype(int).tolist()

        for candidate_position, raw_distance in zip(
            positions, distances, strict=True
        ):
            if not include_self and candidate_position == focus_position:
                continue
            distance = float(raw_distance)
            item = (-distance, -candidate_position, candidate_position, distance)
            if len(retained) < int(k):
                heapq.heappush(retained, item)
            elif item > retained[0]:
                heapq.heapreplace(retained, item)

    ordered = sorted(
        ((item[3], item[2]) for item in retained),
        key=lambda value: (value[0], value[1]),
    )
    if not ordered:
        return pd.DataFrame(columns=[*output_columns, *context_columns])

    result_positions = [position_value for _, position_value in ordered]
    key_frame = artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        metadata_mode="none",
        positions=result_positions,
        form="table",
        include_position=True,
    )
    by_position = {
        int(row["_position"]): row
        for row in key_frame.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    result_row_names = (
        artifact.get_row_names(positions=result_positions)
        if row_name_column is not None
        else [None] * len(result_positions)
    )
    for rank, ((distance, result_position), result_row_name) in enumerate(
        zip(ordered, result_row_names, strict=True), start=1
    ):
        source = by_position[int(result_position)]
        row = {name: source[name] for name in primary_key}
        if row_name_column is not None and row_name_column not in row:
            row[str(row_name_column)] = result_row_name
        row["_position"] = int(result_position)
        row["rank"] = int(rank)
        row["distance"] = float(distance)
        rows.append(row)

    result = pd.DataFrame(rows, columns=output_columns)
    if context_artifact is not None and not result.empty:
        result = _append_context(
            result,
            neighbor_artifact=artifact,
            context_artifact=context_artifact,
            data_columns=context_data_columns,
            metadata_columns=context_metadata_columns,
            metadata_mode=context_metadata_mode,
            context_columns=context_columns,
        )
    return result


def _validate_context_keys(neighbor_artifact: "BaseArtifact", context_artifact: "BaseArtifact") -> None:
    neighbor_pk = tuple(str(v) for v in neighbor_artifact.primary_key)
    context_pk = tuple(str(v) for v in context_artifact.primary_key)
    if not is_prefix(context_pk, neighbor_pk):
        raise ArtifactError(
            "nearest-neighbor context primary key must equal or be a leading prefix "
            f"of the neighbor artifact primary key; context={list(context_pk)}, "
            f"neighbors={list(neighbor_pk)}."
        )


def _selected_context_output_columns(
    context_artifact: "BaseArtifact",
    *,
    data_columns: Any,
    metadata_columns: Any,
    metadata_mode: str,
) -> list[str]:
    if metadata_columns is not False and metadata_mode == "none":
        raise ArtifactError(
            "context_metadata_columns requires context_metadata_mode='local' or 'full'."
        )
    info = context_artifact.query_columns(metadata_mode=metadata_mode)
    selected: list[str] = []
    for namespace, request in (("data", data_columns), ("metadata", metadata_columns)):
        if request is False:
            continue
        cols = [c for c in info.get("columns", []) if c.get("namespace") == namespace]
        if request is True:
            names = [str(c["output_name"]) for c in cols]
        else:
            requested = [request] if isinstance(request, str) else list(request)
            names = []
            for raw in requested:
                name = str(raw)
                exact = [c for c in cols if name in {str(c.get("qualified_name")), str(c.get("output_name"))}]
                base = [c for c in cols if str(c.get("base_name")) == name]
                matches = exact if exact else base
                if len(matches) != 1:
                    if not matches:
                        raise ArtifactError(
                            f"Unknown context {namespace} column {name!r}."
                        )
                    raise ArtifactError(
                        f"Ambiguous context {namespace} column {name!r}; use a qualified name."
                    )
                names.append(str(matches[0]["output_name"]))
        for name in names:
            if name not in selected:
                selected.append(name)
    return selected


def _append_context(
    result: pd.DataFrame,
    *,
    neighbor_artifact: "BaseArtifact",
    context_artifact: "BaseArtifact",
    data_columns: Any,
    metadata_columns: Any,
    metadata_mode: str,
    context_columns: list[str],
) -> pd.DataFrame:
    if not context_columns:
        return result
    context_pk = [str(v) for v in context_artifact.primary_key]
    unique_keys: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for row in result.loc[:, context_pk].itertuples(index=False, name=None):
        key = tuple(int(v) for v in row)
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)
    positions = [
        context_artifact.position_by_key(key[0] if len(key) == 1 else key)
        for key in unique_keys
    ]
    frame = context_artifact.query(
        key_columns=True,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode=metadata_mode,
        positions=positions,
        form="table",
        include_position=False,
    )
    keep = [*context_pk, *context_columns]
    frame = frame.loc[:, keep].copy()
    return result.merge(frame, on=context_pk, how="left", sort=False, validate="many_to_one")
