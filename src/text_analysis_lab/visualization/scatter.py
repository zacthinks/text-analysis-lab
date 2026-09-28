"""Interactive keyed scatter plots for TeAL artifacts."""

from __future__ import annotations

import html
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, MissingDependencyError
from text_analysis_lab.core.types import ColumnSelect, MetadataMode

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_INTERNAL_X = "__teal_scatter_x"
_INTERNAL_Y = "__teal_scatter_y"
_INTERNAL_HOVER_PREFIX = "__teal_scatter_hover_"


def scatter(
    artifact: BaseArtifact,
    *,
    embeddings: BaseArtifact,
    color: str | None = None,
    hover_fields: str | Sequence[str] | None = None,
    include_coordinates: bool = False,
    key_columns: ColumnSelect = True,
    data_columns: ColumnSelect = True,
    metadata_columns: ColumnSelect = False,
    metadata_mode: MetadataMode = "none",
    where: str | None = None,
    order_by: str | Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    sample_n: int | None = None,
    sample_frac: float | None = None,
    random_state: int | None = None,
    limit: int | None = None,
    max_length: int | None = 600,
    wrap_length: int | None = 200,
    width: int | None = None,
    height: int | None = 700,
) -> Any:
    """Plot artifact rows interactively using a keyed two-dimensional embedding.

    The calling ``artifact`` owns the points and all display fields. ``embeddings``
    supplies only the two-dimensional coordinates. Document selection follows the
    normal :meth:`BaseArtifact.query` vocabulary, including column selection,
    filtering, sampling, ordering, positions, and limits. The selected document
    keys are then used to fetch only the matching embedding rows.

    ``color`` and ``hover_fields`` must refer to fields that were selected by
    ``key_columns``, ``data_columns``, and ``metadata_columns``. Metadata therefore
    remains opt-in: request an appropriate ``metadata_mode`` and
    ``metadata_columns`` before using metadata in the plot. The color field and
    projection coordinates are hidden from hover by default; set
    ``include_coordinates=True`` to include the two projection coordinates, and
    include the color field explicitly in ``hover_fields`` if you want it shown.

    Plotly is imported lazily only when this function is called.
    """
    _validate_text_length("max_length", max_length)
    _validate_text_length("wrap_length", wrap_length)
    _validate_width(width)
    _validate_dimension("height", height)
    if artifact.project != embeddings.project:
        raise ArtifactError(
            "scatter() requires the document artifact and embeddings artifact "
            "to belong to the same TeAL project."
        )

    document_key_columns = [str(column) for column in artifact.primary_key]
    embedding_key_columns = [str(column) for column in embeddings.primary_key]
    if not document_key_columns:
        raise ArtifactError(
            "scatter() requires the document artifact to have a primary key."
        )
    if embedding_key_columns != document_key_columns:
        raise ArtifactError(
            "scatter() requires embeddings with the same primary key as the "
            f"document artifact. Documents use {document_key_columns}; embeddings "
            f"use {embedding_key_columns}."
        )

    coordinate_columns = [str(column) for column in embeddings.get_data_columns()]
    if len(coordinate_columns) != 2:
        raise ArtifactError(
            "scatter() requires an embeddings artifact with exactly two data "
            f"columns; found {len(coordinate_columns)}: {coordinate_columns}."
        )

    # Let the ordinary document query own all selection semantics. _position is
    # requested only as an internal handle so a second, tiny key-only query can
    # recover join keys even when key_columns=False for the visible plot fields.
    display = artifact.query(
        key_columns=key_columns,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode=metadata_mode,
        where=where,
        order_by=order_by,
        positions=positions,
        sample_n=sample_n,
        sample_frac=sample_frac,
        random_state=random_state,
        limit=limit,
        form="table",
        include_position=True,
    )
    if not isinstance(display, pd.DataFrame):  # pragma: no cover - query contract
        display = pd.DataFrame(display)
    if "_position" not in display.columns:
        raise ArtifactError("scatter() could not recover selected document positions.")

    selected_positions = [int(value) for value in display["_position"].tolist()]
    display = display.drop(columns=["_position"]).reset_index(drop=True)
    selected_columns = list(display.columns)

    resolved_color = None
    if color is not None:
        resolved_color = _resolve_selected_field(
            artifact,
            str(color),
            selected_columns=selected_columns,
            metadata_mode=metadata_mode,
        )
    resolved_hover = [
        _resolve_selected_field(
            artifact,
            field,
            selected_columns=selected_columns,
            metadata_mode=metadata_mode,
        )
        for field in _normalize_hover_fields(hover_fields)
    ]

    if selected_positions:
        key_frame = artifact.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            metadata_mode="none",
            positions=selected_positions,
            form="table",
            include_position=False,
        )
        keys = [
            {column: int(record[column]) for column in document_key_columns}
            for record in key_frame.to_dict(orient="records")
        ]
        try:
            embedding_positions = artifact.project.query.positions_by_keys(
                embeddings,
                keys,
            )
        except KeyError as exc:
            raise ArtifactError(
                "The embeddings artifact does not contain coordinates for one or "
                "more selected document keys."
            ) from exc

        coordinates = embeddings.query(
            key_columns=False,
            data_columns=coordinate_columns,
            metadata_columns=False,
            metadata_mode="none",
            positions=embedding_positions,
            form="table",
            include_position=False,
        )
        if len(coordinates) != len(display):
            raise ArtifactError(
                "The embeddings artifact returned a different number of coordinate "
                "rows than the selected document rows."
            )
    else:
        coordinates = pd.DataFrame(columns=coordinate_columns)

    plot_frame = display.copy()
    try:
        plot_frame[_INTERNAL_X] = pd.to_numeric(
            coordinates[coordinate_columns[0]], errors="raise"
        ).to_numpy()
        plot_frame[_INTERNAL_Y] = pd.to_numeric(
            coordinates[coordinate_columns[1]], errors="raise"
        ).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ArtifactError("scatter() embedding coordinates must be numeric.") from exc

    hover_data: dict[str, bool] = {
        _INTERNAL_X: include_coordinates,
        _INTERNAL_Y: include_coordinates,
    }
    if resolved_color is not None:
        hover_data[resolved_color] = False

    labels = {
        _INTERNAL_X: coordinate_columns[0],
        _INTERNAL_Y: coordinate_columns[1],
    }
    for index, field in enumerate(resolved_hover):
        shadow = f"{_INTERNAL_HOVER_PREFIX}{index}"
        plot_frame[shadow] = plot_frame[field].map(
            lambda value: _format_hover_value(
                value,
                max_length=max_length,
                wrap_length=wrap_length,
            )
        )
        hover_data[shadow] = True
        labels[shadow] = field

    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MissingDependencyError(
            "Interactive scatter plots require Plotly. Install it with 'pip install plotly'."
        ) from exc

    scatter_kwargs: dict[str, Any] = {
        "x": _INTERNAL_X,
        "y": _INTERNAL_Y,
        "color": resolved_color,
        "hover_data": hover_data,
        "labels": labels,
        "render_mode": "webgl",
        "height": height,
    }
    if width is not None:
        scatter_kwargs["width"] = width

    figure = px.scatter(plot_frame, **scatter_kwargs)
    if width is None:
        figure.update_layout(width=None, autosize=True)
    return figure


def _normalize_hover_fields(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(field) for field in value]


def _validate_text_length(name: str, value: int | None) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer or None.")


def _validate_width(value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError("width must be a positive integer or None.")
    if value is not None and value <= 0:
        raise ValueError("width must be a positive integer or None.")


def _validate_dimension(name: str, value: int | None) -> None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer or None.")


def _resolve_selected_field(
    artifact: BaseArtifact,
    field: str,
    *,
    selected_columns: Sequence[str],
    metadata_mode: MetadataMode,
) -> str:
    """Resolve a key/data/metadata name, but only among selected output columns."""
    name = str(field)
    if not name:
        raise ValueError("Scatter field names must be non-empty strings.")
    selected = {str(column) for column in selected_columns}
    if name in selected:
        return name

    info = artifact.query_columns(metadata_mode=metadata_mode)
    candidates = [
        column
        for column in info["columns"]
        if str(column["namespace"]) in {"key", "data", "metadata"}
        and str(column["output_name"]) in selected
        and name
        in {
            str(column["base_name"]),
            str(column["qualified_name"]),
            str(column["output_name"]),
        }
    ]
    output_names = {str(column["output_name"]) for column in candidates}
    if len(output_names) == 1:
        return next(iter(output_names))
    if len(output_names) > 1:
        options = sorted(str(column["qualified_name"]) for column in candidates)
        raise ArtifactError(
            f"Scatter field {name!r} is ambiguous among the selected columns. "
            f"Use one of: {options}."
        )

    raise ArtifactError(
        f"Scatter field {name!r} was not selected. Choose it with key_columns, "
        "data_columns, or metadata_columns (and metadata_mode for metadata) "
        f"before using it as color or hover data. Selected columns: {sorted(selected)}."
    )


def _format_hover_value(
    value: Any,
    *,
    max_length: int | None,
    wrap_length: int | None,
) -> str:
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    if max_length is not None and len(text) > int(max_length):
        if int(max_length) == 1:
            text = "…"
        else:
            text = text[: int(max_length) - 1].rstrip() + "…"
    if wrap_length is not None:
        lines = text.splitlines() or [""]
        text = "\n".join(
            line
            for source_line in lines
            for line in (
                textwrap.wrap(
                    source_line,
                    width=int(wrap_length),
                    replace_whitespace=False,
                    drop_whitespace=True,
                )
                or [""]
            )
        )
    # Plotly hover labels render a subset of HTML. Escape document content first,
    # then convert our own line breaks so user text cannot inject markup.
    return html.escape(text).replace("\n", "<br>")
