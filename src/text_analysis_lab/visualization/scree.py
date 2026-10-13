"""Artifact-independent scree visualization for component spectra."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import MissingDependencyError


def scree(
    spectrum: pd.DataFrame,
    *,
    measure: Literal["energy", "explained_variance"] = "energy",
    title: str | None = None,
    width: int | None = None,
    height: int | None = 550,
) -> Any:
    """Return a Plotly figure with individual component bars and a cumulative line.

    Accepts any DataFrame following the component/ratio column contract, not
    necessarily an SVD artifact or even a TeAL-derived result.
    """
    if measure not in {"energy", "explained_variance"}:
        raise ValueError("measure must be 'energy' or 'explained_variance'.")
    if not isinstance(spectrum, pd.DataFrame):
        raise TypeError("scree() requires a pandas DataFrame.")
    individual = "energy_ratio" if measure == "energy" else "explained_variance_ratio"
    cumulative = (
        "cumulative_energy_ratio" if measure == "energy"
        else "cumulative_explained_variance_ratio"
    )
    missing = {"component", individual, cumulative} - set(spectrum.columns)
    if missing:
        raise ValueError(f"scree() is missing required columns: {sorted(missing)}.")
    if spectrum.empty:
        raise ValueError("scree() requires at least one component.")
    components = pd.to_numeric(spectrum["component"], errors="raise").to_numpy(dtype=float)
    ratios = pd.to_numeric(spectrum[individual], errors="raise").to_numpy(dtype=float)
    running = pd.to_numeric(spectrum[cumulative], errors="raise").to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(components))
        or not np.array_equal(components, np.arange(1, len(components) + 1))
    ):
        raise ValueError("component must contain consecutive 1-based integers.")
    if (
        not np.all(np.isfinite(ratios))
        or not np.all(np.isfinite(running))
        or np.any(ratios < -1e-10)
        or np.any(running < -1e-10)
        or np.any(ratios > 1 + 1e-10)
        or np.any(running > 1 + 1e-10)
        or not np.allclose(np.cumsum(ratios), running, atol=1e-7)
    ):
        raise ValueError("scree() requires consistent finite component and cumulative ratios.")
    for name, value in (("width", width), ("height", height)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None.")
    try:
        from plotly.graph_objects import Bar, Figure, Scatter
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "Scree plots require Plotly. Install it with 'pip install text-analysis-lab[visualization]'."
        ) from exc

    label = "Energy" if measure == "energy" else "Explained variance"
    figure = Figure()
    figure.add_trace(
        Bar(
            x=components.astype(int),
            y=ratios,
            name=f"Component {label.lower()}",
            marker_color="#7799be",
            hovertemplate="Component %{x}<br>Individual: %{y:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        Scatter(
            x=components.astype(int),
            y=running,
            name=f"Cumulative {label.lower()}",
            mode="lines+markers",
            line={"color": "#c43b43", "width": 2.5},
            marker={"color": "#c43b43", "size": 5},
            yaxis="y2",
            hovertemplate="Component %{x}<br>Cumulative: %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title or f"Scree plot — {label.lower()}",
        xaxis={"title": "Component", "dtick": 1 if len(components) <= 20 else None},
        yaxis={"title": f"Individual {label.lower()}", "tickformat": ".0%", "rangemode": "tozero"},
        yaxis2={
            "title": f"Cumulative {label.lower()}",
            "tickformat": ".0%",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1],
        },
        legend={"orientation": "h", "y": 1.12},
        margin={"t": 105, "r": 85, "b": 65, "l": 85},
        template="plotly_white",
        bargap=0.2,
        width=width,
        height=height,
        autosize=width is None,
    )
    return figure
