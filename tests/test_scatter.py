from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.visualization.scatter import _format_hover_value, scatter


class FakeQueryEngine:
    def __init__(self, positions_by_key: dict[int, int]) -> None:
        self.positions_by_key = positions_by_key
        self.calls: list[tuple[Any, list[dict[str, int]]]] = []

    def positions_by_keys(self, artifact: Any, keys: list[dict[str, int]]) -> list[int]:
        self.calls.append((artifact, keys))
        try:
            return [self.positions_by_key[key["doc_id"]] for key in keys]
        except KeyError as exc:
            raise KeyError("One or more keys were not found.") from exc


@dataclass
class FakeProject:
    query: FakeQueryEngine
    identity: str = "project"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeProject) and self.identity == other.identity


class FakeDocuments:
    primary_key: ClassVar[list[str]] = ["doc_id"]

    def __init__(self, project: FakeProject) -> None:
        self.project = project
        self.calls: list[dict[str, Any]] = []
        self.rows = pd.DataFrame(
            {
                "doc_id": [10, 11, 12, 13],
                "text": ["alpha", "beta", "gamma", "delta"],
                "group": ["a", "b", "a", "b"],
                "_position": [0, 1, 2, 3],
            }
        )

    def query(self, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(dict(kwargs))
        if kwargs.get("data_columns") is False and kwargs.get("key_columns") is True:
            positions = kwargs["positions"]
            return self.rows.iloc[positions][["doc_id"]].reset_index(drop=True)

        frame = self.rows.copy()
        positions = kwargs.get("positions")
        if positions is not None:
            frame = frame.iloc[list(positions)]
        sample_n = kwargs.get("sample_n")
        if sample_n is not None:
            frame = frame.sample(
                n=min(sample_n, len(frame)),
                random_state=kwargs.get("random_state"),
            )
        sample_frac = kwargs.get("sample_frac")
        if sample_frac is not None:
            frame = frame.sample(
                frac=sample_frac, random_state=kwargs.get("random_state")
            )
        limit = kwargs.get("limit")
        if limit is not None:
            frame = frame.head(limit)

        selected: list[str] = []
        if kwargs.get("key_columns") is True:
            selected.append("doc_id")
        elif isinstance(kwargs.get("key_columns"), str):
            selected.append(kwargs["key_columns"])
        elif (
            kwargs.get("key_columns") is not False
            and kwargs.get("key_columns") is not None
        ):
            selected.extend(kwargs["key_columns"])

        if kwargs.get("data_columns") is True:
            selected.append("text")
        elif isinstance(kwargs.get("data_columns"), str):
            selected.append(kwargs["data_columns"])
        elif (
            kwargs.get("data_columns") is not False
            and kwargs.get("data_columns") is not None
        ):
            selected.extend(kwargs["data_columns"])

        metadata = kwargs.get("metadata_columns")
        if metadata is True:
            selected.append("group")
        elif isinstance(metadata, str):
            selected.append(metadata)
        elif metadata is not False and metadata is not None:
            selected.extend(metadata)

        if kwargs.get("include_position"):
            selected.append("_position")
        return frame[selected].reset_index(drop=True)

    def query_columns(self, *, metadata_mode: str = "none") -> dict[str, Any]:
        columns = [
            {
                "namespace": "key",
                "base_name": "doc_id",
                "qualified_name": "key.doc_id",
                "output_name": "doc_id",
            },
            {
                "namespace": "data",
                "base_name": "text",
                "qualified_name": "data.text",
                "output_name": "text",
            },
        ]
        if metadata_mode != "none":
            columns.append(
                {
                    "namespace": "metadata",
                    "base_name": "group",
                    "qualified_name": "metadata.group",
                    "output_name": "group",
                }
            )
        return {"columns": columns}


class FakeEmbeddings:
    primary_key: ClassVar[list[str]] = ["doc_id"]

    def __init__(
        self, project: FakeProject, *, columns: list[str] | None = None
    ) -> None:
        self.project = project
        self.columns = columns or ["umap_0", "umap_1"]
        self.calls: list[dict[str, Any]] = []
        self.values = pd.DataFrame(
            {
                "umap_0": [0.1, 0.2, 0.3, 0.4],
                "umap_1": [1.1, 1.2, 1.3, 1.4],
            }
        )

    def get_data_columns(self) -> list[str]:
        return list(self.columns)

    def query(self, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(dict(kwargs))
        positions = kwargs["positions"]
        return self.values.iloc[positions][self.columns].reset_index(drop=True)


def test_scatter_samples_documents_before_fetching_coordinates() -> None:
    engine = FakeQueryEngine({10: 0, 11: 1, 12: 2, 13: 3})
    project = FakeProject(engine)
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project)

    figure = scatter(
        docs,
        embeddings=embeddings,
        color="group",
        hover_fields=["doc_id", "text"],
        key_columns=True,
        data_columns=["text"],
        metadata_columns=["group"],
        metadata_mode="full",
        sample_n=2,
        random_state=7,
        limit=2,
    )

    first = docs.calls[0]
    assert first["sample_n"] == 2
    assert first["random_state"] == 7
    assert first["limit"] == 2
    assert first["metadata_columns"] == ["group"]
    assert first["metadata_mode"] == "full"

    selected_keys = engine.calls[0][1]
    assert len(selected_keys) == 2
    assert embeddings.calls[0]["positions"] == [
        engine.positions_by_key[key["doc_id"]] for key in selected_keys
    ]
    assert sum(len(trace.x) for trace in figure.data) == 2


def test_scatter_requires_color_and_hover_fields_to_be_selected() -> None:
    project = FakeProject(FakeQueryEngine({10: 0, 11: 1, 12: 2, 13: 3}))
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project)

    with pytest.raises(ArtifactError, match="was not selected"):
        scatter(
            docs,
            embeddings=embeddings,
            color="group",
            metadata_columns=False,
            metadata_mode="none",
            sample_n=1,
            random_state=1,
        )


def test_scatter_requires_exactly_two_coordinate_columns() -> None:
    project = FakeProject(FakeQueryEngine({10: 0, 11: 1, 12: 2, 13: 3}))
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project, columns=["umap_0"])

    with pytest.raises(ArtifactError, match="exactly two"):
        scatter(docs, embeddings=embeddings, sample_n=1, random_state=1)


def test_hover_formatting_truncates_wraps_and_escapes() -> None:
    assert (
        _format_hover_value("<abc def ghi>", max_length=10, wrap_length=5)
        == "&lt;abc<br>def…"
    )


def test_scatter_errors_when_selected_key_is_missing_from_embeddings() -> None:
    project = FakeProject(FakeQueryEngine({10: 0}))
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project)

    with pytest.raises(ArtifactError, match="does not contain coordinates"):
        scatter(docs, embeddings=embeddings, positions=[1])


def test_scatter_display_defaults_are_stable() -> None:
    params = signature(scatter).parameters
    assert params["max_length"].default == 600
    assert params["wrap_length"].default == 200
    assert params["width"].default is None
    assert params["height"].default == 700


def test_scatter_uses_roomy_layout_and_hover_defaults() -> None:
    project = FakeProject(FakeQueryEngine({10: 0, 11: 1, 12: 2, 13: 3}))
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project)

    figure = scatter(
        docs,
        embeddings=embeddings,
        hover_fields=["text"],
        positions=[0],
    )

    assert figure.layout.width is None
    assert figure.layout.autosize is True
    assert figure.layout.height == 700
    hover_text = str(figure.data[0].customdata[0][0])
    assert "alpha" in hover_text


def test_scatter_layout_dimensions_can_be_overridden_or_unset() -> None:
    project = FakeProject(FakeQueryEngine({10: 0, 11: 1, 12: 2, 13: 3}))
    docs = FakeDocuments(project)
    embeddings = FakeEmbeddings(project)

    figure = scatter(
        docs,
        embeddings=embeddings,
        positions=[0],
        width=1200,
        height=800,
    )
    assert figure.layout.width == 1200
    assert figure.layout.height == 800

    responsive = scatter(
        docs,
        embeddings=embeddings,
        positions=[0],
        width=None,
        height=None,
    )
    assert responsive.layout.width is None
    assert responsive.layout.autosize is True
    assert responsive.layout.height is None

    with pytest.raises(TypeError, match="width must be"):
        scatter(
            docs,
            embeddings=embeddings,
            positions=[0],
            width="80%",  # type: ignore[arg-type]
        )
