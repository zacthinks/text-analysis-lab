from __future__ import annotations

import text_analysis_lab as teal
from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.types import ArtifactType


class _DisplayArtifact(BaseArtifact):
    artifact_type = ArtifactType.SPARSE_MATRIX

    def __init__(self):
        pass

    artifact_id = property(lambda self: "art_000123")
    label = property(lambda self: "output")
    aliases = property(lambda self: ["dtm_trimmed"])
    status = property(lambda self: "complete")
    primary_key = property(lambda self: ["row_id"])
    n_rows = property(lambda self: 9560)
    descriptor = {
        "lineage": {
            "lineage_mode": "preserved_key",
            "basis_artifact_ids": ["art_000122"],
        }
    }

    def get_data_columns(self):
        return ["a", "b", "c"]


def test_artifact_str_repr_and_html_are_informative() -> None:
    artifact = _DisplayArtifact()
    assert str(artifact) == "dtm_trimmed [sparse_matrix: 9,560 x 3]"
    rendered = repr(artifact)
    assert "dtm_trimmed" in rendered
    assert "art_000123" in rendered
    assert "rows=9560" in rendered
    assert "preserved_key" in rendered
    html = artifact._repr_html_()
    assert "dtm_trimmed" in html
    assert "9,560" in html
    assert "sparse_matrix" in html


def test_project_str_repr_and_html_are_informative(tmp_path) -> None:
    project = teal.Project.create(tmp_path / "display", name="display")
    try:
        assert "display [TeAL project: open" in str(project)
        assert "artifacts=0" in repr(project)
        assert "display" in project._repr_html_()
    finally:
        project.close()
    assert "closed" in str(project)
    assert "closed" in repr(project)
