from __future__ import annotations

import pandas as pd

from text_analysis_lab.analysis.text_diagnostics import text_diagnostics


class _Project:
    def __init__(self):
        self.artifacts = {}

    def get_artifact(self, value):
        return self.artifacts[value] if isinstance(value, str) else value


class _Artifact:
    def __init__(self, project, artifact_id, frame):
        self.project = project
        self.artifact_id = artifact_id
        self.primary_key = ["row_id"]
        self.frame = frame.copy()
        project.artifacts[artifact_id] = self

    def query(self, **kwargs):
        data = kwargs.get("data_columns")
        cols = ["row_id"]
        if data is not False:
            cols.extend(list(data))
        result = self.frame.loc[:, cols].copy()
        if kwargs.get("include_position"):
            result["_position"] = range(len(result))
        return result


def test_text_diagnostics_separates_empty_and_duplicate_nonempty_texts() -> None:
    project = _Project()
    artifact = _Artifact(
        project,
        "cleaned",
        pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3, 4, 5],
                "text": ["alpha", " alpha ", "", "   ", None, "beta"],
            }
        ),
    )
    result = text_diagnostics(artifact)
    assert result.row_count == 6
    assert result.missing_text_count == 1
    assert result.empty_text_count == 2
    assert result.nonempty_text_count == 3
    assert result.unique_nonempty_text_count == 2
    assert result.duplicate_group_count == 1
    assert result.duplicate_row_count == 2
    assert result.duplicate_excess_count == 1


def test_text_diagnostics_reports_key_set_and_order_preservation() -> None:
    project = _Project()
    before = _Artifact(
        project,
        "before",
        pd.DataFrame({"row_id": [0, 1, 2], "text": ["a", "b", "c"]}),
    )
    reordered = _Artifact(
        project,
        "after",
        pd.DataFrame({"row_id": [2, 0, 1], "text": ["c", "a", "b"]}),
    )
    result = text_diagnostics(reordered, compare_to=before)
    assert result.key_set_preserved is True
    assert result.key_order_preserved is False
    assert result.missing_keys_from_current == 0
    assert result.extra_keys_in_current == 0
