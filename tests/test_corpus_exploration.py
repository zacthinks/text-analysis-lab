from __future__ import annotations

import pandas as pd

from text_analysis_lab.analysis.accessor import ArtifactAnalysis
from text_analysis_lab.visualization.accessor import ArtifactVisualization


class _FakeTabularArtifact:
    def __init__(self) -> None:
        self.batches = [
            pd.DataFrame(
                {
                    "text_words": [2, 4, 6],
                    "record_type": ["paper", "paper", "session"],
                    "session_type": ["panel", "panel", "roundtable"],
                }
            ),
            pd.DataFrame(
                {
                    "text_words": [8, 10, None],
                    "record_type": ["session", "paper", None],
                    "session_type": ["panel", "roundtable", "panel"],
                }
            ),
        ]
        self.batch_calls = 0

    def query_columns(self, *, metadata_mode="none"):
        assert metadata_mode == "full"
        names = ["text_words", "record_type", "session_type"]
        return {
            "columns": [
                {
                    "namespace": "metadata",
                    "base_name": name,
                    "qualified_name": f"metadata.{name}",
                    "output_name": name,
                }
                for name in names
            ]
        }

    def iter_table_batches(self, **kwargs):
        self.batch_calls += 1
        assert kwargs["key_columns"] is False
        assert kwargs["data_columns"] is False
        assert kwargs["metadata_mode"] == "full"
        selected = kwargs["metadata_columns"]
        for batch in self.batches:
            yield batch.loc[:, selected].copy()


def test_crosstab_counts_in_batches_and_supports_margins() -> None:
    artifact = _FakeTabularArtifact()
    result = ArtifactAnalysis(artifact).crosstab(
        "record_type", "session_type", margins=True, batch_size=2
    )
    assert result.loc["paper", "panel"] == 2
    assert result.loc["paper", "roundtable"] == 1
    assert result.loc["session", "panel"] == 1
    assert result.loc["session", "roundtable"] == 1
    assert result.loc["All", "All"] == 5
    assert artifact.batch_calls == 1


def test_histogram_returns_compact_grouped_data_without_concatenating_rows() -> None:
    artifact = _FakeTabularArtifact()
    result = ArtifactVisualization(artifact).histogram(
        "text_words",
        by="record_type",
        bins=[0, 5, 10],
        output="result",
        batch_size=2,
    )
    compact = result.to_frame()
    paper = compact.loc[compact["record_type"] == "paper", "count"].tolist()
    session = compact.loc[compact["record_type"] == "session", "count"].tolist()
    assert paper == [2, 1]
    assert session == [0, 2]
    assert result.observed_count == 5
    assert result.missing_count == 1
    assert artifact.batch_calls == 1


def test_integer_bin_histogram_uses_bounded_two_pass_aggregation() -> None:
    artifact = _FakeTabularArtifact()
    compact = ArtifactVisualization(artifact).histogram(
        "text_words", bins=2, output="data", batch_size=2
    )
    assert compact["count"].sum() == 5
    assert len(compact) == 2
    assert artifact.batch_calls == 2
