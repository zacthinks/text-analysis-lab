from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    CountVectorizer,
    FeatureTrimmer,
    MatrixTranspose,
)


def _build_count_matrix(tmp_path: Path):
    csv_path = tmp_path / "docs.csv"
    pd.DataFrame(
        {
            "text": [
                "alpha beta",
                "beta gamma",
                "alpha gamma gamma",
                "delta gamma",
            ]
        }
    ).to_csv(csv_path, index=False)
    project = teal.Project.create(tmp_path / "project", name="feature_subset")
    docs = project.read_csv(
        csv_path,
        text_fields="text",
        metadata_fields=None,
        output_label="docs",
    )
    counts = project.translate(
        CountVectorizer(text_field="text"),
        docs,
    )["output"]
    counts.add_alias("counts")
    return project, counts


def test_feature_subset_is_lazy_positional_view_and_chains(tmp_path: Path) -> None:
    project, counts = _build_count_matrix(tmp_path)
    try:
        source_columns = counts.get_data_columns()
        assert source_columns == ["alpha", "beta", "delta", "gamma"]
        source = counts.get_matrix().toarray()

        selected = project.feature_subset(
            counts,
            lambda features: features["column"].isin(["alpha", "gamma"]),
            output_label="alpha_gamma",
        )
        assert selected.components.get("data") is None
        assert selected.descriptor["lineage"]["feature_indices"] == [0, 3]
        assert selected.get_data_columns() == ["alpha", "gamma"]
        assert selected.get_feature_frame()["column_index"].tolist() == [0, 1]
        np.testing.assert_array_equal(
            selected.get_matrix().toarray(),
            source[:, [0, 3]],
        )

        gamma = project.feature_subset(
            selected,
            lambda features: features["column"].eq("gamma"),
            output_label="gamma_only",
        )
        assert gamma.components.get("data") is None
        assert gamma.descriptor["lineage"]["feature_indices"] == [1]
        assert gamma.get_data_columns() == ["gamma"]
        np.testing.assert_array_equal(
            gamma.get_matrix().toarray(),
            source[:, [3]],
        )
    finally:
        project.close()


def test_feature_trimmer_outputs_lazy_feature_view_and_replays_by_width(
    tmp_path: Path,
) -> None:
    project, counts = _build_count_matrix(tmp_path)
    try:
        trimmer = FeatureTrimmer(min_df=2)
        trimmed = project.translate(trimmer, counts)["output"]

        # alpha, beta, and gamma occur in >=2 documents; delta occurs once.
        assert trimmer.kept_indices_ == (0, 1, 3)
        assert trimmed.components.get("data") is None
        assert trimmed.descriptor["lineage"]["feature_indices"] == [0, 1, 3]
        assert trimmed.get_data_columns() == ["alpha", "beta", "gamma"]
        np.testing.assert_array_equal(
            trimmed.get_matrix().toarray(),
            counts.get_matrix().toarray()[:, [0, 1, 3]],
        )

        operator_id = str(trimmer.operator_id)
    finally:
        project.close()

    reopened = teal.Project.open(tmp_path / "project")
    try:
        restored = reopened.get_operator(operator_id)
        assert isinstance(restored, FeatureTrimmer)
        assert restored.source_width_ == 4
        assert restored.kept_indices_ == (0, 1, 3)

        replayed = reopened.translate(restored, reopened.get_artifact("counts"))[
            "output"
        ]
        assert replayed.components.get("data") is None
        assert replayed.get_data_columns() == ["alpha", "beta", "gamma"]
    finally:
        reopened.close()


def test_transpose_promotes_projected_feature_frame_to_metadata(tmp_path: Path) -> None:
    project, counts = _build_count_matrix(tmp_path)
    try:
        feature_frame = counts.get_feature_frame()
        feature_frame["ngram_n"] = [1, 1, 2, 2]
        feature_frame["group"] = ["left", "left", "right", "right"]
        feature_frame.to_parquet(counts.storage.data_columns_path, index=False)

        selected = project.feature_subset(
            counts,
            lambda features: features["column"].isin(["alpha", "gamma"]),
            output_label="alpha_gamma_for_transpose",
        )
        assert selected.get_feature_frame()["column"].tolist() == ["alpha", "gamma"]
        assert selected.get_feature_frame()["ngram_n"].tolist() == [1, 2]

        transposed = project.translate(MatrixTranspose(), selected)["output"]
        assert transposed.primary_key == ["feature_id"]
        assert transposed.get_row_names() == ["alpha", "gamma"]
        metadata = transposed.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=True,
            metadata_mode="local",
        )
        assert metadata["feature_id"].tolist() == [0, 1]
        assert metadata["column"].tolist() == ["alpha", "gamma"]
        assert metadata["ngram_n"].tolist() == [1, 2]
        assert metadata["group"].tolist() == ["left", "right"]
        assert "column_index" not in metadata.columns
    finally:
        project.close()
