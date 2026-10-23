"""Persisted scope invariance for public analytic field consumers."""

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.translators import FeatureMatrix


def _source(project):
    return project.register_external([{
        "keys": pd.DataFrame({"doc_id": list(range(8))}),
        "data": pd.DataFrame({
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "score": [10, 20, 30, 40, 50, 60, 70, 80],
        }),
        "metadata": pd.DataFrame({
            "label": [100] * 8,
            "score": [999] * 8,
        }),
    }], artifact_type="table", primary_key="doc_id")


def _label(source, namespace, name):
    return next(c["output_name"] for c in source.query_columns(metadata_mode="full")["columns"]
                if c["namespace"] == namespace and c["base_name"] == name)


def test_feature_matrix_frozen_numeric_field_identity(tmp_path):
    project = teal.Project.create(tmp_path / "fm", name="fm")
    try:
        source = _source(project)
        score = _label(source, "data", "score")
        meta = _label(source, "metadata", "score")
        for selections, expected in [
            ({"data_columns": [score], "metadata_columns": False},
             np.array([[10], [20], [30], [40], [50], [60], [70], [80]])),
            ({"data_columns": [score], "metadata_columns": [meta]},
             np.array([[10, 999], [20, 999], [30, 999], [40, 999],
                       [50, 999], [60, 999], [70, 999], [80, 999]])),
        ]:
            recipe = FeatureMatrix(**selections, matrix_type="dense")
            result = project.translate(recipe, source)["output"]
            np.testing.assert_array_equal(result.get_matrix(), expected)
            restored = project.get_operator(recipe.operator_id)
            assert restored.scalar_data_columns == [score]
    finally:
        project.close()


def test_split_data_stratification_preserves_colliding_label(tmp_path):
    project = teal.Project.create(tmp_path / "split", name="split")
    try:
        source = _source(project)
        label = _label(source, "data", "label")
        children = project.split(
            source, stratify=label, labels=("train", "test"),
            proportions=(0.5, 0.5), random_state=3,
        )
        membership = {}
        for group, child in children.items():
            rows = child.query(key_columns=True, data_columns=False, form="table")
            membership[group] = rows["doc_id"].tolist()
        assert sorted(membership["train"] + membership["test"]) == list(range(8))
        # Each real data stratum contributes two observations to each half.
        for values in membership.values():
            assert sum(v % 2 == 0 for v in values) == 2
            assert sum(v % 2 == 1 for v in values) == 2
    finally:
        project.close()


@pytest.mark.parametrize("metadata_mode", ["none", "local", "full"])
def test_subset_validates_in_requested_metadata_scope(tmp_path, metadata_mode):
    project = teal.Project.create(tmp_path / metadata_mode, name="subset_" + metadata_mode)
    try:
        source = _source(project)
        inventory = source.query_columns(metadata_mode=metadata_mode)
        data_score = next(c["output_name"] for c in inventory["columns"]
                          if c["namespace"] == "data" and c["base_name"] == "score")
        result = project.subset(
            source, lambda frame: frame[data_score] >= 50,
            data_columns=[data_score], metadata_columns=False,
            metadata_mode=metadata_mode, workers=1, batch_size=3,
        )["output"]
        selected = result.query(data_columns=False, form="table")
        assert selected["doc_id"].tolist() == [4, 5, 6, 7]
        if metadata_mode == "none":
            metadata_label = _label(source, "metadata", "score")
            with pytest.raises((ValueError, Exception)):
                project.subset(
                    source, lambda frame: frame[metadata_label] > 0,
                    data_columns=False, metadata_columns=[metadata_label],
                    metadata_mode="none",
                )
    finally:
        project.close()
