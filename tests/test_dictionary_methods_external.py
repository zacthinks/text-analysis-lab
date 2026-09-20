from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import DictionaryTranslator

_VALUES = np.array(
    [
        [2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 4.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
    ]
)
_FEATURES = ["good", "great", "bad", "awful", "economy", "economic", "neutral"]


def _register_dtm(project: teal.Project):
    artifact_id = "art_dictionary_dtm"
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="count_dtm",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    for start in (0, 2):
        writer.write(
            {
                "keys": pd.DataFrame({"doc_id": [start, start + 1]}),
                "data": {
                    "values": sparse.csr_matrix(_VALUES[start : start + 2]),
                    "columns": _FEATURES,
                },
            }
        )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix",
        label="count_dtm",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_dictionary_translation_and_analysis_real_sparse_round_trip(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "dictionary_project"
    project = teal.Project.create(project_path, name="dictionary_project")
    try:
        dtm = _register_dtm(project)
        content = teal.dictionaries.Dictionary(
            {
                "positive": ["good", "great"],
                "negative": ["bad", "awful"],
                "neutral": ["neutral"],
                "economy": ["econom*"],
            }
        )
        polarity_dictionary = teal.dictionaries.PolarityDictionary(
            content,
            positive="positive",
            negative="negative",
            neutral="neutral",
        )
        valence_dictionary = teal.dictionaries.ValenceDictionary(
            {"good": 2.0, "great": 3.0, "bad": -2.0, "awful": -4.0}
        )

        # Pre-translation audit remains ephemeral.
        before = len(project.list_artifacts())
        matches = dtm.analysis.dictionary_matches(content)
        assert {"economy", "economic"}.issubset(
            set(matches.loc[matches["key"] == "economy", "feature"])
        )
        assert len(project.list_artifacts()) == before

        polarity_artifact = project.translate(
            DictionaryTranslator(polarity_dictionary),
            dtm,
            batch_size=2,
        )["output"]
        assert polarity_artifact.primary_key == ["doc_id"]
        assert polarity_artifact.get_data_columns() == [
            "positive",
            "negative",
            "neutral",
        ]
        assert polarity_artifact.get_matrix().toarray().tolist() == [
            [3, 0, 4],
            [0, 3, 1],
            [0, 0, 0],
            [1, 1, 2],
        ]
        local = polarity_artifact.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=["matched", "unmatched", "total"],
            metadata_mode="local",
            form="table",
        )
        assert local["matched"].tolist() == [7, 4, 0, 4]
        assert local["unmatched"].tolist() == [3, 1, 0, 0]
        polarity = polarity_artifact.analysis.polarity(zero_division=0.0, batch_size=2)
        assert polarity["neutral"].tolist() == pytest.approx([4, 1, 0, 2])
        assert polarity["matched_difference"].tolist() == pytest.approx(
            [3 / 7, -3 / 4, 0, 0]
        )
        assert polarity["total_difference"].tolist() == pytest.approx([0.3, -0.6, 0, 0])

        valence_artifact = project.translate(
            DictionaryTranslator(valence_dictionary),
            dtm,
            batch_size=2,
        )["output"]
        assert valence_artifact.get_data_columns() == ["-4", "-2", "2", "3"]
        assert valence_artifact.get_matrix().toarray().tolist() == [
            [0, 0, 2, 1],
            [2, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
        ]
        valence = valence_artifact.analysis.valence(zero_division=0.0, batch_size=2)
        assert valence["weighted_sum"].tolist() == pytest.approx([7, -10, 0, 0])
        assert valence["mean_matched"].tolist() == pytest.approx([7 / 3, -10 / 3, 0, 0])
        assert valence["median"].tolist() == pytest.approx([2, -4, 0, 0])

        polarity_id = polarity_artifact.artifact_id
        valence_id = valence_artifact.artifact_id
        polarity_operator_id = project.catalog.get_operation(
            polarity_artifact.operation_id
        )["operator_id"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        polarity_artifact = reopened.get_artifact(polarity_id)
        valence_artifact = reopened.get_artifact(valence_id)
        assert polarity_artifact.analysis.polarity()[
            "difference"
        ].tolist() == pytest.approx([3, -3, 0, 0])
        assert valence_artifact.analysis.valence()[
            "mean_matched"
        ].tolist() == pytest.approx([7 / 3, -10 / 3, 0, 0])

        # The frozen operator retains the dictionary rules and can be reused.
        operator = reopened.get_operator(str(polarity_operator_id))
        assert isinstance(operator, DictionaryTranslator)
        assert operator.dictionary_kind == "polarity"
        reused = reopened.translate(
            operator, reopened.get_artifact("art_dictionary_dtm"), batch_size=3
        )["output"]
        assert (
            reused.get_matrix().toarray().tolist()
            == polarity_artifact.get_matrix().toarray().tolist()
        )
    finally:
        reopened.close()


def test_dictionary_translation_real_parallel_matches_sequential(
    tmp_path: Path,
) -> None:
    pytest.importorskip("dask.distributed")
    project = teal.Project.create(
        tmp_path / "dictionary_parallel", name="dictionary_parallel"
    )
    try:
        dtm = _register_dtm(project)
        dictionary = teal.dictionaries.ValenceDictionary(
            {"good": 2.0, "great": 3.0, "bad": -2.0, "awful": -4.0}
        )
        sequential = project.translate(
            DictionaryTranslator(dictionary), dtm, workers=1, batch_size=2
        )["output"]
        parallel = project.translate(
            DictionaryTranslator(dictionary),
            dtm,
            workers=2,
            batch_size=2,
            max_outstanding_units=2,
        )["output"]
        assert parallel.get_data_columns() == sequential.get_data_columns()
        assert (
            parallel.get_matrix().toarray().tolist()
            == sequential.get_matrix().toarray().tolist()
        )
        sequential_meta = sequential.query(
            key_columns=False,
            data_columns=False,
            metadata_columns=["matched", "unmatched", "total"],
            metadata_mode="local",
            form="table",
        )
        parallel_meta = parallel.query(
            key_columns=False,
            data_columns=False,
            metadata_columns=["matched", "unmatched", "total"],
            metadata_mode="local",
            form="table",
        )
        assert parallel_meta.to_dict("list") == sequential_meta.to_dict("list")
    finally:
        project.close()
