from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import pandas as pd

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import TextLength


def _seed(project: teal.Project):
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "text": ["one two", "three", "four five six", None],
            "record_type": ["paper", "paper", "session", "session"],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("source"),
        artifact_id="source",
        label="source",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": frame[["row_id"]],
            "data": frame[["text"]],
            "metadata": frame[["record_type"]],
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="source",
        artifact_type="table",
        label="source",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("source")


def test_text_length_artifact_inherits_data_and_supports_query_histogram_crosstab(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="exploration")
    try:
        source = _seed(project)
        lengths = project.translate(
            TextLength({"text": ["characters", "words"]}),
            source,
            batch_size=2,
        )["output"]
        assert set(lengths.components) == {"keys", "metadata"}
        assert lengths.data_artifact.artifact_id == source.artifact_id
        frame = lengths.query(
            key_columns=True,
            data_columns="text",
            metadata_columns=["record_type", "text_characters", "text_words"],
            metadata_mode="full",
            where="text_words <= 2",
            form="table",
        )
        assert frame["row_id"].astype(int).tolist() == [0, 1, 3]
        assert frame["text_words"].astype(int).tolist() == [2, 1, 0]
        compact = lengths.visualize.histogram(
            "text_words", by="record_type", bins=[0, 1, 2, 3], output="data"
        )
        assert compact["count"].sum() == 4
        table = lengths.analysis.crosstab("record_type", "text_words", margins=True)
        assert table.loc["All", "All"] == 4
    finally:
        project.close()


def _seed_sampling(project: teal.Project):
    frame = pd.DataFrame(
        {
            "row_id": list(range(30)),
            "text": [" ".join(["word"] * (i + 1)) for i in range(30)],
            "record_type": ["paper" if i % 2 == 0 else "session" for i in range(30)],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("sampling_source"),
        artifact_id="sampling_source",
        label="sampling_source",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": frame[["row_id"]],
            "data": frame[["text"]],
            "metadata": frame[["record_type"]],
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="sampling_source",
        artifact_type="table",
        label="sampling_source",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    source = project.get_artifact("sampling_source")
    return project.translate(TextLength({"text": ["words"]}), source)["output"]


def _sample_frame(lengths, **kwargs):
    return lengths.query(
        key_columns=True,
        data_columns=False,
        metadata_columns="text_words",
        metadata_mode="full",
        include_position=True,
        form="table",
        **kwargs,
    )


def test_filtered_sample_n_samples_after_where_and_is_reproducible(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="filtered_sampling")
    try:
        lengths = _seed_sampling(project)
        first = _sample_frame(
            lengths,
            where="text_words BETWEEN 10 AND 25",
            sample_n=6,
            random_state=42,
        )
        repeated = _sample_frame(
            lengths,
            where="text_words BETWEEN 10 AND 25",
            sample_n=6,
            random_state=42,
        )
        different = _sample_frame(
            lengths,
            where="text_words BETWEEN 10 AND 25",
            sample_n=6,
            random_state=43,
        )

        assert len(first) == 6
        assert first["text_words"].between(10, 25).all()
        assert (
            first["row_id"].astype(int).tolist()
            == repeated["row_id"].astype(int).tolist()
        )
        assert set(first["row_id"].astype(int)) != set(different["row_id"].astype(int))
    finally:
        project.close()


def test_filtered_sampling_handles_small_empty_and_positions_populations(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="filtered_sampling_edges")
    try:
        lengths = _seed_sampling(project)

        small = _sample_frame(
            lengths,
            where="text_words <= 3",
            sample_n=6,
            random_state=1,
        )
        assert sorted(small["text_words"].astype(int).tolist()) == [1, 2, 3]

        empty = _sample_frame(
            lengths,
            where="text_words > 100",
            sample_n=6,
            random_state=1,
        )
        assert empty.empty

        restricted = _sample_frame(
            lengths,
            positions=list(range(10)),
            where="text_words >= 5",
            sample_n=3,
            random_state=9,
        )
        assert len(restricted) == 3
        assert restricted["row_id"].astype(int).between(4, 9).all()
        assert restricted["text_words"].astype(int).between(5, 10).all()
    finally:
        project.close()


def test_filtered_sampling_applies_order_and_limit_after_sampling(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="filtered_sampling_order")
    try:
        lengths = _seed_sampling(project)
        selected = _sample_frame(
            lengths,
            where="text_words BETWEEN 5 AND 25",
            sample_n=8,
            random_state=17,
        )
        ordered = _sample_frame(
            lengths,
            where="text_words BETWEEN 5 AND 25",
            sample_n=8,
            random_state=17,
            order_by="text_words DESC",
            limit=3,
        )
        expected = (
            selected.sort_values("text_words", ascending=False, kind="stable")
            .head(3)["row_id"]
            .astype(int)
            .tolist()
        )
        assert ordered["row_id"].astype(int).tolist() == expected
        assert ordered["text_words"].astype(int).tolist() == sorted(
            ordered["text_words"].astype(int).tolist(), reverse=True
        )
    finally:
        project.close()


def test_filtered_sample_frac_uses_eligible_count_and_iter_batches_is_exact(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="filtered_sampling_frac")
    try:
        lengths = _seed_sampling(project)
        fraction = _sample_frame(
            lengths,
            where="text_words BETWEEN 11 AND 20",
            sample_frac=0.5,
            random_state=3,
        )
        assert len(fraction) == 5
        assert fraction["text_words"].between(11, 20).all()

        batches = list(
            lengths.query(
                key_columns=True,
                data_columns=False,
                metadata_columns="text_words",
                metadata_mode="full",
                where="text_words >= 10",
                sample_n=7,
                random_state=12,
                include_position=True,
                form="table",
                iter_batches=True,
                batch_size=3,
            )
        )
        combined = pd.concat(batches, ignore_index=True)
        assert len(combined) == 7
        assert combined["text_words"].astype(int).ge(10).all()
    finally:
        project.close()
