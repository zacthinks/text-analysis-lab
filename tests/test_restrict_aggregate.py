from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab import Project
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.translators import DelimiterDecomposer


def _table(artifact, *, data=True, metadata=False):
    return artifact.query(
        key_columns=True,
        data_columns=data,
        metadata_columns=metadata,
        metadata_mode="full" if metadata else "none",
        order_by="_position",
        form="table",
    )


def _build_sentence_project(tmp_path):
    csv_path = tmp_path / "docs.csv"
    pd.DataFrame(
        {
            "text": ["alpha|beta", "gamma", "delta|epsilon"],
            "group": ["A", "B", "A"],
        }
    ).to_csv(csv_path, index=False)
    project = Project.create(
        tmp_path / "project", name="restrict_aggregate", delete_existing=True
    )
    docs = project.read_csv(
        csv_path,
        text_fields="text",
        metadata_fields=["group"],
        output_label="docs",
    )
    sentences = project.translate(
        DelimiterDecomposer(delimiter="|", new_key="sentence_id"), docs
    )["output"]
    return project, docs, sentences


def test_restrict_uses_compatible_key_prefix_and_preserves_source_grain(tmp_path):
    project, docs, sentences = _build_sentence_project(tmp_path)
    try:
        train = project.select_keys(docs, [0, 2], output_label="train")
        restricted = project.restrict(
            sentences, to=train, output_label="train_sentences"
        )

        frame = _table(restricted)
        assert list(restricted.primary_key) == ["row_id", "sentence_id"]
        assert frame[["row_id", "sentence_id"]].astype(int).to_dict("records") == [
            {"row_id": 0, "sentence_id": 0},
            {"row_id": 0, "sentence_id": 1},
            {"row_id": 2, "sentence_id": 0},
            {"row_id": 2, "sentence_id": 1},
        ]
        assert frame["text"].tolist() == ["alpha", "beta", "delta", "epsilon"]
        assert restricted.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        assert restricted.descriptor["lineage"]["basis_artifact_ids"] == [
            sentences.artifact_id
        ]

        sources = {
            row["source_label"]: row["source_artifact_id"]
            for row in project.operation_sources(restricted.operation_id)
        }
        assert sources == {"source": sentences.artifact_id, "domain": train.artifact_id}
    finally:
        project.close()


def test_restrict_allows_domain_rows_with_no_finer_source_rows(tmp_path):
    project, docs, sentences = _build_sentence_project(tmp_path)
    try:
        # Build a legitimate extended-key source in which document 1 has no children.
        only_outer_docs = project.select_keys(docs, [0, 2], output_label="outer_docs")
        sparse_sentences = project.restrict(
            sentences, to=only_outer_docs, output_label="sparse_sentences"
        )

        # Restricting that source to the full document domain is valid. Document 1
        # simply contributes zero child rows; restrict is a semijoin, not a coverage assertion.
        restored_domain = project.restrict(
            sparse_sentences, to=docs, output_label="full_domain_sparse"
        )
        frame = _table(restored_domain)
        assert set(frame["row_id"].astype(int)) == {0, 2}
    finally:
        project.close()


def test_restrict_rejects_non_prefix_domain_key(tmp_path):
    project, docs, sentences = _build_sentence_project(tmp_path)
    try:
        line_zero = project.select_keys(sentences, [(0, 0)], output_label="line_zero")
        with pytest.raises(ArtifactError, match="primary key to be a prefix"):
            project.restrict(docs, to=line_zero)
    finally:
        project.close()


def test_aggregate_reduces_key_by_named_level_and_sums_sentence_scores(tmp_path):
    project, _docs, sentences = _build_sentence_project(tmp_path)
    try:
        sentence_rows = _table(sentences, data=False)
        probabilities = {
            (0, 0): 0.9,
            (0, 1): 0.8,
            (1, 0): 0.2,
            (2, 0): 0.6,
            (2, 1): 0.7,
        }
        labels = sentence_rows[["row_id", "sentence_id"]].copy()
        labels["probability"] = [
            probabilities[(int(row.row_id), int(row.sentence_id))]
            for row in labels.itertuples(index=False)
        ]
        sentence_scores = project.from_keyed_frame(
            sentences,
            labels,
            data_fields=["probability"],
            require_complete=True,
            output_label="sentence_scores",
        )

        document_scores = project.aggregate(
            sentence_scores,
            to_key="row_id",
            aggregations={"probability": {"evidence": "sum"}},
            output_label="document_scores",
        )
        frame = _table(document_scores, metadata=True)

        assert list(document_scores.primary_key) == ["row_id"]
        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["evidence"].tolist() == pytest.approx([1.7, 0.2, 1.3])
        assert frame["n_rows"].astype(int).tolist() == [2, 1, 2]
        assert document_scores.descriptor["lineage"]["lineage_mode"] == "reduced_key"
        assert document_scores.descriptor["lineage"]["basis_artifact_ids"] == [
            sentence_scores.artifact_id
        ]
        # Once the key returns to document grain, coarser source metadata is visible again.
        assert frame["group"].tolist() == ["A", "B", "A"]
    finally:
        project.close()


def test_aggregate_to_key_is_last_retained_key_and_must_reduce(tmp_path):
    project, _docs, sentences = _build_sentence_project(tmp_path)
    try:
        sentence_rows = _table(sentences, data=False)
        labels = sentence_rows[["row_id", "sentence_id"]].copy()
        labels["score"] = 1.0
        source = project.from_keyed_frame(
            sentences,
            labels,
            data_fields=["score"],
            require_complete=True,
        )

        with pytest.raises(ArtifactError, match="not in source primary key"):
            project.aggregate(source, to_key="missing", aggregations={"score": "sum"})
        with pytest.raises(ArtifactError, match="proper prefix"):
            project.aggregate(
                source, to_key="sentence_id", aggregations={"score": "sum"}
            )
    finally:
        project.close()


def test_aggregate_new_output_centric_data_and_metadata_api(tmp_path):
    from text_analysis_lab import agg, concat, literal

    project, _docs, sentences = _build_sentence_project(tmp_path)
    try:
        sentence_rows = _table(sentences, data=False)
        values = sentence_rows[["row_id", "sentence_id"]].copy()
        values["probability"] = [0.9, 0.8, 0.2, 0.6, 0.7]
        values["tag"] = ["x", "y", "z", "x", "x"]
        source = project.from_keyed_frame(
            sentences,
            values,
            data_fields=["probability", "tag"],
            require_complete=True,
            output_label="sentence_values",
        )

        reduced = project.aggregate(
            source,
            to_key="row_id",
            data={
                "probability_sum": agg("probability", "sum"),
                "probability_mean": agg("probability", "mean"),
                "tags": agg("tag", concat("|")),
                "tag_unique": agg("tag", "unique"),
            },
            metadata={
                "group_first": agg("group", "first"),
                "course_number": literal(12),
            },
            batch_size=1,
            output_label="document_values",
        )
        frame = _table(reduced, metadata=True)

        assert frame["row_id"].astype(int).tolist() == [0, 1, 2]
        assert frame["probability_sum"].tolist() == pytest.approx([1.7, 0.2, 1.3])
        assert frame["probability_mean"].tolist() == pytest.approx([0.85, 0.2, 0.65])
        assert frame["tags"].tolist() == ["x|y", "z", "x|x"]
        # DuckDB/Pandas may expose persisted LIST values as Python lists or
        # NumPy arrays depending on platform/backend versions. The public
        # contract here is the ordered collection contents, not the concrete
        # pandas cell container type.
        assert [list(value) for value in frame["tag_unique"].tolist()] == [
            ["x", "y"],
            ["z"],
            ["x"],
        ]
        assert frame["group_first"].tolist() == ["A", "B", "A"]
        assert frame["course_number"].astype(int).tolist() == [12, 12, 12]
        assert frame["n_rows"].astype(int).tolist() == [2, 1, 2]
        assert reduced.descriptor["lineage"]["lineage_mode"] == "reduced_key"

        operation = (
            project.storage.operation_dir(reduced.operation_id) / "operation.json"
        )
        request = __import__("json").loads(operation.read_text())["request"]
        assert request["batch_unit"] == "retained_key_groups"
    finally:
        project.close()


def test_aggregate_first_last_are_primary_key_ordered(tmp_path):
    from text_analysis_lab import agg

    project, _docs, sentences = _build_sentence_project(tmp_path)
    try:
        sentence_rows = _table(sentences, data=False)
        values = sentence_rows[["row_id", "sentence_id"]].copy()
        values["marker"] = ["first-0", "last-0", "only-1", "first-2", "last-2"]
        source = project.from_keyed_frame(
            sentences,
            values,
            data_fields=["marker"],
            require_complete=True,
        )
        reduced = project.aggregate(
            source,
            to_key="row_id",
            data={
                "first_marker": agg("marker", "first"),
                "last_marker": agg("marker", "last"),
            },
            batch_size=1,
        )
        frame = _table(reduced)
        assert frame["first_marker"].tolist() == ["first-0", "only-1", "first-2"]
        assert frame["last_marker"].tolist() == ["last-0", "only-1", "last-2"]
    finally:
        project.close()


def test_matrix_aggregate_sums_all_features_and_aggregates_metadata(tmp_path):
    from text_analysis_lab import agg, literal
    from text_analysis_lab.translators import CountVectorizer

    project, _docs, sentences = _build_sentence_project(tmp_path)
    try:
        sentence_dtm = project.translate(
            CountVectorizer(text_field="text"),
            sentences,
        )["output"]

        document_dtm = project.aggregate(
            sentence_dtm,
            to_key="row_id",
            data="sum",
            metadata={
                "group_first": agg("group", "first"),
                "course_number": literal(12),
            },
            batch_size=1,
            output_label="document_dtm",
        )

        assert document_dtm.artifact_type.value == "sparse_matrix"
        assert list(document_dtm.primary_key) == ["row_id"]
        columns = list(document_dtm.get_data_columns())
        matrix = document_dtm.get_matrix().toarray()
        rows = document_dtm.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=["n_rows", "group_first", "course_number"],
            metadata_mode="local",
            form="table",
        )
        assert rows["row_id"].astype(int).tolist() == [0, 1, 2]
        assert rows["n_rows"].astype(int).tolist() == [2, 1, 2]
        assert rows["group_first"].tolist() == ["A", "B", "A"]
        assert rows["course_number"].astype(int).tolist() == [12, 12, 12]

        by_term = {
            term: matrix[:, index].tolist() for index, term in enumerate(columns)
        }
        assert by_term["alpha"] == pytest.approx([1, 0, 0])
        assert by_term["beta"] == pytest.approx([1, 0, 0])
        assert by_term["gamma"] == pytest.approx([0, 1, 0])
        assert by_term["delta"] == pytest.approx([0, 0, 1])
        assert by_term["epsilon"] == pytest.approx([0, 0, 1])
    finally:
        project.close()
