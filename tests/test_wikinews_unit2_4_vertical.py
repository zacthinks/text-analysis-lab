from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.translators import (
    CountVectorizer,
    DelimiterDecomposer,
    RegexCleaner,
    RegexReplaceRule,
    TextLength,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wikinews_2025_small"
ARTICLES_PATH = FIXTURE_DIR / "articles.jsonl"


def _external_modules():
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")


def _load_articles() -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows).sort_values("article_id").reset_index(drop=True)


def _register_articles(project: teal.Project):
    from text_analysis_lab.core.writer import create_artifact_writer

    articles = _load_articles()
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_articles"),
        artifact_id="art_articles",
        label="articles",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    for start in range(0, len(articles), 4):
        stop = min(len(articles), start + 4)
        part = articles.iloc[start:stop].reset_index(drop=True)
        writer.write(
            {
                "keys": part[["article_id"]],
                "data": part[["title", "text"]],
                "metadata": part[
                    [
                        "published_date",
                        "topic_group",
                        "region_group",
                        "source_url",
                        "license",
                    ]
                ],
            }
        )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_articles",
        artifact_type="table",
        label="articles",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_articles"), articles


def test_wikinews_units_2_to_4_vertical_slice(tmp_path: Path) -> None:
    _external_modules()
    pytest.importorskip("dask.distributed")

    project_path = tmp_path / "wikinews_units_2_4"
    project = teal.Project.create(project_path, name="wikinews_units_2_4")
    try:
        articles_artifact, articles = _register_articles(project)

        # Unit 2: decompose real article text into paragraphs using a literal
        # delimiter.  This should extend article_id with paragraph_id and retain
        # inherited article metadata.
        paragraphs = project.translate(
            DelimiterDecomposer(
                delimiter="\n\n",
                new_key="paragraph_id",
                text_field="text",
                output_text_field="paragraph_text",
            ),
            articles_artifact,
            workers=2,
            batch_size=4,
        )["output"]
        assert paragraphs.primary_key == ["article_id", "paragraph_id"]
        assert paragraphs.n_rows == 39
        paragraph_frame = paragraphs.query(
            key_columns=True,
            data_columns="paragraph_text",
            metadata_columns=["published_date", "topic_group"],
            metadata_mode="full",
            order_by="_position",
            form="table",
        )
        assert len(paragraph_frame) == 39
        assert paragraph_frame.groupby("article_id")["paragraph_id"].min().eq(0).all()
        first_date = articles.loc[articles["article_id"] == 1, "published_date"].iloc[0]
        assert set(
            paragraph_frame.loc[paragraph_frame["article_id"] == 1, "published_date"]
        ) == {first_date}

        # Unit 3: row-preserving vectorized regex cleaning.
        cleaner = RegexCleaner(
            text_field="paragraph_text",
            output_field="clean_text",
            rules=[
                RegexReplaceRule(r"[^\w\s'-]+", " "),
                RegexReplaceRule(r"\s+", " "),
            ],
            strip=True,
        )
        cleaned = project.translate(
            cleaner,
            paragraphs,
            workers=2,
            batch_size=7,
        )["output"]
        assert cleaned.primary_key == paragraphs.primary_key
        assert cleaned.n_rows == paragraphs.n_rows == 39

        # Reusable length measures become durable metadata without copying text.
        artifact_count = len(project.list_artifacts())
        lengths = project.translate(
            TextLength({"clean_text": ["words", "log_words"]}),
            cleaned,
            batch_size=8,
        )["output"]
        length_frame = lengths.query(
            key_columns=False,
            data_columns=False,
            metadata_columns=["clean_text_words", "clean_text_log_words"],
            metadata_mode="local",
            form="table",
        )
        assert len(length_frame) == 39
        assert (length_frame["clean_text_words"] > 0).all()
        np.testing.assert_allclose(
            length_frame["clean_text_log_words"].to_numpy(),
            np.log1p(length_frame["clean_text_words"].to_numpy()),
        )
        assert len(project.list_artifacts()) == artifact_count + 1

        # Unit 3/4: fit a real count DTM.  Vocabulary learning is whole-corpus;
        # the frozen operator can then transform in bounded parallel batches.
        vectorizer = CountVectorizer(
            text_field="clean_text",
            lowercase=True,
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
        )
        dtm = project.translate(vectorizer, cleaned, batch_size=6)["output"]
        assert dtm.artifact_type.value == "sparse_matrix"
        assert dtm.primary_key == cleaned.primary_key
        assert dtm.n_rows == cleaned.n_rows == 39
        matrix = dtm.get_matrix()
        assert sparse.issparse(matrix)
        assert matrix.shape[0] == 39
        assert matrix.shape[1] == len(dtm.get_data_columns())
        assert matrix.nnz > 0
        assert "trump" in dtm.get_data_columns()

        # Reload the frozen fitted operator and exercise real Dask translation.
        frozen = project.get_operator(vectorizer.operator_id)
        assert isinstance(frozen, CountVectorizer)
        dtm_parallel = project.translate(
            frozen,
            cleaned,
            workers=2,
            batch_size=5,
            max_outstanding_units=2,
        )["output"]
        assert dtm_parallel.get_data_columns() == dtm.get_data_columns()
        np.testing.assert_array_equal(
            dtm_parallel.get_matrix().toarray(),
            matrix.toarray(),
        )

        # Unit 4 matrix Analytic Methods should work directly on the count DTM
        # and remain ephemeral.
        before_analytics = len(project.list_artifacts())
        neighbors = dtm.analysis.nearest_neighbors(position=0, k=3, batch_size=8)
        assert len(neighbors) == 3
        assert {"article_id", "paragraph_id", "_position", "rank", "distance"}.issubset(
            neighbors.columns
        )
        cosine = dtm.analysis.cosine_similarity(position=0, other_position=1)
        cosine_distance = dtm.analysis.distance(
            position=0, other_position=1, metric="cosine"
        )
        assert cosine_distance == pytest.approx(1.0 - cosine)
        assert dtm.analysis.distance(
            position=0, other_position=1, metric="euclidean"
        ) >= 0.0

        matrix_stats = dtm.analysis.matrix_summary(batch_size=8)
        assert matrix_stats.n_rows == 39
        assert matrix_stats.n_features == matrix.shape[1]
        assert matrix_stats.nonzero_values == matrix.nnz
        assert 0.0 < matrix_stats.density < 1.0

        row_stats = dtm.analysis.row_summary(batch_size=8)
        assert len(row_stats) == 39
        assert {"article_id", "paragraph_id", "nonzero_features", "l2_norm"}.issubset(
            row_stats.columns
        )

        feature_stats = dtm.analysis.feature_summary(batch_size=8)
        trump_stats = feature_stats.set_index("feature").loc["trump"]
        assert trump_stats["sum"] > 0
        assert trump_stats["nonzero_rows"] > 0
        assert len(project.list_artifacts()) == before_analytics

        cleaned_id = cleaned.artifact_id
        dtm_id = dtm.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        cleaned = reopened.get_artifact(cleaned_id)
        dtm = reopened.get_artifact(dtm_id)
        assert cleaned.n_rows == 39
        assert dtm.n_rows == 39
        assert sparse.issparse(dtm.get_matrix())
        assert "trump" in dtm.get_data_columns()
    finally:
        reopened.close()


def test_count_vectorizer_fit_translate_can_resume_and_freeze_learned_vocabulary(
    tmp_path: Path,
) -> None:
    _external_modules()
    from types import MethodType

    project_path = tmp_path / "wikinews_count_resume"
    project = teal.Project.create(project_path, name="wikinews_count_resume")
    try:
        articles_artifact, _ = _register_articles(project)
        paragraphs = project.translate(
            DelimiterDecomposer(
                delimiter="\n\n",
                new_key="paragraph_id",
                text_field="text",
                output_text_field="paragraph_text",
            ),
            articles_artifact,
            workers=1,
            batch_size=5,
        )["output"]

        vectorizer = CountVectorizer(text_field="paragraph_text", min_df=1)
        original_translate_batch = vectorizer.translate_batch
        failed = {"done": False}

        def fail_after_fit_once(self, inputs, *, mode, request):
            result = original_translate_batch(inputs, mode=mode, request=request)
            if not failed["done"]:
                failed["done"] = True
                raise RuntimeError("intentional CountVectorizer fit interruption")
            return result

        vectorizer.translate_batch = MethodType(fail_after_fit_once, vectorizer)
        with pytest.raises(RuntimeError, match="intentional CountVectorizer fit interruption"):
            project.translate(vectorizer, paragraphs)

        operation_rows = project.catalog.operations_using_operator(vectorizer.operator_id)
        assert len(operation_rows) == 1
        operation_id = str(operation_rows[0]["operation_id"])

        outputs = project.resume_operation(operation_id)
        dtm = outputs["output"]
        assert dtm.status == "complete"
        assert dtm.n_rows == paragraphs.n_rows == 39
        assert sparse.issparse(dtm.get_matrix())

        frozen = project.get_operator(vectorizer.operator_id)
        assert isinstance(frozen, CountVectorizer)
        assert frozen.is_fitted
        assert frozen.vocabulary_
        assert frozen.operator_id == vectorizer.operator_id
    finally:
        project.close()
