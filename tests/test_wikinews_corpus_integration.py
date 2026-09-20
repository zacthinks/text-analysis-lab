from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import text_analysis_lab as teal

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wikinews_2025_small"
ARTICLES_PATH = FIXTURE_DIR / "articles.jsonl"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])")


def _load_articles() -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows).sort_values("article_id").reset_index(drop=True)


def _sentence_frame(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for article in articles.to_dict("records"):
        paragraphs = [
            p.strip() for p in str(article["text"]).split("\n\n") if p.strip()
        ]
        for paragraph_id, paragraph in enumerate(paragraphs):
            sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()]
            for sentence_id, sentence in enumerate(sentences):
                rows.append(
                    {
                        "article_id": int(article["article_id"]),
                        "paragraph_id": int(paragraph_id),
                        "sentence_id": int(sentence_id),
                        "sentence_text": sentence,
                    }
                )
    return pd.DataFrame(rows)


def test_wikinews_fixture_is_frozen_and_structurally_realistic() -> None:
    articles = _load_articles()
    sentences = _sentence_frame(articles)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert len(articles) == manifest["article_count"] == 15
    assert (
        articles["article_id"].astype(int).tolist()
        == manifest["article_ids"]
        == list(range(1, 16))
    )
    assert len(sentences) == manifest["sentence_count"] == 93
    assert (
        sum(manifest["per_article"][str(i)]["paragraphs"] for i in range(1, 16))
        == manifest["paragraph_count"]
        == 39
    )
    assert articles["source"].eq("Wikinews").all()
    assert articles["license"].eq("CC BY 4.0").all()
    assert articles["excerpted"].eq(True).all()
    assert articles["source_url"].str.startswith("https://en.wikinews.org/wiki/").all()
    assert articles["topic_group"].nunique() >= 5
    assert articles["region_group"].nunique() >= 6

    # Real hierarchical irregularity matters more than sheer size: articles have
    # different sentence counts and paragraphs have different sentence counts.
    per_article_sentence_counts = sentences.groupby("article_id").size()
    assert per_article_sentence_counts.nunique() > 1
    per_paragraph_sentence_counts = sentences.groupby(
        ["article_id", "paragraph_id"]
    ).size()
    assert per_paragraph_sentence_counts.nunique() > 1


def _external_modules():
    pyarrow = pytest.importorskip("pyarrow")
    duckdb = pytest.importorskip("duckdb")
    return pyarrow, duckdb


def _register_table_artifact(
    project: teal.Project,
    *,
    artifact_id: str,
    label: str,
    keys: pd.DataFrame,
    data: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    lineage_mode: str = "new_key",
    basis_artifact_ids: tuple[str, ...] = (),
    batch_size: int = 5,
):
    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=basis_artifact_ids,
    )
    for start in range(0, len(keys), batch_size):
        stop = min(len(keys), start + batch_size)
        payload: dict[str, object] = {
            "keys": keys.iloc[start:stop].reset_index(drop=True),
        }
        if data is not None:
            payload["data"] = data.iloc[start:stop].reset_index(drop=True)
        if metadata is not None:
            payload["metadata"] = metadata.iloc[start:stop].reset_index(drop=True)
        writer.write(payload)
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=label,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=basis_artifact_ids,
    )
    return project.get_artifact(artifact_id)


def _seed_wikinews_project(project: teal.Project):
    articles = _load_articles()
    sentences = _sentence_frame(articles)

    article_artifact = _register_table_artifact(
        project,
        artifact_id="art_articles",
        label="articles",
        keys=articles[["article_id"]],
        data=articles[["title", "text"]],
        metadata=articles[
            ["published_date", "topic_group", "region_group", "source_url", "license"]
        ],
        lineage_mode="new_key",
        batch_size=4,
    )
    project.add_artifact_alias(article_artifact, "articles")

    sentence_artifact = _register_table_artifact(
        project,
        artifact_id="art_sentences",
        label="sentences",
        keys=sentences[["article_id", "paragraph_id", "sentence_id"]],
        data=sentences[["sentence_text"]],
        lineage_mode="extended_key",
        basis_artifact_ids=(article_artifact.artifact_id,),
        batch_size=11,
    )
    project.add_artifact_alias(sentence_artifact, "sentences")
    return article_artifact, sentence_artifact, articles, sentences


def _key_tuples(artifact) -> list[tuple[int, ...]]:
    frame = artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        form="table",
        include_position=True,
        order_by="_position",
    )
    return [
        tuple(int(row[col]) for col in artifact.primary_key)
        for _, row in frame.iterrows()
    ]


def test_wikinews_query_lineage_context_kwic_sql_and_reopen(tmp_path: Path) -> None:
    _external_modules()
    project_path = tmp_path / "wikinews_project"
    project = teal.Project.create(project_path, name="wikinews_golden")
    try:
        articles_artifact, sentences_artifact, articles, sentences = (
            _seed_wikinews_project(project)
        )

        # Extended-key metadata should bubble from article rows to their sentences.
        sample = sentences_artifact.query(
            key_columns=True,
            data_columns="sentence_text",
            metadata_columns=["published_date", "topic_group", "region_group"],
            metadata_mode="full",
            order_by="_position",
            limit=12,
            form="table",
        )
        assert {"article_id", "paragraph_id", "sentence_id", "sentence_text"}.issubset(
            sample.columns
        )
        assert {"published_date", "topic_group", "region_group"}.issubset(
            sample.columns
        )
        first_article = articles.iloc[0]
        assert set(sample.loc[sample["article_id"] == 1, "published_date"]) == {
            first_article["published_date"]
        }
        assert set(sample.loc[sample["article_id"] == 1, "topic_group"]) == {
            first_article["topic_group"]
        }

        # Context is deliberately position-based. The primary keys make paragraph
        # and article boundary crossings directly visible without dummy rows.
        first_of_second_paragraph = tuple(
            sentences.loc[
                (sentences["article_id"] == 1) & (sentences["paragraph_id"] == 1)
            ]
            .iloc[0][["article_id", "paragraph_id", "sentence_id"]]
            .astype(int)
        )
        context = sentences_artifact.get_context(
            first_of_second_paragraph, before=1, after=1
        )
        assert context["article_id"].astype(int).tolist() == [1, 1, 1]
        assert context["paragraph_id"].astype(int).tolist()[0] == 0
        assert context["paragraph_id"].astype(int).tolist()[1] == 1

        first_article_2 = tuple(
            sentences.loc[sentences["article_id"] == 2]
            .iloc[0][["article_id", "paragraph_id", "sentence_id"]]
            .astype(int)
        )
        previous = sentences_artifact.get_previous(first_article_2, n=1)
        assert int(previous.iloc[0]["article_id"]) == 1
        assert int(previous.iloc[-1]["article_id"]) != first_article_2[0]

        kwic = sentences_artifact.kwic(
            "Trump",
            search_columns="sentence_text",
            data_columns="sentence_text",
            key_columns=True,
            target_matches=20,
        ).to_frame()
        assert len(kwic) >= 4
        assert set(kwic["article_id"].astype(int)) == {1, 6, 13}

        by_topic = project.sql(
            ["sentences"],
            'SELECT topic_group, COUNT(*) AS n FROM "sentences" GROUP BY topic_group ORDER BY topic_group',
        )
        assert int(by_topic["n"].sum()) == len(sentences)
        assert set(by_topic["topic_group"]) == set(articles["topic_group"])

        summary = teal.analysis.summarize(sentences_artifact)
        assert summary.n_rows == len(sentences) == 93
        assert tuple(summary.primary_key) == (
            "article_id",
            "paragraph_id",
            "sentence_id",
        )
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        assert reopened.project_id == "wikinews_golden"
        reopened_sentences = reopened.get_artifact("sentences")
        assert reopened_sentences.n_rows == 93
        assert _key_tuples(reopened_sentences) == [
            tuple(map(int, row))
            for row in sentences[
                ["article_id", "paragraph_id", "sentence_id"]
            ].itertuples(index=False, name=None)
        ]
    finally:
        reopened.close()


def test_wikinews_split_is_reproducible_exhaustive_and_stratified(
    tmp_path: Path,
) -> None:
    _external_modules()
    project = teal.Project.create(tmp_path / "project", name="wikinews_split")
    try:
        articles_artifact, _, articles, _ = _seed_wikinews_project(project)
        outputs_a = project.split(
            articles_artifact,
            labels=("explore", "confirm"),
            proportions=(0.7, 0.3),
            random_state=2025,
            stratify="topic_group",
        )
        outputs_b = project.split(
            articles_artifact,
            labels=("explore", "confirm"),
            proportions=(0.7, 0.3),
            random_state=2025,
            stratify="topic_group",
        )

        explore_a = {key[0] for key in _key_tuples(outputs_a["explore"])}
        confirm_a = {key[0] for key in _key_tuples(outputs_a["confirm"])}
        explore_b = {key[0] for key in _key_tuples(outputs_b["explore"])}
        confirm_b = {key[0] for key in _key_tuples(outputs_b["confirm"])}

        assert explore_a == explore_b
        assert confirm_a == confirm_b
        assert explore_a.isdisjoint(confirm_a)
        assert explore_a | confirm_a == set(articles["article_id"].astype(int))
        assert explore_a and confirm_a

        # Every topic with at least two source documents should remain represented
        # in the combined split exactly as in the source.
        source_topics = articles.set_index("article_id")["topic_group"].to_dict()
        combined = list(explore_a | confirm_a)
        assert sorted(source_topics[i] for i in combined) == sorted(
            articles["topic_group"].tolist()
        )
    finally:
        project.close()


def test_wikinews_parallel_subset_matches_sequential(tmp_path: Path) -> None:
    _external_modules()
    pytest.importorskip("dask.distributed")

    project = teal.Project.create(tmp_path / "project", name="wikinews_subset")
    try:
        _, sentences_artifact, _, _ = _seed_wikinews_project(project)
        rule_path = tmp_path / "keep_politics.py"
        rule_path.write_text(
            "def keep(packet):\n    return packet['topic_group'] == 'politics'\n",
            encoding="utf-8",
        )

        sequential = project.subset(
            sentences_artifact,
            (rule_path, "keep"),
            data_columns=False,
            metadata_columns="topic_group",
            metadata_mode="full",
            batch_size=7,
            workers=1,
            output_label="politics_sentences_seq",
        )["politics_sentences_seq"]

        parallel = project.subset(
            sentences_artifact,
            (rule_path, "keep"),
            data_columns=False,
            metadata_columns="topic_group",
            metadata_mode="full",
            batch_size=7,
            workers=3,
            output_label="politics_sentences_parallel",
        )["politics_sentences_parallel"]

        seq_keys = _key_tuples(sequential)
        par_keys = _key_tuples(parallel)
        assert seq_keys == par_keys
        assert seq_keys
        assert all(key[0] in {1, 2, 3, 4, 6, 9, 12, 13} for key in seq_keys)
    finally:
        project.close()


def test_wikinews_parallel_subset_fail_reopen_resume(tmp_path: Path) -> None:
    _external_modules()
    pytest.importorskip("dask.distributed")

    sentinel = tmp_path / "failed_once.txt"
    rule_path = tmp_path / "fail_once_real_corpus.py"
    rule_path.write_text(
        "from pathlib import Path\n"
        f"SENTINEL = Path({str(sentinel)!r})\n"
        "def keep(packet):\n"
        "    if (packet['article_id'] == 8).any() and not SENTINEL.exists():\n"
        "        SENTINEL.write_text('failed-once', encoding='utf-8')\n"
        "        raise RuntimeError('intentional Wikinews corpus failure')\n"
        "    return packet['article_id'] % 2 == 1\n",
        encoding="utf-8",
    )

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="wikinews_resume")
    try:
        _, sentences_artifact, _, sentences = _seed_wikinews_project(project)
        with pytest.raises(RuntimeError, match="intentional Wikinews corpus failure"):
            project.subset(
                sentences_artifact,
                (rule_path, "keep"),
                data_columns=False,
                batch_size=9,
                workers=3,
                output_label="odd_articles",
            )
        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["status"] == "failed"
        assert sentinel.exists()
    finally:
        project.close()

    project = teal.Project.open(project_path)
    try:
        resumed = project.resume_operation(operation_id)["odd_articles"]
        assert resumed.status == "complete"
        actual = _key_tuples(resumed)
        expected = [
            tuple(map(int, row))
            for row in sentences.loc[
                sentences["article_id"] % 2 == 1,
                ["article_id", "paragraph_id", "sentence_id"],
            ].itertuples(index=False, name=None)
        ]
        assert actual == expected

        import sqlite3

        plan_path = project.storage.operation_dir(operation_id) / "plan.sqlite"
        with sqlite3.connect(plan_path) as con:
            statuses = [
                row[0]
                for row in con.execute(
                    "SELECT status FROM plan_units ORDER BY unit_index"
                )
            ]
        assert statuses and set(statuses) == {"complete"}
    finally:
        project.close()
