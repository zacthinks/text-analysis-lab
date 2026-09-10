from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.translators import SpacyTranslator
from text_analysis_lab.translators.spacy_translator import (
    SENTENCE_DATA_COLUMNS,
    TOKEN_DATA_COLUMNS,
)


def _external_modules():
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    return pytest.importorskip("spacy")


def _seed_documents(project: teal.Project, rows: pd.DataFrame):
    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_documents"),
        artifact_id="art_documents",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": rows[["document_id"]].reset_index(drop=True),
            "data": rows[["text"]].reset_index(drop=True),
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_documents",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_documents")


def _saved_sentencizer_model(tmp_path: Path) -> Path:
    spacy = _external_modules()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model_dir = tmp_path / "spacy_test_model"
    nlp.to_disk(model_dir)
    return model_dir


def test_real_spacy_saved_pipeline_parallel_two_output_structure_and_reopen(tmp_path: Path) -> None:
    _external_modules()
    pytest.importorskip("dask.distributed")
    model_dir = _saved_sentencizer_model(tmp_path)
    rows = pd.DataFrame(
        {
            "document_id": list(range(1, 7)),
            "text": [
                "Alice sees Bob. They wave.",
                "A second document has one sentence.",
                "Numbers like 42 are tokens. So is $5.",
                "Visit https://example.com. Then email test@example.com.",
                "Punctuation (matters). Quotes \"too\".",
                "Final document. Final sentence.",
            ],
        }
    )

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="spacy_external")
    try:
        source = _seed_documents(project, rows)
        outputs = project.translate(
            SpacyTranslator(model=str(model_dir), spacy_batch_size=2),
            source,
            workers=2,
            batch_size=2,
            max_outstanding_units=2,
        )
        assert list(outputs) == ["sentences", "tokens"]
        sentences = outputs["sentences"]
        tokens = outputs["tokens"]
        assert sentences.status == "complete"
        assert tokens.status == "complete"
        assert sentences.primary_key == ["document_id", "sentence_id"]
        assert tokens.primary_key == ["document_id", "sentence_id", "token_id"]
        assert sentences.descriptor["lineage"]["lineage_mode"] == "extended_key"
        assert sentences.descriptor["lineage"]["basis_artifact_ids"] == [source.artifact_id]
        assert tokens.descriptor["lineage"]["lineage_mode"] == "extended_key"
        assert tokens.descriptor["lineage"]["basis_artifact_ids"] == [sentences.artifact_id]

        sentence_frame = sentences.query(
            key_columns=True,
            data_columns=list(SENTENCE_DATA_COLUMNS),
            order_by="_position",
            form="table",
        )
        token_frame = tokens.query(
            key_columns=True,
            data_columns=list(TOKEN_DATA_COLUMNS),
            order_by="_position",
            form="table",
        )
        assert len(sentence_frame) == 11
        assert len(token_frame) > len(sentence_frame)
        assert set(TOKEN_DATA_COLUMNS).issubset(token_frame.columns)
        assert token_frame.groupby(["document_id", "sentence_id"])["token_id"].min().eq(0).all()
        assert token_frame.groupby(["document_id", "sentence_id"])["token_id"].apply(
            lambda values: values.astype(int).tolist() == list(range(len(values)))
        ).all()
        assert token_frame.groupby(["document_id", "sentence_id"])["head_token_id"].apply(
            lambda values: set(values.astype(int)).issubset(set(range(len(values))))
        ).all()

        # Character offsets are document-relative and recover exact source substrings.
        text_by_id = rows.set_index("document_id")["text"].to_dict()
        for row in sentence_frame.to_dict("records"):
            text = text_by_id[int(row["document_id"])]
            assert text[int(row["char_start"]) : int(row["char_end"])] == row["text"]
        for row in token_frame.to_dict("records"):
            text = text_by_id[int(row["document_id"])]
            assert text[int(row["char_start"]) : int(row["char_end"])] == row["text"]

        sentence_id = sentences.artifact_id
        token_id = tokens.artifact_id
        project.close()
        project = teal.Project.open(project_path)
        reopened_sentences = project.get_artifact(sentence_id)
        reopened_tokens = project.get_artifact(token_id)
        assert reopened_sentences.primary_key == ["document_id", "sentence_id"]
        assert reopened_tokens.primary_key == ["document_id", "sentence_id", "token_id"]
        assert reopened_tokens.descriptor["lineage"]["basis_artifact_ids"] == [sentence_id]
    finally:
        project.close()


def test_spacy_two_output_resume_after_mid_operation_failure(tmp_path: Path, monkeypatch) -> None:
    spacy = _external_modules()
    import text_analysis_lab.translators.spacy_translator as module

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    sentinel = tmp_path / "failed_once.txt"

    class FailOncePipe:
        def pipe(self, texts, *, batch_size, n_process):
            texts = list(texts)
            if any("TRIGGER" in text for text in texts) and not sentinel.exists():
                sentinel.write_text("failed", encoding="utf-8")
                raise RuntimeError("intentional spaCy translator integration failure")
            yield from nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    fake = FailOncePipe()
    monkeypatch.setattr(module, "_load_spacy_pipeline", lambda model, disable: fake)

    rows = pd.DataFrame(
        {
            "document_id": list(range(8)),
            "text": [
                "zero one.",
                "one two.",
                "two three.",
                "three four.",
                "TRIGGER failure here.",
                "five six.",
                "six seven.",
                "seven eight.",
            ],
        }
    )
    project_path = tmp_path / "resume_project"
    project = teal.Project.create(project_path, name="spacy_resume")
    try:
        source = _seed_documents(project, rows)
        with pytest.raises(RuntimeError, match="intentional spaCy translator"):
            project.translate(
                SpacyTranslator(model="fake"),
                source,
                workers=1,
                batch_size=2,
            )
        assert sentinel.exists()
        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["status"] == "failed"

        project.close()
        project = teal.Project.open(project_path)
        outputs = project.resume_operation(operation_id)
        assert set(outputs) == {"sentences", "tokens"}
        assert all(artifact.status == "complete" for artifact in outputs.values())
        assert outputs["sentences"].n_rows == 8
        assert outputs["tokens"].n_rows > 8

        import sqlite3

        plan_path = project.storage.operation_dir(operation_id) / "plan.sqlite"
        with sqlite3.connect(plan_path) as con:
            statuses = [
                row[0]
                for row in con.execute(
                    "SELECT status FROM plan_units ORDER BY unit_index"
                ).fetchall()
            ]
        assert statuses and set(statuses) == {"complete"}
    finally:
        project.close()


def test_real_dask_spacy_failure_then_parallel_resume_after_model_becomes_available(
    tmp_path: Path,
) -> None:
    spacy = _external_modules()
    pytest.importorskip("dask.distributed")

    missing_model_dir = tmp_path / "model_available_after_failure"
    rows = pd.DataFrame(
        {
            "document_id": list(range(1, 7)),
            "text": [f"Document {idx}. Another sentence." for idx in range(1, 7)],
        }
    )
    project_path = tmp_path / "parallel_resume_project"
    project = teal.Project.create(project_path, name="spacy_parallel_resume")
    try:
        source = _seed_documents(project, rows)
        with pytest.raises(Exception, match="Could not load spaCy pipeline"):
            project.translate(
                SpacyTranslator(model=str(missing_model_dir), spacy_batch_size=2),
                source,
                workers=2,
                batch_size=2,
                max_outstanding_units=2,
            )

        operation = project.catalog.list_operations()[-1]
        operation_id = str(operation["operation_id"])
        assert operation["status"] == "failed"

        # The operation configuration is unchanged. Make the originally requested
        # pipeline path available, then reopen and resume the same parallel plan.
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.to_disk(missing_model_dir)

        project.close()
        project = teal.Project.open(project_path)
        outputs = project.resume_operation(operation_id)
        assert set(outputs) == {"sentences", "tokens"}
        assert all(artifact.status == "complete" for artifact in outputs.values())
        assert outputs["sentences"].n_rows == 12
        assert outputs["tokens"].n_rows > 12

        import sqlite3

        plan_path = project.storage.operation_dir(operation_id) / "plan.sqlite"
        with sqlite3.connect(plan_path) as con:
            statuses = [
                row[0]
                for row in con.execute(
                    "SELECT status FROM plan_units ORDER BY unit_index"
                ).fetchall()
            ]
        assert statuses and set(statuses) == {"complete"}
    finally:
        project.close()
