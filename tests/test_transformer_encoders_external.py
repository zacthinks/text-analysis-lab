from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

import text_analysis_lab as teal
from text_analysis_lab.translators import ContextualTransformer, SentenceTransformerEncoder


def _enabled() -> bool:
    return os.environ.get("TEAL_RUN_HF_EXTERNAL", "").strip().lower() in {"1", "true", "yes"}


def _seed_documents(project: teal.Project):
    from text_analysis_lab.core.writer import create_artifact_writer

    rows = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": [
                "A short document about public policy.",
                "Another short document about schools and teachers.",
                "A final sentence about elections.",
            ],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_hf_docs"),
        artifact_id="art_hf_docs",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["doc_id"]], "data": rows[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_hf_docs",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_hf_docs")


@pytest.mark.skipif(not _enabled(), reason="set TEAL_RUN_HF_EXTERNAL=1 to allow real model downloads")
def test_real_sentence_and_contextual_transformers_saved_offline(tmp_path: Path, monkeypatch) -> None:
    model_id = os.environ.get(
        "TEAL_HF_TEST_MODEL", "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    )
    project_path = tmp_path / "hf_project"
    project = teal.Project.create(project_path, name="hf_project")
    try:
        source = _seed_documents(project)
        sentence = project.translate(
            SentenceTransformerEncoder(
                model_id,
                normalize=True,
                truncation="error",
                save_model=True,
            ),
            source,
            batch_size=2,
            device="cpu",
            model_batch_size=2,
        )["output"]
        sentence_values = sentence.get_matrix()
        assert sentence_values.shape[0] == 3
        norms = (sentence_values**2).sum(axis=1) ** 0.5
        assert norms.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)

        contextual_outputs = project.translate(
            ContextualTransformer(model_id, truncation="error", save_model=True),
            source,
            batch_size=2,
            device="cpu",
            model_batch_size=2,
        )
        tokens = contextual_outputs["tokens"]
        contextual = contextual_outputs["contextual_embeddings"]
        assert contextual.get_matrix().shape[0] == len(tokens.query(form="table"))
        assert contextual.primary_key == tokens.primary_key
        assert contextual.descriptor["lineage"]["basis_artifact_ids"] == [tokens.artifact_id]

        sentence_operator_id = str(project.catalog.get_operation(sentence.operation_id)["operator_id"])
        contextual_operator_id = str(project.catalog.get_operation(contextual.operation_id)["operator_id"])
        assert (project.storage.operator_dir(sentence_operator_id) / "assets" / "model").is_dir()
        assert (project.storage.operator_dir(contextual_operator_id) / "assets" / "model").is_dir()
    finally:
        project.close()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    reopened = teal.Project.open(project_path)
    try:
        reopened.get_operator(sentence_operator_id).download()
        reopened.get_operator(contextual_operator_id).download()
    finally:
        reopened.close()
