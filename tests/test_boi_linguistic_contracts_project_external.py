from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.linguistics.contracts import validate_linguistic_contracts
from text_analysis_lab.translators import SemanticRoleHeadResolver, SenseSelector
from tests._boi_linguistic_fixture import build_boi_linguistic_fixture


def _write_table(
    project,
    artifact_id: str,
    label: str,
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    lineage_mode: str = "new_key",
    basis=(),
):
    from text_analysis_lab.core.writer import create_artifact_writer

    data_columns = [column for column in frame.columns if column not in key_columns]
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=tuple(basis),
    )
    writer.write(
        {
            "keys": frame.loc[:, list(key_columns)].reset_index(drop=True),
            "data": frame.loc[:, data_columns].reset_index(drop=True),
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=label,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=tuple(basis),
    )
    return project.get_artifact(artifact_id)


def _frame(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        include_position=False,
        form="table",
    )


def test_boi_contract_fixture_round_trips_through_project(tmp_path: Path) -> None:
    fixture = build_boi_linguistic_fixture()
    project_path = tmp_path / "boi-contract-project"
    project = teal.Project.create(
        project_path,
        name="boi_contracts",
        delete_existing=True,
    )
    try:
        documents = _write_table(
            project,
            "art_docs",
            "documents",
            fixture.documents,
            ("row_id",),
        )
        sentences = _write_table(
            project,
            "art_sentences",
            "sentences",
            fixture.sentences,
            ("row_id", "sentence_id"),
            lineage_mode="extended_key",
            basis=(documents.artifact_id,),
        )
        tokens = _write_table(
            project,
            "art_tokens",
            "tokens",
            fixture.tokens,
            ("row_id", "sentence_id", "token_id"),
            lineage_mode="extended_key",
            basis=(sentences.artifact_id,),
        )
        predicates = _write_table(
            project,
            "art_predicates",
            "predicates",
            fixture.predicates,
            ("row_id", "sentence_id", "predicate_id"),
            lineage_mode="extended_key",
            basis=(sentences.artifact_id,),
        )
        role_spans = _write_table(
            project,
            "art_role_spans",
            "role_spans",
            fixture.role_spans,
            ("row_id", "sentence_id", "predicate_id", "role_id"),
            lineage_mode="extended_key",
            basis=(predicates.artifact_id,),
        )
        candidates = _write_table(
            project,
            "art_candidates",
            "candidates",
            fixture.candidates,
            ("row_id", "sentence_id", "token_id", "candidate_id"),
            lineage_mode="extended_key",
            basis=(tokens.artifact_id,),
        )
        unresolved = _write_table(
            project,
            "art_wsd_unresolved",
            "unresolved",
            fixture.wsd_unresolved,
            ("row_id", "sentence_id", "token_id"),
            lineage_mode="preserved_key",
            basis=(tokens.artifact_id,),
        )
        mentions = _write_table(
            project,
            "art_mentions",
            "mentions",
            fixture.mentions,
            ("row_id", "cluster_id", "mention_id"),
            lineage_mode="extended_key",
            basis=(documents.artifact_id,),
        )
        srl_failures = _write_table(
            project,
            "art_srl_failures",
            "srl_failures",
            fixture.srl_failures,
            ("row_id", "sentence_id"),
            lineage_mode="preserved_key",
            basis=(sentences.artifact_id,),
        )
        coref_failures = _write_table(
            project,
            "art_coref_failures",
            "coref_failures",
            fixture.coref_failures,
            ("row_id",),
            lineage_mode="preserved_key",
            basis=(documents.artifact_id,),
        )

        role_heads = project.translate(
            SemanticRoleHeadResolver(),
            {"role_spans": role_spans, "tokens": tokens},
        )["role_heads"]
        senses = project.translate(
            SenseSelector(),
            {"candidates": candidates},
        )["senses"]

        validate_linguistic_contracts(
            sentences=_frame(sentences),
            tokens=_frame(tokens),
            predicates=_frame(predicates),
            role_spans=_frame(role_spans),
            role_heads=_frame(role_heads),
            candidates=_frame(candidates),
            senses=_frame(senses),
            wsd_unresolved=_frame(unresolved),
            mentions=_frame(mentions),
            srl_failures=_frame(srl_failures),
            coref_failures=_frame(coref_failures),
        )

        assert role_heads.primary_key == (
            "row_id",
            "sentence_id",
            "predicate_id",
            "role_id",
            "head_id",
        )
        assert senses.primary_key == ("row_id", "sentence_id", "token_id")
        assert role_heads.descriptor["lineage"]["basis_artifact_ids"] == [
            role_spans.artifact_id
        ]
        assert senses.descriptor["lineage"]["basis_artifact_ids"] == [
            candidates.artifact_id
        ]

        role_head_id = role_heads.artifact_id
        sense_id = senses.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        validate_linguistic_contracts(
            sentences=_frame(reopened.get_artifact(sentences.artifact_id)),
            tokens=_frame(reopened.get_artifact(tokens.artifact_id)),
            predicates=_frame(reopened.get_artifact(predicates.artifact_id)),
            role_spans=_frame(reopened.get_artifact(role_spans.artifact_id)),
            role_heads=_frame(reopened.get_artifact(role_head_id)),
            candidates=_frame(reopened.get_artifact(candidates.artifact_id)),
            senses=_frame(reopened.get_artifact(sense_id)),
            wsd_unresolved=_frame(reopened.get_artifact(unresolved.artifact_id)),
            mentions=_frame(reopened.get_artifact(mentions.artifact_id)),
            srl_failures=_frame(reopened.get_artifact(srl_failures.artifact_id)),
            coref_failures=_frame(reopened.get_artifact(coref_failures.artifact_id)),
        )
    finally:
        reopened.close()


def test_head_resolution_accepts_structurally_compatible_different_lineage(
    tmp_path: Path,
) -> None:
    fixture = build_boi_linguistic_fixture()
    project = teal.Project.create(
        tmp_path / "boi-lineage-project",
        name="boi_lineage",
        delete_existing=True,
    )
    try:
        role_spans = _write_table(
            project,
            "art_role_spans_unrelated",
            "role_spans",
            fixture.role_spans,
            ("row_id", "sentence_id", "predicate_id", "role_id"),
        )
        tokens = _write_table(
            project,
            "art_tokens_unrelated",
            "tokens",
            fixture.tokens,
            ("row_id", "sentence_id", "token_id"),
        )

        result = project.translate(
            SemanticRoleHeadResolver(),
            {"role_spans": role_spans, "tokens": tokens},
        )["role_heads"]

        frame = _frame(result)
        assert not frame.empty
        assert set(frame["head_token_id"].astype(int)).issubset(
            set(fixture.tokens["token_id"].astype(int))
        )
    finally:
        project.close()
