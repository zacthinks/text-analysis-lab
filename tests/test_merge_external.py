from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError, DuplicatePrimaryKeyError
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import RegexCleaner, RegexReplaceRule


def _register_table(
    project: teal.Project,
    *,
    artifact_id: str,
    label: str,
    keys: pd.DataFrame,
    data: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    lineage_mode: str = "new_key",
    basis_artifact_ids: tuple[str, ...] = (),
):
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode=lineage_mode,
        basis_artifact_ids=basis_artifact_ids,
    )
    payload: dict[str, object] = {"keys": keys.reset_index(drop=True)}
    if data is not None:
        payload["data"] = data.reset_index(drop=True)
    if metadata is not None:
        payload["metadata"] = metadata.reset_index(drop=True)
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


def _build_disjoint_extracted_branches(project: teal.Project):
    root = _register_table(
        project,
        artifact_id="art_root",
        label="inventory",
        keys=pd.DataFrame({"file_id": [0, 1, 2, 3, 4, 5]}),
        data=pd.DataFrame(
            {
                "path": [f"/tmp/{i}.dat" for i in range(6)],
                "extension": [".pdf", ".txt", ".pdf", ".txt", ".pdf", ".txt"],
            }
        ),
        metadata=pd.DataFrame(
            {"speaker": ["a", "b", "c", "d", "e", "f"]}
        ),
    )
    pdf = _register_table(
        project,
        artifact_id="art_pdf",
        label="pdf_text",
        keys=pd.DataFrame({"file_id": [0, 2, 4]}),
        data=pd.DataFrame(
            {
                "text": ["pdf-0", "pdf-2", "pdf-4"],
                "extraction_status": ["ok", "ok", "ok"],
            }
        ),
        lineage_mode="preserved_key",
        basis_artifact_ids=(root.artifact_id,),
    )
    txt = _register_table(
        project,
        artifact_id="art_txt",
        label="txt_text",
        keys=pd.DataFrame({"file_id": [1, 3, 5]}),
        data=pd.DataFrame(
            {
                "text": ["txt-1", "txt-3", "txt-5"],
                "extraction_status": ["ok", "ok", "ok"],
            }
        ),
        lineage_mode="preserved_key",
        basis_artifact_ids=(root.artifact_id,),
    )
    return root, pdf, txt


def test_merge_is_keys_only_and_resolves_branch_data_and_metadata(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="merge_test")
    try:
        _, pdf, txt = _build_disjoint_extracted_branches(project)
        merged = project.merge([pdf, txt], batch_size=2, memo="recombine extracted files")

        assert merged.descriptor["lineage"] == {
            "lineage_mode": "merged_key",
            "basis_artifact_ids": [pdf.artifact_id, txt.artifact_id],
        }
        assert set(merged.components) == {"keys"}
        assert not merged.has_own_data()
        # merged_key is a virtual relational data provider for itself and any
        # later preserved-key descendants, despite owning no physical data.
        assert merged.data_artifact is merged

        frame = merged.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="full",
            form="table",
            include_position=True,
            order_by="_position",
        )
        assert frame["_position"].tolist() == list(range(6))
        assert frame["file_id"].astype(int).tolist() == [0, 2, 4, 1, 3, 5]
        assert frame["text"].tolist() == [
            "pdf-0",
            "pdf-2",
            "pdf-4",
            "txt-1",
            "txt-3",
            "txt-5",
        ]
        assert frame["speaker"].tolist() == ["a", "c", "e", "b", "d", "f"]

        local = merged.query_columns(metadata_mode="local")
        assert local["metadata"] == []
        full = merged.query_columns(metadata_mode="full")
        assert full["data"] == ["text", "extraction_status"]
        assert full["metadata"] == ["speaker"]

        # A normal Translator must be able to consume the merge itself.  This
        # is stronger than merely querying it: translate() must materialize
        # input batches from the merge's virtual branchwise data provider,
        # preserve the merged key order, and keep inherited metadata available
        # through the new preserved-key descendant.
        cleaned = project.translate(
            RegexCleaner(
                text_field="text",
                output_field="clean_text",
                rules=[RegexReplaceRule(r"-", " ")],
            ),
            merged,
            batch_size=2,
            workers=1,
        )["output"]
        assert cleaned.descriptor["lineage"] == {
            "lineage_mode": "preserved_key",
            "basis_artifact_ids": [merged.artifact_id],
        }
        cleaned_frame = cleaned.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="full",
            form="table",
            include_position=True,
            order_by="_position",
        )
        assert cleaned_frame["_position"].tolist() == list(range(6))
        assert cleaned_frame["file_id"].astype(int).tolist() == [0, 2, 4, 1, 3, 5]
        assert cleaned_frame["clean_text"].tolist() == [
            "pdf 0",
            "pdf 2",
            "pdf 4",
            "txt 1",
            "txt 3",
            "txt 5",
        ]
        assert cleaned_frame["speaker"].tolist() == ["a", "c", "e", "b", "d", "f"]

        # A later keys-only operation must inherit the merge's virtual data and
        # still traverse full metadata through the multi-basis boundary.
        later_split = project.split(
            merged,
            labels=("a", "b"),
            proportions=(0.5, 0.5),
            random_state=7,
        )
        expected_text = dict(zip(frame["file_id"], frame["text"], strict=True))
        expected_speaker = dict(zip(frame["file_id"], frame["speaker"], strict=True))
        for child in later_split.values():
            child_frame = child.query(
                key_columns=True,
                data_columns=True,
                metadata_columns=True,
                metadata_mode="full",
                form="table",
                order_by="_position",
            )
            for row in child_frame.itertuples(index=False):
                assert row.text == expected_text[row.file_id]
                assert row.speaker == expected_speaker[row.file_id]

        operation = project.operation_for_artifact(merged)
        assert operation is not None
        assert operation["operation_type"] == "merge"
        assert operation["status"] == "complete"
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        merged = reopened.get_artifact("art_000001")
        frame = merged.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="full",
            form="table",
            include_position=True,
            order_by="_position",
        )
        assert frame["file_id"].astype(int).tolist() == [0, 2, 4, 1, 3, 5]
        assert frame["speaker"].tolist() == ["a", "c", "e", "b", "d", "f"]

        # Persistence must be sufficient for normal downstream execution too,
        # not merely for querying the reopened merge.
        reopened_cleaned = reopened.translate(
            RegexCleaner(
                text_field="text",
                output_field="clean_text",
                rules=[RegexReplaceRule(r"-", " ")],
            ),
            merged,
            batch_size=3,
            workers=1,
        )["output"]
        reopened_cleaned_frame = reopened_cleaned.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="full",
            form="table",
            order_by="_position",
        )
        assert reopened_cleaned_frame["file_id"].astype(int).tolist() == [0, 2, 4, 1, 3, 5]
        assert reopened_cleaned_frame["clean_text"].tolist() == [
            "pdf 0",
            "pdf 2",
            "pdf 4",
            "txt 1",
            "txt 3",
            "txt 5",
        ]
        assert reopened_cleaned_frame["speaker"].tolist() == ["a", "c", "e", "b", "d", "f"]
    finally:
        reopened.close()


def test_merge_rejects_overlapping_primary_keys_and_marks_output_failed(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="merge_collision_test")
    try:
        left = _register_table(
            project,
            artifact_id="art_left",
            label="left",
            keys=pd.DataFrame({"file_id": [0, 1]}),
            data=pd.DataFrame({"text": ["a", "b"]}),
        )
        right = _register_table(
            project,
            artifact_id="art_right",
            label="right",
            keys=pd.DataFrame({"file_id": [1, 2]}),
            data=pd.DataFrame({"text": ["c", "d"]}),
        )

        with pytest.raises(DuplicatePrimaryKeyError):
            project.merge([left, right])

        merges = project.list_operations(operation_type="merge")
        assert len(merges) == 1
        assert merges[0]["status"] == "failed"
        outputs = project.operation_outputs(merges[0]["operation_id"])
        assert len(outputs) == 1
        failed = project.get_artifact(outputs[0]["artifact_id"])
        assert failed.status == "failed"
    finally:
        project.close()


def test_merge_validates_public_api_compatibility_before_creating_operation(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="merge_validation_test")
    try:
        left = _register_table(
            project,
            artifact_id="art_left",
            label="left",
            keys=pd.DataFrame({"file_id": [0, 1]}),
            data=pd.DataFrame({"text": ["a", "b"]}),
            metadata=pd.DataFrame({"speaker": ["x", "y"]}),
        )
        different_key = _register_table(
            project,
            artifact_id="art_different_key",
            label="different_key",
            keys=pd.DataFrame({"document_id": [2, 3]}),
            data=pd.DataFrame({"text": ["c", "d"]}),
            metadata=pd.DataFrame({"speaker": ["z", "w"]}),
        )
        different_data = _register_table(
            project,
            artifact_id="art_different_data",
            label="different_data",
            keys=pd.DataFrame({"file_id": [2, 3]}),
            data=pd.DataFrame({"body": ["c", "d"]}),
            metadata=pd.DataFrame({"speaker": ["z", "w"]}),
        )
        different_metadata = _register_table(
            project,
            artifact_id="art_different_metadata",
            label="different_metadata",
            keys=pd.DataFrame({"file_id": [4, 5]}),
            data=pd.DataFrame({"text": ["e", "f"]}),
            metadata=pd.DataFrame({"site": ["north", "south"]}),
        )

        with pytest.raises(ArtifactError, match="at least two"):
            project.merge([left])
        with pytest.raises(ArtifactError, match="same primary-key fields"):
            project.merge([left, different_key])
        with pytest.raises(ArtifactError, match="compatible effective data/full-metadata schemas"):
            project.merge([left, different_data])
        with pytest.raises(ArtifactError, match="compatible effective data/full-metadata schemas"):
            project.merge([left, different_metadata])
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            project.merge([left, different_data], batch_size=0)

        # Compatibility validation happens before durable operation/output IDs
        # are allocated.  A rejected shape/schema request must not leave failed
        # merge provenance behind.  (Key collisions are different: they are an
        # artifact-wide seal failure and therefore *do* leave a failed merge.)
        assert project.list_operations(operation_type="merge") == []
    finally:
        project.close()
