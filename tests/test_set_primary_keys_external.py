from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError, QueryError
from text_analysis_lab.core.writer import create_artifact_writer


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


def _register_sparse(
    project: teal.Project,
    *,
    artifact_id: str,
    keys: pd.DataFrame,
    values,
    columns: list[str],
    metadata: pd.DataFrame | None = None,
    lineage_mode: str = "new_key",
    basis_artifact_ids: tuple[str, ...] = (),
):
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode=lineage_mode,
        basis_artifact_ids=basis_artifact_ids,
    )
    payload: dict[str, object] = {"keys": keys.reset_index(drop=True)}
    if values is not None:
        payload["data"] = {"values": sparse.csr_matrix(values), "columns": columns}
    if metadata is not None:
        payload["metadata"] = metadata.reset_index(drop=True)
    writer.write(payload)
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix",
        label=artifact_id,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=basis_artifact_ids,
    )
    return project.get_artifact(artifact_id)


def _source_table(project: teal.Project):
    return _register_table(
        project,
        artifact_id="art_source",
        label="source",
        keys=pd.DataFrame({"row_id": list(range(8))}),
        data=pd.DataFrame(
            {
                "text": [f"text-{i}" for i in range(8)],
                "score": [10 + i for i in range(8)],
            }
        ),
        metadata=pd.DataFrame(
            {
                "child": ["B", "A", "A", "A", "B", "B", "A", "A"],
                "book": ["Road", "Ocean", "Ocean", "Ocean", "Ocean", "Road", "Road", "Road"],
                "partner": ["robot", "robot", "robot", "parent", "parent", "parent", "robot", "robot"],
                "speaker": ["child", "child", "interlocutor", "child", "child", "interlocutor", "child", "interlocutor"],
            }
        ),
    )


def _full(artifact):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=True,
        metadata_mode="full",
        include_position=True,
        order_by="_position",
        form="table",
    )


def test_basic_rekey_is_keys_only_local_integer_hierarchy_and_inherits_everything(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="rekey_basic")
    try:
        source = _source_table(project)
        out = project.set_primary_keys(
            source,
            levels={
                "child": "child_id",
                "book": "book_id",
                "partner": "partner_id",
            },
            leaf_key="turn_id",
            output_label="turns",
            batch_size=3,
        )

        assert out.primary_key == ["child_id", "book_id", "partner_id", "turn_id"]
        assert out.descriptor["lineage"] == {
            "lineage_mode": "rekeyed_key",
            "basis_artifact_ids": [source.artifact_id],
        }
        assert set(out.components) == {"keys"}
        assert out.n_rows == source.n_rows == 8

        frame = _full(out)
        assert frame["_position"].tolist() == list(range(8))
        assert frame["text"].tolist() == [f"text-{i}" for i in range(8)]
        assert frame["child"].tolist() == ["B", "A", "A", "A", "B", "B", "A", "A"]

        # Child is sorted A=0, B=1; subordinate levels reset within parent groups.
        assert frame["child_id"].astype(int).tolist() == [1, 0, 0, 0, 1, 1, 0, 0]
        # A: Ocean=0, Road=1. B: Ocean=0, Road=1.
        assert frame["book_id"].astype(int).tolist() == [1, 0, 0, 0, 0, 1, 1, 1]
        # A/Ocean has parent=0, robot=1; A/Road only robot=0.
        # B/Ocean only parent=0; B/Road parent=0, robot=1.
        assert frame["partner_id"].astype(int).tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
        # Only rows 1 and 2 share the same deepest group.
        assert frame["turn_id"].astype(int).tolist() == [0, 0, 1, 0, 0, 0, 0, 1]
    finally:
        project.close()


def test_subset_before_and_after_rekey_keeps_only_relevant_source_rows(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="rekey_subset")
    try:
        source = _source_table(project)
        before = project.select_keys(source, [6, 1, 3, 5], output_label="before")
        rekeyed = project.set_primary_keys(
            before,
            levels={"child": "child_id", "book": "book_id"},
            leaf_key="turn_id",
            output_label="rekeyed",
        )
        frame = _full(rekeyed)
        # select_keys returns source order, not request order.
        assert frame["text"].tolist() == ["text-1", "text-3", "text-5", "text-6"]
        assert frame["child"].tolist() == ["A", "A", "B", "A"]

        keys = frame.loc[[0, 3], rekeyed.primary_key].astype(int).to_dict("records")
        after = project.select_keys(rekeyed, keys, output_label="after")
        final = _full(after)
        assert final["text"].tolist() == ["text-1", "text-6"]
        assert final["child"].tolist() == ["A", "A"]
        assert final["_position"].tolist() == [0, 1]
    finally:
        project.close()


def test_two_rekeys_with_subset_between_translate_back_to_original_data_and_metadata(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="double_rekey")
    try:
        source = _source_table(project)
        first = project.set_primary_keys(
            source,
            levels={"child": "child_id", "book": "book_id"},
            leaf_key="turn_id",
            output_label="first",
        )
        first_frame = _full(first)
        chosen = first_frame.loc[[1, 2, 4, 7], first.primary_key].astype(int).to_dict("records")
        middle = project.select_keys(first, chosen, output_label="middle")

        second = project.set_primary_keys(
            middle,
            levels={"partner": "condition_id", "speaker": "speaker_id"},
            leaf_key="item_id",
            output_label="second",
        )
        second_frame = _full(second)
        assert second_frame["text"].tolist() == ["text-1", "text-2", "text-4", "text-7"]
        assert second_frame["score"].astype(int).tolist() == [11, 12, 14, 17]
        assert second_frame["child"].tolist() == ["A", "A", "B", "A"]

        last_keys = second_frame.loc[[1, 3], second.primary_key].astype(int).to_dict("records")
        final = project.select_keys(second, last_keys, output_label="final")
        final_frame = _full(final)
        assert final_frame["text"].tolist() == ["text-2", "text-7"]
        assert final_frame["book"].tolist() == ["Ocean", "Road"]
    finally:
        project.close()


def test_extended_key_before_rekey_repeats_ancestor_metadata_correctly(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="extended_before")
    try:
        docs = _register_table(
            project,
            artifact_id="art_docs",
            label="docs",
            keys=pd.DataFrame({"doc_id": [0, 1]}),
            metadata=pd.DataFrame({"child": ["B", "A"], "cohort": ["x", "y"]}),
        )
        sentences = _register_table(
            project,
            artifact_id="art_sentences",
            label="sentences",
            keys=pd.DataFrame(
                {"doc_id": [0, 0, 1, 1, 1], "sentence_id": [0, 1, 0, 1, 2]}
            ),
            lineage_mode="extended_key",
            basis_artifact_ids=(docs.artifact_id,),
        )
        rekeyed = project.set_primary_keys(
            sentences,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="rekeyed",
        )
        frame = _full(rekeyed)
        assert frame["child"].tolist() == ["B", "B", "A", "A", "A"]
        assert frame["cohort"].tolist() == ["x", "x", "y", "y", "y"]
        assert frame["child_id"].astype(int).tolist() == [1, 1, 0, 0, 0]
        assert frame["turn_id"].astype(int).tolist() == [0, 1, 0, 1, 2]
    finally:
        project.close()


def test_extended_key_after_rekey_maps_pre_rekey_metadata_to_each_extended_row(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="extended_after")
    try:
        source = _register_table(
            project,
            artifact_id="art_base",
            label="base",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            metadata=pd.DataFrame({"child": ["A", "B", "C"], "origin": ["aa", "bb", "cc"]}),
        )
        rekeyed = project.set_primary_keys(
            source,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="rekeyed",
        )
        extended = _register_table(
            project,
            artifact_id="art_extended",
            label="extended",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 1, 2, 2],
                    "turn_id": [0, 0, 0, 0, 0],
                    "piece_id": [0, 1, 0, 0, 1],
                }
            ),
            lineage_mode="extended_key",
            basis_artifact_ids=(rekeyed.artifact_id,),
        )
        frame = extended.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=True,
            metadata_mode="full",
            order_by="_position",
            form="table",
        )
        assert frame["child"].tolist() == ["A", "A", "B", "C", "C"]
        assert frame["origin"].tolist() == ["aa", "aa", "bb", "cc", "cc"]
    finally:
        project.close()


def test_reduced_key_after_rekey_does_not_guess_cross_keyspace_metadata(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="reduced_boundary")
    try:
        source = _register_table(
            project,
            artifact_id="art_base",
            label="base",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            metadata=pd.DataFrame({"child": ["A", "A", "B"], "old_meta": [1, 2, 3]}),
        )
        rekeyed = project.set_primary_keys(
            source,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="rekeyed",
        )
        reduced = _register_table(
            project,
            artifact_id="art_reduced",
            label="reduced",
            keys=pd.DataFrame({"child_id": [0, 1]}),
            lineage_mode="reduced_key",
            basis_artifact_ids=(rekeyed.artifact_id,),
        )
        columns = reduced.query_columns(metadata_mode="full")
        assert "old_meta" not in columns["metadata"]
        assert "child" not in columns["metadata"]
    finally:
        project.close()


def test_rekeyed_sparse_matrix_inherits_values_and_feature_axis_then_feature_subsets(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="matrix_rekey")
    try:
        values = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 1, 1]], dtype=float)
        source = _register_sparse(
            project,
            artifact_id="art_matrix",
            keys=pd.DataFrame({"row_id": [0, 1, 2, 3]}),
            values=values,
            columns=["a", "b", "c"],
            metadata=pd.DataFrame({"child": ["B", "A", "B", "A"]}),
        )
        rekeyed = project.set_primary_keys(
            source,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="matrix_rekeyed",
        )
        assert rekeyed.get_data_columns() == ["a", "b", "c"]
        np.testing.assert_array_equal(rekeyed.get_matrix().toarray(), values)

        selected = project.feature_subset(
            rekeyed,
            lambda f: f["column"].isin(["a", "c"]),
            output_label="selected",
        )
        assert selected.get_data_columns() == ["a", "c"]
        np.testing.assert_array_equal(selected.get_matrix().toarray(), values[:, [0, 2]])
    finally:
        project.close()


def test_joined_virtual_table_data_survives_rekey_without_materialization(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="joined_rekey")
    try:
        basis = _register_table(
            project,
            artifact_id="art_basis",
            label="basis",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            data=pd.DataFrame({"text": ["a", "b", "c"]}),
            metadata=pd.DataFrame({"child": ["B", "A", "A"]}),
        )
        scores = _register_table(
            project,
            artifact_id="art_scores",
            label="scores",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            data=pd.DataFrame({"score": [0.1, 0.2, 0.3]}),
        )
        joined = project.join(basis, scores, output_label="joined")
        rekeyed = project.set_primary_keys(
            joined,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="joined_rekeyed",
        )
        frame = _full(rekeyed)
        assert frame["text"].tolist() == ["a", "b", "c"]
        assert frame["score"].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert frame["child"].tolist() == ["B", "A", "A"]
    finally:
        project.close()


def test_rekey_survives_project_reopen_and_operator_state_is_serialized(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="reopen")
    try:
        source = _source_table(project)
        out = project.set_primary_keys(
            source,
            levels={"child": "child_id", "book": "book_id"},
            leaf_key="turn_id",
            output_label="turns",
        )
        artifact_id = out.artifact_id
        operation = project.operation_for_artifact(out)
        assert operation is not None
        assert operation["operation_type"] == "rekey"
        operator_id = operation["operator_id"]
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        out = reopened.get_artifact(artifact_id)
        frame = _full(out)
        assert frame["text"].tolist() == [f"text-{i}" for i in range(8)]
        operator = reopened.get_operator(operator_id)
        assert operator.levels == {"child": "child_id", "book": "book_id"}
        assert operator.leaf_key == "turn_id"
    finally:
        reopened.close()


def test_validation_rejects_collisions_nulls_duplicate_names_and_ambiguous_source_fields(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="validation")
    try:
        source = _register_table(
            project,
            artifact_id="art_source_validation",
            label="source",
            keys=pd.DataFrame({"row_id": [0, 1]}),
            data=pd.DataFrame({"child": ["data-a", "data-b"], "x": [1, 2]}),
            metadata=pd.DataFrame({"child": ["meta-a", "meta-b"], "group": ["A", None]}),
        )
        with pytest.raises(ArtifactError, match="ambiguous"):
            project.set_primary_keys(
                source,
                levels={"child": "child_id"},
                leaf_key="turn_id",
            )
        with pytest.raises(ArtifactError, match="contains null"):
            project.set_primary_keys(
                source,
                levels={"group": "group_id"},
                leaf_key="turn_id",
            )
        with pytest.raises(ArtifactError, match="must not collide"):
            project.set_primary_keys(
                source,
                levels={"data.child": "x"},
                leaf_key="turn_id",
            )
        with pytest.raises(ArtifactError, match="must be unique"):
            project.set_primary_keys(
                source,
                levels={"data.child": "turn_id"},
                leaf_key="turn_id",
            )
    finally:
        project.close()


def test_empty_artifact_can_be_rekeyed_without_creating_data_or_metadata(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="empty")
    try:
        source = _register_table(
            project,
            artifact_id="art_empty",
            label="empty",
            keys=pd.DataFrame({"row_id": pd.Series(dtype="int64")}),
            data=pd.DataFrame({"text": pd.Series(dtype="object")}),
            metadata=pd.DataFrame({"child": pd.Series(dtype="object")}),
        )
        out = project.set_primary_keys(
            source,
            levels={"child": "child_id"},
            leaf_key="turn_id",
            output_label="empty_rekeyed",
        )
        assert out.n_rows == 0
        assert out.primary_key == ["child_id", "turn_id"]
        assert set(out.components) == {"keys"}
    finally:
        project.close()


def test_malformed_rekey_row_count_is_rejected_on_cross_boundary_query(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="malformed")
    try:
        source = _register_table(
            project,
            artifact_id="art_base_malformed",
            label="base",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            metadata=pd.DataFrame({"meta": ["a", "b", "c"]}),
        )
        malformed = _register_table(
            project,
            artifact_id="art_bad_rekey",
            label="bad",
            keys=pd.DataFrame({"new_id": [0, 1]}),
            lineage_mode="rekeyed_key",
            basis_artifact_ids=(source.artifact_id,),
        )
        with pytest.raises(QueryError, match="row counts differ"):
            malformed.query(
                key_columns=True,
                data_columns=False,
                metadata_columns=True,
                metadata_mode="full",
                form="table",
            )
    finally:
        project.close()


def test_rekey_lineage_uses_position_even_when_old_and_new_key_names_happen_to_match(tmp_path: Path) -> None:
    """Never infer alignment from coincidentally available key columns."""
    project = teal.Project.create(tmp_path / "project", name="same_named_keys")
    try:
        source = _register_table(
            project,
            artifact_id="art_same_name_source",
            label="source",
            keys=pd.DataFrame({"row_id": [0, 1, 2]}),
            data=pd.DataFrame({"text": ["zero", "one", "two"]}),
            metadata=pd.DataFrame({"meta": ["a", "b", "c"]}),
        )
        # This is a valid structural rekey: row i remains row i, but the new key
        # namespace deliberately reuses the old *name* with unrelated values.
        rekeyed = _register_table(
            project,
            artifact_id="art_same_name_rekey",
            label="rekeyed",
            keys=pd.DataFrame({"row_id": [2, 0, 1]}),
            lineage_mode="rekeyed_key",
            basis_artifact_ids=(source.artifact_id,),
        )
        frame = _full(rekeyed)
        assert frame["row_id"].astype(int).tolist() == [2, 0, 1]
        # Position mapping must win. A key join would incorrectly produce
        # ["two", "zero", "one"] / ["c", "a", "b"].
        assert frame["text"].tolist() == ["zero", "one", "two"]
        assert frame["meta"].tolist() == ["a", "b", "c"]
    finally:
        project.close()


def test_full_metadata_can_mix_sources_from_before_between_and_after_two_rekeys(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="metadata_depths")
    try:
        source = _register_table(
            project,
            artifact_id="art_meta_a",
            label="a",
            keys=pd.DataFrame({"row_id": [0, 1, 2, 3]}),
            metadata=pd.DataFrame(
                {
                    "group1": ["B", "A", "A", "B"],
                    "group2": ["X", "X", "Y", "Y"],
                    "old_meta": ["a0", "a1", "a2", "a3"],
                }
            ),
        )
        b1 = project.set_primary_keys(
            source,
            levels={"group1": "group1_id"},
            leaf_key="b_row_id",
            output_label="b1",
        )
        b1_keys = b1.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        b2 = _register_table(
            project,
            artifact_id="art_meta_b2",
            label="b2",
            keys=b1_keys.loc[:, b1.primary_key],
            metadata=pd.DataFrame({"middle_meta": ["b0", "b1", "b2", "b3"]}),
            lineage_mode="preserved_key",
            basis_artifact_ids=(b1.artifact_id,),
        )
        c1 = project.set_primary_keys(
            b2,
            levels={"group2": "group2_id"},
            leaf_key="c_row_id",
            output_label="c1",
        )
        c1_keys = c1.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            order_by="_position",
            form="table",
        )
        c2 = _register_table(
            project,
            artifact_id="art_meta_c2",
            label="c2",
            keys=c1_keys.loc[:, c1.primary_key],
            metadata=pd.DataFrame({"new_meta": ["c0", "c1", "c2", "c3"]}),
            lineage_mode="preserved_key",
            basis_artifact_ids=(c1.artifact_id,),
        )

        frame = c2.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=True,
            metadata_mode="full",
            order_by="_position",
            form="table",
        )
        assert frame["old_meta"].tolist() == ["a0", "a1", "a2", "a3"]
        assert frame["middle_meta"].tolist() == ["b0", "b1", "b2", "b3"]
        assert frame["new_meta"].tolist() == ["c0", "c1", "c2", "c3"]
    finally:
        project.close()


def test_merged_virtual_branches_can_be_rekeyed_and_still_resolve_branch_data(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="merge_rekey")
    try:
        source = _source_table(project)
        left = project.select_keys(source, [0, 2, 4, 6], output_label="left")
        right = project.select_keys(source, [1, 3, 5, 7], output_label="right")
        merged = project.merge([left, right], output_label="merged")
        rekeyed = project.set_primary_keys(
            merged,
            levels={"child": "child_id", "book": "book_id"},
            leaf_key="turn_id",
            output_label="merged_rekeyed",
        )
        frame = _full(rekeyed)
        # Merge preserves source-list order, then each branch's source order.
        assert frame["text"].tolist() == [
            "text-0", "text-2", "text-4", "text-6",
            "text-1", "text-3", "text-5", "text-7",
        ]
        assert frame["score"].astype(int).tolist() == [10, 12, 14, 16, 11, 13, 15, 17]
        assert frame["child"].tolist() == ["B", "A", "B", "A", "A", "A", "B", "A"]
    finally:
        project.close()
