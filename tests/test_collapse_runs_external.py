from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab import agg, concat
from text_analysis_lab.core.writer import create_artifact_writer


def _register_table(
    project: teal.Project,
    *,
    artifact_id: str,
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
        label=artifact_id,
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
        label=artifact_id,
        lineage_mode=lineage_mode,
        status="complete",
        basis_artifact_ids=basis_artifact_ids,
    )
    return project.get_artifact(artifact_id)


def _frame(artifact, *, metadata_mode="full"):
    return artifact.query(
        key_columns=True,
        data_columns=True,
        metadata_columns=True,
        metadata_mode=metadata_mode,
        include_position=True,
        order_by="_position",
        form="table",
    )


def test_collapse_runs_gapped_keys_aggregates_data_and_metadata(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_gapped")
    try:
        source = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 0, 0, 0],
                    "turn_id": [1, 3, 7, 8, 12],
                }
            ),
            data=pd.DataFrame(
                {
                    "text": ["a", "b", "c", "d", "e"],
                    "score": [1, 2, 3, 4, 5],
                }
            ),
            metadata=pd.DataFrame(
                {
                    "speaker_role": [
                        "child",
                        "child",
                        "interlocutor",
                        "interlocutor",
                        "child",
                    ],
                    "note": ["n1", "n2", "n3", "n4", "n5"],
                }
            ),
        )

        collapsed = project.collapse_runs(
            source,
            by="speaker_role",
            data={
                "text": concat(" "),
                "score_sum": agg("score", "sum"),
            },
            metadata={"notes": agg("note", concat("|"))},
            batch_size=1,
            output_label="collapsed",
        )
        frame = _frame(collapsed)

        assert collapsed.primary_key == ["child_id", "turn_id_start", "turn_id_end"]
        assert collapsed.descriptor["lineage"] == {
            "lineage_mode": "span_key",
            "basis_artifact_ids": [source.artifact_id],
        }
        assert frame[["turn_id_start", "turn_id_end"]].astype(int).values.tolist() == [
            [1, 3],
            [7, 8],
            [12, 12],
        ]
        assert frame["speaker_role"].tolist() == ["child", "interlocutor", "child"]
        assert frame["text"].tolist() == ["a b", "c d", "e"]
        assert frame["score_sum"].tolist() == pytest.approx([3, 7, 5])
        assert frame["notes"].tolist() == ["n1|n2", "n3|n4", "n5"]
        assert frame["n_rows"].astype(int).tolist() == [2, 2, 1]

        # Unaggregated leaf metadata does not bubble across span_key.
        assert "note" not in collapsed.get_full_metadata_columns()
        assert {"speaker_role", "notes", "n_rows"}.issubset(
            set(collapsed.get_full_metadata_columns())
        )
    finally:
        project.close()


def test_collapse_runs_permits_nonmonotonic_keys_and_uses_leaf_range_semantics(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_nonmonotonic")
    try:
        source = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame({"turn_id": [0, 2, 1, 3]}),
            data=pd.DataFrame({"text": ["a", "b", "c", "d"]}),
            metadata=pd.DataFrame({"speaker": ["A", "A", "B", "B"]}),
        )
        collapsed = project.collapse_runs(
            source,
            by="speaker",
            data={"text": concat(" ")},
        )
        frame = _frame(collapsed, metadata_mode="local")

        # No monotonicity validation: adjacent A rows generate [0,2], adjacent B rows [1,3].
        # Data aggregation follows the resulting closed leaf ranges, so overlapping spans are
        # intentionally possible on a reordered source.
        assert frame[["turn_id_start", "turn_id_end"]].astype(int).values.tolist() == [
            [0, 2],
            [1, 3],
        ]
        assert frame["speaker"].tolist() == ["A", "B"]
        assert frame["text"].tolist() == ["a b c", "b c d"]
        assert frame["n_rows"].astype(int).tolist() == [3, 3]
    finally:
        project.close()


def test_collapse_runs_span_resolves_only_against_immediate_subset_basis(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_subset_basis")
    try:
        source = _register_table(
            project,
            artifact_id="source",
            keys=pd.DataFrame({"turn_id": [0, 1, 2]}),
            data=pd.DataFrame({"text": ["zero", "one", "two"]}),
            metadata=pd.DataFrame({"speaker": ["A", "A", "A"]}),
        )
        cleaned = project.select_keys(source, [0, 2], output_label="cleaned")
        collapsed = project.collapse_runs(
            cleaned,
            by="speaker",
            data={"text": concat(" ")},
            output_label="collapsed",
        )
        frame = _frame(collapsed, metadata_mode="local")
        assert frame[["turn_id_start", "turn_id_end"]].astype(int).values.tolist() == [
            [0, 2]
        ]
        assert frame["text"].tolist() == ["zero two"]
        assert frame["n_rows"].astype(int).tolist() == [2]
    finally:
        project.close()


def test_collapse_runs_coarser_metadata_bubbles_but_leaf_metadata_requires_aggregation(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_lineage")
    try:
        docs = _register_table(
            project,
            artifact_id="children",
            keys=pd.DataFrame({"child_id": [0, 1]}),
            metadata=pd.DataFrame({"cohort": ["x", "y"]}),
        )
        turns = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame(
                {
                    "child_id": [0, 0, 0, 1, 1],
                    "turn_id": [0, 1, 2, 0, 1],
                }
            ),
            data=pd.DataFrame({"text": ["a", "b", "c", "d", "e"]}),
            metadata=pd.DataFrame(
                {
                    "speaker": ["A", "A", "B", "A", "A"],
                    "leaf_note": ["0", "1", "2", "3", "4"],
                }
            ),
            lineage_mode="extended_key",
            basis_artifact_ids=(docs.artifact_id,),
        )
        collapsed = project.collapse_runs(
            turns,
            by="speaker",
            data={"text": concat(" ")},
            metadata={"note_joined": agg("leaf_note", concat(""))},
        )
        frame = _frame(collapsed, metadata_mode="full")

        assert frame["cohort"].tolist() == ["x", "x", "y"]
        assert "leaf_note" not in collapsed.get_full_metadata_columns()
        assert frame["note_joined"].tolist() == ["01", "2", "34"]
    finally:
        project.close()


def test_collapse_runs_context_navigation_is_positional_and_atomic_context_uses_span_range(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_context")
    try:
        source = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame({"turn_id": [1, 3, 7, 8, 12]}),
            data=pd.DataFrame({"text": ["a", "b", "c", "d", "e"]}),
            metadata=pd.DataFrame({"speaker": ["A", "A", "B", "B", "A"]}),
        )
        collapsed = project.collapse_runs(
            source,
            by="speaker",
            data={"text": concat(" ")},
        )
        middle = {"turn_id_start": 7, "turn_id_end": 8}

        previous = collapsed.get_previous(middle, n=1, data_columns=False)
        following = collapsed.get_next(middle, n=1, data_columns=False)
        assert previous[["turn_id_start", "turn_id_end"]].astype(
            int
        ).values.tolist() == [[1, 3]]
        assert following[["turn_id_start", "turn_id_end"]].astype(
            int
        ).values.tolist() == [[12, 12]]

        atomic = collapsed.get_context(
            middle,
            before=0,
            after=0,
            context_artifact=source,
            include_focus=True,
            metadata_columns=False,
        )
        assert atomic["turn_id"].astype(int).tolist() == [7, 8]
        assert atomic["text"].tolist() == ["c", "d"]
    finally:
        project.close()


def test_collapse_runs_multiple_by_fields_and_close_reopen(tmp_path: Path) -> None:
    root = tmp_path / "project"
    project = teal.Project.create(root, name="collapse_reopen")
    source = _register_table(
        project,
        artifact_id="turns",
        keys=pd.DataFrame({"turn_id": [0, 1, 2, 3]}),
        data=pd.DataFrame({"text": ["a", "b", "c", "d"]}),
        metadata=pd.DataFrame(
            {
                "speaker": ["A", "A", "A", "B"],
                "mode": ["x", "x", "y", "y"],
            }
        ),
    )
    collapsed = project.collapse_runs(
        source,
        by=["speaker", "mode"],
        data={"text": concat(" ")},
        output_label="collapsed",
    )
    artifact_id = collapsed.artifact_id
    project.close()

    reopened = teal.Project.open(root)
    try:
        restored = reopened.get_artifact(artifact_id)
        frame = _frame(restored, metadata_mode="local")
        assert frame[["turn_id_start", "turn_id_end"]].astype(int).values.tolist() == [
            [0, 1],
            [2, 2],
            [3, 3],
        ]
        assert frame["speaker"].tolist() == ["A", "A", "B"]
        assert frame["mode"].tolist() == ["x", "y", "y"]
        assert frame["text"].tolist() == ["a b", "c", "d"]
    finally:
        reopened.close()


def test_collapse_runs_empty_artifact_preserves_span_schema(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_empty")
    try:
        source = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame(
                {
                    "child_id": pd.Series(dtype="int64"),
                    "turn_id": pd.Series(dtype="int64"),
                }
            ),
            data=pd.DataFrame({"text": pd.Series(dtype="string")}),
            metadata=pd.DataFrame({"speaker": pd.Series(dtype="string")}),
        )
        collapsed = project.collapse_runs(
            source,
            by="speaker",
            data={"text": concat(" ")},
        )
        assert collapsed.n_rows == 0
        assert collapsed.primary_key == ["child_id", "turn_id_start", "turn_id_end"]
        assert set(collapsed.get_metadata_columns()) == {"speaker", "n_rows"}
    finally:
        project.close()


def test_collapse_runs_rejects_invalid_by_and_duplicate_local_output(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="collapse_validation")
    try:
        source = _register_table(
            project,
            artifact_id="turns",
            keys=pd.DataFrame({"turn_id": [0, 1]}),
            data=pd.DataFrame({"text": ["a", "b"]}),
            metadata=pd.DataFrame({"speaker": ["A", "A"]}),
        )
        with pytest.raises(ValueError, match="at least one"):
            project.collapse_runs(source, by=[])
        with pytest.raises(Exception, match="not available"):
            project.collapse_runs(source, by="missing")
        with pytest.raises(Exception, match="carried automatically"):
            project.collapse_runs(
                source,
                by="speaker",
                metadata={"speaker": "first"},
            )
    finally:
        project.close()
