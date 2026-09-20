from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError, LineageError
from text_analysis_lab.core.lineage import (
    iter_metadata_lineage_sources,
    validate_primary_key_relationship,
)
from text_analysis_lab.core.operator import validate_operation_type
from text_analysis_lab.core.set_primary_keys import _build_hierarchical_keys



def test_rekey_operation_type_is_registered() -> None:
    assert validate_operation_type("rekey") == "rekey"

def test_hierarchical_keys_are_sorted_locally_and_leaf_follows_row_order() -> None:
    frame = pd.DataFrame(
        {
            "child": ["B", "A", "A", "A", "B", "B"],
            "book": ["Z", "Y", "X", "X", "A", "Z"],
            "partner": ["robot", "robot", "robot", "parent", "parent", "parent"],
        }
    )
    keys = _build_hierarchical_keys(
        frame,
        source_columns=["child", "book", "partner"],
        key_names=["child_id", "book_id", "partner_id"],
        leaf_key="turn_id",
    )

    # Top-level child values sort A -> 0, B -> 1 regardless of source order.
    assert keys["child_id"].tolist() == [1, 0, 0, 0, 1, 1]
    # Book numbering resets within child: A has X=0,Y=1; B has A=0,Z=1.
    assert keys["book_id"].tolist() == [1, 1, 0, 0, 0, 1]
    # Partner numbering also resets within child+book.
    assert keys["partner_id"].tolist() == [1, 0, 1, 0, 0, 0]
    # Leaf IDs are local running integers in existing row order.
    assert keys["turn_id"].tolist() == [0, 0, 0, 0, 0, 0]


def test_leaf_ids_increment_only_within_deepest_group() -> None:
    frame = pd.DataFrame(
        {
            "child": ["A", "A", "A", "A", "B"],
            "book": ["X", "X", "X", "Y", "X"],
        }
    )
    keys = _build_hierarchical_keys(
        frame,
        source_columns=["child", "book"],
        key_names=["child_id", "book_id"],
        leaf_key="turn_id",
    )
    assert keys["turn_id"].tolist() == [0, 1, 2, 0, 0]
    assert not keys.duplicated(["child_id", "book_id", "turn_id"]).any()


def test_leaf_only_rekey_is_running_integer() -> None:
    frame = pd.DataFrame({"x": [3, 1, 2]})
    keys = _build_hierarchical_keys(
        frame,
        source_columns=[],
        key_names=[],
        leaf_key="turn_id",
    )
    assert keys.to_dict("list") == {"turn_id": [0, 1, 2]}


def test_hierarchical_keys_reject_null_and_unsortable_groups() -> None:
    with pytest.raises(ArtifactError, match="contains null"):
        _build_hierarchical_keys(
            pd.DataFrame({"child": ["A", None]}),
            source_columns=["child"],
            key_names=["child_id"],
            leaf_key="turn_id",
        )

    with pytest.raises(ArtifactError, match="cannot be deterministically sorted"):
        _build_hierarchical_keys(
            pd.DataFrame({"child": [1, "A"]}),
            source_columns=["child"],
            key_names=["child_id"],
            leaf_key="turn_id",
        )


def test_rekeyed_lineage_accepts_unrelated_key_schemas_but_requires_one_basis() -> None:
    validate_primary_key_relationship(
        basis_keys=[["row_id"]],
        output_key=["child_id", "book_id", "turn_id"],
        lineage_mode="rekeyed_key",
    )
    with pytest.raises(LineageError, match="exactly one basis"):
        validate_primary_key_relationship(
            basis_keys=[["row_id"], ["other_id"]],
            output_key=["child_id", "turn_id"],
            lineage_mode="rekeyed_key",
        )


@dataclass
class _FakeArtifact:
    artifact_id: str
    primary_key: list[str]
    mode: str
    basis_ids: tuple[str, ...]
    metadata: bool = False

    @property
    def descriptor(self):
        return {
            "lineage": {
                "lineage_mode": self.mode,
                "basis_artifact_ids": list(self.basis_ids),
            }
        }

    def has_metadata(self) -> bool:
        return self.metadata


class _FakeProject:
    def __init__(self, artifacts: list[_FakeArtifact]):
        self.artifacts = {artifact.artifact_id: artifact for artifact in artifacts}

    def get_artifact(self, artifact_id: str):
        return self.artifacts[artifact_id]


def test_metadata_discovery_crosses_multiple_rekeys_and_resets_key_space() -> None:
    a1 = _FakeArtifact("a1", ["doc_id"], "new_key", (), metadata=True)
    a2 = _FakeArtifact("a2", ["doc_id", "sent_id"], "extended_key", ("a1",))
    b1 = _FakeArtifact("b1", ["child_id", "turn_id"], "rekeyed_key", ("a2",))
    b2 = _FakeArtifact("b2", ["child_id", "turn_id"], "preserved_key", ("b1",))
    c1 = _FakeArtifact("c1", ["group_id", "item_id"], "rekeyed_key", ("b2",))
    c2 = _FakeArtifact("c2", ["group_id", "item_id"], "preserved_key", ("c1",))
    project = _FakeProject([a1, a2, b1, b2, c1, c2])

    sources = iter_metadata_lineage_sources(project, c2)
    assert [source.artifact_id for source in sources] == ["a1"]


def test_reduced_key_after_rekey_blocks_older_cross_keyspace_metadata() -> None:
    a1 = _FakeArtifact("a1", ["doc_id"], "new_key", (), metadata=True)
    a2 = _FakeArtifact("a2", ["doc_id", "sent_id"], "extended_key", ("a1",))
    b1 = _FakeArtifact("b1", ["child_id", "turn_id"], "rekeyed_key", ("a2",))
    reduced = _FakeArtifact("reduced", ["child_id"], "reduced_key", ("b1",))
    project = _FakeProject([a1, a2, b1, reduced])

    sources = iter_metadata_lineage_sources(project, reduced)
    assert sources == []


def test_mapping_sql_collapses_ordinary_segments_and_crosses_each_rekey_by_position(tmp_path) -> None:
    from text_analysis_lab.core.query import QueryEngine
    from text_analysis_lab.core.types import ArtifactType

    @dataclass
    class QArtifact(_FakeArtifact):
        n_rows: int = 4
        artifact_type: ArtifactType = ArtifactType.TABLE

        @property
        def keys_dir(self):
            return tmp_path / self.artifact_id / "keys"

    # C3 -> C1 is ordinary same-key lineage; C1 -> B3 is a rekey;
    # B3 -> B1 is ordinary; B1 -> A3 is a second rekey; A3 -> A1 is
    # ordinary extended-key lineage. The SQL should need only the two bridges.
    a1 = QArtifact("a1", ["doc_id"], "new_key", (), n_rows=2)
    a3 = QArtifact("a3", ["doc_id", "sent_id"], "extended_key", ("a1",), n_rows=4)
    b1 = QArtifact("b1", ["child_id", "turn_id"], "rekeyed_key", ("a3",), n_rows=4)
    b3 = QArtifact("b3", ["child_id", "turn_id"], "preserved_key", ("b1",), n_rows=4)
    c1 = QArtifact("c1", ["group_id", "item_id"], "rekeyed_key", ("b3",), n_rows=4)
    c3 = QArtifact("c3", ["group_id", "item_id", "piece_id"], "extended_key", ("c1",), n_rows=8)

    engine = QueryEngine(project=None)
    sql = engine._mapping_sql_for_path((c3, c1, b3, b1, a3, a1))
    assert sql is not None
    assert "bk._position = rk._position" in sql
    assert sql.count("bk._position = rk._position") == 2
    assert 'm."group_id" = rk."group_id"' in sql
    assert 'm."child_id" = rk."child_id"' in sql
    assert 'm."doc_id" = sk."doc_id"' in sql


def test_mapping_sql_rejects_malformed_rekey_and_unresolvable_reduced_domain(tmp_path) -> None:
    from text_analysis_lab.core.query import QueryEngine
    from text_analysis_lab.core.types import ArtifactType
    from text_analysis_lab.core.errors import QueryError

    @dataclass
    class QArtifact(_FakeArtifact):
        n_rows: int = 3
        artifact_type: ArtifactType = ArtifactType.TABLE

        @property
        def keys_dir(self):
            return tmp_path / self.artifact_id / "keys"

    old = QArtifact("old", ["row_id"], "new_key", (), n_rows=3)
    rekey = QArtifact("rekey", ["child_id", "turn_id"], "rekeyed_key", ("old",), n_rows=2)
    engine = QueryEngine(project=None)
    with pytest.raises(QueryError, match="row counts differ"):
        engine._mapping_sql_for_path((rekey, old))

    # If the downstream target no longer has all rekey-side key fields, there is
    # no unique row at the positional bridge; the path is intentionally unusable.
    good_rekey = QArtifact("good_rekey", ["child_id", "turn_id"], "rekeyed_key", ("old",), n_rows=3)
    reduced = QArtifact("reduced", ["child_id"], "reduced_key", ("good_rekey",), n_rows=2)
    assert engine._mapping_sql_for_path((reduced, good_rekey, old)) is None


def test_new_key_still_hard_stops_metadata_even_when_older_lineage_contains_rekeys() -> None:
    a = _FakeArtifact("a", ["row_id"], "new_key", (), metadata=True)
    b = _FakeArtifact("b", ["child_id", "turn_id"], "rekeyed_key", ("a",))
    c = _FakeArtifact("c", ["fresh_id"], "new_key", ("b",))
    project = _FakeProject([a, b, c])
    assert iter_metadata_lineage_sources(project, c) == []
