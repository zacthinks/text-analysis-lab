from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.lineage import LINEAGE_MODES
from text_analysis_lab.core.types import ArtifactType


def test_public_import_exposes_current_core() -> None:
    assert teal.Project.__module__ == "text_analysis_lab.core.project"
    assert teal.TableArtifact.__module__ == "text_analysis_lab.core.artifact_subclasses"
    assert teal.BaseTranslator.__module__ == "text_analysis_lab.core.operator"


def test_project_catalog_uses_catalog_directory(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        expected = tmp_path / "project" / ".teal" / "catalog" / "catalog.sqlite"
        assert project.catalog.db_path == expected
        assert expected.exists()
        assert not (expected / "catalog.sqlite").exists()
    finally:
        project.close()


def test_ground_truth_lineage_modes_include_structural_merge() -> None:
    assert LINEAGE_MODES == (
        "preserved_key",
        "extended_key",
        "reduced_key",
        "span_key",
        "merged_key",
        "joined_key",
        "rekeyed_key",
        "new_key",
    )


def test_artifact_type_remains_string_like_and_json_serializable() -> None:
    assert isinstance(ArtifactType.TABLE, str)
    assert ArtifactType.TABLE == "table"
    assert json.loads(json.dumps({"type": ArtifactType.TABLE})) == {"type": "table"}


def test_legacy_operator_tree_is_reference_only() -> None:
    runtime_root = Path(teal.__file__).resolve().parent
    assert not (runtime_root / "operators").exists()


def _allocate_artifact_id(manifest_path: str) -> str:
    from text_analysis_lab.core.ids import next_id

    return next_id(Path(manifest_path), "artifact")


def test_transactional_ids_are_unique_across_processes(tmp_path: Path) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    project = teal.Project.create(tmp_path / "project", name="test_project")
    manifest_path = str(project.storage.manifest_path)
    project.close()

    with ProcessPoolExecutor(
        max_workers=8,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        ids = list(executor.map(_allocate_artifact_id, [manifest_path] * 80))

    assert len(ids) == 80
    assert len(set(ids)) == 80
    assert sorted(ids) == [f"art_{index:06d}" for index in range(1, 81)]


def test_transactional_ids_seed_from_legacy_manifest_count(tmp_path: Path) -> None:
    from text_analysis_lab.core.ids import next_id

    project = teal.Project.create(tmp_path / "project", name="test_project")
    manifest_path = project.storage.manifest_path
    project.close()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["artifacts"] = 12
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert next_id(manifest_path, "artifact") == "art_000013"
    assert next_id(manifest_path, "artifact") == "art_000014"


def _seed_writer_for_final_key_validation(
    tmp_path: Path,
    *,
    n_rows: int,
    n_parts: int,
):
    from text_analysis_lab.core.writer import ArtifactWriter

    writer = ArtifactWriter(
        artifact_dir=tmp_path / "artifact",
        artifact_id="art_000001",
        artifact_type="table",
        label="output",
    )
    writer._primary_key = ("id",)
    writer._n_rows = n_rows
    writer._next_position = n_rows
    writer._next_part_index = n_parts
    writer._components["keys"] = {"format": "parquet_dataset", "path": "keys/"}
    writer.keys_dir.mkdir(parents=True, exist_ok=True)
    for part_index in range(n_parts):
        writer.storage.key_part_path(part_index).touch()
    return writer


def test_finalize_seals_artifact_after_full_primary_key_check(
    tmp_path: Path, monkeypatch
) -> None:
    import pandas as pd

    writer = _seed_writer_for_final_key_validation(tmp_path, n_rows=4, n_parts=2)
    frames = {
        0: pd.DataFrame(
            {
                "id": [1, 2],
                "_position": [0, 1],
                "_batch": [0, 0],
                "_row_offset": [0, 1],
            }
        ),
        1: pd.DataFrame(
            {
                "id": [3, 4],
                "_position": [2, 3],
                "_batch": [1, 1],
                "_row_offset": [0, 1],
            }
        ),
    }

    def fake_read_parquet(path, *, columns):
        part_index = int(Path(path).stem.split("-")[-1])
        return frames[part_index].loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    writer.finalize()
    descriptor = json.loads(writer.storage.descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["status"] == "complete"


def test_finalize_marks_artifact_failed_on_cross_batch_duplicate_primary_key(
    tmp_path: Path, monkeypatch
) -> None:
    import pandas as pd
    import pytest

    from text_analysis_lab.core.errors import DuplicatePrimaryKeyError

    writer = _seed_writer_for_final_key_validation(tmp_path, n_rows=4, n_parts=2)
    frames = {
        0: pd.DataFrame(
            {
                "id": [1, 2],
                "_position": [0, 1],
                "_batch": [0, 0],
                "_row_offset": [0, 1],
            }
        ),
        1: pd.DataFrame(
            {
                "id": [2, 3],
                "_position": [2, 3],
                "_batch": [1, 1],
                "_row_offset": [0, 1],
            }
        ),
    }

    def fake_read_parquet(path, *, columns):
        part_index = int(Path(path).stem.split("-")[-1])
        return frames[part_index].loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    with pytest.raises(DuplicatePrimaryKeyError):
        writer.finalize()

    descriptor = json.loads(writer.storage.descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["status"] == "failed"
    assert "Duplicate primary-key" in descriptor["components"]["error"]["message"]


def test_transactional_ids_seed_from_existing_catalog_rows(tmp_path: Path) -> None:
    from text_analysis_lab.core.ids import next_id

    project = teal.Project.create(tmp_path / "project", name="test_project")
    project.catalog.register_artifact(
        artifact_id="art_000012",
        artifact_type="table",
        label="legacy",
        lineage_mode="new_key",
        status="failed",
    )
    manifest_path = project.storage.manifest_path
    project.close()

    assert next_id(manifest_path, "artifact") == "art_000013"


def _write_artifact_descriptor(
    project,
    *,
    artifact_id: str,
    label: str,
    status: str,
    primary_key: list[str],
    lineage_mode: str = "new_key",
    basis_artifact_ids: list[str] | None = None,
    operation_id: str | None = None,
) -> None:
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": label,
        "status": status,
        "primary_key": primary_key,
        "n_rows": 0,
        "components": {"keys": {"format": "parquet_dataset", "path": "keys/"}},
        "lineage": {
            "lineage_mode": lineage_mode,
            "basis_artifact_ids": list(basis_artifact_ids or []),
        },
        "operation_id": operation_id,
        "write": {"mode": "batch", "parts": 0},
    }
    project.storage.artifact_descriptor_path(artifact_id).write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )


def test_project_name_is_required_and_is_project_id(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(TypeError):
        teal.Project.create(tmp_path / "unnamed")

    project = teal.Project.create(tmp_path / "named", name="course_project")
    try:
        assert project.project_id == "course_project"
        assert project.name == "course_project"
        manifest = json.loads(project.manifest.read_text(encoding="utf-8"))
        assert manifest["project"]["project_id"] == "course_project"
        assert manifest["project"]["name"] == "course_project"
    finally:
        project.close()


def test_catalog_is_authoritative_for_artifact_status(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        project.catalog.register_artifact(
            artifact_id="art_000001",
            artifact_type="table",
            label="output",
            lineage_mode="new_key",
            status="incomplete",
        )
        _write_artifact_descriptor(
            project,
            artifact_id="art_000001",
            label="output",
            status="complete",
            primary_key=["id"],
        )
        artifact = project.get_artifact("art_000001")
        assert artifact.descriptor["status"] == "complete"
        assert artifact.status == "incomplete"
        project.catalog.mark_artifact_failed("art_000001")
        assert artifact.status == "failed"
    finally:
        project.close()


def test_keys_only_new_key_artifact_can_load(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        project.catalog.register_artifact(
            artifact_id="art_000001",
            artifact_type="table",
            label="keys_only",
            lineage_mode="new_key",
            status="incomplete",
        )
        _write_artifact_descriptor(
            project,
            artifact_id="art_000001",
            label="keys_only",
            status="incomplete",
            primary_key=["id"],
        )
        artifact = project.get_artifact("art_000001")
        assert artifact.data_artifact is None
        assert artifact.label == "keys_only"
    finally:
        project.close()


def test_artifact_operation_id_uses_writer_descriptor_field(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        project.catalog.register_artifact(
            artifact_id="art_000001",
            artifact_type="table",
            label="output",
            lineage_mode="new_key",
            status="incomplete",
        )
        _write_artifact_descriptor(
            project,
            artifact_id="art_000001",
            label="output",
            status="incomplete",
            primary_key=["id"],
            operation_id="run_000007",
        )
        assert project.get_artifact("art_000001").operation_id == "run_000007"
    finally:
        project.close()


def test_catalog_uses_wal_mode(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        journal_mode = project.catalog.con.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = project.catalog.con.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == 30000
    finally:
        project.close()


def test_final_lineage_validation_reuses_existing_validator(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import pytest

    from text_analysis_lab.core.errors import LineageError
    from text_analysis_lab.core.operator import OutputSpec
    from text_analysis_lab.core.translate import _validate_runtime_output_lineage

    project = teal.Project.create(tmp_path / "project", name="test_project")
    try:
        project.catalog.register_artifact(
            artifact_id="art_000001",
            artifact_type="table",
            label="source",
            lineage_mode="new_key",
            status="complete",
        )
        _write_artifact_descriptor(
            project,
            artifact_id="art_000001",
            label="source",
            status="complete",
            primary_key=["doc_id"],
        )
        project.catalog.register_artifact(
            artifact_id="art_000002",
            artifact_type="table",
            label="output",
            lineage_mode="preserved_key",
            status="incomplete",
            basis_artifact_ids=["art_000001"],
        )
        _write_artifact_descriptor(
            project,
            artifact_id="art_000002",
            label="output",
            status="complete",
            primary_key=["doc_id"],
            lineage_mode="preserved_key",
            basis_artifact_ids=["art_000001"],
        )
        runtime = SimpleNamespace(
            output_artifact_ids={"output": "art_000002"},
            output_specs={"output": OutputSpec(lineage_mode="preserved_key")},
        )
        _validate_runtime_output_lineage(project, runtime, "output")

        descriptor_path = project.storage.artifact_descriptor_path("art_000002")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor["primary_key"] = ["other_id"]
        descriptor_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")

        with pytest.raises(LineageError):
            _validate_runtime_output_lineage(project, runtime, "output")
    finally:
        project.close()


def test_sample_frac_does_not_also_request_default_n() -> None:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    class Dummy:
        def __init__(self) -> None:
            self.kwargs = None

        def query(self, **kwargs):
            self.kwargs = kwargs
            return kwargs

    dummy = Dummy()
    result = BaseArtifact.sample(dummy, frac=0.25, random_state=7)
    assert result["sample_n"] is None
    assert result["sample_frac"] == 0.25


def test_batched_single_form_rejects_multirow_batches() -> None:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    with pytest.raises(ValueError, match="requires batch_size=1"):
        BaseArtifact.query(
            object(),
            form="single",
            iter_batches=True,
            batch_size=2,
        )


def test_default_query_of_no_data_artifact_returns_available_info() -> None:
    from types import SimpleNamespace

    import pandas as pd

    from text_analysis_lab.core.artifact_base import BaseArtifact

    class DummyQuery:
        def artifact_query(self, artifact, **kwargs):
            assert kwargs["data_columns"] is False
            return pd.DataFrame({"unit_id": [1, 2]})

    class Dummy:
        data_artifact = None
        project = SimpleNamespace(query=DummyQuery())

        def _query_can_include_data(self, data_columns):
            return False

        def _format_query_frame(self, frame, *, form, include_position):
            return frame

    result = BaseArtifact._query_materialized(
        Dummy(),
        key_columns=True,
        data_columns=True,
        metadata_columns=False,
        metadata_mode="none",
        where=None,
        order_by=None,
        positions=None,
        limit=None,
        form="table",
        include_position=False,
    )
    assert list(result.columns) == ["unit_id"]
    assert result["unit_id"].tolist() == [1, 2]


def test_incomplete_artifact_views_are_refreshed_and_not_cached(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from text_analysis_lab.core.query import QueryEngine

    class DummyArtifact:
        artifact_id = "art_000001"
        status = "incomplete"
        primary_key: ClassVar[list[str]] = ["unit_id"]
        data_artifact = None

        def __init__(self) -> None:
            self.keys_dir = tmp_path / "keys"
            self.descriptor = {"status": "incomplete"}
            self.refresh_count = 0

        def refresh(self) -> None:
            self.refresh_count += 1

    artifact = DummyArtifact()
    engine = QueryEngine(SimpleNamespace())
    first = engine.build_artifact_view_sql(
        artifact, metadata_mode="none", include_data=False
    )
    second = engine.build_artifact_view_sql(
        artifact, metadata_mode="none", include_data=False
    )

    assert artifact.refresh_count == 2
    assert first is not second
    assert engine._artifact_view_cache == {}


def test_analysis_kwic_reuses_existing_implementation() -> None:
    import text_analysis_lab as teal
    from text_analysis_lab.core.kwic import keyword_in_context

    assert teal.analysis.kwic is keyword_in_context


def test_no_data_artifact_rejects_explicit_data_column_request() -> None:
    from types import SimpleNamespace

    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.errors import MissingDataComponentError

    class Dummy:
        artifact_id = "art_000001"
        data_artifact = None
        project = SimpleNamespace(query=None)

    with pytest.raises(MissingDataComponentError, match="no available data component"):
        BaseArtifact._query_materialized(
            Dummy(),
            key_columns=True,
            data_columns="missing_column",
            metadata_columns=False,
            metadata_mode="none",
            where=None,
            order_by=None,
            positions=None,
            limit=None,
            form="table",
            include_position=False,
        )


def test_metadata_lineage_reuses_non_bubbling_traversal_rules() -> None:
    from text_analysis_lab.core.lineage import iter_metadata_lineage_sources

    class FakeArtifact:
        def __init__(self, artifact_id, primary_key, mode, bases, has_metadata):
            self.artifact_id = artifact_id
            self.primary_key = list(primary_key)
            self.descriptor = {
                "lineage": {
                    "lineage_mode": mode,
                    "basis_artifact_ids": list(bases),
                }
            }
            self._has_metadata = has_metadata

        def has_metadata(self):
            return self._has_metadata

        def __repr__(self):
            return f"FakeArtifact({self.artifact_id})"

    root = FakeArtifact("root", ["doc_id"], "new_key", [], True)
    fine = FakeArtifact(
        "fine", ["doc_id", "sentence_id"], "extended_key", ["root"], True
    )
    reduced = FakeArtifact("reduced", ["doc_id"], "reduced_key", ["fine"], True)
    artifacts = {a.artifact_id: a for a in [root, fine, reduced]}

    class FakeProject:
        def get_artifact(self, artifact_id):
            return artifacts[artifact_id]

    sources = iter_metadata_lineage_sources(FakeProject(), reduced)
    assert [source.artifact_id for source in sources] == ["reduced", "root"]


def test_data_inheritance_remains_preserved_key_only() -> None:
    from text_analysis_lab.core.lineage import find_data_artifact
    from text_analysis_lab.core.types import ArtifactType

    class FakeArtifact:
        artifact_type = ArtifactType.TABLE

        def __init__(self, artifact_id, mode, bases, owns_data):
            self.artifact_id = artifact_id
            self.descriptor = {
                "lineage": {
                    "lineage_mode": mode,
                    "basis_artifact_ids": list(bases),
                }
            }
            self._owns_data = owns_data
            self.project = None

        def has_own_data(self):
            return self._owns_data

        def __repr__(self):
            return f"FakeArtifact({self.artifact_id})"

    parent = FakeArtifact("parent", "new_key", [], True)
    child = FakeArtifact("child", "preserved_key", ["parent"], False)
    extended = FakeArtifact("extended", "extended_key", ["parent"], False)
    artifacts = {a.artifact_id: a for a in [parent, child, extended]}

    class FakeProject:
        def get_artifact(self, artifact_id):
            return artifacts[artifact_id]

    project = FakeProject()
    for artifact in artifacts.values():
        artifact.project = project

    assert find_data_artifact(child) is parent
    assert find_data_artifact(extended) is None


def test_artifact_view_column_names_promote_ambiguous_names() -> None:
    from text_analysis_lab.core.query import _assign_view_output_names

    columns, ambiguous, mapping = _assign_view_output_names(
        [
            {
                "namespace": "key",
                "base_name": "text",
                "qualified_name": "key.text",
                "sql_expr": "k.text",
                "source_artifact_id": "a1",
            },
            {
                "namespace": "data",
                "base_name": "text",
                "qualified_name": "data.text",
                "sql_expr": "d.text",
                "source_artifact_id": "a1",
            },
            {
                "namespace": "metadata",
                "base_name": "text",
                "qualified_name": "metadata.a0.text",
                "sql_expr": "m0.text",
                "source_artifact_id": "a0",
            },
        ]
    )

    assert [column.output_name for column in columns] == [
        "key.text",
        "data.text",
        "metadata.a0.text",
    ]
    assert ambiguous["text"] == ("key.text", "data.text", "metadata.a0.text")
    assert mapping["data.text"] == "data.text"


def test_completed_status_transition_refreshes_stale_descriptor_before_caching(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from text_analysis_lab.core.query import QueryEngine

    class DummyArtifact:
        artifact_id = "art_000001"
        status = "complete"
        primary_key: ClassVar[list[str]] = ["unit_id"]
        data_artifact = None

        def __init__(self) -> None:
            self.keys_dir = tmp_path / "keys"
            self.descriptor = {"status": "incomplete"}
            self.refresh_count = 0

        def refresh(self) -> None:
            self.refresh_count += 1
            self.descriptor["status"] = "complete"

    artifact = DummyArtifact()
    engine = QueryEngine(SimpleNamespace())
    first = engine.build_artifact_view_sql(
        artifact, metadata_mode="none", include_data=False
    )
    second = engine.build_artifact_view_sql(
        artifact, metadata_mode="none", include_data=False
    )

    assert artifact.refresh_count == 1
    assert first is second


def test_sample_defaults_to_five_but_rejects_explicit_n_with_frac() -> None:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    class Dummy:
        def query(self, **kwargs):
            return kwargs

    assert BaseArtifact.sample(Dummy())["sample_n"] == 5
    with pytest.raises(ValueError, match="both n and frac"):
        BaseArtifact.sample(Dummy(), n=3, frac=0.25)


def test_external_data_batches_use_paged_query_to_avoid_invalidating_arrow_reader() -> (
    None
):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from text_analysis_lab.core.artifact_base import BaseArtifact

    class DummyQuery:
        def __init__(self) -> None:
            self.streaming_modes: list[str] = []

        def artifact_query_batches(self, artifact, **kwargs):
            self.streaming_modes.append(str(kwargs["streaming_mode"]))
            yield pd.DataFrame({"_position": [0, 1]})
            yield pd.DataFrame({"_position": [2, 3]})

    query = DummyQuery()

    class Dummy:
        artifact_id = "art_000001"
        data_artifact = object()
        project = SimpleNamespace(query=query)

        def _query_can_include_data(self, data_columns):
            return False

        def _data_native_for_positions(self, positions, *, data_columns=True):
            return np.asarray([[float(position)] for position in positions])

        def _native_from_info_and_data(self, info, data):
            return {"info": info, "matrix": data}

    batches = list(
        BaseArtifact._query_iter_batches(
            Dummy(),
            key_columns=False,
            data_columns=True,
            metadata_columns=False,
            metadata_mode="none",
            where=None,
            order_by=None,
            positions=None,
            limit=None,
            form="native",
            include_position=True,
            batch_size=2,
            streaming_mode="arrow",
        )
    )

    assert query.streaming_modes == ["paged"]
    assert [batch["info"]["_position"].tolist() for batch in batches] == [
        [0, 1],
        [2, 3],
    ]


def test_inline_data_batches_keep_requested_arrow_streaming_mode() -> None:
    from types import SimpleNamespace

    import pandas as pd

    from text_analysis_lab.core.artifact_base import BaseArtifact

    class DummyQuery:
        def __init__(self) -> None:
            self.streaming_modes: list[str] = []

        def artifact_query_batches(self, artifact, **kwargs):
            self.streaming_modes.append(str(kwargs["streaming_mode"]))
            yield pd.DataFrame({"doc_id": [0, 1], "text": ["a", "b"]})

    query = DummyQuery()

    class Dummy:
        data_artifact = object()
        project = SimpleNamespace(query=query)

        def _query_can_include_data(self, data_columns):
            return True

        def _format_query_frame(self, frame, *, form, include_position):
            return frame

    batches = list(
        BaseArtifact._query_iter_batches(
            Dummy(),
            key_columns=True,
            data_columns=True,
            metadata_columns=False,
            metadata_mode="none",
            where=None,
            order_by=None,
            positions=None,
            limit=None,
            form="table",
            include_position=False,
            batch_size=2,
            streaming_mode="arrow",
        )
    )

    assert query.streaming_modes == ["arrow"]
    assert len(batches) == 1


def test_parquet_dataset_queries_union_batch_schemas_by_name(tmp_path: Path) -> None:
    from text_analysis_lab.core.query import _parquet_dataset_expr

    expr = _parquet_dataset_expr(tmp_path / "data")
    assert "read_parquet(" in expr
    assert "union_by_name = true" in expr


def test_project_create_delete_existing_resets_only_teal_state(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir(parents=True)
    sibling = project_path / "keep-me.txt"
    sibling.write_text("user file", encoding="utf-8")

    project = teal.Project.create(project_path, name="first")
    marker = project.storage.teal_dir / "old-state.txt"
    marker.write_text("old", encoding="utf-8")
    project.close()

    replacement = teal.Project.create(
        project_path,
        name="second",
        delete_existing=True,
    )
    try:
        assert replacement.name == "second"
        assert sibling.read_text(encoding="utf-8") == "user file"
        assert not marker.exists()
        assert replacement.storage.teal_dir.exists()
    finally:
        replacement.close()
