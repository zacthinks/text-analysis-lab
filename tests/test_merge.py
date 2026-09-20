from __future__ import annotations

import pytest

from text_analysis_lab.core.errors import LineageError
from text_analysis_lab.core.lineage import validate_primary_key_relationship


def test_merged_key_requires_multiple_identical_basis_key_schemas() -> None:
    validate_primary_key_relationship(
        basis_keys=[("doc_id",), ("doc_id",)],
        output_key=("doc_id",),
        lineage_mode="merged_key",
    )

    with pytest.raises(LineageError, match="at least two"):
        validate_primary_key_relationship(
            basis_keys=[("doc_id",)],
            output_key=("doc_id",),
            lineage_mode="merged_key",
        )

    with pytest.raises(LineageError, match="same primary-key schema"):
        validate_primary_key_relationship(
            basis_keys=[("doc_id",), ("document_id",)],
            output_key=("doc_id",),
            lineage_mode="merged_key",
        )

    with pytest.raises(LineageError, match="must match"):
        validate_primary_key_relationship(
            basis_keys=[("doc_id",), ("doc_id",)],
            output_key=("other_id",),
            lineage_mode="merged_key",
        )


def test_merged_key_is_virtual_data_provider_for_preserved_descendants() -> None:
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

    left = FakeArtifact("left", "new_key", [], True)
    right = FakeArtifact("right", "new_key", [], True)
    merged = FakeArtifact("merged", "merged_key", ["left", "right"], False)
    child = FakeArtifact("child", "preserved_key", ["merged"], False)
    artifacts = {a.artifact_id: a for a in [left, right, merged, child]}

    class FakeProject:
        def get_artifact(self, artifact_id):
            return artifacts[artifact_id]

    project = FakeProject()
    for artifact in artifacts.values():
        artifact.project = project

    assert find_data_artifact(merged) is merged
    assert find_data_artifact(child) is merged


def test_full_metadata_can_cross_merge_boundary_and_deduplicates_common_ancestor() -> (
    None
):
    from text_analysis_lab.core.lineage import iter_metadata_lineage_sources

    class FakeArtifact:
        def __init__(self, artifact_id, mode, bases, has_metadata):
            self.artifact_id = artifact_id
            self.primary_key = ["file_id"]
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

    root = FakeArtifact("root", "new_key", [], True)
    left = FakeArtifact("left", "preserved_key", ["root"], False)
    right = FakeArtifact("right", "preserved_key", ["root"], False)
    merged = FakeArtifact("merged", "merged_key", ["left", "right"], False)
    child = FakeArtifact("child", "preserved_key", ["merged"], False)
    artifacts = {a.artifact_id: a for a in [root, left, right, merged, child]}

    class FakeProject:
        def get_artifact(self, artifact_id):
            return artifacts[artifact_id]

    sources = iter_metadata_lineage_sources(FakeProject(), child)
    assert [source.artifact_id for source in sources] == ["root"]
