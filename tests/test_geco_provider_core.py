from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactNotFoundError
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.integrations.geco import GeCoIntegrationError, TeALGeCoProvider


class _FakeArtifact:
    def __init__(self, artifact_id, artifact_type, matrix, keys):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.primary_key = ["row_id"]
        self._matrix = matrix
        self._keys = list(keys)

    def get_matrix(self, *, positions=None):
        if positions is None:
            return self._matrix
        return self._matrix[list(positions), :]


class _FakeQuery:
    def positions_by_keys(self, artifact, keys):
        index = {int(value): pos for pos, value in enumerate(artifact._keys)}
        try:
            return [index[int(record["row_id"])] for record in keys]
        except KeyError as exc:
            raise KeyError("missing") from exc


class _FakeProject:
    def __init__(self, artifacts):
        self._artifacts = {artifact.artifact_id: artifact for artifact in artifacts}
        self.query = _FakeQuery()

    def get_artifact(self, artifact_id):
        return self._artifacts[artifact_id]


def test_provider_subsets_and_reorders_sparse_geometry_by_stable_key():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
            ]
        )
    )
    artifact = _FakeArtifact(
        "art_geometry", ArtifactType.SPARSE_MATRIX, matrix, [7, 2, 9, 5]
    )
    provider = TeALGeCoProvider(_FakeProject([artifact]))

    result = provider.geometry_matrix(
        {"artifact_id": "art_geometry"},
        [{"row_id": 5}, {"row_id": 7}, {"row_id": 2}],
    )

    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_equal(
        result.toarray(),
        np.array([[40.0, 400.0], [10.0, 100.0], [20.0, 200.0]]),
    )


def test_provider_subsets_and_reorders_dense_view_by_stable_key():
    matrix = np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]])
    artifact = _FakeArtifact("art_view", ArtifactType.DENSE_MATRIX, matrix, [11, 4, 8])
    provider = TeALGeCoProvider(_FakeProject([artifact]))

    result = provider.view_coordinates(
        {"artifact_id": "art_view"},
        [{"row_id": 8}, {"row_id": 11}],
    )
    np.testing.assert_array_equal(result, np.array([[2.1, 2.2], [0.1, 0.2]]))


def test_provider_missing_document_key_fails_instead_of_positional_fallback():
    matrix = sparse.eye(3, format="csr")
    artifact = _FakeArtifact(
        "art_geometry", ArtifactType.SPARSE_MATRIX, matrix, [0, 1, 2]
    )
    provider = TeALGeCoProvider(_FakeProject([artifact]))

    with pytest.raises(GeCoIntegrationError, match="every document key"):
        provider.geometry_matrix(
            {"artifact_id": "art_geometry"},
            [{"row_id": 0}, {"row_id": 99}],
        )


def test_provider_rejects_non_2d_view():
    matrix = np.ones((3, 3))
    artifact = _FakeArtifact("art_view", ArtifactType.DENSE_MATRIX, matrix, [0, 1, 2])
    provider = TeALGeCoProvider(_FakeProject([artifact]))

    with pytest.raises(GeCoIntegrationError, match="exactly 2 columns"):
        provider.view_coordinates(
            {"artifact_id": "art_view"},
            [{"row_id": 0}, {"row_id": 1}],
        )


def test_provider_wraps_only_genuine_missing_artifacts():
    class MissingProject:
        query = _FakeQuery()

        def get_artifact(self, artifact_id):
            raise ArtifactNotFoundError(artifact_id)

    provider = TeALGeCoProvider(MissingProject())
    with pytest.raises(GeCoIntegrationError, match="unavailable TeAL artifact"):
        provider.geometry_matrix(
            {"artifact_id": "missing"},
            [{"row_id": 0}],
        )


def test_provider_does_not_mislabel_catalog_threading_failures_as_missing():
    class BrokenProject:
        query = _FakeQuery()

        def get_artifact(self, artifact_id):
            raise sqlite3.ProgrammingError("wrong thread")

    provider = TeALGeCoProvider(BrokenProject())
    with pytest.raises(sqlite3.ProgrammingError, match="wrong thread"):
        provider.geometry_matrix(
            {"artifact_id": "art_geometry"},
            [{"row_id": 0}],
        )


def test_provider_uses_same_recovered_pipeline_for_text_and_query():
    matrix = sparse.eye(3, format="csr")
    artifact = _FakeArtifact(
        "art_geometry", ArtifactType.SPARSE_MATRIX, matrix, [0, 1, 2]
    )
    text = _FakeArtifact(
        "art_documents", ArtifactType.DENSE_MATRIX, np.zeros((3, 1)), [0, 1, 2]
    )
    calls = []

    class FakePipeline:
        def translate(self, texts, *, project=None):
            assert project is project_obj
            calls.append(list(texts))
            values = sparse.csr_matrix(
                np.arange(len(texts) * 3, dtype=float).reshape(len(texts), 3)
            )
            return {"output": {"values": values}}

    class PipelineProject(_FakeProject):
        def pipeline(self, *, start, end):
            assert start == "art_documents"
            assert end is artifact
            return FakePipeline()

    project_obj = PipelineProject([artifact, text])
    provider = TeALGeCoProvider(project_obj)
    external_ref = {
        "artifact_id": "art_geometry",
        "text_artifact_id": "art_documents",
    }

    query = provider.transform_query(external_ref, "room temperature")
    texts = provider.transform_texts(external_ref, ["a", "b"])

    np.testing.assert_array_equal(query.toarray(), [[0, 1, 2]])
    np.testing.assert_array_equal(texts.toarray(), [[0, 1, 2], [3, 4, 5]])
    assert calls == [["room temperature"], ["a", "b"]]


def test_geco_contract_gate_rejects_pre_capability_build(monkeypatch):
    import sys
    from types import SimpleNamespace

    import text_analysis_lab.integrations.geco as bridge

    class OldGeometricCoder:
        def register_external_geometry(
            self, *, name, external_ref, supports_query=False
        ):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "geometric_coder",
        SimpleNamespace(__version__="0.8.1", GeometricCoder=OldGeometricCoder),
    )
    with pytest.raises(GeCoIntegrationError, match="supports_text_transform"):
        bridge._load_geometric_coder()


def test_geco_contract_gate_accepts_per_geometry_text_capability(monkeypatch):
    import sys
    from types import SimpleNamespace

    import text_analysis_lab.integrations.geco as bridge

    class CurrentGeometricCoder:
        def register_external_geometry(
            self,
            *,
            name,
            external_ref,
            supports_query=False,
            supports_text_transform=False,
        ):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "geometric_coder",
        SimpleNamespace(__version__="0.next", GeometricCoder=CurrentGeometricCoder),
    )
    assert bridge._load_geometric_coder() is CurrentGeometricCoder
