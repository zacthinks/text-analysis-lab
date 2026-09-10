from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from text_analysis_lab.core.transform_like import can_transform_texts_like, transform_texts_like
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import CountVectorizer, FeatureTrimmer, MatrixNormalizer, TfidfTransformer


class _Storage:
    def __init__(self, root: Path):
        self.root = root

    def operation_descriptor_path(self, operation_id: str) -> Path:
        return self.root / operation_id / "operation.json"


class _Artifact:
    def __init__(self, project, artifact_id, artifact_type, columns):
        self.project = project
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self._columns = list(columns)
        project.artifacts[artifact_id] = self

    def get_data_columns(self):
        return list(self._columns)


class _Project:
    def __init__(self, root: Path):
        self.storage = _Storage(root)
        self.artifacts = {}
        self.operations = {}
        self.sources = {}
        self.outputs = {}
        self.operators = {}

    def get_artifact(self, value):
        return self.artifacts[value] if isinstance(value, str) else value

    def operation_for_artifact(self, artifact):
        artifact = self.get_artifact(artifact)
        opid = next((op for op, aid in self.operations.items() if aid == artifact.artifact_id), None)
        return None if opid is None else {"operation_id": opid, "operator_id": f"operator_{opid}"}

    def operation_outputs(self, operation_id):
        return self.outputs[operation_id]

    def operation_sources(self, operation_id):
        return self.sources[operation_id]

    def get_operator(self, operator_id):
        return self.operators[operator_id]


def _record(project, opid, source, output, operator):
    project.operations[opid] = output.artifact_id
    project.sources[opid] = [{"source_label": "source", "source_artifact_id": source.artifact_id}]
    project.outputs[opid] = [{"output_label": "output", "artifact_id": output.artifact_id}]
    project.operators[f"operator_{opid}"] = operator
    path = project.storage.operation_descriptor_path(opid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "request": {"params": {}},
                "output_specs": {"output": {"lineage_mode": "preserved_key"}},
            }
        ),
        encoding="utf-8",
    )


def test_transform_texts_like_replays_frozen_count_tfidf_pipeline(tmp_path: Path) -> None:
    project = _Project(tmp_path)
    table = _Artifact(project, "table", ArtifactType.TABLE, ["text"])

    count = CountVectorizer(vocabulary={"alpha": 0, "beta": 1, "gamma": 2})
    count_matrix = _Artifact(project, "count", ArtifactType.SPARSE_MATRIX, ["alpha", "beta", "gamma"])
    _record(project, "countop", table, count_matrix, count)

    train = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=float))
    tfidf = TfidfTransformer()
    tfidf.source_features_ = ("alpha", "beta", "gamma")
    tfidf._transformer = tfidf._make_transformer().fit(train)
    tfidf_matrix = _Artifact(project, "tfidf", ArtifactType.SPARSE_MATRIX, ["alpha", "beta", "gamma"])
    _record(project, "tfidfop", count_matrix, tfidf_matrix, tfidf)

    expected = tfidf._require_transformer().transform(count._require_vectorizer().transform(["alpha gamma"]))
    actual = transform_texts_like(project, tfidf_matrix, ["alpha gamma"], query=True)
    np.testing.assert_allclose(actual.toarray(), expected.toarray())
    assert can_transform_texts_like(project, tfidf_matrix, query=True)


def test_column_normalization_is_not_replayable_for_single_new_text(tmp_path: Path) -> None:
    project = _Project(tmp_path)
    table = _Artifact(project, "table", ArtifactType.TABLE, ["text"])
    count = CountVectorizer(vocabulary={"alpha": 0, "beta": 1})
    count_matrix = _Artifact(project, "count", ArtifactType.SPARSE_MATRIX, ["alpha", "beta"])
    _record(project, "countop", table, count_matrix, count)
    normalizer = MatrixNormalizer(axis="columns")
    norm_matrix = _Artifact(project, "norm", ArtifactType.SPARSE_MATRIX, ["alpha", "beta"])
    _record(project, "normop", count_matrix, norm_matrix, normalizer)
    assert not can_transform_texts_like(project, norm_matrix)


def test_transform_texts_like_replays_frozen_feature_trim_mask(tmp_path: Path) -> None:
    project = _Project(tmp_path)
    table = _Artifact(project, "table", ArtifactType.TABLE, ["text"])

    count = CountVectorizer(vocabulary={"alpha": 0, "beta": 1, "gamma": 2})
    count_matrix = _Artifact(
        project, "count", ArtifactType.SPARSE_MATRIX, ["alpha", "beta", "gamma"]
    )
    _record(project, "countop", table, count_matrix, count)

    trim = FeatureTrimmer(min_df=2)
    trim.source_features_ = ("alpha", "beta", "gamma")
    trim.kept_indices_ = (0, 2)
    trimmed = _Artifact(project, "trimmed", ArtifactType.SPARSE_MATRIX, ["alpha", "gamma"])
    _record(project, "trimop", count_matrix, trimmed, trim)

    actual = transform_texts_like(project, trimmed, ["alpha beta gamma"], query=True)
    assert actual.shape == (1, 2)
    np.testing.assert_array_equal(actual.toarray(), [[1, 1]])
    assert can_transform_texts_like(project, trimmed, query=True)
