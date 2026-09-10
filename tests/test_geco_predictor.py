from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import BaseOperator, InputBatch, TranslationRequest
from text_analysis_lab.core.translate import _resolve_sources
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.integrations.geco import (
    GeCoPredictorRef,
    LinkedGeCoWorkspace,
)
from text_analysis_lab.translators import GeCoPredictor


class _ColumnProbability:
    def __init__(self, column: int):
        self.column = int(column)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if sparse.issparse(X):
            values = np.asarray(X[:, self.column].toarray()).reshape(-1)
        else:
            values = np.asarray(X)[:, self.column]
        p = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class _FakeArtifact:
    def __init__(
        self, artifact_id: str, *, n_rows: int, kind=ArtifactType.DENSE_MATRIX, n_features: int = 1
    ):
        self.artifact_id = artifact_id
        self.artifact_type = kind
        self.primary_key = ["row_id"]
        self.n_rows = n_rows
        self._n_features = int(n_features)

    def get_data_columns(self):
        return [str(index) for index in range(self._n_features)]


class _FakeProject:
    def __init__(self, artifacts):
        self.artifacts = {a.artifact_id: a for a in artifacts}

    def get_artifact(self, ref):
        if hasattr(ref, "artifact_id"):
            return ref
        return self.artifacts[str(ref)]


def _packet(label: str, values, keys, *, batch_index=0, batch_count=1):
    return InputBatch(
        source_label=label,
        artifact_id=f"art_{label}",
        primary_key=("row_id",),
        data={
            "info": pd.DataFrame({"row_id": list(keys)}),
            "matrix": values,
        },
        batch_index=batch_index,
        batch_count=batch_count,
        is_first=batch_index == 0,
        is_last=batch_index == batch_count - 1,
    )


def _predictor(*, aggregation="single", stacker=None):
    models = [_ColumnProbability(0)] if aggregation == "single" else [
        _ColumnProbability(0),
        _ColumnProbability(0),
    ]
    source_specs = [{"source_index": 0}]
    member_specs = [{"source_index": 0, "positive_class": 1}]
    if aggregation != "single":
        source_specs.append({"source_index": 1})
        member_specs.append({"source_index": 1, "positive_class": 1})
    return GeCoPredictor(
        models,
        source_specs=source_specs,
        member_specs=member_specs,
        aggregation=aggregation,
        stacker=stacker,
    )


def test_sequence_source_binding_normalizes_to_ordered_labels():
    a = _FakeArtifact("a", n_rows=3)
    b = _FakeArtifact("b", n_rows=3)
    project = _FakeProject([a, b])
    resolved = _resolve_sources(project, [a, "b"])
    assert list(resolved) == ["source_0", "source_1"]
    assert resolved["source_0"] is a
    assert resolved["source_1"] is b


def test_geco_predictor_one_source_lineage_and_request():
    predictor = _predictor()
    source = _FakeArtifact("a", n_rows=3)
    spec = predictor.output_specs(
        sources={"source_0": source}, request=TranslationRequest()
    )
    assert spec.lineage_mode == "preserved_key"
    assert spec.basis_labels == ("source_0",)
    request = predictor.input_request(
        sources={"source_0": source},
        mode="translate",
        request=TranslationRequest(batch_size=17),
    )
    assert list(request) == ["source_0"]
    assert request["source_0"].batch_size == 17
    assert request["source_0"].form == "native"


def test_geco_predictor_multi_source_preserves_first_source_key_lineage():
    predictor = _predictor(aggregation="mean")
    a = _FakeArtifact("a", n_rows=3, kind=ArtifactType.SPARSE_MATRIX)
    b = _FakeArtifact("b", n_rows=3, kind=ArtifactType.DENSE_MATRIX)
    spec = predictor.output_specs(
        sources={"source_0": a, "source_1": b}, request=TranslationRequest()
    )
    assert spec.lineage_mode == "preserved_key"
    assert spec.basis_labels == ("source_0",)


def test_fixed_committee_mixed_packets_reproduces_aggregation():
    predictor = _predictor(aggregation="mean")
    inputs = {
        "source_0": _packet(
            "source_0", sparse.csr_matrix([[0.2], [0.8], [0.4]]), [1, 2, 3]
        ),
        "source_1": _packet("source_1", np.array([[0.6], [0.1], [0.9]]), [1, 2, 3]),
    }
    result = predictor.translate_batch(inputs, mode="translate", request=TranslationRequest())
    frame = result.outputs["output"]["data"]
    np.testing.assert_allclose(frame["probability"], [0.4, 0.45, 0.65])
    assert frame["prediction"].tolist() == [0, 0, 1]


def test_logistic_stack_uses_frozen_member_order():
    p0 = np.array([0.1, 0.8, 0.3, 0.9, 0.6])
    p1 = np.array([0.7, 0.2, 0.4, 0.9, 0.1])
    y = np.array([0, 1, 0, 1, 1])
    stacker = LogisticRegression(random_state=0).fit(np.column_stack([p0, p1]), y)
    predictor = _predictor(aggregation="logistic_stack", stacker=stacker)
    inputs = {
        "source_0": _packet("source_0", p0.reshape(-1, 1), range(5)),
        "source_1": _packet("source_1", p1.reshape(-1, 1), range(5)),
    }
    result = predictor.translate_batch(inputs, mode="translate", request=TranslationRequest())
    observed = result.outputs["output"]["data"]["probability"].to_numpy()
    expected = stacker.predict_proba(np.column_stack([p0, p1]))[:, 1]
    np.testing.assert_allclose(observed, expected)


def test_geco_predictor_rejects_misaligned_keys():
    predictor = _predictor(aggregation="mean")
    inputs = {
        "source_0": _packet("source_0", np.array([[0.2], [0.8]]), [1, 2]),
        "source_1": _packet("source_1", np.array([[0.6], [0.1]]), [2, 1]),
    }
    with pytest.raises(ArtifactError, match="not row-aligned"):
        predictor.translate_batch(inputs, mode="translate", request=TranslationRequest())


def test_geco_predictor_rejects_row_count_mismatch_before_batches():
    predictor = _predictor(aggregation="mean")
    a = _FakeArtifact("a", n_rows=3)
    b = _FakeArtifact("b", n_rows=2)
    with pytest.raises(OperatorError, match="same number of rows"):
        predictor.output_specs(
            sources={"source_0": a, "source_1": b}, request=TranslationRequest()
        )




def test_geco_predictor_rejects_feature_width_mismatch_before_batches():
    predictor = GeCoPredictor(
        [_ColumnProbability(0)],
        source_specs=[{"source_index": 0, "geometry_name": "g0", "n_features": 2}],
        member_specs=[{"source_index": 0, "positive_class": 1}],
        aggregation="single",
    )
    source = _FakeArtifact("a", n_rows=3, n_features=3)
    with pytest.raises(OperatorError, match="requires 2 features"):
        predictor.output_specs(
            sources={"source_0": source}, request=TranslationRequest()
        )

def test_geco_predictor_serialization_round_trip(tmp_path):
    predictor = _predictor(aggregation="mean")
    predictor.save_to_dir(tmp_path / "operator", operator_id="op_1")
    restored = BaseOperator.load_from_dir(tmp_path / "operator")
    assert isinstance(restored, GeCoPredictor)
    assert restored.aggregation == "mean"
    assert len(restored.member_models) == 2
    inputs = {
        "source_0": _packet("source_0", np.array([[0.2], [0.8]]), [1, 2]),
        "source_1": _packet("source_1", np.array([[0.6], [0.1]]), [1, 2]),
    }
    result = restored.translate_batch(inputs, mode="translate", request=TranslationRequest())
    np.testing.assert_allclose(result.outputs["output"]["data"]["probability"], [0.4, 0.45])


def test_bridge_consumes_neutral_frozen_export_without_geco_dependency():
    model = _ColumnProbability(0)

    class NativeRef:
        def __init__(self):
            self.kind = "classifier"
            self.id = 7
            self.code_id = 3
            self.name = "quality"

    native_ref = NativeRef()

    class Coder:
        def predictors(self):
            return [native_ref]

        def export_predictor(self, ref, *, allow_stale=False):
            # GeCo 0.8.14 requires the exact native ref object returned by predictors().
            assert ref is native_ref
            assert allow_stale is False
            return SimpleNamespace(
                format_version=1,
                ref=native_ref,
                code_name="quality",
                sources=[
                    SimpleNamespace(
                        source_index=0,
                        geometry_id=2,
                        geometry_name="tfidf",
                        n_features=1,
                        storage_kind="external",
                        external_ref={"artifact_id": "art_tfidf"},
                    )
                ],
                members=[
                    SimpleNamespace(
                        estimator=model,
                        classifier_spec_id=7,
                        classifier_fit_id=11,
                        classifier_name="quality",
                        fit_classifier_name="quality",
                        source_index=0,
                        algorithm="logistic_l2",
                        hyperparameters={"threshold": 0.5},
                        score_kind="probability",
                        positive_class=1,
                    )
                ],
                aggregation=None,
                stacker=None,
                stacker_fit_id=None,
                positive_class=1,
                threshold=0.5,
                output_fields=("prediction", "probability"),
                provenance={"geco_name": "quality"},
                stale_at_export=False,
            )

    manager = SimpleNamespace(project=object())
    linked = LinkedGeCoWorkspace(
        manager,
        {"name": "fake", "workspace_path": "fake.geco", "documents_artifact_id": "x"},
        Coder(),
        None,
    )
    refs = linked.predictors()
    assert refs == [GeCoPredictorRef(kind="classifier", id=7, code_id=3, name="quality")]
    predictor = linked.export_predictor(refs[0])
    assert isinstance(predictor, GeCoPredictor)
    assert predictor.aggregation == "single"
    assert predictor.source_specs[0]["geometry_name"] == "tfidf"
    assert predictor.provenance["geco_name"] == "quality"

def test_export_predictor_rejects_bare_name():
    class Coder:
        def predictors(self):
            return []

    linked = LinkedGeCoWorkspace(
        SimpleNamespace(project=object()),
        {"name": "fake", "workspace_path": "fake.geco", "documents_artifact_id": "x"},
        Coder(),
        None,
    )
    with pytest.raises(TypeError, match="GeCoPredictorRef"):
        linked.export_predictor("quality")  # type: ignore[arg-type]
