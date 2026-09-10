from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

geometric_coder = pytest.importorskip("geometric_coder")
from geometric_coder import GeCoPredictorRef as NativeGeCoPredictorRef
from geometric_coder.predictors import (
    FrozenPredictorExport,
    FrozenPredictorMember,
    PredictorSourceSpec,
)

from text_analysis_lab.integrations.geco import LinkedGeCoWorkspace
from text_analysis_lab.translators import GeCoPredictor


def test_bridge_matches_geco_0814_native_ref_and_frozen_export_contract() -> None:
    X = np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=float)
    y = np.asarray([0, 0, 1, 1], dtype=int)
    model = LogisticRegression(random_state=0).fit(X, y)

    native_ref = NativeGeCoPredictorRef(
        kind="committee", id=5, code_id=1, name="qualitative_classifier"
    )
    frozen = FrozenPredictorExport(
        format_version=1,
        ref=native_ref,
        code_name="qualitative methods",
        sources=(
            PredictorSourceSpec(
                source_index=0,
                geometry_id=7,
                geometry_name="sentence_lsa100",
                n_features=1,
                storage_kind="external",
                external_ref={"artifact_id": "art_lsa"},
            ),
        ),
        members=(
            FrozenPredictorMember(
                classifier_spec_id=2,
                classifier_fit_id=9,
                classifier_name="member",
                fit_classifier_name="member",
                source_index=0,
                algorithm="logistic_l2",
                hyperparameters={"threshold": 0.5},
                score_kind="probability",
                positive_class=1,
                estimator=model,
            ),
        ),
        aggregation="mean",
        stacker=None,
        stacker_fit_id=None,
        positive_class=1,
        threshold=0.5,
        output_fields=("prediction", "probability"),
        stale_at_export=False,
        provenance={"geco_version": getattr(geometric_coder, "__version__", "unknown")},
    )

    class Coder:
        def predictors(self):
            return [native_ref]

        def export_predictor(self, ref, *, allow_stale=False):
            assert ref is native_ref
            assert allow_stale is False
            return frozen

    linked = LinkedGeCoWorkspace(
        SimpleNamespace(project=object()),
        {"name": "fake", "workspace_path": "fake.geco", "documents_artifact_id": "x"},
        Coder(),
        None,
    )

    bridge_ref = linked.predictors()[0]
    predictor = linked.export_predictor(bridge_ref)

    assert isinstance(predictor, GeCoPredictor)
    assert predictor.aggregation == "mean"
    assert predictor.source_specs[0]["geometry_name"] == "sentence_lsa100"
    assert predictor.source_specs[0]["external_ref"] == {"artifact_id": "art_lsa"}
    assert predictor.member_specs[0]["algorithm"] == "logistic_l2"
