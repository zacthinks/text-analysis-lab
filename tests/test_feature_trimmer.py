from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import BaseOperator, InputBatch, TranslationRequest
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import FeatureTrimmer


FEATURES = ["alpha", "beta", "gamma", "delta", "epsilon"]
COUNTS = sparse.csr_matrix(
    np.array(
        [
            [3, 0, 1, 0, 2],
            [0, 2, 0, 1, 1],
            [1, 1, 0, 3, 0],
            [0, 0, 4, 1, 0],
            [2, 1, 1, 0, 1],
            [0, 3, 0, 2, 0],
        ],
        dtype=float,
    )
)


def _source(kind: str = "sparse_matrix", features=FEATURES):
    return SimpleNamespace(
        artifact_type=ArtifactType(kind),
        primary_key=["doc_id"],
        get_data_columns=lambda: list(features),
    )


def _packet(matrix=COUNTS) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_counts",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": list(range(matrix.shape[0]))}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_feature_trimmer_fits_df_mask_and_preserves_original_column_order() -> None:
    trimmer = FeatureTrimmer(min_df=4)
    request = trimmer.input_request(
        sources={"source": _source()},
        mode="fit_translate",
        request=TranslationRequest(batch_size=2),
    )
    assert request.mode == "full_artifact"
    result = trimmer.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert trimmer.kept_indices_ == (1, 3)
    assert result.outputs["output"]["feature_indices"] == [1, 3]
    assert "data" not in result.outputs["output"]


def test_feature_trimmer_fractional_max_df_and_max_features_are_deterministic() -> None:
    low_df = FeatureTrimmer(max_df=0.5)
    low_df.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    )
    low_df.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    assert low_df.kept_indices_ == (0, 2, 4)

    top_one = FeatureTrimmer(min_df=4, max_features=1)
    top_one.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    )
    top_one.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    # beta and delta tie on corpus sum; original position is the deterministic tie-breaker.
    assert top_one.kept_indices_ == (1,)


def test_feature_trimmer_replays_exact_frozen_mask_and_round_trips(tmp_path: Path) -> None:
    trimmer = FeatureTrimmer(min_df=4)
    trimmer.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    )
    trimmer.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )

    new_rows = sparse.csr_matrix([[9, 8, 7, 6, 5], [1, 2, 3, 4, 5]])
    replayed = trimmer.transform_external_matrix(new_rows, query=True)
    assert np.array_equal(replayed.toarray(), new_rows[:, [1, 3]].toarray())

    path = tmp_path / "feature_trimmer"
    trimmer.save_to_dir(path, operator_id="op_trim")
    restored = BaseOperator.load_from_dir(path)
    assert isinstance(restored, FeatureTrimmer)
    assert restored.source_width_ == 5
    assert restored.kept_indices_ == (1, 3)
    assert np.array_equal(
        restored.transform_external_matrix(new_rows).toarray(),
        new_rows[:, [1, 3]].toarray(),
    )


def test_feature_trimmer_dense_external_replay_stays_dense() -> None:
    dense = COUNTS.toarray()
    trimmer = FeatureTrimmer(min_df=4)
    trimmer.input_request(
        sources={"source": _source("dense_matrix")},
        mode="fit_translate",
        request=TranslationRequest(),
    )
    result = trimmer.translate_batch(
        {"source": _packet(dense)}, mode="fit_translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["feature_indices"] == [1, 3]
    replayed = trimmer.transform_external_matrix(dense)
    assert isinstance(replayed, np.ndarray)
    assert np.array_equal(replayed, dense[:, [1, 3]])


def test_feature_trimmer_rejects_empty_or_mismatched_schema() -> None:
    trimmer = FeatureTrimmer(min_df=7, max_df=100)
    trimmer.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    )
    with pytest.raises(ArtifactError, match="retained no features"):
        trimmer.translate_batch(
            {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
        )

    fitted = FeatureTrimmer(min_df=4)
    fitted.input_request(
        sources={"source": _source()}, mode="fit_translate", request=TranslationRequest()
    )
    fitted.translate_batch(
        {"source": _packet()}, mode="fit_translate", request=TranslationRequest()
    )
    # Same width is accepted even when labels differ: feature identity is positional.
    replay_request = fitted.input_request(
        sources={"source": _source(features=[*FEATURES[:-1], "changed"])},
        mode="translate",
        request=TranslationRequest(),
    )
    assert replay_request.columns.data is False
    assert replay_request.form == "table"

    with pytest.raises(OperatorError, match="fitted source width"):
        fitted.input_request(
            sources={"source": _source(features=FEATURES[:-1])},
            mode="translate",
            request=TranslationRequest(),
        )
