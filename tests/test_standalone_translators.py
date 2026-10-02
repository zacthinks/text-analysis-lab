from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import (
    OperatorNotFittedError,
    StandaloneTranslationNotSupportedError,
)
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import (
    FeatureTrimmer,
    MatrixNormalizer,
    RegexCleaner,
    RegexReplaceRule,
    TfidfTransformer,
)


def _matrix_packet(matrix) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": list(range(int(matrix.shape[0])))}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _matrix_source(kind: str, columns: list[str]):
    return SimpleNamespace(
        artifact_type=ArtifactType(kind),
        primary_key=["doc_id"],
        get_data_columns=lambda: list(columns),
    )


class _InternalOnlyTranslator(BaseTranslator):
    """Minimal translator proving standalone support is not required by inheritance."""

    def output_specs(self, *, sources, request):
        _ = sources, request
        return OutputSpec(artifact_type="table", lineage_mode="preserved_key")

    def input_request(self, *, sources, mode, request):
        _ = sources, mode, request
        return SourceRequest(
            artifact_type="table",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        _ = inputs, mode, request
        return BatchResult()

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(self, *, mode, request):
        _ = mode, request
        return None


def test_base_translator_allows_internal_execution_only_subclasses() -> None:
    with pytest.raises(
        StandaloneTranslationNotSupportedError,
        match="does not expose a standalone",
    ):
        _InternalOnlyTranslator().translate(["not", "a", "public", "translator"])


def test_regex_cleaner_standalone_matches_teal_batch_semantics() -> None:
    translator = RegexCleaner(
        rules=[
            RegexReplaceRule(r"hello", "hi", flags=("IGNORECASE",)),
            {"pattern": r"[!-]+", "replacement": " "},
            {"pattern": r"\s+", "replacement": " "},
        ]
    )
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["  HELLO   world!!! ", None, "Hello---TeAL"],
        }
    )

    direct = translator.translate(frame["text"])
    assert isinstance(direct, pd.Series)
    assert direct.tolist()[0] == "hi world"
    assert pd.isna(direct.tolist()[1])
    assert direct.tolist()[2] == "hi TeAL"
    assert translator.translate("  HELLO!!! ") == "hi"
    assert translator.translate(None) is None

    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["text"]
    pd.testing.assert_series_equal(
        batch.reset_index(drop=True),
        direct.reset_index(drop=True),
        check_names=False,
    )


def test_tfidf_standalone_matches_teal_batch_and_validates_state() -> None:
    train = sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    new_rows = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    translator = TfidfTransformer(norm=None)
    translator.source_features_ = ("alpha", "beta", "gamma")
    translator._transformer = translator._make_transformer().fit(train)

    direct = translator.translate(new_rows)
    assert sparse.isspmatrix_csr(direct)

    batch = translator.translate_batch(
        {"source": _matrix_packet(new_rows)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_allclose(batch.toarray(), direct.toarray())

    with pytest.raises(ValueError, match="fitted feature width"):
        translator.translate(sparse.csr_matrix([[1.0, 2.0]]))
    with pytest.raises(OperatorNotFittedError, match="fitted IDF state"):
        TfidfTransformer().translate(new_rows)


def test_feature_trimmer_standalone_is_semantically_equivalent_to_lazy_teal_view() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )
    translator = FeatureTrimmer()
    translator.source_width_ = 4
    translator.kept_indices_ = (0, 2)

    direct = translator.translate(matrix)
    np.testing.assert_array_equal(direct.toarray(), matrix[:, [0, 2]].toarray())

    key_packet = InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id",),
        data=pd.DataFrame({"doc_id": [0, 1]}),
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    lazy_payload = translator.translate_batch(
        {"source": key_packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert lazy_payload["feature_indices"] == [0, 2]
    np.testing.assert_array_equal(
        direct.toarray(),
        matrix[:, lazy_payload["feature_indices"]].toarray(),
    )

    dense = matrix.toarray()
    dense_direct = translator.translate(dense)
    assert isinstance(dense_direct, np.ndarray)
    np.testing.assert_array_equal(dense_direct, dense[:, [0, 2]])


@pytest.mark.parametrize(
    ("axis", "norm", "kind"),
    [
        ("rows", "l2", "sparse_matrix"),
        ("columns", "l1", "dense_matrix"),
    ],
)
def test_matrix_normalizer_standalone_matches_teal_batch(
    axis: str,
    norm: str,
    kind: str,
) -> None:
    dense = np.array(
        [
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    matrix = sparse.csr_matrix(dense) if kind == "sparse_matrix" else dense
    columns = ["alpha", "beta", "gamma"]
    translator = MatrixNormalizer(axis=axis, norm=norm)
    translator.input_request(
        sources={"source": _matrix_source(kind, columns)},
        mode="translate",
        request=TranslationRequest(batch_size=2),
    )

    direct = translator.translate(matrix)
    batch = translator.translate_batch(
        {"source": _matrix_packet(matrix)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]

    if sparse.issparse(direct):
        assert sparse.isspmatrix_csr(batch)
        np.testing.assert_allclose(batch.toarray(), direct.toarray())
    else:
        assert isinstance(batch, np.ndarray)
        np.testing.assert_allclose(batch, direct)

    with pytest.raises(ValueError, match="two-dimensional"):
        translator.translate(np.array([1.0, 2.0, 3.0]))
