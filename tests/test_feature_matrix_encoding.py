"""Isolated acceptance tests for the opt-in FeatureMatrix encoding engine."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.translators._feature_matrix_encoding import (
    FeatureMatrixEncoding,
    append_feature_columns,
)


def test_onehot_freezes_categories_and_ignores_unseen_on_replay():
    train = pd.DataFrame({"gender": ["male", "female", "male"], "age": [30, 40, 25]})
    enc = FeatureMatrixEncoding().fit(train)
    assert enc.columns == ("gender_female", "gender_male", "age")
    np.testing.assert_array_equal(enc.transform(train),
                                  [[0, 1, 30], [1, 0, 40], [0, 1, 25]])
    replay = FeatureMatrixEncoding.from_state(enc.to_state())
    assert replay.columns == enc.columns
    incoming = pd.DataFrame({"gender": ["nonbinary", "female", None], "age": [22, 27, 31]})
    np.testing.assert_array_equal(replay.transform(incoming),
                                  [[0, 0, 22], [1, 0, 27], [0, 0, 31]])


def test_explicit_mapping_unknown_is_nan_and_numeric_override_onehot():
    train = pd.DataFrame({"gender": ["male", "female"], "code": [1, 2]})
    enc = FeatureMatrixEncoding({
        "gender": {"male": 0, "female": 1},
        "code": "onehot",
    }).fit(train)
    assert enc.columns == ("gender", "code_1", "code_2")
    output = enc.transform(pd.DataFrame({"gender": ["other"], "code": [3]}))
    assert np.isnan(output[0, 0])
    np.testing.assert_array_equal(output[0, 1:], [0, 0])


def test_sparse_assembly_preserves_sparse_values_and_does_not_mutate_parent():
    original = sparse.csr_matrix([[0, 4, 0], [3, 0, 0]], dtype=float)
    enc = FeatureMatrixEncoding({"group": "onehot"}).fit(
        pd.DataFrame({"group": ["x", "y"]})
    )
    extra = enc.transform(pd.DataFrame({"group": ["x", "y"]}), sparse_output=True)
    combined = append_feature_columns(original, extra, sparse_output=True)
    assert sparse.isspmatrix_csr(combined)
    assert combined.shape == (2, 5)
    np.testing.assert_array_equal(
        combined.toarray(), [[0, 4, 0, 1, 0], [3, 0, 0, 0, 1]]
    )
    assert original.shape == (2, 3)
    assert original.nnz == 2


def test_dense_assembly_and_table_only():
    extra = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(append_feature_columns(None, extra, sparse_output=False), extra)
    combined = append_feature_columns(np.array([[9], [8]]), extra, sparse_output=False)
    np.testing.assert_array_equal(combined, [[9, 1, 2], [8, 3, 4]])


def test_mismatched_rows_and_unfitted_state_fail():
    with pytest.raises(ArtifactError):
        FeatureMatrixEncoding().transform(pd.DataFrame({"x": [1]}))
    with pytest.raises(ArtifactError, match="matching row counts"):
        append_feature_columns(np.zeros((3, 2)), np.ones((2, 1)), sparse_output=True)
