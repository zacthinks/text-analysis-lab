"""The computed-matrix provider must be opt-in and query bounded source rows."""

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.translators._feature_matrix_encoding import FeatureMatrixEncoding
from text_analysis_lab.translators._feature_matrix_provider import render_feature_matrix


class _Type:
    def __init__(self, value):
        self.value = value


class _Source:
    def __init__(self, matrix=None):
        self.artifact_type = _Type("sparse_matrix" if matrix is not None else "table")
        self.matrix = matrix
        self.info = pd.DataFrame(
            {"document_id": [10, 20, 30],
             "income": [4.0, 5.0, 6.0],
             "gender": ["f", "m", "f"]}
        )
        self.calls = []

    def query(self, *, positions, key_columns, data_columns, metadata_columns,
              form, metadata_mode="none"):
        self.calls.append((list(positions), form))
        if form == "native":
            assert self.matrix is not None
            return {"matrix": self.matrix[list(positions), :]}
        columns = (
            (["document_id"] if key_columns is True else list(key_columns) if key_columns else [])
            + (["income"] if data_columns is True else list(data_columns) if data_columns else [])
            + (["gender"] if metadata_columns is True else list(metadata_columns) if metadata_columns else [])
        )
        return self.info.iloc[list(positions)][columns].reset_index(drop=True)


def test_lazy_sparse_output_only_queries_requested_rows():
    source = _Source(sparse.csr_matrix([[0, 1], [2, 0], [3, 4]]))
    encoder = FeatureMatrixEncoding({"gender": "onehot"})
    encoder.fit(source.info[["gender"]])
    result = render_feature_matrix(
        source, positions=[2, 0], matrix_columns=True,
        metadata_columns=["gender"], encoder=encoder,
        sparse_output=True, output_columns=["a", "b", *encoder.columns],
    )
    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_equal(result.toarray(), [[3, 4, 1, 0], [0, 1, 1, 0]])
    assert source.calls == [([2, 0], "native"), ([2, 0], "table")]


def test_table_only_dense_with_selected_keys_as_features():
    source = _Source()
    encoder = FeatureMatrixEncoding().fit(source.info[["document_id", "income"]])
    result = render_feature_matrix(
        source, positions=[1], matrix_columns=False,
        data_columns=["income"], key_columns=["document_id"],
        encoder=encoder, sparse_output=False,
    )
    np.testing.assert_array_equal(result, [[20, 5.0]])
    assert source.calls == [([1], "table")]
