"""ArgMax numeric, artifact, and replay contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import ArgMax


def _named(values, names=("z", "a", "m")):
    return {"values": values, "columns": list(names)}


@pytest.mark.parametrize("factory", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_numeric_ties_zeros_implicit_and_negative(factory):
    values = np.array([
        [4, 4, 0],
        [0, 0, 0],
        [-2, 0, -1],
        [-2, -4, -1],
        [0, 1, 1],
    ], dtype=float)
    result = ArgMax().translate(_named(factory(values)))
    assert result["max_feature"].tolist()[:1] == ["a"]
    assert pd.isna(result["max_feature"].iloc[1])
    assert result["max_feature"].iloc[2:].tolist() == ["a", "m", "a"]
    assert result["max_value"].tolist() == pytest.approx([4, 0, 0, -1, 1])


def test_sparse_zero_maximum_with_negative_stored_entries():
    matrix = sparse.csr_matrix(([-2., -1.], ([0, 0], [0, 2])), shape=(1, 3))
    result = ArgMax().translate(_named(matrix))
    assert result["max_feature"].tolist() == ["a"]
    assert result["max_value"].tolist() == [0]


def test_sparse_duplicate_coordinate_summed_before_ranking():
    matrix = sparse.coo_matrix(
        ([2., 3., 4.], ([0, 0, 0], [0, 0, 1])), shape=(1, 3)
    )
    result = ArgMax().translate(_named(matrix))
    assert result["max_feature"].tolist() == ["z"]
    assert result["max_value"].tolist() == [5]


def test_sparse_does_not_densify(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Sparse ArgMax may not densify")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject)
    matrix = sparse.csr_matrix(([3.], ([0], [999_999])), shape=(2, 1_000_000))
    names = [f"f_{i:07d}" for i in range(1_000_000)]
    result = ArgMax().translate(matrix, features=names)
    assert result["max_feature"].iloc[0] == "f_0999999"
    assert pd.isna(result["max_feature"].iloc[1])


@pytest.mark.parametrize("values", [
    np.array([[np.nan, 1.]]),
    np.array([[np.inf, 1.]]),
    np.array([["bad", "value"]]),
    sparse.csr_matrix(([np.inf], ([0], [0])), shape=(1, 2)),
])
def test_invalid_values_rejected(values):
    with pytest.raises((ValueError, TypeError), match="finite numeric"):
        ArgMax().translate(_named(values, names=("a", "b")))


def test_missing_duplicate_or_inconsistent_features_rejected():
    model = ArgMax()
    with pytest.raises(ValueError, match="feature names"):
        model.translate(np.eye(2))
    with pytest.raises(ValueError, match="unique"):
        model.translate(np.eye(2), features=["a", "a"])
    with pytest.raises(ValueError, match="feature count"):
        model.translate(np.eye(2), features=["a"])
    with pytest.raises(ValueError, match="at least one"):
        model.translate(np.empty((2, 0)), features=[])


def test_named_metadata_schema_and_source_immutability():
    values = np.array([[.2, .8]])
    frame = pd.DataFrame({"column_index": [0, 1], "column": ["topic_0", "topic_1"]})
    before = values.copy()
    result = ArgMax().translate({"values": values, "feature_metadata": frame})
    assert result["max_feature"].tolist() == ["topic_1"]
    assert result["max_value"].tolist() == pytest.approx([.8])
    np.testing.assert_array_equal(values, before)


def test_batch_preserves_composite_keys_and_worker_parity():
    from types import SimpleNamespace

    source = SimpleNamespace(
        artifact_type=SimpleNamespace(value="sparse_matrix"),
        get_data_columns=lambda: ["topic_0", "topic_1"],
    )
    op = ArgMax()
    request = TranslationRequest(batch_size=2)
    spec = op.output_specs(sources={"source": source}, request=request)
    assert spec.lineage_mode == "preserved_key"
    binding = op.input_request(sources={"source": source}, mode="translate", request=request)
    assert binding.mode == "batches"
    assert binding.form == "native"
    info = pd.DataFrame({"doc_id": [7, 3], "segment_id": [2, 4]})
    packet = InputBatch(
        source_label="source", artifact_id="unit", primary_key=("doc_id", "segment_id"),
        data={"info": info, "matrix": sparse.csr_matrix([[.1, .9], [0., 0.]])},
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )
    direct = op.translate_batch({"source": packet}, mode="translate", request=request).outputs["output"]
    parallel = op.make_translate_worker(mode="translate", request=request)
    alternate = parallel.translate_batch({"source": packet}, mode="translate", request=request).outputs["output"]
    pd.testing.assert_frame_equal(direct["keys"], info)
    pd.testing.assert_frame_equal(direct["data"], alternate["data"])
    assert direct["data"]["max_feature"].iloc[0] == "topic_1"
    assert pd.isna(direct["data"]["max_feature"].iloc[1])
    assert ArgMax.from_json_state(op.to_json_state()).execution_capabilities().reusable


def _register_matrix(project, artifact_id, matrix, columns):
    from text_analysis_lab.core.writer import create_artifact_writer
    kind = "sparse_matrix" if sparse.issparse(matrix) else "dense_matrix"
    writer = create_artifact_writer(
        artifact_type=kind, artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id, label="source_matrix",
        lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({
        "keys": pd.DataFrame({"doc_id": [19, 11, 37]}),
        "data": {"values": matrix, "columns": columns},
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id, artifact_type=kind,
        label="source_matrix", lineage_mode="new_key",
        status="complete", basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_real_project_batches_reopen_and_frozen_reuse(tmp_path, sparse_input):
    import text_analysis_lab as teal

    location = tmp_path / "argmax_project"
    project = teal.Project.create(location, name="argmax")
    values = np.array([[.1, .9], [0., 0.], [.8, .2]])
    if sparse_input:
        values = sparse.csr_matrix(values)
    try:
        source = _register_matrix(project, "argmax_input", values, ["topic_0", "topic_1"])
        op = ArgMax()
        output = project.translate(op, source, batch_size=1)["output"]
        assert output.artifact_type.value == "table"
        assert output.primary_key == ["doc_id"]
        assert output.descriptor["lineage"]["lineage_mode"] == "preserved_key"
        artifact_id = output.artifact_id
        operator_id = str(op.operator_id)
        expected = output.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
        )
        assert expected["doc_id"].tolist() == [19, 11, 37]
        assert expected["max_feature"].iloc[0] == "topic_1"
        assert pd.isna(expected["max_feature"].iloc[1])
        assert expected["max_value"].tolist() == pytest.approx([.9, 0, .8])
    finally:
        project.close()

    reopened = teal.Project.open(location)
    try:
        persisted = reopened.get_artifact(artifact_id)
        assert persisted.status == "complete"
        again = reopened.translate(
            reopened.get_operator(operator_id),
            reopened.get_artifact("argmax_input"),
            batch_size=2,
        )["output"]
        actual = again.query(
            key_columns=True, data_columns=True,
            metadata_columns=False, metadata_mode="none",
        )
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False,
        )
    finally:
        reopened.close()
