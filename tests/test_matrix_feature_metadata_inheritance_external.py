from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.core.translate import _output_specs_from_dict
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import EmbeddingLookup


def _matrix_source(project: teal.Project):
    artifact_id = "art_matrix_source"
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="matrix_source",
        lineage_mode="new_key",
        feature_metadata_mode="own",
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": list(range(6))}),
            "metadata": pd.DataFrame(
                {"group": ["a", "a", "b", "b", "c", "c"]}
            ),
            "data": {
                "values": sparse.csr_matrix(
                    np.asarray(
                        [
                            [1, 0, 2],
                            [0, 3, 0],
                            [4, 0, 5],
                            [0, 1, 1],
                            [2, 2, 0],
                            [0, 0, 6],
                        ],
                        dtype=float,
                    )
                ),
                "columns": ["alpha", "beta", "gamma"],
            },
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix",
        label="matrix_source",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    source = project.get_artifact(artifact_id)
    feature_metadata = pd.DataFrame(
        {
            "column_index": np.arange(3, dtype=np.int64),
            "column": ["alpha", "beta", "gamma"],
            "family": ["lexical", "lexical", "phrase"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    feature_metadata.to_parquet(source.storage.feature_metadata_path, index=False)
    return source, feature_metadata


def _assert_inherits_feature_metadata(artifact, expected: pd.DataFrame) -> None:
    assert artifact.descriptor["lineage"]["feature_metadata_mode"] == "inherit"
    assert not artifact.storage.feature_metadata_path.exists()
    pd.testing.assert_frame_equal(artifact.get_feature_metadata(), expected)
    assert artifact.get_matrix().shape[1] == len(expected)


def test_legacy_stored_matrix_output_spec_defaults_to_owned_metadata() -> None:
    specs = _output_specs_from_dict(
        {
            "output": {
                "artifact_type": "dense_matrix",
                "lineage_mode": "new_key",
                "basis_labels": [],
            }
        }
    )
    assert specs["output"].feature_metadata_mode == "own"


def test_stored_output_spec_preserves_explicit_feature_metadata_basis() -> None:
    specs = _output_specs_from_dict(
        {
            "output": {
                "artifact_type": "dense_matrix",
                "lineage_mode": "preserved_key",
                "basis_labels": ["tokens"],
                "feature_metadata_mode": "inherit",
                "feature_metadata_basis_label": "embeddings",
            }
        }
    )
    spec = specs["output"]
    assert spec.basis_labels == ("tokens",)
    assert spec.feature_metadata_mode == "inherit"
    assert spec.feature_metadata_basis_label == "embeddings"


def test_row_preserving_matrix_operations_inherit_rich_feature_metadata(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "project", name="feature_metadata_rows")
    try:
        source, expected = _matrix_source(project)

        sampled = project.sample(
            source,
            n=3,
            random_state=7,
            output_label="sampled",
        )
        selected = project.select_keys(
            source,
            [0, 2, 4],
            output_label="selected",
        )
        restricted = project.restrict(
            source,
            to=selected,
            output_label="restricted",
        )
        subset = project.subset(
            source,
            lambda frame: frame["row_id"].astype(int) % 2 == 0,
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            output_label="subset",
        )["subset"]
        split = project.split(
            source,
            labels=("left", "right"),
            proportions=(0.5, 0.5),
            random_state=11,
        )
        probability = project.probability_split(
            source,
            n=2,
            remainder_label="train",
            sample_label="audit",
            random_state=13,
        )
        rekeyed = project.set_primary_keys(
            source,
            levels={"group": "group_id"},
            leaf_key="item_id",
            output_label="rekeyed",
        )

        children = [
            sampled,
            selected,
            restricted,
            subset,
            split["left"],
            split["right"],
            probability["train"],
            probability["audit"],
            rekeyed,
        ]
        for child in children:
            _assert_inherits_feature_metadata(child, expected)
    finally:
        project.close()



def test_embedding_lookup_inherits_feature_metadata_from_embedding_source(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(tmp_path / "embedding_project", name="embedding_basis")
    try:
        token_id = "art_tokens"
        token_writer = create_artifact_writer(
            artifact_type="table",
            artifact_dir=project.storage.artifact_dir(token_id),
            artifact_id=token_id,
            label="tokens",
            lineage_mode="new_key",
        )
        token_writer.write(
            {
                "keys": pd.DataFrame(
                    {
                        "doc_id": [1, 1, 1],
                        "token_id": [0, 1, 2],
                    }
                ),
                "data": pd.DataFrame({"lemma": ["b", "missing", "a"]}),
            }
        )
        token_writer.finalize()
        project.catalog.register_artifact(
            artifact_id=token_id,
            artifact_type="table",
            label="tokens",
            lineage_mode="new_key",
            status="complete",
            basis_artifact_ids=(),
        )
        tokens = project.get_artifact(token_id)

        embedding_id = "art_embeddings"
        embedding_writer = create_artifact_writer(
            artifact_type="dense_matrix",
            artifact_dir=project.storage.artifact_dir(embedding_id),
            artifact_id=embedding_id,
            label="embeddings",
            lineage_mode="new_key",
            feature_metadata_mode="own",
        )
        embedding_writer.write(
            {
                "keys": pd.DataFrame({"word_id": [0, 1]}),
                "data": {
                    "values": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                    "columns": ["d0", "d1"],
                    "row_names": ["a", "b"],
                    "row_name": "word",
                },
            }
        )
        embedding_writer.finalize()
        project.catalog.register_artifact(
            artifact_id=embedding_id,
            artifact_type="dense_matrix",
            label="embeddings",
            lineage_mode="new_key",
            status="complete",
            basis_artifact_ids=(),
        )
        embeddings = project.get_artifact(embedding_id)
        expected = embeddings.get_feature_metadata()
        expected["family"] = ["semantic", "semantic"]
        expected["source"] = ["word2vec", "word2vec"]
        expected.to_parquet(embeddings.storage.feature_metadata_path, index=False)

        looked_up = project.translate(
            EmbeddingLookup(field="lemma"),
            {"tokens": tokens, "embeddings": embeddings},
        )["output"]

        assert looked_up.descriptor["lineage"]["basis_artifact_ids"] == [token_id]
        assert (
            looked_up.descriptor["lineage"]["feature_metadata_basis_artifact_id"]
            == embedding_id
        )
        assert looked_up.descriptor["lineage"]["feature_metadata_mode"] == "inherit"
        assert not looked_up.storage.feature_metadata_path.exists()
        pd.testing.assert_frame_equal(looked_up.get_feature_metadata(), expected)
        assert looked_up.get_matrix().shape == (3, 2)
    finally:
        project.close()
