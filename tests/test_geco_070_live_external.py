from __future__ import annotations

import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer

if sys.version_info < (3, 11):
    pytest.skip("GeCo requires Python >=3.11", allow_module_level=True)

geometric_coder = pytest.importorskip("geometric_coder")

register_external_geometry = getattr(
    geometric_coder.GeometricCoder, "register_external_geometry", None
)
try:
    _geco_geometry_parameters = inspect.signature(register_external_geometry).parameters
except (TypeError, ValueError):
    _geco_geometry_parameters = {}
if "supports_text_transform" not in _geco_geometry_parameters:
    pytest.skip(
        "TeAL live bridge tests require the current pre-1.0 GeCo external-resource "
        "contract with per-geometry supports_text_transform; "
        f"found GeCo {getattr(geometric_coder, '__version__', 'unknown')}",
        allow_module_level=True,
    )


def _seed_table(project: teal.Project):
    rows = pd.DataFrame(
        {
            "row_id": list(range(8)),
            "text": [f"linked document {i}" for i in range(8)],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("live_docs"),
        artifact_id="live_docs",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["row_id"]], "data": rows[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="live_docs",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("live_docs")


def _seed_matrix(project, artifact_id, label, values, columns):
    is_sparse = sparse.issparse(values)
    artifact_type = "sparse_matrix" if is_sparse else "dense_matrix"
    writer = create_artifact_writer(
        artifact_type=artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": list(range(values.shape[0]))}),
            "data": {"values": values, "columns": list(columns)},
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        label=label,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def _keys(artifact):
    frame = artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        order_by="_position",
        include_position=True,
        form="table",
    )
    return frame["row_id"].astype(int).tolist()


def test_real_geco_external_create_register_and_reopen(tmp_path: Path):
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="live_geco")
    try:
        F = _seed_table(project)
        geometry_values = sparse.csr_matrix(
            np.array([[i, i % 2, (i + 1) % 3] for i in range(8)], dtype=float)
        )
        geometry = _seed_matrix(
            project, "live_geometry", "tfidf", geometry_values, ["a", "b", "c"]
        )
        view_values = np.array([[float(i), float(i * i)] for i in range(8)])
        view = _seed_matrix(project, "live_view", "umap", view_values, ["x", "y"])
        split = project.probability_split(
            F,
            n=2,
            remainder_label="train",
            sample_label="audit",
            random_state=31,
        )
        T = split["train"]
        t_keys = _keys(T)

        linked = project.geco.create(
            "live",
            documents=T,
            text_field="text",
            geometry=geometry,
            geometry_name="tfidf",
            projections={"umap": view},
        )
        assert linked.manifest["schema_version"] == 2
        assert linked.manifest["created_with_geco_version"] == str(
            getattr(geometric_coder, "__version__", "")
        )
        assert "geometries" not in linked.manifest
        assert "projections" not in linked.manifest
        assert linked.path.exists()
        with ThreadPoolExecutor(max_workers=2) as pool:
            matrix_future = pool.submit(
                linked.external_provider.geometry_matrix,
                {"artifact_id": geometry.artifact_id},
                [{"row_id": key} for key in t_keys],
            )
            coordinates_future = pool.submit(
                linked.external_provider.view_coordinates,
                {"artifact_id": view.artifact_id},
                [{"row_id": key} for key in t_keys],
            )
            matrix = matrix_future.result()
            coordinates = coordinates_future.result()
        np.testing.assert_array_equal(
            matrix.toarray(), geometry_values[t_keys, :].toarray()
        )
        np.testing.assert_array_equal(coordinates, view_values[t_keys, :])
        t_id = T.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        linked = reopened.geco.open("live")
        t_keys = _keys(reopened.get_artifact(t_id))
        with ThreadPoolExecutor(max_workers=1) as pool:
            matrix = pool.submit(
                linked.external_provider.geometry_matrix,
                {"artifact_id": "live_geometry"},
                [{"row_id": key} for key in reversed(t_keys)],
            ).result()
        np.testing.assert_array_equal(
            matrix.toarray(), geometry_values[list(reversed(t_keys)), :].toarray()
        )
    finally:
        reopened.close()
