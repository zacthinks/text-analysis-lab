from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.integrations import (
    GeCoIntegrationError,
    GeCoPredictorRef,
    TeALGeCoProvider,
)
from text_analysis_lab.translators import (
    SVD,
    CountVectorizer,
    GeCoPredictor,
    TfidfTransformer,
)


class _FakeGeometricCoder:
    """Minimal current GeCo public seam used to test TeAL's bridge contract."""

    _registry: dict[str, _FakeGeometricCoder] = {}

    def __init__(self, project_dir, data, keys, text, metadata, provider):
        self.project_dir = Path(project_dir)
        self.data = data.copy()
        self.keys = list(keys)
        self.text = text
        self.metadata = list(metadata)
        self.provider = provider
        self._geometries = {}
        self._views = {}
        self.launched = False
        self.launch_kwargs = None
        self.closed = False

    @classmethod
    def create_external(
        cls,
        *,
        project_dir,
        data,
        keys,
        text,
        metadata,
        external_provider,
        overwrite=False,
        **kwargs,
    ):
        assert overwrite is False
        path = Path(project_dir)
        path.mkdir(parents=True, exist_ok=False)
        coder = cls(path, data, keys, text, metadata, external_provider)
        cls._registry[str(path.resolve())] = coder
        return coder

    @classmethod
    def open(cls, project_dir, external_provider=None, **kwargs):
        coder = cls._registry[str(Path(project_dir).resolve())]
        coder.provider = external_provider
        coder.closed = False
        return coder

    def _ordered_keys(self):
        return self.data.loc[:, self.keys].to_dict(orient="records")

    def register_external_geometry(
        self,
        *,
        name,
        external_ref,
        supports_query=False,
        supports_text_transform=False,
        public=True,
    ):
        existing = self._geometries.get(name)
        declaration = {
            "external_ref": external_ref,
            "supports_query": bool(supports_query),
            "supports_text_transform": bool(supports_text_transform),
            "public": bool(public),
        }
        if existing is not None:
            if all(existing[key] == value for key, value in declaration.items()):
                return existing["geometry_id"]
            raise ValueError(f"Conflicting geometry registration for {name!r}")
        matrix = self.provider.geometry_matrix(external_ref, self._ordered_keys())
        assert matrix.shape[0] == len(self.data)
        geometry_id = len(self._geometries) + 1
        self._geometries[name] = {
            "geometry_id": geometry_id,
            "name": name,
            "matrix": matrix,
            **declaration,
        }
        return geometry_id

    def register_external_view(self, *, geometry_id, name, external_ref):
        existing = self._views.get(name)
        declaration = {
            "geometry_id": int(geometry_id),
            "external_ref": external_ref,
        }
        if existing is not None:
            if all(existing[key] == value for key, value in declaration.items()):
                return existing["view_id"]
            raise ValueError(f"Conflicting view registration for {name!r}")
        coordinates = self.provider.view_coordinates(external_ref, self._ordered_keys())
        assert coordinates.shape == (len(self.data), 2)
        view_id = len(self._views) + 1
        self._views[name] = {
            "view_id": view_id,
            "name": name,
            "coordinates": coordinates,
            **declaration,
        }
        return view_id

    def geometries(self, *, public_only=True):
        records = list(self._geometries.values())
        if public_only:
            records = [record for record in records if record["public"]]
        return [dict(record) for record in records]

    def views(self, geometry_id=None):
        records = list(self._views.values())
        if geometry_id is not None:
            records = [
                record
                for record in records
                if int(record["geometry_id"]) == int(geometry_id)
            ]
        return [dict(record) for record in records]

    def geometry_matrix(self, geometry):
        if isinstance(geometry, int):
            record = next(
                record
                for record in self._geometries.values()
                if int(record["geometry_id"]) == int(geometry)
            )
        else:
            record = self._geometries[str(geometry)]
        return self.provider.geometry_matrix(
            record["external_ref"], self._ordered_keys()
        )

    def view_coordinates(self, view_id):
        record = next(
            record
            for record in self._views.values()
            if int(record["view_id"]) == int(view_id)
        )
        return self.provider.view_coordinates(
            record["external_ref"], self._ordered_keys()
        )

    def export_codes(self, code):
        frame = self.data.loc[:3, self.keys].copy()
        frame["label"] = [1, 0, 1, 0]
        return frame

    def predictors(self):
        return [
            SimpleNamespace(
                kind="classifier",
                id=17,
                code_id=4,
                name="room_temp_logistic",
            )
        ]

    def export_predictor(self, ref, *, allow_stale=False):
        assert ref.kind == "classifier"
        assert int(ref.id) == 17
        assert int(ref.code_id) == 4
        assert ref.name == "room_temp_logistic"
        assert allow_stale is False
        geometry = self._geometries.get("tfidf") or next(
            iter(self._geometries.values())
        )
        X = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        y = np.array([0, 0, 1, 1, 1, 0])
        model = LogisticRegression(random_state=0).fit(X, y)
        return {
            "format_version": 1,
            "sources": [
                {
                    "source_index": 0,
                    "geometry_id": int(geometry["geometry_id"]),
                    "geometry_name": str(geometry["name"]),
                    "n_features": 3,
                    "storage_kind": "external",
                    "external_ref": dict(geometry["external_ref"]),
                }
            ],
            "members": [
                {
                    "estimator": model,
                    "classifier_spec_id": 17,
                    "classifier_fit_id": 21,
                    "classifier_name": "room_temp_logistic",
                    "fit_classifier_name": "room_temp_logistic",
                    "source_index": 0,
                    "algorithm": "logistic_l2",
                    "hyperparameters": {"threshold": 0.5},
                    "score_kind": "probability",
                    "positive_class": 1,
                }
            ],
            "aggregation": None,
            "stacker": None,
            "positive_class": 1,
            "threshold": 0.5,
            "output_fields": ["prediction", "probability"],
            "provenance": {
                "geco_predictor_kind": "classifier",
                "geco_predictor_id": 17,
                "geco_code_id": 4,
                "geco_name": "room_temp_logistic",
            },
            "stale_at_export": False,
        }

    def export_classifier(self, name):
        ref = self.predictors()[0]
        return self.export_predictor(ref, allow_stale=False)["members"][0]["estimator"]

    def launch(self, *, host="127.0.0.1", port=8050, debug=False):
        self.launched = True
        self.launch_kwargs = {"host": host, "port": port, "debug": debug}
        return "launched"

    def close(self):
        self.closed = True


def _seed_table(project: teal.Project):
    rows = pd.DataFrame(
        {
            "row_id": list(range(10)),
            "text": [f"document {i}" for i in range(10)],
            "year": [2020 + (i % 3) for i in range(10)],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_docs"),
        artifact_id="art_docs",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": rows[["row_id"]],
            "data": rows[["text"]],
            "metadata": rows[["year"]],
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_docs",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_docs")


def _seed_matrix(
    project: teal.Project, artifact_id: str, label: str, values, *, columns
):
    is_sparse = sparse.issparse(values)
    writer = create_artifact_writer(
        artifact_type="sparse_matrix" if is_sparse else "dense_matrix",
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
        artifact_type="sparse_matrix" if is_sparse else "dense_matrix",
        label=label,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def _frame(artifact, fields=False):
    return artifact.query(
        key_columns=True,
        data_columns=fields,
        metadata_columns=False,
        order_by="_position",
        include_position=True,
        form="table",
    )


def test_linked_geco_create_export_reopen_and_apply(tmp_path: Path, monkeypatch):
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.next-test")

    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="geco_bridge")
    try:
        F = _seed_table(project)
        geometry_values = sparse.csr_matrix(
            np.array(
                [[i % 2, (i // 2) % 2, (i // 3) % 2] for i in range(10)], dtype=float
            )
        )
        geometry = _seed_matrix(
            project,
            "art_geometry",
            "tfidf",
            geometry_values,
            columns=["a", "b", "c"],
        )
        view_values = np.array([[float(i), -float(i)] for i in range(10)])
        view = _seed_matrix(
            project,
            "art_view",
            "umap",
            view_values,
            columns=["x", "y"],
        )

        split = project.probability_split(
            F, n=3, sample_label="audit", remainder_label="train", random_state=7
        )
        T = split["train"]
        t_keys = _frame(T)["row_id"].astype(int).tolist()

        linked = project.geco.create(
            "roomtemp",
            documents=T,
            text_field="text",
            metadata_fields=["year"],
            geometry=geometry,
            geometry_name="tfidf",
            projections={"umap": view},
        )

        assert linked.name == "roomtemp"
        assert linked.coder.data["row_id"].astype(int).tolist() == t_keys
        assert linked.coder.data["text"].tolist() == [f"document {i}" for i in t_keys]
        assert linked.coder.data["year"].astype(int).tolist() == [
            2020 + (i % 3) for i in t_keys
        ]
        np.testing.assert_array_equal(
            linked.coder._geometries["tfidf"]["matrix"].toarray(),
            geometry_values[t_keys, :].toarray(),
        )
        np.testing.assert_array_equal(
            linked.coder._views["umap"]["coordinates"],
            view_values[t_keys, :],
        )
        assert project.geco.list()[0]["name"] == "roomtemp"
        assert linked.launch() == "launched"
        assert linked.coder.launch_kwargs == {
            "host": "127.0.0.1",
            "port": 8050,
            "debug": False,
        }
        assert linked.launch(host="127.0.0.1", port=8051, debug=True) == "launched"
        assert linked.coder.launch_kwargs == {
            "host": "127.0.0.1",
            "port": 8051,
            "debug": True,
        }
        diagnostics = linked.resource_diagnostics()
        assert diagnostics["name"].tolist() == ["tfidf", "umap"]
        assert diagnostics["rows"].tolist() == [T.n_rows, T.n_rows]
        assert diagnostics["columns"].tolist() == [3, 2]
        assert diagnostics["finite"].tolist() == [True, True]

        L_T = linked.export_codes("room temperature", output_label="human_labels")
        assert L_T.descriptor["lineage"]["basis_artifact_ids"] == [T.artifact_id]
        labels = _frame(L_T, ["label"])
        assert labels["row_id"].astype(int).tolist() == t_keys[:4]
        assert labels["label"].astype(int).tolist() == [1, 0, 1, 0]

        refs = linked.predictors()
        assert refs == [
            GeCoPredictorRef(
                kind="classifier", id=17, code_id=4, name="room_temp_logistic"
            )
        ]
        predictor = linked.export_predictor(refs[0])
        assert isinstance(predictor, GeCoPredictor)
        L_F = project.translate(predictor, [geometry])["output"]
        predicted = _frame(L_F, ["prediction", "probability"])
        assert len(predicted) == 10
        assert predicted["probability"].between(0.0, 1.0).all()

        descriptor = linked.manifest
        assert descriptor["schema_version"] == 2
        assert descriptor["documents_artifact_id"] == T.artifact_id
        assert descriptor["created_with_geco_version"] == "0.next-test"
        assert "geometries" not in descriptor
        assert "projections" not in descriptor
        assert "geco_geometry_id" not in descriptor
        assert "geco_view_id" not in descriptor

        listing = project.geco.list()[0]
        assert listing["name"] == "roomtemp"
        assert listing["documents_artifact_id"] == T.artifact_id
        assert "geometries" not in listing
        assert "projections" not in listing

        ids = {"T": T.artifact_id, "L_T": L_T.artifact_id, "L_F": L_F.artifact_id}
    finally:
        project.close()


def test_three_record_linked_provider_is_thread_local_and_survives_project_close(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the same cross-thread boundary used by GeCo Dash callbacks."""
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.next-test")

    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "text": [
                "apple orchard fruit",
                "engine piston motor",
                "ocean coral reef",
            ],
            "kind": ["fruit", "machine", "sea"],
        }
    ).to_csv(csv_path, index=False)

    project_path = tmp_path / "tiny_project"
    project = teal.Project.create(project_path, name="tiny_geco")
    provider = None
    try:
        documents = project.read_csv(
            csv_path,
            text_fields="text",
            metadata_fields="kind",
            output_label="tiny_documents",
        )
        geometry = project.translate(
            CountVectorizer(text_field="text", min_df=1), documents
        )["output"]
        view = project.translate(SVD(n_components=2, random_state=7), geometry)[
            "output"
        ]

        linked = project.geco.create(
            "tiny",
            documents=documents,
            text_field="text",
            metadata_fields=["kind"],
            geometry=geometry,
            geometry_name="counts",
            projections={"svd2": view},
        )
        provider = linked.external_provider

        # Force both the original catalog and query engine to be used on the
        # notebook/main thread before worker-thread provider requests begin.
        project.list_artifacts()
        documents.query(data_columns=False, include_position=True, form="table")

        requested = [{"row_id": 2}, {"row_id": 0}, {"row_id": 1}]
        expected_geometry = geometry.get_matrix(positions=[2, 0, 1]).toarray()
        expected_view = np.asarray(view.get_matrix(positions=[2, 0, 1]), dtype=float)

        with ThreadPoolExecutor(max_workers=2) as pool:
            geometry_future = pool.submit(
                provider.geometry_matrix,
                {"artifact_id": geometry.artifact_id},
                requested,
            )
            view_future = pool.submit(
                provider.view_coordinates,
                {"artifact_id": view.artifact_id},
                requested,
            )
            threaded_geometry = geometry_future.result()
            threaded_view = view_future.result()

        assert sparse.issparse(threaded_geometry)
        np.testing.assert_array_equal(threaded_geometry.toarray(), expected_geometry)
        np.testing.assert_allclose(threaded_view, expected_view)
        assert threaded_view.dtype == np.float64
        assert threaded_view.flags.c_contiguous

        diagnostics = linked.resource_diagnostics()
        assert diagnostics[["name", "rows", "columns"]].to_dict("records") == [
            {"name": "counts", "rows": 3, "columns": expected_geometry.shape[1]},
            {"name": "svd2", "rows": 3, "columns": 2},
        ]
    finally:
        project.close()

    assert provider is not None
    # The provider stores the durable project path, so callbacks remain safe even
    # if the original notebook Project object has already been closed.
    with ThreadPoolExecutor(max_workers=1) as pool:
        after_close = pool.submit(
            provider.geometry_matrix,
            {"artifact_id": geometry.artifact_id},
            [{"row_id": 2}, {"row_id": 0}, {"row_id": 1}],
        ).result()
    np.testing.assert_array_equal(after_close.toarray(), expected_geometry)

    reopened = teal.Project.open(project_path)
    try:
        monkeypatch.setattr(
            bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder
        )
        linked2 = reopened.geco.open("tiny")
        assert isinstance(linked2.external_provider, TeALGeCoProvider)
        diagnostics2 = linked2.resource_diagnostics()
        assert diagnostics2[["name", "rows", "columns"]].to_dict("records") == [
            {"name": "counts", "rows": 3, "columns": expected_geometry.shape[1]},
            {"name": "svd2", "rows": 3, "columns": 2},
        ]
        with ThreadPoolExecutor(max_workers=1) as pool:
            reopened_matrix = pool.submit(
                linked2.external_provider.geometry_matrix,
                {"artifact_id": geometry.artifact_id},
                [{"row_id": 2}, {"row_id": 0}, {"row_id": 1}],
            ).result()
        np.testing.assert_array_equal(reopened_matrix.toarray(), expected_geometry)
    finally:
        reopened.close()


def test_linked_geco_add_resources_use_geco_registry_without_mutating_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.next-test")
    project = teal.Project.create(tmp_path / "project", name="geco_registry")
    try:
        F = _seed_table(project)
        geometry1 = project.translate(CountVectorizer(text_field="text", min_df=1), F)[
            "output"
        ]
        geometry2 = _seed_matrix(
            project, "g2", "g2", sparse.csr_matrix(np.ones((10, 2))), columns=["u", "v"]
        )
        view1 = _seed_matrix(
            project,
            "v1",
            "v1",
            np.column_stack([np.arange(10), np.arange(10)]),
            columns=["x", "y"],
        )
        view2 = _seed_matrix(
            project,
            "v2",
            "v2",
            np.column_stack([np.arange(10), -np.arange(10)]),
            columns=["x", "y"],
        )
        linked = project.geco.create(
            "registry",
            documents=F,
            text_field="text",
            geometry=geometry1,
            geometry_name="first",
            projections={"first_view": view1},
        )
        before = linked.manifest
        second_id = linked.add_geometry("second", geometry2)
        second_view_id = linked.add_projection("second_view", view2, geometry="second")
        assert second_id == 2
        assert second_view_id == 2
        assert linked.add_geometry("second", geometry2) == second_id
        assert (
            linked.add_projection("second_view", view2, geometry="second")
            == second_view_id
        )
        with pytest.raises(ValueError, match="Conflicting geometry registration"):
            linked.add_geometry("second", geometry1)
        with pytest.raises(ValueError, match="Conflicting view registration"):
            linked.add_projection("second_view", view1, geometry="second")
        assert linked.manifest == before
        geometry_records = linked.coder.geometries()
        assert [record["name"] for record in geometry_records] == ["first", "second"]
        assert [record["supports_text_transform"] for record in geometry_records] == [
            True,
            False,
        ]
        assert [record["name"] for record in linked.coder.views()] == [
            "first_view",
            "second_view",
        ]
        diagnostics = linked.resource_diagnostics()
        assert diagnostics["name"].tolist() == [
            "first",
            "second",
            "first_view",
            "second_view",
        ]
    finally:
        project.close()


def test_create_registration_failure_removes_new_workspace_and_link(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.next-test")

    def fail_view_registration(self, *, geometry_id, name, external_ref):
        raise RuntimeError("injected view registration failure")

    monkeypatch.setattr(
        _FakeGeometricCoder, "register_external_view", fail_view_registration
    )
    project = teal.Project.create(tmp_path / "project", name="geco_transaction")
    try:
        F = _seed_table(project)
        geometry = _seed_matrix(
            project, "g", "g", sparse.eye(10, 3, format="csr"), columns=["a", "b", "c"]
        )
        view = _seed_matrix(
            project,
            "v",
            "v",
            np.column_stack([np.arange(10), np.arange(10)]),
            columns=["x", "y"],
        )
        with pytest.raises(RuntimeError, match="injected view registration failure"):
            project.geco.create(
                "broken",
                documents=F,
                text_field="text",
                geometry=geometry,
                projections={"view": view},
            )
        assert not (
            project.storage.teal_dir / "geco" / "links" / "broken.json"
        ).exists()
        assert not (
            project.storage.teal_dir / "geco" / "workspaces" / "broken.geco"
        ).exists()
    finally:
        project.close()


def test_old_link_descriptor_is_rejected_without_migration(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "project", name="geco_old_link")
    try:
        link = project.storage.teal_dir / "geco" / "links" / "old.json"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.write_text(
            '{"schema_version": 1, "name": "old", "workspace_path": "workspaces/old.geco", '
            '"documents_artifact_id": "art_docs"}',
            encoding="utf-8",
        )
        with pytest.raises(
            GeCoIntegrationError,
            match=r"Incompatible pre-1\.0 linked GeCo descriptor schema 1",
        ):
            project.geco.open("old")
    finally:
        project.close()


def test_linked_geco_rejects_query_support_for_nonreplayable_geometry(
    tmp_path: Path, monkeypatch
):
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    project = teal.Project.create(tmp_path / "project", name="geco_no_query")
    try:
        F = _seed_table(project)
        geometry = _seed_matrix(
            project,
            "art_geometry",
            "tfidf",
            sparse.eye(10, 3, format="csr"),
            columns=["a", "b", "c"],
        )
        with pytest.raises(
            GeCoIntegrationError, match="cannot replay semantic queries"
        ):
            project.geco.create(
                "roomtemp",
                documents=F,
                text_field="text",
                geometry=geometry,
                supports_query=True,
            )
    finally:
        project.close()


def test_linked_geco_replayable_geometry_supports_query_and_text_transform(
    tmp_path: Path, monkeypatch
):
    import text_analysis_lab.integrations.geco as bridge

    _FakeGeometricCoder._registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: _FakeGeometricCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.next-test")
    project = teal.Project.create(tmp_path / "project", name="geco_query")
    try:
        F = _seed_table(project)
        counts = project.translate(CountVectorizer(text_field="text"), F)["output"]
        tfidf = project.translate(TfidfTransformer(), counts)["output"]
        linked = project.geco.create(
            "query_room",
            documents=F,
            text_field="text",
            geometry=tfidf,
            geometry_name="tfidf",
            supports_query=True,
        )
        assert isinstance(linked.external_provider, TeALGeCoProvider)
        geometry_record = linked.coder.geometries()[0]
        assert geometry_record["supports_query"] is True
        assert geometry_record["supports_text_transform"] is True
        query = linked.external_provider.transform_query(
            {"artifact_id": tfidf.artifact_id}, "linked document"
        )
        texts = linked.external_provider.transform_texts(
            {"artifact_id": tfidf.artifact_id}, ["linked document 1", "unseen words"]
        )
        assert query.shape == (1, len(tfidf.get_data_columns()))
        assert texts.shape == (2, len(tfidf.get_data_columns()))
    finally:
        project.close()
