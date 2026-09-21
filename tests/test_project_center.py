from __future__ import annotations

import json
from urllib.request import Request, urlopen

from text_analysis_lab import Project


def _json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_project_center_serves_tabs_and_initializes_project_memo(tmp_path):
    project = Project.create(tmp_path / "project", name="memo_test")
    assert project.get_project_memo() is None

    server = project.launch_memo_center(open_browser=False)
    try:
        with urlopen(server.url, timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "TeAL Project Center" in html
        assert 'data-tab="artifacts"' in html
        assert 'data-tab="memos"' in html
        assert 'id="targetType"' not in html
        assert 'id="targetId"' not in html
        assert 'class="graph-legend"' in html
        assert "graph-legend-shape artifact" in html
        assert "graph-legend-shape transformation" not in html
        assert "graph-legend-shape operation" in html
        assert '<option value="lineage">Lineage</option>' in html
        assert 'id="focusSelected"' in html
        assert "app.graph.lineage_edges" in html
        assert "function graphNodeMetrics(node)" in html
        assert "barycentric sweeps" in html
        assert server.memo_url.endswith("?tab=memos")

        state = _json(server.url + "api/state")
        assert state["project_name"] == "memo_test"
        project_memo = next(
            memo for memo in state["memos"] if memo["target_type"] == "project"
        )
        assert project_memo["target_id"] == "project"
        assert project_memo["title"] == "Project Memo"
        assert "memo_test" in project_memo["body"]
        assert "Project created:" in project_memo["body"]
    finally:
        server.close()
        project.close()


def test_project_center_versions_existing_project_memo(tmp_path):
    project = Project.create(tmp_path / "project", name="memo_test")
    server = project.launch_memo_center(open_browser=False)
    try:
        first = _post_json(
            server.url + "api/save",
            {
                "target_type": "project",
                "target_id": "project",
                "title": "Development",
                "body": "Initial memo",
            },
        )
        second = _post_json(
            server.url + "api/save",
            {
                "target_type": "project",
                "target_id": "project",
                "title": "Development",
                "body": "Revised memo",
            },
        )
        assert int(second["memo"]["memo_id"]) > int(first["memo"]["memo_id"])
        assert project.get_project_memo() == "Revised memo"

        history = _json(
            server.url + "api/history?target_type=project&target_id=project"
        )["versions"]
        assert [row["body"] for row in history[:2]] == [
            "Revised memo",
            "Initial memo",
        ]
    finally:
        project.close()
    assert server.closed


def test_project_center_creates_and_edits_standalone_memo(tmp_path):
    project = Project.create(tmp_path / "project", name="memo_test")
    server = project.launch_memo_center(open_browser=False)
    try:
        created = _post_json(
            server.url + "api/save",
            {
                "target_type": "standalone",
                "target_id": None,
                "title": "Construct notes",
                "body": "C0 is provisional.",
            },
        )["memo"]
        target_id = str(created["target_id"])
        assert target_id

        revised = _post_json(
            server.url + "api/save",
            {
                "target_type": "standalone",
                "target_id": target_id,
                "title": "Construct notes",
                "body": "C1 revises the boundary.",
            },
        )["memo"]
        assert str(revised["target_id"]) == target_id
        assert project.get_standalone_memo(target_id) == "C1 revises the boundary."
    finally:
        project.close()


def test_project_center_artifact_graph_and_artifact_memo(tmp_path):
    project = Project.create(tmp_path / "project", name="memo_test")
    source_id = "art_000001"
    output_id = "art_000002"
    operator_id = "op_000001"
    operation_id = "run_000001"
    project.catalog.register_artifact(
        artifact_id=source_id,
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
    )
    project.catalog.add_artifact_alias(source_id, "docs")
    project.catalog.register_artifact(
        artifact_id=output_id,
        artifact_type="table",
        label="sentences",
        lineage_mode="extended_key",
        status="complete",
        basis_artifact_ids=[source_id],
    )
    project.catalog.add_artifact_alias(output_id, "sentences")
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="translate",
    )
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="translate",
        operator_id=operator_id,
        status="complete",
    )
    project.catalog.add_operation_source(operation_id, "source", source_id)
    project.catalog.add_operation_output(
        operation_id,
        "output",
        output_id,
        ordinal=0,
    )
    operation_dir = project.storage.operation_dir(operation_id)
    operation_dir.mkdir(parents=True)
    (operation_dir / "operation.json").write_text(
        json.dumps(
            {
                "operation_id": operation_id,
                "translator_class": {
                    "module": "example.translators",
                    "qualname": "SentenceSegmenter",
                },
            }
        ),
        encoding="utf-8",
    )

    server = project.launch_memo_center(open_browser=False)
    try:
        graph = _json(server.url + "api/artifacts")
        assert {node["id"] for node in graph["artifact_nodes"]} == {
            source_id,
            output_id,
        }
        assert {node["id"] for node in graph["operation_nodes"]} == {operation_id}
        assert {
            (edge["source"], edge["target"]) for edge in graph["lineage_edges"]
        } == {(source_id, output_id)}
        assert {
            (edge["source"], edge["target"]) for edge in graph["provenance_edges"]
        } == {(source_id, operation_id), (operation_id, output_id)}
        assert {
            (edge["source"], edge["target"]) for edge in graph["overview_edges"]
        } == {(source_id, output_id)}
        assert graph["overview_edges"][0]["operation_id"] == operation_id
        assert graph["overview_edges"][0]["label"] == "SentenceSegmenter"
        operation_node = next(
            node for node in graph["operation_nodes"] if node["id"] == operation_id
        )
        assert operation_node["node_type"] == "operation"
        assert operation_node["operator_id"] == operator_id
        assert operation_node["label"] == "SentenceSegmenter"

        operation_detail = _json(
            server.url + f"api/operation?operation_id={operation_id}"
        )
        assert operation_detail["operation"]["operator_id"] == operator_id
        assert operation_detail["operator"]["operator_id"] == operator_id
        assert operation_detail["display_label"] == "SentenceSegmenter"
        assert operation_detail["operator_use_count"] == 1
        assert operation_detail["sources"][0]["source_artifact_id"] == source_id
        assert operation_detail["outputs"][0]["artifact_id"] == output_id

        detail = _json(server.url + f"api/artifact?artifact_id={output_id}")
        assert detail["aliases"] == ["sentences"]
        assert detail["memo"] is None
        assert detail["operation"]["operation_id"] == operation_id

        saved = _post_json(
            server.url + "api/save",
            {
                "target_type": "artifact",
                "target_id": output_id,
                "title": "Sentence artifact",
                "body": "Created during segmentation.",
            },
        )["memo"]
        assert saved["target_id"] == output_id

        detail = _json(server.url + f"api/artifact?artifact_id={output_id}")
        assert detail["memo"]["body"] == "Created during segmentation."
    finally:
        project.close()


def test_project_center_launch_aliases_open_expected_tabs(tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        "text_analysis_lab.gui.project_center.webbrowser.open",
        lambda url: opened.append(str(url)),
    )

    project = Project.create(tmp_path / "project", name="center_aliases")
    general = project.launch_project_center()
    memos = project.launch_memo_center()
    artifacts = project.launch_artifact_map()
    try:
        assert opened == [
            general.artifact_url,
            memos.memo_url,
            artifacts.artifact_url,
        ]
        assert general.url_for("memos") == general.memo_url
        assert general.url_for("artifacts") == general.artifact_url
    finally:
        project.close()

    assert general.closed
    assert memos.closed
    assert artifacts.closed


def test_project_center_artifact_size_and_descriptor_endpoint(tmp_path):
    project = Project.create(tmp_path / "project", name="artifact_details")
    artifact_id = "art_000001"
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
    )
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True)
    descriptor = {
        "artifact_id": artifact_id,
        "artifact_type": "table",
        "label": "documents",
        "status": "complete",
        "n_rows": 2,
        "primary_key": ["row_id"],
        "components": {},
        "lineage": {"lineage_mode": "new_key", "basis_artifact_ids": []},
    }
    (artifact_dir / "artifact.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    (artifact_dir / "payload.bin").write_bytes(b"x" * 2048)

    server = project.launch_artifact_map(open_browser=False)
    try:
        graph = _json(server.url + "api/artifacts")
        node = next(item for item in graph["nodes"] if item["id"] == artifact_id)
        assert node["size_bytes"] >= 2048
        assert node["size_display"].endswith(("KB", "MB"))

        detail = _json(server.url + f"api/artifact?artifact_id={artifact_id}")
        assert detail["size_bytes"] == node["size_bytes"]
        assert detail["descriptor_available"] is True
        assert detail["table_preview_available"] is True

        loaded = _json(server.url + f"api/artifact-json?artifact_id={artifact_id}")
        assert loaded["descriptor"] == descriptor
    finally:
        project.close()


def test_table_preview_queries_only_requested_page_and_components(
    tmp_path, monkeypatch
):
    import pandas as pd

    from text_analysis_lab.core.project import Project as ProjectClass
    from text_analysis_lab.gui.project_center import _table_preview_payload

    calls: list[dict] = []

    class ArtifactTypeValue:
        value = "table"

    class FakeArtifact:
        artifact_type = ArtifactTypeValue()
        n_rows = 63

        def query(self, **kwargs):
            calls.append(kwargs)
            positions = list(kwargs["positions"])
            return pd.DataFrame(
                {
                    "row_id": positions,
                    "group": ["g"] * len(positions),
                }
            )

    class FakeProject:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def get_artifact(self, artifact_id, *, include_deleted=False):
            assert artifact_id == "art_000001"
            assert include_deleted is True
            return FakeArtifact()

    monkeypatch.setattr(
        ProjectClass,
        "open",
        classmethod(lambda cls, path: FakeProject()),
    )

    manifest_path = tmp_path / ".teal" / "manifest.json"
    payload = _table_preview_payload(
        manifest_path,
        {
            "artifact_id": ["art_000001"],
            "page": ["2"],
            "page_size": ["25"],
            "keys": ["true"],
            "data": ["false"],
            "metadata": ["true"],
            "full_metadata": ["true"],
        },
    )

    assert payload["page"] == 2
    assert payload["page_count"] == 3
    assert payload["n_rows"] == 63
    assert calls == [
        {
            "key_columns": True,
            "data_columns": False,
            "metadata_columns": True,
            "metadata_mode": "full",
            "positions": list(range(25, 50)),
            "form": "table",
            "include_position": False,
        }
    ]
    assert payload["rows"][0] == [25, "g"]


def test_matrix_artifact_detail_reports_dimensions(tmp_path, monkeypatch):
    from text_analysis_lab.core.project import Project as ProjectClass
    from text_analysis_lab.gui.project_center import _artifact_detail_payload

    project = Project.create(tmp_path / "project", name="matrix_detail")
    artifact_id = "art_000001"
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="dense_matrix",
        label="embeddings",
        lineage_mode="new_key",
        status="complete",
    )
    artifact_dir = project.storage.artifact_dir(artifact_id)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "artifact.json").write_text("{}", encoding="utf-8")

    class FakeArtifact:
        n_rows = 12

        def get_data_columns(self):
            return ["a", "b", "c", "d"]

    class FakeProject:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def get_artifact(self, artifact_id, *, include_deleted=False):
            return FakeArtifact()

    monkeypatch.setattr(
        ProjectClass,
        "open",
        classmethod(lambda cls, path: FakeProject()),
    )

    try:
        detail = _artifact_detail_payload(
            project.storage.catalog_dir,
            project.storage.manifest_path,
            artifact_id,
        )
        assert detail["dimensions"] == {"rows": 12, "columns": 4}
    finally:
        project.close()


def test_project_center_operation_and_operator_storage_and_json(tmp_path):
    project = Project.create(tmp_path / "project", name="operation_details")
    source_id = "art_000001"
    output_id = "art_000002"
    operator_id = "optr_000001"
    operation_id = "run_000001"
    project.catalog.register_artifact(
        artifact_id=source_id,
        artifact_type="table",
        label="source",
        lineage_mode="new_key",
        status="complete",
    )
    project.catalog.register_artifact(
        artifact_id=output_id,
        artifact_type="table",
        label="output",
        lineage_mode="preserved_key",
        status="complete",
        basis_artifact_ids=[source_id],
    )
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="translate",
    )
    project.catalog.add_operator_alias(operator_id, "translator")
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="translate",
        operator_id=operator_id,
        status="complete",
    )
    project.catalog.add_operation_source(operation_id, "source", source_id)
    project.catalog.add_operation_output(operation_id, "output", output_id, ordinal=0)

    operation_dir = project.storage.operation_dir(operation_id)
    operator_dir = project.storage.operator_dir(operator_id)
    operation_dir.mkdir(parents=True)
    operator_dir.mkdir(parents=True)
    operation_descriptor = {"operation_id": operation_id, "route": "transform"}
    operator_descriptor = {"operator_id": operator_id, "state": {"fitted": True}}
    (operation_dir / "operation.json").write_text(
        json.dumps(operation_descriptor), encoding="utf-8"
    )
    (operator_dir / "operator.json").write_text(
        json.dumps(operator_descriptor), encoding="utf-8"
    )
    (operation_dir / "checkpoint.bin").write_bytes(b"x" * 1024)
    (operator_dir / "model.bin").write_bytes(b"y" * 4096)

    server = project.launch_artifact_map(open_browser=False)
    try:
        graph = _json(server.url + "api/artifacts")
        assert graph["storage"]["operations_bytes"] >= 1024
        assert graph["storage"]["operators_bytes"] >= 4096
        assert graph["storage"]["total_bytes"] >= (
            graph["storage"]["operations_bytes"] + graph["storage"]["operators_bytes"]
        )

        detail = _json(server.url + f"api/operation?operation_id={operation_id}")
        assert detail["size_bytes"] >= 1024
        assert detail["operator_size_bytes"] >= 4096
        assert detail["operator_aliases"] == ["translator"]
        assert detail["descriptor_available"] is True
        assert detail["operator_descriptor_available"] is True

        loaded_operation = _json(
            server.url + f"api/operation-json?operation_id={operation_id}"
        )
        loaded_operator = _json(
            server.url + f"api/operator-json?operator_id={operator_id}"
        )
        assert loaded_operation["descriptor"] == operation_descriptor
        assert loaded_operator["descriptor"] == operator_descriptor
    finally:
        project.close()


def test_project_center_end_session_endpoint_stops_server(tmp_path):
    project = Project.create(tmp_path / "project", name="end_session")
    server = project.launch_project_center(open_browser=False)
    assert not server.closed

    result = _post_json(server.url + "api/end-session", {})
    assert result == {"status": "ending"}
    assert server.wait(timeout=5.0)
    assert server.closed

    project.close()
