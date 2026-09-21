"""Local browser-based project interface for TeAL projects."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from text_analysis_lab.core.catalog import ProjectCatalog


class ProjectCenterServer:
    """Handle for one running TeAL Project Center server."""

    def __init__(
        self,
        *,
        catalog_dir: str | Path,
        project_name: str,
        manifest_path: str | Path,
        port: int = 0,
    ) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.project_name = str(project_name)
        self.manifest_path = Path(manifest_path)
        _ensure_project_memo(
            catalog_dir=self.catalog_dir,
            project_name=self.project_name,
            manifest_path=self.manifest_path,
        )
        handler = _handler_factory(
            catalog_dir=self.catalog_dir,
            project_name=self.project_name,
            manifest_path=self.manifest_path,
            end_session=self._request_close,
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"teal-project-center-{self._server.server_port}",
            daemon=True,
        )
        self._closed = False
        self._closed_event = threading.Event()
        self._close_lock = threading.Lock()

    @property
    def url(self) -> str:
        """Return the local base URL for this Project Center."""
        return f"http://127.0.0.1:{self._server.server_port}/"

    def url_for(self, tab: str) -> str:
        """Return the local URL opening directly to one Project Center tab."""
        if tab not in {"artifacts", "memos"}:
            raise ValueError("Project Center tab must be 'artifacts' or 'memos'.")
        return f"{self.url}?tab={tab}"

    @property
    def memo_url(self) -> str:
        """Return the local URL opening directly to the Memos tab."""
        return self.url_for("memos")

    @property
    def artifact_url(self) -> str:
        """Return the local URL opening directly to the Artifacts tab."""
        return self.url_for("artifacts")

    @property
    def closed(self) -> bool:
        """Return whether the server has been closed."""
        return self._closed

    def start(self) -> ProjectCenterServer:
        """Start serving requests in a daemon thread and return this handle."""
        if not self._thread.is_alive() and not self._closed:
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop the local server."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._server.shutdown()
            self._server.server_close()
            if (
                self._thread.is_alive()
                and threading.current_thread() is not self._thread
            ):
                self._thread.join(timeout=5.0)
        finally:
            self._closed_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the Project Center session ends."""
        return self._closed_event.wait(timeout)

    def _request_close(self) -> None:
        """Schedule server shutdown after an HTTP response has been sent."""
        threading.Thread(
            target=self.close,
            name=f"teal-project-center-close-{self._server.server_port}",
            daemon=True,
        ).start()

    def __enter__(self) -> ProjectCenterServer:  # noqa: PYI034 - Python 3.10 support
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


# Backward-compatible name for callers that imported the first implementation.
MemoCenterServer = ProjectCenterServer


def launch_project_center(
    *,
    catalog_dir: str | Path,
    project_name: str,
    manifest_path: str | Path,
    initial_tab: str = "artifacts",
    port: int = 0,
    open_browser: bool = True,
) -> ProjectCenterServer:
    """Launch the localhost-only TeAL Project Center."""
    server = ProjectCenterServer(
        catalog_dir=catalog_dir,
        project_name=project_name,
        manifest_path=manifest_path,
        port=port,
    ).start()
    destination = server.url_for(initial_tab)
    if open_browser:
        webbrowser.open(destination)
    return server


def launch_memo_center(
    *,
    catalog_dir: str | Path,
    project_name: str,
    manifest_path: str | Path,
    port: int = 0,
    open_browser: bool = True,
) -> ProjectCenterServer:
    """Backward-compatible launcher opening the Project Center on Memos."""
    return launch_project_center(
        catalog_dir=catalog_dir,
        project_name=project_name,
        manifest_path=manifest_path,
        initial_tab="memos",
        port=port,
        open_browser=open_browser,
    )


def _handler_factory(
    *,
    catalog_dir: Path,
    project_name: str,
    manifest_path: Path,
    end_session: Callable[[], None],
):
    class ProjectCenterHandler(BaseHTTPRequestHandler):
        server_version = "TeALProjectCenter/1.0"

        def log_message(self, format: str, *args: object) -> None:
            # Keep notebook/terminal output quiet during ordinary UI use.
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(_INDEX_HTML)
                    return
                if parsed.path == "/api/state":
                    self._send_json(_state_payload(catalog_dir, project_name))
                    return
                if parsed.path == "/api/history":
                    query = parse_qs(parsed.query)
                    target_type = _single_query_value(query, "target_type")
                    target_id = _single_query_value(query, "target_id")
                    with _catalog(catalog_dir) as catalog:
                        versions = catalog.get_memo_all_versions(
                            target_type=target_type,
                            target_id=target_id,
                        )
                    self._send_json({"versions": versions})
                    return
                if parsed.path == "/api/artifacts":
                    self._send_json(_artifact_graph_payload(catalog_dir, manifest_path))
                    return
                if parsed.path == "/api/artifact":
                    query = parse_qs(parsed.query)
                    artifact_id = _single_query_value(query, "artifact_id")
                    self._send_json(
                        _artifact_detail_payload(
                            catalog_dir, manifest_path, artifact_id
                        )
                    )
                    return
                if parsed.path == "/api/artifact-json":
                    query = parse_qs(parsed.query)
                    artifact_id = _single_query_value(query, "artifact_id")
                    self._send_json(_artifact_json_payload(manifest_path, artifact_id))
                    return
                if parsed.path == "/api/operation":
                    query = parse_qs(parsed.query)
                    operation_id = _single_query_value(query, "operation_id")
                    self._send_json(
                        _operation_detail_payload(
                            catalog_dir, manifest_path, operation_id
                        )
                    )
                    return
                if parsed.path == "/api/operation-json":
                    query = parse_qs(parsed.query)
                    operation_id = _single_query_value(query, "operation_id")
                    self._send_json(
                        _operation_json_payload(manifest_path, operation_id)
                    )
                    return
                if parsed.path == "/api/operator-json":
                    query = parse_qs(parsed.query)
                    operator_id = _single_query_value(query, "operator_id")
                    self._send_json(_operator_json_payload(manifest_path, operator_id))
                    return
                if parsed.path == "/api/table-preview":
                    query = parse_qs(parsed.query)
                    self._send_json(_table_preview_payload(manifest_path, query))
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "Unknown Project Center route.")
            except (KeyError, TypeError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary must return JSON errors
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/end-session":
                    self._send_json({"status": "ending"})
                    end_session()
                    return
                if parsed.path != "/api/save":
                    self._send_error(
                        HTTPStatus.NOT_FOUND, "Unknown Project Center route."
                    )
                    return
                payload = self._read_json_body()
                result = _save_memo(catalog_dir, payload)
                self._send_json(result, status=HTTPStatus.CREATED)
            except (KeyError, TypeError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary must return JSON errors
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def _read_json_body(self) -> Mapping[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be valid UTF-8 JSON.") from exc
            if not isinstance(payload, Mapping):
                raise TypeError("Request body must be a JSON object.")
            return payload

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(
            self,
            payload: Mapping[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status=status)

    return ProjectCenterHandler


class _CatalogContext:
    def __init__(self, catalog_dir: Path) -> None:
        self.catalog = ProjectCatalog(catalog_dir)

    def __enter__(self) -> ProjectCatalog:
        return self.catalog

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.catalog.close()


def _catalog(catalog_dir: Path) -> _CatalogContext:
    return _CatalogContext(catalog_dir)


def _single_query_value(query: Mapping[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values or len(values) != 1 or not values[0]:
        raise ValueError(f"Query parameter {name!r} is required exactly once.")
    return values[0]


def _ensure_project_memo(
    *,
    catalog_dir: Path,
    project_name: str,
    manifest_path: Path,
) -> None:
    with _catalog(catalog_dir) as catalog:
        if catalog.get_memo(target_type="project", target_id="project") is not None:
            return
        created_at = None
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            project = payload.get("project", {}) if isinstance(payload, Mapping) else {}
            if isinstance(project, Mapping):
                created_at = project.get("created_at")
        lines = [f"# {project_name}", "", "TeAL project memo."]
        if created_at:
            lines.extend(["", f"Project created: {created_at}"])
        catalog.add_memo(
            target_type="project",
            target_id="project",
            title="Project Memo",
            body="\n".join(lines),
        )


def _state_payload(catalog_dir: Path, project_name: str) -> dict[str, Any]:
    with _catalog(catalog_dir) as catalog:
        memos = catalog.list_memos(latest_only=True)
        artifacts = catalog.list_artifacts(include_deleted=True)
        operators = catalog.list_operators()
        operations = catalog.list_operations()

        targets = {
            "project": {"project": project_name},
            "standalone": {},
            "artifact": {
                str(row["artifact_id"]): _artifact_label(catalog, row)
                for row in artifacts
            },
            "operator": {
                str(row["operator_id"]): _operator_label(catalog, row)
                for row in operators
            },
            "operation": {
                str(row["operation_id"]): (
                    f"{row['operation_type']} · {row['operation_id']}"
                )
                for row in operations
            },
        }

    return {
        "project_name": project_name,
        "memos": memos,
        "target_labels": targets,
    }


def _artifact_label(catalog: ProjectCatalog, row: Mapping[str, Any]) -> str:
    artifact_id = str(row["artifact_id"])
    aliases = catalog.aliases_for_artifact(artifact_id)
    preferred = aliases[0] if aliases else str(row["label"])
    return f"{preferred} · {row['artifact_type']} · {artifact_id}"


def _operator_label(catalog: ProjectCatalog, row: Mapping[str, Any]) -> str:
    operator_id = str(row["operator_id"])
    aliases = catalog.aliases_for_operator(operator_id)
    preferred = aliases[0] if aliases else str(row["operation_type"])
    return f"{preferred} · {operator_id}"


def _operation_graph_label(
    catalog: ProjectCatalog,
    operation: Mapping[str, Any],
    operation_dir: Path,
) -> str:
    """Return the most useful short human label for a provenance operation."""

    operator_id = str(operation["operator_id"])
    aliases = catalog.aliases_for_operator(operator_id)
    if aliases:
        return aliases[0]

    descriptor_path = operation_dir / "operation.json"
    if descriptor_path.is_file():
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            descriptor = None
        if isinstance(descriptor, Mapping):
            translator_class = descriptor.get("translator_class")
            if isinstance(translator_class, Mapping):
                qualname = translator_class.get("qualname")
                if isinstance(qualname, str) and qualname.strip():
                    return qualname.rsplit(".", 1)[-1]

    return str(operation["operation_type"])


def _overview_edge_label(
    operation_label: str,
    source_label: str,
    output_label: str,
) -> str:
    generic_source = {"source", "input"}
    generic_output = {"output"}
    source_role = "" if source_label in generic_source else source_label
    output_role = "" if output_label in generic_output else output_label
    # User-facing output aliases can also be stored as port labels. Those are
    # already visible on the Artifact node and make poor edge annotations.
    if len(source_role) > 24:
        source_role = ""
    if len(output_role) > 24:
        output_role = ""
    if source_role and output_role:
        return f"{operation_label} · {source_role}→{output_role}"
    if source_role:
        return f"{operation_label} · {source_role}"
    if output_role:
        return f"{operation_label} · {output_role}"
    return operation_label


def _artifact_graph_payload(
    catalog_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    teal_dir = manifest_path.parent
    artifacts_dir = teal_dir / "artifacts"
    operations_dir = teal_dir / "operations"
    operators_dir = teal_dir / "operators"
    with _catalog(catalog_dir) as catalog:
        artifacts = catalog.list_artifacts()
        artifact_nodes = []
        for row in artifacts:
            artifact_id = str(row["artifact_id"])
            aliases = catalog.aliases_for_artifact(artifact_id)
            preferred = aliases[0] if aliases else str(row["label"])
            size_bytes = _directory_size(artifacts_dir / artifact_id)
            artifact_nodes.append(
                {
                    "id": artifact_id,
                    "node_type": "artifact",
                    "label": preferred,
                    "artifact_type": str(row["artifact_type"]),
                    "status": str(row["status"]),
                    "lineage_mode": str(row["lineage_mode"]),
                    "size_bytes": size_bytes,
                    "size_display": _format_bytes(size_bytes),
                }
            )

        live_ids = {node["id"] for node in artifact_nodes}
        lineage_edges = []
        for node in artifact_nodes:
            for basis in catalog.artifact_basis(node["id"]):
                source = str(basis["basis_artifact_id"])
                if source in live_ids:
                    lineage_edges.append(
                        {
                            "source": source,
                            "target": node["id"],
                            "label": str(node["lineage_mode"]),
                        }
                    )

        operation_nodes = []
        provenance_edges = []
        overview_edges = []
        operator_ids: set[str] = set()
        for operation in catalog.list_operations():
            operation_id = str(operation["operation_id"])
            operator_id = str(operation["operator_id"])
            operator_ids.add(operator_id)
            size_bytes = _directory_size(operations_dir / operation_id)
            operation_label = _operation_graph_label(
                catalog,
                operation,
                operations_dir / operation_id,
            )
            operation_nodes.append(
                {
                    "id": operation_id,
                    "node_type": "operation",
                    "label": operation_label,
                    "operation_type": str(operation["operation_type"]),
                    "operator_id": operator_id,
                    "status": str(operation["status"]),
                    "size_bytes": size_bytes,
                    "size_display": _format_bytes(size_bytes),
                }
            )
            source_rows = catalog.operation_sources(operation_id)
            output_rows = catalog.operation_outputs(operation_id)
            for source_row in source_rows:
                source = str(source_row["source_artifact_id"])
                if source in live_ids:
                    provenance_edges.append(
                        {
                            "source": source,
                            "target": operation_id,
                            "label": str(source_row["source_label"]),
                        }
                    )
            for output_row in output_rows:
                output = str(output_row["artifact_id"])
                if output in live_ids:
                    provenance_edges.append(
                        {
                            "source": operation_id,
                            "target": output,
                            "label": str(output_row["output_label"]),
                        }
                    )

            for source_row in source_rows:
                source = str(source_row["source_artifact_id"])
                if source not in live_ids:
                    continue
                for output_row in output_rows:
                    output = str(output_row["artifact_id"])
                    if output not in live_ids:
                        continue
                    source_label = str(source_row["source_label"])
                    output_label = str(output_row["output_label"])
                    overview_edges.append(
                        {
                            "source": source,
                            "target": output,
                            "operation_id": operation_id,
                            "operation_label": operation_label,
                            "source_label": source_label,
                            "output_label": output_label,
                            "label": _overview_edge_label(
                                operation_label,
                                source_label,
                                output_label,
                            ),
                        }
                    )

    artifact_bytes = sum(int(node["size_bytes"]) for node in artifact_nodes)
    operation_bytes = sum(int(node["size_bytes"]) for node in operation_nodes)
    operator_bytes = sum(
        _directory_size(operators_dir / operator_id) for operator_id in operator_ids
    )
    total_bytes = _directory_size(teal_dir)
    other_bytes = max(
        0, total_bytes - artifact_bytes - operation_bytes - operator_bytes
    )
    storage = {
        "artifacts_bytes": artifact_bytes,
        "artifacts_display": _format_bytes(artifact_bytes),
        "operations_bytes": operation_bytes,
        "operations_display": _format_bytes(operation_bytes),
        "operators_bytes": operator_bytes,
        "operators_display": _format_bytes(operator_bytes),
        "other_bytes": other_bytes,
        "other_display": _format_bytes(other_bytes),
        "total_bytes": total_bytes,
        "total_display": _format_bytes(total_bytes),
    }

    return {
        "nodes": [*artifact_nodes, *operation_nodes],
        "artifact_nodes": artifact_nodes,
        "operation_nodes": operation_nodes,
        "lineage_edges": lineage_edges,
        "overview_edges": overview_edges,
        "provenance_edges": provenance_edges,
        "storage": storage,
    }


def _artifact_detail_payload(
    catalog_dir: Path,
    manifest_path: Path,
    artifact_id: str,
) -> dict[str, Any]:
    with _catalog(catalog_dir) as catalog:
        row = catalog.resolve_artifact(artifact_id, include_deleted=True)
        aliases = catalog.aliases_for_artifact(artifact_id)
        basis = catalog.artifact_basis(artifact_id)
        operation = catalog.operation_for_artifact(artifact_id)
        sources = (
            []
            if operation is None
            else catalog.operation_sources(str(operation["operation_id"]))
        )
        memo = catalog.get_memo(target_type="artifact", target_id=artifact_id)

    artifact_dir = manifest_path.parent / "artifacts" / artifact_id
    size_bytes = _directory_size(artifact_dir)
    artifact_type = str(row["artifact_type"])
    dimensions = None
    if artifact_dir.exists() and artifact_type in {"dense_matrix", "sparse_matrix"}:
        from text_analysis_lab.core.project import Project

        with Project.open(manifest_path.parent.parent) as project:
            artifact = project.get_artifact(artifact_id, include_deleted=True)
            dimensions = {
                "rows": artifact.n_rows,
                "columns": len(artifact.get_data_columns()),
            }

    return {
        "artifact": row,
        "aliases": aliases,
        "basis": basis,
        "operation": operation,
        "sources": sources,
        "memo": memo,
        "size_bytes": size_bytes,
        "size_display": _format_bytes(size_bytes),
        "dimensions": dimensions,
        "descriptor_available": (artifact_dir / "artifact.json").is_file(),
        "table_preview_available": artifact_type == "table" and artifact_dir.exists(),
    }


def _artifact_json_payload(
    manifest_path: Path,
    artifact_id: str,
) -> dict[str, Any]:
    descriptor_path = manifest_path.parent / "artifacts" / artifact_id / "artifact.json"
    if not descriptor_path.is_file():
        raise ValueError(
            f"Artifact {artifact_id!r} does not have an artifact.json file."
        )
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read artifact descriptor for {artifact_id!r}."
        ) from exc
    if not isinstance(descriptor, Mapping):
        raise TypeError(
            f"Artifact descriptor for {artifact_id!r} is not a JSON object."
        )
    return {"artifact_id": artifact_id, "descriptor": descriptor}


def _operation_detail_payload(
    catalog_dir: Path,
    manifest_path: Path,
    operation_id: str,
) -> dict[str, Any]:
    teal_dir = manifest_path.parent
    operation_dir = teal_dir / "operations" / operation_id
    with _catalog(catalog_dir) as catalog:
        operation = catalog.get_operation(operation_id)
        sources = catalog.operation_sources(operation_id)
        outputs = catalog.operation_outputs(operation_id)
        operator_id = str(operation["operator_id"])
        operator = catalog.resolve_operator(operator_id, include_deleted=True)
        operator_aliases = catalog.aliases_for_operator(operator_id)
        operator_use_count = len(catalog.operations_using_operator(operator_id))
        display_label = _operation_graph_label(catalog, operation, operation_dir)

    operator_dir = teal_dir / "operators" / operator_id
    operation_size = _directory_size(operation_dir)
    operator_size = _directory_size(operator_dir)
    return {
        "operation": operation,
        "display_label": display_label,
        "sources": sources,
        "outputs": outputs,
        "size_bytes": operation_size,
        "size_display": _format_bytes(operation_size),
        "descriptor_available": (operation_dir / "operation.json").is_file(),
        "operator": operator,
        "operator_aliases": operator_aliases,
        "operator_use_count": operator_use_count,
        "operator_size_bytes": operator_size,
        "operator_size_display": _format_bytes(operator_size),
        "operator_descriptor_available": (operator_dir / "operator.json").is_file(),
    }


def _json_file_payload(
    path: Path,
    *,
    object_name: str,
    object_id: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"{object_name} {object_id!r} does not have a JSON descriptor."
        )
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read {object_name.lower()} descriptor for {object_id!r}."
        ) from exc
    if not isinstance(descriptor, Mapping):
        raise TypeError(
            f"{object_name} descriptor for {object_id!r} is not a JSON object."
        )
    return {"descriptor": descriptor}


def _operation_json_payload(
    manifest_path: Path,
    operation_id: str,
) -> dict[str, Any]:
    payload = _json_file_payload(
        manifest_path.parent / "operations" / operation_id / "operation.json",
        object_name="Operation",
        object_id=operation_id,
    )
    return {"operation_id": operation_id, **payload}


def _operator_json_payload(
    manifest_path: Path,
    operator_id: str,
) -> dict[str, Any]:
    payload = _json_file_payload(
        manifest_path.parent / "operators" / operator_id / "operator.json",
        object_name="Operator",
        object_id=operator_id,
    )
    return {"operator_id": operator_id, **payload}


def _table_preview_payload(
    manifest_path: Path,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    artifact_id = _single_query_value(query, "artifact_id")
    page = _positive_int_query(query, "page", default=1)
    page_size = _positive_int_query(query, "page_size", default=25, maximum=250)
    show_keys = _bool_query(query, "keys", default=True)
    show_data = _bool_query(query, "data", default=True)
    show_metadata = _bool_query(query, "metadata", default=False)
    full_metadata = _bool_query(query, "full_metadata", default=False)
    if full_metadata:
        show_metadata = True
    if not any((show_keys, show_data, show_metadata)):
        raise ValueError("Table preview must display at least one component.")

    from text_analysis_lab.core.project import Project

    with Project.open(manifest_path.parent.parent) as project:
        artifact = project.get_artifact(artifact_id, include_deleted=True)
        if artifact.artifact_type.value != "table":
            raise ValueError(
                f"Artifact {artifact_id!r} is {artifact.artifact_type.value!r}, not a table."
            )
        n_rows = artifact.n_rows
        if n_rows is None:
            raise ValueError(
                f"Table artifact {artifact_id!r} does not record its row count."
            )
        page_count = max(1, (int(n_rows) + page_size - 1) // page_size)
        page = min(page, page_count)
        start = (page - 1) * page_size
        stop = min(start + page_size, int(n_rows))
        positions = list(range(start, stop)) if start < int(n_rows) else []
        metadata_mode = (
            "full" if full_metadata else ("local" if show_metadata else "none")
        )
        frame = artifact.query(
            key_columns=show_keys,
            data_columns=show_data,
            metadata_columns=show_metadata,
            metadata_mode=metadata_mode,
            positions=positions,
            form="table",
            include_position=False,
        )

    split = json.loads(
        frame.to_json(
            orient="split",
            date_format="iso",
            default_handler=str,
        )
    )
    return {
        "artifact_id": artifact_id,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "n_rows": int(n_rows),
        "columns": [str(value) for value in split.get("columns", [])],
        "rows": split.get("data", []),
        "options": {
            "keys": show_keys,
            "data": show_data,
            "metadata": show_metadata,
            "full_metadata": full_metadata,
        },
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(size: int) -> str:
    value = float(max(0, int(size)))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(size)} B"


def _positive_int_query(
    query: Mapping[str, list[str]],
    name: str,
    *,
    default: int,
    maximum: int | None = None,
) -> int:
    values = query.get(name)
    raw = str(default) if not values else values[-1]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Query parameter {name!r} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"Query parameter {name!r} must be positive.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Query parameter {name!r} may not exceed {maximum}.")
    return value


def _bool_query(
    query: Mapping[str, list[str]],
    name: str,
    *,
    default: bool,
) -> bool:
    values = query.get(name)
    if not values:
        return default
    raw = values[-1].strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Query parameter {name!r} must be boolean.")


def _save_memo(catalog_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target_type = str(payload.get("target_type", ""))
    target_id_raw = payload.get("target_id")
    target_id = None if target_id_raw is None else str(target_id_raw)
    title_raw = payload.get("title")
    title = None if title_raw is None else str(title_raw)
    body = str(payload.get("body", ""))

    with _catalog(catalog_dir) as catalog:
        memo_id = catalog.add_memo(
            target_type=target_type,
            target_id=target_id,
            title=title,
            body=body,
        )
        if target_id is None:
            row = next(
                (
                    item
                    for item in catalog.list_memos(
                        target_type="standalone",
                        latest_only=False,
                    )
                    if int(item["memo_id"]) == memo_id
                ),
                None,
            )
            if row is None:
                raise RuntimeError(
                    "Saved memo could not be recovered from the catalog."
                )
            target_id = str(row["target_id"])
        current = catalog.get_memo(target_type=target_type, target_id=target_id)

    if current is None:
        raise RuntimeError("Saved memo could not be recovered from the catalog.")
    return {"memo": current}


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TeAL Project Center</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f5f6f8;
  --panel: #ffffff;
  --text: #1d2430;
  --muted: #687386;
  --line: #d9dee7;
  --accent: #315f86;
  --accent-soft: #e8f0f7;
  --danger: #a33a3a;
  --success: #2e6b45;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15181d;
    --panel: #1e232a;
    --text: #edf1f5;
    --muted: #a5afbc;
    --line: #39414c;
    --accent: #88b6dd;
    --accent-soft: #263b4d;
    --danger: #ef9a9a;
    --success: #8bc49f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
button, input, select, textarea { font: inherit; }
button { cursor: pointer; }
input, select, textarea { border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); padding: 8px 10px; }
button { border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); padding: 8px 11px; }
button.primary { background: var(--accent); color: white; border-color: var(--accent); }
button.active { background: var(--accent-soft); border-color: var(--accent); }
.app { height: 100vh; display: grid; grid-template-rows: auto minmax(0, 1fr); }
.header { background: var(--panel); border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 18px; padding: 10px 16px; }
.brand { min-width: 220px; }
.brand strong { display: block; font-size: 18px; }
.brand span { color: var(--muted); font-size: 12px; }
.tabs { display: flex; gap: 5px; }
.tab-button { border: 0; background: transparent; padding: 8px 12px; }
.tab-button.active { color: var(--accent); border-bottom: 2px solid var(--accent); border-radius: 0; background: transparent; }
.global-status { margin-left: auto; color: var(--muted); font-size: 13px; }
.global-status.error { color: var(--danger); }
.tab { min-height: 0; display: none; }
.tab.active { display: block; height: 100%; }

/* Memo Center */
.memo-shell { height: 100%; display: grid; grid-template-columns: 320px minmax(0, 1fr); }
.sidebar { border-right: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-height: 0; }
.toolbar { padding: 12px; display: grid; gap: 8px; border-bottom: 1px solid var(--line); }
.toolbar-row { display: flex; gap: 8px; }
.toolbar input, .toolbar select { width: 100%; }
.memo-list { overflow: auto; padding: 8px; }
.memo-item { display: block; width: 100%; text-align: left; margin: 0 0 4px; padding: 10px; border: 0; background: transparent; border-radius: 6px; }
.memo-item:hover, .memo-item.active { background: var(--accent-soft); }
.memo-title { font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.memo-meta { color: var(--muted); font-size: 12px; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.empty { color: var(--muted); padding: 16px 10px; }
.memo-main { min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); }
.memo-topbar { background: var(--panel); border-bottom: 1px solid var(--line); padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.target-note { color: var(--muted); font-size: 12px; margin-top: 2px; }
.editor { min-height: 0; overflow: auto; padding: 18px; }
.editor-card { max-width: 1050px; margin: 0 auto; background: var(--panel); border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
.fields { display: grid; grid-template-columns: 90px minmax(0, 1fr); gap: 10px; padding: 16px; border-bottom: 1px solid var(--line); }
.fields label { color: var(--muted); align-self: center; }
.fields input { width: 100%; }
.editor-mode { padding: 10px 16px 0; display: flex; gap: 6px; }
.body-wrap { padding: 12px 16px 16px; }
textarea.memo-body { width: 100%; min-height: 54vh; resize: vertical; line-height: 1.55; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.preview { min-height: 54vh; border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; background: var(--panel); overflow-wrap: anywhere; }
.preview[hidden], textarea[hidden] { display: none; }
.markdown-body h1, .markdown-body h2, .markdown-body h3 { line-height: 1.25; margin: 1.2em 0 .5em; }
.markdown-body h1:first-child, .markdown-body h2:first-child, .markdown-body h3:first-child { margin-top: 0; }
.markdown-body p { margin: .7em 0; }
.markdown-body code { background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; }
.markdown-body pre { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 10px; overflow: auto; }
.markdown-body pre code { border: 0; padding: 0; }
.markdown-body blockquote { margin: .8em 0; padding-left: 12px; border-left: 3px solid var(--line); color: var(--muted); }
.markdown-body ul, .markdown-body ol { padding-left: 24px; }
.actions { padding: 12px 16px; border-top: 1px solid var(--line); display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
.history { border-top: 1px solid var(--line); padding: 12px 16px 16px; display: none; }
.history.open { display: block; }
.history h2 { font-size: 15px; margin: 0 0 8px; }
.history-row { width: 100%; display: flex; justify-content: space-between; gap: 12px; text-align: left; margin: 4px 0; border: 1px solid var(--line); }
.history-row span:last-child { color: var(--muted); font-size: 12px; }

/* Artifacts */
.artifact-shell { height: 100%; display: grid; grid-template-columns: minmax(0, 1fr) 380px; }
.graph-pane { min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); }
.graph-toolbar { background: var(--panel); border-bottom: 1px solid var(--line); padding: 10px 14px; display: flex; gap: 8px; align-items: center; }
.graph-toolbar input { min-width: 220px; }
.graph-toolbar .spacer { flex: 1; }
.graph-legend { display: inline-flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; white-space: nowrap; }
.graph-legend-item { display: inline-flex; align-items: center; gap: 5px; }
.graph-legend-shape { display: inline-block; width: 18px; height: 12px; border: 1.5px solid var(--muted); background: var(--panel); }
.graph-legend-shape.artifact { border-radius: 2px; clip-path: polygon(0 0, 72% 0, 100% 35%, 100% 100%, 0 100%); }
.graph-legend-shape.operation { width: 20px; clip-path: polygon(10% 0, 90% 0, 100% 50%, 90% 100%, 10% 100%, 0 50%); }
.graph-legend-shape.transformation { width: 24px; height: 14px; border-radius: 999px; background: var(--accent-soft); }
.graph-wrap { overflow: auto; padding: 16px; }
.graph-canvas { min-width: 100%; min-height: 100%; }
.node-shape { fill: var(--panel); stroke: var(--line); stroke-width: 1.5; }
.node.operation .node-shape { fill: var(--accent-soft); }
.node:hover .node-shape, .node.selected .node-shape { stroke: var(--accent); stroke-width: 2.5; }
.node-fold { fill: none; stroke: var(--line); stroke-width: 1.2; pointer-events: none; }
.node:hover .node-fold, .node.selected .node-fold { stroke: var(--accent); }
.node text { fill: var(--text); pointer-events: none; }
.node .node-sub { fill: var(--muted); font-size: 11px; }
.edge { stroke: var(--muted); stroke-width: 1.4; fill: none; opacity: .64; }
.edge.selected { stroke: var(--accent); stroke-width: 2.5; opacity: 1; }
.edge-label { fill: var(--muted); font-size: 10px; paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; }
.edge-badge { cursor: pointer; }
.edge-badge rect { fill: var(--accent-soft); stroke: var(--line); stroke-width: 1; }
.edge-badge:hover rect, .edge-badge.selected rect { stroke: var(--accent); stroke-width: 2; }
.edge-badge text { fill: var(--text); font-size: 10px; font-weight: 600; pointer-events: none; }
.graph-focus-note { color: var(--muted); font-size: 12px; white-space: nowrap; }
.artifact-detail { border-left: 1px solid var(--line); background: var(--panel); overflow: auto; padding: 16px; }
.artifact-detail h2 { margin: 0 0 4px; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0 14px; }
.badge { border: 1px solid var(--line); border-radius: 999px; padding: 2px 7px; font-size: 11px; color: var(--muted); }
.detail-grid { display: grid; grid-template-columns: 90px 1fr; gap: 6px 10px; margin: 12px 0 18px; }
.detail-grid dt { color: var(--muted); }
.detail-grid dd { margin: 0; overflow-wrap: anywhere; }
.artifact-memo { border-top: 1px solid var(--line); padding-top: 14px; }
.artifact-memo h3 { margin: 0 0 10px; }
.artifact-memo input, .artifact-memo textarea { width: 100%; margin-bottom: 8px; }
.artifact-memo textarea { min-height: 220px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.artifact-memo-actions { display: flex; gap: 8px; justify-content: flex-end; }
.artifact-actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 16px; }
.artifact-json { border-top: 1px solid var(--line); padding-top: 12px; margin-top: 8px; }
.artifact-json[hidden] { display: none; }
.inspector-section { border-top: 1px solid var(--line); margin-top: 16px; padding-top: 14px; }
.inspector-section h3 { margin: 0 0 10px; }
.json-tree { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.json-tree details { margin-left: 12px; }
.json-tree summary { cursor: pointer; padding: 2px 0; }
.json-scalar { margin-left: 28px; padding: 2px 0; overflow-wrap: anywhere; }
.json-key { color: var(--muted); }
.preview-dialog { width: min(1180px, 94vw); max-width: 1180px; height: min(760px, 90vh); border: 1px solid var(--line); border-radius: 10px; padding: 0; background: var(--panel); color: var(--text); }
.preview-dialog::backdrop { background: rgb(0 0 0 / .38); }
.preview-shell { height: 100%; display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto; }
.preview-header { display: flex; align-items: center; gap: 10px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.preview-header strong { font-size: 16px; }
.preview-header .spacer { flex: 1; }
.preview-controls { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); }
.preview-controls label { display: inline-flex; align-items: center; gap: 6px; }
.preview-controls input[type=checkbox] { width: auto; }
.preview-table-wrap { overflow: auto; padding: 0; }
.preview-table { border-collapse: collapse; width: max-content; min-width: 100%; font-size: 12px; }
.preview-table th, .preview-table td { border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; max-width: 420px; overflow-wrap: anywhere; white-space: pre-wrap; }
.preview-table th { position: sticky; top: 0; z-index: 1; background: var(--panel); }
.preview-footer { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--line); }
.preview-footer .spacer { flex: 1; }
.preview-error { color: var(--danger); padding: 16px; }

@media (max-width: 900px) {
  .memo-shell { grid-template-columns: 280px minmax(0, 1fr); }
  .artifact-shell { grid-template-columns: 1fr; grid-template-rows: 55% 45%; }
  .artifact-detail { border-left: 0; border-top: 1px solid var(--line); }
}
@media (max-width: 700px) {
  .header { flex-wrap: wrap; }
  .global-status { width: 100%; margin-left: 0; }
  .memo-shell { grid-template-columns: 1fr; grid-template-rows: 38% 62%; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
  .fields { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="brand"><strong>TeAL Project Center</strong><span id="projectName"></span></div>
    <nav class="tabs">
      <button class="tab-button" data-tab="artifacts">Artifacts</button>
      <button class="tab-button" data-tab="memos">Memos</button>
    </nav>
    <div class="global-status" id="status"></div>
    <button id="endSession" title="Stop the local Project Center server">End session</button>
  </header>

  <section class="tab" id="tab-memos">
    <div class="memo-shell">
      <aside class="sidebar">
        <div class="toolbar">
          <div class="toolbar-row"><button class="primary" id="newMemo">New Memo</button><button id="refreshMemos">Refresh</button></div>
          <input id="searchMemos" type="search" placeholder="Search memos">
          <select id="filterMemoType">
            <option value="">All memos</option>
            <option value="project">Project</option>
            <option value="standalone">Standalone</option>
            <option value="artifact">Artifact</option>
            <option value="operation">Operation</option>
            <option value="operator">Operator</option>
          </select>
        </div>
        <div class="memo-list" id="memoList"></div>
      </aside>
      <main class="memo-main">
        <div class="memo-topbar">
          <div><strong id="memoHeading">Memo</strong><div class="target-note" id="memoTarget"></div></div>
          <div class="target-note" id="memoSaved"></div>
        </div>
        <div class="editor">
          <div class="editor-card">
            <div class="fields"><label for="memoTitle">Title</label><input id="memoTitle" maxlength="500" placeholder="Optional title"></div>
            <div class="editor-mode"><button id="memoRaw" class="active">Markdown</button><button id="memoPreview">Preview</button></div>
            <div class="body-wrap">
              <textarea class="memo-body" id="memoBody" placeholder="Write your memo in Markdown..."></textarea>
              <div class="preview markdown-body" id="memoPreviewPane" hidden></div>
            </div>
            <div class="actions"><button id="memoHistoryButton">History</button><button class="primary" id="saveMemo">Save</button></div>
            <div class="history" id="memoHistory"><h2>Version history</h2><div id="memoHistoryRows"></div></div>
          </div>
        </div>
      </main>
    </div>
  </section>

  <section class="tab" id="tab-artifacts">
    <div class="artifact-shell">
      <div class="graph-pane">
        <div class="graph-toolbar">
          <label for="graphMode">View</label>
          <select id="graphMode"><option value="lineage">Lineage</option><option value="provenance">Provenance</option></select>
          <button id="focusSelected" disabled>Focus selected</button>
          <span class="graph-focus-note" id="focusNote"></span>
          <input id="artifactSearch" type="search" placeholder="Find artifact or transformation">
          <span class="target-note" id="storageSummary"></span>
          <span class="graph-legend" aria-label="Graph node legend">
            <span class="graph-legend-item"><span class="graph-legend-shape artifact"></span>Artifact</span>
            <span class="graph-legend-item" id="legendOperation" hidden><span class="graph-legend-shape operation"></span>Operation</span>
          </span>
          <div class="spacer"></div><button id="refreshArtifacts">Refresh</button>
        </div>
        <div class="graph-wrap"><svg class="graph-canvas" id="artifactGraph"></svg></div>
      </div>
      <aside class="artifact-detail" id="artifactDetail"><div class="empty">Select an artifact or operation to inspect it.</div></aside>
    </div>
  </section>

  <dialog class="preview-dialog" id="tablePreviewDialog">
    <div class="preview-shell">
      <div class="preview-header">
        <strong id="tablePreviewTitle">Table preview</strong>
        <span class="target-note" id="tablePreviewMeta"></span>
        <div class="spacer"></div>
        <button id="closeTablePreview">Close</button>
      </div>
      <div class="preview-controls">
        <label><input type="checkbox" id="previewKeys" checked> Keys</label>
        <label><input type="checkbox" id="previewData" checked> Data</label>
        <label><input type="checkbox" id="previewMetadata"> Metadata</label>
        <label><input type="checkbox" id="previewFullMetadata" disabled> Full metadata</label>
        <label>Rows per page
          <select id="previewPageSize"><option>25</option><option>50</option><option>100</option></select>
        </label>
      </div>
      <div class="preview-table-wrap" id="tablePreviewBody"></div>
      <div class="preview-footer">
        <button id="previewPrevious">Previous</button>
        <button id="previewNext">Next</button>
        <span id="previewPageLabel"></span>
        <div class="spacer"></div>
        <span class="target-note">Only the visible page is queried.</span>
      </div>
    </div>
  </dialog>
</div>
<script>
const app = {
  state: null,
  selectedMemo: null,
  memoBaseline: {title: '', body: ''},
  memoHistory: [],
  graph: null,
  selectedArtifact: null,
  selectedOperation: null,
  artifactDetail: null,
  operationDetail: null,
  artifactMemoBaseline: {title: '', body: ''},
  artifactDescriptor: null,
  operationDescriptor: null,
  operatorDescriptor: null,
  tablePreview: {page: 1, pageCount: 1},
  graphFocused: false,
  sessionEnding: false,
};
const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
function setStatus(message, error = false) {
  $('status').textContent = message;
  $('status').className = error ? 'global-status error' : 'global-status';
}
async function endSession() {
  const dirty = memoIsDirty() || artifactMemoIsDirty();
  const prompt = dirty
    ? 'You have unsaved changes. End this Project Center session and discard them?'
    : 'End this Project Center session?';
  if (!window.confirm(prompt)) return;
  try {
    app.sessionEnding = true;
    $('endSession').disabled = true;
    const payload = await api('/api/end-session', {method: 'POST'});
    if (payload.status !== 'ending') throw new Error('Project Center did not acknowledge shutdown.');
    setStatus('Session ended. You can close this browser tab.');
  } catch (error) {
    app.sessionEnding = false;
    $('endSession').disabled = false;
    setStatus(error.message, true);
  }
}
function formatWhen(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
function displayTitle(memo) {
  return memo.title && memo.title.trim() ? memo.title : targetLabel(memo.target_type, memo.target_id);
}
function targetLabel(type, id) {
  const map = app.state?.target_labels?.[type] || {};
  return map[id] || (type === 'standalone' ? `Standalone memo ${id}` : `${type}: ${id}`);
}
function memoIsDirty() {
  return $('memoTitle').value !== app.memoBaseline.title || $('memoBody').value !== app.memoBaseline.body;
}
function artifactMemoIsDirty() {
  const title = $('artifactMemoTitle');
  const body = $('artifactMemoBody');
  if (!title || !body) return false;
  return title.value !== app.artifactMemoBaseline.title || body.value !== app.artifactMemoBaseline.body;
}
function confirmMemoDiscard() {
  return !memoIsDirty() || window.confirm('Discard unsaved memo changes?');
}
function confirmArtifactMemoDiscard() {
  return !artifactMemoIsDirty() || window.confirm('Discard unsaved artifact memo changes?');
}
function setMemoBaseline(title, body) {
  app.memoBaseline = {title: title || '', body: body || ''};
}
function setArtifactMemoBaseline(title, body) {
  app.artifactMemoBaseline = {title: title || '', body: body || ''};
}

async function loadState(preserveMemo = true) {
  const previous = preserveMemo && app.selectedMemo ? `${app.selectedMemo.target_type}:${app.selectedMemo.target_id}` : null;
  app.state = await api('/api/state');
  $('projectName').textContent = app.state.project_name;
  if (previous) {
    app.selectedMemo = app.state.memos.find(m => `${m.target_type}:${m.target_id}` === previous) || null;
  }
  renderMemoList();
  if (app.selectedMemo) loadMemoIntoEditor(app.selectedMemo, false);
}
function renderMemoList() {
  const query = $('searchMemos').value.trim().toLowerCase();
  const type = $('filterMemoType').value;
  const selectedKey = app.selectedMemo ? `${app.selectedMemo.target_type}:${app.selectedMemo.target_id}` : '';
  const rows = app.state.memos.filter(memo => {
    if (type && memo.target_type !== type) return false;
    if (!query) return true;
    const label = targetLabel(memo.target_type, memo.target_id);
    return [memo.title, memo.body, memo.target_type, memo.target_id, label].some(v => String(v || '').toLowerCase().includes(query));
  });
  const container = $('memoList');
  container.innerHTML = '';
  if (!rows.length) {
    container.innerHTML = '<div class="empty">No memos match.</div>';
    return;
  }
  for (const memo of rows) {
    const button = document.createElement('button');
    button.className = 'memo-item' + (`${memo.target_type}:${memo.target_id}` === selectedKey ? ' active' : '');
    const title = document.createElement('div');
    title.className = 'memo-title';
    title.textContent = displayTitle(memo);
    const meta = document.createElement('div');
    meta.className = 'memo-meta';
    meta.textContent = `${memo.target_type} · ${targetLabel(memo.target_type, memo.target_id)} · ${formatWhen(memo.created_at)}`;
    button.append(title, meta);
    button.addEventListener('click', () => selectMemo(memo));
    container.appendChild(button);
  }
}
function newStandaloneMemo() {
  if (!confirmMemoDiscard()) return;
  app.selectedMemo = null;
  $('memoHeading').textContent = 'New Memo';
  $('memoTarget').textContent = 'Standalone memo';
  $('memoSaved').textContent = 'Not yet saved';
  $('memoTitle').value = '';
  $('memoBody').value = '';
  setMemoBaseline('', '');
  closeMemoHistory();
  setMemoMode('raw');
  renderMemoList();
  setStatus('');
  $('memoTitle').focus();
}
function selectMemo(memo) {
  if (!confirmMemoDiscard()) return;
  app.selectedMemo = memo;
  loadMemoIntoEditor(memo, true);
}
function loadMemoIntoEditor(memo, rerenderList = true) {
  $('memoHeading').textContent = displayTitle(memo);
  $('memoTarget').textContent = `${memo.target_type} · ${targetLabel(memo.target_type, memo.target_id)}`;
  $('memoSaved').textContent = `Saved ${formatWhen(memo.created_at)}`;
  $('memoTitle').value = memo.title || '';
  $('memoBody').value = memo.body || '';
  setMemoBaseline(memo.title || '', memo.body || '');
  updateMemoPreview();
  closeMemoHistory();
  if (rerenderList) renderMemoList();
}
async function saveMemo() {
  const body = $('memoBody').value;
  if (!body.trim()) {
    setStatus('Memo body cannot be empty.', true);
    return;
  }
  const targetType = app.selectedMemo ? app.selectedMemo.target_type : 'standalone';
  const targetId = app.selectedMemo ? app.selectedMemo.target_id : null;
  setStatus('Saving...');
  try {
    const payload = await api('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({target_type: targetType, target_id: targetId, title: $('memoTitle').value || null, body})
    });
    app.selectedMemo = payload.memo;
    setMemoBaseline(payload.memo.title || '', payload.memo.body || '');
    await loadState(true);
    setStatus(`Saved memo version ${payload.memo.memo_id}.`);
  } catch (error) {
    setStatus(error.message, true);
  }
}
async function toggleMemoHistory() {
  const panel = $('memoHistory');
  if (panel.classList.contains('open')) {
    closeMemoHistory();
    return;
  }
  if (!app.selectedMemo) {
    setStatus('Save the new memo before viewing history.', true);
    return;
  }
  try {
    const params = new URLSearchParams({target_type: app.selectedMemo.target_type, target_id: app.selectedMemo.target_id});
    const payload = await api(`/api/history?${params}`);
    app.memoHistory = payload.versions;
    renderMemoHistory();
    panel.classList.add('open');
  } catch (error) {
    setStatus(error.message, true);
  }
}
function renderMemoHistory() {
  const container = $('memoHistoryRows');
  container.innerHTML = '';
  for (let index = 0; index < app.memoHistory.length; index++) {
    const version = app.memoHistory[index];
    const button = document.createElement('button');
    button.className = 'history-row';
    const label = document.createElement('span');
    label.textContent = `${index === 0 ? 'Current' : `Previous ${index}`} · ${version.title || 'Untitled'}`;
    const when = document.createElement('span');
    when.textContent = formatWhen(version.created_at);
    button.append(label, when);
    button.addEventListener('click', () => loadHistoricalMemoVersion(version));
    container.appendChild(button);
  }
}
function loadHistoricalMemoVersion(version) {
  if (!confirmMemoDiscard()) return;
  $('memoTitle').value = version.title || '';
  $('memoBody').value = version.body || '';
  updateMemoPreview();
  setStatus(version.memo_id === app.selectedMemo.memo_id ? 'Current version loaded.' : `Loaded historical version ${version.memo_id}; Save to restore it as a new version.`);
}
function closeMemoHistory() {
  $('memoHistory').classList.remove('open');
  $('memoHistoryRows').innerHTML = '';
  app.memoHistory = [];
}
function setMemoMode(mode) {
  const preview = mode === 'preview';
  $('memoRaw').classList.toggle('active', !preview);
  $('memoPreview').classList.toggle('active', preview);
  $('memoBody').hidden = preview;
  $('memoPreviewPane').hidden = !preview;
  if (preview) updateMemoPreview();
}
function updateMemoPreview() {
  $('memoPreviewPane').innerHTML = renderMarkdown($('memoBody').value);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function inlineMarkdown(value) {
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  text = text.replace(/_([^_]+)_/g, '<em>$1</em>');
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return text;
}
function renderMarkdown(source) {
  const lines = String(source || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let inFence = false;
  let fence = [];
  let listType = null;
  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };
  for (const line of lines) {
    if (line.startsWith('```')) {
      closeList();
      if (inFence) {
        out.push(`<pre><code>${escapeHtml(fence.join('\n'))}</code></pre>`);
        fence = [];
        inFence = false;
      } else {
        inFence = true;
      }
      continue;
    }
    if (inFence) { fence.push(line); continue; }
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.*)$/);
    if (unordered) {
      if (listType !== 'ul') { closeList(); out.push('<ul>'); listType = 'ul'; }
      out.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
      continue;
    }
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ordered) {
      if (listType !== 'ol') { closeList(); out.push('<ol>'); listType = 'ol'; }
      out.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
      continue;
    }
    closeList();
    if (!line.trim()) { out.push(''); continue; }
    if (line.startsWith('> ')) {
      out.push(`<blockquote>${inlineMarkdown(line.slice(2))}</blockquote>`);
    } else {
      out.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  }
  closeList();
  if (inFence) out.push(`<pre><code>${escapeHtml(fence.join('\n'))}</code></pre>`);
  return out.join('\n');
}

async function loadArtifactGraph(preserve = true) {
  const previousArtifact = preserve ? app.selectedArtifact : null;
  const previousOperation = preserve ? app.selectedOperation : null;
  app.graph = await api('/api/artifacts');
  renderStorageSummary();
  renderArtifactGraph();
  if (previousOperation && app.graph.operation_nodes.some(node => node.id === previousOperation)) {
    await selectOperation(previousOperation, false);
  } else if (previousArtifact && app.graph.artifact_nodes.some(node => node.id === previousArtifact)) {
    await selectArtifact(previousArtifact, false);
  }
}
function renderStorageSummary() {
  const storage = app.graph?.storage;
  if (!storage) { $('storageSummary').textContent = ''; return; }
  $('storageSummary').textContent = `Project ${storage.total_display} · artifacts ${storage.artifacts_display} · operations ${storage.operations_display} · operators ${storage.operators_display} · other ${storage.other_display}`;
}
function graphEdges() {
  return $('graphMode').value === 'lineage' ? app.graph.lineage_edges : app.graph.provenance_edges;
}
function graphNodes() {
  return $('graphMode').value === 'lineage' ? app.graph.artifact_nodes : app.graph.nodes;
}
function graphSelectedSeeds(edges) {
  if (app.selectedArtifact) return [app.selectedArtifact];
  if (!app.selectedOperation) return [];
  if ($('graphMode').value === 'provenance') return [app.selectedOperation];
  const seeds = new Set();
  for (const edge of edges) {
    if (edge.operation_id === app.selectedOperation) {
      seeds.add(edge.source); seeds.add(edge.target);
    }
  }
  return Array.from(seeds);
}
function graphNeighborhood(nodes, edges, seeds, radius = 2) {
  const ids = new Set(nodes.map(node => node.id));
  const adjacent = new Map(nodes.map(node => [node.id, new Set()]));
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    adjacent.get(edge.source).add(edge.target);
    adjacent.get(edge.target).add(edge.source);
  }
  const seen = new Set(seeds.filter(id => ids.has(id)));
  let frontier = Array.from(seen);
  for (let step = 0; step < radius; step++) {
    const next = [];
    for (const id of frontier) {
      for (const neighbor of adjacent.get(id) || []) {
        if (seen.has(neighbor)) continue;
        seen.add(neighbor); next.push(neighbor);
      }
    }
    frontier = next;
    if (!frontier.length) break;
  }
  return seen;
}
function graphData() {
  let nodes = graphNodes();
  let edges = graphEdges();
  if (app.graphFocused) {
    const seeds = graphSelectedSeeds(edges);
    if (seeds.length) {
      const keep = graphNeighborhood(nodes, edges, seeds, 2);
      nodes = nodes.filter(node => keep.has(node.id));
      edges = edges.filter(edge => keep.has(edge.source) && keep.has(edge.target));
    }
  }

  const query = $('artifactSearch').value.trim().toLowerCase();
  if (query) {
    const matched = new Set();
    for (const node of nodes) {
      const fields = [node.id, node.label, node.node_type, node.artifact_type, node.operation_type, node.operator_id, node.status];
      if (fields.some(value => String(value || '').toLowerCase().includes(query))) matched.add(node.id);
    }
    for (const edge of edges) {
      const fields = [edge.label, edge.operation_label, edge.operation_id, edge.source_label, edge.output_label];
      if (fields.some(value => String(value || '').toLowerCase().includes(query))) {
        matched.add(edge.source); matched.add(edge.target);
      }
    }
    const keep = graphNeighborhood(nodes, edges, Array.from(matched), 1);
    nodes = nodes.filter(node => keep.has(node.id));
    edges = edges.filter(edge => keep.has(edge.source) && keep.has(edge.target));
  }
  return {nodes, edges};
}
function renderGraphControls() {
  const hasSelection = Boolean(app.selectedArtifact || ($('graphMode').value === 'provenance' && app.selectedOperation));
  $('focusSelected').disabled = !hasSelection;
  $('focusSelected').textContent = app.graphFocused ? 'Show all' : 'Focus selected';
  $('focusSelected').classList.toggle('active', app.graphFocused);
  $('focusNote').textContent = app.graphFocused && hasSelection ? 'Showing local neighborhood' : '';
  const lineage = $('graphMode').value === 'lineage';
  $('legendOperation').hidden = lineage;
}
function provenanceEdgeLabel(edge, nodeById) {
  const value = String(edge.label || '').trim();
  if (!value || ['source', 'output', 'input'].includes(value.toLowerCase())) return '';
  const from = nodeById.get(edge.source); const to = nodeById.get(edge.target);
  if (value === String(from?.label || '') || value === String(to?.label || '')) return '';
  return truncate(value, 24);
}
function renderArtifactGraph() {
  const svg = $('artifactGraph');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!app.graph) return;
  renderGraphControls();
  const {nodes: visibleNodes, edges} = graphData();
  const visibleIds = new Set(visibleNodes.map(node => node.id));
  const nodeById = new Map(visibleNodes.map(node => [node.id, node]));
  const positions = graphLayout(visibleNodes, edges);
  const width = Math.max(760, ...visibleNodes.map(node => {
    const pos = positions.get(node.id); const metrics = graphNodeMetrics(node);
    return pos ? pos.x + metrics.width + 35 : 0;
  }));
  const height = Math.max(500, ...visibleNodes.map(node => {
    const pos = positions.get(node.id); const metrics = graphNodeMetrics(node);
    return pos ? pos.y + metrics.height + 35 : 0;
  }));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', 'arrow'); marker.setAttribute('viewBox', '0 0 10 10'); marker.setAttribute('refX', '9'); marker.setAttribute('refY', '5'); marker.setAttribute('markerWidth', '6'); marker.setAttribute('markerHeight', '6'); marker.setAttribute('orient', 'auto-start-reverse');
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z'); arrow.setAttribute('fill', 'currentColor');
  marker.appendChild(arrow); defs.appendChild(marker); svg.appendChild(defs);

  for (const edge of edges) {
    const from = positions.get(edge.source); const to = positions.get(edge.target);
    if (!from || !to) continue;
    const fromNode = nodeById.get(edge.source);
    const toNode = nodeById.get(edge.target);
    if (!fromNode || !toNode) continue;
    const fromMetrics = graphNodeMetrics(fromNode); const toMetrics = graphNodeMetrics(toNode);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const x1 = from.x + fromMetrics.width; const y1 = from.y + fromMetrics.height / 2;
    const x2 = to.x; const y2 = to.y + toMetrics.height / 2;
    const horizontal = Math.max(0, x2 - x1);
    const bend = Math.max(50, Math.min(150, horizontal / 2));
    line.setAttribute('d', `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
    line.setAttribute('class', 'edge'); line.setAttribute('marker-end', 'url(#arrow)');
    svg.appendChild(line);
    const middleX = (x1 + x2) / 2; const middleY = (y1 + y2) / 2 - 5;
    const edgeLabel = $('graphMode').value === 'lineage'
      ? truncate(String(edge.label || ''), 24)
      : provenanceEdgeLabel(edge, nodeById);
    if (!edgeLabel) continue;
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', String(middleX)); label.setAttribute('y', String(middleY)); label.setAttribute('text-anchor', 'middle'); label.setAttribute('class', 'edge-label'); label.textContent = edgeLabel;
    svg.appendChild(label);
  }
  for (const node of visibleNodes) {
    const pos = positions.get(node.id);
    if (!pos) continue;
    const metrics = graphNodeMetrics(node);
    const selected = node.node_type === 'operation' ? node.id === app.selectedOperation : node.id === app.selectedArtifact;
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `node ${node.node_type}${selected ? ' selected' : ''}`);
    group.setAttribute('transform', `translate(${pos.x},${pos.y})`);
    group.style.cursor = 'pointer';
    if (node.node_type === 'operation') {
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const inset = 18; const mid = metrics.height / 2;
      shape.setAttribute('class', 'node-shape');
      shape.setAttribute('d', `M ${inset} 0 H ${metrics.width - inset} L ${metrics.width} ${mid} L ${metrics.width - inset} ${metrics.height} H ${inset} L 0 ${mid} Z`);
      group.appendChild(shape);
    } else {
      const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const fold = 22;
      shape.setAttribute('class', 'node-shape');
      shape.setAttribute('d', `M 0 0 H ${metrics.width - fold} L ${metrics.width} ${fold} V ${metrics.height} H 0 Z`);
      const foldLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      foldLine.setAttribute('class', 'node-fold');
      foldLine.setAttribute('d', `M ${metrics.width - fold} 0 V ${fold} H ${metrics.width}`);
      group.append(shape, foldLine);
    }
    const textX = node.node_type === 'operation' ? 24 : 12;
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    title.setAttribute('x', String(textX)); title.setAttribute('y', node.node_type === 'operation' ? '20' : '22'); title.textContent = truncate(node.label, node.node_type === 'operation' ? 20 : 28);
    const sub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    sub.setAttribute('x', String(textX)); sub.setAttribute('y', node.node_type === 'operation' ? '37' : '43'); sub.setAttribute('class', 'node-sub');
    sub.textContent = node.node_type === 'operation'
      ? (node.operation_type === node.label ? 'operation' : node.operation_type)
      : `${node.artifact_type} · ${node.size_display}${node.status === 'complete' ? '' : ` · ${node.status}`}`;
    group.append(title, sub);
    group.addEventListener('click', () => node.node_type === 'operation' ? selectOperation(node.id) : selectArtifact(node.id));
    svg.appendChild(group);
  }
}
function graphNodeMetrics(node) {
  return node.node_type === 'operation'
    ? {width: 150, height: 46}
    : {width: 220, height: 58};
}
function graphLayout(nodes, edges) {
  if (!nodes.length) return new Map();
  const ids = new Set(nodes.map(node => node.id));
  const depth = new Map(nodes.map(node => [node.id, 0]));
  for (let iteration = 0; iteration < nodes.length; iteration++) {
    let changed = false;
    for (const edge of edges) {
      if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
      const candidate = (depth.get(edge.source) || 0) + 1;
      if (candidate > (depth.get(edge.target) || 0) && candidate <= nodes.length) {
        depth.set(edge.target, candidate); changed = true;
      }
    }
    if (!changed) break;
  }
  const levels = new Map();
  for (const node of nodes.slice().sort((a, b) => a.label.localeCompare(b.label))) {
    const level = depth.get(node.id) || 0;
    if (!levels.has(level)) levels.set(level, []);
    levels.get(level).push(node);
  }

  const incoming = new Map(nodes.map(node => [node.id, []]));
  const outgoing = new Map(nodes.map(node => [node.id, []]));
  for (const edge of edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    outgoing.get(edge.source).push(edge.target);
    incoming.get(edge.target).push(edge.source);
  }
  const levelKeys = Array.from(levels.keys()).sort((a, b) => a - b);
  const rebuildOrder = () => {
    const order = new Map();
    for (const level of levelKeys) {
      levels.get(level).forEach((node, index) => order.set(node.id, index));
    }
    return order;
  };
  const sortLevelByNeighbors = (level, neighbors, order) => {
    const group = levels.get(level);
    const previousIndex = new Map(group.map((node, index) => [node.id, index]));
    const barycenter = node => {
      const positions = (neighbors.get(node.id) || [])
        .map(id => order.get(id))
        .filter(value => Number.isFinite(value));
      if (!positions.length) return null;
      return positions.reduce((total, value) => total + value, 0) / positions.length;
    };
    group.sort((a, b) => {
      const aCenter = barycenter(a); const bCenter = barycenter(b);
      if (aCenter === null && bCenter === null) return previousIndex.get(a.id) - previousIndex.get(b.id);
      if (aCenter === null) return 1;
      if (bCenter === null) return -1;
      if (aCenter !== bCenter) return aCenter - bCenter;
      return previousIndex.get(a.id) - previousIndex.get(b.id);
    });
  };

  // Repeated left/right barycentric sweeps keep connected branches close together
  // and substantially reduce crossings compared with alphabetical rank ordering.
  for (let sweep = 0; sweep < 6; sweep++) {
    let order = rebuildOrder();
    for (const level of levelKeys.slice(1)) {
      sortLevelByNeighbors(level, incoming, order);
      order = rebuildOrder();
    }
    order = rebuildOrder();
    for (const level of levelKeys.slice(0, -1).reverse()) {
      sortLevelByNeighbors(level, outgoing, order);
      order = rebuildOrder();
    }
  }

  const verticalGap = 34;
  const levelHeights = new Map();
  let maxLevelHeight = 0;
  for (const level of levelKeys) {
    const group = levels.get(level);
    const total = group.reduce((sum, node) => sum + graphNodeMetrics(node).height, 0)
      + Math.max(0, group.length - 1) * verticalGap;
    levelHeights.set(level, total);
    maxLevelHeight = Math.max(maxLevelHeight, total);
  }
  const positions = new Map();
  let x = 35;
  const horizontalGap = $('graphMode').value === 'lineage' ? 110 : 80;
  for (const level of levelKeys) {
    const group = levels.get(level);
    const levelWidth = Math.max(...group.map(node => graphNodeMetrics(node).width));
    let y = 35 + (maxLevelHeight - levelHeights.get(level)) / 2;
    for (const node of group) {
      positions.set(node.id, {x, y});
      y += graphNodeMetrics(node).height + verticalGap;
    }
    x += levelWidth + horizontalGap;
  }
  return positions;
}
function truncate(value, max) {
  const text = String(value || '');
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}
async function selectArtifact(artifactId, ask = true) {
  if (ask && !confirmArtifactMemoDiscard()) return;
  try {
    app.selectedArtifact = artifactId;
    app.selectedOperation = null;
    app.operationDetail = null;
    app.operationDescriptor = null;
    app.operatorDescriptor = null;
    app.artifactDescriptor = null;
    app.artifactDetail = await api(`/api/artifact?${new URLSearchParams({artifact_id: artifactId})}`);
    renderArtifactDetail();
    renderArtifactGraph();
  } catch (error) {
    setStatus(error.message, true);
  }
}
async function selectOperation(operationId, ask = true) {
  if (ask && !confirmArtifactMemoDiscard()) return;
  try {
    app.selectedOperation = operationId;
    app.selectedArtifact = null;
    app.artifactDetail = null;
    app.artifactDescriptor = null;
    app.operationDescriptor = null;
    app.operatorDescriptor = null;
    app.operationDetail = await api(`/api/operation?${new URLSearchParams({operation_id: operationId})}`);
    renderOperationDetail();
    renderArtifactGraph();
  } catch (error) {
    setStatus(error.message, true);
  }
}
function renderOperationDetail() {
  const detail = app.operationDetail;
  const container = $('artifactDetail');
  if (!detail) {
    container.innerHTML = '<div class="empty">Select an artifact or operation to inspect it.</div>';
    return;
  }
  const operation = detail.operation;
  const operator = detail.operator;
  container.innerHTML = '';
  const heading = document.createElement('h2'); heading.textContent = detail.display_label || operation.operation_type;
  const badges = document.createElement('div'); badges.className = 'badges';
  for (const value of ['operation', operation.status]) { const span = document.createElement('span'); span.className = 'badge'; span.textContent = value; badges.appendChild(span); }
  const dl = document.createElement('dl'); dl.className = 'detail-grid';
  addDetail(dl, 'ID', operation.operation_id);
  addDetail(dl, 'Created', formatWhen(operation.created_at) || '—');
  addDetail(dl, 'Completed', formatWhen(operation.completed_at) || '—');
  addDetail(dl, 'Size on disk', detail.size_display);
  addDetail(dl, 'Sources', detail.sources.length ? detail.sources.map(row => `${row.source_label}: ${row.source_artifact_id}`).join(', ') : '—');
  addDetail(dl, 'Outputs', detail.outputs.length ? detail.outputs.map(row => `${row.output_label}: ${row.artifact_id}`).join(', ') : '—');
  if (operation.error) addDetail(dl, 'Error', operation.error);
  const actionRow = document.createElement('div'); actionRow.className = 'artifact-actions';
  const operationDetailsButton = document.createElement('button'); operationDetailsButton.textContent = 'Load operation details'; operationDetailsButton.disabled = !detail.descriptor_available; operationDetailsButton.addEventListener('click', loadOperationJson); actionRow.appendChild(operationDetailsButton);
  const operationJsonBox = document.createElement('section'); operationJsonBox.className = 'artifact-json'; operationJsonBox.id = 'operationJsonBox'; operationJsonBox.hidden = true;

  const operatorSection = document.createElement('section'); operatorSection.className = 'inspector-section';
  const operatorHeading = document.createElement('h3'); operatorHeading.textContent = 'Operator';
  const operatorName = detail.operator_aliases.length ? detail.operator_aliases[0] : (detail.display_label || operator.operation_type);
  const operatorDl = document.createElement('dl'); operatorDl.className = 'detail-grid';
  addDetail(operatorDl, 'Name', operatorName);
  addDetail(operatorDl, 'ID', operator.operator_id);
  addDetail(operatorDl, 'Type', operator.operation_type);
  addDetail(operatorDl, 'Snapshot', operator.snapshot_status);
  addDetail(operatorDl, 'Size on disk', detail.operator_size_display);
  addDetail(operatorDl, 'Uses', `${detail.operator_use_count} operation${detail.operator_use_count === 1 ? '' : 's'}`);
  const operatorActions = document.createElement('div'); operatorActions.className = 'artifact-actions';
  const operatorDetailsButton = document.createElement('button'); operatorDetailsButton.textContent = 'Load operator details'; operatorDetailsButton.disabled = !detail.operator_descriptor_available; operatorDetailsButton.addEventListener('click', loadOperatorJson); operatorActions.appendChild(operatorDetailsButton);
  const operatorJsonBox = document.createElement('section'); operatorJsonBox.className = 'artifact-json'; operatorJsonBox.id = 'operatorJsonBox'; operatorJsonBox.hidden = true;
  operatorSection.append(operatorHeading, operatorDl, operatorActions, operatorJsonBox);

  container.append(heading, badges, dl, actionRow, operationJsonBox, operatorSection);
}
async function loadOperationJson() {
  if (!app.selectedOperation) return;
  const box = $('operationJsonBox');
  if (app.operationDescriptor) { box.hidden = !box.hidden; return; }
  try {
    const payload = await api(`/api/operation-json?${new URLSearchParams({operation_id: app.selectedOperation})}`);
    app.operationDescriptor = payload.descriptor;
    box.innerHTML = '';
    const heading = document.createElement('h3'); heading.textContent = 'operation.json';
    box.append(heading, jsonTree(payload.descriptor, 'operation.json', false));
    box.hidden = false;
  } catch (error) { setStatus(error.message, true); }
}
async function loadOperatorJson() {
  const operatorId = app.operationDetail?.operator?.operator_id;
  if (!operatorId) return;
  const box = $('operatorJsonBox');
  if (app.operatorDescriptor) { box.hidden = !box.hidden; return; }
  try {
    const payload = await api(`/api/operator-json?${new URLSearchParams({operator_id: operatorId})}`);
    app.operatorDescriptor = payload.descriptor;
    box.innerHTML = '';
    const heading = document.createElement('h3'); heading.textContent = 'operator.json';
    box.append(heading, jsonTree(payload.descriptor, 'operator.json', false));
    box.hidden = false;
  } catch (error) { setStatus(error.message, true); }
}
function renderArtifactDetail() {
  const detail = app.artifactDetail;
  const container = $('artifactDetail');
  if (!detail) {
    container.innerHTML = '<div class="empty">Select an artifact or operation to inspect it.</div>';
    return;
  }
  const a = detail.artifact;
  const preferred = detail.aliases.length ? detail.aliases[0] : a.label;
  const operation = detail.operation ? `${detail.operation.operation_type} · ${detail.operation.operation_id}` : '—';
  const sources = detail.sources.length ? detail.sources.map(row => row.source_artifact_id).join(', ') : '—';
  const basis = detail.basis.length ? detail.basis.map(row => row.basis_artifact_id).join(', ') : '—';
  container.innerHTML = '';
  const heading = document.createElement('h2'); heading.textContent = preferred;
  const badges = document.createElement('div'); badges.className = 'badges';
  for (const value of [a.artifact_type, a.status, a.lineage_mode]) { const span = document.createElement('span'); span.className = 'badge'; span.textContent = value; badges.appendChild(span); }
  const dl = document.createElement('dl'); dl.className = 'detail-grid';
  addDetail(dl, 'ID', a.artifact_id);
  addDetail(dl, 'Label', a.label);
  addDetail(dl, 'Aliases', detail.aliases.join(', ') || '—');
  addDetail(dl, 'Size on disk', detail.size_display);
  if (detail.dimensions) addDetail(dl, 'Dimensions', `${Number(detail.dimensions.rows || 0).toLocaleString()} × ${Number(detail.dimensions.columns || 0).toLocaleString()}`);
  addDetail(dl, 'Operation', operation);
  addDetail(dl, 'Sources', sources);
  addDetail(dl, 'Basis', basis);
  const actionRow = document.createElement('div'); actionRow.className = 'artifact-actions';
  const detailsButton = document.createElement('button'); detailsButton.textContent = 'Load details'; detailsButton.disabled = !detail.descriptor_available; detailsButton.addEventListener('click', loadArtifactJson); actionRow.appendChild(detailsButton);
  if (detail.table_preview_available) { const previewButton = document.createElement('button'); previewButton.textContent = 'Preview table'; previewButton.addEventListener('click', openTablePreview); actionRow.appendChild(previewButton); }
  const jsonBox = document.createElement('section'); jsonBox.className = 'artifact-json'; jsonBox.id = 'artifactJsonBox'; jsonBox.hidden = true;
  const memoBox = document.createElement('section'); memoBox.className = 'artifact-memo'; memoBox.id = 'artifactMemoBox';
  container.append(heading, badges, dl, actionRow, jsonBox, memoBox);
  renderArtifactMemo();
}
function addDetail(dl, term, value) {
  const dt = document.createElement('dt'); dt.textContent = term;
  const dd = document.createElement('dd'); dd.textContent = value;
  dl.append(dt, dd);
}
async function loadArtifactJson() {
  if (!app.selectedArtifact) return;
  const box = $('artifactJsonBox');
  if (app.artifactDescriptor) {
    box.hidden = !box.hidden;
    return;
  }
  try {
    const payload = await api(`/api/artifact-json?${new URLSearchParams({artifact_id: app.selectedArtifact})}`);
    app.artifactDescriptor = payload.descriptor;
    box.innerHTML = '';
    const heading = document.createElement('h3'); heading.textContent = 'artifact.json';
    box.append(heading, jsonTree(payload.descriptor, 'artifact.json', false));
    box.hidden = false;
  } catch (error) { setStatus(error.message, true); }
}
function jsonTree(value, label, root = false) {
  if (value !== null && typeof value === 'object') {
    const details = document.createElement('details'); details.className = 'json-tree';
    if (root) details.open = true;
    const summary = document.createElement('summary');
    const size = Array.isArray(value) ? value.length : Object.keys(value).length;
    summary.textContent = `${label} ${Array.isArray(value) ? `[${size}]` : `{${size}}`}`;
    details.appendChild(summary);
    for (const [key, child] of Object.entries(value)) details.appendChild(jsonTree(child, key, false));
    return details;
  }
  const row = document.createElement('div'); row.className = 'json-scalar';
  const key = document.createElement('span'); key.className = 'json-key'; key.textContent = `${label}: `;
  const scalar = document.createElement('span'); scalar.textContent = value === null ? 'null' : JSON.stringify(value);
  row.append(key, scalar);
  return row;
}
function openTablePreview() {
  if (!app.selectedArtifact) return;
  app.tablePreview = {page: 1, pageCount: 1};
  $('previewKeys').checked = true;
  $('previewData').checked = true;
  $('previewMetadata').checked = false;
  $('previewFullMetadata').checked = false;
  $('previewFullMetadata').disabled = true;
  $('previewPageSize').value = '25';
  const preferred = app.artifactDetail.aliases.length ? app.artifactDetail.aliases[0] : app.artifactDetail.artifact.label;
  $('tablePreviewTitle').textContent = `Preview · ${preferred}`;
  $('tablePreviewDialog').showModal();
  loadTablePreview();
}
async function loadTablePreview() {
  if (!app.selectedArtifact) return;
  const showMetadata = $('previewMetadata').checked;
  $('previewFullMetadata').disabled = !showMetadata;
  if (!showMetadata) $('previewFullMetadata').checked = false;
  if (!$('previewKeys').checked && !$('previewData').checked && !showMetadata) {
    $('tablePreviewBody').innerHTML = '<div class="preview-error">Select at least one of Keys, Data, or Metadata.</div>';
    return;
  }
  const params = new URLSearchParams({
    artifact_id: app.selectedArtifact,
    page: String(app.tablePreview.page),
    page_size: $('previewPageSize').value,
    keys: String($('previewKeys').checked),
    data: String($('previewData').checked),
    metadata: String(showMetadata),
    full_metadata: String($('previewFullMetadata').checked),
  });
  $('tablePreviewBody').innerHTML = '<div class="empty">Loading page…</div>';
  try {
    const payload = await api(`/api/table-preview?${params}`);
    app.tablePreview.page = payload.page;
    app.tablePreview.pageCount = payload.page_count;
    renderTablePreview(payload);
  } catch (error) {
    $('tablePreviewBody').innerHTML = '';
    const message = document.createElement('div'); message.className = 'preview-error'; message.textContent = error.message; $('tablePreviewBody').appendChild(message);
  }
}
function renderTablePreview(payload) {
  $('tablePreviewMeta').textContent = `${Number(payload.n_rows).toLocaleString()} rows`;
  $('previewPageLabel').textContent = `Page ${payload.page.toLocaleString()} of ${payload.page_count.toLocaleString()}`;
  $('previewPrevious').disabled = payload.page <= 1;
  $('previewNext').disabled = payload.page >= payload.page_count;
  const wrap = $('tablePreviewBody'); wrap.innerHTML = '';
  if (!payload.columns.length) { wrap.innerHTML = '<div class="empty">No columns selected.</div>'; return; }
  const table = document.createElement('table'); table.className = 'preview-table';
  const thead = document.createElement('thead'); const header = document.createElement('tr');
  for (const column of payload.columns) { const th = document.createElement('th'); th.textContent = column; header.appendChild(th); }
  thead.appendChild(header); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const row of payload.rows) {
    const tr = document.createElement('tr');
    for (const value of row) { const td = document.createElement('td'); td.textContent = previewCell(value); tr.appendChild(td); }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody); wrap.appendChild(table);
}
function previewCell(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
function renderArtifactMemo() {
  const box = $('artifactMemoBox');
  const memo = app.artifactDetail.memo;
  box.innerHTML = '';
  const heading = document.createElement('h3'); heading.textContent = 'Artifact memo'; box.appendChild(heading);
  if (!memo) {
    const text = document.createElement('div'); text.className = 'empty'; text.textContent = 'No memo has been created for this artifact.';
    const button = document.createElement('button'); button.className = 'primary'; button.textContent = 'Add Memo'; button.addEventListener('click', () => showArtifactMemoEditor(null));
    box.append(text, button);
    setArtifactMemoBaseline('', '');
    return;
  }
  showArtifactMemoEditor(memo);
}
function showArtifactMemoEditor(memo, resetBaseline = true) {
  const box = $('artifactMemoBox'); box.innerHTML = '<h3>Artifact memo</h3>';
  const title = document.createElement('input'); title.id = 'artifactMemoTitle'; title.placeholder = 'Optional title'; title.value = memo?.title || '';
  const body = document.createElement('textarea'); body.id = 'artifactMemoBody'; body.placeholder = 'Write artifact memo in Markdown...'; body.value = memo?.body || '';
  const preview = document.createElement('div'); preview.id = 'artifactMemoPreviewPane'; preview.className = 'preview markdown-body'; preview.hidden = true;
  const modes = document.createElement('div'); modes.className = 'editor-mode';
  const raw = document.createElement('button'); raw.id = 'artifactMemoRaw'; raw.className = 'active'; raw.textContent = 'Markdown';
  const previewButton = document.createElement('button'); previewButton.id = 'artifactMemoPreview'; previewButton.textContent = 'Preview';
  raw.addEventListener('click', () => setArtifactMemoMode('raw')); previewButton.addEventListener('click', () => setArtifactMemoMode('preview'));
  modes.append(raw, previewButton);
  const actions = document.createElement('div'); actions.className = 'artifact-memo-actions';
  if (memo) {
    const history = document.createElement('button'); history.textContent = 'History'; history.addEventListener('click', showArtifactMemoHistory); actions.appendChild(history);
  }
  const save = document.createElement('button'); save.className = 'primary'; save.textContent = 'Save'; save.addEventListener('click', saveArtifactMemo); actions.appendChild(save);
  box.append(title, modes, body, preview, actions);
  if (resetBaseline) setArtifactMemoBaseline(title.value, body.value);
}
function setArtifactMemoMode(mode) {
  const body = $('artifactMemoBody'); const preview = $('artifactMemoPreviewPane');
  if (!body || !preview) return;
  const showPreview = mode === 'preview';
  $('artifactMemoRaw').classList.toggle('active', !showPreview); $('artifactMemoPreview').classList.toggle('active', showPreview);
  body.hidden = showPreview; preview.hidden = !showPreview;
  if (showPreview) preview.innerHTML = renderMarkdown(body.value);
}
async function saveArtifactMemo() {
  const title = $('artifactMemoTitle'); const body = $('artifactMemoBody');
  if (!title || !body || !body.value.trim()) { setStatus('Artifact memo body cannot be empty.', true); return; }
  try {
    const payload = await api('/api/save', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({target_type:'artifact', target_id:app.selectedArtifact, title:title.value || null, body:body.value})});
    app.artifactDetail.memo = payload.memo;
    setArtifactMemoBaseline(payload.memo.title || '', payload.memo.body || '');
    await loadState(true);
    renderArtifactDetail();
    setStatus(`Saved artifact memo version ${payload.memo.memo_id}.`);
  } catch (error) { setStatus(error.message, true); }
}
async function showArtifactMemoHistory() {
  const memo = app.artifactDetail.memo;
  if (!memo) return;
  try {
    const params = new URLSearchParams({target_type:'artifact', target_id:app.selectedArtifact});
    const payload = await api(`/api/history?${params}`);
    const choices = payload.versions.map((version, index) => `${index}: ${index === 0 ? 'Current' : `Previous ${index}`} — ${formatWhen(version.created_at)}`).join('\n');
    const raw = window.prompt(`Artifact memo versions:\n${choices}\n\nEnter a version number to load:`, '0');
    if (raw === null) return;
    const index = Number(raw);
    if (!Number.isInteger(index) || index < 0 || index >= payload.versions.length) { setStatus('Invalid history selection.', true); return; }
    if (!confirmArtifactMemoDiscard()) return;
    showArtifactMemoEditor(payload.versions[index], false);
    setStatus(index === 0 ? 'Current artifact memo loaded.' : `Loaded historical artifact memo version ${payload.versions[index].memo_id}; Save to restore it as a new version.`);
  } catch (error) { setStatus(error.message, true); }
}

function activateTab(tab) {
  if (tab !== 'memos' && tab !== 'artifacts') tab = 'memos';
  if (tab !== activeTab()) {
    if (activeTab() === 'memos' && !confirmMemoDiscard()) return;
    if (activeTab() === 'artifacts' && !confirmArtifactMemoDiscard()) return;
  }
  for (const button of document.querySelectorAll('.tab-button')) button.classList.toggle('active', button.dataset.tab === tab);
  for (const panel of document.querySelectorAll('.tab')) panel.classList.toggle('active', panel.id === `tab-${tab}`);
  const url = new URL(window.location.href); url.searchParams.set('tab', tab); history.replaceState(null, '', url);
}
function activeTab() {
  return document.querySelector('.tab-button.active')?.dataset.tab || null;
}

$('endSession').addEventListener('click', endSession);
$('newMemo').addEventListener('click', newStandaloneMemo);
$('refreshMemos').addEventListener('click', async () => { if (confirmMemoDiscard()) { await loadState(true); setStatus('Memos refreshed.'); } });
$('searchMemos').addEventListener('input', renderMemoList);
$('filterMemoType').addEventListener('change', renderMemoList);
$('memoTitle').addEventListener('input', () => {});
$('memoBody').addEventListener('input', updateMemoPreview);
$('saveMemo').addEventListener('click', saveMemo);
$('memoHistoryButton').addEventListener('click', toggleMemoHistory);
$('memoRaw').addEventListener('click', () => setMemoMode('raw'));
$('memoPreview').addEventListener('click', () => setMemoMode('preview'));
$('graphMode').addEventListener('change', renderArtifactGraph);
$('focusSelected').addEventListener('click', () => {
  if (!(app.selectedArtifact || app.selectedOperation)) return;
  app.graphFocused = !app.graphFocused;
  renderArtifactGraph();
});
$('artifactSearch').addEventListener('input', renderArtifactGraph);
$('refreshArtifacts').addEventListener('click', async () => { if (confirmArtifactMemoDiscard()) { await loadArtifactGraph(true); setStatus('Artifacts refreshed.'); } });
$('closeTablePreview').addEventListener('click', () => $('tablePreviewDialog').close());
for (const id of ['previewKeys', 'previewData', 'previewMetadata', 'previewFullMetadata']) $(id).addEventListener('change', () => { app.tablePreview.page = 1; loadTablePreview(); });
$('previewPageSize').addEventListener('change', () => { app.tablePreview.page = 1; loadTablePreview(); });
$('previewPrevious').addEventListener('click', () => { if (app.tablePreview.page > 1) { app.tablePreview.page -= 1; loadTablePreview(); } });
$('previewNext').addEventListener('click', () => { if (app.tablePreview.page < app.tablePreview.pageCount) { app.tablePreview.page += 1; loadTablePreview(); } });
for (const button of document.querySelectorAll('.tab-button')) button.addEventListener('click', () => activateTab(button.dataset.tab));
window.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
    event.preventDefault();
    if (activeTab() === 'memos') saveMemo();
    else if (activeTab() === 'artifacts' && $('artifactMemoBody')) saveArtifactMemo();
  }
});
window.addEventListener('beforeunload', event => {
  if (!app.sessionEnding && (memoIsDirty() || artifactMemoIsDirty())) { event.preventDefault(); event.returnValue = ''; }
});

Promise.all([loadState(false), loadArtifactGraph(false)]).then(() => {
  const requested = new URL(window.location.href).searchParams.get('tab') || 'memos';
  activateTab(requested);
  const projectMemo = app.state.memos.find(memo => memo.target_type === 'project' && memo.target_id === 'project');
  if (projectMemo) { app.selectedMemo = projectMemo; loadMemoIntoEditor(projectMemo, true); }
  else newStandaloneMemo();
}).catch(error => setStatus(error.message, true));
</script>
</body>
</html>
"""
