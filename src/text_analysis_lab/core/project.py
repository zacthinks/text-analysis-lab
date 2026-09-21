"""Project lifecycle and lookup shell for TextAnalysisLab (TeAL).

`Project` owns project-local services: storage paths, the SQLite catalog, the
DuckDB query engine, high-level lookup helpers, memo access, and thin execution
facades over operation-specific modules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.aggregate import (
    AggregateField,
    ConcatReducer,
    FieldAggregationSpec,
    LiteralValue,
)
from text_analysis_lab.core.aggregate import (
    aggregate as _aggregate,
)
from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.artifact_subclasses import load_artifact
from text_analysis_lab.core.binary_code import binary_code as _binary_code
from text_analysis_lab.core.catalog import ProjectCatalog
from text_analysis_lab.core.collapse_runs import collapse_runs as _collapse_runs
from text_analysis_lab.core.errors import (
    ArtifactError,
    ArtifactNotFoundError,
    OperatorError,
    OperatorNotFoundError,
)
from text_analysis_lab.core.feature_subset import (
    FeatureSubsetFunction,
)
from text_analysis_lab.core.feature_subset import (
    feature_subset as _feature_subset,
)
from text_analysis_lab.core.idempotence import (
    AliasSpec,
    finalize_alias_plan,
    prepare_alias_plan,
    reused_outputs,
)
from text_analysis_lab.core.importers import (
    folder_inventory as _folder_inventory,
)
from text_analysis_lab.core.importers import (
    read_csv as _read_csv,
)
from text_analysis_lab.core.importers import (
    read_csv_folder as _read_csv_folder,
)
from text_analysis_lab.core.importers import (
    read_excel as _read_excel,
)
from text_analysis_lab.core.importers import (
    read_excel_folder as _read_excel_folder,
)
from text_analysis_lab.core.importers import (
    read_jsonl as _read_jsonl,
)
from text_analysis_lab.core.importers import (
    read_parquet as _read_parquet,
)
from text_analysis_lab.core.join import join as _join
from text_analysis_lab.core.keyed_frame import from_keyed_frame as _from_keyed_frame
from text_analysis_lab.core.keyed_metadata import attach_metadata as _attach_metadata
from text_analysis_lab.core.merge import merge as _merge
from text_analysis_lab.core.operator import BaseOperator, BaseTranslator
from text_analysis_lab.core.probability_split import (
    probability_split as _probability_split,
)
from text_analysis_lab.core.query import QueryEngine
from text_analysis_lab.core.register_external import (
    register_external as _register_external,
)
from text_analysis_lab.core.restrict import restrict as _restrict
from text_analysis_lab.core.sample import sample as _sample
from text_analysis_lab.core.select_keys import select_keys as _select_keys
from text_analysis_lab.core.set_primary_keys import (
    set_primary_keys as _set_primary_keys,
)
from text_analysis_lab.core.split import split as _split
from text_analysis_lab.core.storage import ProjectStorage
from text_analysis_lab.core.subset import FunctionSpec
from text_analysis_lab.core.subset import subset as _subset
from text_analysis_lab.core.transform_like import (
    can_transform_texts_like as _can_transform_texts_like,
)
from text_analysis_lab.core.transform_like import (
    transform_texts_like as _transform_texts_like,
)
from text_analysis_lab.core.translate import (
    resume_translate as _resume_translate,
)
from text_analysis_lab.core.translate import (
    translate as _translate,
)
from text_analysis_lab.core.types import (
    DEFAULT_OUTPUT_LABEL,
    ArtifactStatus,
    ArtifactType,
    ColumnSelect,
    LineageMode,
    MetadataMode,
    OperationStatus,
    OperationType,
    OperatorSnapshotStatus,
    QueryForm,
)

if TYPE_CHECKING:
    from text_analysis_lab.gui import ProjectCenterServer


class Project:
    """Persistent TeAL workspace.

    The project object is deliberately small. It provides access to storage,
    catalog lookup, query services, memo services, and thin operation facades.
    """

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str,
        *,
        delete_existing: bool = False,
    ) -> Project:
        """Create or reopen a named project directory.

        ``delete_existing=True`` removes only TeAL-owned state under
        ``path/.teal`` before creating the project. Files outside ``.teal`` in
        the containing directory are preserved.
        """
        return cls(
            ProjectStorage.initialize(
                path,
                name=name,
                delete_existing=delete_existing,
            )
        )

    @classmethod
    def open(cls, path: str | Path) -> Project:
        """Open an existing TeAL project directory."""
        return cls(ProjectStorage.open(path))

    def __init__(self, storage: ProjectStorage) -> None:
        self.storage = storage
        self.catalog = ProjectCatalog(self.storage.catalog_dir)
        self.query = QueryEngine(self)
        self._geco_manager = None
        self._project_centers: list[ProjectCenterServer] = []
        self._closed = False

    def __enter__(self) -> Project:  # noqa: PYI034 - keep Python 3.10 base deps minimal
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort fallback only
        with suppress(Exception):
            self.close()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return False
        return self.storage.teal_dir.resolve() == other.storage.teal_dir.resolve()

    def __hash__(self) -> int:
        return hash(self.storage.teal_dir.resolve())

    def __str__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"{self.name} [TeAL project: {state}, {self.path}]"

    def __repr__(self) -> str:
        artifact_count = len(self.list_artifacts()) if not self._closed else None
        count_text = (
            "closed" if artifact_count is None else f"artifacts={artifact_count:,}"
        )
        return f"Project(name={self.name!r}, path={str(self.path)!r}, {count_text})"

    def _repr_html_(self) -> str:
        from html import escape

        artifact_count = len(self.list_artifacts()) if not self._closed else None
        fields = [
            ("Project", self.name),
            ("Path", str(self.path)),
            ("State", "closed" if self._closed else "open"),
            ("Artifacts", "-" if artifact_count is None else f"{artifact_count:,}"),
        ]
        rows_html = "".join(
            f"<tr><th style='text-align:left;padding:2px 10px 2px 0'>{escape(str(key))}</th>"
            f"<td style='text-align:left;padding:2px 0'>{escape(str(value))}</td></tr>"
            for key, value in fields
        )
        return f"<table>{rows_html}</table>"

    def close(self) -> None:
        """Close project-owned database connections and clear query caches."""
        if self._closed:
            return
        if self._geco_manager is not None:
            self._geco_manager.close()
        for project_center in self._project_centers:
            with suppress(Exception):
                project_center.close()
        self._project_centers.clear()
        self.query.close()
        self.catalog.close()
        self._closed = True

    @property
    def path(self) -> Path:
        return self.storage.project_path

    @property
    def manifest(self) -> Path:
        return self.storage.manifest_path

    @property
    def project_id(self) -> str:
        """Return the stable project id recorded in the project manifest."""
        payload = json.loads(self.storage.manifest_path.read_text(encoding="utf-8"))
        project = payload.get("project", {})
        project_id = project.get("project_id")
        if not isinstance(project_id, str) or not project_id:
            raise RuntimeError("Project manifest is missing project.project_id.")
        return project_id

    @property
    def name(self) -> str:
        """Return the project name; project names are also project IDs."""
        return self.project_id

    @property
    def operations(self) -> Path:
        return self.storage.operations_dir

    @property
    def geco(self):
        """Return the manager for optional TeAL-linked GeCo workspaces."""
        if self._geco_manager is None:
            from text_analysis_lab.integrations.geco import GeCoManager

            self._geco_manager = GeCoManager(self)
        return self._geco_manager

    # ------------------------------------------------------------------
    # Artifact access
    # ------------------------------------------------------------------

    def resolve_artifact_id(
        self,
        ref: BaseArtifact | str,
        *,
        include_deleted: bool = False,
    ) -> str:
        """Resolve an artifact object, ID, or alias to an artifact ID."""
        if isinstance(ref, BaseArtifact):
            if ref.project != self:
                raise ArtifactError("Artifact belongs to a different project.")
            return ref.artifact_id
        row = self.catalog.resolve_artifact(ref, include_deleted=include_deleted)
        return str(row["artifact_id"])

    def get_artifact(
        self,
        ref: BaseArtifact | str,
        *,
        include_deleted: bool = False,
    ) -> BaseArtifact:
        """Load an artifact by object, artifact ID, or catalog alias."""
        if isinstance(ref, BaseArtifact):
            if ref.project != self:
                raise ArtifactError("Artifact belongs to a different project.")
            return ref

        artifact_id = self.resolve_artifact_id(ref, include_deleted=include_deleted)
        artifact_dir = self.storage.artifact_dir(artifact_id)
        if not artifact_dir.exists():
            raise ArtifactNotFoundError(artifact_id)
        return load_artifact(self, artifact_dir)

    def list_artifacts(
        self,
        *,
        status: ArtifactStatus | None = None,
        include_deleted: bool = False,
        artifact_type: ArtifactType | str | None = None,
        lineage_mode: LineageMode | str | None = None,
        basis_artifact_id: str | None = None,
        source_artifact_id: str | None = None,
        operation_id: str | None = None,
        operator_id: str | None = None,
        include_aliases: bool = True,
    ) -> list[dict[str, Any]]:
        """Return catalog records for project artifacts.

        Artifact descriptors remain on disk and are loaded by `get_artifact(...)`.
        The catalog is the source of truth for lookup, filtering, deletion state,
        and project graph fields.
        """
        rows = self.catalog.list_artifacts(
            status=status,
            include_deleted=include_deleted,
            artifact_type=artifact_type,
            lineage_mode=lineage_mode,
            basis_artifact_id=basis_artifact_id,
            source_artifact_id=source_artifact_id,
            operation_id=operation_id,
            operator_id=operator_id,
        )
        if not include_aliases:
            return rows
        return [
            {
                **row,
                "aliases": self.catalog.aliases_for_artifact(str(row["artifact_id"])),
            }
            for row in rows
        ]

    def add_artifact_alias(self, ref: BaseArtifact | str, alias: str) -> None:
        """Add a project-level alias for an artifact."""
        self.catalog.add_artifact_alias(self.resolve_artifact_id(ref), alias)

    def remove_artifact_alias(self, alias: str) -> None:
        """Remove a project-level artifact alias."""
        self.catalog.remove_artifact_alias(alias)

    def prefer_artifact_alias(self, alias: str) -> None:
        """Mark an artifact alias as the preferred display alias."""
        self.catalog.touch_artifact_alias(alias)

    def delete_artifact(self, ref: BaseArtifact | str) -> None:
        """Mark an artifact as deleted in the catalog and remove its aliases."""
        self.catalog.mark_artifact_deleted(self.resolve_artifact_id(ref))

    def sql(
        self,
        artifacts: Sequence[BaseArtifact | str],
        query: str,
    ):
        """Run SQL against explicitly registered artifact views.

        Each artifact is registered under `str(ref)`: artifact objects use their
        artifact ID, while string refs use the exact ID or alias supplied by the
        caller. This keeps SQL registration explicit and avoids parsing SQL to
        guess artifact names.
        """
        return self.query.project_sql(artifacts, query)

    # ------------------------------------------------------------------
    # Operator access
    # ------------------------------------------------------------------

    def resolve_operator_id(
        self,
        ref: BaseOperator | str,
        *,
        include_deleted: bool = False,
    ) -> str:
        """Resolve an operator object, ID, or alias to an operator ID."""
        if isinstance(ref, BaseOperator):
            if ref.operator_id is None:
                raise OperatorNotFoundError("Operator object has no operator_id.")
            if not include_deleted:
                self.catalog.resolve_operator(ref.operator_id, include_deleted=False)
            return str(ref.operator_id)
        row = self.catalog.resolve_operator(ref, include_deleted=include_deleted)
        return str(row["operator_id"])

    def get_operator(
        self,
        ref: BaseOperator | str,
        *,
        include_deleted: bool = False,
    ) -> BaseOperator:
        """Load a frozen operator by object, operator ID, or catalog alias."""
        if isinstance(ref, BaseOperator):
            if ref.operator_id is None:
                raise OperatorNotFoundError("Operator object has no operator_id.")
            row = self.catalog.resolve_operator(
                ref.operator_id, include_deleted=include_deleted
            )
            snapshot_status = str(row.get("snapshot_status", "serialized"))
            if snapshot_status != "serialized":
                raise OperatorError(
                    f"Operator {ref.operator_id} is not loadable because "
                    f"snapshot_status={snapshot_status!r}."
                )
            return ref

        row = self.catalog.resolve_operator(ref, include_deleted=include_deleted)
        operator_id = str(row["operator_id"])
        snapshot_status = str(row.get("snapshot_status", "serialized"))
        if snapshot_status != "serialized":
            raise OperatorError(
                f"Operator {operator_id} is not loadable because "
                f"snapshot_status={snapshot_status!r}."
            )
        operator_dir = self.storage.operator_dir(operator_id)
        if not operator_dir.exists():
            raise OperatorNotFoundError(operator_id)
        return BaseOperator.load_from_dir(operator_dir)

    def list_operators(
        self,
        *,
        include_deleted: bool = False,
        operation_type: OperationType | str | None = None,
        snapshot_status: OperatorSnapshotStatus | str | None = None,
        include_aliases: bool = True,
    ) -> list[dict[str, Any]]:
        """Return catalog records for project operators."""
        rows = self.catalog.list_operators(
            include_deleted=include_deleted,
            operation_type=operation_type,
            snapshot_status=snapshot_status,
        )
        if not include_aliases:
            return rows
        return [
            {
                **row,
                "aliases": self.catalog.aliases_for_operator(str(row["operator_id"])),
            }
            for row in rows
        ]

    def add_operator_alias(self, ref: BaseOperator | str, alias: str) -> None:
        """Add a project-level alias for a frozen operator."""
        self.catalog.add_operator_alias(self.resolve_operator_id(ref), alias)

    def remove_operator_alias(self, alias: str) -> None:
        """Remove a project-level operator alias."""
        self.catalog.remove_operator_alias(alias)

    def prefer_operator_alias(self, alias: str) -> None:
        """Mark an operator alias as the preferred display alias."""
        self.catalog.touch_operator_alias(alias)

    def delete_operator(self, ref: BaseOperator | str) -> None:
        """Mark an operator as deleted in the catalog and remove its aliases."""
        self.catalog.mark_operator_deleted(self.resolve_operator_id(ref))

    # ------------------------------------------------------------------
    # Operation records
    # ------------------------------------------------------------------

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        """Return one operation catalog record."""
        return self.catalog.get_operation(str(operation_id))

    def list_operations(
        self,
        *,
        operation_type: OperationType | str | None = None,
        operator: BaseOperator | str | None = None,
        status: OperationStatus | str | None = None,
    ) -> list[dict[str, Any]]:
        """Return operation catalog records with optional filters."""
        operator_id = None if operator is None else self.resolve_operator_id(operator)
        return self.catalog.list_operations(
            operation_type=operation_type,
            operator_id=operator_id,
            status=status,
        )

    def operation_sources(self, operation_id: str) -> list[dict[str, Any]]:
        """Return source artifact edges for an operation."""
        return self.catalog.operation_sources(str(operation_id))

    def operation_outputs(self, operation_id: str) -> list[dict[str, Any]]:
        """Return output artifact edges for an operation."""
        return self.catalog.operation_outputs(str(operation_id))

    def operation_for_artifact(
        self,
        artifact: BaseArtifact | str,
    ) -> dict[str, Any] | None:
        """Return the operation that produced an artifact, if one is recorded."""
        return self.catalog.operation_for_artifact(self.resolve_artifact_id(artifact))

    # ------------------------------------------------------------------
    # Memo access
    # ------------------------------------------------------------------

    def update_project_memo(self, body: str, *, title: str | None = None) -> int:
        """Append a new version of the project memo and return the memo row id."""
        return self.catalog.add_memo(
            target_type="project",
            target_id="project",
            title=title,
            body=body,
        )

    def get_project_memo(self) -> str | None:
        """Return the latest project memo body, if one exists."""
        row = self.catalog.get_memo(
            target_type="project",
            target_id="project",
        )
        return None if row is None else str(row["body"])

    def add_standalone_memo(self, body: str, *, title: str | None = None) -> int:
        """Create a standalone memo and return its memo row id."""
        return self.catalog.add_memo(
            target_type="standalone",
            target_id=None,
            title=title,
            body=body,
        )

    def list_standalone_memos(self) -> list[dict[str, Any]]:
        """Return the latest version of each standalone memo."""
        return self.catalog.list_memos(
            target_type="standalone",
            latest_only=True,
        )

    def get_standalone_memo(self, target_id: str | int) -> str:
        """Return the latest body for one standalone memo target id."""
        row = self.catalog.get_memo(
            target_type="standalone",
            target_id=str(target_id),
        )
        if row is None:
            raise KeyError(f"No standalone memo found for target_id={target_id!r}.")
        return str(row["body"])

    def _launch_project_center(
        self,
        *,
        initial_tab: str,
        port: int,
        open_browser: bool,
    ) -> ProjectCenterServer:
        from text_analysis_lab.gui import launch_project_center

        project_center = launch_project_center(
            catalog_dir=self.storage.catalog_dir,
            project_name=self.name,
            manifest_path=self.storage.manifest_path,
            initial_tab=initial_tab,
            port=port,
            open_browser=open_browser,
        )
        self._project_centers.append(project_center)
        print(f"TeAL Project Center running at {project_center.url_for(initial_tab)}")
        return project_center

    def launch_project_center(
        self,
        *,
        port: int = 0,
        open_browser: bool = True,
    ) -> ProjectCenterServer:
        """Launch the local TeAL Project Center on its Artifacts tab.

        The Project Center is TeAL's localhost-only interactive interface. The
        returned handle exposes ``url`` and ``close()`` and is closed
        automatically when the project closes. ``port=0`` asks the operating
        system to choose an available local port.
        """
        return self._launch_project_center(
            initial_tab="artifacts",
            port=port,
            open_browser=open_browser,
        )

    def launch_memo_center(
        self,
        *,
        port: int = 0,
        open_browser: bool = True,
    ) -> ProjectCenterServer:
        """Launch the TeAL Project Center directly on its Memos tab."""
        return self._launch_project_center(
            initial_tab="memos",
            port=port,
            open_browser=open_browser,
        )

    def launch_artifact_map(
        self,
        *,
        port: int = 0,
        open_browser: bool = True,
    ) -> ProjectCenterServer:
        """Launch the TeAL Project Center directly on its Artifacts tab."""
        return self._launch_project_center(
            initial_tab="artifacts",
            port=port,
            open_browser=open_browser,
        )

    # ------------------------------------------------------------------
    # Execution facades
    # ------------------------------------------------------------------

    def read_csv(
        self,
        path: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        duckdb_options: Mapping[str, Any] | None = None,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import explicitly selected CSV text/metadata fields via DuckDB."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _read_csv(
            self,
            path,
            text_fields=text_fields,
            metadata_fields=metadata_fields,
            batch_size=batch_size,
            output_label=output_label,
            duckdb_options=duckdb_options,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def read_csv_folder(
        self,
        root: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        pattern: str = "*.csv",
        recursive: bool = True,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        duckdb_options: Mapping[str, Any] | None = None,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import a deterministic folder of CSV files via DuckDB.

        Adds generic ``source_file`` (relative path) and zero-based ``source_row``
        metadata. Filename semantics remain corpus-specific downstream logic.
        """
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _read_csv_folder(
            self,
            root,
            text_fields=text_fields,
            metadata_fields=metadata_fields,
            pattern=pattern,
            recursive=recursive,
            batch_size=batch_size,
            output_label=output_label,
            duckdb_options=duckdb_options,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def read_excel(
        self,
        path: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        sheets: int | str | Sequence[int | str] | None = None,
        header_row: int = 0,
        missing_fields: Any = ...,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import selected fields from one XLSX workbook.

        ``sheets=None`` imports all workbook sheets. Sheet selectors may mix exact
        names and zero-based integer positions. Excel source provenance is attached
        as sheet name, sheet index, and zero-based source row metadata.
        """
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        kwargs: dict[str, Any] = {
            "text_fields": text_fields,
            "metadata_fields": metadata_fields,
            "sheets": sheets,
            "header_row": header_row,
            "batch_size": batch_size,
            "output_label": output_label,
            "memo": memo,
        }
        if missing_fields is not ...:
            kwargs["missing_fields"] = missing_fields
        result = _read_excel(self, path, **kwargs)
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def read_excel_folder(
        self,
        root: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        sheets: int | str | Sequence[int | str] | None = None,
        header_row: int = 0,
        missing_fields: Any = ...,
        pattern: str = "*.xlsx",
        recursive: bool = True,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import selected sheets from a deterministic folder of XLSX workbooks.

        Adds ``source_file``, ``source_sheet``, ``source_sheet_index``, and
        zero-based ``source_row`` metadata. Workbooks are visited in sorted
        relative-path order and selected sheets in requested/workbook order.
        """
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        kwargs: dict[str, Any] = {
            "text_fields": text_fields,
            "metadata_fields": metadata_fields,
            "sheets": sheets,
            "header_row": header_row,
            "pattern": pattern,
            "recursive": recursive,
            "batch_size": batch_size,
            "output_label": output_label,
            "memo": memo,
        }
        if missing_fields is not ...:
            kwargs["missing_fields"] = missing_fields
        result = _read_excel_folder(self, root, **kwargs)
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def read_jsonl(
        self,
        path: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        duckdb_options: Mapping[str, Any] | None = None,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import explicitly selected JSONL text/metadata fields via DuckDB."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _read_jsonl(
            self,
            path,
            text_fields=text_fields,
            metadata_fields=metadata_fields,
            batch_size=batch_size,
            output_label=output_label,
            duckdb_options=duckdb_options,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def read_parquet(
        self,
        path: str | Path,
        *,
        text_fields: str | Sequence[str],
        metadata_fields: str | Sequence[str] | None,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        duckdb_options: Mapping[str, Any] | None = None,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import explicitly selected Parquet text/metadata fields via DuckDB."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _read_parquet(
            self,
            path,
            text_fields=text_fields,
            metadata_fields=metadata_fields,
            batch_size=batch_size,
            output_label=output_label,
            duckdb_options=duckdb_options,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def folder_inventory(
        self,
        root: str | Path,
        *,
        patterns: str | Sequence[str] = "*",
        recursive: bool = True,
        batch_size: int = 10_000,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Inventory matching file paths without reading file contents."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _folder_inventory(
            self,
            root,
            patterns=patterns,
            recursive=recursive,
            batch_size=batch_size,
            output_label=output_label,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def translate(
        self,
        translator: BaseTranslator,
        sources: BaseArtifact
        | str
        | Sequence[BaseArtifact | str]
        | Mapping[str, BaseArtifact | str],
        *,
        workers: int = 1,
        batch_size: int | None = None,
        max_outstanding_units: int | None = None,
        memo: str | None = None,
        alias: AliasSpec = None,
        overwrite: bool = False,
        **params: Any,
    ) -> Mapping[str, BaseArtifact]:
        """Run a translation operation and return output artifacts by label.

        ``params`` contains translator-specific operation arguments. Callers may
        use this generic public API directly, or a user-facing translator may
        expose a typed ``translate(project, ...)`` convenience method that
        delegates here.
        """
        return _translate(
            self,
            translator,
            sources,
            workers=workers,
            batch_size=batch_size,
            max_outstanding_units=max_outstanding_units,
            memo=memo,
            alias=alias,
            overwrite=overwrite,
            **params,
        )

    def resume_operation(
        self,
        operation_id: str,
    ) -> Mapping[str, BaseArtifact]:
        """Resume a previously recorded resumable operation as configured."""
        operation = self.catalog.get_operation(str(operation_id))
        operation_type = str(operation["operation_type"])
        if operation_type in {"translate", "subset"}:
            return _resume_translate(self, str(operation_id))
        raise OperatorError(
            f"Operation {operation_id!r} has unsupported operation_type "
            f"{operation_type!r} for resume_operation(...)."
        )

    def subset(
        self,
        source: BaseArtifact | str,
        function: FunctionSpec,
        *,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        form: QueryForm = "table",
        iter_batches: bool = True,
        batch_size: int | None = None,
        include_position: bool = True,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        workers: int = 1,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> Mapping[str, BaseArtifact]:
        """Create a keys-only subset artifact using a boolean-mask function."""
        return _subset(
            self,
            source,
            function,
            key_columns=key_columns,
            data_columns=data_columns,
            metadata_columns=metadata_columns,
            metadata_mode=metadata_mode,
            form=form,
            iter_batches=iter_batches,
            batch_size=batch_size,
            include_position=include_position,
            output_label=output_label,
            workers=workers,
            memo=memo,
            alias=alias,
            overwrite=overwrite,
        )

    def feature_subset(
        self,
        source: BaseArtifact | str,
        function: FeatureSubsetFunction,
        *,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Create a lazy positional feature view of a matrix artifact."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _feature_subset(
            self,
            source,
            function,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def set_primary_keys(
        self,
        source: BaseArtifact | str,
        *,
        levels: Mapping[str, str],
        leaf_key: str,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Replace the primary-key namespace without copying data or metadata."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _set_primary_keys(
            self,
            source,
            levels=levels,
            leaf_key=leaf_key,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def collapse_runs(
        self,
        source: BaseArtifact | str,
        *,
        by: str | Sequence[str],
        data: FieldAggregationSpec | None = None,
        metadata: FieldAggregationSpec | None = None,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Collapse maximal adjacent runs into a span-key table artifact."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _collapse_runs(
            self,
            source,
            by=by,
            data=data,
            metadata=metadata,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def select_keys(
        self,
        source: BaseArtifact | str,
        keys: Sequence[Any],
        *,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Select exact stable keys into a keys-only preserved-key child."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _select_keys(
            self,
            source,
            keys,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def sample(
        self,
        source: BaseArtifact | str,
        *,
        n: int,
        random_state: int | None = None,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Draw a random sample without replacement into one keys-only child."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _sample(
            self,
            source,
            n=n,
            random_state=random_state,
            output_label=output_label,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def restrict(
        self,
        source: BaseArtifact | str,
        *,
        to: BaseArtifact | str,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Restrict a source artifact to another artifact's compatible key domain."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _restrict(
            self,
            source,
            to=to,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def aggregate(
        self,
        source: BaseArtifact | str,
        *,
        to_key: str | Sequence[str],
        data: str
        | Mapping[str, str | ConcatReducer | AggregateField | LiteralValue]
        | None = None,
        metadata: Mapping[str, str | ConcatReducer | AggregateField | LiteralValue]
        | None = None,
        aggregations: Mapping[str, str | Sequence[str] | Mapping[str, str]]
        | None = None,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Reduce an artifact to a retained key prefix using group-safe aggregation.

        Table/JSONL data and metadata use field-level aggregation mappings. Matrix
        data takes one uniform reducer, ``"sum"`` or ``"mean"``, while metadata
        remains field-level. ``aggregations=`` retains the older relational syntax
        for backward compatibility.
        """
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _aggregate(
            self,
            source,
            to_key=to_key,
            data=data,
            metadata=metadata,
            aggregations=aggregations,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def split(
        self,
        source: BaseArtifact | str,
        *,
        labels: Sequence[str] = ("train", "test"),
        proportions: Sequence[float] = (0.8, 0.2),
        random_state: int | None = None,
        stratify: str | Sequence[str] | None = None,
        workers: int = 1,
        memo: str | None = None,
        alias: Mapping[str, str] | None = None,
        overwrite: bool = False,
    ) -> Mapping[str, BaseArtifact]:
        """Split one source artifact into keys-only child artifacts."""
        return _split(
            self,
            source,
            labels=labels,
            proportions=proportions,
            random_state=random_state,
            stratify=stratify,
            workers=workers,
            memo=memo,
            alias=alias,
            overwrite=overwrite,
        )

    def probability_split(
        self,
        source: BaseArtifact | str,
        *,
        n: int,
        remainder_label: str = "remainder",
        sample_label: str = "sample",
        strata: BaseArtifact | str | None = None,
        allocation: Mapping[Any, float] | None = None,
        random_state: int | None = None,
        workers: int = 1,
        memo: str | None = None,
        alias: Mapping[str, str] | None = None,
        overwrite: bool = False,
    ) -> Mapping[str, BaseArtifact]:
        """Draw a fixed-size probability audit sample and inclusion probabilities."""
        return _probability_split(
            self,
            source,
            n=n,
            remainder_label=remainder_label,
            sample_label=sample_label,
            strata=strata,
            allocation=allocation,
            random_state=random_state,
            workers=workers,
            memo=memo,
            alias=alias,
            overwrite=overwrite,
        )

    def register_external(
        self,
        external: Any,
        *,
        artifact_type: ArtifactType | str = ArtifactType.TABLE,
        primary_key: str | Sequence[str],
        data_fields: str | Sequence[str] | None = None,
        metadata_fields: str | Sequence[str] | None = None,
        format: str | None = None,
        batch_size: int = 10_000,
        duckdb_options: Mapping[str, Any] | None = None,
        sources: Mapping[str, BaseArtifact | str] | None = None,
        lineage_mode: LineageMode | str = "new_key",
        basis_labels: str | Sequence[str] | None = None,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Register an externally produced result as a durable TeAL artifact.

        ``external`` may be a DataFrame, a supported tabular path/dataset, one
        writer-shaped payload, or an iterable/generator of payload batches.
        External computations own their own execution/checkpointing; TeAL owns
        registration, lineage/provenance, serialization, and sealing.
        """
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _register_external(
            self,
            external,
            artifact_type=artifact_type,
            primary_key=primary_key,
            data_fields=data_fields,
            metadata_fields=metadata_fields,
            format=format,
            batch_size=batch_size,
            duckdb_options=duckdb_options,
            sources=sources,
            lineage_mode=lineage_mode,
            basis_labels=basis_labels,
            output_label=output_label,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def from_keyed_frame(
        self,
        source: BaseArtifact | str,
        frame: Any,
        *,
        data_fields: str | Sequence[str],
        require_complete: bool = False,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Import same-key measurement data aligned by stable TeAL primary keys."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _from_keyed_frame(
            self,
            source,
            frame,
            data_fields=data_fields,
            require_complete=require_complete,
            output_label=output_label,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def attach_metadata(
        self,
        source: BaseArtifact | str,
        frame: Any,
        *,
        metadata_fields: str | Sequence[str],
        require_complete: bool = True,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Attach descriptive metadata by stable key without copying source data."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _attach_metadata(
            self,
            source,
            frame,
            metadata_fields=metadata_fields,
            require_complete=require_complete,
            output_label=output_label,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def can_transform_texts_like(
        self,
        artifact: BaseArtifact | str,
        *,
        query: bool = False,
    ) -> bool:
        """Return whether a frozen matrix lineage can replay new text in memory."""
        return _can_transform_texts_like(self, artifact, query=query)

    def transform_texts_like(
        self,
        artifact: BaseArtifact | str,
        texts: Sequence[str],
        *,
        query: bool = False,
    ) -> Any:
        """Replay an artifact's frozen fitted representation pipeline on new text."""
        return _transform_texts_like(self, artifact, texts, query=query)

    def binary_code(
        self,
        source: BaseArtifact | str,
        *,
        text_source: BaseArtifact | str,
        text_field: str,
        instructions: str,
        context_before: int = 2,
        context_after: int = 2,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Interactively code every source observation as binary 0/1."""
        _label = DEFAULT_OUTPUT_LABEL
        plan = prepare_alias_plan(self, (_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[_label]
        result = _binary_code(
            self,
            source,
            text_source=text_source,
            text_field=text_field,
            instructions=instructions,
            context_before=context_before,
            context_after=context_after,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {_label: result})
        return result

    def merge(
        self,
        sources: Sequence[BaseArtifact | str],
        *,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """N-way merge compatible disjoint table artifacts into a flat keys-only artifact."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _merge(
            self,
            sources,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result

    def join(
        self,
        basis: BaseArtifact | str,
        *others: BaseArtifact | str,
        output_label: str = DEFAULT_OUTPUT_LABEL,
        batch_size: int = 10_000,
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Lazily compose same-key relational fields onto the basis row universe."""
        plan = prepare_alias_plan(self, (output_label,), alias, overwrite=overwrite)
        if plan is not None and plan.reuse:
            return reused_outputs(self, plan)[output_label]
        result = _join(
            self,
            basis,
            *others,
            output_label=output_label,
            batch_size=batch_size,
            memo=memo,
        )
        finalize_alias_plan(self, plan, {output_label: result})
        return result
