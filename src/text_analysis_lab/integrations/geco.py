"""TeAL <-> GeCo linked-workspace integration.

GeCo owns mutable exploratory/coding state while TeAL owns durable analytic
artifacts. Exploratory workspaces may use TeAL-backed numerical resources; compact
focus-coder workspaces are geometry-free local GeCo snapshots of a fixed TeAL key
universe for independent human audit coding.
"""

from __future__ import annotations

import inspect
import json
import re
import shutil
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import ArtifactError, ArtifactNotFoundError
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.core.utils import utc_now_iso
from text_analysis_lab.translators.geco_predictor import GeCoPredictor

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


_LINK_SCHEMA_VERSION = 2
_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MATRIX_TYPES = {ArtifactType.SPARSE_MATRIX, ArtifactType.DENSE_MATRIX}


class GeCoIntegrationError(RuntimeError):
    """Raised when a linked GeCo workspace cannot be created or reopened safely."""


@dataclass(frozen=True, slots=True)
class GeCoPredictorRef:
    """Stable bridge reference to one exportable GeCo predictor."""

    kind: str
    id: int
    code_id: int | None
    name: str


class TeALGeCoProvider:
    """Resolve opaque GeCo external refs against one TeAL project.

    The provider implements the complete external numerical protocol. GeCo owns
    per-geometry capability declarations and decides whether query or arbitrary-text
    transformation may be requested for a given resource. Every request opens a
    callback-local TeAL project session so no notebook-owned SQLite or DuckDB
    connection crosses a server-thread boundary.
    """

    def __init__(self, project: "Project") -> None:
        self.project = project
        project_path = getattr(project, "path", None)
        self._project_path = Path(project_path) if project_path is not None else None

    @contextmanager
    def _project_session(self):
        """Open connection-local TeAL access in the provider caller's thread.

        Real TeAL projects are reopened from their durable project path for every
        provider request. This keeps SQLite and DuckDB connection creation, use,
        and closure in the same thread as the GeCo callback. The fallback exists
        only for lightweight test doubles that do not expose a project path.
        """
        if self._project_path is None:
            yield self.project
            return

        from text_analysis_lab.core.project import Project

        project = Project.open(self._project_path)
        try:
            yield project
        finally:
            project.close()

    def geometry_matrix(
        self,
        external_ref: Any,
        ordered_user_keys: list[dict[str, Any]],
    ) -> Any:
        """Return a TeAL matrix subset/reordered to GeCo's requested user keys."""
        with self._project_session() as project:
            artifact = self._matrix_artifact(project, external_ref, purpose="geometry")
            positions = self._positions_for_keys(project, artifact, ordered_user_keys)
            values = artifact.get_matrix(positions=positions)
            # Detach the result from any project-owned storage/query lifecycle
            # before the call-local project is closed.
            return values.copy() if hasattr(values, "copy") else values

    def view_coordinates(
        self,
        external_ref: Any,
        ordered_user_keys: list[dict[str, Any]],
    ) -> np.ndarray:
        """Return a dense ``(n, 2)`` TeAL view aligned to GeCo's requested keys."""
        with self._project_session() as project:
            artifact = self._matrix_artifact(project, external_ref, purpose="view")
            positions = self._positions_for_keys(project, artifact, ordered_user_keys)
            values = artifact.get_matrix(positions=positions)
            if sparse.issparse(values):
                values = values.toarray()
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[1] != 2:
                raise GeCoIntegrationError(
                    "External GeCo view must resolve to a two-dimensional TeAL matrix "
                    f"with exactly 2 columns; got shape={array.shape!r} for artifact "
                    f"{artifact.artifact_id!r}."
                )
            try:
                coordinates = np.ascontiguousarray(array, dtype="float64")
            except (TypeError, ValueError) as exc:
                raise GeCoIntegrationError(
                    f"External GeCo view {artifact.artifact_id!r} could not be converted "
                    "to numeric coordinates."
                ) from exc
            if not np.isfinite(coordinates).all():
                raise GeCoIntegrationError(
                    f"External GeCo view {artifact.artifact_id!r} contains non-finite "
                    "coordinates; Plotly cannot display this view reliably."
                )
            return coordinates

    def _matrix_artifact(
        self,
        project: "Project",
        external_ref: Any,
        *,
        purpose: str,
    ) -> BaseArtifact:
        artifact_id = _artifact_id_from_external_ref(external_ref)
        try:
            artifact = project.get_artifact(artifact_id)
        except ArtifactNotFoundError as exc:
            raise GeCoIntegrationError(
                f"External GeCo {purpose} ref points to unavailable TeAL artifact "
                f"{artifact_id!r}."
            ) from exc
        if artifact.artifact_type not in _MATRIX_TYPES:
            raise GeCoIntegrationError(
                f"External GeCo {purpose} requires a dense_matrix or sparse_matrix "
                f"TeAL artifact; {artifact.artifact_id!r} is "
                f"{artifact.artifact_type.value!r}."
            )
        return artifact

    def _positions_for_keys(
        self,
        project: "Project",
        artifact: BaseArtifact,
        ordered_user_keys: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        keys = _normalize_ordered_keys(ordered_user_keys, artifact.primary_key)
        try:
            return project.query.positions_by_keys(artifact, keys)
        except KeyError as exc:
            raise GeCoIntegrationError(
                f"TeAL artifact {artifact.artifact_id!r} does not cover every document "
                "key requested by the linked GeCo workspace. Geometry/view artifacts "
                "may contain extra rows, but every GeCo document key must be present."
            ) from exc


    def transform_texts(self, external_ref: Any, texts: list[str]) -> Any:
        """Replay one registered geometry's frozen TeAL lineage on new text."""
        with self._project_session() as project:
            artifact = self._matrix_artifact(project, external_ref, purpose="geometry")
            try:
                values = project.transform_texts_like(artifact, texts, query=False)
            except Exception as exc:
                raise GeCoIntegrationError(
                    f"Could not transform new text through TeAL geometry "
                    f"{artifact.artifact_id!r}."
                ) from exc
            return values.copy() if hasattr(values, "copy") else values

    def transform_query(self, external_ref: Any, query: str) -> Any:
        """Replay one registered geometry's frozen TeAL query transformation."""
        with self._project_session() as project:
            artifact = self._matrix_artifact(project, external_ref, purpose="geometry")
            try:
                values = project.transform_texts_like(artifact, [str(query)], query=True)
            except Exception as exc:
                raise GeCoIntegrationError(
                    f"Could not transform semantic query through TeAL geometry "
                    f"{artifact.artifact_id!r}."
                ) from exc
            return values.copy() if hasattr(values, "copy") else values


class LinkedGeCoWorkspace:
    """TeAL-side handle for one mutable externally backed GeCo project."""

    def __init__(
        self,
        manager: "GeCoManager",
        manifest: Mapping[str, Any],
        coder: Any,
        provider: TeALGeCoProvider | None,
    ) -> None:
        self._manager = manager
        self._project = manager.project
        self._manifest = dict(manifest)
        self._coder = coder
        self._provider = provider
        self._closed = False

    @property
    def name(self) -> str:
        return str(self._manifest["name"])

    @property
    def path(self) -> Path:
        return self._manager._workspace_path_from_manifest(self._manifest)

    @property
    def documents(self) -> BaseArtifact:
        return self._project.get_artifact(str(self._manifest["documents_artifact_id"]))

    @property
    def coder(self) -> Any:
        """Expose the ordinary GeCo project for interactive exploratory work."""
        return self._coder

    @property
    def external_provider(self) -> TeALGeCoProvider | None:
        """Return the TeAL provider for externally backed workspaces, otherwise None."""
        return self._provider

    @property
    def mode(self) -> str:
        """Return the linked workspace mode (``explore`` or ``focus_coder``)."""
        return str(self._manifest.get("mode", "explore"))

    @property
    def is_focus_coder(self) -> bool:
        return self.mode == "focus_coder"

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate directly to GeCo's public ``launch(...)`` method."""
        launch = getattr(self._coder, "launch", None)
        if not callable(launch):
            raise GeCoIntegrationError(
                "Installed GeCo object does not expose launch(...)."
            )
        return launch(*args, **kwargs)

    def launch_focus(self, *args: Any, **kwargs: Any) -> Any:
        """Launch the linked finite audit task in GeCo Focus Coding.

        GeCo 0.8.15+ exposes a dedicated finite-task API. New TeAL focus
        workspaces configure that task at creation time and persist its session ID
        in the link manifest. Older linked workspaces without that field can still
        be bootstrapped by GeCo's public launcher from the persisted code list and
        the local imported document order.
        """
        if not self.is_focus_coder:
            raise GeCoIntegrationError(
                "launch_focus() is available only for a focus-coder workspace."
            )

        focus_launch = getattr(self._coder, "launch_focus_coder", None)
        if not callable(focus_launch):
            # Compatibility with the short-lived integrated-UI GeCo seam used by
            # 0.8.14. The workspace remains geometry-free and TeAL's export check
            # still enforces complete binary judgments.
            launch = getattr(self._coder, "launch", None)
            if not callable(launch):
                raise GeCoIntegrationError(
                    "Installed GeCo does not expose a supported Focus Coding launcher."
                )
            return launch(*args, **kwargs)

        options = dict(kwargs)
        focus_session_id = self._manifest.get("focus_session_id")
        if focus_session_id is not None:
            options.setdefault("session_id", int(focus_session_id))
        else:
            # GeCo 0.8.15 can bootstrap a finite task for older TeAL-created
            # geometry-free workspaces. Use public code IDs, not an obsolete
            # ``code_ids`` launcher keyword.
            options.setdefault(
                "codes",
                [int(row["code_id"]) for row in self._manifest.get("codes", [])],
            )
            options.setdefault(
                "allow_unsure", bool(self._manifest.get("allow_unsure", False))
            )
        return focus_launch(*args, **options)

    def export_focus_labels(
        self,
        *,
        fields: Mapping[str, str] | None = None,
        output_label: str = "output",
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Freeze a complete multi-code focus task into one binary TeAL artifact.

        ``Unsure`` remains a human review state, not a numeric halfway label. Because
        GeCo's binary export omits Unsure and unlabeled cases, this method requires
        every focus document to have a Present/Absent judgment for every requested
        code before producing a DSL-ready artifact.
        """
        if not self.is_focus_coder:
            raise GeCoIntegrationError(
                "export_focus_labels() is available only for a focus-coder workspace."
            )
        code_rows = [dict(row) for row in self._manifest.get("codes", [])]
        if not code_rows:
            raise GeCoIntegrationError("Focus-coder manifest contains no codes.")
        requested = dict(fields or {})
        keys = list(self.documents.primary_key)
        base = self.documents.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            order_by="_position", include_position=True, form="table",
        ).sort_values("_position", kind="stable").drop(columns=["_position"])
        output = base.reset_index(drop=True)
        data_fields: list[str] = []
        for row in code_rows:
            code_name = str(row["name"])
            field = _validate_field_name(
                str(requested.get(code_name, code_name)), kind="focus output field"
            )
            if field in keys or field in data_fields:
                raise GeCoIntegrationError(
                    f"Focus output field {field!r} collides with a key or another code."
                )
            exported = self._coder.export_codes(int(row["code_id"]))
            if not isinstance(exported, pd.DataFrame):
                exported = pd.DataFrame(exported)
            required = [*keys, "label"]
            missing_columns = [column for column in required if column not in exported.columns]
            if missing_columns:
                raise GeCoIntegrationError(
                    f"GeCo export for focus code {code_name!r} omitted {missing_columns!r}."
                )
            exported = exported.loc[:, required].copy()
            if exported.duplicated(subset=keys).any():
                raise GeCoIntegrationError(
                    f"GeCo export for focus code {code_name!r} contains duplicate keys."
                )
            exported = exported.rename(columns={"label": field})
            output = output.merge(exported, on=keys, how="left", validate="one_to_one")
            missing_count = int(output[field].isna().sum())
            if missing_count:
                raise GeCoIntegrationError(
                    f"Focus code {code_name!r} is not binary-complete: {missing_count} of "
                    f"{len(output)} audit documents are unlabeled or Unsure. Resolve them "
                    "to Present/Absent before DSL export."
                )
            labels = pd.to_numeric(output[field], errors="coerce")
            if labels.isna().any() or (~labels.isin([0, 1])).any():
                raise GeCoIntegrationError(
                    f"Focus code {code_name!r} contains non-binary exported labels."
                )
            output[field] = labels.astype("int64")
            data_fields.append(field)
        provenance = (
            f"Imported complete multi-code human audit labels from linked GeCo focus "
            f"workspace {self.name!r}."
        )
        if memo:
            provenance = f"{provenance}\n\n{memo}"
        return self._project.from_keyed_frame(
            self.documents, output, data_fields=data_fields, require_complete=True,
            output_label=output_label, memo=provenance, alias=alias, overwrite=overwrite,
        )

    def resource_diagnostics(self) -> pd.DataFrame:
        """Exercise GeCo's authoritative registry through the complete external bridge."""
        rows: list[dict[str, Any]] = []
        for record in self._geometries():
            geometry_id = int(record["geometry_id"])
            values = self._coder.geometry_matrix(geometry_id)
            shape = tuple(int(value) for value in values.shape)
            rows.append(
                {
                    "resource_type": "geometry",
                    "name": str(record["name"]),
                    "artifact_id": _artifact_id_from_external_ref(record["external_ref"]),
                    "rows": shape[0],
                    "columns": shape[1],
                    "finite": _matrix_is_finite(values),
                    "supports_query": bool(record.get("supports_query", False)),
                    "supports_text_transform": bool(
                        record.get("supports_text_transform", False)
                    ),
                }
            )
        for record in self._views():
            view_id = int(record["view_id"])
            values = self._coder.view_coordinates(view_id)
            shape = tuple(int(value) for value in values.shape)
            rows.append(
                {
                    "resource_type": "view",
                    "name": str(record["name"]),
                    "artifact_id": _artifact_id_from_external_ref(record["external_ref"]),
                    "rows": shape[0],
                    "columns": shape[1],
                    "finite": bool(np.isfinite(values).all()),
                    "supports_query": None,
                    "supports_text_transform": None,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "resource_type",
                "name",
                "artifact_id",
                "rows",
                "columns",
                "finite",
                "supports_query",
                "supports_text_transform",
            ],
        )

    def export_codes(
        self,
        code: int | str,
        *,
        output_label: str = "output",
        memo: str | None = None,
        alias: str | None = None,
        overwrite: bool = False,
    ) -> BaseArtifact:
        """Freeze current atomic human Present/Absent judgments into TeAL."""
        frame = self._coder.export_codes(code)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        keys = self.documents.primary_key
        required = [*keys, "label"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise GeCoIntegrationError(
                f"GeCo export_codes(...) omitted required column(s) {missing!r}."
            )
        exported = frame.loc[:, required].copy()
        if exported.empty:
            raise GeCoIntegrationError(
                f"GeCo code {code!r} has no current atomic Present/Absent judgments to export."
            )
        labels = pd.to_numeric(exported["label"], errors="coerce")
        invalid = labels.isna() | ~labels.isin([0, 1])
        if invalid.any():
            raise GeCoIntegrationError(
                "GeCo export_codes(...) must contain only binary label values 0/1."
            )
        exported["label"] = labels.astype("int64")
        provenance = (
            f"Imported human binary codes from linked GeCo workspace {self.name!r}; "
            f"code={code!r}."
        )
        if memo:
            provenance = f"{provenance}\n\n{memo}"
        return self._project.from_keyed_frame(
            self.documents,
            exported,
            data_fields=["label"],
            require_complete=False,
            output_label=output_label,
            memo=provenance,
            alias=alias,
            overwrite=overwrite,
        )

    def predictors(self) -> list[GeCoPredictorRef]:
        """Return stable references for predictors this linked workspace can export.

        Current GeCo builds may expose a unified ``predictors()`` registry. During
        the transition, older classifier-only GeCo builds are adapted from
        ``classifier_specs()`` so existing linked workspaces remain usable.
        """
        method = getattr(self._coder, "predictors", None)
        if callable(method):
            return [_normalize_predictor_ref(record) for record in method()]

        classifier_specs = getattr(self._coder, "classifier_specs", None)
        if not callable(classifier_specs):
            raise GeCoIntegrationError(
                "Installed GeCo exposes neither predictors() nor classifier_specs()."
            )
        refs: list[GeCoPredictorRef] = []
        for record in classifier_specs():
            row = dict(record)
            refs.append(
                GeCoPredictorRef(
                    kind="classifier",
                    id=int(row["classifier_spec_id"]),
                    code_id=int(row["code_id"]),
                    name=str(row["name"]),
                )
            )
        return refs

    def export_predictor(
        self,
        ref: GeCoPredictorRef,
        *,
        allow_stale: bool = False,
    ) -> GeCoPredictor:
        """Freeze one retained GeCo predictor as a serializable TeAL translator.

        Export never retrains. A current GeCo predictor export is preferred. Older
        classifier-only GeCo builds are supported as a one-member compatibility
        path until they implement the neutral frozen-predictor export contract.
        """
        resolved = _require_predictor_ref(ref)
        method = getattr(self._coder, "export_predictor", None)
        if callable(method):
            # GeCo 0.8.14+ requires the exact native GeCoPredictorRef returned by
            # coder.predictors(); it intentionally does not accept decomposed
            # kind/id keyword arguments. Resolve a fresh native reference so the
            # bridge also catches renamed/deleted predictors before export.
            native_ref = _resolve_native_predictor_ref(self._coder, resolved)
            frozen = method(native_ref, allow_stale=bool(allow_stale))
            return _geco_predictor_from_export(frozen)

        if resolved.kind != "classifier":
            raise GeCoIntegrationError(
                "Installed GeCo can export only individual classifiers through its legacy "
                "export_classifier(...) API. Update GeCo to export committees/scalers."
            )
        return self._export_legacy_classifier_predictor(
            resolved, allow_stale=allow_stale
        )

    def export_classifier(
        self,
        name: str,
        *,
        probability_class: Any | None = 1,
    ) -> GeCoPredictor:
        """Compatibility wrapper for the pre-``export_predictor`` classifier API."""
        matches = [
            ref for ref in self.predictors()
            if ref.kind == "classifier" and ref.name == str(name)
        ]
        if len(matches) != 1:
            raise GeCoIntegrationError(
                f"Expected exactly one active GeCo classifier named {name!r}; "
                f"found {len(matches)}."
            )
        predictor = self.export_predictor(matches[0])
        if probability_class is not None and predictor.member_specs:
            # Preserve the old explicit override only for this compatibility wrapper.
            predictor.member_specs[0]["positive_class"] = probability_class
        return predictor

    def _export_legacy_classifier_predictor(
        self,
        ref: GeCoPredictorRef,
        *,
        allow_stale: bool,
    ) -> GeCoPredictor:
        classifier_specs = getattr(self._coder, "classifier_specs", None)
        export_classifier = getattr(self._coder, "export_classifier", None)
        if not callable(classifier_specs) or not callable(export_classifier):
            raise GeCoIntegrationError(
                "Installed GeCo does not expose the legacy classifier export contract."
            )
        rows = [
            dict(row) for row in classifier_specs()
            if int(row.get("classifier_spec_id", -1)) == int(ref.id)
        ]
        if len(rows) != 1:
            raise GeCoIntegrationError(
                f"Could not resolve GeCo classifier id {ref.id!r} for export."
            )
        spec = rows[0]
        status = _legacy_classifier_status(self._coder, spec)
        if status == "stale" and not allow_stale:
            raise GeCoIntegrationError(
                f"GeCo classifier {ref.name!r} has a stale retained fit. Retrain it or "
                "pass allow_stale=True to export the retained fit explicitly."
            )
        estimator = export_classifier(ref.name)
        provenance = {
            "geco_predictor_kind": "classifier",
            "geco_predictor_id": int(ref.id),
            "geco_code_id": ref.code_id,
            "geco_name": ref.name,
            "geco_geometry_id": spec.get("geometry_id"),
            "geco_geometry_name": spec.get("geometry_name"),
            "created_with_geco_version": _installed_geco_version(),
            "legacy_classifier_export": True,
        }
        return GeCoPredictor.single_classifier(
            estimator,
            geometry_id=spec.get("geometry_id"),
            geometry_name=(
                None if spec.get("geometry_name") is None else str(spec.get("geometry_name"))
            ),
            positive_class=1,
            threshold=0.5,
            provenance=provenance,
            stale_at_export=(status == "stale"),
        )

    def add_geometry(
        self,
        name: str,
        artifact: BaseArtifact | str,
        *,
        supports_query: bool | None = None,
    ) -> int:
        """Register another TeAL-backed geometry in GeCo's authoritative registry."""
        geometry_name = _validate_resource_name(name, kind="geometry")
        geometry = self._manager._validate_numerical_resource(
            artifact, self.documents, purpose=f"geometry {geometry_name!r}"
        )
        if self._provider is None:
            raise GeCoIntegrationError("This GeCo workspace has no external TeAL provider.")
        keys = self._manager._document_keys(self.documents)
        self._provider.geometry_matrix(_external_ref(geometry), keys)
        text_replay = self._project.can_transform_texts_like(geometry, query=False)
        query_replay = self._project.can_transform_texts_like(geometry, query=True)
        resolved_supports_query = (
            query_replay if supports_query is None else bool(supports_query)
        )
        if resolved_supports_query and not query_replay:
            raise GeCoIntegrationError(
                f"TeAL geometry {geometry.artifact_id!r} cannot replay semantic queries "
                "through its frozen operator lineage."
            )
        return int(
            self._coder.register_external_geometry(
                name=geometry_name,
                external_ref=_external_ref(geometry),
                supports_query=resolved_supports_query,
                supports_text_transform=text_replay,
            )
        )

    def add_projection(
        self,
        name: str,
        artifact: BaseArtifact | str,
        *,
        geometry: str | None = None,
    ) -> int:
        """Register another TeAL-backed 2D view in GeCo's authoritative registry."""
        view_name = _validate_resource_name(name, kind="projection")
        geometry_record = self._resolve_geometry_record(geometry)
        view = self._manager._validate_numerical_resource(
            artifact, self.documents, purpose=f"projection {view_name!r}"
        )
        if self._provider is None:
            raise GeCoIntegrationError("This GeCo workspace has no external TeAL provider.")
        keys = self._manager._document_keys(self.documents)
        self._provider.view_coordinates(_external_ref(view), keys)
        return int(
            self._coder.register_external_view(
                geometry_id=int(geometry_record["geometry_id"]),
                name=view_name,
                external_ref=_external_ref(view),
            )
        )

    def _geometries(self) -> list[dict[str, Any]]:
        method = getattr(self._coder, "geometries", None)
        if not callable(method):
            raise GeCoIntegrationError(
                "Installed GeCo does not expose the required geometries() registry API."
            )
        return [dict(record) for record in method(public_only=True)]

    def _views(self, geometry_id: int | None = None) -> list[dict[str, Any]]:
        method = getattr(self._coder, "views", None)
        if not callable(method):
            raise GeCoIntegrationError(
                "Installed GeCo does not expose the required views() registry API."
            )
        return [dict(record) for record in method(geometry_id)]

    def _resolve_geometry_record(self, geometry: str | None) -> dict[str, Any]:
        records = self._geometries()
        if geometry is None:
            if len(records) != 1:
                raise GeCoIntegrationError(
                    "geometry= is required when a linked GeCo workspace has anything "
                    "other than exactly one registered geometry."
                )
            return records[0]
        matches = [record for record in records if str(record.get("name")) == geometry]
        if not matches:
            available = sorted(str(record.get("name")) for record in records)
            raise GeCoIntegrationError(
                f"Unknown linked GeCo geometry {geometry!r}; available={available}."
            )
        return matches[0]

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._coder, "close", None)
        if callable(close):
            close()
        self._closed = True
        self._manager._forget_handle(self.name, self)

    def __enter__(self) -> "LinkedGeCoWorkspace":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class GeCoManager:
    """Create, reopen, list, and manage TeAL-linked GeCo workspaces."""

    def __init__(self, project: "Project") -> None:
        self.project = project
        self.root = project.storage.teal_dir / "geco"
        self.links_dir = self.root / "links"
        self.workspaces_dir = self.root / "workspaces"
        self.links_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, LinkedGeCoWorkspace] = {}

    def create(
        self,
        name: str,
        *,
        documents: BaseArtifact | str,
        text_field: str,
        geometry: BaseArtifact | str,
        projections: Mapping[str, BaseArtifact | str] | None = None,
        metadata_fields: Sequence[str] | None = None,
        geometry_name: str | None = None,
        supports_query: bool | None = None,
        overwrite: bool = False,
    ) -> LinkedGeCoWorkspace:
        """Create a fixed-universe GeCo workspace linked to TeAL numerical artifacts."""
        workspace_name = _validate_workspace_name(name)
        text_field = _validate_field_name(text_field, kind="text_field")
        metadata = _normalize_metadata_fields(metadata_fields)
        documents_artifact = self.project.get_artifact(documents)
        if not documents_artifact.primary_key:
            raise GeCoIntegrationError("GeCo documents artifact must have a primary key.")
        if not documents_artifact.n_rows:
            raise GeCoIntegrationError("GeCo documents artifact must contain at least one row.")

        geometry_artifact = self._validate_numerical_resource(
            geometry, documents_artifact, purpose="geometry"
        )
        external_geometry_name = _validate_resource_name(
            geometry_name or _default_resource_name(geometry_artifact), kind="geometry"
        )
        projection_artifacts: dict[str, BaseArtifact] = {}
        for raw_name, ref in dict(projections or {}).items():
            view_name = _validate_resource_name(str(raw_name), kind="projection")
            if view_name in projection_artifacts:
                raise GeCoIntegrationError(f"Duplicate projection name {view_name!r}.")
            projection_artifacts[view_name] = self._validate_numerical_resource(
                ref, documents_artifact, purpose=f"projection {view_name!r}"
            )

        manifest_path = self._manifest_path(workspace_name)
        workspace_path = self._workspace_path(workspace_name)
        if manifest_path.exists() or workspace_path.exists():
            if not overwrite:
                raise GeCoIntegrationError(
                    f"Linked GeCo workspace {workspace_name!r} already exists. "
                    "Use project.geco.open(name) or create(..., overwrite=True)."
                )
            existing = self._handles.pop(workspace_name, None)
            if existing is not None:
                existing.close()
            manifest_path.unlink(missing_ok=True)
            if workspace_path.exists():
                shutil.rmtree(workspace_path)

        document_frame = self._document_frame(
            documents_artifact,
            text_field=text_field,
            metadata_fields=metadata,
        )
        ordered_keys = _key_records(document_frame, documents_artifact.primary_key)
        text_replay = self.project.can_transform_texts_like(geometry_artifact, query=False)
        query_replay = self.project.can_transform_texts_like(geometry_artifact, query=True)
        provider = TeALGeCoProvider(self.project)
        resolved_supports_query = query_replay if supports_query is None else bool(supports_query)
        if resolved_supports_query and not query_replay:
            raise GeCoIntegrationError(
                f"TeAL geometry {geometry_artifact.artifact_id!r} cannot replay semantic "
                "queries through its frozen operator lineage."
            )

        # Preflight the exact stable-key order before creating any mutable GeCo state.
        matrix = provider.geometry_matrix(_external_ref(geometry_artifact), ordered_keys)
        _validate_matrix_rows(
            matrix, expected_rows=len(document_frame), purpose="geometry"
        )
        for view_name, view in projection_artifacts.items():
            coordinates = provider.view_coordinates(_external_ref(view), ordered_keys)
            _validate_matrix_rows(
                coordinates,
                expected_rows=len(document_frame),
                purpose=f"projection {view_name!r}",
            )

        GeometricCoder = _load_geometric_coder()
        coder = None
        try:
            coder = GeometricCoder.create_external(
                project_dir=workspace_path,
                data=document_frame,
                keys=list(documents_artifact.primary_key),
                text=text_field,
                metadata=list(metadata),
                external_provider=provider,
                overwrite=False,
            )
            geometry_id = int(
                coder.register_external_geometry(
                    name=external_geometry_name,
                    external_ref=_external_ref(geometry_artifact),
                    supports_query=resolved_supports_query,
                    supports_text_transform=text_replay,
                )
            )
            for view_name, view in projection_artifacts.items():
                coder.register_external_view(
                    geometry_id=geometry_id,
                    name=view_name,
                    external_ref=_external_ref(view),
                )

            manifest: dict[str, Any] = {
                "schema_version": _LINK_SCHEMA_VERSION,
                "name": workspace_name,
                "mode": "explore",
                "workspace_path": str(workspace_path.relative_to(self.root)),
                "documents_artifact_id": documents_artifact.artifact_id,
                "text_field": text_field,
                "metadata_fields": list(metadata),
                "created_with_geco_version": _installed_geco_version(),
                "created_at": utc_now_iso(),
            }
            self._write_manifest(manifest)
            handle = LinkedGeCoWorkspace(self, manifest, coder, provider)
            self._handles[workspace_name] = handle
            self.project.storage.touch_manifest()
            return handle
        except Exception:
            if coder is not None:
                close = getattr(coder, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            manifest_path.unlink(missing_ok=True)
            if workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)
            raise

    def create_focus(
        self,
        name: str,
        *,
        documents: BaseArtifact | str,
        text_field: str,
        codes: Mapping[str, str] | Sequence[Mapping[str, str]],
        text_source: BaseArtifact | str | None = None,
        metadata_fields: Sequence[str] | None = None,
        allow_unsure: bool = False,
        overwrite: bool = False,
    ) -> LinkedGeCoWorkspace:
        """Create a geometry-free local GeCo workspace for multi-code audit coding.

        ``documents`` defines the exact fixed audit universe and canonical order.
        ``text_source`` may point to a larger same-key artifact (for example the full
        eligible corpus) when the audit artifact itself contains only sampled keys.
        """
        workspace_name = _validate_workspace_name(name)
        text_field = _validate_field_name(text_field, kind="text_field")
        metadata = _normalize_metadata_fields(metadata_fields)
        documents_artifact = self.project.get_artifact(documents)
        source_artifact = (
            documents_artifact if text_source is None else self.project.get_artifact(text_source)
        )
        if not documents_artifact.primary_key or not documents_artifact.n_rows:
            raise GeCoIntegrationError(
                "Focus coder documents must contain at least one row and a primary key."
            )
        if source_artifact.primary_key != documents_artifact.primary_key:
            raise GeCoIntegrationError(
                "Focus coder text_source must use the same primary key as documents."
            )
        code_specs = _normalize_focus_codes(codes)
        manifest_path = self._manifest_path(workspace_name)
        workspace_path = self._workspace_path(workspace_name)
        if manifest_path.exists() or workspace_path.exists():
            if not overwrite:
                raise GeCoIntegrationError(
                    f"Linked GeCo workspace {workspace_name!r} already exists. "
                    "Use project.geco.open(name) or create_focus(..., overwrite=True)."
                )
            existing = self._handles.pop(workspace_name, None)
            if existing is not None:
                existing.close()
            manifest_path.unlink(missing_ok=True)
            if workspace_path.exists():
                shutil.rmtree(workspace_path)

        document_frame = self._focus_document_frame(
            documents_artifact, source_artifact, text_field=text_field, metadata_fields=metadata
        )
        GeometricCoder = _load_geometric_coder()
        coder = None
        try:
            coder = GeometricCoder.create(
                project_dir=workspace_path,
                data=document_frame,
                keys=list(documents_artifact.primary_key),
                text=text_field,
                metadata=list(metadata),
                geometries=None,
                overwrite=False,
            )
            persisted_codes: list[dict[str, Any]] = []
            for spec in code_specs:
                code_id = int(coder.create_code(spec["name"], spec["description"]))
                persisted_codes.append({**spec, "code_id": code_id})

            # GeCo 0.8.15+ has a first-class finite Focus Coding protocol. Configure
            # the exact ordered audit universe once at creation time. Stable TeAL
            # primary keys are the cross-system identity; GeCo-local unit IDs are
            # never persisted as TeAL identity.
            focus_session_id: int | None = None
            configure_focus = getattr(coder, "configure_focus", None)
            if callable(configure_focus):
                ordered_user_keys = (
                    document_frame.loc[:, list(documents_artifact.primary_key)]
                    .to_dict(orient="records")
                )
                focus_session_id = int(
                    configure_focus(
                        codes=[int(row["code_id"]) for row in persisted_codes],
                        allow_unsure=bool(allow_unsure),
                        user_keys=ordered_user_keys,
                        title=workspace_name,
                    )
                )
            else:
                # Compatibility with GeCo 0.8.14, which temporarily integrated
                # audit coding into the ordinary UI rather than exposing
                # configure_focus(...). Prime that UI as before.
                sessions = getattr(coder, "sessions", None)
                patch_session_state = getattr(coder, "patch_session_state", None)
                if callable(sessions) and callable(patch_session_state):
                    session_rows = list(sessions())
                    if session_rows:
                        patch_session_state(
                            int(session_rows[0]["session_id"]),
                            {
                                "workspace": "explore",
                                "explore_code_palette": [
                                    int(row["code_id"]) for row in persisted_codes
                                ],
                                "navigation_new_only": True,
                            },
                        )
            manifest: dict[str, Any] = {
                "schema_version": _LINK_SCHEMA_VERSION,
                "name": workspace_name,
                "mode": "focus_coder",
                "workspace_path": str(workspace_path.relative_to(self.root)),
                "documents_artifact_id": documents_artifact.artifact_id,
                "text_source_artifact_id": source_artifact.artifact_id,
                "text_field": text_field,
                "metadata_fields": list(metadata),
                "codes": persisted_codes,
                "allow_unsure": bool(allow_unsure),
                "focus_session_id": focus_session_id,
                "created_with_geco_version": _installed_geco_version(),
                "created_at": utc_now_iso(),
            }
            self._write_manifest(manifest)
            handle = LinkedGeCoWorkspace(self, manifest, coder, None)
            self._handles[workspace_name] = handle
            self.project.storage.touch_manifest()
            return handle
        except Exception:
            if coder is not None:
                close = getattr(coder, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            manifest_path.unlink(missing_ok=True)
            if workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)
            raise

    def open(self, name: str) -> LinkedGeCoWorkspace:
        """Reopen a linked GeCo project with a newly constructed TeAL provider."""
        workspace_name = _validate_workspace_name(name)
        cached = self._handles.get(workspace_name)
        if cached is not None and not cached._closed:
            return cached
        manifest = self._read_manifest(workspace_name)
        self.project.get_artifact(str(manifest["documents_artifact_id"]))
        mode = str(manifest.get("mode", "explore"))
        provider = TeALGeCoProvider(self.project) if mode == "explore" else None
        GeometricCoder = _load_geometric_coder()
        if provider is None:
            coder = GeometricCoder.open(self._workspace_path_from_manifest(manifest))
        else:
            coder = GeometricCoder.open(
                self._workspace_path_from_manifest(manifest), external_provider=provider
            )
        handle = LinkedGeCoWorkspace(self, manifest, coder, provider)
        self._handles[workspace_name] = handle
        return handle

    def list(self) -> list[dict[str, Any]]:
        """List persisted linked-workspace manifests without importing GeCo."""
        rows: list[dict[str, Any]] = []
        for path in sorted(self.links_dir.glob("*.json")):
            try:
                manifest = _read_json(path)
            except Exception as exc:
                raise GeCoIntegrationError(
                    f"Could not read linked GeCo manifest {path}."
                ) from exc
            name = _validate_workspace_name(str(manifest.get("name", path.stem)))
            manifest = self._read_manifest(name)
            rows.append(
                {
                    "name": manifest["name"],
                    "mode": manifest.get("mode", "explore"),
                    "documents_artifact_id": manifest["documents_artifact_id"],
                    "codes": [dict(row) for row in manifest.get("codes", [])],
                    "allow_unsure": manifest.get("allow_unsure"),
                    "text_field": manifest.get("text_field"),
                    "metadata_fields": list(manifest.get("metadata_fields", [])),
                    "workspace_path": str(self._workspace_path_from_manifest(manifest)),
                    "created_with_geco_version": manifest.get(
                        "created_with_geco_version"
                    ),
                    "created_at": manifest.get("created_at"),
                }
            )
        return rows

    def close(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def _forget_handle(self, name: str, handle: LinkedGeCoWorkspace) -> None:
        if self._handles.get(name) is handle:
            self._handles.pop(name, None)

    def _focus_document_frame(
        self,
        documents: BaseArtifact,
        source: BaseArtifact,
        *,
        text_field: str,
        metadata_fields: Sequence[str],
    ) -> pd.DataFrame:
        """Resolve source text/metadata onto a fixed target key universe in target order."""
        key_columns = list(documents.primary_key)
        overlap = sorted(set(metadata_fields).intersection([*key_columns, text_field]))
        if text_field in key_columns or overlap:
            raise GeCoIntegrationError(
                "Focus coder text/metadata fields may not collide with primary-key columns."
            )
        target = documents.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            order_by="_position", include_position=True, form="table",
        ).sort_values("_position", kind="stable")
        target = target.loc[:, [*key_columns, "_position"]].rename(
            columns={"_position": "_focus_position"}
        )
        try:
            source_frame = source.query(
                key_columns=True, data_columns=[text_field],
                metadata_columns=list(metadata_fields) if metadata_fields else False,
                metadata_mode="full" if metadata_fields else "none",
                order_by="_position", include_position=False, form="table",
            )
        except Exception as exc:
            raise GeCoIntegrationError(
                f"Could not resolve focus text_field={text_field!r} and requested metadata "
                f"from source artifact {source.artifact_id!r}."
            ) from exc
        expected = [*key_columns, text_field, *metadata_fields]
        missing = [column for column in expected if column not in source_frame.columns]
        if missing:
            raise GeCoIntegrationError(
                f"Focus coder source is missing requested column(s) {missing!r}."
            )
        if source_frame.duplicated(subset=key_columns).any():
            raise GeCoIntegrationError("Focus coder text_source contains duplicate stable keys.")
        merged = target.merge(
            source_frame.loc[:, expected], on=key_columns, how="left", validate="one_to_one"
        ).sort_values("_focus_position", kind="stable")
        if merged[text_field].isna().any():
            missing_keys = int(merged[text_field].isna().sum())
            raise GeCoIntegrationError(
                f"Focus coder text_source does not cover {missing_keys} document key(s)."
            )
        merged[text_field] = merged[text_field].astype(str)
        return merged.drop(columns=["_focus_position"]).loc[:, expected].reset_index(drop=True)

    def _document_frame(
        self,
        documents: BaseArtifact,
        *,
        text_field: str,
        metadata_fields: Sequence[str],
    ) -> pd.DataFrame:
        key_columns = list(documents.primary_key)
        if text_field in key_columns:
            raise GeCoIntegrationError("text_field cannot also be a primary-key column.")
        overlap = sorted(set(metadata_fields).intersection([*key_columns, text_field]))
        if overlap:
            raise GeCoIntegrationError(
                f"metadata_fields overlap TeAL key/text columns: {overlap!r}."
            )
        try:
            frame = documents.query(
                key_columns=True,
                data_columns=[text_field],
                metadata_columns=list(metadata_fields) if metadata_fields else False,
                metadata_mode="full" if metadata_fields else "none",
                order_by="_position",
                include_position=True,
                form="table",
            )
        except Exception as exc:
            raise GeCoIntegrationError(
                f"Could not resolve text_field={text_field!r} and requested metadata from "
                f"documents artifact {documents.artifact_id!r}."
            ) from exc
        if "_position" not in frame.columns:
            raise GeCoIntegrationError("TeAL document query did not expose canonical _position.")
        frame = frame.sort_values("_position", kind="stable").drop(columns=["_position"])
        expected = [*key_columns, text_field, *metadata_fields]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise GeCoIntegrationError(
                f"Resolved GeCo document frame is missing requested column(s) {missing!r}; "
                "avoid key/data/metadata name collisions for linked workspaces."
            )
        frame = frame.loc[:, expected].reset_index(drop=True)
        if frame[text_field].isna().any():
            raise GeCoIntegrationError("GeCo text_field contains null values.")
        frame[text_field] = frame[text_field].astype(str)
        if frame.duplicated(subset=key_columns).any():
            raise GeCoIntegrationError("GeCo documents frame contains duplicate stable keys.")
        return frame

    def _document_keys(self, documents: BaseArtifact) -> list[dict[str, Any]]:
        frame = documents.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            order_by="_position",
            include_position=True,
            form="table",
        )
        frame = frame.sort_values("_position", kind="stable")
        return _key_records(frame, documents.primary_key)

    def _validate_numerical_resource(
        self,
        ref: BaseArtifact | str,
        documents: BaseArtifact,
        *,
        purpose: str,
    ) -> BaseArtifact:
        artifact = self.project.get_artifact(ref)
        if artifact.artifact_type not in _MATRIX_TYPES:
            raise GeCoIntegrationError(
                f"Linked GeCo {purpose} must be a dense_matrix or sparse_matrix TeAL "
                f"artifact; got {artifact.artifact_type.value!r}."
            )
        if artifact.primary_key != documents.primary_key:
            raise GeCoIntegrationError(
                f"Linked GeCo {purpose} primary key {artifact.primary_key!r} does not "
                f"match documents primary key {documents.primary_key!r}."
            )
        return artifact

    def _manifest_path(self, name: str) -> Path:
        return self.links_dir / f"{name}.json"

    def _workspace_path(self, name: str) -> Path:
        return self.workspaces_dir / f"{name}.geco"

    def _workspace_path_from_manifest(self, manifest: Mapping[str, Any]) -> Path:
        relative = Path(str(manifest["workspace_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise GeCoIntegrationError("Linked GeCo manifest has an unsafe workspace path.")
        path = (self.root / relative).resolve()
        root = self.root.resolve()
        if path != root and root not in path.parents:
            raise GeCoIntegrationError("Linked GeCo manifest workspace escapes .teal/geco.")
        return path

    def _read_manifest(self, name: str) -> dict[str, Any]:
        path = self._manifest_path(name)
        if not path.exists():
            raise GeCoIntegrationError(f"No linked GeCo workspace named {name!r} exists.")
        manifest = _read_json(path)
        if int(manifest.get("schema_version", -1)) != _LINK_SCHEMA_VERSION:
            raise GeCoIntegrationError(
                "Incompatible pre-1.0 linked GeCo descriptor schema "
                f"{manifest.get('schema_version')!r}; this TeAL build supports only "
                f"schema {_LINK_SCHEMA_VERSION}. Recreate the linked workspace."
            )
        if manifest.get("name") != name:
            raise GeCoIntegrationError(
                f"Linked GeCo manifest name mismatch: expected {name!r}, "
                f"got {manifest.get('name')!r}."
            )
        return manifest

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        name = _validate_workspace_name(str(manifest["name"]))
        path = self._manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        temp.replace(path)


def _normalize_predictor_ref(record: Any) -> GeCoPredictorRef:
    if isinstance(record, GeCoPredictorRef):
        return record
    if isinstance(record, Mapping):
        row = dict(record)
        kind = row.get("kind")
        predictor_id = row.get("id", row.get("predictor_id"))
        code_id = row.get("code_id")
        name = row.get("name")
    else:
        kind = getattr(record, "kind", None)
        predictor_id = getattr(record, "id", getattr(record, "predictor_id", None))
        code_id = getattr(record, "code_id", None)
        name = getattr(record, "name", None)
    if not isinstance(kind, str) or not kind.strip():
        raise GeCoIntegrationError("GeCo predictor registry row is missing a non-empty kind.")
    if predictor_id is None:
        raise GeCoIntegrationError("GeCo predictor registry row is missing an id.")
    if not isinstance(name, str) or not name.strip():
        raise GeCoIntegrationError("GeCo predictor registry row is missing a non-empty name.")
    return GeCoPredictorRef(
        kind=kind.strip(),
        id=int(predictor_id),
        code_id=(None if code_id is None else int(code_id)),
        name=name.strip(),
    )


def _require_predictor_ref(ref: GeCoPredictorRef) -> GeCoPredictorRef:
    if not isinstance(ref, GeCoPredictorRef):
        raise TypeError(
            "linked.export_predictor(...) requires a GeCoPredictorRef returned by "
            "linked.predictors(); bare predictor names are intentionally not accepted."
        )
    return ref


def _resolve_native_predictor_ref(coder: Any, ref: GeCoPredictorRef) -> Any:
    """Resolve a bridge ref back to GeCo's exact native predictor-reference object."""
    method = getattr(coder, "predictors", None)
    if not callable(method):
        raise GeCoIntegrationError(
            "Installed GeCo exposes export_predictor(...) but not predictors()."
        )
    matches: list[Any] = []
    for native_ref in method():
        normalized = _normalize_predictor_ref(native_ref)
        if (
            normalized.kind == ref.kind
            and int(normalized.id) == int(ref.id)
            and normalized.code_id == ref.code_id
            and normalized.name == ref.name
        ):
            matches.append(native_ref)
    if len(matches) != 1:
        raise GeCoIntegrationError(
            "The selected GeCo predictor reference is no longer uniquely current. "
            "Refresh linked.predictors() and select the predictor again."
        )
    return matches[0]


def _legacy_classifier_status(coder: Any, spec: Mapping[str, Any]) -> str | None:
    method = getattr(coder, "classifier_status", None)
    if not callable(method):
        return None
    try:
        result = method(
            code_id=int(spec["code_id"]),
            classifier_spec_ids=[int(spec["classifier_spec_id"])],
        )
    except Exception:
        return None
    if not isinstance(result, Mapping):
        return None
    rows = result.get("classifiers")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        return None
    row = rows[0]
    if not isinstance(row, Mapping):
        return None
    value = row.get("status")
    return None if value is None else str(value)


def _geco_predictor_from_export(frozen: Any) -> GeCoPredictor:
    """Convert GeCo's neutral frozen-export contract into TeAL state.

    Contract version 1 intentionally uses ordinary attributes/mapping fields so
    GeCo does not import TeAL. Fitted estimators remain live Python objects only
    during this handoff; TeAL serializes them as its own operator assets.
    """
    format_version = int(_export_value(frozen, "format_version", 1))
    if format_version != 1:
        raise GeCoIntegrationError(
            f"Unsupported GeCo frozen predictor format_version={format_version}; expected 1."
        )

    # GeCo 0.8.14 names the ordered unique source tuple ``sources``. Accept the
    # earlier proposal-era ``source_specs`` spelling only as a compatibility read.
    raw_sources = _export_value(
        frozen, "sources", _export_value(frozen, "source_specs", None)
    )
    raw_members = _export_value(frozen, "members", None)
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise GeCoIntegrationError("GeCo frozen predictor export must provide source_specs.")
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        raise GeCoIntegrationError("GeCo frozen predictor export must provide members.")

    source_specs = [_plain_mapping(item, name="source spec") for item in raw_sources]
    member_models: list[Any] = []
    member_specs: list[dict[str, Any]] = []
    for index, raw_member in enumerate(raw_members):
        member = _object_mapping(raw_member)
        estimator = member.pop("estimator", member.pop("model", None))
        if estimator is None:
            estimator = getattr(raw_member, "estimator", getattr(raw_member, "model", None))
        if estimator is None:
            raise GeCoIntegrationError(
                f"GeCo frozen predictor member {index} is missing its fitted estimator."
            )
        if "source_index" not in member:
            source_index = getattr(raw_member, "source_index", None)
            if source_index is None:
                raise GeCoIntegrationError(
                    f"GeCo frozen predictor member {index} is missing source_index."
                )
            member["source_index"] = source_index
        if "positive_class" not in member:
            member["positive_class"] = getattr(raw_member, "positive_class", 1)
        member_models.append(estimator)
        member_specs.append(member)

    aggregation_raw = _export_value(frozen, "aggregation", None)
    if aggregation_raw is None:
        aggregation = "single"
    elif isinstance(aggregation_raw, Mapping):
        aggregation = str(aggregation_raw.get("kind", "single"))
    else:
        aggregation = str(aggregation_raw)

    stacker = _export_value(frozen, "stacker", None)
    # GeCo 0.8.14 exposes one positive_class for the frozen predictor/stacker.
    stacker_positive_class = _export_value(
        frozen, "stacker_positive_class", _export_value(frozen, "positive_class", 1)
    )
    threshold = float(_export_value(frozen, "threshold", 0.5))
    output_fields = _export_value(frozen, "output_fields", ("prediction", "probability"))
    if not isinstance(output_fields, Sequence) or isinstance(output_fields, (str, bytes)):
        raise GeCoIntegrationError("GeCo frozen predictor output_fields must be a sequence.")
    output_semantics = {
        "kind": "binary_classification",
        "output_fields": [str(value) for value in output_fields],
    }
    provenance = _export_value(frozen, "provenance", {})
    stale_at_export = bool(_export_value(frozen, "stale_at_export", False))

    if not isinstance(output_semantics, Mapping):
        raise GeCoIntegrationError("GeCo frozen predictor output_semantics must be a mapping.")
    if not isinstance(provenance, Mapping):
        raise GeCoIntegrationError("GeCo frozen predictor provenance must be a mapping.")

    return GeCoPredictor(
        member_models,
        source_specs=source_specs,
        member_specs=member_specs,
        aggregation=aggregation,
        stacker=stacker,
        stacker_positive_class=stacker_positive_class,
        threshold=threshold,
        output_semantics=dict(output_semantics),
        provenance=dict(provenance),
        stale_at_export=stale_at_export,
        format_version=format_version,
    )


def _export_value(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _object_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    fields = getattr(obj, "__dict__", None)
    if isinstance(fields, Mapping):
        return dict(fields)
    names = (
        # FrozenPredictorMember fields (estimator intentionally excluded).
        "classifier_spec_id",
        "classifier_fit_id",
        "classifier_name",
        "fit_classifier_name",
        "source_index",
        "algorithm",
        "hyperparameters",
        "score_kind",
        "positive_class",
        # PredictorSourceSpec fields.
        "geometry_id",
        "geometry_name",
        "n_features",
        "storage_kind",
        "external_ref",
        # Proposal/legacy compatibility fields.
        "member_id",
        "name",
        "original_external_ref",
    )
    return {name: getattr(obj, name) for name in names if hasattr(obj, name)}


def _plain_mapping(obj: Any, *, name: str) -> dict[str, Any]:
    value = _object_mapping(obj)
    if not value:
        raise GeCoIntegrationError(f"GeCo frozen predictor {name} must be mapping-like.")
    return value

def _normalize_focus_codes(
    codes: Mapping[str, str] | Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Normalize ordered focus-code names/descriptions without inventing labels."""
    if isinstance(codes, Mapping):
        rows = [{"name": str(name), "description": str(description)} for name, description in codes.items()]
    else:
        rows = []
        for raw in codes:
            if not isinstance(raw, Mapping) or "name" not in raw:
                raise GeCoIntegrationError(
                    "Each focus code must provide at least a name and optional description."
                )
            rows.append({
                "name": str(raw["name"]),
                "description": str(raw.get("description", raw.get("definition", ""))),
            })
    if not rows:
        raise GeCoIntegrationError("Focus coder requires at least one code.")
    seen: set[str] = set()
    for row in rows:
        name = row["name"].strip()
        if not name:
            raise GeCoIntegrationError("Focus code names may not be empty.")
        if name in seen:
            raise GeCoIntegrationError(f"Duplicate focus code name {name!r}.")
        seen.add(name)
        row["name"] = name
        row["description"] = row["description"].strip()
    return rows


def _load_geometric_coder() -> Any:
    try:
        import geometric_coder
    except ImportError as exc:
        raise GeCoIntegrationError(
            "TeAL's linked-GeCo integration requires a current pre-1.0 GeCo build "
            "with the external-resource capability contract. Install the "
            "`geometric-coder` package from zacthinks/GeCo in the environment that "
            "runs the linked workspace."
        ) from exc
    coder = getattr(geometric_coder, "GeometricCoder", None)
    if coder is None:
        raise GeCoIntegrationError(
            "Installed geometric_coder package does not expose GeometricCoder."
        )
    register = getattr(coder, "register_external_geometry", None)
    if not callable(register):
        raise GeCoIntegrationError(
            "Installed GeCo does not expose register_external_geometry(...)."
        )
    try:
        parameters = inspect.signature(register).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "supports_text_transform" not in parameters:
        version = str(getattr(geometric_coder, "__version__", "")) or "unknown"
        raise GeCoIntegrationError(
            "Installed GeCo does not implement TeAL's current pre-1.0 integration "
            "contract: register_external_geometry(...) must accept the persisted "
            f"supports_text_transform capability (installed version {version!r}). "
            "Older development-era GeCo workspaces/builds are not backward compatible; "
            "update GeCo and recreate the linked workspace."
        )
    return coder


def _installed_geco_version() -> str | None:
    try:
        import geometric_coder
    except ImportError:
        return None
    value = getattr(geometric_coder, "__version__", None)
    return str(value) if value is not None else None


def _matrix_is_finite(values: Any) -> bool:
    if sparse.issparse(values):
        return bool(np.isfinite(values.data).all())
    try:
        return bool(np.isfinite(np.asarray(values, dtype="float64")).all())
    except (TypeError, ValueError):
        return False



def _artifact_id_from_external_ref(external_ref: Any) -> str:
    if not isinstance(external_ref, Mapping):
        raise GeCoIntegrationError(
            "TeAL-backed GeCo external_ref must be a JSON-like mapping containing artifact_id."
        )
    artifact_id = external_ref.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise GeCoIntegrationError(
            "TeAL-backed GeCo external_ref must contain a non-empty string artifact_id."
        )
    return artifact_id


def _external_ref(artifact: BaseArtifact) -> dict[str, str]:
    return {"artifact_id": artifact.artifact_id}


def _normalize_ordered_keys(
    ordered_user_keys: Sequence[Mapping[str, Any]],
    primary_key: Sequence[str],
) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    expected = list(primary_key)
    for index, raw in enumerate(ordered_user_keys):
        if not isinstance(raw, Mapping):
            raise GeCoIntegrationError(
                f"GeCo ordered_user_keys[{index}] is not a mapping."
            )
        missing = [column for column in expected if column not in raw]
        if missing:
            raise GeCoIntegrationError(
                f"GeCo ordered_user_keys[{index}] is missing TeAL primary-key "
                f"column(s) {missing!r}."
            )
        record: dict[str, Any] = {}
        for column in expected:
            value = raw[column]
            if isinstance(value, np.generic):
                value = value.item()
            record[column] = value
        keys.append(record)
    return keys


def _key_records(frame: pd.DataFrame, primary_key: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for values in frame.loc[:, list(primary_key)].itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for column, value in zip(primary_key, values, strict=True):
            if isinstance(value, np.generic):
                value = value.item()
            record[str(column)] = value
        records.append(record)
    return records


def _validate_matrix_rows(values: Any, *, expected_rows: int, purpose: str) -> None:
    shape = getattr(values, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise GeCoIntegrationError(
            f"Linked GeCo {purpose} provider result must be a 2D matrix; got shape={shape!r}."
        )
    if int(shape[0]) != int(expected_rows):
        raise GeCoIntegrationError(
            f"Linked GeCo {purpose} provider result has {shape[0]} rows; expected "
            f"{expected_rows} document rows."
        )


def _validate_workspace_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Linked GeCo workspace name must be a string.")
    name = value.strip()
    if not _WORKSPACE_NAME_RE.fullmatch(name):
        raise ValueError(
            "Linked GeCo workspace name must start with an alphanumeric character "
            "and contain only letters, numbers, '.', '_', or '-'."
        )
    return name


def _validate_resource_name(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Linked GeCo {kind} name must be a non-empty string.")
    return value.strip()


def _validate_field_name(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} must be a non-empty string.")
    return value.strip()


def _normalize_metadata_fields(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    fields = tuple(str(field).strip() for field in value)
    if any(not field for field in fields):
        raise ValueError("metadata_fields cannot contain empty names.")
    if len(set(fields)) != len(fields):
        raise ValueError("metadata_fields cannot contain duplicates.")
    return fields


def _default_resource_name(artifact: BaseArtifact) -> str:
    aliases = artifact.aliases
    if aliases:
        return str(aliases[0])
    return artifact.label


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GeCoIntegrationError(f"Expected JSON object in {path}.")
    return value
