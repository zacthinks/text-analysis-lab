"""Project storage initialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil

from text_analysis_lab.core.utils import utc_now_iso


def ensure_project_dir(
    path: Path,
    name: str,
    *,
    delete_existing: bool = False,
) -> Path:
    """Create or reopen the hidden ``.teal`` directory for a project.

    When ``delete_existing=True``, only the existing TeAL-owned ``.teal``
    directory is removed. Other files in the containing project directory are
    never deleted.
    """
    path = Path(path)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Project name must be a non-empty string.")
    name = name.strip()
    path.mkdir(parents=True, exist_ok=True)

    teal = path / ".teal"
    if delete_existing and teal.exists():
        if teal.is_symlink():
            raise ValueError(
                f"Refusing to delete symlinked TeAL state directory: {teal}."
            )
        shutil.rmtree(teal)
    teal.mkdir(exist_ok=True)

    (teal / "artifacts").mkdir(exist_ok=True)
    (teal / "operators").mkdir(exist_ok=True)
    (teal / "operations").mkdir(exist_ok=True)
    (teal / "catalog").mkdir(exist_ok=True)

    manifest = teal / "manifest.json"
    if not manifest.exists():
        now = utc_now_iso()
        manifest.write_text(
            json.dumps(
                {
                    "project": {
                        "project_id": name,
                        "name": name,
                        "created_at": now,
                        "updated_at": now,
                    },
                    "storage_version": "0.1",
                    "counts": {"artifacts": 0, "operators": 0, "operations": 0},
                    "paths": {
                        "artifacts": str(teal / "artifacts"),
                        "operators": str(teal / "operators"),
                        "catalog": str(teal / "catalog"),
                        "operations": str(teal / "operations"),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        existing = payload.get("project", {})
        existing_id = existing.get("project_id")
        if existing_id != name:
            raise ValueError(
                f"Existing TeAL project has project_id={existing_id!r}, not {name!r}. "
                "Use Project.open(...) to open an existing project."
            )

    return teal


def touch_manifest(manifest_path: Path) -> None:
    """Update manifest project.updated_at without changing counters."""
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("project", {})["updated_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ProjectStorage:
    """Path vocabulary for a TeAL project.

    This wraps the existing project initialization behavior. It owns path
    conventions only; it does not write artifact components.
    """

    project_path: Path
    teal_dir: Path

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        name: str,
        *,
        delete_existing: bool = False,
    ) -> "ProjectStorage":
        project_path = Path(path)
        teal_dir = ensure_project_dir(
            project_path,
            name=name,
            delete_existing=delete_existing,
        )
        return cls(project_path=project_path, teal_dir=teal_dir)

    @classmethod
    def open(cls, path: str | Path) -> "ProjectStorage":
        project_path = Path(path)
        teal_dir = project_path / ".teal"
        if not teal_dir.exists():
            raise FileNotFoundError(
                f"No TeAL project found at {project_path!s}; expected {teal_dir!s}."
            )
        return cls(project_path=project_path, teal_dir=teal_dir)

    @property
    def manifest_path(self) -> Path:
        return self.teal_dir / "manifest.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.teal_dir / "artifacts"

    @property
    def operators_dir(self) -> Path:
        return self.teal_dir / "operators"
    
    @property
    def operations_dir(self) -> Path:
        return self.teal_dir / "operations"

    @property
    def catalog_dir(self) -> Path:
        return self.teal_dir / "catalog"

    @property
    def catalog_db_path(self) -> Path:
        return self.catalog_dir / "catalog.sqlite"

    def artifact_dir(self, artifact_id: str) -> Path:
        return self.artifacts_dir / artifact_id

    def artifact_descriptor_path(self, artifact_id: str) -> Path:
        return self.artifact_dir(artifact_id) / "artifact.json"

    def artifact_component_dir(self, artifact_id: str, component: str) -> Path:
        return self.artifact_dir(artifact_id) / component

    def operator_dir(self, operator_id: str) -> Path:
        return self.operators_dir / operator_id

    def operator_descriptor_path(self, operator_id: str) -> Path:
        return self.operator_dir(operator_id) / "operator.json"

    def operation_dir(self, operation_id: str) -> Path:
        return self.operations_dir / operation_id

    def operation_descriptor_path(self, operation_id: str) -> Path:
        return self.operation_dir(operation_id) / "operation.json"

    def operation_plan_path(self, operation_id: str) -> Path:
        return self.operation_dir(operation_id) / "plan.sqlite"
    
    def operation_temp_dir(self, operation_id: str) -> Path:
        return self.operation_dir(operation_id) / "temp"
    
    def operation_temp_dir_for_operator(self, operation_id: str) -> Path:
        return self.operation_temp_dir(operation_id) / "operator"
    
    def operation_temp_dir_for_writers(self, operation_id: str) -> Path:
        return self.operation_temp_dir(operation_id) / "writers"

    def touch_manifest(self) -> None:
        touch_manifest(self.manifest_path)


@dataclass(frozen=True)
class ArtifactStorage:
    """Path vocabulary for one artifact directory.

    This owns filesystem conventions inside a single artifact directory.
    It does not read or interpret artifact descriptors.
    """

    artifact_dir: Path

    @classmethod
    def open(cls, artifact_dir: str | Path) -> "ArtifactStorage":
        return cls(artifact_dir=Path(artifact_dir))

    @property
    def descriptor_path(self) -> Path:
        return self.artifact_dir / "artifact.json"

    @property
    def keys_dir(self) -> Path:
        return self.artifact_dir / "keys"

    @property
    def data_dir(self) -> Path:
        return self.artifact_dir / "data"

    @property
    def metadata_dir(self) -> Path:
        return self.artifact_dir / "metadata"

    @property
    def data_values_dir(self) -> Path:
        return self.data_dir / "values"

    @property
    def data_columns_path(self) -> Path:
        return self.data_dir / "columns.parquet"

    @property
    def data_row_names_dir(self) -> Path:
        return self.data_dir / "row_names"

    @staticmethod
    def part_name(index: int, suffix: str) -> str:
        return f"part-{int(index):06d}.{suffix}"

    def part_path(self, component: str, index: int, suffix: str) -> Path:
        return self.artifact_dir / component / self.part_name(index, suffix)

    def key_part_path(self, index: int, suffix: str = "parquet") -> Path:
        return self.keys_dir / self.part_name(index, suffix)

    def data_part_path(self, index: int, suffix: str) -> Path:
        return self.data_dir / self.part_name(index, suffix)

    def metadata_part_path(self, index: int, suffix: str = "parquet") -> Path:
        return self.metadata_dir / self.part_name(index, suffix)

    def data_value_part_path(self, index: int, suffix: str) -> Path:
        return self.data_values_dir / self.part_name(index, suffix)

    def data_row_names_part_path(
        self, index: int, suffix: str = "parquet"
    ) -> Path:
        return self.data_row_names_dir / self.part_name(index, suffix)

    def ensure_artifact_dir(self) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.artifact_dir

    def ensure_keys_dir(self) -> Path:
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        return self.keys_dir

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def ensure_metadata_dir(self) -> Path:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self.metadata_dir

    def ensure_data_values_dir(self) -> Path:
        self.data_values_dir.mkdir(parents=True, exist_ok=True)
        return self.data_values_dir
