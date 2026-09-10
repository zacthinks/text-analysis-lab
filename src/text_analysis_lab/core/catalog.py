"""Project-level SQLite catalog for TeAL artifacts, operators, operations, and memos."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, cast, get_args
from collections.abc import Iterable

from text_analysis_lab.core.errors import (
    AliasOverwriteBlockedError,
    ArtifactNotFoundError,
    InvalidAliasError,
    OperatorNotFoundError,
)
from text_analysis_lab.core.lineage import validate_lineage_mode
from text_analysis_lab.core.types import (
    ArtifactStatus,
    ArtifactType,
    LineageMode,
    MemoTargetType,
    OperationStatus,
    OperationType,
    OperatorSnapshotStatus,
)
from text_analysis_lab.core.utils import utc_now_iso

_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_ARTIFACT_ID_RE = re.compile(r"^art_[0-9]{6,}$")
_RESERVED_OPERATOR_ID_RE = re.compile(r"^optr_[0-9]{6,}$")

ARTIFACT_STATUS_VALUES: tuple[ArtifactStatus, ...] = get_args(ArtifactStatus)
OPERATION_STATUS_VALUES: tuple[OperationStatus, ...] = get_args(OperationStatus)
OPERATOR_SNAPSHOT_STATUS_VALUES: tuple[OperatorSnapshotStatus, ...] = get_args(OperatorSnapshotStatus)
OPERATION_TYPE_VALUES: tuple[OperationType, ...] = get_args(OperationType)
MEMO_TARGET_TYPE_VALUES: tuple[MemoTargetType, ...] = get_args(MemoTargetType)


class ProjectCatalog:
    """Project-level lookup and provenance tables.

    The catalog stores project-level relational metadata: artifact/operator lookup,
    aliases, operation records, operation input/output edges, artifact-basis edges,
    and append-only memos. Artifact and operator descriptors remain the local
    storage records for object-specific details.
    """

    def __init__(self, catalog_dir: str | Path) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.catalog_dir / "catalog.sqlite"
        self._con: sqlite3.Connection | None = None
        self._init_db()

    @property
    def con(self) -> sqlite3.Connection:
        """Return the project-local SQLite connection, opening it lazily."""
        if self._con is None:
            self._con = sqlite3.connect(self.db_path, timeout=30.0)
            self._con.row_factory = sqlite3.Row
            self._con.execute("PRAGMA busy_timeout = 30000")
            self._con.execute("PRAGMA journal_mode = WAL")
            self._con.execute("PRAGMA foreign_keys = ON")
        return self._con

    def close(self) -> None:
        """Close the active SQLite connection, if one is open."""
        if self._con is not None:
            self._con.close()
            self._con = None

    def _init_db(self) -> None:
        with self.con as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS operators (
                    operator_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    snapshot_status TEXT NOT NULL DEFAULT 'serialized',
                    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1))
                );

                CREATE INDEX IF NOT EXISTS idx_operators_operation_type
                ON operators(operation_type);

                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    FOREIGN KEY (operator_id) REFERENCES operators(operator_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operations_operator_id
                ON operations(operator_id);

                CREATE INDEX IF NOT EXISTS idx_operations_operation_type
                ON operations(operation_type);

                CREATE INDEX IF NOT EXISTS idx_operations_status
                ON operations(status);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    lineage_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1))
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_artifact_type
                ON artifacts(artifact_type);

                CREATE INDEX IF NOT EXISTS idx_artifacts_lineage_mode
                ON artifacts(lineage_mode);

                CREATE INDEX IF NOT EXISTS idx_artifacts_status
                ON artifacts(status);

                CREATE TABLE IF NOT EXISTS artifact_basis (
                    artifact_id TEXT NOT NULL,
                    basis_artifact_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (artifact_id, ordinal),
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id),
                    FOREIGN KEY (basis_artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE INDEX IF NOT EXISTS idx_artifact_basis_basis_artifact_id
                ON artifact_basis(basis_artifact_id);

                CREATE TABLE IF NOT EXISTS operation_sources (
                    operation_id TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    source_artifact_id TEXT NOT NULL,
                    PRIMARY KEY (operation_id, source_label),
                    FOREIGN KEY (operation_id) REFERENCES operations(operation_id),
                    FOREIGN KEY (source_artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operation_sources_source_artifact_id
                ON operation_sources(source_artifact_id);

                CREATE TABLE IF NOT EXISTS operation_outputs (
                    operation_id TEXT NOT NULL,
                    output_label TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (operation_id, output_label),
                    UNIQUE (artifact_id),
                    FOREIGN KEY (operation_id) REFERENCES operations(operation_id),
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operation_outputs_operation_id
                ON operation_outputs(operation_id);

                CREATE TABLE IF NOT EXISTS artifact_aliases (
                    alias TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    touched_at TEXT NOT NULL,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                );

                CREATE INDEX IF NOT EXISTS idx_artifact_aliases_artifact_id
                ON artifact_aliases(artifact_id);

                CREATE INDEX IF NOT EXISTS idx_artifact_aliases_touched_at
                ON artifact_aliases(artifact_id, touched_at DESC, alias);

                CREATE TABLE IF NOT EXISTS operator_aliases (
                    alias TEXT PRIMARY KEY,
                    operator_id TEXT NOT NULL,
                    touched_at TEXT NOT NULL,
                    FOREIGN KEY (operator_id) REFERENCES operators(operator_id)
                );

                CREATE INDEX IF NOT EXISTS idx_operator_aliases_operator_id
                ON operator_aliases(operator_id);

                CREATE INDEX IF NOT EXISTS idx_operator_aliases_touched_at
                ON operator_aliases(operator_id, touched_at DESC, alias);

                CREATE TABLE IF NOT EXISTS memos (
                    memo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    title TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memos_target
                ON memos(target_type, target_id, created_at DESC, memo_id DESC);

                CREATE INDEX IF NOT EXISTS idx_memos_title
                ON memos(title, created_at DESC, memo_id DESC);
                """
            )
            self._ensure_column(
                con,
                table="operators",
                column="snapshot_status",
                definition="TEXT NOT NULL DEFAULT 'serialized'",
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        if "deleted" in out:
            out["deleted"] = bool(out["deleted"])
        return out

    @staticmethod
    def _dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [ProjectCatalog._dict(row) or {} for row in rows]

    @staticmethod
    def _ensure_column(
        con: sqlite3.Connection,
        *,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        existing = {str(row["name"]) for row in con.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _validate_alias(alias: str, *, reserved_id_pattern: re.Pattern[str]) -> str:
        alias = str(alias)
        if not _ALIAS_RE.match(alias):
            raise InvalidAliasError(
                f"Alias {alias!r} is not valid. Aliases must match {_ALIAS_RE.pattern!r}."
            )
        if reserved_id_pattern.match(alias):
            raise InvalidAliasError(
                f"Alias {alias!r} is reserved because it matches TeAL ID syntax."
            )
        return alias

    @staticmethod
    def _raise_artifact_alias_integrity_error(
        alias: str, exc: sqlite3.IntegrityError
    ) -> None:
        message = str(exc).lower()
        if "foreign key" in message:
            raise ArtifactNotFoundError(
                f"Alias {alias!r} points to an unknown catalog target."
            ) from exc
        raise InvalidAliasError(f"Alias {alias!r} already exists.") from exc

    @staticmethod
    def _raise_operator_alias_integrity_error(
        alias: str, exc: sqlite3.IntegrityError
    ) -> None:
        message = str(exc).lower()
        if "foreign key" in message:
            raise OperatorNotFoundError(
                f"Alias {alias!r} points to an unknown catalog target."
            ) from exc
        raise InvalidAliasError(f"Alias {alias!r} already exists.") from exc

    @staticmethod
    def _validate_artifact_status(status: str) -> ArtifactStatus:
        status = str(status)
        if status not in ARTIFACT_STATUS_VALUES:
            raise ValueError(f"status must be one of {ARTIFACT_STATUS_VALUES}.")
        return cast(ArtifactStatus, status)

    @staticmethod
    def _validate_operation_status(status: str) -> OperationStatus:
        status = str(status)
        if status not in OPERATION_STATUS_VALUES:
            raise ValueError(
                f"operation status must be one of {OPERATION_STATUS_VALUES}."
            )
        return cast(OperationStatus, status)

    @staticmethod
    def _validate_operator_snapshot_status(
        snapshot_status: str,
    ) -> OperatorSnapshotStatus:
        snapshot_status = str(snapshot_status)
        if snapshot_status not in OPERATOR_SNAPSHOT_STATUS_VALUES:
            raise ValueError(
                f"operator snapshot_status must be one of {OPERATOR_SNAPSHOT_STATUS_VALUES}."
            )
        return cast(OperatorSnapshotStatus, snapshot_status)

    @staticmethod
    def _validate_operation_type(operation_type: str) -> OperationType:
        operation_type = str(operation_type)
        if operation_type not in OPERATION_TYPE_VALUES:
            raise ValueError(f"operation_type must be one of {OPERATION_TYPE_VALUES}.")
        return cast(OperationType, operation_type)

    @staticmethod
    def _validate_memo_target_type(target_type: str | None) -> MemoTargetType | None:
        if target_type is None:
            return None
        target_type = str(target_type)
        if target_type not in MEMO_TARGET_TYPE_VALUES:
            raise ValueError(
                f"target_type must be None or one of {MEMO_TARGET_TYPE_VALUES}."
            )
        return cast(MemoTargetType, target_type)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def register_operator(
        self,
        *,
        operator_id: str,
        operation_type: OperationType | str,
        snapshot_status: OperatorSnapshotStatus | str = "serialized",
    ) -> None:
        """Register one project-owned operator snapshot in the catalog.

        ``snapshot_status`` distinguishes a reserved/pending operator ID from a
        serialized operator snapshot that can be loaded and reused. ``pending``
        is used for fit-translate operations before the fitted operator has been
        finalized and written to disk.
        """
        operation_type_value = self._validate_operation_type(str(operation_type))
        snapshot_status_value = self._validate_operator_snapshot_status(
            str(snapshot_status)
        )
        with self.con as con:
            con.execute(
                """
                INSERT INTO operators(operator_id, operation_type, snapshot_status, deleted)
                VALUES (?, ?, ?, 0)
                """,
                (str(operator_id), operation_type_value, snapshot_status_value),
            )

    def resolve_operator(
        self,
        ref: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Resolve an operator ID or alias to an operator catalog row."""
        ref = str(ref)
        with self.con as con:
            row = con.execute(
                "SELECT * FROM operators WHERE operator_id = ?",
                (ref,),
            ).fetchone()
            if row is None:
                row = con.execute(
                    """
                    SELECT o.*
                    FROM operator_aliases oa
                    JOIN operators o ON o.operator_id = oa.operator_id
                    WHERE oa.alias = ?
                    """,
                    (ref,),
                ).fetchone()

        record = self._dict(row)
        if record is None or (record["deleted"] and not include_deleted):
            raise OperatorNotFoundError(ref)
        return record

    def list_operators(
        self,
        *,
        include_deleted: bool = False,
        operation_type: OperationType | str | None = None,
        snapshot_status: OperatorSnapshotStatus | str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            where.append("deleted = 0")
        if operation_type is not None:
            where.append("operation_type = ?")
            params.append(self._validate_operation_type(str(operation_type)))
        if snapshot_status is not None:
            where.append("snapshot_status = ?")
            params.append(
                self._validate_operator_snapshot_status(str(snapshot_status))
            )

        sql = "SELECT * FROM operators"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY operator_id"

        with self.con as con:
            return self._dicts(con.execute(sql, params).fetchall())

    def _set_operator_snapshot_status(
        self,
        operator_id: str,
        snapshot_status: OperatorSnapshotStatus | str,
    ) -> None:
        """Set the persisted snapshot status for one operator."""
        status_value = self._validate_operator_snapshot_status(str(snapshot_status))
        with self.con as con:
            cur = con.execute(
                "UPDATE operators SET snapshot_status = ? WHERE operator_id = ?",
                (status_value, str(operator_id)),
            )
            if cur.rowcount == 0:
                raise OperatorNotFoundError(str(operator_id))

    def operator_snapshot_status(self, operator_id: str) -> OperatorSnapshotStatus:
        """Return whether an operator snapshot is pending, serialized, or failed."""
        row = self.resolve_operator(str(operator_id), include_deleted=True)
        return self._validate_operator_snapshot_status(str(row["snapshot_status"]))

    def mark_operator_snapshot_pending(self, operator_id: str) -> None:
        self._set_operator_snapshot_status(operator_id, "pending")

    def mark_operator_serialized(self, operator_id: str) -> None:
        self._set_operator_snapshot_status(operator_id, "serialized")

    def mark_operator_snapshot_failed(self, operator_id: str) -> None:
        self._set_operator_snapshot_status(operator_id, "failed")

    def mark_operator_deleted(self, operator_id: str) -> None:
        """Mark an operator deleted and remove aliases pointing to it."""
        operator_id = str(operator_id)
        with self.con as con:
            cur = con.execute(
                "UPDATE operators SET deleted = 1 WHERE operator_id = ?",
                (operator_id,),
            )
            if cur.rowcount == 0:
                raise OperatorNotFoundError(operator_id)
            con.execute(
                "DELETE FROM operator_aliases WHERE operator_id = ?",
                (operator_id,),
            )

    def add_operator_alias(self, operator_id: str, alias: str) -> None:
        alias = self._validate_alias(
            alias, reserved_id_pattern=_RESERVED_OPERATOR_ID_RE
        )
        try:
            with self.con as con:
                con.execute(
                    """
                    INSERT INTO operator_aliases(alias, operator_id, touched_at)
                    VALUES (?, ?, ?)
                    """,
                    (alias, str(operator_id), utc_now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            self._raise_operator_alias_integrity_error(alias, exc)

    def remove_operator_alias(self, alias: str) -> None:
        alias = str(alias)
        with self.con as con:
            con.execute("DELETE FROM operator_aliases WHERE alias = ?", (alias,))

    def touch_operator_alias(self, alias: str) -> None:
        """Make an operator alias the most recently touched display alias."""
        alias = str(alias)
        with self.con as con:
            cur = con.execute(
                "UPDATE operator_aliases SET touched_at = ? WHERE alias = ?",
                (utc_now_iso(), alias),
            )
            if cur.rowcount == 0:
                raise InvalidAliasError(f"Operator alias {alias!r} does not exist.")

    def aliases_for_operator(self, operator_id: str) -> list[str]:
        with self.con as con:
            rows = con.execute(
                """
                SELECT alias FROM operator_aliases
                WHERE operator_id = ?
                ORDER BY touched_at DESC, alias
                """,
                (str(operator_id),),
            ).fetchall()
        return [str(row["alias"]) for row in rows]

    def remove_aliases_for_operator(self, operator_id: str) -> int:
        """Remove all aliases pointing to one operator and return the number removed."""
        with self.con as con:
            cur = con.execute(
                "DELETE FROM operator_aliases WHERE operator_id = ?",
                (str(operator_id),),
            )
            return int(cur.rowcount)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def register_operation(
        self,
        *,
        operation_id: str,
        operation_type: OperationType | str,
        operator_id: str,
        status: OperationStatus | str = "incomplete",
        created_at: str | None = None,
    ) -> None:
        """Register one operation execution record."""
        operation_type_value = self._validate_operation_type(str(operation_type))
        status_value = self._validate_operation_status(str(status))
        with self.con as con:
            con.execute(
                """
                INSERT INTO operations(
                    operation_id,
                    operation_type,
                    operator_id,
                    status,
                    created_at,
                    completed_at,
                    error
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    str(operation_id),
                    operation_type_value,
                    str(operator_id),
                    status_value,
                    created_at or utc_now_iso(),
                ),
            )

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        """Return one operation record."""
        with self.con as con:
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
        record = self._dict(row)
        if record is None:
            raise KeyError(str(operation_id))
        return record

    def list_operations(
        self,
        *,
        operation_type: OperationType | str | None = None,
        operator_id: str | None = None,
        status: OperationStatus | str | None = None,
    ) -> list[dict[str, Any]]:
        """List operation records with optional filters."""
        where: list[str] = []
        params: list[Any] = []

        if operation_type is not None:
            where.append("operation_type = ?")
            params.append(self._validate_operation_type(str(operation_type)))
        if operator_id is not None:
            where.append("operator_id = ?")
            params.append(str(operator_id))
        if status is not None:
            where.append("status = ?")
            params.append(self._validate_operation_status(str(status)))

        sql = "SELECT * FROM operations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, operation_id"

        with self.con as con:
            return self._dicts(con.execute(sql, params).fetchall())

    def _set_operation_status(
        self,
        operation_id: str,
        status: OperationStatus,
        *,
        error: BaseException | str | None = None,
        completed_at: str | None = None,
    ) -> None:
        status_value = self._validate_operation_status(status)
        with self.con as con:
            cur = con.execute(
                """
                UPDATE operations
                SET status = ?, completed_at = ?, error = ?
                WHERE operation_id = ?
                """,
                (
                    status_value,
                    completed_at or utc_now_iso(),
                    None if error is None else str(error),
                    str(operation_id),
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(str(operation_id))

    def mark_operation_complete(self, operation_id: str) -> None:
        self._set_operation_status(str(operation_id), "complete")

    def mark_operation_failed(
        self,
        operation_id: str,
        error: BaseException | str | None = None,
    ) -> None:
        self._set_operation_status(str(operation_id), "failed", error=error)

    def add_operation_source(
        self,
        operation_id: str,
        source_label: str,
        source_artifact_id: str,
    ) -> None:
        """Record one labeled source artifact consumed by an operation."""
        if not isinstance(source_label, str) or not source_label:
            raise ValueError("source_label must be a non-empty string.")
        with self.con as con:
            con.execute(
                """
                INSERT INTO operation_sources(
                    operation_id, source_label, source_artifact_id
                ) VALUES (?, ?, ?)
                """,
                (str(operation_id), source_label, str(source_artifact_id)),
            )

    def operation_sources(self, operation_id: str) -> list[dict[str, Any]]:
        """Return source artifacts for an operation, ordered by source label."""
        with self.con as con:
            rows = con.execute(
                """
                SELECT * FROM operation_sources
                WHERE operation_id = ?
                ORDER BY operation_id
                """,
                (str(operation_id),),
            ).fetchall()
        return self._dicts(rows)

    def add_operation_output(
        self,
        operation_id: str,
        output_label: str,
        artifact_id: str,
        *,
        ordinal: int,
    ) -> None:
        """Record one output artifact produced by an operation."""
        with self.con as con:
            con.execute(
                """
                INSERT INTO operation_outputs(operation_id, output_label, artifact_id, ordinal)
                VALUES (?, ?, ?, ?)
                """,
                (str(operation_id), str(output_label), str(artifact_id), int(ordinal)),
            )

    def operation_outputs(self, operation_id: str) -> list[dict[str, Any]]:
        """Return output artifacts for an operation, ordered by output ordinal."""
        with self.con as con:
            rows = con.execute(
                """
                SELECT * FROM operation_outputs
                WHERE operation_id = ?
                ORDER BY ordinal
                """,
                (str(operation_id),),
            ).fetchall()
        return self._dicts(rows)

    def operation_for_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return the operation record that produced an artifact, if recorded."""
        with self.con as con:
            row = con.execute(
                """
                SELECT o.*
                FROM operation_outputs oo
                JOIN operations o ON o.operation_id = oo.operation_id
                WHERE oo.artifact_id = ?
                """,
                (str(artifact_id),),
            ).fetchone()
        return self._dict(row)

    def operations_using_operator(self, operator_id: str) -> list[dict[str, Any]]:
        """Return operations that used one operator."""
        return self.list_operations(operator_id=str(operator_id))

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        *,
        artifact_id: str,
        artifact_type: ArtifactType | str,
        label: str,
        lineage_mode: LineageMode | str,
        status: ArtifactStatus,
        basis_artifact_ids: Iterable[str] = (),
    ) -> None:
        """Register one project-owned artifact and its lineage basis edges."""
        artifact_type_value = ArtifactType(artifact_type).value
        lineage_mode_value = validate_lineage_mode(str(lineage_mode))
        status_value = self._validate_artifact_status(status)
        basis_ids = [str(item) for item in basis_artifact_ids]

        with self.con as con:
            con.execute(
                """
                INSERT INTO artifacts(
                    artifact_id,
                    artifact_type,
                    label,
                    lineage_mode,
                    status,
                    deleted
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    str(artifact_id),
                    artifact_type_value,
                    str(label),
                    lineage_mode_value,
                    status_value,
                ),
            )
            for ordinal, basis_artifact_id in enumerate(basis_ids):
                con.execute(
                    """
                    INSERT INTO artifact_basis(artifact_id, basis_artifact_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (str(artifact_id), basis_artifact_id, ordinal),
                )

    def _set_artifact_status(self, artifact_id: str, status: ArtifactStatus) -> None:
        status_value = self._validate_artifact_status(status)
        with self.con as con:
            cur = con.execute(
                "UPDATE artifacts SET status = ? WHERE artifact_id = ?",
                (status_value, str(artifact_id)),
            )
            if cur.rowcount == 0:
                raise ArtifactNotFoundError(str(artifact_id))

    def mark_artifact_incomplete(self, artifact_id: str) -> None:
        self._set_artifact_status(artifact_id, "incomplete")

    def mark_artifact_complete(self, artifact_id: str) -> None:
        self._set_artifact_status(artifact_id, "complete")

    def mark_artifact_failed(self, artifact_id: str) -> None:
        self._set_artifact_status(artifact_id, "failed")

    def resolve_artifact(
        self,
        ref: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Resolve an artifact ID or alias to an artifact catalog row."""
        ref = str(ref)
        with self.con as con:
            row = con.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (ref,),
            ).fetchone()
            if row is None:
                row = con.execute(
                    """
                    SELECT a.*
                    FROM artifact_aliases aa
                    JOIN artifacts a ON a.artifact_id = aa.artifact_id
                    WHERE aa.alias = ?
                    """,
                    (ref,),
                ).fetchone()

        record = self._dict(row)
        if record is None or (record["deleted"] and not include_deleted):
            raise ArtifactNotFoundError(ref)
        return record

    def list_artifacts(
        self,
        *,
        status: ArtifactStatus | None = None,
        include_deleted: bool = False,
        artifact_type: ArtifactType | str | None = None,
        lineage_mode: LineageMode | str | None = None,
        label: str | None = None,
        basis_artifact_id: str | None = None,
        source_artifact_id: str | None = None,
        operation_id: str | None = None,
        operator_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List artifact catalog rows with optional project-graph filters."""
        joins: list[str] = []
        where: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            where.append("a.deleted = 0")
        if status is not None:
            where.append("a.status = ?")
            params.append(self._validate_artifact_status(str(status)))
        if artifact_type is not None:
            where.append("a.artifact_type = ?")
            params.append(ArtifactType(artifact_type).value)
        if lineage_mode is not None:
            where.append("a.lineage_mode = ?")
            params.append(validate_lineage_mode(str(lineage_mode)))
        if label is not None:
            where.append("a.label = ?")
            params.append(str(label))

        if basis_artifact_id is not None:
            joins.append(
                "JOIN artifact_basis ab_filter ON ab_filter.artifact_id = a.artifact_id"
            )
            where.append("ab_filter.basis_artifact_id = ?")
            params.append(str(basis_artifact_id))

        if source_artifact_id is not None:
            joins.append(
                "JOIN operation_outputs oo_source ON oo_source.artifact_id = a.artifact_id"
            )
            joins.append(
                "JOIN operation_sources os_filter ON os_filter.operation_id = oo_source.operation_id"
            )
            where.append("os_filter.source_artifact_id = ?")
            params.append(str(source_artifact_id))

        if operation_id is not None:
            joins.append(
                "JOIN operation_outputs oo_operation ON oo_operation.artifact_id = a.artifact_id"
            )
            where.append("oo_operation.operation_id = ?")
            params.append(str(operation_id))

        if operator_id is not None:
            joins.append(
                "JOIN operation_outputs oo_operator ON oo_operator.artifact_id = a.artifact_id"
            )
            joins.append(
                "JOIN operations op_filter ON op_filter.operation_id = oo_operator.operation_id"
            )
            where.append("op_filter.operator_id = ?")
            params.append(str(operator_id))

        sql = "SELECT DISTINCT a.* FROM artifacts a"
        if joins:
            sql += " " + " ".join(joins)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY a.artifact_id"

        with self.con as con:
            return self._dicts(con.execute(sql, params).fetchall())

    def mark_artifact_deleted(self, artifact_id: str) -> None:
        """Mark an artifact deleted and remove aliases pointing to it."""
        artifact_id = str(artifact_id)
        with self.con as con:
            cur = con.execute(
                "UPDATE artifacts SET deleted = 1 WHERE artifact_id = ?",
                (artifact_id,),
            )
            if cur.rowcount == 0:
                raise ArtifactNotFoundError(artifact_id)
            con.execute(
                "DELETE FROM artifact_aliases WHERE artifact_id = ?",
                (artifact_id,),
            )

    def validate_artifact_alias(self, alias: str) -> str:
        """Validate and normalize a public artifact alias without mutating the catalog."""
        return self._validate_alias(alias, reserved_id_pattern=_RESERVED_ARTIFACT_ID_RE)

    def replace_artifact_alias_bundle(
        self,
        bindings: dict[str, str],
        *,
        expected_existing: dict[str, str | None],
        retire_artifact_ids: set[str],
    ) -> None:
        """Atomically bind a complete alias bundle and optionally retire replaced artifacts."""
        normalized = {
            self.validate_artifact_alias(alias): str(artifact_id)
            for alias, artifact_id in bindings.items()
        }
        if set(normalized) != set(expected_existing):
            raise InvalidAliasError("Alias bundle expectation keys do not match bindings.")
        now = utc_now_iso()
        with self.con as con:
            for alias, artifact_id in normalized.items():
                target = con.execute(
                    "SELECT status, deleted FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
                if target is None or bool(target["deleted"]):
                    raise ArtifactNotFoundError(artifact_id)
                if str(target["status"]) != "complete":
                    raise InvalidAliasError(
                        f"Cannot bind alias {alias!r} to incomplete artifact {artifact_id}."
                    )

            for alias, expected in expected_existing.items():
                row = con.execute(
                    "SELECT artifact_id FROM artifact_aliases WHERE alias = ?", (alias,)
                ).fetchone()
                current = None if row is None else str(row["artifact_id"])
                if current != expected:
                    raise InvalidAliasError(
                        f"Alias {alias!r} changed while the operation was running: "
                        f"expected {expected!r}, found {current!r}."
                    )

            retire_ids = {str(value) for value in retire_artifact_ids}
            if retire_ids:
                allowed_aliases: dict[str, set[str]] = {artifact_id: set() for artifact_id in retire_ids}
                for alias, artifact_id in expected_existing.items():
                    if artifact_id is not None and str(artifact_id) in retire_ids:
                        allowed_aliases[str(artifact_id)].add(alias)
                for artifact_id in sorted(retire_ids):
                    rows = con.execute(
                        "SELECT alias FROM artifact_aliases WHERE artifact_id = ?",
                        (artifact_id,),
                    ).fetchall()
                    extras = {str(row["alias"]) for row in rows} - allowed_aliases[artifact_id]
                    if extras:
                        raise AliasOverwriteBlockedError(
                            f"Cannot overwrite {artifact_id}: additional aliases appeared "
                            f"while the operation was running: {sorted(extras)}."
                        )
                    dependent_rows = con.execute(
                        """
                        SELECT DISTINCT dependent_id FROM (
                            SELECT a.artifact_id AS dependent_id
                            FROM artifact_basis ab
                            JOIN artifacts a ON a.artifact_id = ab.artifact_id
                            WHERE ab.basis_artifact_id = ? AND a.deleted = 0
                            UNION
                            SELECT a2.artifact_id AS dependent_id
                            FROM operation_sources os
                            JOIN operation_outputs oo ON oo.operation_id = os.operation_id
                            JOIN artifacts a2 ON a2.artifact_id = oo.artifact_id
                            WHERE os.source_artifact_id = ? AND a2.deleted = 0
                        )
                        """,
                        (artifact_id, artifact_id),
                    ).fetchall()
                    external = sorted(
                        str(row["dependent_id"])
                        for row in dependent_rows
                        if str(row["dependent_id"]) not in retire_ids
                    )
                    if external:
                        raise AliasOverwriteBlockedError(
                            f"Cannot overwrite {artifact_id}: live dependents appeared while "
                            f"the operation was running: {external}."
                        )

            for alias in normalized:
                con.execute("DELETE FROM artifact_aliases WHERE alias = ?", (alias,))
            for alias, artifact_id in normalized.items():
                con.execute(
                    "INSERT INTO artifact_aliases(alias, artifact_id, touched_at) VALUES (?, ?, ?)",
                    (alias, artifact_id, now),
                )

            for artifact_id in retire_artifact_ids:
                cur = con.execute(
                    "UPDATE artifacts SET deleted = 1 WHERE artifact_id = ? AND deleted = 0",
                    (str(artifact_id),),
                )
                if cur.rowcount == 0:
                    raise ArtifactNotFoundError(str(artifact_id))
                con.execute(
                    "DELETE FROM artifact_aliases WHERE artifact_id = ?",
                    (str(artifact_id),),
                )

    def add_artifact_alias(self, artifact_id: str, alias: str) -> None:
        alias = self._validate_alias(
            alias, reserved_id_pattern=_RESERVED_ARTIFACT_ID_RE
        )
        try:
            with self.con as con:
                con.execute(
                    """
                    INSERT INTO artifact_aliases(alias, artifact_id, touched_at)
                    VALUES (?, ?, ?)
                    """,
                    (alias, str(artifact_id), utc_now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            self._raise_artifact_alias_integrity_error(alias, exc)

    def remove_artifact_alias(self, alias: str) -> None:
        alias = str(alias)
        with self.con as con:
            con.execute("DELETE FROM artifact_aliases WHERE alias = ?", (alias,))

    def touch_artifact_alias(self, alias: str) -> None:
        """Make an artifact alias the most recently touched display alias."""
        alias = str(alias)
        with self.con as con:
            cur = con.execute(
                "UPDATE artifact_aliases SET touched_at = ? WHERE alias = ?",
                (utc_now_iso(), alias),
            )
            if cur.rowcount == 0:
                raise InvalidAliasError(f"Artifact alias {alias!r} does not exist.")

    def aliases_for_artifact(self, artifact_id: str) -> list[str]:
        with self.con as con:
            rows = con.execute(
                """
                SELECT alias FROM artifact_aliases
                WHERE artifact_id = ?
                ORDER BY touched_at DESC, alias
                """,
                (str(artifact_id),),
            ).fetchall()
        return [str(row["alias"]) for row in rows]

    def remove_aliases_for_artifact(self, artifact_id: str) -> int:
        """Remove all aliases pointing to one artifact and return the number removed."""
        with self.con as con:
            cur = con.execute(
                "DELETE FROM artifact_aliases WHERE artifact_id = ?",
                (str(artifact_id),),
            )
            return int(cur.rowcount)

    def artifact_basis(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return basis artifacts for an artifact, ordered by basis ordinal."""
        with self.con as con:
            rows = con.execute(
                """
                SELECT * FROM artifact_basis
                WHERE artifact_id = ?
                ORDER BY ordinal
                """,
                (str(artifact_id),),
            ).fetchall()
        return self._dicts(rows)

    def set_artifact_basis(
        self, artifact_id: str, basis_artifact_ids: Iterable[str]
    ) -> None:
        """Replace the recorded basis edges for an artifact."""
        artifact_id = str(artifact_id)
        basis_ids = [str(item) for item in basis_artifact_ids]
        with self.con as con:
            exists = con.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ? LIMIT 1",
                (artifact_id,),
            ).fetchone()
            if exists is None:
                raise ArtifactNotFoundError(artifact_id)
            con.execute(
                "DELETE FROM artifact_basis WHERE artifact_id = ?", (artifact_id,)
            )
            for ordinal, basis_artifact_id in enumerate(basis_ids):
                con.execute(
                    """
                    INSERT INTO artifact_basis(artifact_id, basis_artifact_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (artifact_id, basis_artifact_id, ordinal),
                )

    def artifacts_based_on(
        self, artifact_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """Return artifacts whose lineage basis includes the given artifact."""
        artifact_id = str(artifact_id)
        where = "ab.basis_artifact_id = ?"
        params: list[Any] = [artifact_id]
        if not include_deleted:
            where += " AND a.deleted = 0"
        with self.con as con:
            rows = con.execute(
                f"""
                SELECT a.*
                FROM artifact_basis ab
                JOIN artifacts a ON a.artifact_id = ab.artifact_id
                WHERE {where}
                ORDER BY a.artifact_id
                """,
                params,
            ).fetchall()
        return self._dicts(rows)

    def artifacts_sourced_from(
        self, artifact_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """Return artifacts produced by operations that consumed the given artifact."""
        artifact_id = str(artifact_id)
        where = "os.source_artifact_id = ?"
        params: list[Any] = [artifact_id]
        if not include_deleted:
            where += " AND a.deleted = 0"
        with self.con as con:
            rows = con.execute(
                f"""
                SELECT DISTINCT a.*
                FROM operation_sources os
                JOIN operation_outputs oo ON oo.operation_id = os.operation_id
                JOIN artifacts a ON a.artifact_id = oo.artifact_id
                WHERE {where}
                ORDER BY a.artifact_id
                """,
                params,
            ).fetchall()
        return self._dicts(rows)

    def artifact_dependents(
        self, artifact_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """Return artifacts that depend on an artifact as basis or operation source."""
        artifact_id = str(artifact_id)
        deleted_basis = "" if include_deleted else " AND a.deleted = 0"
        deleted_source = "" if include_deleted else " AND a2.deleted = 0"
        with self.con as con:
            rows = con.execute(
                f"""
                SELECT DISTINCT * FROM (
                    SELECT a.*
                    FROM artifact_basis ab
                    JOIN artifacts a ON a.artifact_id = ab.artifact_id
                    WHERE ab.basis_artifact_id = ?{deleted_basis}
                    UNION
                    SELECT a2.*
                    FROM operation_sources os
                    JOIN operation_outputs oo ON oo.operation_id = os.operation_id
                    JOIN artifacts a2 ON a2.artifact_id = oo.artifact_id
                    WHERE os.source_artifact_id = ?{deleted_source}
                )
                ORDER BY artifact_id
                """,
                (artifact_id, artifact_id),
            ).fetchall()
        return self._dicts(rows)

    def has_artifact_dependents(
        self, artifact_id: str, *, include_deleted: bool = False
    ) -> bool:
        """Return True if any artifact depends on this artifact as basis or source."""
        return bool(
            self.artifact_dependents(artifact_id, include_deleted=include_deleted)
        )

    def artifacts_created_by_operator(
        self, operator_id: str, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """Return artifacts produced by operations using the given operator."""
        operator_id = str(operator_id)
        where = "o.operator_id = ?"
        params: list[Any] = [operator_id]
        if not include_deleted:
            where += " AND a.deleted = 0"
        with self.con as con:
            rows = con.execute(
                f"""
                SELECT DISTINCT a.*
                FROM operations o
                JOIN operation_outputs oo ON oo.operation_id = o.operation_id
                JOIN artifacts a ON a.artifact_id = oo.artifact_id
                WHERE {where}
                ORDER BY a.artifact_id
                """,
                params,
            ).fetchall()
        return self._dicts(rows)

    def has_operator_outputs(
        self, operator_id: str, *, include_deleted: bool = False
    ) -> bool:
        """Return True if any artifact was produced by this operator."""
        return bool(
            self.artifacts_created_by_operator(
                operator_id,
                include_deleted=include_deleted,
            )
        )

    # ------------------------------------------------------------------
    # Memos
    # ------------------------------------------------------------------

    def add_memo(
        self,
        *,
        target_type: MemoTargetType | str,
        target_id: str | None = None,
        body: str,
        title: str | None = None,
    ) -> int:
        """Add one memo version and return its row-level memo_id.

        Memos are versioned by their logical ``(target_type, target_id)`` pair.
        Adding another memo with the same pair creates a newer version rather
        than overwriting the older memo. Standalone memos may omit ``target_id``;
        the catalog assigns the next numeric standalone target id.
        """
        target_type_value = self._validate_memo_target_type(target_type)
        if target_type_value is None:
            raise ValueError("target_type must not be None.")

        body_value = str(body)
        if not body_value.strip():
            raise ValueError("Memo body must be non-empty.")

        title_value = None if title is None else str(title)

        with self.con as con:
            if target_id is None:
                if target_type_value != "standalone":
                    raise ValueError(
                        "target_id is required unless target_type='standalone'."
                    )
                target_id_value = self._next_standalone_memo_target_id(con)
            else:
                target_id_value = str(target_id)
                if not target_id_value:
                    raise ValueError("target_id must be non-empty.")

            cur = con.execute(
                """
                INSERT INTO memos(target_type, target_id, title, body, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    target_type_value,
                    target_id_value,
                    title_value,
                    body_value,
                    utc_now_iso(),
                ),
            )
            memo_id = cur.lastrowid
            if memo_id is None:
                raise RuntimeError(
                    "SQLite did not return a memo_id after inserting memo."
                )
            return int(memo_id)

    def get_memo_all_versions(
        self,
        *,
        target_type: MemoTargetType | str,
        target_id: str,
    ) -> list[dict[str, Any]]:
        """Return all memo versions for one logical target, newest first."""
        target_type_value = self._validate_memo_target_type(str(target_type))
        if target_type_value is None:
            raise ValueError("target_type must not be None.")
        target_id_value = str(target_id)
        if not target_id_value:
            raise ValueError("target_id must be non-empty.")

        with self.con as con:
            rows = con.execute(
                """
                SELECT * FROM memos
                WHERE target_type = ? AND target_id = ?
                ORDER BY memo_id DESC
                """,
                (target_type_value, target_id_value),
            ).fetchall()
        return self._dicts(rows)

    def get_memo(
        self,
        *,
        target_type: MemoTargetType | str,
        target_id: str,
    ) -> dict[str, Any] | None:
        """Return the newest memo version for one logical target, if any."""
        versions = self.get_memo_all_versions(
            target_type=target_type,
            target_id=target_id,
        )
        return versions[0] if versions else None

    def list_memos(
        self,
        *,
        target_type: MemoTargetType | str | None = None,
        latest_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List memo rows, optionally filtered by target type.

        If latest_only=True, return only the newest memo version for each
        (target_type, target_id) pair.
        """
        where: list[str] = []
        params: list[Any] = []

        if target_type is not None:
            where.append("m.target_type = ?")
            params.append(self._validate_memo_target_type(str(target_type)))

        if latest_only:
            sql = """
                SELECT m.*
                FROM memos m
                JOIN (
                    SELECT target_type, target_id, MAX(memo_id) AS memo_id
                    FROM memos
                    GROUP BY target_type, target_id
                ) latest
                ON latest.memo_id = m.memo_id
            """
        else:
            sql = "SELECT m.* FROM memos m"

        if where:
            sql += " WHERE " + " AND ".join(where)

        sql += " ORDER BY m.created_at DESC, m.memo_id DESC"

        with self.con as con:
            return self._dicts(con.execute(sql, params).fetchall())

    def _next_standalone_memo_target_id(self, con: sqlite3.Connection) -> str:
        """Return the next numeric target_id for a new standalone memo."""
        row = con.execute(
            """
            SELECT COALESCE(MAX(CAST(target_id AS INTEGER)), 0) AS max_target_id
            FROM memos
            WHERE target_type = 'standalone'
            """
        ).fetchone()
        max_target_id = int(row["max_target_id"] if row is not None else 0)
        return str(max_target_id + 1)
