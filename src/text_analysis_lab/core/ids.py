"""Sequential ID helpers for TeAL projects."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

IdKind = Literal["artifact", "operator", "operation"]

_DEFAULT_COUNTS = {"artifacts": 0, "operators": 0, "operations": 0}


def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"counts": dict(_DEFAULT_COUNTS)}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def next_id(manifest_path: Path, kind: IdKind) -> str:
    """Return the next sequential TeAL ID using a transactional SQLite counter.

    The legacy implementation updated ``manifest.json`` directly, which could
    lose increments or allocate duplicate IDs when multiple TeAL processes used
    the same project concurrently.  Keep the existing function boundary, but
    move the mutable counter into the project catalog database where allocation
    can be serialized with ``BEGIN IMMEDIATE``.

    Existing manifest counts are used only to seed a counter the first time a
    project uses the SQLite allocator, preserving the legacy sequence.
    """
    manifest_path = Path(manifest_path)
    if kind not in ("artifact", "operator", "operation"):  # pragma: no cover
        raise ValueError(f"Unknown ID kind: {kind!r}")

    count_key = {
        "artifact": "artifacts",
        "operator": "operators",
        "operation": "operations",
    }[kind]
    prefix = {
        "artifact": "art_",
        "operator": "optr_",
        "operation": "run_",
    }[kind]
    catalog_source = {
        "artifact": ("artifacts", "artifact_id", 5),
        "operator": ("operators", "operator_id", 6),
        "operation": ("operations", "operation_id", 5),
    }[kind]

    manifest = _load_manifest(manifest_path)
    legacy_count = int(manifest.get("counts", {}).get(count_key, 0))

    catalog_db = manifest_path.parent / "catalog" / "catalog.sqlite"
    catalog_db.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(catalog_db, timeout=30.0)
    try:
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS id_counters (
                kind TEXT PRIMARY KEY,
                value INTEGER NOT NULL CHECK (value >= 0)
            )
            """
        )
        row = con.execute(
            "SELECT value FROM id_counters WHERE kind = ?",
            (kind,),
        ).fetchone()
        if row is None:
            table, id_column, suffix_start = catalog_source
            try:
                catalog_row = con.execute(
                    f"SELECT MAX(CAST(substr({id_column}, ?) AS INTEGER)) FROM {table}",
                    (suffix_start,),
                ).fetchone()
                catalog_count = (
                    0
                    if catalog_row is None or catalog_row[0] is None
                    else int(catalog_row[0])
                )
            except sqlite3.OperationalError:
                catalog_count = 0
            current = max(legacy_count, catalog_count)
            con.execute(
                "INSERT INTO id_counters(kind, value) VALUES (?, ?)",
                (kind, current),
            )
        else:
            current = int(row[0])

        next_value = current + 1
        con.execute(
            "UPDATE id_counters SET value = ? WHERE kind = ?",
            (next_value, kind),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    return f"{prefix}{next_value:06d}"
