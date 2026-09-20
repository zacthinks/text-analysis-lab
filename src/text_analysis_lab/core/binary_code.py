"""Small resumable local binary coding interface for audit artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


def binary_code(
    project: Project,
    source: BaseArtifact | str,
    *,
    text_source: BaseArtifact | str,
    text_field: str,
    instructions: str,
    context_before: int = 2,
    context_after: int = 2,
    memo: str | None = None,
) -> BaseArtifact:
    """Interactively code every source observation as 0 or 1.

    Progress is committed after each decision to a project-local SQLite file.
    Interrupting the cell/process leaves that state intact; calling the same
    coding request after reopen resumes at the first uncoded key.  Once every
    key is coded, labels are committed through ``from_keyed_frame`` and the
    transient state is removed.
    """
    audit = project.get_artifact(source)
    text_artifact = project.get_artifact(text_source)
    audit.require_complete()
    text_artifact.require_complete()
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string.")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string.")
    context_before = _nonnegative_int(context_before, name="context_before")
    context_after = _nonnegative_int(context_after, name="context_after")
    keys = tuple(str(value) for value in audit.primary_key)
    if tuple(str(value) for value in text_artifact.primary_key) != keys:
        raise ArtifactError(
            "binary_code source and text_source must use the same primary-key columns."
        )
    if text_field not in [str(v) for v in text_artifact.get_data_columns()]:
        raise ArtifactError(
            f"binary_code text_source does not expose text field {text_field!r}."
        )

    audit_keys = audit.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        form="table",
        include_position=True,
    )
    if not isinstance(audit_keys, pd.DataFrame):
        raise ArtifactError("binary_code could not materialize audit keys.")
    audit_keys = audit_keys.sort_values("_position", kind="stable").reset_index(
        drop=True
    )
    audit_keys = audit_keys.loc[:, list(keys)]
    if audit_keys.empty:
        raise ArtifactError("binary_code source contains no observations.")

    state_path = _state_path(
        project,
        audit_id=audit.artifact_id,
        text_id=text_artifact.artifact_id,
        text_field=text_field,
        instructions=instructions,
        context_before=context_before,
        context_after=context_after,
    )
    con = _open_state(
        state_path,
        metadata={
            "audit_artifact_id": audit.artifact_id,
            "text_artifact_id": text_artifact.artifact_id,
            "text_field": text_field,
            "instructions": instructions,
            "context_before": context_before,
            "context_after": context_after,
            "primary_key": list(keys),
        },
    )
    try:
        completed = _load_labels(con)
        total = len(audit_keys)
        for ordinal, row in enumerate(
            audit_keys.itertuples(index=False, name=None), start=1
        ):
            key_tuple = tuple(int(value) for value in row)
            encoded = json.dumps(key_tuple, separators=(",", ":"))
            if encoded in completed:
                continue
            _render_item(
                text_artifact,
                key_tuple=key_tuple,
                keys=keys,
                text_field=text_field,
                instructions=instructions,
                before=context_before,
                after=context_after,
                ordinal=ordinal,
                total=total,
            )
            label = _prompt_binary()
            with con:
                con.execute(
                    "INSERT OR REPLACE INTO labels(key_json, label) VALUES (?, ?)",
                    (encoded, int(label)),
                )
            completed[encoded] = int(label)

        if len(completed) != total:
            raise ArtifactError(
                f"binary_code cannot finalize until every observation is coded; "
                f"coded={len(completed)}, total={total}."
            )
        labels = audit_keys.copy()
        labels["label"] = [
            completed[json.dumps(tuple(int(v) for v in row), separators=(",", ":"))]
            for row in audit_keys.itertuples(index=False, name=None)
        ]
    finally:
        con.close()

    artifact = project.from_keyed_frame(
        audit,
        labels,
        data_fields=["label"],
        require_complete=True,
        output_label=DEFAULT_OUTPUT_LABEL,
        memo=memo,
    )
    state_path.unlink(missing_ok=True)
    try:
        state_path.parent.rmdir()
    except OSError:
        pass
    return artifact


def _render_item(
    text_artifact: BaseArtifact,
    *,
    key_tuple: tuple[int, ...],
    keys: tuple[str, ...],
    text_field: str,
    instructions: str,
    before: int,
    after: int,
    ordinal: int,
    total: int,
) -> None:
    key_arg: Any = key_tuple[0] if len(key_tuple) == 1 else key_tuple
    focus_position = text_artifact.position_by_key(key_arg)
    start = max(0, focus_position - before)
    stop = min(int(text_artifact.n_rows or 0), focus_position + after + 1)
    positions = list(range(start, stop))
    context = text_artifact.query(
        key_columns=True,
        data_columns=[text_field],
        metadata_columns=False,
        positions=positions,
        form="table",
        include_position=True,
    )
    if not isinstance(context, pd.DataFrame):
        raise ArtifactError("binary_code could not materialize readable text context.")

    print("\n" + "=" * 72)
    print(f"Binary audit coding [{ordinal}/{total}]")
    print("Instructions:")
    print(instructions)
    print("-" * 72)
    for _, row in context.iterrows():
        marker = ">>>" if int(row["_position"]) == focus_position else "   "
        key_display = ", ".join(f"{key}={int(row[key])}" for key in keys)
        print(f"{marker} {key_display}: {row[text_field]}")
    print("-" * 72)


def _prompt_binary() -> int:
    while True:
        value = input("Code 0 or 1: ").strip()
        if value in {"0", "1"}:
            return int(value)
        print("Please enter exactly 0 or 1.")


def _state_path(
    project: Project,
    *,
    audit_id: str,
    text_id: str,
    text_field: str,
    instructions: str,
    context_before: int,
    context_after: int,
) -> Path:
    payload = json.dumps(
        {
            "audit": audit_id,
            "text": text_id,
            "field": text_field,
            "instructions": instructions,
            "before": context_before,
            "after": context_after,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    root = project.storage.teal_dir / "coding"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"binary-{digest}.sqlite"


def _open_state(path: Path, *, metadata: dict[str, Any]) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    with con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS labels (
                key_json TEXT PRIMARY KEY,
                label INTEGER NOT NULL CHECK (label IN (0, 1))
            );
            """
        )
        encoded = json.dumps(metadata, sort_keys=True)
        row = con.execute(
            "SELECT payload FROM session_metadata WHERE id = 1"
        ).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO session_metadata(id, payload) VALUES (1, ?)",
                (encoded,),
            )
        elif str(row["payload"]) != encoded:
            con.close()
            raise ArtifactError(
                "Existing binary coding state does not match this coding request."
            )
    return con


def _load_labels(con: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row["key_json"]): int(row["label"])
        for row in con.execute("SELECT key_json, label FROM labels").fetchall()
    }


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)
