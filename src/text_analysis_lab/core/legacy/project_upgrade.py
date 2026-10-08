"""Explicit project-level upgrade support for pre-strict operator snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.legacy.operator_snapshot_v1 import (
    classify_v1_operator_snapshot,
    migrate_v1_operator_snapshot,
)


def _valid_prior_migrations(project: Any, legacy_operator_id: str) -> list[str]:
    """Return strict operator IDs that explicitly record migration from one legacy ID."""
    matches: list[str] = []
    for row in project.catalog.list_operators(include_deleted=False):
        operator_id = str(row["operator_id"])
        if str(row.get("snapshot_status")) != "serialized":
            continue
        sidecar = project.storage.operator_dir(operator_id) / "legacy_migration.json"
        if not sidecar.is_file():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("migrated_from_operator_id")) != str(legacy_operator_id):
            continue
        if int(payload.get("migrated_to_snapshot_schema", 0)) != 2:
            continue
        descriptor_path = project.storage.operator_descriptor_path(operator_id)
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(descriptor, dict):
            continue
        if int(descriptor.get("schema_version", 0)) != 2:
            continue
        if str(descriptor.get("operator_id", "")) != operator_id:
            continue
        matches.append(operator_id)
    return sorted(set(matches))


def _descriptor_payload(project: Any, operation_id: str) -> dict[str, Any]:
    path = project.storage.operation_descriptor_path(operation_id)
    if not path.is_file():
        raise OperatorError(f"Operation {operation_id} has no operation descriptor.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorError(
            f"Operation {operation_id} has an unreadable operation descriptor."
        ) from exc
    if not isinstance(payload, dict):
        raise OperatorError(
            f"Operation {operation_id} operation descriptor must contain a JSON object."
        )
    return payload


def _stage_rebound_descriptor(
    project: Any,
    *,
    operation_id: str,
    legacy_operator_id: str,
    strict_operator_id: str,
) -> Path:
    """Prepare one replacement operation descriptor without changing the live file."""
    path = project.storage.operation_descriptor_path(operation_id)
    payload = _descriptor_payload(project, operation_id)
    current = str(payload.get("operator_id", ""))
    if current not in {str(legacy_operator_id), str(strict_operator_id)}:
        raise OperatorError(
            f"Operation {operation_id} descriptor references {current!r}, not "
            f"{legacy_operator_id!r} or {strict_operator_id!r}."
        )
    payload["operator_id"] = str(strict_operator_id)
    payload["legacy_operator_upgrade"] = {
        "legacy_operator_id": str(legacy_operator_id),
        "strict_operator_id": str(strict_operator_id),
    }

    staged = path.with_name(path.name + ".legacy-upgrade.tmp")
    staged.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return staged


def _commit_staged_descriptor(
    project: Any,
    *,
    operation_id: str,
    staged: Path,
) -> None:
    path = project.storage.operation_descriptor_path(operation_id)
    os.replace(staged, path)
    project.catalog.mark_legacy_rebind_descriptor_synced(operation_id)


def _write_rebound_descriptor(
    project: Any,
    *,
    operation_id: str,
    legacy_operator_id: str,
    strict_operator_id: str,
) -> None:
    """Prepare and atomically synchronize one operation descriptor."""
    staged = _stage_rebound_descriptor(
        project,
        operation_id=operation_id,
        legacy_operator_id=legacy_operator_id,
        strict_operator_id=strict_operator_id,
    )
    _commit_staged_descriptor(
        project,
        operation_id=operation_id,
        staged=staged,
    )


def _recover_pending_descriptor_sync(project: Any) -> list[str]:
    recovered: list[str] = []
    for row in project.catalog.legacy_operation_rebinds(descriptor_synced=False):
        operation_id = str(row["operation_id"])
        _write_rebound_descriptor(
            project,
            operation_id=operation_id,
            legacy_operator_id=str(row["legacy_operator_id"]),
            strict_operator_id=str(row["strict_operator_id"]),
        )
        recovered.append(operation_id)
    return recovered


def _canonical_strict_candidate(
    project: Any,
    legacy_operator_id: str,
) -> tuple[str | None, str | None]:
    """Return (strict_id, error) without creating new migration state."""
    mapping = project.catalog.legacy_operator_upgrade(legacy_operator_id)
    if mapping is not None:
        return str(mapping["strict_operator_id"]), None

    prior = _valid_prior_migrations(project, legacy_operator_id)
    if len(prior) == 1:
        return prior[0], None
    if len(prior) > 1:
        return None, (
            f"multiple strict migrations already exist for {legacy_operator_id}: "
            + ", ".join(prior)
        )
    return None, None


def _plan_item(project: Any, row: dict[str, Any]) -> dict[str, Any]:
    legacy_id = str(row["operator_id"])
    classification = classify_v1_operator_snapshot(
        project.storage.operator_dir(legacy_id)
    )
    operations = project.catalog.operations_using_operator(legacy_id)
    complete_operations = [
        str(item["operation_id"])
        for item in operations
        if str(item.get("status")) == "complete"
    ]
    incomplete_operations = [
        str(item["operation_id"])
        for item in operations
        if str(item.get("status")) != "complete"
    ]
    aliases = project.catalog.aliases_for_operator(legacy_id)
    strict_id, ambiguity = _canonical_strict_candidate(project, legacy_id)
    canonical = project.catalog.legacy_operator_upgrade(legacy_id)
    already_rebound = (
        canonical is not None
        and not complete_operations
        and not aliases
    )

    return {
        "legacy_operator_id": legacy_id,
        "classification": classification,
        "strict_operator_id": strict_id,
        "complete_operations": complete_operations,
        "incomplete_operations": incomplete_operations,
        "aliases": aliases,
        "status": (
            "ambiguous_prior_migrations"
            if ambiguity
            else "already_upgraded"
            if already_rebound
            else "historical_only"
            if classification == "historical_only"
            else "invalid"
            if classification == "invalid"
            else "ready"
        ),
        "reason": ambiguity,
    }


def upgrade_legacy_operators(project: Any, *, dry_run: bool = False) -> dict[str, Any]:
    """Upgrade exactly migratable legacy operators for current project replay.

    This is explicit project maintenance. It never runs automatically from ordinary
    loading, operator lookup, or Pipeline reconstruction.
    """
    pending_descriptor_operations = [
        str(row["operation_id"])
        for row in project.catalog.legacy_operation_rebinds(descriptor_synced=False)
    ]
    recovered: list[str] = []
    if not dry_run:
        recovered = _recover_pending_descriptor_sync(project)

    all_rows = project.catalog.list_operators(include_deleted=False)
    legacy_rows = [
        row
        for row in all_rows
        if str(row.get("reuse_status", "legacy_unknown")) == "legacy_unknown"
    ]
    already_current_operator_ids = [
        str(row["operator_id"])
        for row in all_rows
        if str(row.get("reuse_status", "legacy_unknown")) != "legacy_unknown"
    ]
    items = [_plan_item(project, row) for row in legacy_rows]

    if dry_run:
        return {
            "dry_run": True,
            "already_current_operator_ids": already_current_operator_ids,
            "pending_descriptor_operations": pending_descriptor_operations,
            "recovered_descriptor_operations": [],
            "items": items,
        }

    for item in items:
        if item["status"] != "ready":
            continue

        legacy_id = str(item["legacy_operator_id"])
        strict_id = item["strict_operator_id"]
        if strict_id is None:
            migrated = migrate_v1_operator_snapshot(project, legacy_id)
            if migrated.operator_id is None:
                raise OperatorError(
                    f"Migration of {legacy_id} did not produce an operator ID."
                )
            strict_id = str(migrated.operator_id)
            item["strict_operator_id"] = strict_id

        strict_row = project.catalog.resolve_operator(strict_id, include_deleted=False)
        reuse_status = str(strict_row.get("reuse_status", "legacy_unknown"))
        if reuse_status != "reusable":
            item["status"] = "migrated_not_reusable"
            item["reason"] = (
                f"strict replacement {strict_id} has reuse_status={reuse_status!r}"
            )
            continue

        complete_operations = list(item["complete_operations"])

        # Stage every descriptor before catalog mutation so serialization and
        # filesystem writes are proven viable before durable references change.
        staged_descriptors: dict[str, Path] = {}
        try:
            for operation_id in complete_operations:
                staged_descriptors[operation_id] = _stage_rebound_descriptor(
                    project,
                    operation_id=operation_id,
                    legacy_operator_id=legacy_id,
                    strict_operator_id=strict_id,
                )

            project.catalog.apply_legacy_operator_upgrade(
                legacy_operator_id=legacy_id,
                strict_operator_id=strict_id,
                operation_ids=complete_operations,
            )
        except Exception:
            for staged in staged_descriptors.values():
                staged.unlink(missing_ok=True)
            raise

        for operation_id, staged in staged_descriptors.items():
            _commit_staged_descriptor(
                project,
                operation_id=operation_id,
                staged=staged,
            )

        item["status"] = "upgraded"
        item["reason"] = None

    current_after = [
        str(row["operator_id"])
        for row in project.catalog.list_operators(include_deleted=False)
        if str(row.get("reuse_status", "legacy_unknown")) != "legacy_unknown"
    ]
    return {
        "dry_run": False,
        "already_current_operator_ids": current_after,
        "pending_descriptor_operations": pending_descriptor_operations,
        "recovered_descriptor_operations": recovered,
        "items": items,
    }
