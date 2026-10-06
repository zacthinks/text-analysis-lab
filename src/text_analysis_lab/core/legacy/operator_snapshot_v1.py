"""Temporary migration support for pre-strict operator snapshot schema v1.

This module is intentionally isolated so it can be removed after the pre-launch
pilot cohort no longer needs compatibility with v1 operator snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.operator import BaseOperator

LegacyMigrationStatus = Literal[
    "exactly_migratable",
    "historical_only",
    "invalid",
]

_EXACT_CLASS_NAMES = frozenset(
    {
        "RegexCleaner",
        "DelimiterDecomposer",
        "TextLength",
        "CountVectorizer",
        "ArtifactCountVectorizer",
        "DictionaryTranslator",
    }
)


def classify_v1_operator_snapshot(
    operator_dir: str | Path,
) -> LegacyMigrationStatus:
    """Classify whether a schema-v1 snapshot can be migrated exactly."""
    path = Path(operator_dir)
    descriptor_path = path / "operator.json"
    if not descriptor_path.is_file():
        return "invalid"

    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if not isinstance(descriptor, Mapping):
        return "invalid"
    if int(descriptor.get("schema_version", 1)) != 1:
        return "invalid"

    class_info = descriptor.get("class")
    if not isinstance(class_info, Mapping):
        return "invalid"
    qualname = class_info.get("qualname")
    if not isinstance(qualname, str) or not qualname:
        return "invalid"
    class_name = qualname.rsplit(".", 1)[-1]
    if class_name not in _EXACT_CLASS_NAMES:
        return "historical_only"

    state = descriptor.get("json_state", {})
    assets = descriptor.get("assets", {})
    if not isinstance(state, Mapping) or not isinstance(assets, Mapping):
        return "invalid"

    if class_name in {"CountVectorizer", "ArtifactCountVectorizer"}:
        if bool(state.get("is_fitted", False)):
            filename = assets.get("vocabulary_file")
            if not isinstance(filename, str) or not (path / "assets" / filename).is_file():
                return "historical_only"

    if class_name == "DictionaryTranslator":
        dictionary_state = state.get("dictionary")
        if not isinstance(dictionary_state, Mapping):
            return "invalid"
        storage = str(dictionary_state.get("storage", ""))
        if storage == "embedded":
            return "exactly_migratable"
        if storage != "external":
            return "invalid"
        filename = assets.get("dictionary_file")
        if not isinstance(filename, str) or not (path / "assets" / filename).is_file():
            return "historical_only"

    return "exactly_migratable"


def migrate_v1_operator_snapshot(
    project: Any,
    operator_id: str,
) -> BaseOperator:
    """Create a new strict v2 operator snapshot from an exact legacy v1 snapshot.

    The original operator and all historical operation/artifact provenance remain
    untouched. The new operator receives a new operator ID and a small migration
    sidecar that records its legacy source.
    """
    old_id = str(operator_id)
    row = project.catalog.resolve_operator(old_id, include_deleted=False)
    if str(row.get("snapshot_status", "serialized")) != "serialized":
        raise OperatorError(
            f"Legacy operator {old_id} is not migratable because its snapshot is not serialized."
        )
    if str(row.get("reuse_status", "legacy_unknown")) != "legacy_unknown":
        raise OperatorError(
            f"Operator {old_id} is not a legacy-unknown snapshot requiring v1 migration."
        )

    old_dir = project.storage.operator_dir(old_id)
    status = classify_v1_operator_snapshot(old_dir)
    if status != "exactly_migratable":
        if status == "historical_only":
            raise OperatorError(
                f"Legacy operator {old_id} is historical-only under strict freeze semantics "
                "and cannot be migrated exactly."
            )
        raise OperatorError(
            f"Legacy operator {old_id} is invalid/corrupt and cannot be migrated."
        )

    descriptor = json.loads(
        (old_dir / "operator.json").read_text(encoding="utf-8")
    )
    loaded = BaseOperator.load_from_dir(old_dir)
    state = descriptor.get("json_state", {})
    assets = descriptor.get("assets", {})
    if not isinstance(state, Mapping) or not isinstance(assets, Mapping):
        raise OperatorError(f"Legacy operator {old_id} has malformed snapshot state.")

    migrated = loaded.__class__.from_json_state(state)
    if not isinstance(migrated, BaseOperator):
        raise OperatorError(
            f"Legacy operator {old_id} did not reconstruct to a BaseOperator."
        )
    migrated.load_assets(old_dir / "assets", assets)

    new_id = next_id(project.storage.manifest_path, "operator")
    project.catalog.register_operator(
        operator_id=new_id,
        operation_type=migrated.operation_type,
        snapshot_status="pending",
        reuse_status="pending",
    )
    try:
        new_dir = project.storage.operator_dir(new_id)
        migrated.save_to_dir(new_dir, operator_id=new_id)
        project.catalog.mark_operator_serialized(new_id)
        caps = getattr(migrated, "execution_capabilities", None)
        if callable(caps) and not bool(caps(project=project).reusable):
            project.catalog.mark_operator_provenance_only(new_id)
        else:
            project.catalog.mark_operator_reusable(new_id)
        (new_dir / "legacy_migration.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "migrated_from_operator_id": old_id,
                    "migrated_from_snapshot_schema": 1,
                    "migrated_to_snapshot_schema": 2,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except Exception:
        project.catalog.mark_operator_snapshot_failed(new_id)
        raise

    return migrated
