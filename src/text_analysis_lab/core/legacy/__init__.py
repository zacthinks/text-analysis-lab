"""Temporary compatibility helpers for pre-strict TeAL snapshots."""

from text_analysis_lab.core.legacy.operator_snapshot_v1 import (
    classify_v1_operator_snapshot,
    migrate_v1_operator_snapshot,
)

__all__ = [
    "classify_v1_operator_snapshot",
    "migrate_v1_operator_snapshot",
]
