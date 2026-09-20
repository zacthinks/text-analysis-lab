"""Best-effort cleanup helpers for preserving primary operation failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def attempt_failure_cleanup(
    action: Callable[[], Any],
    *,
    primary_error: BaseException,
    label: str,
) -> None:
    """Run secondary failure bookkeeping without masking ``primary_error``.

    Failure recording and cleanup run only after some other operation has already
    failed.  A secondary exception must therefore never replace the primary one.
    On Python versions that support ``BaseException.add_note``, retain the
    secondary failure as diagnostic context on the original exception.
    """

    try:
        action()
    except Exception as cleanup_error:  # noqa: BLE001 - this is a failure boundary by design
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(
                f"{label} failed during error cleanup: "
                f"{cleanup_error.__class__.__name__}: {cleanup_error}"
            )


def mark_operation_failed_best_effort(
    project: Any,
    *,
    writer: Any,
    artifact_id: str,
    operation_id: str,
    error: BaseException,
) -> None:
    """Record writer/artifact/operation failure without masking ``error``."""

    attempt_failure_cleanup(
        lambda: writer.mark_failed(error),
        primary_error=error,
        label="writer failure marking",
    )
    attempt_failure_cleanup(
        lambda: project.catalog.mark_artifact_failed(artifact_id),
        primary_error=error,
        label=f"artifact {artifact_id!r} failure marking",
    )
    attempt_failure_cleanup(
        lambda: project.catalog.mark_operation_failed(operation_id, error),
        primary_error=error,
        label=f"operation {operation_id!r} failure marking",
    )
