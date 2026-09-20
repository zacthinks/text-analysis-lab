from __future__ import annotations

from types import SimpleNamespace

from text_analysis_lab.core.failure_cleanup import (
    attempt_failure_cleanup,
    mark_operation_failed_best_effort,
)


def test_attempt_failure_cleanup_does_not_replace_primary_error() -> None:
    primary = RuntimeError("primary")

    def fail() -> None:
        raise ValueError("secondary")

    attempt_failure_cleanup(fail, primary_error=primary, label="cleanup")

    notes = getattr(primary, "__notes__", [])
    if notes:
        assert "cleanup failed during error cleanup" in notes[0]
        assert "ValueError: secondary" in notes[0]


def test_mark_operation_failed_best_effort_attempts_every_step() -> None:
    calls: list[str] = []

    class Writer:
        def mark_failed(self, error: BaseException) -> None:
            calls.append("writer")
            raise RuntimeError("writer failed")

    class Catalog:
        def mark_artifact_failed(self, artifact_id: str) -> None:
            calls.append(f"artifact:{artifact_id}")
            raise RuntimeError("artifact failed")

        def mark_operation_failed(
            self, operation_id: str, error: BaseException
        ) -> None:
            calls.append(f"operation:{operation_id}")

    primary = ValueError("primary")
    project = SimpleNamespace(catalog=Catalog())

    mark_operation_failed_best_effort(
        project,
        writer=Writer(),
        artifact_id="art_1",
        operation_id="op_1",
        error=primary,
    )

    assert calls == ["writer", "artifact:art_1", "operation:op_1"]
