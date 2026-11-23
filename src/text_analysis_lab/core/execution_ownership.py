"""Single-writer OS locks for one TeAL translation operation.

Only the owning coordinator holds this lock. Unlike a durable PID/lease row,
an OS advisory lock is released when its process terminates, allowing explicit
checkpoint recovery after abrupt termination. A process-local guard additionally
covers independently opened Project sessions in the same Python process.
The project directory must reside on a filesystem supporting these locks.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from text_analysis_lab.core.errors import OperatorError

_GUARD = Lock()
_OWNED: set[str] = set()


@contextmanager
def exclusive_operation_execution(operation_dir: Path) -> Iterator[None]:
    """Refuse a competing live owner before inspecting or mutating checkpoints."""
    lock_path = operation_dir / ".execution.lock"
    key = str(lock_path.resolve())
    with _GUARD:
        if key in _OWNED:
            raise OperatorError(
                "Operation is already executing in another Project session; "
                "concurrent resume is not permitted."
            )
        _OWNED.add(key)
    try:
        with lock_path.open("a+b") as handle:
            # msvcrt.locking locks one existing byte from the current offset.
            if os.name == "nt":
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\x00")
                    handle.flush()
                handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise OperatorError(
                    "Operation is already executing in another process; "
                    "concurrent resume is not permitted."
                ) from exc
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        with _GUARD:
            _OWNED.remove(key)
