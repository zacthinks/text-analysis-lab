from typing import Any
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone

from text_analysis_lab.core.errors import DuckDBRegexValidationError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def str_keys(record: Any) -> dict[str, Any]:
    """Normalize a pandas/dict-like row record to a string-keyed dict."""
    return {str(key): value for key, value in dict(record).items()}


def quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def unique_column_name(
    name: str,
    existing: set[str],
    resolve_column_conflicts: bool = True,
) -> str:
    """Return a unique column name by appending _1, _2, ... as needed."""
    candidate = str(name)
    suffix = 1

    while candidate in existing:
        if not resolve_column_conflicts:
            raise ValueError(f"Column name collision: {name!r}.")
        candidate = f"{name}_{suffix}"
        suffix += 1

    return candidate


def resolve_names(
    base_names: Sequence[str],
    qualified_names: Sequence[str],
    reserved_names: Sequence[str] = (),
    error_cls: type[Exception] = ValueError,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Resolve base names to unique output names, avoiding reserved names."""
    n = len(base_names)
    if n != len(qualified_names):
        raise ValueError("base_names and qualified_names must have the same length.")
    names = list(base_names)
    promoted = [False] * n
    ambiguous: dict[str, tuple[str, ...]] = {}
    first_round = True

    while True:
        counts = Counter([*reserved_names, *names])
        duplicates = {name for name, count in counts.items() if count > 1}

        if not duplicates:
            break

        to_promote = [
            i for i, name in enumerate(names) if name in duplicates and not promoted[i]
        ]

        if not to_promote:
            raise error_cls(
                "Could not disambiguate artifact view columns. "
                f"Duplicate names: {sorted(duplicates)}."
            )

        if first_round:
            for duplicate in sorted(duplicates):
                options = [
                    qualified_names[i]
                    for i, name in enumerate(names)
                    if name == duplicate
                ]
                if duplicate in reserved_names:
                    options = [duplicate, *options]
                if len(options) > 1:
                    ambiguous[duplicate] = tuple(options)

        for i in to_promote:
            names[i] = qualified_names[i]
            promoted[i] = True

        first_round = False

    return tuple(names), ambiguous


def validate_duckdb_regex(
    pattern: str,
    *,
    connection: Any,
    options: str = "c",
) -> None:
    """Validate that DuckDB/RE2 can compile a regex pattern."""
    try:
        connection.execute(
            "SELECT regexp_matches(?, ?, ?)",
            ["this is a harmless regex validation string", pattern, options],
        ).fetchone()
    except Exception as exc:
        if type(exc).__name__ == "InvalidInputException":
            raise DuckDBRegexValidationError(str(exc)) from exc
        raise
