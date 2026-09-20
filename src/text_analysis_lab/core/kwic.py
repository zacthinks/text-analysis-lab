"""Keyword-in-context search and result helpers for TeAL artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.types import (
    ColumnSelect,
    MetadataMode,
    StreamingMode,
    StructuralColumn,
)
from text_analysis_lab.core.utils import str_keys

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

STRUCTURAL_COLUMNS: frozenset[str] = frozenset(get_args(StructuralColumn))


def _kwic_pattern_body(
    pattern: str,
    *,
    valuetype: Literal["fixed", "regex"],
    enforce_word_boundary: bool,
) -> str:
    """Return the regex body used for Python matching and SQL prefiltering."""
    if valuetype == "fixed":
        body = re.escape(pattern)
    elif valuetype == "regex":
        body = pattern
    else:
        raise ValueError("valuetype must be 'fixed' or 'regex'.")

    if enforce_word_boundary:
        body = r"\b(?:" + body + r")\b"
    return body


def _compile_kwic_regex(
    pattern: str,
    *,
    valuetype: Literal["fixed", "regex"],
    case_sensitive: bool,
    enforce_word_boundary: bool,
) -> re.Pattern[str]:
    """Compile the Python KWIC matcher.

    By default, KWIC searches are bounded by word boundaries in both fixed and
    regex mode. Regex mode can opt out with ``enforce_word_boundary=False`` for
    substring, wildcard, or custom-boundary searches.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(
        _kwic_pattern_body(
            pattern,
            valuetype=valuetype,
            enforce_word_boundary=enforce_word_boundary,
        ),
        flags,
    )


def _is_word_char(value: str) -> bool:
    return value in {"_", "-", "'"} or value.isalnum()


def _context_slice(
    text: str,
    boundary: int,
    n: int,
    *,
    direction: Literal["previous", "next"],
) -> str:
    """Return up to ``n`` context tokens adjacent to ``boundary``.

    Alphanumeric/underscore runs count as one token. Punctuation marks count as
    individual tokens. Whitespace separates tokens but is not itself counted.
    The scanner stops as soon as it has consumed the requested number of
    context tokens and returns the original text slice.
    """
    if n <= 0:
        return ""

    if direction == "previous":
        cursor = max(0, min(boundary, len(text)))
        tokens_seen = 0
        start = cursor

        while cursor > 0 and tokens_seen < n:
            while cursor > 0 and text[cursor - 1].isspace():
                cursor -= 1
            if cursor <= 0:
                break

            if _is_word_char(text[cursor - 1]):
                while cursor > 0 and _is_word_char(text[cursor - 1]):
                    cursor -= 1
            else:
                cursor -= 1

            start = cursor
            tokens_seen += 1

        return text[start:boundary].strip()

    if direction == "next":
        cursor = max(0, min(boundary, len(text)))
        tokens_seen = 0
        end = cursor
        length = len(text)

        while cursor < length and tokens_seen < n:
            while cursor < length and text[cursor].isspace():
                cursor += 1
            if cursor >= length:
                break

            if _is_word_char(text[cursor]):
                while cursor < length and _is_word_char(text[cursor]):
                    cursor += 1
            else:
                cursor += 1

            end = cursor
            tokens_seen += 1

        return text[boundary:end].strip()

    raise ValueError("direction must be 'previous' or 'next'.")


def _normalize_search_columns(value: ColumnSelect | None) -> list[str] | None:
    if value is None:
        return None
    if value is True:
        return None
    if value is False:
        raise ValueError("search_columns=False is not meaningful for KWIC.")
    if isinstance(value, str):
        return [value]
    return [str(col) for col in value]


def _string_search_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in frame.columns:
        name = str(col)
        if name in STRUCTURAL_COLUMNS:
            continue
        series = frame[col]
        if pd.api.types.is_string_dtype(series):
            columns.append(name)
            continue
        non_null = series.dropna()
        if (
            not non_null.empty
            and non_null.map(lambda value: isinstance(value, str)).any()
        ):
            columns.append(name)
    return columns


def _explicit_search_columns(
    frame: pd.DataFrame, requested: Sequence[str]
) -> list[str]:
    available = {str(col) for col in frame.columns}
    missing = [col for col in requested if col not in available]
    if missing:
        raise ArtifactError(
            f"Requested KWIC search column(s) are not present in the query result: {missing}. "
            f"Available columns: {list(frame.columns)}."
        )
    return list(dict.fromkeys(str(col) for col in requested))


class KWICResult(list[dict[str, Any]]):
    """List-like keyword-in-context result with display helpers."""

    def __init__(
        self,
        hits: Iterable[dict[str, Any]] = (),
        *,
        display_limit: int = 20,
    ) -> None:
        super().__init__(hits)
        self.display_limit = display_limit

    def to_frame(
        self, *, expand_key: bool = True, expand_metadata: bool = True
    ) -> pd.DataFrame:
        """Return KWIC hits as a flat DataFrame.

        KWIC is a table-like result, so requested metadata is expanded by default.
        Metadata fields keep their original column names unless they collide with
        key or KWIC columns, in which case the metadata column is prefixed with
        ``metadata_``.
        """
        core_fields = (
            "_position",
            "field",
            "pre",
            "match",
            "post",
            "match_start",
            "match_end",
            "match_index",
        )
        rows: list[dict[str, Any]] = []
        for hit in self:
            row: dict[str, Any] = {}

            key = hit.get("key", {})
            if expand_key and isinstance(key, dict):
                for key_name, key_value in key.items():
                    out_name = (
                        key_name if key_name not in core_fields else f"key_{key_name}"
                    )
                    row[out_name] = key_value
            else:
                row["key"] = key

            for field in core_fields:
                if field in hit:
                    row[field] = hit[field]

            if expand_metadata and isinstance(hit.get("metadata"), dict):
                for meta_key, meta_value in hit["metadata"].items():
                    out_name = (
                        meta_key if meta_key not in row else f"metadata_{meta_key}"
                    )
                    row[out_name] = meta_value

            rows.append(row)
        return pd.DataFrame(rows)

    def _format_lines(
        self,
        hits: Sequence[dict[str, Any]],
        *,
        max_rows: int | None,
        context_width: int,
        key: bool,
    ) -> str:
        total = len(self)
        shown_hits = list(hits if max_rows is None else hits[:max_rows])

        header = (
            f"#> Keyword-in-context with {total} match{'es' if total != 1 else ''}."
        )
        if max_rows is not None and total > len(shown_hits):
            header += f" Showing {len(shown_hits)}."

        if not shown_hits:
            return header

        def trim_left(value: Any) -> str:
            text = str(value)
            return text if len(text) <= context_width else "..." + text[-context_width:]

        def trim_right(value: Any) -> str:
            text = str(value)
            return text if len(text) <= context_width else text[:context_width] + "..."

        key_labels: list[str] = []
        pre_values: list[str] = []
        match_values: list[str] = []
        post_values: list[str] = []

        for hit in shown_hits:
            hit_key = hit.get("key", {})
            if key and isinstance(hit_key, dict):
                key_label = "[" + ", ".join(str(hit_key[col]) for col in hit_key) + "]"
            elif key:
                key_label = f"[{hit_key}]"
            else:
                key_label = ""
            key_labels.append(key_label)
            pre_values.append(trim_left(hit.get("pre", "")))
            match_values.append(str(hit.get("match", "")))
            post_values.append(trim_right(hit.get("post", "")))

        key_width = max([len(value) for value in key_labels], default=0) if key else 0
        pre_width = max([len(value) for value in pre_values], default=3)
        match_width = max([len(value) for value in match_values], default=5)

        lines = [header]
        for key_label, pre, match, post in zip(
            key_labels, pre_values, match_values, post_values, strict=True
        ):
            prefix = f"#> {key_label:>{key_width}}  " if key else "#> "
            lines.append(
                f"{prefix}{pre:>{pre_width}} | {match:^{match_width}} | {post}"
            )
        return "\n".join(lines)

    def display(
        self,
        *,
        max_rows: int | None = None,
        context_width: int = 60,
        key: bool = True,
    ) -> str:
        """Return an aligned plain-text KWIC concordance display."""
        if max_rows is None:
            max_rows = self.display_limit
        return self._format_lines(
            list(self),
            max_rows=max_rows,
            context_width=context_width,
            key=key,
        )

    def sample_display(
        self,
        n: int = 20,
        *,
        random_state: int | None = None,
        context_width: int = 60,
        key: bool = True,
    ) -> str:
        """Return an aligned display for a reproducible random sample of KWIC hits."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        if not self or n == 0:
            return self._format_lines(
                [], max_rows=0, context_width=context_width, key=key
            )
        rng = np.random.default_rng(random_state)
        sample_n = min(int(n), len(self))
        indices = rng.choice(len(self), size=sample_n, replace=False).tolist()
        sampled = [self[int(index)] for index in indices]
        return self._format_lines(
            sampled,
            max_rows=None,
            context_width=context_width,
            key=key,
        )

    def __repr__(self) -> str:
        return self.display(max_rows=self.display_limit)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        p.text(self.__repr__())

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(self.__repr__())}</pre>"


def keyword_in_context(
    artifact: BaseArtifact,
    pattern: str,
    *,
    window: int = 5,
    before: int | None = None,
    after: int | None = None,
    valuetype: Literal["fixed", "regex"] = "fixed",
    case_sensitive: bool = False,
    enforce_word_boundary: bool = True,
    key_columns: ColumnSelect = True,
    data_columns: ColumnSelect = True,
    metadata_columns: ColumnSelect = False,
    metadata_mode: MetadataMode = "none",
    search_columns: ColumnSelect | None = None,
    where: str | None = None,
    order_by: str | Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    sample_n: int | None = None,
    sample_frac: float | None = None,
    random_state: int | None = None,
    limit: int | None = None,
    batch_size: int = 100_000,
    streaming_mode: StreamingMode = "auto",
    target_matches: int | None = None,
) -> KWICResult:
    """Return keyword-in-context hits from the normal artifact query stream.

    KWIC is intentionally a thin wrapper around ``artifact.iter_table_batches``.
    The query arguments select which columns are returned. ``search_columns``
    names the returned columns to search. If ``search_columns`` is ``None``,
    KWIC searches every string-like non-structural column in each batch.

    ``target_matches`` is a batch-level stopping target: when set, KWIC stops
    after finishing the first batch that brings the accumulated hit count to at
    least that value.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty.")
    if before is None:
        before = window
    if after is None:
        after = window
    if before < 0 or after < 0:
        raise ValueError("KWIC context windows must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if target_matches is not None and int(target_matches) < 0:
        raise ValueError("target_matches must be non-negative.")
    if target_matches == 0:
        return KWICResult()

    regex = _compile_kwic_regex(
        pattern,
        valuetype=valuetype,
        case_sensitive=case_sensitive,
        enforce_word_boundary=enforce_word_boundary,
    )
    requested_search_columns = _normalize_search_columns(search_columns)

    query_info = artifact.query_columns(metadata_mode=metadata_mode)
    key_output_columns = {str(col) for col in query_info.get("key", [])}
    metadata_output_columns = {str(col) for col in query_info.get("metadata", [])}

    hits = KWICResult()
    saw_any_row = False
    saw_searchable_text = False

    for batch in artifact.iter_table_batches(
        batch_size=int(batch_size),
        key_columns=key_columns,
        data_columns=data_columns,
        metadata_columns=metadata_columns,
        metadata_mode=metadata_mode,
        where=where,
        order_by=order_by,
        positions=positions,
        sample_n=sample_n,
        sample_frac=sample_frac,
        random_state=random_state,
        limit=limit,
        include_position=True,
        streaming_mode=streaming_mode,
    ):
        if batch.empty:
            continue
        saw_any_row = True

        if requested_search_columns is None:
            batch_search_columns = _string_search_columns(batch)
        else:
            batch_search_columns = _explicit_search_columns(
                batch, requested_search_columns
            )

        if not batch_search_columns:
            continue

        for raw_record in batch.to_dict(orient="records"):
            record = str_keys(raw_record)
            key = {
                column: record.get(column)
                for column in batch.columns
                if str(column) in key_output_columns and str(column) in record
            }
            row_metadata = {
                column: record.get(column)
                for column in batch.columns
                if str(column) in metadata_output_columns and str(column) in record
            }

            for field in batch_search_columns:
                value = record.get(field)
                if not isinstance(value, str):
                    continue
                saw_searchable_text = True

                for match_index, match in enumerate(regex.finditer(value)):
                    hit = {
                        "key": key,
                        "field": field,
                        "pre": _context_slice(
                            value, match.start(), before, direction="previous"
                        ),
                        "match": match.group(0),
                        "post": _context_slice(
                            value, match.end(), after, direction="next"
                        ),
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "match_index": match_index,
                        "_position": record.get("_position"),
                    }
                    if row_metadata:
                        hit["metadata"] = row_metadata
                    hits.append(hit)

        if target_matches is not None and len(hits) >= int(target_matches):
            break

    if saw_any_row and not saw_searchable_text:
        raise ArtifactError(
            f"KWIC found no string-like searchable columns in the query result for artifact "
            f"{artifact.artifact_id}. Pass search_columns with columns returned by query()."
        )

    return hits
