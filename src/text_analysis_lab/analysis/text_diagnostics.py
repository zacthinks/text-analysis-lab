"""Lightweight text-cleaning diagnostics for course workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


@dataclass(frozen=True)
class TextDiagnosticsResult:
    row_count: int
    text_field: str
    missing_text_count: int
    empty_text_count: int
    nonempty_text_count: int
    unique_nonempty_text_count: int
    duplicate_group_count: int
    duplicate_row_count: int
    duplicate_excess_count: int
    compared_to_artifact_id: str | None = None
    key_set_preserved: bool | None = None
    key_order_preserved: bool | None = None
    missing_keys_from_current: int | None = None
    extra_keys_in_current: int | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def text_diagnostics(
    artifact: BaseArtifact,
    *,
    text_field: str = "text",
    compare_to: BaseArtifact | str | None = None,
    strip: bool = True,
) -> TextDiagnosticsResult:
    """Report empty/duplicate text and optional stable-key preservation checks.

    Diagnostics never remove or rewrite observations. ``compare_to`` is useful
    after a row-preserving cleaning step: it checks both key-set equality and
    whether the current artifact retains the same canonical key order.
    """
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string.")
    frame = artifact.query(
        key_columns=True,
        data_columns=[text_field],
        metadata_columns=False,
        form="table",
        include_position=True,
    )
    if not isinstance(frame, pd.DataFrame) or text_field not in frame.columns:
        raise ArtifactError(f"Could not materialize text field {text_field!r}.")
    frame = frame.sort_values("_position", kind="stable").reset_index(drop=True)
    raw = frame[text_field]
    missing = raw.isna()
    strings = raw.fillna("").astype(str)
    comparable = strings.str.strip() if strip else strings
    empty = (~missing) & comparable.eq("")
    nonmissing = ~missing
    nonempty = nonmissing & ~empty

    observed = comparable.loc[nonempty]
    counts = observed.value_counts(dropna=False)
    duplicate_counts = counts[counts > 1]
    duplicate_groups = len(duplicate_counts)
    duplicate_rows = int(duplicate_counts.sum()) if duplicate_groups else 0
    duplicate_excess = int((duplicate_counts - 1).sum()) if duplicate_groups else 0

    compared_id: str | None = None
    key_set_preserved: bool | None = None
    key_order_preserved: bool | None = None
    missing_keys: int | None = None
    extra_keys: int | None = None
    if compare_to is not None:
        other = artifact.project.get_artifact(compare_to)
        current_keys = tuple(str(value) for value in artifact.primary_key)
        other_keys = tuple(str(value) for value in other.primary_key)
        if current_keys != other_keys:
            raise ArtifactError(
                "Row-preservation comparison requires identical primary-key columns; "
                f"current={current_keys}, compare_to={other_keys}."
            )
        other_frame = other.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            form="table",
            include_position=True,
        )
        if not isinstance(other_frame, pd.DataFrame):
            raise ArtifactError("Could not materialize comparison artifact keys.")
        other_frame = other_frame.sort_values("_position", kind="stable").reset_index(
            drop=True
        )
        key_columns = list(current_keys)
        current_order = [
            tuple(row)
            for row in frame.loc[:, key_columns].itertuples(index=False, name=None)
        ]
        other_order = [
            tuple(row)
            for row in other_frame.loc[:, key_columns].itertuples(
                index=False, name=None
            )
        ]
        current_set = set(current_order)
        other_set = set(other_order)
        key_set_preserved = current_set == other_set
        key_order_preserved = current_order == other_order
        missing_keys = len(other_set - current_set)
        extra_keys = len(current_set - other_set)
        compared_id = str(other.artifact_id)

    return TextDiagnosticsResult(
        row_count=len(frame),
        text_field=text_field,
        missing_text_count=int(missing.sum()),
        empty_text_count=int(empty.sum()),
        nonempty_text_count=int(nonempty.sum()),
        unique_nonempty_text_count=int(observed.nunique(dropna=False)),
        duplicate_group_count=duplicate_groups,
        duplicate_row_count=duplicate_rows,
        duplicate_excess_count=duplicate_excess,
        compared_to_artifact_id=compared_id,
        key_set_preserved=key_set_preserved,
        key_order_preserved=key_order_preserved,
        missing_keys_from_current=missing_keys,
        extra_keys_in_current=extra_keys,
    )
