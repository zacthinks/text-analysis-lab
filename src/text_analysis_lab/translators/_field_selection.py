"""Scoped selection of data and metadata labels returned by query_columns().

Column names are established by the complete query naming universe before
selection. Translators must never infer a namespace by parsing a label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.types import MetadataMode


def resolve_text_fields(
    artifact: Any,
    fields: Sequence[str],
    *,
    metadata_mode: MetadataMode,
    translator: str,
) -> dict[str, str]:
    """Resolve exact displayed source labels to their data/metadata namespaces."""
    if metadata_mode not in {"none", "local", "full"}:
        raise ValueError("metadata_mode must be 'none', 'local', or 'full'.")
    inventory = artifact.query_columns(metadata_mode=metadata_mode)
    choices: dict[str, str] = {}
    for column in inventory.get("columns", []):
        if not isinstance(column, Mapping):
            continue
        namespace = column.get("namespace")
        if namespace not in {"data", "metadata"}:
            continue
        label = str(column.get("output_name"))
        if label in choices:
            raise OperatorError(
                f"{translator}: query scope contains repeated output label {label!r}."
            )
        choices[label] = namespace

    selected: dict[str, str] = {}
    for label in fields:
        if label not in choices:
            raise OperatorError(
                f"{translator} field {label!r} is unavailable with "
                f"metadata_mode={metadata_mode!r}. "
                f"Available fields: {sorted(choices)}."
            )
        selected[label] = choices[label]
    return selected



def canonical_key_output_names(
    artifact: Any, *, metadata_mode: MetadataMode, translator: str
) -> list[str]:
    """Return query-scope key labels in structural primary-key order."""
    inventory = artifact.query_columns(metadata_mode=metadata_mode)
    key_columns = [
        col for col in inventory.get("columns", ())
        if col.get("namespace") == "key"
    ]
    raw_keys = [str(value) for value in artifact.primary_key]
    if (
        len(key_columns) != len(raw_keys)
        or any(str(col.get("base_name")) != name
               for col, name in zip(key_columns, raw_keys, strict=True))
    ):
        raise OperatorError(
            f"{translator}: query scope has an inconsistent structural key schema."
        )
    return [str(col["output_name"]) for col in key_columns]


def structural_key_frame(
    frame: Any,
    primary_key: Sequence[str],
    *,
    output_names: Sequence[str] | None,
    translator: str,
) -> Any:
    """Extract scoped query keys, restoring structural key names for output."""
    raw = [str(name) for name in primary_key]
    names = raw if output_names is None else [str(name) for name in output_names]
    if len(names) != len(raw) or len(set(names)) != len(names):
        raise OperatorError(f"{translator}: invalid source key label mapping.")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise OperatorError(
            f"{translator}: source batch is missing key column(s) {missing}."
        )
    keys = frame.loc[:, names].copy().reset_index(drop=True)
    keys.columns = raw
    return keys
