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
