from __future__ import annotations

from enum import Enum
from typing import Literal, Final
from collections.abc import Sequence

MetadataMode = Literal["none", "local", "full"]

LineageMode = Literal[
    "preserved_key",
    "extended_key",
    "reduced_key",
    "span_key",
    "merged_key",
    "joined_key",
    "new_key",
]

QueryForm = Literal["table", "records", "native", "single"]

StreamingMode = Literal["auto", "arrow", "paged"]

ArtifactStatus = Literal["incomplete", "complete", "failed"]

OperationType = Literal["import", "split", "subset", "merge", "join", "translate"]

OperationStatus = Literal["incomplete", "complete", "failed"]

OperatorSnapshotStatus = Literal["pending", "serialized", "failed"]

MemoTargetType = Literal["project", "artifact", "operator", "operation", "standalone"]

StructuralColumn = Literal["_position", "_batch", "_row_offset"]

ColumnSelect = bool | str | Sequence[str]

DEFAULT_SOURCE_LABEL: Final = "source"
DEFAULT_OUTPUT_LABEL: Final = "output"

class ArtifactType(str, Enum):
    TABLE = "table"
    JSONL = "jsonl"
    SPARSE_MATRIX = "sparse_matrix"
    DENSE_MATRIX = "dense_matrix"
    OTHER = "other"
