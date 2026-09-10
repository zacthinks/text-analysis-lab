"""Public API for Text Analysis Lab (TeAL).

TeAL provides persistent, lineage-aware artifacts and operators for reproducible
computational text-analysis workflows.
"""

from text_analysis_lab.core.project import Project
from text_analysis_lab.core.aggregate import (
    AggregateField,
    ConcatReducer,
    LiteralValue,
    agg,
    concat,
    literal,
)
from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.artifact_subclasses import (
    DenseMatrixArtifact,
    JsonlArtifact,
    OtherArtifact,
    SparseMatrixArtifact,
    TableArtifact,
    load_artifact,
)
from text_analysis_lab.core.operator import (
    BaseOperator,
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from . import analysis, dictionaries, translators, visualization

__all__ = [
    "Project",
    "agg",
    "literal",
    "concat",
    "AggregateField",
    "LiteralValue",
    "ConcatReducer",
    "BaseArtifact",
    "TableArtifact",
    "JsonlArtifact",
    "SparseMatrixArtifact",
    "DenseMatrixArtifact",
    "OtherArtifact",
    "load_artifact",
    "BaseOperator",
    "BaseTranslator",
    "ColumnRequest",
    "SourceRequest",
    "TranslationRequest",
    "InputBatch",
    "BatchResult",
    "OutputSpec",
    "analysis",
    "dictionaries",
    "translators",
    "visualization",
]
