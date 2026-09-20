"""Public API for Text Analysis Lab (TeAL).

TeAL provides persistent, lineage-aware artifacts and operators for reproducible
computational text-analysis workflows.
"""

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
from text_analysis_lab.core.project import Project

from . import analysis, dictionaries, translators, visualization

__all__ = [
    "AggregateField",
    "BaseArtifact",
    "BaseOperator",
    "BaseTranslator",
    "BatchResult",
    "ColumnRequest",
    "ConcatReducer",
    "DenseMatrixArtifact",
    "InputBatch",
    "JsonlArtifact",
    "LiteralValue",
    "OtherArtifact",
    "OutputSpec",
    "Project",
    "SourceRequest",
    "SparseMatrixArtifact",
    "TableArtifact",
    "TranslationRequest",
    "agg",
    "analysis",
    "concat",
    "dictionaries",
    "literal",
    "load_artifact",
    "translators",
    "visualization",
]
