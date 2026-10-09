"""Public API for Text Analysis Lab (TeAL).

TeAL provides persistent, lineage-aware artifacts and operators for reproducible
computational text-analysis workflows.
"""

from importlib.metadata import PackageNotFoundError, version as _distribution_version

try:
    __version__ = _distribution_version("text-analysis-lab")
except PackageNotFoundError:
    # Source-tree imports outside an installed development environment have no
    # distribution metadata. The authoritative version remains pyproject.toml.
    __version__ = "0+unknown"

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
    ExecutionCapabilities,
    InputBatch,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.pipeline import (
    Pipeline,
    PipelineCapabilities,
    PipelineCapabilityIssue,
    PipelinePort,
)
from text_analysis_lab.core.project import Project
from text_analysis_lab.linguistics import (
    SemanticHeadRules,
    export_default_semantic_head_rules,
    get_default_semantic_head_rules,
    load_semantic_head_rules,
)

from . import analysis, dictionaries, translators, visualization

__all__ = [
    "__version__",
    "AggregateField",
    "BaseArtifact",
    "BaseOperator",
    "BaseTranslator",
    "BatchResult",
    "ColumnRequest",
    "ConcatReducer",
    "DenseMatrixArtifact",
    "ExecutionCapabilities",
    "InputBatch",
    "JsonlArtifact",
    "LiteralValue",
    "OtherArtifact",
    "OutputSpec",
    "Pipeline",
    "PipelineCapabilities",
    "PipelineCapabilityIssue",
    "PipelinePort",
    "Project",
    "SemanticHeadRules",
    "SourceRequest",
    "SparseMatrixArtifact",
    "TableArtifact",
    "TranslationRequest",
    "agg",
    "analysis",
    "concat",
    "dictionaries",
    "export_default_semantic_head_rules",
    "get_default_semantic_head_rules",
    "literal",
    "load_semantic_head_rules",
    "load_artifact",
    "translators",
    "visualization",
]
