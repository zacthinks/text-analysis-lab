"""Custom exceptions for TextAnalysisLab (TeAL)."""

from __future__ import annotations


class TeALError(Exception):
    """Base exception for TextAnalysisLab."""


class OperatorError(TeALError):
    """Base exception for operator-related errors."""


class OperatorNotFittedError(OperatorError):
    """Raised when an operator that requires fitting is used before fitting."""


class FrozenOperatorError(OperatorError):
    """Raised when attempting to mutate a committed/frozen operator."""


class OperatorNotFoundError(OperatorError):
    """Raised when an operator ID cannot be found in the project."""


class UnsupportedExecutionModeError(OperatorError):
    """Raised when an operator declares an unsupported execution mode."""


class FunctionMapperError(OperatorError):
    """Raised when a FunctionMapper cannot normalize or validate user function output."""


class FunctionSerializationError(OperatorError):
    """Raised when a user-supplied function cannot be saved or reloaded."""


class LineageError(OperatorError):
    """Raised when declared artifact lineage is invalid or violated."""


class OperationProvenanceError(OperatorError):
    """Raised when operation-run creation metadata is invalid."""


class OutputSpecError(OperatorError):
    """Raised when an operator declares invalid output specifications."""


class ArtifactError(TeALError):
    """Base exception for artifact-related errors."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact ID cannot be found in the project."""


class InvalidAliasError(ArtifactError):
    """Raised when an artifact alias is not valid."""


class AliasBundleError(InvalidAliasError):
    """Raised when an idempotent alias bundle is malformed or only partly present."""


class AliasOverwriteBlockedError(InvalidAliasError):
    """Raised when overwrite would break live lineage or other alias references."""


class QueryError(ArtifactError):
    """Raised when an artifact or project query cannot be constructed or executed."""


class UnsupportedArtifactTypeError(ArtifactError):
    """Raised when an artifact descriptor names an unsupported artifact type."""


class UnsupportedArtifactOperationError(ArtifactError):
    """Raised when an artifact does not support a requested operation."""


class IncompleteArtifactError(ArtifactError):
    """Raised when an operation requires a complete artifact."""


class MissingPrimaryKeyError(ArtifactError):
    """Raised when a primary key is missing or malformed."""


class DuplicatePrimaryKeyError(ArtifactError):
    """Raised when artifact keys contain duplicate primary-key values."""


class MissingDataComponentError(ArtifactError):
    """Raised when an artifact has no owned data component and cannot inherit one."""


class DataInheritanceError(ArtifactError):
    """Raised when missing artifact data cannot be resolved through lineage."""


class MissingDependencyError(TeALError):
    """Raised when an optional dependency is needed but unavailable."""


class MetadataError(TeALError):
    """Base exception for metadata-related errors."""


class MissingMetadataError(MetadataError):
    """Raised when requested metadata is unavailable."""


class MetadataAggregationError(MetadataError):
    """Raised when metadata aggregation for recomposition fails."""


class DuckDBRegexValidationError(ValueError):
    """Raised when DuckDB/RE2 cannot compile a regex pattern."""
