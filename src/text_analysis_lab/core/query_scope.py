"""An operation-local naming scope shared by discovery and materialization.

This is deliberately a thin wrapper. QueryEngine remains the sole authority
for column identities and SQL, and SourceRequest remains the durable contract
for translator input. Projection is specified independently for each query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.operator import ColumnRequest, SourceRequest
from text_analysis_lab.core.types import MetadataMode

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_SCOPE_KEYS = frozenset({"include_data", "metadata_mode", "with_artifacts"})


@dataclass(frozen=True)
class QueryScope:
    """Bind one artifact and naming universe, not a fixed output projection."""

    artifact: BaseArtifact
    include_data: bool = True
    metadata_mode: MetadataMode = "none"
    with_artifacts: Any = None

    def _scope_kwargs(self) -> dict[str, Any]:
        # Omit defaults so older narrow query test doubles continue to work.
        # Actual query methods supply the same default values.
        kwargs: dict[str, Any] = {"metadata_mode": self.metadata_mode}
        if not self.include_data:
            kwargs["include_data"] = False
        if self.with_artifacts is not None:
            kwargs["with_artifacts"] = self.with_artifacts
        return kwargs

    def columns(self) -> dict[str, Any]:
        """Discover canonical output labels in this exact naming scope."""
        return self.artifact.query_columns(**self._scope_kwargs())

    def query(self, **projection_and_execution: Any) -> Any:
        """Execute with this scope; each call may select different columns."""
        unexpected = _SCOPE_KEYS.intersection(projection_and_execution)
        if unexpected:
            raise TypeError(
                "QueryScope fixes naming-scope arguments; do not override "
                f"{sorted(unexpected)} on query()."
            )
        return self.artifact.query(
            **self._scope_kwargs(), **projection_and_execution
        )

    def source_request(
        self, *, columns: ColumnRequest, **request_options: Any
    ) -> SourceRequest:
        """Build a durable SourceRequest with this scope and a projection."""
        if self.with_artifacts is not None:
            raise ValueError(
                "SourceRequest cannot express with_artifacts; use query() "
                "for attached sources."
            )
        unexpected = _SCOPE_KEYS.intersection(request_options)
        if unexpected:
            raise TypeError(
                "QueryScope fixes naming-scope arguments; do not override "
                f"{sorted(unexpected)} on source_request()."
            )
        artifact_type = request_options.pop("artifact_type", self.artifact.artifact_type)
        return SourceRequest(
            artifact_type=artifact_type,
            columns=columns,
            include_data=self.include_data,
            metadata_mode=self.metadata_mode,
            **request_options,
        )
