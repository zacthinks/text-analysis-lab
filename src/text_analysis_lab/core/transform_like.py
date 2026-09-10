"""Replay a frozen matrix-producing TeAL pipeline on new text without artifacts.

This is intentionally narrower than ordinary translation.  It exists for
runtime consumers such as externally backed GeCo geometries that need to embed
one query or a few newly composed texts in the *same fitted representation* as
an existing TeAL matrix artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.types import ArtifactType

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project

_MATRIX_TYPES = {ArtifactType.SPARSE_MATRIX, ArtifactType.DENSE_MATRIX}


class TransformLikeError(OperatorError):
    """Raised when an artifact's frozen lineage cannot transform new text."""


def can_transform_texts_like(
    project: "Project",
    artifact: BaseArtifact | str,
    *,
    query: bool = False,
) -> bool:
    """Return whether TeAL can replay ``artifact``'s frozen lineage on new text."""
    try:
        _build_transform_plan(project, project.get_artifact(artifact), query=query)
    except (ArtifactError, OperatorError, TransformLikeError, KeyError, ValueError):
        return False
    return True


def transform_texts_like(
    project: "Project",
    artifact: BaseArtifact | str,
    texts: Sequence[str],
    *,
    query: bool = False,
) -> Any:
    """Transform raw texts into the representation of an existing matrix artifact.

    The existing artifact is not modified and no new TeAL artifact is written.
    Every fitted operator is loaded from its durable snapshot and executed with
    the operation parameters that produced the corresponding lineage stage.
    """
    target = project.get_artifact(artifact)
    if target.artifact_type not in _MATRIX_TYPES:
        raise TransformLikeError(
            "transform_texts_like(...) requires a dense_matrix or sparse_matrix "
            f"artifact; got {target.artifact_type.value!r}."
        )
    normalized = ["" if value is None else str(value) for value in texts]
    plan = _build_transform_plan(project, target, query=query)
    values: Any = normalized
    for stage in plan:
        operator = stage["operator"]
        params = stage["params"]
        kind = stage["kind"]
        if kind == "texts":
            values = operator.transform_external_texts(
                values,
                query=query,
                params=params,
            )
        else:
            values = operator.transform_external_matrix(
                values,
                query=query,
                params=params,
            )
    _validate_transformed_rows(values, len(normalized), target)
    return values


def _build_transform_plan(
    project: "Project",
    artifact: BaseArtifact,
    *,
    query: bool,
) -> list[dict[str, Any]]:
    if artifact.artifact_type not in _MATRIX_TYPES:
        raise TransformLikeError(
            f"Artifact {artifact.artifact_id!r} is not a matrix representation."
        )

    operation = project.operation_for_artifact(artifact)
    if operation is None:
        raise TransformLikeError(
            f"Artifact {artifact.artifact_id!r} has no producing operation to replay."
        )
    operation_id = str(operation["operation_id"])
    outputs = project.operation_outputs(operation_id)
    output_edge = next(
        (row for row in outputs if str(row.get("artifact_id")) == artifact.artifact_id),
        None,
    )
    if output_edge is None:
        raise TransformLikeError(
            f"Operation {operation_id!r} does not record artifact {artifact.artifact_id!r} "
            "as an output."
        )

    descriptor = _operation_descriptor(project, operation_id)
    output_label = str(output_edge.get("output_label"))
    spec = dict(descriptor.get("output_specs", {})).get(output_label)
    if isinstance(spec, Mapping) and str(spec.get("lineage_mode")) != "preserved_key":
        raise TransformLikeError(
            "Only preserved-key representation outputs can be replayed on new documents; "
            f"artifact {artifact.artifact_id!r} is output {output_label!r} with "
            f"lineage_mode={spec.get('lineage_mode')!r}."
        )

    sources = project.operation_sources(operation_id)
    if len(sources) != 1:
        raise TransformLikeError(
            "New-text replay currently supports only single-source representation "
            f"pipelines; operation {operation_id!r} has {len(sources)} sources."
        )
    source = project.get_artifact(str(sources[0]["source_artifact_id"]))
    operator = project.get_operator(str(operation["operator_id"]))
    raw_params = descriptor.get("request", {}).get("params", {})
    if not isinstance(raw_params, Mapping):
        raise TransformLikeError(
            f"Operation {operation_id!r} has invalid persisted request parameters."
        )
    params = dict(operator.deserialize_operation_params(raw_params))

    if source.artifact_type == ArtifactType.TABLE:
        method = getattr(operator, "transform_external_texts", None)
        if not callable(method):
            raise TransformLikeError(
                f"Frozen operator {operator.__class__.__name__} cannot transform new raw text."
            )
        if not _operator_allows_external_transform(operator, query=query, input_kind="texts"):
            raise TransformLikeError(
                f"Frozen operator {operator.__class__.__name__} does not support "
                f"{'query' if query else 'new-text'} replay for this configuration."
            )
        return [{"kind": "texts", "operator": operator, "params": params}]

    if source.artifact_type in _MATRIX_TYPES:
        method = getattr(operator, "transform_external_matrix", None)
        if not callable(method):
            raise TransformLikeError(
                f"Frozen operator {operator.__class__.__name__} cannot replay new matrix rows."
            )
        if not _operator_allows_external_transform(operator, query=query, input_kind="matrix"):
            raise TransformLikeError(
                f"Frozen operator {operator.__class__.__name__} does not support "
                f"{'query' if query else 'new-text'} replay for this configuration."
            )
        return [
            *_build_transform_plan(project, source, query=query),
            {"kind": "matrix", "operator": operator, "params": params},
        ]

    raise TransformLikeError(
        "New-text replay requires a representation lineage rooted in a table and "
        f"continuing through matrix artifacts; source {source.artifact_id!r} has type "
        f"{source.artifact_type.value!r}."
    )


def _operator_allows_external_transform(
    operator: Any,
    *,
    query: bool,
    input_kind: str,
) -> bool:
    checker = getattr(operator, "supports_external_transform", None)
    if callable(checker):
        return bool(checker(query=query, input_kind=input_kind))
    return True


def _operation_descriptor(project: "Project", operation_id: str) -> Mapping[str, Any]:
    path = project.storage.operation_descriptor_path(operation_id)
    if not path.exists():
        raise TransformLikeError(
            f"Operation {operation_id!r} has no durable operation descriptor."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TransformLikeError(
            f"Operation descriptor for {operation_id!r} must contain a mapping."
        )
    return payload


def _validate_transformed_rows(values: Any, expected: int, artifact: BaseArtifact) -> None:
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != int(expected):
        raise TransformLikeError(
            f"Replaying artifact {artifact.artifact_id!r} expected {expected} transformed "
            f"row(s), received shape={shape!r}."
        )
    expected_columns = len(artifact.get_data_columns())
    if int(shape[1]) != expected_columns:
        raise TransformLikeError(
            f"Replaying artifact {artifact.artifact_id!r} expected {expected_columns} "
            f"feature columns, received shape={shape!r}."
        )
