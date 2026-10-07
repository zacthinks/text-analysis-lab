"""Private frozen-representation replay for TeAL-backed GeCo geometries.

This is intentionally narrower than ordinary translation.  It exists for
runtime consumers such as externally backed GeCo geometries that need to embed
one newly composed texts in the *same fitted representation* as
an existing TeAL matrix artifact.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import BaseTranslator
from text_analysis_lab.core.types import ArtifactType

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project

_MATRIX_TYPES = {ArtifactType.SPARSE_MATRIX, ArtifactType.DENSE_MATRIX}


class _RepresentationReplayError(OperatorError):
    """Raised when a frozen representation lineage cannot replay new text."""


def _can_replay_texts_like(
    project: Project,
    artifact: BaseArtifact | str,
) -> bool:
    """Return whether TeAL can replay ``artifact``'s frozen lineage on new text."""
    try:
        _build_replay_plan(project, project.get_artifact(artifact))
    except (ArtifactError, OperatorError, _RepresentationReplayError, KeyError, ValueError):
        return False
    return True


def _replay_texts_like(
    project: Project,
    artifact: BaseArtifact | str,
    texts: Sequence[str],
) -> Any:
    """Replay raw texts into the representation of an existing matrix artifact.

    The existing artifact is not modified and no new TeAL artifact is written.
    Every fitted operator is loaded from its durable snapshot and executed with
    the operation parameters that produced the corresponding lineage stage.
    """
    target = project.get_artifact(artifact)
    if target.artifact_type not in _MATRIX_TYPES:
        raise _RepresentationReplayError(
            "Frozen representation replay requires a dense_matrix or sparse_matrix "
            f"artifact; got {target.artifact_type.value!r}."
        )
    normalized = ["" if value is None else str(value) for value in texts]
    plan = _build_replay_plan(project, target)
    values: Any = normalized
    for stage in plan:
        operator = stage["operator"]
        params = stage["params"]
        kind = stage["kind"]
        values = _replay_stage(
            operator,
            values,
            kind=kind,
            params=params,
        )
    _validate_replayed_rows(values, len(normalized), target)
    if isinstance(values, Mapping) and "values" in values:
        return values["values"]
    return values


def _build_replay_plan(
    project: Project,
    artifact: BaseArtifact,
    *
) -> list[dict[str, Any]]:
    if artifact.artifact_type not in _MATRIX_TYPES:
        raise _RepresentationReplayError(
            f"Artifact {artifact.artifact_id!r} is not a matrix representation."
        )

    operation = project.operation_for_artifact(artifact)
    if operation is None:
        raise _RepresentationReplayError(
            f"Artifact {artifact.artifact_id!r} has no producing operation to replay."
        )
    operation_id = str(operation["operation_id"])
    outputs = project.operation_outputs(operation_id)
    output_edge = next(
        (row for row in outputs if str(row.get("artifact_id")) == artifact.artifact_id),
        None,
    )
    if output_edge is None:
        raise _RepresentationReplayError(
            f"Operation {operation_id!r} does not record artifact {artifact.artifact_id!r} "
            "as an output."
        )

    descriptor = _operation_descriptor(project, operation_id)
    output_label = str(output_edge.get("output_label"))
    spec = dict(descriptor.get("output_specs", {})).get(output_label)
    if isinstance(spec, Mapping) and str(spec.get("lineage_mode")) != "preserved_key":
        raise _RepresentationReplayError(
            "Only preserved-key representation outputs can be replayed on new documents; "
            f"artifact {artifact.artifact_id!r} is output {output_label!r} with "
            f"lineage_mode={spec.get('lineage_mode')!r}."
        )

    sources = project.operation_sources(operation_id)
    if len(sources) != 1:
        raise _RepresentationReplayError(
            "New-text replay currently supports only single-source representation "
            f"pipelines; operation {operation_id!r} has {len(sources)} sources."
        )
    source = project.get_artifact(str(sources[0]["source_artifact_id"]))
    operator = project.get_operator(str(operation["operator_id"]))
    raw_params = descriptor.get("request", {}).get("params", {})
    if not isinstance(raw_params, Mapping):
        raise _RepresentationReplayError(
            f"Operation {operation_id!r} has invalid persisted request parameters."
        )
    params = dict(operator.deserialize_operation_params(raw_params))

    if source.artifact_type == ArtifactType.TABLE:
        if not _supports_text_root(operator):
            raise _RepresentationReplayError(
                f"Frozen operator {operator.__class__.__name__} cannot transform new raw text."
            )
        return [{"kind": "texts", "operator": operator, "params": params}]

    if source.artifact_type in _MATRIX_TYPES:
        if not _supports_matrix_stage(operator):
            raise _RepresentationReplayError(
                f"Frozen operator {operator.__class__.__name__} cannot replay new matrix rows."
            )
        return [
            *_build_replay_plan(project, source),
            {"kind": "matrix", "operator": operator, "params": params},
        ]

    raise _RepresentationReplayError(
        "New-text replay requires a representation lineage rooted in a table and "
        f"continuing through matrix artifacts; source {source.artifact_id!r} has type "
        f"{source.artifact_type.value!r}."
    )


def _replay_stage(
    operator: Any,
    values: Any,
    *,
    kind: str
    params: Mapping[str, Any],
) -> Any:
    """Execute one frozen replay stage through the ordinary translation contract."""
    if kind == "texts":
        if operator.__class__.__name__ == "SentenceTransformerEncoder":
            return operator.translate(
                values,
                device="auto",
                model_batch_size=int(params.get("model_batch_size", 32)),
            )
        return operator.translate(values)

    if _has_standalone_translate(operator):
        return operator.translate(values)

    # Core positional feature-subset operators are internal TeAL operations rather
    # than user-facing translators. Preserve their narrow in-memory replay hook
    # without requiring them to invent a public standalone translator contract.
    fallback = getattr(operator, "transform_external_matrix", None)
    if callable(fallback):
        return fallback(values, params=params)
    raise _RepresentationReplayError(
        f"Frozen operator {operator.__class__.__name__} cannot replay matrix rows."
    )


def _supports_text_root(operator: Any) -> bool:
    """Return whether an operator is a raw-text representation root."""
    if operator.__class__.__name__ not in {
        "CountVectorizer",
        "SentenceTransformerEncoder",
    }:
        return False
    if bool(getattr(operator, "requires_fit", False)) and not bool(
        getattr(operator, "is_fitted", False)
    ):
        return False
    return _has_standalone_translate(operator)


def _supports_matrix_stage(operator: Any) -> bool:
    """Return whether a frozen one-input matrix stage can replay new rows."""
    if getattr(operator, "axis", None) == "columns":
        # Column normalization depends on the fitted corpus row population.
        return False
    if bool(getattr(operator, "requires_fit", False)) and not bool(
        getattr(operator, "is_fitted", False)
    ):
        return False
    if _has_standalone_translate(operator):
        signature = inspect.signature(operator.translate)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        # Bound method signatures omit self. Replay can supply only the matrix.
        return (
            len(required) == 1
            and required[0].kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )

    # Internal positional feature-subset operators intentionally remain outside
    # the standalone translator contract.
    return callable(getattr(operator, "transform_external_matrix", None))


def _has_standalone_translate(operator: Any) -> bool:
    method = getattr(type(operator), "translate", None)
    return callable(method) and method is not BaseTranslator.translate


def _operation_descriptor(project: Project, operation_id: str) -> Mapping[str, Any]:
    path = project.storage.operation_descriptor_path(operation_id)
    if not path.exists():
        raise _RepresentationReplayError(
            f"Operation {operation_id!r} has no durable operation descriptor."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise _RepresentationReplayError(
            f"Operation descriptor for {operation_id!r} must contain a mapping."
        )
    return payload


def _validate_replayed_rows(
    values: Any, expected: int, artifact: BaseArtifact
) -> None:
    if isinstance(values, Mapping) and "values" in values:
        values = values["values"]
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != int(expected):
        raise _RepresentationReplayError(
            f"Replaying artifact {artifact.artifact_id!r} expected {expected} transformed "
            f"row(s), received shape={shape!r}."
        )
    expected_columns = len(artifact.get_data_columns())
    if int(shape[1]) != expected_columns:
        raise _RepresentationReplayError(
            f"Replaying artifact {artifact.artifact_id!r} expected {expected_columns} "
            f"feature columns, received shape={shape!r}."
        )
