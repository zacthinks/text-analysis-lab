"""Lazy positional feature subsetting for matrix artifacts.

Feature subsetting is deliberately positional. A matrix feature axis is defined
by column order; feature labels and any future feature metadata annotate that
axis but do not establish cross-artifact identity. The derived artifact stores
only its keys and an ordered integer projection into its basis feature axis.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.types import ArtifactType, DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


FeatureSubsetFunction = Callable[[pd.DataFrame], Iterable[bool]]


class FeatureSubsetOperator(BaseOperator):
    """Frozen positional feature projection used by ``Project.feature_subset``."""

    operation_type = "subset"

    def __init__(
        self,
        *,
        source_width: int,
        selected_indices: Sequence[int],
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.source_width = int(source_width)
        if self.source_width <= 0:
            raise ValueError("source_width must be positive.")
        self.selected_indices = _normalize_indices(
            selected_indices,
            source_width=self.source_width,
        )

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = sources[DEFAULT_SOURCE_LABEL]
        return OutputSpec(
            artifact_type=source.artifact_type,
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "selected_indices": list(self.selected_indices),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "FeatureSubsetOperator":
        raw = state.get("selected_indices")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise OperatorError("FeatureSubsetOperator state is missing selected_indices.")
        return cls(
            source_width=int(state["source_width"]),
            selected_indices=[int(value) for value in raw],
        )

    def transform_external_matrix(
        self,
        matrix: Any,
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Apply the frozen positional projection to new matrix rows."""
        _ = query, params
        shape = getattr(matrix, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[1]) != self.source_width:
            raise OperatorError(
                "FeatureSubsetOperator replay requires source width "
                f"{self.source_width}; got shape={shape!r}."
            )
        return slice_matrix_features(matrix, self.selected_indices)

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "matrix"


def feature_subset(
    project: "Project",
    source: "BaseArtifact | str",
    function: FeatureSubsetFunction,
    *,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> "BaseArtifact":
    """Create a lazy positional feature view of a matrix artifact.

    ``function`` receives the source ``feature_frame`` in source-column order and
    must return exactly one boolean value per feature. The selected source
    positions are frozen into the derived artifact. No feature-name matching is
    performed during creation or replay.
    """
    artifact = project.get_artifact(source)
    artifact.require_complete()
    if artifact.artifact_type not in {
        ArtifactType.SPARSE_MATRIX,
        ArtifactType.DENSE_MATRIX,
    }:
        raise ArtifactError(
            "feature_subset requires a sparse_matrix or dense_matrix source."
        )
    getter = getattr(artifact, "get_feature_frame", None)
    if not callable(getter):
        raise ArtifactError(
            f"Matrix artifact {artifact.artifact_id} does not expose a feature frame."
        )
    if not callable(function):
        raise TypeError("feature_subset function must be callable.")

    feature_frame = getter()
    if not isinstance(feature_frame, pd.DataFrame):
        raise ArtifactError("get_feature_frame() must return a pandas DataFrame.")
    source_width = len(feature_frame)
    if source_width <= 0:
        raise ArtifactError("feature_subset cannot operate on an empty feature axis.")

    raw_mask = function(feature_frame.copy())
    indices = _indices_from_mask(raw_mask, expected_len=source_width)
    label = validate_output_label(output_label)
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    return _create_feature_view(
        project,
        artifact,
        indices=indices,
        source_width=source_width,
        output_label=label,
        batch_size=int(batch_size),
        memo=memo,
    )


def _create_feature_view(
    project: "Project",
    source: "BaseArtifact",
    *,
    indices: Sequence[int],
    source_width: int,
    output_label: str,
    batch_size: int,
    memo: str | None,
) -> "BaseArtifact":
    """Persist one keys-only positional feature view."""
    normalized = _normalize_indices(indices, source_width=source_width)
    operator = FeatureSubsetOperator(
        source_width=source_width,
        selected_indices=normalized,
    )
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="subset",
        snapshot_status="pending",
    )
    try:
        operator.save_to_dir(
            project.storage.operator_dir(operator_id),
            operator_id=operator_id,
        )
        project.catalog.mark_operator_serialized(operator_id)
    except Exception:
        project.catalog.mark_operator_snapshot_failed(operator_id)
        raise

    operation_id = next_id(project.storage.manifest_path, "operation")
    op_dir = project.storage.operation_dir(operation_id)
    op_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="subset",
        operator_id=operator_id,
        status="incomplete",
    )
    project.catalog.add_operation_source(
        operation_id,
        DEFAULT_SOURCE_LABEL,
        source.artifact_id,
    )

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=source.artifact_type,
        label=output_label,
        lineage_mode="preserved_key",
        status="incomplete",
        basis_artifact_ids=(source.artifact_id,),
    )
    project.catalog.add_operation_output(
        operation_id,
        output_label,
        artifact_id,
        ordinal=0,
    )

    writer = create_artifact_writer(
        artifact_type=source.artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=output_label,
        operation_id=operation_id,
        lineage={"feature_indices": list(normalized)},
        lineage_mode="preserved_key",
        basis_artifact_ids=(source.artifact_id,),
    )

    spec = operator.output_specs(
        sources={DEFAULT_SOURCE_LABEL: source},
        request=TranslationRequest(),
    )
    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "subset",
        "operator_id": operator_id,
        "kind": "feature_subset",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: source.artifact_id},
        "output_artifact_ids": {output_label: artifact_id},
        "output_specs": {output_label: spec.to_dict()},
        "request": {
            "params": {},
            "output_label": output_label,
            "source_width": int(source_width),
            "selected_indices": list(normalized),
        },
    }
    descriptor_path = op_dir / "operation.json"
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation",
                target_id=operation_id,
                body=memo,
            )
        for frame in source.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            metadata_mode="none",
            order_by="_position",
            form="table",
            iter_batches=True,
            batch_size=batch_size,
            include_position=False,
        ):
            if frame.empty:
                continue
            writer.write({"keys": frame.loc[:, source.primary_key].reset_index(drop=True)})

        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        project.storage.touch_manifest()
        return project.get_artifact(artifact_id)
    except Exception as exc:
        try:
            writer.mark_failed(exc)
        except Exception:
            pass
        try:
            project.catalog.mark_artifact_failed(artifact_id)
        except Exception:
            pass
        try:
            project.catalog.mark_operation_failed(operation_id, exc)
        except Exception:
            pass
        descriptor["status"] = "failed"
        descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


def _indices_from_mask(raw_mask: Iterable[bool], *, expected_len: int) -> tuple[int, ...]:
    if isinstance(raw_mask, pd.Series):
        values = raw_mask.tolist()
    elif isinstance(raw_mask, np.ndarray):
        if raw_mask.ndim != 1:
            raise ArtifactError("feature_subset mask must be one-dimensional.")
        values = raw_mask.tolist()
    else:
        try:
            values = list(raw_mask)
        except TypeError as exc:
            raise ArtifactError(
                "feature_subset function must return an iterable boolean mask."
            ) from exc

    if len(values) != int(expected_len):
        raise ArtifactError(
            "feature_subset mask length does not match source feature width: "
            f"{len(values)} != {expected_len}."
        )
    mask: list[bool] = []
    for value in values:
        if not isinstance(value, (bool, np.bool_)):
            raise ArtifactError(
                "feature_subset function must return only boolean mask values."
            )
        mask.append(bool(value))
    indices = tuple(index for index, keep in enumerate(mask) if keep)
    if not indices:
        raise ArtifactError("feature_subset retained no features.")
    return indices


def _normalize_indices(
    indices: Sequence[int],
    *,
    source_width: int,
) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes, bytearray)):
        raise TypeError("feature indices must be a sequence of integers.")
    out: list[int] = []
    for raw in indices:
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            raise ArtifactError("feature indices must contain only integers.")
        index = int(raw)
        if index < 0 or index >= int(source_width):
            raise ArtifactError(
                f"feature index {index} is outside source width {source_width}."
            )
        out.append(index)
    if not out:
        raise ArtifactError("feature subset must retain at least one feature.")
    if any(right <= left for left, right in zip(out, out[1:])):
        raise ArtifactError(
            "feature indices must be unique and strictly increasing so source order is preserved."
        )
    return tuple(out)


def slice_matrix_features(matrix: Any, indices: Sequence[int]) -> Any:
    """Return ``matrix[:, indices]`` while preserving sparse/dense representation."""
    resolved = [int(index) for index in indices]
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix)[:, resolved].tocsr()
    dense = np.asarray(matrix)
    if dense.ndim != 2:
        raise ArtifactError("Feature projection requires a two-dimensional matrix.")
    return np.asarray(dense[:, resolved])
