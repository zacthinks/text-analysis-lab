"""Random sampling into a keys-only preserved-key child artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


class RandomSampleOperator(BaseOperator):
    """Frozen descriptor for an unstratified random sample without replacement."""

    operation_type = "subset"

    def __init__(
        self,
        *,
        n: int,
        random_state: int | None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.n = int(n)
        self.random_state = None if random_state is None else int(random_state)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
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
        return {"n": self.n, "random_state": self.random_state}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> RandomSampleOperator:
        raw_state = state.get("random_state")
        return cls(
            n=int(state["n"]),
            random_state=None if raw_state is None else int(raw_state),
        )


def sample(
    project: Project,
    source: BaseArtifact | str,
    *,
    n: int,
    random_state: int | None = None,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    memo: str | None = None,
) -> BaseArtifact:
    """Draw ``n`` source rows without replacement into one child artifact.

    Sampling determines membership only. The selected rows are written in source
    order, preserving the source primary-key schema and inheriting representation
    data through preserved-key lineage.

    This is a convenience sampling operator, not a probability-audit design.
    Use ``Project.probability_split(...)`` when inclusion probabilities or
    stratified probability sampling are part of the analysis.
    """
    artifact = project.get_artifact(source)
    artifact.require_complete()

    pk = tuple(str(column) for column in artifact.primary_key)
    if not pk:
        raise ArtifactError("sample source artifact has no primary key.")

    if isinstance(n, bool):
        raise TypeError("n must be an integer, not bool.")
    try:
        sample_size = int(n)
    except (TypeError, ValueError) as exc:
        raise TypeError("n must be an integer.") from exc
    if sample_size != n:
        raise ValueError("n must be an integer.")
    if sample_size <= 0:
        raise ValueError("n must be positive.")

    total = int(artifact.n_rows or 0)
    if sample_size > total:
        raise ValueError(
            f"n={sample_size} exceeds source row count {total}; "
            "sampling is without replacement."
        )

    seed = None if random_state is None else int(random_state)
    label = validate_output_label(output_label)

    operator = RandomSampleOperator(n=sample_size, random_state=seed)
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
        artifact.artifact_id,
    )

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact.artifact_type,
        label=label,
        lineage_mode="preserved_key",
        status="incomplete",
        basis_artifact_ids=(artifact.artifact_id,),
    )
    project.catalog.add_operation_output(
        operation_id,
        label,
        artifact_id,
        ordinal=0,
    )

    writer = create_artifact_writer(
        artifact_type=artifact.artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        operation_id=operation_id,
        lineage_mode="preserved_key",
        basis_artifact_ids=(artifact.artifact_id,),
    )

    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "subset",
        "operator_id": operator_id,
        "kind": "sample",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: artifact.artifact_id},
        "output_artifact_ids": {label: artifact_id},
        "request": {
            "output_label": label,
            "n": sample_size,
            "random_state": seed,
            "replace": False,
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

        frame = artifact.query(
            key_columns=True,
            data_columns=False,
            metadata_columns=False,
            metadata_mode="none",
            sample_n=sample_size,
            random_state=seed,
            order_by="_position",
            form="table",
            include_position=False,
        )
        if not isinstance(frame, pd.DataFrame):
            raise ArtifactError("sample expected a tabular key query result.")
        if len(frame) != sample_size:
            raise ArtifactError(
                "sample query returned an unexpected number of rows: "
                f"expected {sample_size}, got {len(frame)}."
            )

        writer.write({"keys": frame.loc[:, list(pk)].reset_index(drop=True)})
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
    except Exception:
        project.catalog.mark_artifact_failed(artifact_id)
        project.catalog.mark_operation_failed(operation_id)
        descriptor["status"] = "failed"
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise
