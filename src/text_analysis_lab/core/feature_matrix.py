"""Durable keys-only FeatureMatrix operations.

This is a narrowly dispatched alternative to the regular matrix writer:
the output stores keys and a frozen recipe, never a numeric matrix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.failure_cleanup import mark_operation_failed_best_effort
from text_analysis_lab.core.idempotence import (
    finalize_alias_plan, prepare_alias_plan, reused_outputs,
)
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.operator import OutputSpec, TranslationRequest
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators.feature_matrix import FeatureMatrix


def feature_matrix(
    project: Any,
    translator: FeatureMatrix,
    sources: Any,
    *,
    workers: int = 1,
    batch_size: int | None = None,
    batch_by: str | None = None,
    max_outstanding_units: int | None = None,
    memo: str | None = None,
    alias: Any = None,
    overwrite: bool = False,
    **params: Any,
) -> Mapping[str, Any]:
    """Create an ordinary matrix artifact with an opt-in computed-data provider."""
    if isinstance(sources, Mapping):
        if set(sources) != {DEFAULT_SOURCE_LABEL}:
            raise OperatorError("FeatureMatrix requires exactly one 'source' artifact.")
        sources = sources[DEFAULT_SOURCE_LABEL]
    if isinstance(sources, (list, tuple)):
        raise OperatorError("FeatureMatrix requires one source artifact.")
    source = project.get_artifact(sources)
    source.require_complete()
    if source.artifact_type.value not in {
        "table", "jsonl", "dense_matrix", "sparse_matrix",
    }:
        raise OperatorError("FeatureMatrix requires a matrix, table or JSONL source.")
    if params or workers != 1 or batch_by is not None or max_outstanding_units is not None:
        raise OperatorError(
            "Lazy FeatureMatrix accepts no execution parameters, parallel workers "
            "or grouped batching."
        )
    size = 10_000 if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    label = DEFAULT_OUTPUT_LABEL
    alias_plan = prepare_alias_plan(project, (label,), alias, overwrite=overwrite)
    if alias_plan is not None and alias_plan.reuse:
        return reused_outputs(project, alias_plan)

    if translator.operator_id is None:
        translator.fit_source(source, batch_size=size)
    else:
        row = project.catalog.resolve_operator(translator.operator_id)
        if row.get("snapshot_status") != "serialized":
            raise OperatorError("FeatureMatrix operator has not been serialized.")
        if project.catalog.operator_reuse_status(translator.operator_id) != "reusable":
            raise OperatorError("FeatureMatrix operator is not reusable.")
        if not translator.is_frozen or not translator.is_fitted:
            raise OperatorError("Reload a frozen fitted FeatureMatrix operator.")
        translator.validate_source(source)
    if not translator.is_fitted:
        raise OperatorError("FeatureMatrix fitting did not produce a feature schema.")

    output_kind = "sparse_matrix" if translator.sparse_output else "dense_matrix"
    operator_id = translator.operator_id
    if operator_id is None:
        operator_id = next_id(project.storage.manifest_path, "operator")
        translator.assign_operator_id(operator_id)
        project.catalog.register_operator(
            operator_id=operator_id,
            operation_type="translate",
            snapshot_status="pending",
        )
        try:
            translator.save_to_dir(
                project.storage.operator_dir(operator_id), operator_id=operator_id
            )
            project.catalog.mark_operator_serialized(operator_id)
        except Exception:
            project.catalog.mark_operator_snapshot_failed(operator_id)
            raise

    operation_id = next_id(project.storage.manifest_path, "operation")
    operation_dir = project.storage.operation_dir(operation_id)
    operation_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id, operation_type="translate",
        operator_id=operator_id, status="incomplete",
    )
    project.catalog.add_operation_source(
        operation_id, DEFAULT_SOURCE_LABEL, source.artifact_id
    )

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id, artifact_type=output_kind, label=label,
        lineage_mode="preserved_key", status="incomplete",
        basis_artifact_ids=(source.artifact_id,),
    )
    project.catalog.add_operation_output(operation_id, label, artifact_id, ordinal=0)
    writer = create_artifact_writer(
        artifact_type=output_kind,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        operation_id=operation_id,
        lineage={
            "feature_matrix": {
                "operator_id": operator_id,
                "source_artifact_id": source.artifact_id,
                "feature_columns": list(translator.feature_columns),
            },
        },
        lineage_mode="preserved_key",
        basis_artifact_ids=(source.artifact_id,),
        feature_metadata_mode="own",
    )
    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "translate",
        "kind": "feature_matrix",
        "operator_id": operator_id,
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: source.artifact_id},
        "output_artifact_ids": {label: artifact_id},
        "output_specs": {
            label: OutputSpec(
                artifact_type=output_kind,
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
                feature_metadata_mode="own",
            ).to_dict()
        },
        "request": {"params": {}, "output_label": label},
    }
    descriptor_path = operation_dir / "operation.json"

    def save_descriptor() -> None:
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
        )

    save_descriptor()
    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )
        for frame in source.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            metadata_mode="none", order_by="_position", form="table",
            iter_batches=True, batch_size=size,
        ):
            if len(frame):
                writer.write({
                    "keys": frame.loc[:, source.primary_key].reset_index(drop=True)
                })
        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        save_descriptor()
        project.storage.touch_manifest()
        outputs = {label: project.get_artifact(artifact_id)}
        finalize_alias_plan(project, alias_plan, outputs)
        return outputs
    except Exception as exc:
        mark_operation_failed_best_effort(
            project, writer=writer, artifact_id=artifact_id,
            operation_id=operation_id, error=exc,
        )
        descriptor["status"] = "failed"
        descriptor["error"] = f"{type(exc).__name__}: {exc}"
        save_descriptor()
        raise
