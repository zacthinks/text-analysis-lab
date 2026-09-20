"""Narrow stable-key selection convenience for preserved-key subsets."""

from __future__ import annotations

import json
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

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


class SelectKeysOperator(BaseOperator):
    """Frozen descriptor for exact stable-key selection."""

    operation_type = "subset"

    def __init__(self, *, requested_count: int, operator_id: str | None = None) -> None:
        super().__init__(operator_id=operator_id)
        self.requested_count = int(requested_count)

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        source = sources[DEFAULT_SOURCE_LABEL]
        return OutputSpec(
            artifact_type=source.artifact_type,
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    def to_json_state(self) -> dict[str, Any]:
        return {"requested_count": self.requested_count}

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SelectKeysOperator:
        return cls(requested_count=int(state.get("requested_count", 0)))


def select_keys(
    project: Project,
    source: BaseArtifact | str,
    keys: Sequence[Any],
    *,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> BaseArtifact:
    """Create a keys-only preserved-key child containing exactly requested keys.

    Result order follows source artifact order, not request order. Unknown or
    duplicate requested keys fail clearly rather than being silently ignored.
    """
    artifact = project.get_artifact(source)
    artifact.require_complete()
    pk = tuple(str(col) for col in artifact.primary_key)
    if not pk:
        raise ArtifactError("select_keys source artifact has no primary key.")
    requested = _normalize_requested_keys(keys, pk)
    if not requested:
        raise ArtifactError("select_keys requires at least one key.")
    label = validate_output_label(output_label)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    operator = SelectKeysOperator(requested_count=len(requested))
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id, operation_type="subset", snapshot_status="pending"
    )
    try:
        operator.save_to_dir(
            project.storage.operator_dir(operator_id), operator_id=operator_id
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
        operation_id, DEFAULT_SOURCE_LABEL, artifact.artifact_id
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
    project.catalog.add_operation_output(operation_id, label, artifact_id, ordinal=0)
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
        "kind": "select_keys",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: artifact.artifact_id},
        "output_artifact_ids": {label: artifact_id},
        "request": {"output_label": label, "requested_count": len(requested)},
    }
    (op_dir / "operation.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
    )

    requested_set = set(requested)
    found: set[tuple[int, ...]] = set()
    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )
        for frame in artifact.query(
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
            tuples = [
                tuple(int(v) for v in row)
                for row in frame.loc[:, list(pk)].itertuples(index=False, name=None)
            ]
            mask = [value in requested_set for value in tuples]
            if any(mask):
                chosen = frame.loc[mask, list(pk)].reset_index(drop=True)
                found.update(
                    value for value, keep in zip(tuples, mask, strict=True) if keep
                )
                writer.write({"keys": chosen})
        missing = requested_set - found
        if missing:
            examples = sorted(missing)[:10]
            raise ArtifactError(
                f"select_keys requested {len(missing)} key(s) not present in source; examples={examples}."
            )
        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        (op_dir / "operation.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
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
        (op_dir / "operation.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


def _normalize_requested_keys(
    keys: Sequence[Any], primary_key: Sequence[str]
) -> list[tuple[int, ...]]:
    if isinstance(keys, (str, bytes, bytearray)):
        raise TypeError(
            "select_keys keys must be a sequence of key values, not a string."
        )
    out: list[tuple[int, ...]] = []
    for raw in keys:
        if len(primary_key) == 1:
            values = (raw,)
        elif isinstance(raw, Mapping):
            missing = [col for col in primary_key if col not in raw]
            if missing:
                raise ArtifactError(
                    f"select_keys composite key is missing columns {missing}."
                )
            values = tuple(raw[col] for col in primary_key)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if len(raw) != len(primary_key):
                raise ArtifactError(
                    f"select_keys expected {len(primary_key)} values per composite key; got {len(raw)}."
                )
            values = tuple(raw)
        else:
            raise ArtifactError(
                f"select_keys requires composite keys as mappings/tuples for primary key {list(primary_key)}."
            )
        if any(
            not isinstance(value, numbers.Integral) or isinstance(value, bool)
            for value in values
        ):
            raise ArtifactError(
                "TeAL primary-key values supplied to select_keys must be integers."
            )
        out.append(tuple(int(value) for value in values))
    if len(set(out)) != len(out):
        raise ArtifactError("select_keys request contains duplicate primary keys.")
    return out
