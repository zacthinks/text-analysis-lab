"""Stable-key import of externally produced measurements."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
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


_RESERVED = {"_position", "_batch", "_row_offset"}


class KeyedFrameOperator(BaseOperator):
    """Frozen descriptor for one stable-key external-data import shape."""

    operation_type = "translate"

    def __init__(
        self,
        *,
        data_fields: Sequence[str],
        require_complete: bool,
        output_label: str,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.data_fields = tuple(str(value) for value in data_fields)
        self.require_complete = bool(require_complete)
        self.output_label = validate_output_label(output_label)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        if set(sources) != {DEFAULT_SOURCE_LABEL}:
            raise ArtifactError(
                "KeyedFrameOperator requires exactly one source artifact."
            )
        return {
            self.output_label: OutputSpec(
                artifact_type="table",
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
        }

    def to_json_state(self) -> dict[str, Any]:
        return {
            "data_fields": list(self.data_fields),
            "require_complete": self.require_complete,
            "output_label": self.output_label,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> KeyedFrameOperator:
        return cls(
            data_fields=tuple(str(v) for v in state.get("data_fields", ())),
            require_complete=bool(state.get("require_complete", False)),
            output_label=str(state.get("output_label", DEFAULT_OUTPUT_LABEL)),
        )


def from_keyed_frame(
    project: Project,
    source: BaseArtifact | str,
    frame: pd.DataFrame,
    *,
    data_fields: str | Sequence[str],
    require_complete: bool = False,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    memo: str | None = None,
) -> BaseArtifact:
    """Create a same-key table measurement aligned to an existing artifact."""
    source_artifact = project.get_artifact(source)
    source_artifact.require_complete()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    fields = _normalize_fields(data_fields)
    label = validate_output_label(output_label)
    keys = tuple(str(v) for v in source_artifact.primary_key)
    if not keys:
        raise ArtifactError("source artifact has no primary key.")
    _validate_frame_columns(frame, keys=keys, data_fields=fields)

    source_keys = source_artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        form="table",
        include_position=True,
    )
    if not isinstance(source_keys, pd.DataFrame):
        raise ArtifactError("Could not materialize source keys for from_keyed_frame.")
    source_keys = source_keys.sort_values("_position", kind="stable").reset_index(
        drop=True
    )
    source_keys = source_keys.loc[:, list(keys)]
    source_keys = _coerce_keys(source_keys, keys)

    incoming = frame.loc[:, [*keys, *fields]].copy().reset_index(drop=True)
    incoming.loc[:, list(keys)] = _coerce_keys(incoming.loc[:, list(keys)], keys)
    if incoming.duplicated(subset=list(keys)).any():
        raise ArtifactError("from_keyed_frame frame contains duplicate primary keys.")

    source_tuples = _tuples(source_keys, keys)
    incoming_tuples = _tuples(incoming, keys)
    source_set = set(source_tuples)
    incoming_set = set(incoming_tuples)
    unknown = incoming_set - source_set
    if unknown:
        raise ArtifactError(
            f"from_keyed_frame contains {len(unknown)} key(s) not present in source."
        )
    if require_complete and incoming_set != source_set:
        raise ArtifactError(
            "from_keyed_frame require_complete=True requires the frame key set to "
            f"exactly equal source; missing={len(source_set - incoming_set)}."
        )
    if not incoming_tuples:
        raise ArtifactError("from_keyed_frame frame must contain at least one row.")

    indexed = incoming.set_index(list(keys), drop=False)
    selected_source_order = [key for key in source_tuples if key in incoming_set]
    if len(keys) == 1:
        order_index: Any = [key[0] for key in selected_source_order]
    else:
        order_index = pd.MultiIndex.from_tuples(selected_source_order, names=list(keys))
    aligned = indexed.loc[order_index].reset_index(drop=True)
    key_payload = aligned.loc[:, list(keys)].copy()
    data_payload = aligned.loc[:, list(fields)].copy()

    operator = KeyedFrameOperator(
        data_fields=fields,
        require_complete=require_complete,
        output_label=label,
    )
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id, operation_type="translate", snapshot_status="pending"
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
        operation_type="translate",
        operator_id=operator_id,
        status="incomplete",
    )
    project.catalog.add_operation_source(
        operation_id, DEFAULT_SOURCE_LABEL, source_artifact.artifact_id
    )

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="table",
        label=label,
        lineage_mode="preserved_key",
        status="incomplete",
        basis_artifact_ids=(source_artifact.artifact_id,),
    )
    project.catalog.add_operation_output(operation_id, label, artifact_id, ordinal=0)
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=label,
        operation_id=operation_id,
        lineage_mode="preserved_key",
        basis_artifact_ids=(source_artifact.artifact_id,),
    )
    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "translate",
        "operator_id": operator_id,
        "kind": "from_keyed_frame",
        "status": "incomplete",
        "resumable": False,
        "sources": {DEFAULT_SOURCE_LABEL: source_artifact.artifact_id},
        "output_artifact_ids": {label: artifact_id},
        "request": {
            "data_fields": list(fields),
            "require_complete": bool(require_complete),
            "output_label": label,
            "input_row_count": len(incoming),
        },
    }
    (op_dir / "operation.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
    )

    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )
        writer.write({"keys": key_payload, "data": data_payload})
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


def _normalize_fields(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(str(v) for v in value)
    else:
        raise TypeError("data_fields must be a string or sequence of strings.")
    if not values or any(not v for v in values):
        raise ValueError("data_fields must contain at least one non-empty field.")
    if len(set(values)) != len(values):
        raise ValueError("data_fields cannot contain duplicates.")
    return values


def _validate_frame_columns(
    frame: pd.DataFrame, *, keys: Sequence[str], data_fields: Sequence[str]
) -> None:
    columns = [str(v) for v in frame.columns]
    if len(set(columns)) != len(columns):
        raise ArtifactError("from_keyed_frame frame must have unique column names.")
    missing = [v for v in [*keys, *data_fields] if v not in columns]
    if missing:
        raise ArtifactError(
            f"from_keyed_frame frame is missing required column(s) {missing}."
        )
    overlap = sorted(set(keys).intersection(data_fields))
    if overlap:
        raise ArtifactError(
            f"data_fields cannot include primary-key column(s) {overlap}."
        )
    reserved = sorted(set(data_fields).intersection(_RESERVED))
    if reserved:
        raise ArtifactError(
            f"data_fields cannot include reserved structural column(s) {reserved}."
        )


def _coerce_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for key in keys:
        values = pd.to_numeric(out[key], errors="coerce")
        bad = values.isna() | ~np.isfinite(values) | (values != np.floor(values))
        if bad.any():
            raise ArtifactError(
                f"Primary-key column {key!r} must contain non-null integers."
            )
        out[key] = values.astype("int64")
    return out


def _tuples(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[int, ...]]:
    return [
        tuple(int(v) for v in row)
        for row in frame.loc[:, list(keys)].itertuples(index=False, name=None)
    ]
