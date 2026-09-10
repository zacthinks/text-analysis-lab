"""Key-domain restriction for row-addressable TeAL artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.operator import BaseOperator, OutputSpec, TranslationRequest, validate_output_label
from text_analysis_lab.core.query import _parquet_dataset_expr, quote_identifier
from text_analysis_lab.core.types import ArtifactType, DEFAULT_OUTPUT_LABEL
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


_ALLOWED_TYPES = {
    ArtifactType.TABLE,
    ArtifactType.JSONL,
    ArtifactType.SPARSE_MATRIX,
    ArtifactType.DENSE_MATRIX,
}


class RestrictOperator(BaseOperator):
    """Frozen descriptor for key-domain restriction."""

    operation_type = "subset"

    def __init__(
        self,
        *,
        source_primary_key: tuple[str, ...] = (),
        domain_primary_key: tuple[str, ...] = (),
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.source_primary_key = tuple(str(value) for value in source_primary_key)
        self.domain_primary_key = tuple(str(value) for value in domain_primary_key)

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = sources["source"]
        return OutputSpec(
            artifact_type=source.artifact_type,
            lineage_mode="preserved_key",
            basis_labels="source",
        )

    def to_json_state(self) -> dict[str, Any]:
        return {
            "source_primary_key": list(self.source_primary_key),
            "domain_primary_key": list(self.domain_primary_key),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "RestrictOperator":
        return cls(
            source_primary_key=tuple(str(value) for value in state.get("source_primary_key", ())),
            domain_primary_key=tuple(str(value) for value in state.get("domain_primary_key", ())),
        )


def restrict(
    project: "Project",
    source: "BaseArtifact | str",
    *,
    to: "BaseArtifact | str",
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> "BaseArtifact":
    """Restrict ``source`` to the key domain represented by ``to``.

    ``to`` must use a non-empty primary-key prefix of ``source``.  The result
    preserves the source grain and full source primary key; ``to`` only defines
    which prefix values are admissible.  A domain row is allowed to have zero
    finer-grained source rows, which is important for legitimate zero-or-more
    extended-key recompositions.
    """

    source_artifact = project.get_artifact(source)
    domain_artifact = project.get_artifact(to)
    source_artifact.require_complete()
    domain_artifact.require_complete()

    source_pk = tuple(str(value) for value in source_artifact.primary_key)
    domain_pk = tuple(str(value) for value in domain_artifact.primary_key)
    if not source_pk or not domain_pk:
        raise ArtifactError("restrict requires both artifacts to have non-empty primary keys.")
    if len(domain_pk) > len(source_pk) or source_pk[: len(domain_pk)] != domain_pk:
        raise ArtifactError(
            "restrict(to=...) requires the domain primary key to be a prefix of the "
            f"source primary key; source={list(source_pk)}, domain={list(domain_pk)}."
        )
    if source_artifact.artifact_type not in _ALLOWED_TYPES:
        raise ArtifactError(
            "restrict currently supports table, jsonl, sparse_matrix, and dense_matrix "
            f"sources; got {source_artifact.artifact_type.value!r}."
        )

    label = validate_output_label(output_label)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    operator = RestrictOperator(
        source_primary_key=source_pk,
        domain_primary_key=domain_pk,
    )
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="subset",
        snapshot_status="pending",
    )
    try:
        operator.save_to_dir(project.storage.operator_dir(operator_id), operator_id=operator_id)
        project.catalog.mark_operator_serialized(operator_id)
    except Exception:
        project.catalog.mark_operator_snapshot_failed(operator_id)
        raise

    operation_id = next_id(project.storage.manifest_path, "operation")
    operation_dir = project.storage.operation_dir(operation_id)
    operation_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="subset",
        operator_id=operator_id,
        status="incomplete",
    )
    project.catalog.add_operation_source(operation_id, "source", source_artifact.artifact_id)
    project.catalog.add_operation_source(operation_id, "domain", domain_artifact.artifact_id)

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=source_artifact.artifact_type,
        label=label,
        lineage_mode="preserved_key",
        status="incomplete",
        basis_artifact_ids=(source_artifact.artifact_id,),
    )
    project.catalog.add_operation_output(operation_id, label, artifact_id, ordinal=0)
    writer = create_artifact_writer(
        artifact_type=source_artifact.artifact_type,
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
        "operation_type": "subset",
        "operator_id": operator_id,
        "kind": "restrict",
        "status": "incomplete",
        "resumable": False,
        "sources": {
            "source": source_artifact.artifact_id,
            "domain": domain_artifact.artifact_id,
        },
        "output_artifact_ids": {label: artifact_id},
        "request": {
            "output_label": label,
            "batch_size": batch_size,
            "domain_primary_key": list(domain_pk),
        },
    }
    _write_descriptor(operation_dir, descriptor)

    try:
        if memo is not None:
            project.catalog.add_memo(target_type="operation", target_id=operation_id, body=memo)

        source_keys = _parquet_dataset_expr(source_artifact.keys_dir)
        domain_keys = _parquet_dataset_expr(domain_artifact.keys_dir)
        predicates = " AND ".join(
            f"s.{quote_identifier(col)} = d.{quote_identifier(col)}" for col in domain_pk
        )
        selected = ", ".join(
            f"s.{quote_identifier(col)} AS {quote_identifier(col)}" for col in source_pk
        )
        domain_selected = ", ".join(quote_identifier(col) for col in domain_pk)
        sql = (
            f"SELECT {selected} "
            f"FROM {source_keys} s "
            f"INNER JOIN (SELECT {domain_selected} FROM {domain_keys}) d ON {predicates} "
            "ORDER BY s._position"
        )
        reader = project.query.con.execute(sql).to_arrow_reader(batch_size=batch_size)
        wrote_any = False
        for batch in reader:
            frame = batch.to_pandas().reset_index(drop=True)
            if frame.empty:
                continue
            writer.write({"keys": frame.loc[:, list(source_pk)]})
            wrote_any = True
        if not wrote_any:
            writer.write({"keys": pd.DataFrame(columns=list(source_pk))})

        writer.finalize()
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        _write_descriptor(operation_dir, descriptor)
        project.storage.touch_manifest()
        project.query.clear_cache()
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
        _write_descriptor(operation_dir, descriptor)
        raise


def _write_descriptor(operation_dir, descriptor: Mapping[str, Any]) -> None:
    (operation_dir / "operation.json").write_text(
        json.dumps(dict(descriptor), indent=2, sort_keys=True),
        encoding="utf-8",
    )
