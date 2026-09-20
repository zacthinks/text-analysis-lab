"""Lazy basis-first horizontal composition for same-key relational artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.failure_cleanup import mark_operation_failed_best_effort
from text_analysis_lab.core.ids import next_id
from text_analysis_lab.core.lineage import validate_primary_key_relationship
from text_analysis_lab.core.operator import (
    BaseOperator,
    OutputSpec,
    TranslationRequest,
    validate_output_label,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, ArtifactType
from text_analysis_lab.core.writer import create_artifact_writer

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


class JoinOperator(BaseOperator):
    """Durable operator snapshot for TeAL's constrained lazy same-key join."""

    operation_type = "join"

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        if len(sources) < 2:
            raise ArtifactError(
                "Join requires a basis artifact and at least one additional artifact."
            )
        first = next(iter(sources.values()))
        return {
            DEFAULT_OUTPUT_LABEL: OutputSpec(
                artifact_type=first.artifact_type,
                lineage_mode="joined_key",
                basis_labels=tuple(sources),
            )
        }


def join(
    project: Project,
    basis: BaseArtifact | str,
    *others: BaseArtifact | str,
    output_label: str = DEFAULT_OUTPUT_LABEL,
    batch_size: int = 10_000,
    memo: str | None = None,
) -> BaseArtifact:
    """Lazily compose fields onto the first artifact's exact key universe/order.

    Every source must be a relational (table/JSONL) artifact with exactly the
    same ordered primary-key schema. The first artifact defines result rows.
    Later-only keys are ignored; missing later matches resolve as null. The
    result owns keys only and resolves source data/metadata lazily at query time.
    """
    sources = [
        project.get_artifact(basis),
        *[project.get_artifact(ref) for ref in others],
    ]
    if len(sources) < 2:
        raise ArtifactError("join requires at least two artifacts.")
    output_label = validate_output_label(output_label)
    batch_size = _validate_batch_size(batch_size)
    _validate_sources(project, sources)

    operator = JoinOperator()
    operator_id = next_id(project.storage.manifest_path, "operator")
    operator.assign_operator_id(operator_id)
    project.catalog.register_operator(
        operator_id=operator_id,
        operation_type="join",
        snapshot_status="pending",
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
    operation_dir = project.storage.operation_dir(operation_id)
    operation_dir.mkdir(parents=True, exist_ok=True)
    project.catalog.register_operation(
        operation_id=operation_id,
        operation_type="join",
        operator_id=operator_id,
        status="incomplete",
    )
    source_labels = [
        "basis",
        *[f"source_{index:04d}" for index in range(1, len(sources))],
    ]
    for label, source in zip(source_labels, sources, strict=True):
        project.catalog.add_operation_source(operation_id, label, source.artifact_id)

    artifact_id = next_id(project.storage.manifest_path, "artifact")
    basis_ids = [source.artifact_id for source in sources]
    artifact_type = sources[0].artifact_type
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        label=output_label,
        lineage_mode="joined_key",
        status="incomplete",
        basis_artifact_ids=basis_ids,
    )
    project.catalog.add_operation_output(
        operation_id, output_label, artifact_id, ordinal=0
    )
    writer = create_artifact_writer(
        artifact_type=artifact_type,
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=output_label,
        operation_id=operation_id,
        lineage_mode="joined_key",
        basis_artifact_ids=basis_ids,
    )

    descriptor = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_type": "join",
        "operator_id": operator_id,
        "status": "incomplete",
        "resumable": False,
        "source_order": basis_ids,
        "sources": {
            label: source.artifact_id
            for label, source in zip(source_labels, sources, strict=True)
        },
        "output_artifact_ids": {output_label: artifact_id},
        "request": {"output_label": output_label, "batch_size": batch_size},
        "scope": "lazy_basis_first_same_key_join",
        "row_universe_artifact_id": sources[0].artifact_id,
    }
    _write_json(operation_dir / "operation.json", descriptor)

    try:
        if memo is not None:
            project.catalog.add_memo(
                target_type="operation", target_id=operation_id, body=memo
            )
        key_columns = list(sources[0].primary_key)
        for frame in sources[0].query(
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
            if not frame.empty:
                writer.write({"keys": frame.loc[:, key_columns]})
        writer.finalize()
        validate_primary_key_relationship(
            basis_keys=[source.primary_key for source in sources],
            output_key=sources[0].primary_key,
            lineage_mode="joined_key",
        )
        project.catalog.mark_artifact_complete(artifact_id)
        project.catalog.mark_operation_complete(operation_id)
        descriptor["status"] = "complete"
        _write_json(operation_dir / "operation.json", descriptor)
        project.storage.touch_manifest()
        project.query.clear_cache()
        return project.get_artifact(artifact_id)
    except Exception as exc:
        mark_operation_failed_best_effort(
            project,
            writer=writer,
            artifact_id=artifact_id,
            operation_id=operation_id,
            error=exc,
        )
        descriptor["status"] = "failed"
        descriptor["error"] = f"{exc.__class__.__name__}: {exc}"
        _write_json(operation_dir / "operation.json", descriptor)
        raise


def _validate_sources(project: Project, sources: Sequence[BaseArtifact]) -> None:
    first = sources[0]
    first.require_complete()
    allowed = {ArtifactType.TABLE, ArtifactType.JSONL}
    if first.artifact_type not in allowed:
        raise ArtifactError(
            "join currently supports relational table/JSONL artifacts only; "
            f"got {first.artifact_type.value!r}."
        )
    expected_pk = tuple(first.primary_key)
    if not expected_pk:
        raise ArtifactError("join source artifacts must have a primary key.")
    for source in sources[1:]:
        source.require_complete()
        if source.artifact_type not in allowed:
            raise ArtifactError(
                "join currently supports relational table/JSONL artifacts only; "
                f"got {source.artifact_type.value!r} for {source.artifact_id}."
            )
        if tuple(source.primary_key) != expected_pk:
            raise ArtifactError(
                "join sources must use exactly the same primary-key fields in the same order: "
                f"expected {list(expected_pk)}, got {list(source.primary_key)} "
                f"for {source.artifact_id}."
            )
    _validate_field_collisions(project, sources)


def _validate_field_collisions(
    project: Project, sources: Sequence[BaseArtifact]
) -> None:
    """Reject new same-namespace base-name collisions across joined sources.

    A single source may already expose qualified ambiguous columns; ``join`` does
    not try to repair or reinterpret that source.  Across sources, however, the
    same base field name is safe only when every branch resolves it to exactly
    the same underlying owner(s), which represents the same inherited field
    arriving through more than one lineage path.
    """
    seen: dict[tuple[str, str], frozenset[str | None]] = {}
    for source in sources:
        info = source.query_columns(metadata_mode="full")
        by_name: dict[tuple[str, str], set[str | None]] = {}
        for raw in info.get("columns", []):
            if not isinstance(raw, Mapping):
                continue
            namespace = str(raw.get("namespace"))
            if namespace not in {"data", "metadata"}:
                continue
            base_name = str(raw.get("base_name"))
            owner = raw.get("source_artifact_id")
            owner_id = None if owner is None else str(owner)
            by_name.setdefault((namespace, base_name), set()).add(owner_id)

        for key, owners in by_name.items():
            owner_set = frozenset(owners)
            previous = seen.get(key)
            if previous is None:
                seen[key] = owner_set
                continue
            if previous == owner_set:
                # The same inherited field(s) reached through multiple branches.
                continue
            namespace, base_name = key
            raise ArtifactError(
                "join field-name collision across source artifacts: "
                f"{namespace} field {base_name!r} resolves from distinct owners "
                f"{sorted(str(value) for value in previous)} and "
                f"{sorted(str(value) for value in owner_set)}. Rename before joining."
            )


def _validate_batch_size(value: int) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("batch_size must be positive.")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(payload), indent=2, sort_keys=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
