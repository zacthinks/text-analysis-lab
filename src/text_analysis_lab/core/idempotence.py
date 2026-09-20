"""Alias-backed idempotent output handling for TeAL operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from text_analysis_lab.core.errors import (
    AliasBundleError,
    AliasOverwriteBlockedError,
    ArtifactNotFoundError,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project

AliasSpec = str | Mapping[str, str] | None


@dataclass(frozen=True)
class AliasPlan:
    output_labels: tuple[str, ...]
    aliases: dict[str, str]
    overwrite: bool
    existing_artifact_ids: dict[str, str]
    reuse: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "output_labels": list(self.output_labels),
            "aliases": dict(self.aliases),
            "overwrite": bool(self.overwrite),
            "existing_artifact_ids": dict(self.existing_artifact_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AliasPlan:
        labels = tuple(str(x) for x in data.get("output_labels", ()))
        raw_aliases = data.get("aliases", {})
        raw_existing = data.get("existing_artifact_ids", {})
        if not isinstance(raw_aliases, Mapping) or not isinstance(
            raw_existing, Mapping
        ):
            raise AliasBundleError("Saved alias plan is malformed.")
        return cls(
            output_labels=labels,
            aliases={str(k): str(v) for k, v in raw_aliases.items()},
            overwrite=bool(data.get("overwrite", False)),
            existing_artifact_ids={str(k): str(v) for k, v in raw_existing.items()},
            reuse=False,
        )


def prepare_alias_plan(
    project: Project,
    output_labels: Sequence[str],
    alias: AliasSpec,
    *,
    overwrite: bool = False,
) -> AliasPlan | None:
    labels = tuple(str(x) for x in output_labels)
    if not labels:
        raise AliasBundleError(
            "An aliased operation must declare at least one output label."
        )
    if alias is None:
        if overwrite:
            raise AliasBundleError("overwrite=True requires alias=...")
        return None

    if len(labels) == 1:
        if not isinstance(alias, str):
            raise AliasBundleError(
                f"Single-output operation {labels!r} requires alias='name', not a mapping."
            )
        aliases = {labels[0]: alias}
    else:
        if isinstance(alias, str) or not isinstance(alias, Mapping):
            raise AliasBundleError(
                "Multi-output operations require alias={output_label: alias, ...}."
            )
        aliases = {str(k): str(v) for k, v in alias.items()}
        expected = set(labels)
        actual = set(aliases)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise AliasBundleError(
                "Multi-output alias mapping must cover the complete output bundle; "
                f"missing={missing}, extra={extra}."
            )

    if len(set(aliases.values())) != len(aliases):
        raise AliasBundleError("Each output must use a distinct artifact alias.")
    for value in aliases.values():
        project.catalog.validate_artifact_alias(value)

    existing: dict[str, str] = {}
    missing: list[str] = []
    for label in labels:
        name = aliases[label]
        try:
            artifact = project.get_artifact(name)
        except ArtifactNotFoundError:
            missing.append(label)
        else:
            artifact.require_complete()
            existing[label] = artifact.artifact_id

    if existing and missing:
        raise AliasBundleError(
            "Aliased output bundle is only partially present. TeAL will not execute or "
            f"overwrite a partial bundle; found={sorted(existing)}, missing={sorted(missing)}."
        )

    if existing:
        if len(set(existing.values())) != len(existing):
            raise AliasBundleError(
                "Multiple requested output aliases currently resolve to the same artifact; "
                "the bundle is inconsistent and cannot be reused safely."
            )
        if not overwrite:
            _print_reuse(project, labels, aliases, existing)
            return AliasPlan(labels, aliases, False, existing, reuse=True)
        _check_overwrite_safety(project, aliases, existing)
        return AliasPlan(labels, aliases, True, existing, reuse=False)

    return AliasPlan(labels, aliases, bool(overwrite), {}, reuse=False)


def reused_outputs(project: Project, plan: AliasPlan) -> dict[str, BaseArtifact]:
    return {
        label: project.get_artifact(plan.aliases[label]) for label in plan.output_labels
    }


def finalize_alias_plan(
    project: Project,
    plan: AliasPlan | None,
    outputs: Mapping[str, BaseArtifact],
) -> None:
    if plan is None or plan.reuse:
        return
    if set(outputs) != set(plan.output_labels):
        raise AliasBundleError(
            f"Operation outputs {sorted(outputs)} do not match aliased bundle "
            f"{sorted(plan.output_labels)}."
        )
    bindings = {
        plan.aliases[label]: outputs[label].artifact_id for label in plan.output_labels
    }
    expected = {
        plan.aliases[label]: plan.existing_artifact_ids.get(label)
        for label in plan.output_labels
    }
    if plan.existing_artifact_ids:
        # Recheck immediately before the destructive catalog transaction.
        _check_overwrite_safety(project, plan.aliases, plan.existing_artifact_ids)
        project.catalog.replace_artifact_alias_bundle(
            bindings,
            expected_existing=expected,
            retire_artifact_ids=set(plan.existing_artifact_ids.values()),
        )
        print(
            "Rebuilt aliased output bundle and retired the previous artifact(s): "
            + ", ".join(f"{a!r}->{bindings[a]}" for a in bindings)
        )
    else:
        project.catalog.replace_artifact_alias_bundle(
            bindings,
            expected_existing=expected,
            retire_artifact_ids=set(),
        )


def _check_overwrite_safety(
    project: Project,
    aliases: Mapping[str, str],
    existing: Mapping[str, str],
) -> None:
    bundle_ids = set(existing.values())
    allowed_aliases_by_id: dict[str, set[str]] = {}
    for label, artifact_id in existing.items():
        allowed_aliases_by_id.setdefault(artifact_id, set()).add(aliases[label])

    problems: list[str] = []
    for artifact_id in sorted(bundle_ids):
        extra_aliases = (
            set(project.catalog.aliases_for_artifact(artifact_id))
            - allowed_aliases_by_id[artifact_id]
        )
        if extra_aliases:
            problems.append(f"{artifact_id} also has aliases {sorted(extra_aliases)}")
        external = [
            row
            for row in project.catalog.artifact_dependents(artifact_id)
            if str(row["artifact_id"]) not in bundle_ids
        ]
        if external:
            details = [
                f"{row['artifact_id']} ({row.get('label', 'output')})"
                for row in external
            ]
            problems.append(f"{artifact_id} has live dependents {details}")
    if problems:
        raise AliasOverwriteBlockedError(
            "Cannot overwrite aliased artifact bundle because replacement would break "
            "durable project references. " + "; ".join(problems)
        )


def _print_reuse(
    project: Project,
    labels: Sequence[str],
    aliases: Mapping[str, str],
    existing: Mapping[str, str],
) -> None:
    if len(labels) == 1:
        label = labels[0]
        artifact = project.get_artifact(existing[label])
        print(
            f"Alias {aliases[label]!r} already exists; returning existing artifact "
            f"{artifact.artifact_id}. This operation was not executed and the current "
            "settings were ignored. Pass overwrite=True to rebuild this alias."
        )
        return
    print(
        "All requested output aliases already exist; returning the existing output bundle. "
        "This operation was not executed and the current settings were ignored. "
        "Pass overwrite=True to rebuild the entire bundle."
    )
    for label in labels:
        print(f"  {label}: {aliases[label]!r} -> {existing[label]}")
