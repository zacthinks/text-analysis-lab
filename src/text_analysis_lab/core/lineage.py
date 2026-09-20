"""Artifact lineage helpers for TeAL artifacts.

This module is the single source of truth for artifact-basis lineage: which
artifact(s) another artifact is defined on, and how their keys relate.

Operation provenance -- which operator run created an artifact and which source
artifacts an operation consumed -- is recorded separately in the project catalog.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, get_args

from text_analysis_lab.core.errors import (
    DataInheritanceError,
    LineageError,
    MissingDataComponentError,
)
from text_analysis_lab.core.types import ArtifactType, LineageMode

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


LINEAGE_MODES: tuple[str, ...] = get_args(LineageMode)

BUBBLING_LINEAGE_MODES: frozenset[LineageMode] = frozenset(
    {"preserved_key", "extended_key"}
)

NON_BUBBLING_LINEAGE_MODES: frozenset[LineageMode] = frozenset(
    {"reduced_key", "span_key"}
)

BREAKING_LINEAGE_MODES: frozenset[LineageMode] = frozenset({"new_key"})


def validate_lineage_mode(mode: str) -> LineageMode:
    """Validate and return an artifact lineage mode value."""
    if mode not in LINEAGE_MODES:
        raise LineageError(
            f"Invalid lineage mode {mode!r}. Expected one of {list(LINEAGE_MODES)}."
        )
    return cast(LineageMode, mode)


def artifact_lineage(artifact: "BaseArtifact") -> dict[str, Any]:
    """Return the lineage descriptor for an artifact."""
    lineage = artifact.descriptor.get("lineage")
    if lineage is None:
        raise LineageError(
            f"Artifact {artifact!r} has no lineage information; cannot determine lineage."
        )
    if not isinstance(lineage, dict):
        raise LineageError(
            f"Artifact {artifact!r} lineage information must be a mapping; got {type(lineage).__name__}."
        )
    return dict(lineage)


def lineage_mode_for_artifact(artifact: "BaseArtifact") -> LineageMode:
    """Return the recorded basis-lineage mode for an artifact descriptor."""
    lineage = artifact_lineage(artifact)
    return validate_lineage_mode(str(lineage.get("lineage_mode", "MISSING")))


def basis_artifact_ids(artifact: "BaseArtifact") -> tuple[str, ...]:
    """Return the artifact IDs this artifact is defined on."""
    lineage = artifact_lineage(artifact)

    if "basis_artifact_ids" in lineage:
        value = lineage.get("basis_artifact_ids")
        if value is None:
            return ()
        if isinstance(value, str):
            raise LineageError(
                "lineage['basis_artifact_ids'] must be a sequence of artifact IDs, not a string."
            )
        if not isinstance(value, Sequence):
            raise LineageError(
                "lineage['basis_artifact_ids'] must be a sequence of artifact IDs."
            )
        return tuple(str(item) for item in value)
    else:
        raise LineageError(
            f"Artifact {artifact!r} has no basis_artifact_ids in its lineage descriptor."
        )


def is_key_subset(
    candidate_key: Sequence[str],
    current_key: Sequence[str],
) -> bool:
    """Return True when every candidate key column is present in current_key."""
    candidate = tuple(candidate_key)
    if not candidate:
        return False
    current = set(tuple(current_key))
    return all(col in current for col in candidate)


def is_prefix(
    prefix: Sequence[str],
    full: Sequence[str],
) -> bool:
    """Return True when ``prefix`` is an ordered prefix of ``full``."""
    prefix_tuple = tuple(prefix)
    full_tuple = tuple(full)
    return (
        len(prefix_tuple) <= len(full_tuple)
        and full_tuple[: len(prefix_tuple)] == prefix_tuple
    )


def lineage_paths_to_ancestor(
    project: "Project",
    descendant: "BaseArtifact",
    ancestor: "BaseArtifact",
) -> list[tuple["BaseArtifact", ...]]:
    """Return all basis-lineage paths from ``descendant`` to ``ancestor``.

    Paths include both endpoints. ``new_key`` lineage is a hard boundary and is
    never traversed. The helper is intentionally structural: callers remain
    responsible for deciding whether a particular path supports metadata/data
    inheritance.
    """
    target_id = str(ancestor.artifact_id)
    paths: list[tuple["BaseArtifact", ...]] = []

    def walk(current: "BaseArtifact", path: tuple["BaseArtifact", ...]) -> None:
        current_id = str(current.artifact_id)
        if any(str(item.artifact_id) == current_id for item in path):
            raise LineageError(
                f"Cycle detected in artifact lineage at artifact {current_id!r}."
            )
        next_path = (*path, current)
        if current_id == target_id:
            paths.append(next_path)
            return
        if lineage_mode_for_artifact(current) in BREAKING_LINEAGE_MODES:
            return
        for basis_id in basis_artifact_ids(current):
            try:
                basis = project.get_artifact(basis_id)
            except Exception as exc:
                raise LineageError(
                    f"Artifact {current_id!r} declares basis artifact {basis_id!r}, "
                    "but that basis artifact could not be loaded."
                ) from exc
            walk(basis, next_path)

    walk(descendant, ())
    return paths


def iter_metadata_lineage_sources(
    project: "Project",
    artifact: "BaseArtifact",
) -> list["BaseArtifact"]:
    """Return artifacts whose local metadata can attach to ``artifact``.

    Ordinary lineage keeps the historical key-compatibility rules. A
    ``rekeyed_key`` edge is the one exception: the immediate basis represents
    exactly the same rows in the same order under a different key system, so
    traversal crosses that edge and resets compatibility checks to the basis
    key space.

    A prior non-bubbling reduction/span followed by a rekey is deliberately a
    boundary for older metadata: once row identity has been collapsed, there is
    no unique positional correspondence back across the rekey.
    """
    metadata_sources: list[BaseArtifact] = []
    included_artifact_ids: set[str] = set()

    def include_if_has_metadata(candidate: "BaseArtifact") -> None:
        if (
            candidate.artifact_id not in included_artifact_ids
            and candidate.has_metadata()
        ):
            metadata_sources.append(candidate)
            included_artifact_ids.add(candidate.artifact_id)

    include_if_has_metadata(artifact)

    def walk(
        current: "BaseArtifact",
        *,
        compatibility_only: bool,
        target_key: tuple[str, ...],
        path: frozenset[str],
    ) -> None:
        current_id = current.artifact_id
        if current_id in path:
            raise LineageError(
                f"Cycle detected in artifact lineage at artifact {current_id!r}."
            )
        next_path = frozenset((*path, current_id))

        mode = lineage_mode_for_artifact(current)
        if mode in BREAKING_LINEAGE_MODES:
            return

        basis_ids = basis_artifact_ids(current)
        if not basis_ids:
            raise LineageError(
                f"Artifact {current_id!r} has lineage_mode={mode!r} "
                "but no basis_artifact_ids."
            )

        if mode in {"merged_key", "joined_key"}:
            if len(basis_ids) < 2:
                raise LineageError(
                    f"Artifact {current_id!r} has {mode} lineage but only "
                    f"{len(basis_ids)} basis artifact(s)."
                )
        elif len(basis_ids) != 1:
            return

        for basis_id in basis_ids:
            try:
                basis = project.get_artifact(basis_id)
            except Exception as exc:
                raise LineageError(
                    f"Artifact {current_id!r} declares basis artifact {basis_id!r}, "
                    "but that basis artifact could not be loaded."
                ) from exc

            basis_key = tuple(str(col) for col in basis.primary_key)

            if mode == "rekeyed_key":
                if compatibility_only:
                    # A reduced/span row universe cannot be uniquely translated
                    # backward through an earlier one-to-one rekey boundary.
                    continue
                include_if_has_metadata(basis)
                walk(
                    basis,
                    compatibility_only=False,
                    target_key=basis_key,
                    path=next_path,
                )
                continue

            key_compatible = is_key_subset(basis_key, target_key)
            next_compatibility_only = compatibility_only

            if compatibility_only:
                if key_compatible:
                    include_if_has_metadata(basis)

            elif mode in BUBBLING_LINEAGE_MODES or mode in {"merged_key", "joined_key"}:
                if not key_compatible:
                    raise LineageError(
                        f"Artifact {basis_id!r} is reached through {mode!r}, "
                        f"but its primary key {list(basis_key)} is not compatible with "
                        f"target key space {list(target_key)}."
                    )
                include_if_has_metadata(basis)

            elif mode in NON_BUBBLING_LINEAGE_MODES:
                next_compatibility_only = True

            else:
                raise LineageError(f"Unsupported lineage mode {mode!r}.")

            walk(
                basis,
                compatibility_only=next_compatibility_only,
                target_key=target_key,
                path=next_path,
            )

    walk(
        artifact,
        compatibility_only=False,
        target_key=tuple(str(col) for col in artifact.primary_key),
        path=frozenset(),
    )
    return metadata_sources


def expected_span_key(source_primary_key: Sequence[str]) -> list[str]:
    """Return the canonical output key for a span recomposition of source keys."""
    source = [str(col) for col in source_primary_key]
    if not source:
        raise LineageError("span_key lineage requires a non-empty basis primary key.")
    last = source[-1]
    if last.endswith("_start") or last.endswith("_end"):
        raise LineageError(
            "span_key lineage cannot be applied to an existing span key."
        )
    return [*source[:-1], f"{last}_start", f"{last}_end"]


def validate_primary_key_relationship(
    *,
    basis_keys: Sequence[Sequence[str]],
    output_key: Sequence[str],
    lineage_mode: LineageMode | str,
) -> None:
    """Validate that an output primary-key schema satisfies its lineage mode."""
    mode = validate_lineage_mode(str(lineage_mode))

    output_pk = tuple(output_key)
    if not output_pk:
        raise LineageError("Output primary key must contain at least one column.")

    if mode == "new_key":
        return

    basis_pks = tuple(tuple(basis_key) for basis_key in basis_keys)

    if mode == "rekeyed_key":
        if len(basis_pks) != 1:
            raise LineageError(
                f"rekeyed_key lineage requires exactly one basis key; got {len(basis_pks)}."
            )
        # Rekeying intentionally replaces the key namespace, so there is no
        # schema relationship to validate beyond the single-basis invariant.
        return

    if mode in {"merged_key", "joined_key"}:
        if len(basis_pks) < 2:
            raise LineageError(
                f"{mode} lineage requires at least two basis keys; got {len(basis_pks)}."
            )
        if len(set(basis_pks)) != 1:
            raise LineageError(
                f"{mode} lineage requires every basis artifact to use the same "
                f"primary-key schema; got {basis_pks}."
            )
        if output_pk != basis_pks[0]:
            raise LineageError(
                f"{mode} output primary key must match its basis artifacts: "
                f"expected {list(basis_pks[0])}, got {list(output_pk)}."
            )
        return

    if len(basis_pks) != 1:
        raise LineageError(
            f"{mode} lineage requires exactly one basis key; got {len(basis_pks)}."
        )

    basis_pk = basis_pks[0]

    if mode == "preserved_key":
        if output_pk != basis_pk:
            raise LineageError(
                f"preserved_key output must keep basis primary key {list(basis_pk)}; "
                f"got {list(output_pk)}."
            )
        return

    if mode == "extended_key":
        if not is_prefix(basis_pk, output_pk) or len(output_pk) == len(basis_pk):
            raise LineageError(
                f"extended_key output primary key must be a proper extension of "
                f"basis key {list(basis_pk)}; got {list(output_pk)}."
            )
        return

    if mode == "reduced_key":
        if not is_prefix(output_pk, basis_pk) or len(output_pk) == len(basis_pk):
            raise LineageError(
                f"reduced_key output primary key must be a proper retained prefix "
                f"of basis key {list(basis_pk)}; got {list(output_pk)}."
            )
        return

    if mode == "span_key":
        expected = expected_span_key(basis_pk)
        if list(output_pk) != expected:
            raise LineageError(
                f"span_key output primary key must be {expected} for basis key "
                f"{list(basis_pk)}; got {list(output_pk)}."
            )
        return

    raise LineageError(f"Unsupported lineage mode {mode!r}.")


def find_data_artifact(artifact: "BaseArtifact") -> "BaseArtifact | None":
    """Return the nearest artifact that can provide representation data.

    Representation data inheritance is intentionally stricter than metadata
    inheritance. Data can only be inherited through row-preserving single-basis
    ``preserved_key`` or ``rekeyed_key`` edges and only while artifact type remains
    unchanged. ``rekeyed_key`` preserves row identity by exact position rather than
    by key equality.
    A table/JSONL ``merged_key`` artifact is a virtual relational data provider:
    it still owns no physical data component, but QueryEngine can resolve its
    branch data lazily. Returning that merge artifact here lets preserved-key
    descendants continue to inherit the merged logical representation.
    """
    seen: set[str] = set()
    current = artifact

    while True:
        if current.artifact_id in seen:
            raise DataInheritanceError(
                f"Cycle detected while resolving data artifact for {artifact.artifact_id}."
            )
        seen.add(current.artifact_id)

        if current.has_own_data():
            return current

        lineage = artifact_lineage(current)
        mode = lineage.get("lineage_mode")
        if mode in {"merged_key", "joined_key"}:
            if current.artifact_type in {ArtifactType.TABLE, ArtifactType.JSONL}:
                return current
            return None

        if mode not in {"preserved_key", "rekeyed_key"}:
            # Data is optional. Only row-preserving lineage can inherit the same
            # representation data lazily. Rekeying preserves rows positionally.
            return None

        basis_ids = basis_artifact_ids(current)
        if not basis_ids:
            raise MissingDataComponentError(
                f"Artifact {artifact.artifact_id} has no owned data component and "
                f"no basis artifact from which to inherit data."
            )
        if len(basis_ids) != 1:
            raise DataInheritanceError(
                f"Artifact {artifact.artifact_id} cannot inherit data through "
                f"{current.artifact_id}: data inheritance requires exactly one basis artifact, "
                f"got {len(basis_ids)}."
            )

        current = current.project.get_artifact(basis_ids[0])

        if current.artifact_type != artifact.artifact_type:
            raise DataInheritanceError(
                f"Artifact {artifact.artifact_id} cannot inherit data from "
                f"{current.artifact_id}: artifact type changed from "
                f"{artifact.artifact_type!r} to {current.artifact_type!r}."
            )
