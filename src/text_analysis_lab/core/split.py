"""Random split operations backed by the TeAL translation runner.

The public operation is ``Project.split(...)``. Internally, a split is a full-
artifact translation: materialize source keys, optionally materialize resolved
stratification columns, assign source rows to output labels, and write keys-only
child artifacts that inherit representation data through preserved-key lineage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError, QueryError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_SOURCE_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


def split(
    project: Project,
    source: BaseArtifact | str,
    *,
    labels: Sequence[str] = ("explore", "confirm"),
    proportions: Sequence[float],
    random_state: int | None = None,
    stratify: str | Sequence[str] | None = None,
    workers: int = 1,
    memo: str | None = None,
    alias: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Mapping[str, BaseArtifact]:
    """Split one source artifact into keys-only child artifacts.

    ``labels``, ``proportions``, and ``random_state`` configure the reusable
    random-split operator. ``stratify`` is resolved against the source bound to
    this operation and may name key, data, or metadata output columns.
    """
    translator = RandomSplitTranslator(
        labels=labels,
        proportions=proportions,
        random_state=random_state,
    )
    return project.translate(
        translator,
        source,
        workers=workers,
        memo=memo,
        alias=alias,
        overwrite=overwrite,
        stratify=stratify,
    )


class RandomSplitTranslator(BaseTranslator):
    """Translator-backed random split.

    The translator always requests the source as one full-artifact packet. That
    keeps split proportions exact over the whole source, and exact within each
    stratum when stratification is requested.
    """

    operation_type = "split"

    def __init__(
        self,
        *,
        labels: Sequence[str],
        proportions: Sequence[float],
        random_state: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.labels: tuple[str, ...] = _validate_labels(labels)
        self.proportions: tuple[float, ...] = _validate_proportions(
            proportions,
            expected_count=len(self.labels),
        )
        self.random_state: int | None = _validate_random_state(random_state)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        source = _source_artifact(sources)
        return {
            label: OutputSpec(
                artifact_type=source.artifact_type,
                lineage_mode="preserved_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            )
            for label in self.labels
        }

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        unknown = sorted(set(params) - {"stratify"})
        if unknown:
            raise OperatorError(f"Unknown split operation parameter(s): {unknown}.")

        source = _source_artifact(sources)
        raw_stratify = params.get("stratify")
        stratify_columns, column_request = _resolve_stratify_columns(
            source=source,
            stratify=cast(str | Sequence[str] | None, raw_stratify),
        )
        requested = (
            None
            if raw_stratify is None
            else list(
                _as_tuple(cast(str | Sequence[str], raw_stratify), name="stratify")
            )
        )
        return {
            "stratify": requested,
            "stratify_columns": list(stratify_columns),
            "column_request": {
                "keys": column_request.keys,
                "data": column_request.data,
                "metadata": column_request.metadata,
            },
        }

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _source_artifact(sources)
        raw_columns = request.params.get("column_request", {})
        if not isinstance(raw_columns, Mapping):
            raise OperatorError("Split operation column_request must be a mapping.")
        columns = ColumnRequest(
            keys=raw_columns.get("keys", True),
            data=raw_columns.get("data", False),
            metadata=raw_columns.get("metadata", False),
        )
        return SourceRequest(
            artifact_type=source.artifact_type,
            mode="full_artifact",
            columns=columns,
            batch_size=None,
            form="table",
            metadata_mode="full" if columns.metadata is not False else "none",
            include_position=True,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = mode
        if set(inputs) != {DEFAULT_SOURCE_LABEL}:
            raise OperatorError(
                f"RandomSplitTranslator expected one input label {DEFAULT_SOURCE_LABEL!r}; "
                f"got {tuple(inputs)}."
            )

        source_batch = inputs[DEFAULT_SOURCE_LABEL]
        frame = _require_table_packet(source_batch.data)
        frame = _sort_by_position_if_present(frame).reset_index(drop=True)
        keys = _extract_key_frame(frame, primary_key=source_batch.primary_key)
        stratify_columns = tuple(
            str(column) for column in request.params.get("stratify_columns", ())
        )

        groups = _groups_for_frame(frame, stratify_columns=stratify_columns)
        assignments = _assign_split_indices(
            labels=self.labels,
            proportions=self.proportions,
            groups=groups,
            random_state=self.random_state,
        )

        outputs: dict[str, dict[str, pd.DataFrame]] = {}
        for label in self.labels:
            selected = sorted(assignments[label])
            outputs[label] = {
                "keys": keys.iloc[selected].reset_index(drop=True),
            }
        return BatchResult(outputs=outputs)

    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def to_json_state(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "proportions": list(self.proportions),
            "random_state": self.random_state,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> RandomSplitTranslator:
        return cls(
            labels=cast(Sequence[str], state["labels"]),
            proportions=cast(Sequence[float], state["proportions"]),
            random_state=cast(int | None, state.get("random_state")),
        )


# ---------------------------------------------------------------------------
# Stratification resolution
# ---------------------------------------------------------------------------


def _source_artifact(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            "RandomSplitTranslator requires exactly one source under "
            f"{DEFAULT_SOURCE_LABEL!r}; got {tuple(sources)}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _resolve_stratify_columns(
    *,
    source: BaseArtifact,
    stratify: str | Sequence[str] | None,
) -> tuple[tuple[str, ...], ColumnRequest]:
    if stratify is None:
        return (), ColumnRequest(keys=True, data=False, metadata=False)

    requested = _as_tuple(stratify, name="stratify")
    try:
        info = source.project.query.query_columns(source, metadata_mode="full")
    except Exception as exc:
        raise ArtifactError(
            f"Cannot inspect columns for artifact {source.artifact_id!r}."
        ) from exc

    output_names = set(info.get("output", ()))
    ambiguous = info.get("ambiguous", {})
    output_to_namespace = {
        column["output_name"]: column["namespace"] for column in info.get("columns", ())
    }

    stratify_columns: list[str] = []
    data_columns: list[str] = []
    metadata_columns: list[str] = []

    for name in requested:
        if name not in output_names:
            if name in ambiguous:
                raise QueryError(
                    f"Stratify column {name!r} is ambiguous for artifact "
                    f"{source.artifact_id!r}. Use one of these qualified names "
                    f"instead: {ambiguous[name]}."
                )
            raise QueryError(
                f"Stratify column {name!r} is not available for artifact "
                f"{source.artifact_id!r}. Available output columns are: "
                f"{sorted(output_names)}."
            )

        namespace = output_to_namespace[name]
        stratify_columns.append(name)
        if namespace == "key":
            continue
        if namespace == "data":
            data_columns.append(name)
            continue
        if namespace == "metadata":
            metadata_columns.append(name)
            continue
        raise QueryError(
            f"Stratify column {name!r} has unsupported namespace {namespace!r}."
        )

    if len(set(stratify_columns)) != len(stratify_columns):
        raise ValueError(f"Duplicate stratify columns: {stratify_columns!r}.")

    return (
        tuple(stratify_columns),
        ColumnRequest(
            keys=True,
            data=data_columns or False,
            metadata=metadata_columns or False,
        ),
    )


# ---------------------------------------------------------------------------
# Split assignment helpers
# ---------------------------------------------------------------------------


def _assign_split_indices(
    *,
    labels: tuple[str, ...],
    proportions: tuple[float, ...],
    groups: Mapping[tuple[Any, ...], Sequence[int]],
    random_state: int | None,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(random_state)
    assignments: dict[str, list[int]] = {label: [] for label in labels}

    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=int).copy()
        rng.shuffle(shuffled)
        counts = _counts_from_proportions(len(shuffled), proportions)
        start = 0
        for label, count in zip(labels, counts, strict=True):
            stop = start + count
            assignments[label].extend(int(idx) for idx in shuffled[start:stop].tolist())
            start = stop

    return assignments


def _counts_from_proportions(n: int, proportions: Sequence[float]) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative.")
    raw = np.asarray(proportions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw - counts), kind="stable")
        for idx in fractional_order[:remainder]:
            counts[int(idx)] += 1
    return [int(value) for value in counts.tolist()]


def _groups_for_frame(
    frame: pd.DataFrame,
    *,
    stratify_columns: tuple[str, ...],
) -> dict[tuple[Any, ...], list[int]]:
    if not stratify_columns:
        return {("__all__",): list(range(len(frame)))}

    missing = [column for column in stratify_columns if column not in frame.columns]
    if missing:
        raise ArtifactError(
            f"Split packet is missing resolved stratify column(s) {missing}. "
            f"Available columns: {list(frame.columns)}."
        )

    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(
        frame.loc[:, list(stratify_columns)].itertuples(index=False)
    ):
        groups[tuple(_stratum_value(value) for value in row)].append(idx)
    return dict(groups)


def _stratum_value(value: Any) -> Any:
    return None if pd.isna(value) else value


# ---------------------------------------------------------------------------
# Packet/key helpers
# ---------------------------------------------------------------------------


def _require_table_packet(packet: Any) -> pd.DataFrame:
    if not isinstance(packet, pd.DataFrame):
        raise ArtifactError(
            f"RandomSplitTranslator requires form='table'; got packet type "
            f"{type(packet).__name__}."
        )
    return packet


def _sort_by_position_if_present(frame: pd.DataFrame) -> pd.DataFrame:
    if "_position" not in frame.columns:
        return frame
    return frame.sort_values("_position", kind="stable")


def _extract_key_frame(
    frame: pd.DataFrame, *, primary_key: tuple[str, ...]
) -> pd.DataFrame:
    missing = [column for column in primary_key if column not in frame.columns]
    if missing:
        raise ArtifactError(
            f"Split packet is missing primary key column(s) {missing}. "
            f"Available columns: {list(frame.columns)}."
        )
    keys = frame.loc[:, list(primary_key)].copy().reset_index(drop=True)
    if keys.isna().any().any():
        raise ArtifactError("Split key frame contains NA primary key values.")
    return keys


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_tuple(value: str | Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        out = (value,)
    else:
        out = tuple(str(item) for item in value)
    if not out:
        raise ValueError(f"{name} must contain at least one value.")
    if any(not isinstance(item, str) or not item for item in out):
        raise ValueError(f"{name} must contain non-empty strings.")
    return out


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    out = _as_tuple(labels, name="labels")
    if len(set(out)) != len(out):
        raise ValueError(f"Split labels must be unique: {out!r}.")
    if DEFAULT_SOURCE_LABEL in out:
        raise ValueError(
            f"Split label {DEFAULT_SOURCE_LABEL!r} is reserved for the input source label."
        )
    return out


def _validate_proportions(
    proportions: Sequence[float],
    *,
    expected_count: int,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in proportions)
    if len(values) != expected_count:
        raise ValueError(
            f"labels and proportions must have the same length; got "
            f"{expected_count} labels and {len(values)} proportions."
        )
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("proportions must be finite non-negative numbers.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("At least one split proportion must be positive.")
    return tuple((arr / total).tolist())


def _validate_random_state(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("random_state must be an integer or None.")
    return int(value)
