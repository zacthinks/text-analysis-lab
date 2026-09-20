"""Probability audit sampling for finite-corpus supervised measurement.

This operation differs from ordinary ``Project.split``: it draws one fixed-size
sample, optionally with disproportionate stratum allocation, and materializes the
realized inclusion probability for every sampled observation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
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

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


_DOCUMENTS = "documents"
_STRATA = "strata"
_PI_LABEL = "pi"


def probability_split(
    project: Project,
    source: BaseArtifact | str,
    *,
    n: int,
    remainder_label: str = "remainder",
    sample_label: str = "sample",
    strata: BaseArtifact | str | None = None,
    allocation: Mapping[Any, float] | None = None,
    random_state: int | None = None,
    workers: int = 1,
    memo: str | None = None,
    alias: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Mapping[str, BaseArtifact]:
    """Draw a fixed-size probability audit sample without replacement.

    With no ``strata``, every source observation has inclusion probability
    ``n / N``.  With a one-column same-key strata artifact, ``allocation`` gives
    relative requested sample shares; realized integer counts are obtained by
    deterministic largest-remainder rounding and ``pi_i = n_h / N_h``.
    """
    if (strata is None) != (allocation is None):
        if strata is None:
            raise ValueError("allocation must be None when strata=None.")
        raise ValueError("allocation is required when strata is supplied.")

    translator = ProbabilitySplitTranslator(
        n=n,
        remainder_label=remainder_label,
        sample_label=sample_label,
        allocation=allocation,
        random_state=random_state,
    )
    sources: dict[str, BaseArtifact | str] = {_DOCUMENTS: source}
    if strata is not None:
        sources[_STRATA] = strata
    return project.translate(
        translator,
        sources,
        workers=workers,
        memo=memo,
        alias=alias,
        overwrite=overwrite,
    )


class ProbabilitySplitTranslator(BaseTranslator):
    """Full-artifact translator implementing fixed-size probability sampling."""

    operation_type = "split"

    def __init__(
        self,
        *,
        n: int,
        remainder_label: str = "remainder",
        sample_label: str = "sample",
        allocation: Mapping[Any, float] | Sequence[tuple[Any, float]] | None = None,
        random_state: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.n = _positive_int(n, name="n")
        self.remainder_label = _output_label(remainder_label, name="remainder_label")
        self.sample_label = _output_label(sample_label, name="sample_label")
        if self.remainder_label == self.sample_label:
            raise ValueError("remainder_label and sample_label must be different.")
        if _PI_LABEL in {self.remainder_label, self.sample_label}:
            raise ValueError(
                "remainder_label and sample_label cannot use reserved label 'pi'."
            )
        if {_DOCUMENTS, _STRATA}.intersection(
            {self.remainder_label, self.sample_label}
        ):
            raise ValueError(
                "remainder_label/sample_label cannot use internal source labels "
                f"{_DOCUMENTS!r} or {_STRATA!r}."
            )
        self.allocation = _normalize_allocation(allocation)
        self.random_state = _random_state(random_state)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        source = _documents_source(sources)
        _validate_source_shape(sources)
        return {
            self.remainder_label: OutputSpec(
                artifact_type=source.artifact_type,
                lineage_mode="preserved_key",
                basis_labels=_DOCUMENTS,
            ),
            self.sample_label: OutputSpec(
                artifact_type=source.artifact_type,
                lineage_mode="preserved_key",
                basis_labels=_DOCUMENTS,
            ),
            _PI_LABEL: OutputSpec(
                artifact_type="table",
                lineage_mode="preserved_key",
                basis_labels=self.sample_label,
            ),
        }

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        if params:
            raise OperatorError(
                "ProbabilitySplitTranslator does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        _validate_source_shape(sources)
        source = _documents_source(sources)
        strata = sources.get(_STRATA)
        if strata is None:
            if self.allocation is not None:
                raise OperatorError(
                    "Probability split allocation requires a strata source."
                )
        else:
            if self.allocation is None:
                raise OperatorError(
                    "Probability split strata source requires allocation."
                )
            if strata.artifact_type.value != "table":
                raise OperatorError(
                    "Probability split strata must be a table artifact."
                )
            if tuple(strata.primary_key) != tuple(source.primary_key):
                raise ArtifactError(
                    "Probability split strata must use the same primary-key columns as source."
                )
            columns = [str(value) for value in strata.get_data_columns()]
            if len(columns) != 1:
                raise ArtifactError(
                    "Probability split strata must be a one-column same-key table artifact; "
                    f"got data columns {columns}."
                )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> Mapping[str, SourceRequest]:
        _ = mode, request
        source = _documents_source(sources)
        result: dict[str, SourceRequest] = {
            _DOCUMENTS: SourceRequest(
                artifact_type=source.artifact_type,
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=False, metadata=False),
                batch_size=None,
                form="table",
                metadata_mode="none",
                include_position=True,
            )
        }
        if _STRATA in sources:
            result[_STRATA] = SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=True, metadata=False),
                batch_size=None,
                form="table",
                metadata_mode="none",
                include_position=False,
            )
        return result

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Unsupported probability split mode {mode!r}.")
        expected = {_DOCUMENTS} if self.allocation is None else {_DOCUMENTS, _STRATA}
        if set(inputs) != expected:
            raise OperatorError(
                f"Probability split expected input labels {sorted(expected)}; got {sorted(inputs)}."
            )

        source_packet = inputs[_DOCUMENTS]
        source_frame = _table(source_packet.data, name="documents")
        source_frame = _sort_source(source_frame).reset_index(drop=True)
        keys = _key_frame(source_frame, source_packet.primary_key, name="documents")
        _require_unique_keys(keys, source_packet.primary_key, name="documents")
        N = len(keys)
        if self.n >= N:
            raise ValueError(
                f"probability_split requires 0 < n < N so both sample and remainder "
                f"are non-empty; got n={self.n}, N={N}."
            )

        if self.allocation is None:
            selected, pi_by_index = _srs_indices(
                N=N,
                n=self.n,
                random_state=self.random_state,
            )
        else:
            strata_packet = inputs[_STRATA]
            strata_frame = _table(strata_packet.data, name="strata")
            strata_keys = _key_frame(
                strata_frame, strata_packet.primary_key, name="strata"
            )
            _require_unique_keys(strata_keys, strata_packet.primary_key, name="strata")
            data_columns = [
                column
                for column in strata_frame.columns
                if column not in set(strata_packet.primary_key)
                and column not in {"_position", "_batch", "_row_offset"}
            ]
            if len(data_columns) != 1:
                raise ArtifactError(
                    "Probability split strata packet must contain exactly one data column; "
                    f"got {data_columns}."
                )
            stratum_column = data_columns[0]
            aligned = _align_strata(
                source_keys=keys,
                strata_frame=strata_frame,
                primary_key=source_packet.primary_key,
                stratum_column=stratum_column,
            )
            selected, pi_by_index = _stratified_indices(
                aligned,
                n=self.n,
                allocation=self.allocation,
                random_state=self.random_state,
            )

        selected_set = set(selected)
        remainder = [index for index in range(N) if index not in selected_set]
        sample_keys = keys.iloc[selected].reset_index(drop=True)
        remainder_keys = keys.iloc[remainder].reset_index(drop=True)
        pi = pd.DataFrame({"pi": [float(pi_by_index[index]) for index in selected]})

        return BatchResult(
            outputs={
                self.remainder_label: {"keys": remainder_keys},
                self.sample_label: {"keys": sample_keys},
                _PI_LABEL: {"keys": sample_keys.copy(), "data": pi},
            }
        )

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
            "n": self.n,
            "remainder_label": self.remainder_label,
            "sample_label": self.sample_label,
            "allocation": None
            if self.allocation is None
            else [
                {"stratum": value, "weight": weight}
                for value, weight in self.allocation
            ],
            "random_state": self.random_state,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> ProbabilitySplitTranslator:
        raw = state.get("allocation")
        allocation = None
        if raw is not None:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise OperatorError(
                    "Probability split allocation state must be a sequence."
                )
            pairs: list[tuple[Any, float]] = []
            for record in raw:
                if not isinstance(record, Mapping):
                    raise OperatorError(
                        "Probability split allocation records must be mappings."
                    )
                pairs.append((record.get("stratum"), float(record["weight"])))
            allocation = pairs
        return cls(
            n=int(state["n"]),
            remainder_label=str(state.get("remainder_label", "remainder")),
            sample_label=str(state.get("sample_label", "sample")),
            allocation=allocation,
            random_state=state.get("random_state"),
        )


def _validate_source_shape(sources: Mapping[str, Any]) -> None:
    labels = set(sources)
    if labels not in ({_DOCUMENTS}, {_DOCUMENTS, _STRATA}):
        raise OperatorError(
            "Probability split requires source label 'documents' and optional label 'strata'; "
            f"got {sorted(labels)}."
        )


def _documents_source(sources: Mapping[str, Any]):
    _validate_source_shape(sources)
    return sources[_DOCUMENTS]


def _table(value: Any, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"Probability split {name} input must materialize as a table."
        )
    return value


def _sort_source(frame: pd.DataFrame) -> pd.DataFrame:
    if "_position" in frame.columns:
        return frame.sort_values("_position", kind="stable")
    return frame


def _key_frame(
    frame: pd.DataFrame, primary_key: Sequence[str], *, name: str
) -> pd.DataFrame:
    columns = [str(value) for value in primary_key]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ArtifactError(
            f"Probability split {name} input is missing key column(s) {missing}."
        )
    keys = frame.loc[:, columns].copy().reset_index(drop=True)
    if keys.isna().any().any():
        raise ArtifactError(f"Probability split {name} keys contain null values.")
    return keys


def _require_unique_keys(
    frame: pd.DataFrame, primary_key: Sequence[str], *, name: str
) -> None:
    columns = [str(value) for value in primary_key]
    if frame.duplicated(subset=columns).any():
        raise ArtifactError(
            f"Probability split {name} contains duplicate primary keys."
        )


def _key_tuples(
    frame: pd.DataFrame, primary_key: Sequence[str]
) -> list[tuple[int, ...]]:
    columns = [str(value) for value in primary_key]
    return [
        tuple(int(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def _align_strata(
    *,
    source_keys: pd.DataFrame,
    strata_frame: pd.DataFrame,
    primary_key: Sequence[str],
    stratum_column: str,
) -> list[Any]:
    source_tuples = _key_tuples(source_keys, primary_key)
    strata_tuples = _key_tuples(strata_frame, primary_key)
    source_set = set(source_tuples)
    strata_set = set(strata_tuples)
    if source_set != strata_set:
        missing = len(source_set - strata_set)
        extra = len(strata_set - source_set)
        raise ArtifactError(
            "Probability split strata key set must exactly equal the source key set; "
            f"missing={missing}, extra={extra}."
        )
    by_key = {
        key: _stratum_scalar(value)
        for key, value in zip(
            strata_tuples, strata_frame[stratum_column].tolist(), strict=True
        )
    }
    return [by_key[key] for key in source_tuples]


def _srs_indices(
    *, N: int, n: int, random_state: int | None
) -> tuple[list[int], dict[int, float]]:
    rng = np.random.default_rng(random_state)
    selected = sorted(
        int(value) for value in rng.choice(N, size=n, replace=False).tolist()
    )
    pi = float(n) / float(N)
    return selected, {index: pi for index in selected}


def _stratified_indices(
    strata: Sequence[Any],
    *,
    n: int,
    allocation: tuple[tuple[Any, float], ...],
    random_state: int | None,
) -> tuple[list[int], dict[int, float]]:
    groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    value_for_token: dict[tuple[str, Any], Any] = {}
    for index, value in enumerate(strata):
        token = _stratum_token(value)
        groups[token].append(index)
        value_for_token[token] = value

    requested: dict[tuple[str, Any], float] = {}
    for value, weight in allocation:
        token = _stratum_token(value)
        if token in requested:
            raise ValueError(f"Duplicate allocation stratum {value!r}.")
        requested[token] = float(weight)

    observed_tokens = set(groups)
    requested_tokens = set(requested)
    missing = observed_tokens - requested_tokens
    unknown = requested_tokens - observed_tokens
    if missing:
        values = [
            value_for_token[token] for token in sorted(missing, key=_token_sort_key)
        ]
        raise ValueError(f"allocation is missing observed stratum/strata {values!r}.")
    if unknown:
        values = [value for value, _ in allocation if _stratum_token(value) in unknown]
        raise ValueError(f"allocation contains unknown stratum/strata {values!r}.")

    ordered = sorted(observed_tokens, key=_token_sort_key)
    weights = np.asarray([requested[token] for token in ordered], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(
            "Every observed stratum must have a finite positive allocation weight."
        )
    raw = weights / weights.sum() * int(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder:
        fractional = raw - counts
        order = sorted(
            range(len(ordered)),
            key=lambda idx: (-float(fractional[idx]), _token_sort_key(ordered[idx])),
        )
        for idx in order[:remainder]:
            counts[idx] += 1

    for token, count in zip(ordered, counts.tolist(), strict=True):
        N_h = len(groups[token])
        value = value_for_token[token]
        if count <= 0:
            raise ValueError(
                f"Requested n={n} and allocation yield zero sampled observations for "
                f"observed stratum {value!r}; every observed stratum must have pi_i > 0."
            )
        if count > N_h:
            raise ValueError(
                f"Requested allocation requires n_h={count} observations from stratum "
                f"{value!r}, but only N_h={N_h} are available."
            )

    rng = np.random.default_rng(random_state)
    pi_by_index: dict[int, float] = {}
    selected: list[int] = []
    for token, count in zip(ordered, counts.tolist(), strict=True):
        pool = np.asarray(groups[token], dtype=int)
        chosen = rng.choice(pool, size=int(count), replace=False)
        N_h = len(pool)
        pi_h = float(count) / float(N_h)
        for index in chosen.tolist():
            idx = int(index)
            selected.append(idx)
            pi_by_index[idx] = pi_h
    selected.sort()
    return selected, pi_by_index


def _normalize_allocation(
    allocation: Mapping[Any, float] | Sequence[tuple[Any, float]] | None,
) -> tuple[tuple[Any, float], ...] | None:
    if allocation is None:
        return None
    items = (
        list(allocation.items())
        if isinstance(allocation, Mapping)
        else list(allocation)
    )
    if not items:
        raise ValueError("allocation must contain at least one stratum.")
    out: list[tuple[Any, float]] = []
    seen: set[tuple[str, Any]] = set()
    for value, raw_weight in items:
        scalar = _stratum_scalar(value)
        token = _stratum_token(scalar)
        if token in seen:
            raise ValueError(f"allocation contains duplicate stratum {scalar!r}.")
        seen.add(token)
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("allocation weights must be finite and strictly positive.")
        out.append((scalar, weight))
    return tuple(out)


def _stratum_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Probability split strata cannot contain null/NaN values.")
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Probability split strata must be finite scalars.")
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        "Probability split strata/allocation values must be scalar bool/int/float/str values."
    )


def _stratum_token(value: Any) -> tuple[str, Any]:
    scalar = _stratum_scalar(value)
    # Numeric strata should match ordinary Python/pandas scalar equality. This
    # matters for the Unit 6 H proxy, which may materialize as bool, integer, or
    # integral float while notebooks naturally specify allocation={0: ..., 1: ...}.
    if isinstance(scalar, bool):
        return ("number", int(scalar))
    if isinstance(scalar, int):
        return ("number", scalar)
    if isinstance(scalar, float):
        if scalar.is_integer():
            return ("number", int(scalar))
        return ("number", scalar)
    return ("str", scalar)


def _token_sort_key(token: tuple[str, Any]) -> tuple[str, str]:
    return (token[0], repr(token[1]))


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return out


def _random_state(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("random_state must be an integer or None.")
    return int(value)


def _output_label(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value
