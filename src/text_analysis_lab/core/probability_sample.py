"""Design-aware probability draws, distinct from the legacy probability_split API.

Inputs infer the design; an explicit design is only an assertion. The saved
operator state records the resolved design for deterministic historical replay.
"""

from __future__ import annotations

from collections import Counter
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
from text_analysis_lab.core.probability_split import (
    _key_frame,
    _key_tuples,
    _normalize_allocation,
    _output_label,
    _positive_int,
    _random_state,
    _require_unique_keys,
    _sort_source,
    _srs_indices,
    _stratified_indices,
    _stratum_scalar,
    _stratum_token,
    _table,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.project import Project


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("probability must be a finite numeric value.")
    p = float(value)
    if not np.isfinite(p) or not 0 < p <= 1:
        raise ValueError("probability must satisfy 0 < p <= 1.")
    return p


def _pairs(value, *, counts: bool = False):
    items = list(value.items()) if isinstance(value, Mapping) else list(value)
    if not items:
        raise ValueError("Stratum mapping cannot be empty.")
    out, seen = [], set()
    for raw_key, raw_value in items:
        key = _stratum_scalar(raw_key)
        token = _stratum_token(key)
        if token in seen:
            raise ValueError(f"Duplicate stratum {raw_key!r}.")
        seen.add(token)
        val = (
            _positive_int(raw_value, name=f"n_by_stratum[{key!r}]")
            if counts
            else _probability(raw_value)
        )
        out.append((key, val))
    return tuple(out)


def _aligned(frame_keys: pd.DataFrame, packet: InputBatch, *, label: str):
    frame = _table(packet.data, name=label)
    side_keys = _key_frame(frame, packet.primary_key, name=label)
    _require_unique_keys(side_keys, packet.primary_key, name=label)
    fields = [
        col for col in frame.columns
        if col not in set(packet.primary_key)
        and col not in {"_position", "_batch", "_row_offset"}
    ]
    if len(fields) != 1:
        raise ArtifactError(f"{label} must contain one data column.")
    left = _key_tuples(frame_keys, packet.primary_key)
    right = _key_tuples(side_keys, packet.primary_key)
    if set(left) != set(right):
        raise ArtifactError(f"{label} key set must equal the population key set.")
    values = dict(zip(right, frame[fields[0]].tolist(), strict=True))
    return [values[key] for key in left]


def probability_sample(
    project: Project,
    source: BaseArtifact | str,
    *,
    n: int | None = None,
    fraction: float | None = None,
    probability: float | Mapping[Any, float] | BaseArtifact | str | None = None,
    strata: BaseArtifact | str | None = None,
    allocation: str | Mapping[Any, float] | None = None,
    n_by_stratum: Mapping[Any, int] | None = None,
    design: str | None = None,
    remainder_label: str = "remainder",
    sample_label: str = "sample",
    random_state: int | None = None,
    workers: int = 1,
    memo: str | None = None,
    alias: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Mapping[str, BaseArtifact]:
    """Sample fixed-size SRS/stratified SRS, or independent Bernoulli units.

    n/fraction/n_by_stratum produce fixed-size without-replacement samples.
    probability is an independent inclusion probability (scalar, per-stratum
    mapping, or a full-frame single-field keyed artifact). A sample drawn with
    fraction has a fixed realized size, not a Bernoulli sample size.
    """
    artifact_probability = (
        probability is not None
        and not isinstance(
            probability, (Mapping, int, float, np.integer, np.floating)
        )
    )
    operator = ProbabilitySampleTranslator(
        n=n, fraction=fraction,
        probability=None if artifact_probability else probability,
        probability_source=artifact_probability, has_strata=strata is not None,
        allocation=allocation, n_by_stratum=n_by_stratum, design=design,
        remainder_label=remainder_label, sample_label=sample_label,
        random_state=random_state,
    )
    sources: dict[str, BaseArtifact | str] = {"documents": source}
    if strata is not None:
        sources["strata"] = strata
    if artifact_probability:
        sources["probabilities"] = probability
    return project.translate(
        operator, sources, workers=workers, memo=memo, alias=alias,
        overwrite=overwrite,
    )


class ProbabilitySampleTranslator(BaseTranslator):
    """Frozen operator with a fully resolved probability-sampling design."""

    operation_type = "split"

    def __init__(
        self,
        *,
        n: int | None = None,
        fraction: float | None = None,
        probability: float | Mapping[Any, float] | None = None,
        probability_source: bool = False,
        has_strata: bool = False,
        allocation: str | Mapping[Any, float] | Sequence[tuple[Any, float]] | None = None,
        n_by_stratum: Mapping[Any, int] | Sequence[tuple[Any, int]] | None = None,
        design: str | None = None,
        remainder_label: str = "remainder",
        sample_label: str = "sample",
        random_state: int | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.n = None if n is None else _positive_int(n, name="n")
        self.fraction = None if fraction is None else float(fraction)
        if self.fraction is not None and (
            not np.isfinite(self.fraction) or not 0 < self.fraction < 1
        ):
            raise ValueError("fraction must satisfy 0 < fraction < 1.")
        self.has_strata = bool(has_strata)
        self.probability_source = bool(probability_source)
        self.probability_kind = None
        self.scalar_probability = None
        self.stratum_probabilities = None
        if probability is not None:
            if isinstance(probability, Mapping):
                self.probability_kind = "strata"
                self.stratum_probabilities = _pairs(probability)
            else:
                self.probability_kind = "scalar"
                self.scalar_probability = _probability(probability)
        if self.probability_source:
            if probability is not None:
                raise ValueError("probability_source conflicts with probability.")
            self.probability_kind = "artifact"

        self.n_by_stratum = (
            None if n_by_stratum is None else _pairs(n_by_stratum, counts=True)
        )
        if allocation is None:
            self.allocation_mode, self.allocation_weights = "proportional", None
        elif isinstance(allocation, str):
            if allocation not in {"proportional", "equal"}:
                raise ValueError(
                    "allocation must be 'proportional', 'equal', or weights."
                )
            self.allocation_mode, self.allocation_weights = allocation, None
        else:
            self.allocation_mode = "weighted"
            self.allocation_weights = _normalize_allocation(allocation)

        count = sum((
            n is not None, fraction is not None, n_by_stratum is not None,
            self.probability_kind is not None,
        ))
        if count != 1:
            raise ValueError(
                "Supply exactly one of n, fraction, n_by_stratum, or probability."
            )
        inferred = (
            "bernoulli" if self.probability_kind is not None else
            "stratified_srswor" if self.has_strata else "srswor"
        )
        if design is not None and design != inferred:
            raise ValueError(
                f"Arguments imply design={inferred!r}, not asserted design={design!r}."
            )
        self.design = inferred
        if self.n_by_stratum is not None and not self.has_strata:
            raise ValueError("n_by_stratum requires strata.")
        if inferred == "bernoulli":
            if allocation is not None or self.n_by_stratum is not None:
                raise ValueError("Bernoulli does not use fixed-size allocation.")
            if self.probability_kind == "strata" and not self.has_strata:
                raise ValueError("Stratum-specific probability requires strata.")
        elif allocation is not None and not self.has_strata:
            raise ValueError("allocation requires strata.")
        if self.n_by_stratum is not None and allocation is not None:
            raise ValueError("n_by_stratum conflicts with allocation.")

        self.remainder_label = _output_label(remainder_label, name="remainder_label")
        self.sample_label = _output_label(sample_label, name="sample_label")
        forbidden = {"documents", "strata", "probabilities", "pi"}
        if (
            self.sample_label == self.remainder_label
            or self.sample_label in forbidden
            or self.remainder_label in forbidden
        ):
            raise ValueError("Sample/remainder labels must be distinct and nonreserved.")
        self.random_state = _random_state(random_state)

    def _required(self):
        labels = {"documents"}
        if self.has_strata:
            labels.add("strata")
        if self.probability_kind == "artifact":
            labels.add("probabilities")
        return labels

    def _check(self, sources):
        if set(sources) != self._required():
            raise OperatorError(
                f"Expected probability sampling sources {sorted(self._required())}; "
                f"got {sorted(sources)}."
            )

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ) -> Mapping[str, OutputSpec]:
        _ = request
        self._check(sources)
        source = sources["documents"]
        features = (
            "inherit"
            if source.artifact_type.value in {"sparse_matrix", "dense_matrix"}
            else None
        )
        return {
            self.sample_label: OutputSpec(
                artifact_type=source.artifact_type, lineage_mode="preserved_key",
                basis_labels="documents", feature_metadata_mode=features,
            ),
            self.remainder_label: OutputSpec(
                artifact_type=source.artifact_type, lineage_mode="preserved_key",
                basis_labels="documents", feature_metadata_mode=features,
            ),
            "pi": OutputSpec(
                artifact_type="table", lineage_mode="preserved_key",
                basis_labels=self.sample_label,
            ),
        }

    def validate_operation_params(
        self, params: Mapping[str, Any], *,
        sources: Mapping[str, BaseArtifact], mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = mode
        if params:
            raise OperatorError("Probability sample accepts no operation parameters.")
        self._check(sources)
        for label in ("strata", "probabilities"):
            if label not in sources:
                continue
            source, side = sources["documents"], sources[label]
            if side.artifact_type.value != "table":
                raise ArtifactError(f"{label} must be a table artifact.")
            if tuple(source.primary_key) != tuple(side.primary_key):
                raise ArtifactError(f"{label} must have the population primary key.")
            if len(side.get_data_columns()) != 1:
                raise ArtifactError(f"{label} must have exactly one data column.")
        return {}

    def input_request(
        self, *, sources: Mapping[str, BaseArtifact],
        mode: TranslationMode, request: TranslationRequest,
    ) -> Mapping[str, SourceRequest]:
        _ = mode, request
        self._check(sources)
        out = {
            "documents": SourceRequest(
                artifact_type=sources["documents"].artifact_type,
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=False, metadata=False),
                batch_size=None, form="table", metadata_mode="none",
                include_position=True,
            )
        }
        for label in ("strata", "probabilities"):
            if label in sources:
                out[label] = SourceRequest(
                    artifact_type="table", mode="full_artifact",
                    columns=ColumnRequest(keys=True, data=True, metadata=False),
                    batch_size=None, form="table", metadata_mode="none",
                    include_position=False,
                )
        return out

    def translate_batch(
        self, inputs: Mapping[str, InputBatch], *,
        mode: TranslationMode, request: TranslationRequest,
    ) -> BatchResult:
        _ = request
        if mode != "translate":
            raise OperatorError(f"Invalid probability sample mode {mode!r}.")
        self._check(inputs)
        packet = inputs["documents"]
        table = _sort_source(_table(packet.data, name="documents"))
        keys = _key_frame(table, packet.primary_key, name="documents")
        _require_unique_keys(keys, packet.primary_key, name="documents")
        N = len(keys)
        if not N:
            raise ValueError("Cannot probability-sample an empty population.")
        strata = None
        if self.has_strata:
            strata = [
                _stratum_scalar(x)
                for x in _aligned(keys, inputs["strata"], label="strata")
            ]

        if self.design == "bernoulli":
            if self.probability_kind == "scalar":
                probs = np.full(N, self.scalar_probability)
            elif self.probability_kind == "strata":
                lookup = {
                    _stratum_token(k): p for k, p in self.stratum_probabilities
                }
                observed = {_stratum_token(v) for v in strata}
                if set(lookup) != observed:
                    raise ValueError(
                        "Probability mapping must exactly cover the observed strata."
                    )
                probs = np.asarray(
                    [lookup[_stratum_token(v)] for v in strata], dtype=float
                )
            else:
                probs = np.asarray(
                    _aligned(keys, inputs["probabilities"], label="probabilities"),
                    dtype=float,
                )
                if (
                    np.any(~np.isfinite(probs))
                    or np.any(probs <= 0)
                    or np.any(probs > 1)
                ):
                    raise ValueError(
                        "Per-unit inclusion probabilities must satisfy 0 < p <= 1."
                    )
            selected = np.flatnonzero(
                np.random.default_rng(self.random_state).random(N) < probs
            ).tolist()
            pi = {i: float(probs[i]) for i in selected}
        else:
            n = self.n
            if self.fraction is not None:
                n = int(np.floor(self.fraction * N + 0.5))
            if self.n_by_stratum is not None:
                n = sum(count for _, count in self.n_by_stratum)
            if not 0 < n < N:
                raise ValueError(
                    f"Fixed probability sample requires 0 < n < N; got n={n}, N={N}."
                )
            if strata is None:
                selected, pi = _srs_indices(
                    N=N, n=n, random_state=self.random_state
                )
            else:
                if self.n_by_stratum is not None:
                    weights = self.n_by_stratum
                elif self.allocation_mode == "weighted":
                    weights = self.allocation_weights
                else:
                    population_counts = Counter(_stratum_token(v) for v in strata)
                    representatives = {
                        _stratum_token(v): v for v in strata
                    }
                    weights = [
                        (
                            representatives[k],
                            count if self.allocation_mode == "proportional" else 1,
                        )
                        for k, count in population_counts.items()
                    ]
                selected, pi = _stratified_indices(
                    strata, n=n, allocation=tuple(weights),
                    random_state=self.random_state,
                )

        # Every realized Bernoulli draw is valid, including all-selected and
        # none-selected. Never condition on a desired sample size or redraw.
        # Zero-row table outputs are supported by ArtifactWriter. For other
        # artifact types, its explicit zero-row check remains authoritative.
        selected.sort()
        picked = set(selected)
        remaining = [i for i in range(N) if i not in picked]
        sample_keys = keys.iloc[selected].reset_index(drop=True)
        rest_keys = keys.iloc[remaining].reset_index(drop=True)
        return BatchResult(
            outputs={
                self.sample_label: {"keys": sample_keys},
                self.remainder_label: {"keys": rest_keys},
                "pi": {
                    "keys": sample_keys.copy(),
                    "data": pd.DataFrame({"pi": [pi[i] for i in selected]}),
                },
            }
        )

    def handle_batch_result(
        self, result: BatchResult, *, batch_index: int,
        mode: TranslationMode, request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(
        self, *, mode: TranslationMode, request: TranslationRequest
    ) -> OutputMap | None:
        _ = mode, request
        return None

    def to_json_state(self) -> dict[str, Any]:
        def pack(pairs):
            return None if pairs is None else [
                {"stratum": k, "value": v} for k, v in pairs
            ]

        return {
            "design": self.design,
            "n": self.n, "fraction": self.fraction,
            "probability_kind": self.probability_kind,
            "scalar_probability": self.scalar_probability,
            "stratum_probabilities": pack(self.stratum_probabilities),
            "has_strata": self.has_strata,
            "allocation_mode": self.allocation_mode,
            "allocation_weights": pack(self.allocation_weights),
            "n_by_stratum": pack(self.n_by_stratum),
            "sample_label": self.sample_label,
            "remainder_label": self.remainder_label,
            "random_state": self.random_state,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> ProbabilitySampleTranslator:
        def unpack(records):
            return None if records is None else [
                (item["stratum"], item["value"]) for item in records
            ]

        kind = state["probability_kind"]
        probability = (
            state["scalar_probability"] if kind == "scalar" else
            dict(unpack(state["stratum_probabilities"])) if kind == "strata" else
            None
        )
        # The serialized "proportional" mode is also the default sentinel
        # for SRS, Bernoulli, and exact-count strata. It was not necessarily
        # a user-supplied allocation; passing it back to the constructor as
        # an explicit argument would contradict those designs.
        allocation = (
            unpack(state["allocation_weights"])
            if state["allocation_mode"] == "weighted"
            else "equal"
            if state["allocation_mode"] == "equal"
            else None
        )
        return cls(
            n=state["n"], fraction=state["fraction"],
            probability=probability, probability_source=kind == "artifact",
            has_strata=state["has_strata"], allocation=allocation,
            n_by_stratum=unpack(state["n_by_stratum"]),
            design=state["design"], sample_label=state["sample_label"],
            remainder_label=state["remainder_label"],
            random_state=state.get("random_state"),
        )
