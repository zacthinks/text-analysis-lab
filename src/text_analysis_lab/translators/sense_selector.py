"""Cheap revisable sense selection over stored WSD candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.translators.word_sense_disambiguator import SENSE_DATA_COLUMNS

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

CANDIDATES = "candidates"
SENSES = "senses"
CANDIDATE_ID = "candidate_id"

SELECTION_DATA_COLUMNS = (
    *SENSE_DATA_COLUMNS,
    "model_selected",
    "selection_reason",
)

_REQUIRED_CANDIDATE_COLUMNS = (
    "surface_form",
    "parser_lemma",
    "pos",
    "sense_id",
    "synset_id",
    "sense_label",
    "candidate_lemma",
    "aliases",
    "ili",
    "ontology_id",
    "ontology_version",
    "gloss_text",
    "normalized_score",
    "rank",
    "selected",
    "top1_margin",
    "candidate_kind",
    "model_name",
    "model_revision",
)


class SenseSelector(BaseTranslator):
    """Select one stored WSD candidate per token without rerunning WSD.

    Selection is deterministic and intentionally small in scope:

    1. an explicit token-level force wins when configured;
    2. otherwise globally excluded senses are removed;
    3. optional lemma and exact-token allowlists are intersected with the
       remaining candidates;
    4. the lowest-rank remaining candidate is selected, with sense_id as a
       deterministic tie-breaker.

    Token rules use complete stable token-key mappings rather than row
    positions. A configured token rule may be absent from a particular input
    artifact, which makes reusable policies practical across subsets. If the
    token is present, however, a forced sense must exist among its stored
    candidates.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        excluded_sense_ids: Sequence[str] = (),
        allowed_sense_ids_by_lemma: Mapping[str, Sequence[str]] | None = None,
        allowed_sense_ids_by_token: Sequence[Mapping[str, Any]] = (),
        forced_sense_ids_by_token: Sequence[Mapping[str, Any]] = (),
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        self.excluded_sense_ids = _normalize_sense_ids(
            excluded_sense_ids, name="excluded_sense_ids"
        )
        self.allowed_sense_ids_by_lemma = _normalize_lemma_rules(
            allowed_sense_ids_by_lemma or {}
        )
        self.allowed_sense_ids_by_token = _normalize_token_allow_rules(
            allowed_sense_ids_by_token
        )
        self.forced_sense_ids_by_token = _normalize_token_force_rules(
            forced_sense_ids_by_token
        )

    def translate(
        self,
        candidates: pd.DataFrame,
        *,
        candidate_keys: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Select senses from an ordinary WSD-candidate table."""
        if not isinstance(candidates, pd.DataFrame):
            raise TypeError(
                "SenseSelector.translate(...) requires a pandas DataFrame."
            )
        keys = _standalone_candidate_keys(
            candidates,
            candidate_keys=candidate_keys,
        )
        token_keys = _validate_candidate_key(keys)
        payload = self._translate_frame(
            candidates.reset_index(drop=True),
            token_keys=token_keys,
        )[SENSES]
        return {
            SENSES: pd.concat(
                [
                    payload["keys"].reset_index(drop=True),
                    payload["data"].reset_index(drop=True),
                ],
                axis=1,
            )
        }

    def _translate_frame(
        self,
        candidates: pd.DataFrame,
        *,
        token_keys: Sequence[str],
    ) -> dict[str, Mapping[str, Any]]:
        token_keys = [str(value) for value in token_keys]
        required = [*token_keys, CANDIDATE_ID, *_REQUIRED_CANDIDATE_COLUMNS]
        missing = [name for name in required if name not in candidates.columns]
        if missing:
            raise ArtifactError(
                f"SenseSelector candidate source is missing {missing}."
            )
        if candidates.duplicated([*token_keys, CANDIDATE_ID]).any():
            raise ArtifactError(
                "SenseSelector candidate source contains duplicate candidate keys."
            )

        token_allow = _rules_for_token_keys(
            self.allowed_sense_ids_by_token,
            token_keys=token_keys,
            value_name="sense_ids",
        )
        token_force = _rules_for_token_keys(
            self.forced_sense_ids_by_token,
            token_keys=token_keys,
            value_name="sense_id",
        )

        selected_keys: list[dict[str, Any]] = []
        selected_data: list[dict[str, Any]] = []

        grouped = candidates.groupby(token_keys, sort=False, dropna=False)
        for raw_key, group in grouped:
            key_tuple = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            key_record = {
                name: _plain_scalar(value)
                for name, value in zip(token_keys, key_tuple, strict=True)
            }
            exact_key = tuple(key_record[name] for name in token_keys)
            ordered = group.sort_values(
                ["rank", "sense_id", CANDIDATE_ID],
                kind="stable",
            ).reset_index(drop=True)
            _validate_candidate_group(ordered, key_record=key_record)

            forced_sense = token_force.get(exact_key)
            if forced_sense is not None:
                matching = ordered[
                    ordered["sense_id"].astype(str) == str(forced_sense)
                ]
                if matching.empty:
                    raise ArtifactError(
                        "SenseSelector forced sense "
                        f"{forced_sense!r} is not present among candidates for "
                        f"token {key_record}."
                    )
                chosen = matching.iloc[0]
                reason = "forced_token"
            else:
                pool = ordered
                if self.excluded_sense_ids:
                    pool = pool[
                        ~pool["sense_id"]
                        .astype(str)
                        .isin(self.excluded_sense_ids)
                    ]

                lemma = _normalized(str(ordered.iloc[0]["parser_lemma"]))
                lemma_allow = self.allowed_sense_ids_by_lemma.get(lemma)
                if lemma_allow is not None:
                    pool = pool[
                        pool["sense_id"].astype(str).isin(lemma_allow)
                    ]

                exact_allow = token_allow.get(exact_key)
                if exact_allow is not None:
                    pool = pool[
                        pool["sense_id"].astype(str).isin(exact_allow)
                    ]

                if pool.empty:
                    continue

                chosen = pool.iloc[0]
                if exact_allow is not None:
                    reason = "token_restriction"
                elif lemma_allow is not None:
                    reason = "lemma_restriction"
                elif self.excluded_sense_ids:
                    reason = "top_ranked_after_exclusion"
                else:
                    reason = "top_ranked"

            selected_keys.append(dict(key_record))
            selected_data.append(_sense_record(chosen, reason=reason))

        return {
            SENSES: {
                "keys": pd.DataFrame.from_records(
                    selected_keys,
                    columns=token_keys,
                ),
                "data": pd.DataFrame.from_records(
                    selected_data,
                    columns=list(SELECTION_DATA_COLUMNS),
                ),
            }
        }

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        candidates = _validate_sources(sources)
        _validate_candidate_key(list(candidates.primary_key))
        return {
            SENSES: OutputSpec(
                artifact_type="table",
                lineage_mode="reduced_key",
                basis_labels=CANDIDATES,
            )
        }

    def validate_operation_params(self, params, *, sources, mode):
        _ = sources, mode
        if params:
            raise OperatorError(
                "SenseSelector does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        return {}

    def input_request(self, *, sources, mode, request):
        _ = mode, request
        candidates = _validate_sources(sources)
        _validate_candidate_key(list(candidates.primary_key))
        return {
            CANDIDATES: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(
                    keys=True,
                    data=_REQUIRED_CANDIDATE_COLUMNS,
                    metadata=False,
                ),
                form="table",
                metadata_mode="none",
                include_position=False,
            )
        }

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        if set(inputs) != {CANDIDATES}:
            raise OperatorError(
                f"SenseSelector expects one input under {CANDIDATES!r}."
            )
        packet = inputs[CANDIDATES]
        candidates = _frame(packet.data)
        candidate_keys = [str(value) for value in packet.primary_key]
        token_keys = _validate_candidate_key(candidate_keys)
        return BatchResult(
            outputs=self._translate_frame(
                candidates,
                token_keys=token_keys,
            )
        )

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs or None

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def to_json_state(self) -> dict[str, Any]:
        return {
            "excluded_sense_ids": list(self.excluded_sense_ids),
            "allowed_sense_ids_by_lemma": {
                lemma: list(sense_ids)
                for lemma, sense_ids in self.allowed_sense_ids_by_lemma.items()
            },
            "allowed_sense_ids_by_token": [
                {
                    "key": dict(key_items),
                    "sense_ids": list(sense_ids),
                }
                for key_items, sense_ids in self.allowed_sense_ids_by_token
            ],
            "forced_sense_ids_by_token": [
                {
                    "key": dict(key_items),
                    "sense_id": sense_id,
                }
                for key_items, sense_id in self.forced_sense_ids_by_token
            ],
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "SenseSelector":
        return cls(**cast(dict[str, Any], dict(state)))


def _sense_record(row: pd.Series, *, reason: str) -> dict[str, Any]:
    parser_lemma = str(row["parser_lemma"])
    resolved_lemma = str(row["candidate_lemma"])
    return {
        "surface_form": str(row["surface_form"]),
        "parser_lemma": parser_lemma,
        "resolved_lemma": resolved_lemma,
        "lemma_overrides_parser": _normalized(parser_lemma)
        != _normalized(resolved_lemma),
        "pos": str(row["pos"]),
        "sense_id": str(row["sense_id"]),
        "synset_id": str(row["synset_id"]),
        "sense_label": _nullable_string(row["sense_label"]),
        "aliases": str(row["aliases"]),
        "ili": _nullable_string(row["ili"]),
        "ontology_id": str(row["ontology_id"]),
        "ontology_version": str(row["ontology_version"]),
        "gloss_text": str(row["gloss_text"]),
        "normalized_score": float(row["normalized_score"]),
        "top1_margin": (
            None if pd.isna(row["top1_margin"]) else float(row["top1_margin"])
        ),
        "candidate_kind": str(row["candidate_kind"]),
        "model_name": str(row["model_name"]),
        "model_revision": _nullable_string(row["model_revision"]),
        "model_selected": bool(row["selected"]),
        "selection_reason": str(reason),
    }


def _validate_sources(sources):
    if set(sources) != {CANDIDATES}:
        raise OperatorError(
            f"SenseSelector expects source label {CANDIDATES!r}; "
            f"got {sorted(sources)}."
        )
    candidates = sources[CANDIDATES]
    if candidates.artifact_type.value != "table":
        raise OperatorError("SenseSelector requires a table candidate artifact.")
    return candidates


def _standalone_candidate_keys(
    candidates: pd.DataFrame,
    *,
    candidate_keys: Sequence[str] | None,
) -> list[str]:
    if candidate_keys is not None:
        return [str(value) for value in candidate_keys]

    columns = [str(value) for value in candidates.columns]
    if CANDIDATE_ID not in columns:
        raise ValueError(
            "SenseSelector.translate(...) could not infer candidate_keys because "
            f"{CANDIDATE_ID!r} is absent. Pass candidate_keys explicitly."
        )
    candidate_index = columns.index(CANDIDATE_ID)
    inferred = columns[: candidate_index + 1]
    required_positions = [
        columns.index(name)
        for name in _REQUIRED_CANDIDATE_COLUMNS
        if name in columns
    ]
    if any(position < candidate_index for position in required_positions):
        raise ValueError(
            "SenseSelector.translate(...) can infer candidate_keys only from "
            "TeAL-style frames with key columns before candidate data columns. "
            "Pass candidate_keys explicitly for reordered frames."
        )
    _validate_candidate_key(inferred)
    return inferred


def _validate_candidate_key(candidate_keys: Sequence[str]) -> list[str]:
    keys = [str(value) for value in candidate_keys]
    if len(keys) < 2 or keys[-1] != CANDIDATE_ID:
        raise OperatorError(
            "SenseSelector requires candidate primary keys to end in "
            f"{CANDIDATE_ID!r}; got {keys}."
        )
    return keys[:-1]


def _validate_candidate_group(
    group: pd.DataFrame, *, key_record: Mapping[str, Any]
) -> None:
    for column in ("surface_form", "parser_lemma", "pos"):
        values = {
            None if pd.isna(value) else str(value)
            for value in group[column].tolist()
        }
        if len(values) > 1:
            raise ArtifactError(
                f"SenseSelector candidates for token {dict(key_record)} disagree "
                f"on {column!r}: {sorted(str(value) for value in values)}."
            )


def _normalize_sense_ids(
    values: Sequence[str], *, name: str
) -> tuple[str, ...]:
    raw_values = [str(value) for value in values]
    if any(not value for value in raw_values):
        raise ValueError(f"{name} cannot contain empty sense IDs.")
    return tuple(sorted(set(raw_values)))


def _normalize_lemma_rules(
    rules: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_lemma, raw_sense_ids in rules.items():
        lemma = _normalized(str(raw_lemma))
        if not lemma:
            raise ValueError("allowed_sense_ids_by_lemma cannot use an empty lemma.")
        if lemma in normalized:
            raise ValueError(
                "allowed_sense_ids_by_lemma contains duplicate normalized lemma "
                f"{lemma!r}."
            )
        sense_ids = _normalize_sense_ids(
            raw_sense_ids,
            name=f"allowed_sense_ids_by_lemma[{raw_lemma!r}]",
        )
        if not sense_ids:
            raise ValueError(
                f"allowed_sense_ids_by_lemma[{raw_lemma!r}] cannot be empty."
            )
        normalized[lemma] = sense_ids
    return normalized


def _normalize_token_allow_rules(
    rules: Sequence[Mapping[str, Any]],
):
    normalized = []
    seen = set()
    for index, raw in enumerate(rules):
        if set(raw) != {"key", "sense_ids"}:
            raise ValueError(
                "Each allowed_sense_ids_by_token entry must contain exactly "
                "'key' and 'sense_ids'."
            )
        key_items = _normalize_rule_key(raw["key"], name=f"allow rule {index}")
        if key_items in seen:
            raise ValueError(
                f"Duplicate allowed_sense_ids_by_token key: {dict(key_items)}."
            )
        seen.add(key_items)
        sense_ids = _normalize_sense_ids(
            cast(Sequence[str], raw["sense_ids"]),
            name=f"allowed_sense_ids_by_token[{index}].sense_ids",
        )
        if not sense_ids:
            raise ValueError(
                "allowed_sense_ids_by_token sense_ids cannot be empty."
            )
        normalized.append((key_items, sense_ids))
    return tuple(normalized)


def _normalize_token_force_rules(
    rules: Sequence[Mapping[str, Any]],
):
    normalized = []
    seen = set()
    for index, raw in enumerate(rules):
        if set(raw) != {"key", "sense_id"}:
            raise ValueError(
                "Each forced_sense_ids_by_token entry must contain exactly "
                "'key' and 'sense_id'."
            )
        key_items = _normalize_rule_key(raw["key"], name=f"force rule {index}")
        if key_items in seen:
            raise ValueError(
                f"Duplicate forced_sense_ids_by_token key: {dict(key_items)}."
            )
        seen.add(key_items)
        sense_id = str(raw["sense_id"])
        if not sense_id:
            raise ValueError(
                "forced_sense_ids_by_token sense_id cannot be empty."
            )
        normalized.append((key_items, sense_id))
    return tuple(normalized)


def _normalize_rule_key(value: Any, *, name: str) -> tuple[tuple[str, Any], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} key must be a non-empty mapping.")
    items: list[tuple[str, Any]] = []
    for raw_name, raw_value in value.items():
        key_name = str(raw_name)
        if not key_name:
            raise ValueError(f"{name} contains an empty key-column name.")
        scalar = _plain_scalar(raw_value)
        try:
            hash(scalar)
        except TypeError as exc:
            raise ValueError(
                f"{name} value for {key_name!r} must be a hashable scalar."
            ) from exc
        items.append((key_name, scalar))
    return tuple(sorted(items))


def _rules_for_token_keys(
    rules,
    *,
    token_keys: Sequence[str],
    value_name: str,
):
    expected = set(token_keys)
    result = {}
    for key_items, value in rules:
        key = dict(key_items)
        if set(key) != expected:
            raise ArtifactError(
                f"SenseSelector {value_name} token rule must identify exactly "
                f"the token key columns {list(token_keys)}; got {sorted(key)}."
            )
        exact_key = tuple(key[name] for name in token_keys)
        result[exact_key] = value
    return result


def _frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            "SenseSelector expected a pandas DataFrame candidate source."
        )
    return value


def _nullable_string(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _normalized(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split()).casefold()
