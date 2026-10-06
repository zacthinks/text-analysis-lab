"""Cheap configurable semantic-head resolution for SRL role spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import json

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
from text_analysis_lab.linguistics.heads import (
    SemanticHeadRules,
    load_semantic_head_rules,
    resolve_semantic_heads,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

ROLE_SPANS = "role_spans"
TOKENS = "tokens"
ROLE_HEADS = "role_heads"

ROLE_HEAD_DATA_COLUMNS = (
    "role",
    "syntactic_root_token_id",
    "head_token_id",
    "head_text",
    "head_lemma",
    "head_pos",
    "ent_type",
    "rule",
    "resolution_path",
    "rules_fingerprint",
)


class SemanticRoleHeadResolver(BaseTranslator):
    """Resolve semantic heads from stored SRL spans without rerunning SRL."""

    operation_type = "translate"

    def __init__(
        self,
        *,
        sentence_key: str = "sentence_id",
        token_key: str = "token_id",
        head_rules: SemanticHeadRules | Mapping[str, Any] | str | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not sentence_key or not token_key or sentence_key == token_key:
            raise ValueError(
                "sentence_key and token_key must be distinct non-empty names."
            )
        self.sentence_key = str(sentence_key)
        self.token_key = str(token_key)
        self.head_rules = load_semantic_head_rules(head_rules)

    def translate(
        self,
        role_spans: pd.DataFrame,
        tokens: pd.DataFrame,
        *,
        sentence_keys: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Resolve heads from ordinary role-span and token tables.

        sentence_keys may be supplied explicitly for unusual table layouts.
        Otherwise it is inferred from the role-span key prefix ending immediately
        before predicate_id and validated against the token table.
        """

        if not isinstance(role_spans, pd.DataFrame) or not isinstance(
            tokens, pd.DataFrame
        ):
            raise TypeError(
                "SemanticRoleHeadResolver.translate(...) requires pandas DataFrames "
                "for role_spans and tokens."
            )
        if sentence_keys is None:
            columns = [str(value) for value in role_spans.columns]
            required_key_columns = {self.sentence_key, "predicate_id", "role_id"}
            missing = sorted(required_key_columns - set(columns))
            if missing:
                raise ValueError(
                    "Cannot infer sentence_keys because role_spans is missing "
                    f"key column(s) {missing}."
                )
            sentence_position = columns.index(self.sentence_key)
            predicate_position = columns.index("predicate_id")
            role_position = columns.index("role_id")
            if not (
                predicate_position == sentence_position + 1
                and role_position == predicate_position + 1
            ):
                raise ValueError(
                    "Cannot infer sentence_keys: expected role-span key columns "
                    "to end with [sentence_key, 'predicate_id', 'role_id']."
                )
            keys = columns[: sentence_position + 1]
        else:
            keys = [str(value) for value in sentence_keys]
        if not keys or keys[-1] != self.sentence_key:
            raise ValueError(
                f"sentence_keys must end in {self.sentence_key!r}; got {keys}."
            )
        expected_token_keys = [*keys, self.token_key]
        missing_token_keys = [
            name for name in expected_token_keys if name not in tokens.columns
        ]
        if missing_token_keys:
            raise ValueError(
                "SemanticRoleHeadResolver inferred sentence keys that are not "
                f"present on tokens: {missing_token_keys}."
            )
        payload = self._translate_frames(
            role_spans.reset_index(drop=True),
            tokens.reset_index(drop=True),
            sentence_keys=keys,
        )[ROLE_HEADS]
        return {
            ROLE_HEADS: pd.concat(
                [
                    payload["keys"].reset_index(drop=True),
                    payload["data"].reset_index(drop=True),
                ],
                axis=1,
            )
        }

    def _translate_frames(
        self,
        role_spans: pd.DataFrame,
        tokens: pd.DataFrame,
        *,
        sentence_keys: Sequence[str],
    ) -> dict[str, Mapping[str, Any]]:
        sentence_keys = [str(value) for value in sentence_keys]
        span_keys = [*sentence_keys, "predicate_id", "role_id"]
        token_keys = [*sentence_keys, self.token_key]
        required_spans = [
            *span_keys,
            "role",
            "token_start_id",
            "token_end_id",
        ]
        required_tokens = [
            *token_keys,
            "text",
            "lemma",
            "pos",
            "dep",
            "head_token_id",
            "ent_type",
        ]
        missing_spans = [name for name in required_spans if name not in role_spans]
        missing_tokens = [name for name in required_tokens if name not in tokens]
        if missing_spans or missing_tokens:
            raise ArtifactError(
                "SemanticRoleHeadResolver missing role-span columns "
                f"{missing_spans} and token columns {missing_tokens}."
            )

        grouped_tokens: dict[tuple[Any, ...], pd.DataFrame] = {}
        for raw_key, group in tokens.groupby(sentence_keys, sort=False, dropna=False):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            grouped_tokens[tuple(key)] = group.sort_values(self.token_key).reset_index(
                drop=True
            )

        head_keys: list[dict[str, Any]] = []
        head_data: list[dict[str, Any]] = []
        for _, span in role_spans.iterrows():
            key_record = {name: span[name] for name in sentence_keys}
            key_tuple = tuple(key_record[name] for name in sentence_keys)
            sentence = grouped_tokens.get(key_tuple)
            if sentence is None or sentence.empty:
                raise ArtifactError(
                    "SemanticRoleHeadResolver could not find tokens for role span "
                    f"{key_record}."
                )

            start_id = int(span["token_start_id"])
            end_id = int(span["token_end_id"])
            token_ids = sentence[self.token_key].astype(int).tolist()
            span_positions = [
                index
                for index, token_id in enumerate(token_ids)
                if start_id <= token_id < end_id
            ]
            if not span_positions:
                raise ArtifactError(
                    "SemanticRoleHeadResolver role span does not overlap any tokens: "
                    f"[{start_id}, {end_id})."
                )
            start = min(span_positions)
            end = max(span_positions) + 1
            resolutions = resolve_semantic_heads(
                start=start,
                end=end,
                token_ids=token_ids,
                head_token_ids=[
                    None if pd.isna(value) else int(value)
                    for value in sentence["head_token_id"]
                ],
                dependencies=[
                    None if pd.isna(value) else str(value)
                    for value in sentence["dep"]
                ],
                pos=[
                    None if pd.isna(value) else str(value)
                    for value in sentence["pos"]
                ],
                text=[str(value) for value in sentence["text"]],
                lemmas=[
                    None if pd.isna(value) else str(value)
                    for value in sentence["lemma"]
                ],
                ent_types=[
                    None if pd.isna(value) else str(value)
                    for value in sentence["ent_type"]
                ],
                source="srl",
                role=str(span["role"]),
                rules=self.head_rules,
            )
            for head_id, resolution in enumerate(resolutions):
                head = sentence.iloc[int(resolution.semantic_head_index)]
                syntactic_root = sentence.iloc[int(resolution.syntactic_root_index)]
                resolution_path = json.dumps(
                    [
                        {
                            "rule": move.rule_id,
                            "kind": move.kind,
                            "from_token_id": int(
                                sentence.iloc[int(move.from_index)][self.token_key]
                            ),
                            "to_token_id": int(
                                sentence.iloc[int(move.to_index)][self.token_key]
                            ),
                        }
                        for move in resolution.resolution_path
                    ],
                    separators=(",", ":"),
                )
                head_keys.append(
                    {
                        **key_record,
                        "predicate_id": int(span["predicate_id"]),
                        "role_id": int(span["role_id"]),
                        "head_id": int(head_id),
                    }
                )
                head_data.append(
                    {
                        "role": str(span["role"]),
                        "syntactic_root_token_id": int(
                            syntactic_root[self.token_key]
                        ),
                        "head_token_id": int(head[self.token_key]),
                        "head_text": str(head["text"]),
                        "head_lemma": None
                        if pd.isna(head["lemma"])
                        else str(head["lemma"]),
                        "head_pos": None
                        if pd.isna(head["pos"])
                        else str(head["pos"]),
                        "ent_type": None
                        if pd.isna(head["ent_type"])
                        else str(head["ent_type"]),
                        "rule": resolution.rule_id,
                        "resolution_path": resolution_path,
                        "rules_fingerprint": resolution.rules_fingerprint,
                    }
                )

        return {
            ROLE_HEADS: {
                "keys": pd.DataFrame.from_records(
                    head_keys,
                    columns=[*sentence_keys, "predicate_id", "role_id", "head_id"],
                ),
                "data": pd.DataFrame.from_records(
                    head_data, columns=list(ROLE_HEAD_DATA_COLUMNS)
                ),
            }
        }

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        role_spans, _tokens = _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        if "head_id" in role_spans.primary_key:
            raise OperatorError(
                "SemanticRoleHeadResolver head_id collides with role-span keys."
            )
        return {
            ROLE_HEADS: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=ROLE_SPANS,
            )
        }

    def validate_operation_params(self, params, *, sources, mode):
        _ = sources, mode
        if params:
            raise OperatorError(
                "SemanticRoleHeadResolver does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        return {}

    def input_request(self, *, sources, mode, request):
        _ = mode, request
        _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        return {
            ROLE_SPANS: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(
                    keys=True,
                    data=("role", "token_start_id", "token_end_id"),
                    metadata=False,
                ),
                form="table",
                metadata_mode="none",
                include_position=False,
            ),
            TOKENS: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(
                    keys=True,
                    data=(
                        "text",
                        "lemma",
                        "pos",
                        "dep",
                        "head_token_id",
                        "ent_type",
                    ),
                    metadata=False,
                ),
                form="table",
                metadata_mode="none",
                include_position=False,
            ),
        }

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        if set(inputs) != {ROLE_SPANS, TOKENS}:
            raise OperatorError(
                f"SemanticRoleHeadResolver expects inputs {ROLE_SPANS!r} and {TOKENS!r}."
            )
        span_packet = inputs[ROLE_SPANS]
        token_packet = inputs[TOKENS]
        role_spans = _frame(span_packet.data, ROLE_SPANS)
        tokens = _frame(token_packet.data, TOKENS)
        span_key = [str(value) for value in span_packet.primary_key]
        if len(span_key) < 3 or span_key[-2:] != ["predicate_id", "role_id"]:
            raise ArtifactError(
                "SemanticRoleHeadResolver requires role-span primary keys to end in "
                "['predicate_id', 'role_id']."
            )
        sentence_keys = span_key[:-2]
        if not sentence_keys or sentence_keys[-1] != self.sentence_key:
            raise ArtifactError(
                "SemanticRoleHeadResolver requires role-span sentence keys to end in "
                f"{self.sentence_key!r}."
            )
        expected_token_keys = [*sentence_keys, self.token_key]
        if list(token_packet.primary_key) != expected_token_keys:
            raise ArtifactError(
                "SemanticRoleHeadResolver requires token primary keys to equal "
                f"sentence keys + {self.token_key!r}; got "
                f"{list(token_packet.primary_key)}."
            )
        return BatchResult(
            outputs=self._translate_frames(
                role_spans,
                tokens,
                sentence_keys=sentence_keys,
            )
        )

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs or None

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def to_json_state(self) -> dict[str, Any]:
        return {
            "sentence_key": self.sentence_key,
            "token_key": self.token_key,
            "head_rules": self.head_rules.to_dict(),
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "SemanticRoleHeadResolver":
        values = cast(dict[str, Any], dict(state))
        rules = values.get("head_rules")
        if rules is not None:
            values["head_rules"] = SemanticHeadRules.from_dict(rules)
        return cls(**values)


def _validate_sources(sources, *, sentence_key: str, token_key: str):
    if set(sources) != {ROLE_SPANS, TOKENS}:
        raise OperatorError(
            "SemanticRoleHeadResolver expects source labels "
            f"{ROLE_SPANS!r} and {TOKENS!r}; got {sorted(sources)}."
        )
    role_spans = sources[ROLE_SPANS]
    tokens = sources[TOKENS]
    if (
        role_spans.artifact_type.value != "table"
        or tokens.artifact_type.value != "table"
    ):
        raise OperatorError("SemanticRoleHeadResolver requires table artifacts.")
    span_key = list(role_spans.primary_key)
    if len(span_key) < 3 or span_key[-2:] != ["predicate_id", "role_id"]:
        raise OperatorError(
            "SemanticRoleHeadResolver requires role-span primary keys to end in "
            "['predicate_id', 'role_id']."
        )
    sentence_keys = span_key[:-2]
    if not sentence_keys or sentence_keys[-1] != sentence_key:
        raise OperatorError(
            "SemanticRoleHeadResolver requires role-span sentence keys to end in "
            f"{sentence_key!r}."
        )
    expected = [*sentence_keys, token_key]
    if list(tokens.primary_key) != expected:
        raise OperatorError(
            "SemanticRoleHeadResolver requires token primary keys to equal sentence "
            f"keys + {token_key!r}."
        )
    return role_spans, tokens


def _frame(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"SemanticRoleHeadResolver expected {label} as a pandas DataFrame."
        )
    return value
