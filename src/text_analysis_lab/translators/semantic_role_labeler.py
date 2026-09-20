"""TeAL-native semantic-role labeling using the validated Bag of Ideas runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from text_analysis_lab._linguistics.cache import user_cache_paths
from text_analysis_lab._linguistics.srl.resources import prepare_project_srl_runtime
from text_analysis_lab._linguistics.srl.runtime import AllenNlpSrlRuntime
from text_analysis_lab._linguistics.srl.structures import content_head_indices
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

SENTENCES = "sentences"
TOKENS = "tokens"
PREDICATES = "predicates"
ROLES = "roles"
FAILURES = "failures"

PREDICATE_DATA_COLUMNS = (
    "predicate_token_id",
    "char_start",
    "char_end",
    "text",
    "lemma",
    "pos",
)
ROLE_DATA_COLUMNS = (
    "role",
    "token_start_id",
    "token_end_id",
    "char_start",
    "char_end",
    "text",
    "head_token_id",
    "head_text",
    "head_lemma",
    "head_pos",
    "ent_type",
    "score",
)
FAILURE_DATA_COLUMNS = ("reason", "detail")


class SemanticRoleLabeler(BaseTranslator):
    """Run predicate-conditioned AllenNLP BERT SRL over TeAL sentences.

    The published 2020 AllenNLP checkpoint is converted once into the lightweight runtime
    already validated in Bag of Ideas.  TeAL owns the sentence/token artifacts and writes
    normalized predicate/role tables suitable for later semantic-head analysis.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        sentence_key: str = "sentence_id",
        token_key: str = "token_id",
        device: str = "auto",
        strict_device: bool = False,
        predicate_pos: Sequence[str] = ("VERB", "AUX"),
        max_length: int | None = None,
        mixed_precision: bool = True,
        show_progress: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not sentence_key or not token_key or sentence_key == token_key:
            raise ValueError(
                "sentence_key and token_key must be distinct non-empty names."
            )
        normalized_pos = tuple(str(value).upper() for value in predicate_pos)
        if not normalized_pos:
            raise ValueError("predicate_pos cannot be empty.")
        if max_length is not None and (
            isinstance(max_length, bool) or int(max_length) < 1
        ):
            raise ValueError("max_length must be a positive integer or None.")
        self.sentence_key = str(sentence_key)
        self.token_key = str(token_key)
        self.device = str(device)
        self.strict_device = bool(strict_device)
        self.predicate_pos = normalized_pos
        self.max_length = None if max_length is None else int(max_length)
        self.mixed_precision = bool(mixed_precision)
        self.show_progress = bool(show_progress)

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        sentences, tokens = _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        _ = tokens
        if "predicate_id" in sentences.primary_key:
            raise OperatorError(
                "SemanticRoleLabeler predicate_id collides with sentence keys."
            )
        return {
            PREDICATES: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=SENTENCES,
            ),
            ROLES: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=PREDICATES,
            ),
            FAILURES: OutputSpec(
                artifact_type="table",
                lineage_mode="preserved_key",
                basis_labels=SENTENCES,
            ),
        }

    def validate_operation_params(self, params, *, sources, mode):
        _ = sources, mode
        if params:
            raise OperatorError(
                f"SemanticRoleLabeler does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(self, *, sources, mode, request):
        _ = mode, request
        _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        return {
            SENTENCES: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(
                    keys=True, data=("text", "char_start", "char_end"), metadata=False
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
                        "tag",
                        "dep",
                        "head_token_id",
                        "ent_type",
                        "char_start",
                        "char_end",
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
        if set(inputs) != {SENTENCES, TOKENS}:
            raise OperatorError(
                f"SemanticRoleLabeler expects inputs {SENTENCES!r} and {TOKENS!r}."
            )
        sentence_packet = inputs[SENTENCES]
        token_packet = inputs[TOKENS]
        sentences = _frame(sentence_packet.data, "sentences")
        tokens = _frame(token_packet.data, "tokens")
        sentence_keys = list(sentence_packet.primary_key)
        expected_token_keys = [*sentence_keys, self.token_key]
        if list(token_packet.primary_key) != expected_token_keys:
            raise ArtifactError(
                "SemanticRoleLabeler requires token primary keys to equal sentence keys + "
                f"{self.token_key!r}; got {list(token_packet.primary_key)}."
            )

        required_sentence = [*sentence_keys, "text", "char_start", "char_end"]
        required_token = [
            *expected_token_keys,
            "text",
            "lemma",
            "pos",
            "tag",
            "dep",
            "head_token_id",
            "ent_type",
            "char_start",
            "char_end",
        ]
        missing_sentence = [name for name in required_sentence if name not in sentences]
        missing_token = [name for name in required_token if name not in tokens]
        if missing_sentence or missing_token:
            raise ArtifactError(
                f"SemanticRoleLabeler missing sentence columns {missing_sentence} and token columns {missing_token}."
            )

        runtime = _make_runtime(
            device=self.device,
            strict_device=self.strict_device,
            mixed_precision=self.mixed_precision,
            max_length=self.max_length,
            show_progress=self.show_progress,
        )

        grouped_tokens = tokens.groupby(sentence_keys, sort=False, dropna=False)
        token_groups: dict[tuple[int, ...], pd.DataFrame] = {}
        for raw_key, group in grouped_tokens:
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            token_groups[tuple(int(v) for v in key)] = group.sort_values(
                self.token_key
            ).reset_index(drop=True)

        predicate_keys: list[dict[str, Any]] = []
        predicate_data: list[dict[str, Any]] = []
        role_keys: list[dict[str, Any]] = []
        role_data: list[dict[str, Any]] = []
        failures: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        sentence_key_records: dict[tuple[int, ...], dict[str, int]] = {}

        for _, sentence_row in sentences.iterrows():
            key_record = {name: int(sentence_row[name]) for name in sentence_keys}
            key_tuple = tuple(key_record[name] for name in sentence_keys)
            sentence_key_records[key_tuple] = key_record
            sentence_tokens = token_groups.get(key_tuple)
            if sentence_tokens is None or sentence_tokens.empty:
                _issue(
                    failures,
                    key_tuple,
                    "missing_tokens",
                    "Sentence has no aligned token rows.",
                )
                continue
            predicate_indices = [
                idx
                for idx, row in sentence_tokens.iterrows()
                if _is_predicate_candidate(
                    pos=None if pd.isna(row["pos"]) else str(row["pos"]),
                    tag=None if pd.isna(row["tag"]) else str(row["tag"]),
                    dependency=None if pd.isna(row["dep"]) else str(row["dep"]),
                    lemma=None if pd.isna(row["lemma"]) else str(row["lemma"]),
                    allowed_pos=self.predicate_pos,
                )
            ]
            if not predicate_indices:
                continue
            token_texts = [str(v) for v in sentence_tokens["text"]]
            try:
                encoded = runtime.encode_tokens(token_texts)
            except ValueError as exc:
                _issue(failures, key_tuple, "sentence_too_long", str(exc))
                continue
            try:
                predictions = runtime.predict_encoded_batch(
                    tuple((encoded, int(index)) for index in predicate_indices)
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _clear_cuda_cache()
                    _issue(
                        failures,
                        key_tuple,
                        "cuda_out_of_memory",
                        f"{type(exc).__name__}: {exc}",
                    )
                    continue
                raise

            for predicate_id, (predicate_index, prediction) in enumerate(
                zip(predicate_indices, predictions, strict=True)
            ):
                predicate_row = sentence_tokens.iloc[int(predicate_index)]
                predicate_key = {**key_record, "predicate_id": int(predicate_id)}
                predicate_keys.append(predicate_key)
                predicate_data.append(
                    {
                        "predicate_token_id": int(predicate_row[self.token_key]),
                        "char_start": int(predicate_row["char_start"]),
                        "char_end": int(predicate_row["char_end"]),
                        "text": str(predicate_row["text"]),
                        "lemma": None
                        if pd.isna(predicate_row["lemma"])
                        else str(predicate_row["lemma"]),
                        "pos": None
                        if pd.isna(predicate_row["pos"])
                        else str(predicate_row["pos"]),
                    }
                )
                if prediction.bio_repairs:
                    _issue(
                        failures,
                        key_tuple,
                        "projected_bio_repaired",
                        json.dumps(
                            {
                                "predicate_id": int(predicate_id),
                                "predicate_token_id": int(
                                    predicate_row[self.token_key]
                                ),
                                "repairs": [
                                    {
                                        "position": int(repair.position),
                                        "original_tag": repair.original_tag,
                                        "repaired_tag": repair.repaired_tag,
                                        "reason": repair.reason,
                                    }
                                    for repair in prediction.bio_repairs
                                ],
                            },
                            sort_keys=True,
                        ),
                    )
                for role_id, span in enumerate(prediction.spans):
                    head_indices = content_head_indices(
                        start=int(span.start),
                        end=int(span.end),
                        token_ids=[int(v) for v in sentence_tokens[self.token_key]],
                        head_token_ids=[
                            None if pd.isna(v) else int(v)
                            for v in sentence_tokens["head_token_id"]
                        ],
                        dependencies=[
                            None if pd.isna(v) else str(v)
                            for v in sentence_tokens["dep"]
                        ],
                        pos=[
                            None if pd.isna(v) else str(v)
                            for v in sentence_tokens["pos"]
                        ],
                        text=token_texts,
                        role=str(span.label),
                    )
                    start_row = sentence_tokens.iloc[int(span.start)]
                    end_row = sentence_tokens.iloc[int(span.end) - 1]
                    sentence_start = int(sentence_row["char_start"])
                    local_start = int(start_row["char_start"]) - sentence_start
                    local_end = int(end_row["char_end"]) - sentence_start
                    sentence_text = str(sentence_row["text"])
                    span_text = sentence_text[local_start:local_end]
                    span_score = float(
                        np.mean(prediction.word_scores[int(span.start) : int(span.end)])
                    )
                    for head_id, head_index in enumerate(head_indices):
                        head = sentence_tokens.iloc[int(head_index)]
                        role_keys.append(
                            {
                                **predicate_key,
                                "role_id": int(role_id),
                                "head_id": int(head_id),
                            }
                        )
                        role_data.append(
                            {
                                "role": str(span.label),
                                "token_start_id": int(start_row[self.token_key]),
                                "token_end_id": int(end_row[self.token_key]) + 1,
                                "char_start": int(start_row["char_start"]),
                                "char_end": int(end_row["char_end"]),
                                "text": span_text,
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
                                "score": span_score,
                            }
                        )

        failure_keys: list[dict[str, Any]] = []
        failure_data: list[dict[str, Any]] = []
        if failures:
            for key_tuple, issues in failures.items():
                failure_keys.append(sentence_key_records[key_tuple])
                reasons = list(dict.fromkeys(str(item["reason"]) for item in issues))
                failure_data.append(
                    {
                        "reason": reasons[0] if len(reasons) == 1 else "multiple",
                        "detail": json.dumps(issues, sort_keys=True),
                    }
                )
        outputs: dict[str, Mapping[str, Any]] = {
            PREDICATES: {
                "keys": pd.DataFrame.from_records(
                    predicate_keys, columns=[*sentence_keys, "predicate_id"]
                ),
                "data": pd.DataFrame.from_records(
                    predicate_data, columns=list(PREDICATE_DATA_COLUMNS)
                ),
            },
            ROLES: {
                "keys": pd.DataFrame.from_records(
                    role_keys,
                    columns=[*sentence_keys, "predicate_id", "role_id", "head_id"],
                ),
                "data": pd.DataFrame.from_records(
                    role_data, columns=list(ROLE_DATA_COLUMNS)
                ),
            },
            FAILURES: {
                "keys": pd.DataFrame.from_records(failure_keys, columns=sentence_keys),
                "data": pd.DataFrame.from_records(
                    failure_data, columns=list(FAILURE_DATA_COLUMNS)
                ),
            },
        }
        return BatchResult(outputs=outputs)

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs or None

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def to_json_state(self) -> dict[str, Any]:
        return {
            "sentence_key": self.sentence_key,
            "token_key": self.token_key,
            "device": self.device,
            "strict_device": self.strict_device,
            "predicate_pos": list(self.predicate_pos),
            "max_length": self.max_length,
            "mixed_precision": self.mixed_precision,
            "show_progress": self.show_progress,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SemanticRoleLabeler:
        return cls(**cast(dict[str, Any], dict(state)))


def _make_runtime(*, device, strict_device, mixed_precision, max_length, show_progress):
    cache = user_cache_paths()
    paths = prepare_project_srl_runtime(
        cache.root,
        cache_dir=cache.root,
        download=True,
        show_progress=show_progress,
    )
    return AllenNlpSrlRuntime(
        paths.runtime_dir,
        device=device,
        strict_device=strict_device,
        mixed_precision=mixed_precision,
        max_length=max_length,
        show_progress=show_progress,
    )


def _validate_sources(sources, *, sentence_key: str, token_key: str):
    if set(sources) != {SENTENCES, TOKENS}:
        raise OperatorError(
            f"SemanticRoleLabeler expects source labels {SENTENCES!r} and {TOKENS!r}; got {sorted(sources)}."
        )
    sentences = sources[SENTENCES]
    tokens = sources[TOKENS]
    if (
        sentences.artifact_type.value != "table"
        or tokens.artifact_type.value != "table"
    ):
        raise OperatorError("SemanticRoleLabeler requires table artifacts.")
    if not sentences.primary_key or sentences.primary_key[-1] != sentence_key:
        raise OperatorError(
            f"SemanticRoleLabeler expects sentence primary key to end in {sentence_key!r}."
        )
    if list(tokens.primary_key) != [*sentences.primary_key, token_key]:
        raise OperatorError(
            "SemanticRoleLabeler requires token primary keys to equal sentence keys + "
            f"{token_key!r}."
        )
    return sentences, tokens


def _is_predicate_candidate(*, pos, tag, dependency, lemma, allowed_pos):
    normalized_pos = (pos or "").upper()
    if normalized_pos not in allowed_pos:
        return False
    if normalized_pos == "VERB":
        return True
    normalized_tag = (tag or "").upper()
    normalized_dep = (dependency or "").lower()
    normalized_lemma = (lemma or "").lower()
    if normalized_tag == "MD" or normalized_dep in {"aux", "auxpass"}:
        return False
    if normalized_lemma == "be":
        return True
    return normalized_dep == "root"


def _issue(store, key, reason, detail):
    store.setdefault(key, []).append({"reason": str(reason), "detail": str(detail)})


def _frame(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"SemanticRoleLabeler expected {label} as a pandas DataFrame."
        )
    return value


def _is_cuda_oom(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "cuda out of memory" in str(exc).lower()
    )


def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
