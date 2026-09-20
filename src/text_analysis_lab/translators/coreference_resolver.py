"""TeAL-native coreference translation built from the Bag of Ideas runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab._linguistics.coreference.runtime import FastCorefRuntime
from text_analysis_lab._linguistics.srl.structures import content_head_indices
from text_analysis_lab._linguistics.cache import user_cache_paths
from text_analysis_lab._linguistics.device import resolve_devices
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BatchResult,
    BaseTranslator,
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

DOCUMENTS = "documents"
TOKENS = "tokens"
MENTIONS = "mentions"
FAILURES = "failures"

MENTION_DATA_COLUMNS = (
    "sentence_id",
    "char_start",
    "char_end",
    "token_start_id",
    "token_end_id",
    "text",
    "head_token_id",
    "head_text",
    "head_lemma",
    "head_pos",
    "ent_type",
    "is_first_mention",
)
FAILURE_DATA_COLUMNS = ("reason", "detail", "model_tokens", "max_model_tokens")


class CoreferenceResolver(BaseTranslator):
    """Resolve document-level coreference without rewriting source text.

    This is intentionally a teaching-scale translator.  It materializes the selected
    document and token artifacts once, which keeps the alignment rules transparent and
    makes failures atomic per document.  Later Bag-of-Ideas migration work can add a
    sharded backend without changing these artifact contracts.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        model: str = "lingmess",
        text_field: str = "text",
        sentence_key: str = "sentence_id",
        token_key: str = "token_id",
        device: str = "auto",
        strict_device: bool = False,
        compile_model: bool = False,
        max_tokens_in_batch: int = 2_000,
        show_progress: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        normalized_model = model.strip().lower().replace("-", "").replace("_", "")
        if normalized_model in {"fcoref", "fastcoref"}:
            normalized_model = "fcoref"
        elif normalized_model in {"lingmess", "lingmesscoref"}:
            normalized_model = "lingmess"
        else:
            raise ValueError("model must be 'lingmess' or 'fcoref'.")
        if not text_field:
            raise ValueError("text_field must be non-empty.")
        if not sentence_key or not token_key or sentence_key == token_key:
            raise ValueError("sentence_key and token_key must be distinct non-empty names.")
        if isinstance(max_tokens_in_batch, bool) or int(max_tokens_in_batch) < 1:
            raise ValueError("max_tokens_in_batch must be a positive integer.")
        self.model = normalized_model
        self.text_field = str(text_field)
        self.sentence_key = str(sentence_key)
        self.token_key = str(token_key)
        self.device = str(device)
        self.strict_device = bool(strict_device)
        self.compile_model = bool(compile_model)
        self.max_tokens_in_batch = int(max_tokens_in_batch)
        self.show_progress = bool(show_progress)

    def output_specs(self, *, sources: Mapping[str, "BaseArtifact"], request: TranslationRequest):
        _ = request
        docs, tokens = _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        collisions = [name for name in ("cluster_id", "mention_id") if name in docs.primary_key]
        if collisions:
            raise OperatorError(f"Coreference output keys collide with document keys: {collisions}.")
        _ = tokens
        return {
            MENTIONS: OutputSpec(
                artifact_type="table", lineage_mode="extended_key", basis_labels=DOCUMENTS
            ),
            FAILURES: OutputSpec(
                artifact_type="table", lineage_mode="preserved_key", basis_labels=DOCUMENTS
            ),
        }

    def validate_operation_params(self, params, *, sources, mode):
        _ = sources, mode
        if params:
            raise OperatorError(
                f"CoreferenceResolver does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(self, *, sources, mode, request):
        _ = mode, request
        _validate_sources(sources, sentence_key=self.sentence_key, token_key=self.token_key)
        return {
            DOCUMENTS: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
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
        if set(inputs) != {DOCUMENTS, TOKENS}:
            raise OperatorError(
                f"CoreferenceResolver expects inputs {DOCUMENTS!r} and {TOKENS!r}."
            )
        doc_packet = inputs[DOCUMENTS]
        token_packet = inputs[TOKENS]
        documents = _frame(doc_packet.data, "documents")
        tokens = _frame(token_packet.data, "tokens")
        doc_keys = list(doc_packet.primary_key)
        expected_token_keys = [*doc_keys, self.sentence_key, self.token_key]
        if list(token_packet.primary_key) != expected_token_keys:
            raise ArtifactError(
                "CoreferenceResolver requires token primary keys to equal document keys + "
                f"{self.sentence_key!r} + {self.token_key!r}; got {list(token_packet.primary_key)}."
            )
        missing_docs = [name for name in [*doc_keys, self.text_field] if name not in documents]
        missing_tokens = [
            name
            for name in [
                *expected_token_keys,
                "text",
                "lemma",
                "pos",
                "dep",
                "head_token_id",
                "ent_type",
                "char_start",
                "char_end",
            ]
            if name not in tokens
        ]
        if missing_docs or missing_tokens:
            raise ArtifactError(
                f"CoreferenceResolver missing document columns {missing_docs} and token columns {missing_tokens}."
            )

        resolved_device = resolve_devices(self.device, strict=self.strict_device).primary
        runtime = _make_runtime(
            model=self.model,
            device=resolved_device,
            compile_model=self.compile_model,
            show_progress=self.show_progress,
        )

        token_groups = {
            key: group.sort_values([self.sentence_key, self.token_key]).reset_index(drop=True)
            for key, group in tokens.groupby(doc_keys, sort=False, dropna=False)
        }
        if len(doc_keys) == 1:
            token_groups = {
                (key if isinstance(key, tuple) else (key,)): value
                for key, value in token_groups.items()
            }

        mention_keys: list[dict[str, Any]] = []
        mention_data: list[dict[str, Any]] = []
        failure_keys: list[dict[str, Any]] = []
        failure_data: list[dict[str, Any]] = []

        for _, row in documents.iterrows():
            key_record = {name: int(row[name]) for name in doc_keys}
            key_tuple = tuple(key_record[name] for name in doc_keys)
            text = "" if pd.isna(row[self.text_field]) else str(row[self.text_field])
            model_tokens: int | None = None
            max_model_tokens: int | None = None
            try:
                counts, configured_limit = runtime.document_token_counts([text])
                model_tokens = int(counts[0]) if counts else 0
                max_model_tokens = None if configured_limit is None else int(configured_limit)
                if max_model_tokens is not None and model_tokens > max_model_tokens:
                    _append_failure(
                        failure_keys,
                        failure_data,
                        key_record,
                        reason="document_too_long",
                        detail=(
                            f"Document has {model_tokens} model tokens; configured maximum is "
                            f"{max_model_tokens}."
                        ),
                        model_tokens=model_tokens,
                        max_model_tokens=max_model_tokens,
                    )
                    continue
                batch = runtime.predict_texts(
                    [text], max_tokens_in_batch=self.max_tokens_in_batch, release_logits=True
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _clear_cuda_cache()
                    _append_failure(
                        failure_keys,
                        failure_data,
                        key_record,
                        reason="cuda_out_of_memory",
                        detail=f"{type(exc).__name__}: {exc}",
                        model_tokens=model_tokens,
                        max_model_tokens=max_model_tokens,
                    )
                    continue
                raise
            prediction = batch.documents[0]
            if prediction.normalization_issue is not None:
                _append_failure(
                    failure_keys,
                    failure_data,
                    key_record,
                    reason=prediction.normalization_issue.reason,
                    detail=prediction.normalization_issue.detail,
                    model_tokens=model_tokens,
                    max_model_tokens=max_model_tokens,
                )
                continue
            doc_tokens = token_groups.get(key_tuple)
            for mention in prediction.mentions:
                aligned = _align_mention(
                    doc_tokens,
                    start_char=int(mention.start_char),
                    end_char=int(mention.end_char),
                    sentence_key=self.sentence_key,
                    token_key=self.token_key,
                )
                mention_keys.append(
                    {
                        **key_record,
                        "cluster_id": int(mention.cluster_id),
                        "mention_id": int(mention.mention_id),
                    }
                )
                mention_data.append(
                    {
                        **aligned,
                        "char_start": int(mention.start_char),
                        "char_end": int(mention.end_char),
                        "text": str(mention.text),
                        "is_first_mention": bool(mention.mention_id == 0),
                    }
                )

        outputs: dict[str, Mapping[str, Any]] = {
            MENTIONS: {
                "keys": pd.DataFrame.from_records(
                    mention_keys, columns=[*doc_keys, "cluster_id", "mention_id"]
                ),
                "data": pd.DataFrame.from_records(
                    mention_data, columns=list(MENTION_DATA_COLUMNS)
                ),
            },
            FAILURES: {
                "keys": pd.DataFrame.from_records(failure_keys, columns=doc_keys),
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
        return None

    def to_json_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text_field": self.text_field,
            "sentence_key": self.sentence_key,
            "token_key": self.token_key,
            "device": self.device,
            "strict_device": self.strict_device,
            "compile_model": self.compile_model,
            "max_tokens_in_batch": self.max_tokens_in_batch,
            "show_progress": self.show_progress,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "CoreferenceResolver":
        return cls(**cast(dict[str, Any], dict(state)))


def _make_runtime(*, model: str, device: str, compile_model: bool, show_progress: bool):
    paths = user_cache_paths()
    return FastCorefRuntime(
        paths.root,
        model=model,
        device=device,
        compile_model=compile_model,
        show_progress=show_progress,
        cache_dir=paths.huggingface_hub,
    )


def _validate_sources(sources, *, sentence_key: str, token_key: str):
    if set(sources) != {DOCUMENTS, TOKENS}:
        raise OperatorError(
            f"CoreferenceResolver expects source labels {DOCUMENTS!r} and {TOKENS!r}; got {sorted(sources)}."
        )
    documents = sources[DOCUMENTS]
    tokens = sources[TOKENS]
    if documents.artifact_type.value != "table" or tokens.artifact_type.value != "table":
        raise OperatorError("CoreferenceResolver requires table artifacts.")
    expected = [*documents.primary_key, sentence_key, token_key]
    if list(tokens.primary_key) != expected:
        raise OperatorError(
            "CoreferenceResolver requires token primary keys to equal document keys + "
            f"{sentence_key!r} + {token_key!r}."
        )
    return documents, tokens


def _frame(value: Any, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(f"CoreferenceResolver expected {label} as a pandas DataFrame.")
    return value


def _append_failure(keys, data, key_record, *, reason, detail, model_tokens, max_model_tokens):
    keys.append(dict(key_record))
    data.append(
        {
            "reason": str(reason),
            "detail": None if detail is None else str(detail),
            "model_tokens": model_tokens,
            "max_model_tokens": max_model_tokens,
        }
    )


def _align_mention(
    tokens: pd.DataFrame | None,
    *,
    start_char: int,
    end_char: int,
    sentence_key: str,
    token_key: str,
) -> dict[str, Any]:
    empty = {
        "sentence_id": None,
        "token_start_id": None,
        "token_end_id": None,
        "head_token_id": None,
        "head_text": None,
        "head_lemma": None,
        "head_pos": None,
        "ent_type": None,
    }
    if tokens is None or tokens.empty:
        return empty
    overlap = tokens.loc[(tokens["char_start"] < end_char) & (tokens["char_end"] > start_char)]
    if overlap.empty or overlap[sentence_key].nunique(dropna=False) != 1:
        return empty
    sentence_id = int(overlap.iloc[0][sentence_key])
    sentence = tokens.loc[tokens[sentence_key] == sentence_id].sort_values(token_key).reset_index(drop=True)
    span_positions = sentence.index[
        (sentence["char_start"] < end_char) & (sentence["char_end"] > start_char)
    ].tolist()
    if not span_positions:
        return empty
    span_start = min(span_positions)
    span_end = max(span_positions) + 1
    head_index: int | None = None
    try:
        heads = content_head_indices(
            start=span_start,
            end=span_end,
            token_ids=[int(v) for v in sentence[token_key]],
            head_token_ids=[None if pd.isna(v) else int(v) for v in sentence["head_token_id"]],
            dependencies=[None if pd.isna(v) else str(v) for v in sentence["dep"]],
            pos=[None if pd.isna(v) else str(v) for v in sentence["pos"]],
            text=[str(v) for v in sentence["text"]],
            role="COREF",
        )
        head_index = heads[0] if heads else None
    except (ValueError, IndexError, KeyError):
        head_index = None
    head = None if head_index is None else sentence.iloc[int(head_index)]
    token_values = [int(v) for v in overlap[token_key]]
    return {
        "sentence_id": sentence_id,
        "token_start_id": min(token_values),
        "token_end_id": max(token_values) + 1,
        "head_token_id": None if head is None else int(head[token_key]),
        "head_text": None if head is None else str(head["text"]),
        "head_lemma": None if head is None or pd.isna(head["lemma"]) else str(head["lemma"]),
        "head_pos": None if head is None or pd.isna(head["pos"]) else str(head["pos"]),
        "ent_type": None if head is None or pd.isna(head["ent_type"]) else str(head["ent_type"]),
    }


def _is_cuda_oom(exc: Exception) -> bool:
    return exc.__class__.__name__ == "OutOfMemoryError" or "cuda out of memory" in str(exc).lower()


def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
