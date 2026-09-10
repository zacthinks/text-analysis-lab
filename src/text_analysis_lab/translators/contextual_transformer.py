"""Model-native tokenization plus contextual token embeddings in one operation."""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BatchResult,
    BaseTranslator,
    ColumnRequest,
    InputBatch,
    OutputMap,
    OutputSpec,
    RunRoute,
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._hf_utils import (
    ContextWindowExceededError,
    TransformerResourceError,
    count_tokens,
    detect_context_limit,
    hidden_size,
    require_frame,
    resolve_device,
    resolved_commit_hash,
    single_input,
    single_source,
    warn_if_unpinned,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

TruncationPolicy = Literal["error", "truncate"]

TOKENS_LABEL = "tokens"
CONTEXTUAL_EMBEDDINGS_LABEL = "contextual_embeddings"
TOKEN_COLUMNS = (
    "token",
    "input_id",
    "char_start",
    "char_end",
    "is_special",
    "token_type_id",
)
TOKEN_METADATA_COLUMNS = (
    "source_token_count",
    "embedded_token_count",
    "truncated",
)


class ContextualTransformer(BaseTranslator):
    """Tokenize text and emit one contextual embedding per retained model token.

    The tokenizer and encoder are loaded from the same Hugging Face model id and
    revision inside one translator. This intentionally prevents mismatched
    tokenizer/model combinations. The two durable outputs are:

    ``tokens``
        A table with the source key extended by ``token_id``. It contains the
        exact model tokens (including structural special tokens), input ids,
        offsets, and token-type ids where applicable. Padding tokens are never
        persisted.

    ``contextual_embeddings``
        A dense matrix with the same key as ``tokens`` and one row per retained
        model token. Its basis is the ``tokens`` output.
    """

    operation_type = "translate"

    def __init__(
        self,
        model: str,
        *,
        text_field: str = "text",
        token_key: str = "token_id",
        revision: str | None = None,
        truncation: TruncationPolicy | str = "error",
        max_length: int | None = None,
        save_model: bool = False,
        trust_remote_code: bool = False,
        operator_id: str | None = None,
        resolved_revision: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty Hugging Face model id or local path.")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        if not isinstance(token_key, str) or not token_key or token_key.startswith("_"):
            raise ValueError("token_key must be a non-empty non-structural column name.")
        policy = str(truncation).lower()
        if policy not in {"error", "truncate"}:
            raise ValueError("truncation must be either 'error' or 'truncate'.")
        if max_length is not None and int(max_length) <= 0:
            raise ValueError("max_length must be positive or None.")

        self.model = model.strip()
        self.text_field = text_field
        self.token_key = token_key
        self.revision = None if revision is None else str(revision)
        self.resolved_revision = None if resolved_revision is None else str(resolved_revision)
        self.truncation = cast(TruncationPolicy, policy)
        self.max_length = None if max_length is None else int(max_length)
        self.save_model = bool(save_model)
        self.trust_remote_code = bool(trust_remote_code)

        self._local_model_dir: Path | None = None
        self._runtime_model: Any = None
        self._runtime_tokenizer: Any = None
        self._runtime_device: str | None = None
        self._effective_context_limit: int | None = None
        self._warned_unpinned = False

    @property
    def supports_parallel_translate(self) -> bool:
        # Avoid multiplying large model weights across TeAL process workers.
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route == "sequential"

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        source = single_source(sources, translator_name="ContextualTransformer")
        if self.token_key in set(str(name) for name in source.primary_key):
            raise OperatorError(
                f"ContextualTransformer token_key {self.token_key!r} already exists in the source primary key."
            )
        return {
            TOKENS_LABEL: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            ),
            CONTEXTUAL_EMBEDDINGS_LABEL: OutputSpec(
                artifact_type="dense_matrix",
                lineage_mode="preserved_key",
                basis_labels=TOKENS_LABEL,
            ),
        }

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        unknown = sorted(set(params) - {"device", "model_batch_size"})
        if unknown:
            raise OperatorError(
                f"ContextualTransformer received unknown operation parameter(s): {unknown}."
            )
        model_batch_size = int(params.get("model_batch_size", 16))
        if model_batch_size <= 0:
            raise OperatorError("model_batch_size must be a positive integer.")
        return {
            "device": resolve_device(str(params.get("device", "auto"))),
            "model_batch_size": model_batch_size,
        }

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = single_source(sources, translator_name="ContextualTransformer")
        if source.artifact_type.value != "table":
            raise OperatorError("ContextualTransformer requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 128,
            form="table",
            metadata_mode="none",
            include_position=False,
        )

    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        if mode != "translate":
            raise OperatorError(f"Unsupported ContextualTransformer mode {mode!r}.")
        packet = single_input(inputs, translator_name="ContextualTransformer")
        frame = require_frame(packet.data, translator_name="ContextualTransformer")
        key_columns = [str(name) for name in packet.primary_key]
        missing = [name for name in [*key_columns, self.text_field] if name not in frame.columns]
        if missing:
            raise ArtifactError(f"ContextualTransformer source batch is missing columns {missing}.")

        device = str(request.params.get("device", "cpu"))
        model_batch_size = int(request.params.get("model_batch_size", 16))
        model, tokenizer = self._runtime_components(device=device)
        context_limit = self._context_limit(tokenizer=tokenizer, model=model)

        texts = frame[self.text_field].fillna("").astype(str).tolist()
        source_counts = count_tokens(tokenizer, texts)
        too_long = [i for i, count in enumerate(source_counts) if count > context_limit]
        if too_long and self.truncation == "error":
            examples: list[str] = []
            for index in too_long[:5]:
                key = {name: int(frame.iloc[index][name]) for name in key_columns}
                examples.append(f"{key}: {source_counts[index]} tokens")
            raise ContextWindowExceededError(
                "ContextualTransformer refuses silent truncation: "
                f"{len(too_long)} of {len(texts)} source row(s) exceed the effective "
                f"context limit of {context_limit} tokens (including special tokens). "
                f"Examples: {', '.join(examples)}. Decompose/chunk the text upstream, "
                "choose a longer-context model, or explicitly set truncation='truncate'."
            )
        if too_long:
            warnings.warn(
                "ContextualTransformer is explicitly truncating "
                f"{len(too_long)} of {len(texts)} row(s) to {context_limit} model tokens. "
                "The retained token artifact and audit metadata record the truncation.",
                UserWarning,
                stacklevel=2,
            )

        key_records = frame.loc[:, key_columns].to_dict(orient="records")
        token_key_rows: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        token_metadata_rows: list[dict[str, Any]] = []
        embedding_blocks: list[np.ndarray] = []

        for start in range(0, len(texts), model_batch_size):
            stop = min(start + model_batch_size, len(texts))
            block = self._encode_block(
                model=model,
                tokenizer=tokenizer,
                texts=texts[start:stop],
                device=device,
                context_limit=context_limit,
            )
            for local_index, row in enumerate(block):
                source_index = start + local_index
                source_key = key_records[source_index]
                embedded_count = len(row["input_ids"])
                is_truncated = embedded_count < source_counts[source_index]
                for token_id in range(embedded_count):
                    key = dict(source_key)
                    key[self.token_key] = token_id
                    token_key_rows.append(key)
                    special = bool(row["special_tokens_mask"][token_id])
                    offset = row["offsets"][token_id]
                    token_rows.append(
                        {
                            "token": row["tokens"][token_id],
                            "input_id": int(row["input_ids"][token_id]),
                            "char_start": None if special else int(offset[0]),
                            "char_end": None if special else int(offset[1]),
                            "is_special": special,
                            "token_type_id": (
                                None
                                if row["token_type_ids"] is None
                                else int(row["token_type_ids"][token_id])
                            ),
                        }
                    )
                    token_metadata_rows.append(
                        {
                            "source_token_count": int(source_counts[source_index]),
                            "embedded_token_count": int(embedded_count),
                            "truncated": bool(is_truncated),
                        }
                    )
                embedding_blocks.append(row["embeddings"])

        token_keys = pd.DataFrame(token_key_rows, columns=[*key_columns, self.token_key])
        token_data = pd.DataFrame(token_rows, columns=list(TOKEN_COLUMNS))
        token_metadata = pd.DataFrame(token_metadata_rows, columns=list(TOKEN_METADATA_COLUMNS))
        if not token_data.empty:
            token_data["input_id"] = token_data["input_id"].astype("int64")
            token_data["char_start"] = token_data["char_start"].astype("Int64")
            token_data["char_end"] = token_data["char_end"].astype("Int64")
            token_data["is_special"] = token_data["is_special"].astype(bool)
            token_data["token_type_id"] = token_data["token_type_id"].astype("Int64")
        if embedding_blocks:
            values = np.concatenate(embedding_blocks, axis=0).astype(np.float32, copy=False)
        else:  # pragma: no cover - standard tokenizers emit at least special tokens
            values = np.empty((0, hidden_size(model)), dtype=np.float32)
        if len(values) != len(token_keys):
            raise ArtifactError(
                "ContextualTransformer internal alignment failure: token rows and contextual "
                f"embedding rows differ ({len(token_keys)} vs {len(values)})."
            )
        columns = [f"dim_{index}" for index in range(values.shape[1])]

        return BatchResult(
            outputs={
                TOKENS_LABEL: {
                    "keys": token_keys,
                    "data": token_data,
                    "metadata": token_metadata,
                },
                CONTEXTUAL_EMBEDDINGS_LABEL: {
                    "keys": token_keys.copy(),
                    "data": {"values": values, "columns": columns},
                },
            }
        )

    def _encode_block(
        self,
        *,
        model: Any,
        tokenizer: Any,
        texts: Sequence[str],
        device: str,
        context_limit: int,
    ) -> list[dict[str, Any]]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise TransformerResourceError(
                "ContextualTransformer requires PyTorch. Install TeAL's 'transformers' extra."
            ) from exc

        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=self.truncation == "truncate",
            max_length=context_limit if self.truncation == "truncate" else None,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise TransformerResourceError("Tokenizer output did not include input_ids.")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise TransformerResourceError(
                "ContextualTransformer requires a fast Hugging Face tokenizer that provides "
                "return_offsets_mapping so durable model tokens can be aligned to source text."
            )
        special_tokens_mask = encoded.get("special_tokens_mask")
        token_type_ids = encoded.get("token_type_ids")

        model_input_names = set(getattr(tokenizer, "model_input_names", ()))
        if not model_input_names:
            model_input_names = {"input_ids", "attention_mask", "token_type_ids"}
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in model_input_names and hasattr(value, "to")
        }
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = attention_mask.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                try:
                    hidden = outputs[0]
                except Exception as exc:
                    raise TransformerResourceError(
                        "Transformer model output does not expose last_hidden_state."
                    ) from exc
        hidden = hidden.detach().to("cpu", dtype=torch.float32)

        rows: list[dict[str, Any]] = []
        for index in range(len(texts)):
            length = int(attention_mask[index].sum().item())
            ids = input_ids[index, :length].detach().cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids)
            row_offsets = offsets[index, :length].detach().cpu().tolist()
            if special_tokens_mask is None:
                row_special = tokenizer.get_special_tokens_mask(
                    ids, already_has_special_tokens=True
                )
            else:
                row_special = special_tokens_mask[index, :length].detach().cpu().tolist()
            row_type_ids = (
                None
                if token_type_ids is None
                else token_type_ids[index, :length].detach().cpu().tolist()
            )
            rows.append(
                {
                    "input_ids": ids,
                    "tokens": [str(value) for value in tokens],
                    "offsets": row_offsets,
                    "special_tokens_mask": row_special,
                    "token_type_ids": row_type_ids,
                    "embeddings": hidden[index, :length, :].numpy(),
                }
            )
        return rows

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

    def download(self) -> str | None:
        self._runtime_components(device="cpu")
        return self.resolved_revision

    def to_json_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text_field": self.text_field,
            "token_key": self.token_key,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "save_model": self.save_model,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "ContextualTransformer":
        return cls(
            model=str(state.get("model", "")),
            text_field=str(state.get("text_field", "text")),
            token_key=str(state.get("token_key", "token_id")),
            revision=cast(str | None, state.get("revision")),
            resolved_revision=cast(str | None, state.get("resolved_revision")),
            truncation=str(state.get("truncation", "error")),
            max_length=cast(int | None, state.get("max_length")),
            save_model=bool(state.get("save_model", False)),
            trust_remote_code=bool(state.get("trust_remote_code", False)),
        )

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.save_model:
            return {}
        model, tokenizer = self._runtime_components(device="cpu")
        model_dir = assets_dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(model_dir, safe_serialization=True)
        except TypeError:  # pragma: no cover
            model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        return {"model_dir": model_dir.name, "resolved_revision": self.resolved_revision}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            return
        model_dir = manifest.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise OperatorError("ContextualTransformer asset manifest is missing model_dir.")
        path = assets_dir / model_dir
        if not path.exists():
            raise OperatorError(f"Saved ContextualTransformer model directory is missing: {path}.")
        self._local_model_dir = path
        if manifest.get("resolved_revision") is not None:
            self.resolved_revision = str(manifest["resolved_revision"])

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        _ = mode, route
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        state = self.to_json_state()
        state["operator_id"] = str(operator_id)
        state["local_model_dir"] = (
            None if self._local_model_dir is None else str(self._local_model_dir)
        )
        (intermediate_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> "ContextualTransformer":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise OperatorError("ContextualTransformer intermediate state must be a mapping.")
        obj = cls.from_json_state(state)
        local_model_dir = state.get("local_model_dir")
        if isinstance(local_model_dir, str) and local_model_dir and Path(local_model_dir).exists():
            obj._local_model_dir = Path(local_model_dir)
        obj.operator_id = str(operator_id)
        return obj

    def _runtime_components(self, *, device: str) -> tuple[Any, Any]:
        if (
            self._runtime_model is not None
            and self._runtime_tokenizer is not None
            and self._runtime_device == device
        ):
            return self._runtime_model, self._runtime_tokenizer
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise TransformerResourceError(
                "ContextualTransformer requires Hugging Face Transformers. Install with "
                "`uv sync --extra transformers` or `pip install text-analysis-lab[transformers]`."
            ) from exc

        if self._local_model_dir is not None:
            source: str | Path = self._local_model_dir
            common = {
                "local_files_only": True,
                "trust_remote_code": self.trust_remote_code,
            }
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **common)
                model = AutoModel.from_pretrained(source, **common)
            except Exception as exc:  # pragma: no cover
                raise TransformerResourceError(
                    f"Could not load operator-local ContextualTransformer model from {source}."
                ) from exc
        else:
            revision = self.resolved_revision or self.revision
            common = {"revision": revision, "trust_remote_code": self.trust_remote_code}
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model, use_fast=True, local_files_only=True, **common
                )
                model = AutoModel.from_pretrained(
                    self.model, local_files_only=True, **common
                )
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model, use_fast=True, **common
                    )
                    model = AutoModel.from_pretrained(self.model, **common)
                except Exception as exc:  # pragma: no cover
                    raise TransformerResourceError(
                        f"Could not load Hugging Face model {self.model!r} at revision "
                        f"{revision or 'main'!r}. TeAL checked the local Hugging Face cache "
                        "first and then attempted the normal provider load/download."
                    ) from exc
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise TransformerResourceError(
                "ContextualTransformer requires a fast Hugging Face tokenizer so it can "
                "persist character offsets for model-native tokens."
            )
        try:
            model = model.to(device)
        except Exception as exc:
            raise TransformerResourceError(
                f"Could not move Hugging Face model {self.model!r} to device {device!r}."
            ) from exc
        commit = resolved_commit_hash(model, tokenizer)
        if commit:
            self.resolved_revision = commit
        self._warned_unpinned = warn_if_unpinned(
            model_name=self.model,
            requested_revision=self.revision,
            resolved_revision=self.resolved_revision,
            warned=self._warned_unpinned,
            kind="Contextual transformer model",
        )
        self._runtime_model = model
        self._runtime_tokenizer = tokenizer
        self._runtime_device = device
        self._effective_context_limit = None
        return model, tokenizer

    def _context_limit(self, *, tokenizer: Any, model: Any) -> int:
        if self._effective_context_limit is None:
            self._effective_context_limit = detect_context_limit(
                tokenizer,
                getattr(model, "config", None),
                explicit=self.max_length,
            )
        return self._effective_context_limit
