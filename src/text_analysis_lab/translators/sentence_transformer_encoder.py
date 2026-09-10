"""SentenceTransformers-backed translator for one embedding per source row."""

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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL
from text_analysis_lab.translators._hf_utils import (
    ContextWindowExceededError,
    TransformerResourceError,
    count_tokens,
    detect_context_limit,
    require_frame,
    resolve_device,
    resolved_commit_hash,
    single_input,
    single_source,
    warn_if_unpinned,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

SentenceTask = Literal["document", "query", "generic"]
TruncationPolicy = Literal["error", "truncate"]


class SentenceTransformerEncoder(BaseTranslator):
    """Encode each source text as one SentenceTransformers embedding.

    The SentenceTransformers model owns the embedding recipe: transformer,
    pooling, normalization modules, prompts, and task routing are loaded from
    the model itself. TeAL does not reconstruct that recipe manually.

    ``task='document'`` is the default because TeAL's primary course use is
    document/corpus representation. Models with document/passsage/corpus
    prompts or Router modules therefore receive their document-side behavior.
    """

    operation_type = "translate"

    def __init__(
        self,
        model: str,
        *,
        text_field: str = "text",
        revision: str | None = None,
        task: SentenceTask | str = "document",
        prompt_name: str | None = None,
        prompt: str | None = None,
        normalize: bool = False,
        truncation: TruncationPolicy | str = "error",
        max_length: int | None = None,
        save_model: bool = False,
        trust_remote_code: bool = False,
        operator_id: str | None = None,
        resolved_revision: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty SentenceTransformers model id or path.")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        task_value = str(task).lower()
        if task_value not in {"document", "query", "generic"}:
            raise ValueError("task must be 'document', 'query', or 'generic'.")
        policy = str(truncation).lower()
        if policy not in {"error", "truncate"}:
            raise ValueError("truncation must be either 'error' or 'truncate'.")
        if prompt_name is not None and prompt is not None:
            raise ValueError("Specify at most one of prompt_name or prompt.")
        if max_length is not None and int(max_length) <= 0:
            raise ValueError("max_length must be positive or None.")

        self.model = model.strip()
        self.text_field = text_field
        self.revision = None if revision is None else str(revision)
        self.resolved_revision = None if resolved_revision is None else str(resolved_revision)
        self.task = cast(SentenceTask, task_value)
        self.prompt_name = None if prompt_name is None else str(prompt_name)
        self.prompt = None if prompt is None else str(prompt)
        self.normalize = bool(normalize)
        self.truncation = cast(TruncationPolicy, policy)
        self.max_length = None if max_length is None else int(max_length)
        self.save_model = bool(save_model)
        self.trust_remote_code = bool(trust_remote_code)

        self._local_model_dir: Path | None = None
        self._runtime_model: Any = None
        self._runtime_device: str | None = None
        self._effective_context_limit: int | None = None
        self._warned_unpinned = False

    @property
    def supports_parallel_translate(self) -> bool:
        # SentenceTransformers can already batch efficiently on CPU/GPU. Avoid
        # loading a separate model into every TeAL process worker by default.
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route == "sequential"

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        single_source(sources, translator_name="SentenceTransformerEncoder")
        return OutputSpec(
            artifact_type="dense_matrix",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

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
                f"SentenceTransformerEncoder received unknown operation parameter(s): {unknown}."
            )
        model_batch_size = int(params.get("model_batch_size", 32))
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
        source = single_source(sources, translator_name="SentenceTransformerEncoder")
        if source.artifact_type.value != "table":
            raise OperatorError("SentenceTransformerEncoder requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 256,
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
            raise OperatorError(f"Unsupported SentenceTransformerEncoder mode {mode!r}.")
        packet = single_input(inputs, translator_name="SentenceTransformerEncoder")
        frame = require_frame(packet.data, translator_name="SentenceTransformerEncoder")
        key_columns = [str(name) for name in packet.primary_key]
        missing = [name for name in [*key_columns, self.text_field] if name not in frame.columns]
        if missing:
            raise ArtifactError(
                f"SentenceTransformerEncoder source batch is missing columns {missing}."
            )

        device = str(request.params.get("device", "cpu"))
        model_batch_size = int(request.params.get("model_batch_size", 32))
        model = self._runtime_component(device=device)
        tokenizer = _sentence_tokenizer(model)
        context_limit = self._context_limit(model=model, tokenizer=tokenizer)
        prompt_name, prompt, prompt_prefix = _resolve_prompt(
            model,
            task=self.task,
            prompt_name=self.prompt_name,
            prompt=self.prompt,
        )

        texts = frame[self.text_field].fillna("").astype(str).tolist()
        counted_texts = [f"{prompt_prefix}{text}" for text in texts]
        token_counts = count_tokens(tokenizer, counted_texts)
        too_long = [index for index, count in enumerate(token_counts) if count > context_limit]
        if too_long and self.truncation == "error":
            examples: list[str] = []
            for index in too_long[:5]:
                key = {name: int(frame.iloc[index][name]) for name in key_columns}
                examples.append(f"{key}: {token_counts[index]} tokens")
            raise ContextWindowExceededError(
                "SentenceTransformerEncoder refuses silent truncation: "
                f"{len(too_long)} of {len(texts)} source row(s) exceed the effective "
                f"context limit of {context_limit} model tokens (including any model prompt "
                f"and special tokens). Examples: {', '.join(examples)}. Decompose/chunk the "
                "text upstream, choose a longer-context model, or explicitly set "
                "truncation='truncate'."
            )
        if too_long:
            warnings.warn(
                "SentenceTransformerEncoder is explicitly allowing the model to truncate "
                f"{len(too_long)} of {len(texts)} row(s) to {context_limit} model tokens. "
                "The original and embedded token counts are recorded in output metadata.",
                UserWarning,
                stacklevel=2,
            )

        values = self._encode(
            model=model,
            texts=texts,
            device=device,
            model_batch_size=model_batch_size,
            prompt_name=prompt_name,
            prompt=prompt,
        )
        if values.ndim != 2 or values.shape[0] != len(texts):
            raise TransformerResourceError(
                "SentenceTransformerEncoder expected one 2D embedding row per source text; "
                f"received shape {tuple(values.shape)}."
            )
        values = values.astype(np.float32, copy=False)
        embedded_counts = [min(count, context_limit) for count in token_counts]
        metadata = pd.DataFrame(
            {
                "token_count": np.asarray(token_counts, dtype=np.int64),
                "embedded_token_count": np.asarray(embedded_counts, dtype=np.int64),
                "truncated": np.asarray(token_counts, dtype=np.int64)
                > np.asarray(embedded_counts, dtype=np.int64),
            }
        )
        columns = [f"dim_{index}" for index in range(values.shape[1])]
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": frame.loc[:, key_columns].reset_index(drop=True),
                    "metadata": metadata,
                    "data": {"values": values, "columns": columns},
                }
            }
        )

    def _encode(
        self,
        *,
        model: Any,
        texts: Sequence[str],
        device: str,
        model_batch_size: int,
        prompt_name: str | None,
        prompt: str | None,
        task: SentenceTask | None = None,
    ) -> np.ndarray:
        kwargs = {
            "batch_size": model_batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize,
            "device": device,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        if prompt is not None:
            kwargs["prompt"] = prompt
        effective_task = self.task if task is None else task
        if effective_task == "document" and callable(getattr(model, "encode_document", None)):
            values = model.encode_document(list(texts), **kwargs)
        elif effective_task == "query" and callable(getattr(model, "encode_query", None)):
            values = model.encode_query(list(texts), **kwargs)
        else:
            values = model.encode(list(texts), **kwargs)
        return np.asarray(values)

    def transform_external_texts(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode new documents/queries with the frozen SentenceTransformer recipe."""
        values_in = ["" if value is None else str(value) for value in texts]
        operation_params = dict(params or {})
        # Device choice is an execution detail, not representation semantics. Re-resolve
        # it at replay time so a frozen GPU run remains usable on a CPU-only machine.
        device = resolve_device("auto")
        model_batch_size = int(operation_params.get("model_batch_size", 32))
        model = self._runtime_component(device=device)
        tokenizer = _sentence_tokenizer(model)
        context_limit = self._context_limit(model=model, tokenizer=tokenizer)
        task = cast(SentenceTask, "query" if query else self.task)
        prompt_name, prompt, prompt_prefix = _resolve_prompt(
            model,
            task=task,
            prompt_name=(None if query else self.prompt_name),
            prompt=(None if query else self.prompt),
        )
        counted = [f"{prompt_prefix}{text}" for text in values_in]
        token_counts = count_tokens(tokenizer, counted)
        too_long = [index for index, count in enumerate(token_counts) if count > context_limit]
        if too_long and self.truncation == "error":
            raise ContextWindowExceededError(
                "SentenceTransformerEncoder refuses silent truncation while replaying "
                f"new text: {len(too_long)} of {len(values_in)} row(s) exceed the "
                f"effective context limit of {context_limit} model tokens."
            )
        if too_long:
            warnings.warn(
                "SentenceTransformerEncoder is explicitly allowing truncation while "
                f"replaying {len(too_long)} new text row(s).",
                UserWarning,
                stacklevel=2,
            )
        result = self._encode(
            model=model,
            texts=values_in,
            device=device,
            model_batch_size=model_batch_size,
            prompt_name=prompt_name,
            prompt=prompt,
            task=task,
        )
        return np.asarray(result, dtype=np.float32)

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "texts"

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
        self._runtime_component(device="cpu")
        return self.resolved_revision

    def to_json_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text_field": self.text_field,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "task": self.task,
            "prompt_name": self.prompt_name,
            "prompt": self.prompt,
            "normalize": self.normalize,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "save_model": self.save_model,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "SentenceTransformerEncoder":
        return cls(
            model=str(state.get("model", "")),
            text_field=str(state.get("text_field", "text")),
            revision=cast(str | None, state.get("revision")),
            resolved_revision=cast(str | None, state.get("resolved_revision")),
            task=str(state.get("task", "document")),
            prompt_name=cast(str | None, state.get("prompt_name")),
            prompt=cast(str | None, state.get("prompt")),
            normalize=bool(state.get("normalize", False)),
            truncation=str(state.get("truncation", "error")),
            max_length=cast(int | None, state.get("max_length")),
            save_model=bool(state.get("save_model", False)),
            trust_remote_code=bool(state.get("trust_remote_code", False)),
        )

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.save_model:
            return {}
        model = self._runtime_component(device="cpu")
        model_dir = assets_dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(model_dir, safe_serialization=True)
        except TypeError:  # pragma: no cover
            model.save_pretrained(model_dir)
        return {"model_dir": model_dir.name, "resolved_revision": self.resolved_revision}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            return
        model_dir = manifest.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise OperatorError("SentenceTransformerEncoder asset manifest is missing model_dir.")
        path = assets_dir / model_dir
        if not path.exists():
            raise OperatorError(
                f"Saved SentenceTransformerEncoder model directory is missing: {path}."
            )
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
    ) -> "SentenceTransformerEncoder":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise OperatorError("SentenceTransformerEncoder intermediate state must be a mapping.")
        obj = cls.from_json_state(state)
        local_model_dir = state.get("local_model_dir")
        if isinstance(local_model_dir, str) and local_model_dir and Path(local_model_dir).exists():
            obj._local_model_dir = Path(local_model_dir)
        obj.operator_id = str(operator_id)
        return obj

    def _runtime_component(self, *, device: str) -> Any:
        if self._runtime_model is not None and self._runtime_device == device:
            return self._runtime_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise TransformerResourceError(
                "SentenceTransformerEncoder requires sentence-transformers. Install with "
                "`uv sync --extra transformers` or `pip install text-analysis-lab[transformers]`."
            ) from exc

        if self._local_model_dir is not None:
            source: str | Path = self._local_model_dir
            try:
                model = SentenceTransformer(
                    str(source),
                    device=device,
                    local_files_only=True,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:  # pragma: no cover
                raise TransformerResourceError(
                    f"Could not load operator-local SentenceTransformer model from {source}."
                ) from exc
        else:
            revision = self.resolved_revision or self.revision
            common = {
                "device": device,
                "revision": revision,
                "trust_remote_code": self.trust_remote_code,
            }
            try:
                model = SentenceTransformer(self.model, local_files_only=True, **common)
            except Exception:
                try:
                    model = SentenceTransformer(self.model, **common)
                except Exception as exc:  # pragma: no cover
                    raise TransformerResourceError(
                        f"Could not load SentenceTransformers model {self.model!r} at revision "
                        f"{revision or 'main'!r}. TeAL checked the local cache first and then "
                        "attempted the normal provider load/download."
                    ) from exc

        tokenizer = _sentence_tokenizer(model)
        backbone = _sentence_backbone(model)
        commit = resolved_commit_hash(backbone, tokenizer)
        if commit:
            self.resolved_revision = commit
        self._warned_unpinned = warn_if_unpinned(
            model_name=self.model,
            requested_revision=self.revision,
            resolved_revision=self.resolved_revision,
            warned=self._warned_unpinned,
            kind="SentenceTransformers model",
        )
        self._runtime_model = model
        self._runtime_device = device
        self._effective_context_limit = None
        return model

    def _context_limit(self, *, model: Any, tokenizer: Any) -> int:
        if self._effective_context_limit is not None:
            return self._effective_context_limit
        backbone = _sentence_backbone(model)
        config = getattr(backbone, "config", None)
        st_limit = _finite_int(getattr(model, "max_seq_length", None))
        try:
            detected = detect_context_limit(tokenizer, config, explicit=None)
        except OperatorError:
            if st_limit is None:
                raise
            detected = st_limit
        if st_limit is not None:
            detected = min(detected, st_limit)
        if self.max_length is not None:
            if self.max_length > detected:
                raise OperatorError(
                    f"SentenceTransformerEncoder max_length={self.max_length} exceeds the "
                    f"detected model context limit of {detected}."
                )
            effective = self.max_length
        else:
            effective = detected
        # SentenceTransformers performs its own truncation at max_seq_length.
        # Pin the runtime value to the exact limit TeAL audited above.
        try:
            model.max_seq_length = int(effective)
        except Exception as exc:  # pragma: no cover - unusual custom modules
            raise TransformerResourceError(
                "Could not set SentenceTransformer max_seq_length to TeAL's audited context limit."
            ) from exc
        self._effective_context_limit = int(effective)
        return self._effective_context_limit


def _sentence_tokenizer(model: Any) -> Any:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        first = _sentence_first_module(model)
        tokenizer = getattr(first, "tokenizer", None)
    if tokenizer is None:
        raise TransformerResourceError(
            "SentenceTransformerEncoder requires a text SentenceTransformer with an accessible "
            "Hugging Face tokenizer so TeAL can audit context-window usage."
        )
    return tokenizer


def _sentence_backbone(model: Any) -> Any:
    first = _sentence_first_module(model)
    return getattr(first, "auto_model", first)


def _sentence_first_module(model: Any) -> Any:
    try:
        return model[0]
    except Exception:
        modules = getattr(model, "_modules", None)
        if isinstance(modules, Mapping) and modules:
            return next(iter(modules.values()))
    return model


def _resolve_prompt(
    model: Any,
    *,
    task: SentenceTask,
    prompt_name: str | None,
    prompt: str | None,
) -> tuple[str | None, str | None, str]:
    prompts = getattr(model, "prompts", {})
    if not isinstance(prompts, Mapping):
        prompts = {}
    if prompt is not None:
        return None, prompt, prompt
    if prompt_name is not None:
        if prompt_name not in prompts:
            raise OperatorError(
                f"SentenceTransformer prompt_name {prompt_name!r} is not available. "
                f"Available prompts: {sorted(str(key) for key in prompts)}."
            )
        return prompt_name, None, str(prompts[prompt_name])

    selected: str | None = None
    if task == "document":
        for candidate in ("document", "passage", "corpus"):
            if candidate in prompts:
                selected = candidate
                break
    elif task == "query" and "query" in prompts:
        selected = "query"
    elif task == "generic":
        default_name = getattr(model, "default_prompt_name", None)
        if isinstance(default_name, str) and default_name in prompts:
            selected = default_name
    if selected is None:
        return None, None, ""
    return selected, None, str(prompts[selected])


def _finite_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if 0 < result < 1_000_000_000:
        return result
    return None
