"""Inference-only implementation of Babelscape's WSL reader checkpoint.

This module intentionally implements only the disambiguation path needed by Bag of
Ideas: one predefined target span and an explicit candidate list. It does not import
or depend on Babelscape's WSL Python package, and it does not include span detection,
retrieval, training, or experiment-management code.

The pretrained checkpoint and tokenizer remain subject to Babelscape's
CC BY-NC-SA 4.0 license. See https://huggingface.co/Babelscape/wsl-reader-deberta-v3-base.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WSL_READER_MODEL = "Babelscape/wsl-reader-deberta-v3-base"
DEFAULT_WSL_READER_REVISION = "809d05bd12f261d26b42e28dc2b31db430c1585c"
DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION = "5f32929a4206c500f4044bf7778298aedc77531a"
DEFAULT_MAX_LENGTH = 1000
DEFAULT_MAX_CANDIDATE_SUBWORDS = 22
MAX_CANDIDATES = 100
NME_SYMBOL = "--NME--"


@dataclass(frozen=True)
class WSLReaderInput:
    """A single predefined target packed for the frozen WSL reader."""

    input_ids: Any
    attention_mask: Any
    token_type_ids: Any
    target_start_position: int
    target_end_position: int
    no_meaning_position: int
    candidate_symbol_positions: tuple[int, ...]


@dataclass(frozen=True)
class WSLReaderScores:
    """Raw reader outputs for explicit candidates."""

    candidate_logits: tuple[float, ...]
    candidate_probabilities: tuple[float, ...]
    no_entity_logit: float
    no_entity_probability: float
    no_meaning_logit: float
    no_meaning_probability: float

    @property
    def abstention_probability(self) -> float:
        """Probability assigned to either NONE or the explicit NME class."""

        return self.no_entity_probability + self.no_meaning_probability


class LocalWSLReaderRuntime:
    """Load and run the WSL reader checkpoint without the upstream WSL package."""

    def __init__(
        self,
        model_name: str = DEFAULT_WSL_READER_MODEL,
        *,
        revision: str | None = DEFAULT_WSL_READER_REVISION,
        device: str = "cpu",
        precision: str | int = 32,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_candidate_subwords: int = DEFAULT_MAX_CANDIDATE_SUBWORDS,
    ) -> None:
        torch, transformers, hf_hub_download, load_safetensors = _runtime_imports()
        self._torch = torch
        self.model_name = model_name
        self.revision = revision
        self.device = str(device)
        self.precision = str(precision)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.max_length = int(max_length)
        self.max_candidate_subwords = int(max_candidate_subwords)

        common_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            common_kwargs["revision"] = self.revision

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            add_prefix_space=True,
            **common_kwargs,
        )
        config_path, weights_path = _resolve_checkpoint_files(
            self.model_name,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            hf_hub_download=hf_hub_download,
        )
        with config_path.open("r", encoding="utf-8") as handle:
            reader_config = json.load(handle)

        base_model_name = str(reader_config["transformer_model"])
        base_config_kwargs: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if base_model_name == "microsoft/deberta-v3-base":
            base_config_kwargs["revision"] = DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION
        base_config = transformers.AutoConfig.from_pretrained(
            base_model_name,
            **base_config_kwargs,
        )
        self.model = _build_disambiguation_model(
            torch,
            transformers,
            base_config=base_config,
            additional_special_symbols=int(reader_config["additional_special_symbols"]),
            linears_hidden_size=int(reader_config["linears_hidden_size"]),
            activation=str(reader_config["activation"]),
            use_last_k_layers=int(reader_config["use_last_k_layers"]),
        )
        state_dict = load_safetensors(str(weights_path), device="cpu")
        incompatibility = self.model.load_state_dict(state_dict, strict=False)
        _validate_checkpoint_load(incompatibility)

        self._dtype = _resolve_dtype(torch, self.precision, self.device)
        self.model.eval()
        self.model.to(device=self.device, dtype=self._dtype)

    def score_candidates(
        self,
        *,
        tokens: Sequence[str],
        target_start: int,
        target_end: int,
        candidates: Sequence[str],
    ) -> WSLReaderScores:
        packed = build_wsl_reader_input(
            self.tokenizer,
            tokens=tokens,
            target_start=target_start,
            target_end=target_end,
            candidates=candidates,
            max_length=self.max_length,
            max_candidate_subwords=self.max_candidate_subwords,
        )
        torch = self._torch
        model_inputs = {
            "input_ids": packed.input_ids.to(self.device),
            "attention_mask": packed.attention_mask.to(self.device),
            "token_type_ids": packed.token_type_ids.to(self.device),
        }
        with torch.inference_mode():
            (
                candidate_logits,
                candidate_probabilities,
                no_entity_logit,
                no_entity_probability,
                no_meaning_logit,
                no_meaning_probability,
            ) = self.model.score_predefined_target(
                **model_inputs,
                target_start_position=packed.target_start_position,
                target_end_position=packed.target_end_position,
                no_meaning_position=packed.no_meaning_position,
                candidate_symbol_positions=packed.candidate_symbol_positions,
            )
        return WSLReaderScores(
            candidate_logits=tuple(
                float(value) for value in candidate_logits.cpu().tolist()
            ),
            candidate_probabilities=tuple(
                float(value) for value in candidate_probabilities.cpu().tolist()
            ),
            no_entity_logit=float(no_entity_logit.cpu().item()),
            no_entity_probability=float(no_entity_probability.cpu().item()),
            no_meaning_logit=float(no_meaning_logit.cpu().item()),
            no_meaning_probability=float(no_meaning_probability.cpu().item()),
        )


def build_wsl_reader_input(
    tokenizer: Any,
    *,
    tokens: Sequence[str],
    target_start: int,
    target_end: int,
    candidates: Sequence[str],
    max_length: int = DEFAULT_MAX_LENGTH,
    max_candidate_subwords: int = DEFAULT_MAX_CANDIDATE_SUBWORDS,
) -> WSLReaderInput:
    """Reproduce the WSL reader's sentence/candidate packing for one target."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without ML runtime
        raise RuntimeError("The local WSL reader requires PyTorch.") from exc

    tokens = tuple(str(token) for token in tokens)
    candidates = tuple(str(candidate) for candidate in candidates)
    if not tokens:
        raise ValueError("WSL reader input requires at least one sentence token.")
    if not (0 <= target_start < target_end <= len(tokens)):
        raise ValueError(
            "Invalid target token interval "
            f"[{target_start}, {target_end}) for {len(tokens)} tokens."
        )
    if not candidates:
        raise ValueError("WSL reader input requires at least one candidate.")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"The released WSL reader has {MAX_CANDIDATES} candidate symbols; "
            f"received {len(candidates)} candidates."
        )

    sentence_encoding = tokenizer(
        list(tokens),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_piece_ids = sentence_encoding["input_ids"]
    if token_piece_ids and isinstance(token_piece_ids[0], int):
        raise RuntimeError(
            "The WSL tokenizer treated the sentence-token list as one sequence. "
            "A fast tokenizer that batch-encodes each supplied token is required."
        )
    token_piece_ids = [list(piece_ids) for piece_ids in token_piece_ids]
    empty_tokens = [
        index for index, piece_ids in enumerate(token_piece_ids) if not piece_ids
    ]
    if empty_tokens:
        # This is an input-specific encoding limitation, not evidence that the
        # checkpoint/runtime is broken.  score_wsd_targets treats ValueError as
        # an unresolved target while still failing fast on systemic model errors.
        raise ValueError(
            f"WSL tokenizer produced no subwords for token indices {empty_tokens[:5]}."
        )

    sentence_ids = [piece for piece_ids in token_piece_ids for piece in piece_ids]
    target_start_position = 1 + sum(
        len(piece_ids) for piece_ids in token_piece_ids[:target_start]
    )
    target_end_position = (
        1 + sum(len(piece_ids) for piece_ids in token_piece_ids[:target_end]) - 1
    )

    # The released reader was trained with two abstention classes before the
    # explicit senses: class 0 is represented by CLS (NONE), and class 1 by
    # the standalone --NME-- token. Actual candidates begin at class 2 and use
    # [E-0], [E-1], ... prefix symbols.
    nme_id = tokenizer.convert_tokens_to_ids(NME_SYMBOL)
    if nme_id is None or nme_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"The WSL checkpoint tokenizer does not contain required symbol {NME_SYMBOL!r}."
        )
    nme_ids = list(
        tokenizer(
            NME_SYMBOL,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )
    if nme_ids != [nme_id]:
        raise RuntimeError(
            f"WSL tokenizer must encode {NME_SYMBOL!r} as one special token; got {nme_ids}."
        )

    candidate_encodings: list[list[int]] = []
    candidate_symbol_positions: list[int] = []
    no_meaning_position = 1 + len(sentence_ids) + 1
    next_position = no_meaning_position + len(nme_ids)
    for index, candidate in enumerate(candidates):
        symbol = f"[E-{index}]"
        symbol_id = tokenizer.convert_tokens_to_ids(symbol)
        if symbol_id is None or symbol_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"The WSL checkpoint tokenizer does not contain required symbol {symbol!r}."
            )
        candidate_ids = list(
            tokenizer(
                f"{symbol} {candidate}",
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )
        if max_candidate_subwords > 0:
            candidate_ids = candidate_ids[:max_candidate_subwords]
        if not candidate_ids or candidate_ids[0] != symbol_id:
            raise RuntimeError(
                f"WSL candidate encoding did not begin with its required symbol {symbol!r}."
            )
        candidate_symbol_positions.append(next_position)
        candidate_encodings.append(candidate_ids)
        next_position += len(candidate_ids)

    flat_candidate_ids = nme_ids + [
        piece for ids in candidate_encodings for piece in ids
    ]
    input_ids = (
        [tokenizer.cls_token_id]
        + sentence_ids
        + [tokenizer.sep_token_id]
        + flat_candidate_ids
        + [tokenizer.sep_token_id]
    )
    effective_max_length = min(
        int(max_length),
        int(getattr(tokenizer, "model_max_length", max_length)),
    )
    if len(input_ids) > effective_max_length:
        raise ValueError(
            "The target and all explicit candidate glosses do not fit in one WSL reader input: "
            f"{len(input_ids)} tokens exceed the configured maximum of {effective_max_length}. "
            "This minimal experiment does not split a target's candidate set because probabilities "
            "from separate joint-reader calls are not directly comparable."
        )

    sentence_segment_length = len(sentence_ids) + 2
    token_type_ids = [0] * sentence_segment_length + [1] * (
        len(input_ids) - sentence_segment_length
    )
    return WSLReaderInput(
        input_ids=torch.tensor([input_ids], dtype=torch.long),
        attention_mask=torch.ones((1, len(input_ids)), dtype=torch.long),
        token_type_ids=torch.tensor([token_type_ids], dtype=torch.long),
        target_start_position=target_start_position,
        target_end_position=target_end_position,
        no_meaning_position=no_meaning_position,
        candidate_symbol_positions=tuple(candidate_symbol_positions),
    )


def _build_disambiguation_model(
    torch: Any,
    transformers: Any,
    *,
    base_config: Any,
    additional_special_symbols: int,
    linears_hidden_size: int,
    activation: str,
    use_last_k_layers: int,
) -> Any:
    from transformers.activations import ClippedGELUActivation, GELUActivation

    class WSLDisambiguationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_last_k_layers = use_last_k_layers
            self.transformer_model = transformers.AutoModel.from_config(base_config)
            self.transformer_model.resize_token_embeddings(
                self.transformer_model.config.vocab_size + additional_special_symbols
            )
            self.ed_start_projector = self._projection()
            self.ed_end_projector = self._projection()

        def _projection(self) -> Any:
            if activation == "gelu":
                activation_layer = GELUActivation()
            elif activation == "relu":
                activation_layer = torch.nn.ReLU()
            elif activation == "gelu_10":
                activation_layer = ClippedGELUActivation(-10, 10)
            else:
                raise ValueError(f"Unsupported WSL reader activation: {activation!r}")
            return torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(
                    self.transformer_model.config.hidden_size * self.use_last_k_layers,
                    linears_hidden_size,
                ),
                activation_layer,
                torch.nn.Dropout(0.1),
                torch.nn.Linear(linears_hidden_size, linears_hidden_size),
                torch.nn.LayerNorm(
                    linears_hidden_size,
                    self.transformer_model.config.layer_norm_eps,
                ),
            )

        def _features(
            self, input_ids: Any, attention_mask: Any, token_type_ids: Any
        ) -> Any:
            output = self.transformer_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=self.use_last_k_layers > 1,
            )
            if self.use_last_k_layers > 1:
                return torch.cat(
                    output.hidden_states[-self.use_last_k_layers :], dim=-1
                )
            return output.last_hidden_state

        def score_predefined_target(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
            target_start_position: int,
            target_end_position: int,
            no_meaning_position: int,
            candidate_symbol_positions: Sequence[int],
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            features = self._features(input_ids, attention_mask, token_type_ids)
            target_start = features[
                :, target_start_position : target_start_position + 1, :
            ]
            target_end = features[:, target_end_position : target_end_position + 1, :]
            # Match the released reader's class layout exactly: CLS is NONE,
            # --NME-- is the explicit no-meaning class, and candidates begin at class 2.
            class_positions = (0, no_meaning_position, *candidate_symbol_positions)
            class_features = features[:, class_positions, :]

            target_representation = torch.cat(
                [
                    self.ed_start_projector(target_start),
                    self.ed_end_projector(target_end),
                ],
                dim=-1,
            )
            class_representation = torch.cat(
                [
                    self.ed_start_projector(class_features),
                    self.ed_end_projector(class_features),
                ],
                dim=-1,
            )
            logits = torch.bmm(
                target_representation,
                class_representation.transpose(1, 2),
            )[0, 0]
            probabilities = torch.softmax(logits, dim=-1)
            return (
                logits[2:],
                probabilities[2:],
                logits[0],
                probabilities[0],
                logits[1],
                probabilities[1],
            )

    return WSLDisambiguationModel()


def _runtime_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional live runtime
        missing = exc.name or "unknown"
        raise RuntimeError(
            f"The local WSL reader requires the TeAL NLP runtime; `{missing}` is missing. "
            "Run `uv sync --extra nlp`, then restart Python or the Jupyter kernel."
        ) from exc
    return torch, transformers, hf_hub_download, load_file


def _resolve_checkpoint_files(
    model_name: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    hf_hub_download: Any,
) -> tuple[Path, Path]:
    local_path = Path(model_name)
    if local_path.exists():
        config_path = local_path / "config.json"
        weights_path = local_path / "model.safetensors"
    else:
        kwargs: dict[str, Any] = {
            "repo_id": model_name,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if revision is not None:
            kwargs["revision"] = revision
        config_path = Path(hf_hub_download(filename="config.json", **kwargs))
        weights_path = Path(hf_hub_download(filename="model.safetensors", **kwargs))
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"WSL reader checkpoint is incomplete at {model_name!r}; config.json and "
            "model.safetensors are both required."
        )
    return config_path, weights_path


def _validate_checkpoint_load(incompatibility: Any) -> None:
    required_prefixes = (
        "transformer_model.",
        "ed_start_projector.",
        "ed_end_projector.",
    )
    missing_required = [
        key for key in incompatibility.missing_keys if key.startswith(required_prefixes)
    ]
    if missing_required:
        raise RuntimeError(
            "The WSL checkpoint is missing parameters required by the local reader: "
            + ", ".join(missing_required[:10])
        )
    unexpected = [
        key
        for key in incompatibility.unexpected_keys
        if not key.startswith(("ned_start_classifier.", "ned_end_classifier."))
    ]
    if unexpected:
        raise RuntimeError(
            "The WSL checkpoint contains unrecognized parameters outside the intentionally "
            "discarded span detector: " + ", ".join(unexpected[:10])
        )


def _resolve_dtype(torch: Any, precision: str, device: str) -> Any:
    normalized = str(precision).lower().replace("-mixed", "")
    if normalized in {"16", "fp16", "float16"}:
        if not device.startswith("cuda"):
            raise ValueError("WSL float16 inference is supported only on CUDA devices.")
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"32", "fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported WSL inference precision: {precision!r}")
