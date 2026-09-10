"""Shared Hugging Face runtime helpers for TeAL transformer translators."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import InputBatch

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class TransformerResourceError(OperatorError):
    """Raised when a Hugging Face/PyTorch model resource cannot be loaded."""


class ContextWindowExceededError(ArtifactError):
    """Raised when text exceeds a model context limit without opt-in truncation."""


def resolve_device(value: str) -> str:
    requested = str(value).lower()
    try:
        import torch
    except ImportError as exc:
        raise TransformerResourceError(
            "Transformer inference requires PyTorch. Install TeAL's 'transformers' extra."
        ) from exc
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise TransformerResourceError("device='cuda' was requested but CUDA is unavailable.")
    if requested == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise TransformerResourceError("device='mps' was requested but MPS is unavailable.")
    if requested not in {"cpu", "cuda", "mps"} and not re.fullmatch(r"cuda:\d+", requested):
        raise OperatorError(
            "device must be 'auto', 'cpu', 'cuda', 'mps', or an explicit CUDA device such as 'cuda:1'."
        )
    return requested


def count_tokens(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        verbose=False,
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TransformerResourceError("Tokenizer output did not include input_ids.")
    return [len(row) for row in input_ids]


def detect_context_limit(tokenizer: Any, config: Any = None, explicit: int | None = None) -> int:
    candidates: list[int] = []
    _append_finite_limit(candidates, getattr(tokenizer, "model_max_length", None))
    if config is not None:
        for name in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "max_seq_len",
            "seq_length",
        ):
            _append_finite_limit(candidates, getattr(config, name, None))
    detected = min(candidates) if candidates else None
    if explicit is not None:
        explicit = int(explicit)
        if explicit <= 0:
            raise OperatorError("max_length must be positive or None.")
        if detected is not None and explicit > detected:
            raise OperatorError(
                f"max_length={explicit} exceeds the detected model/tokenizer context limit "
                f"of {detected}. TeAL will not override a known model limit."
            )
        return explicit
    if detected is None:
        raise OperatorError(
            "TeAL could not determine a finite context-window limit from this model/tokenizer. "
            "Supply max_length explicitly so context handling cannot fail silently."
        )
    return int(detected)


def _append_finite_limit(candidates: list[int], value: Any) -> None:
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return
    # Hugging Face tokenizers use huge sentinel integers to mean "unknown".
    if 0 < limit < 1_000_000_000:
        candidates.append(limit)


def resolved_commit_hash(model: Any, tokenizer: Any = None) -> str | None:
    candidates: list[Any] = []
    if model is not None:
        candidates.extend(
            [
                getattr(getattr(model, "config", None), "_commit_hash", None),
                getattr(model, "_commit_hash", None),
            ]
        )
    if tokenizer is not None:
        candidates.append(getattr(tokenizer, "_commit_hash", None))
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            candidates.append(init_kwargs.get("_commit_hash"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def warn_if_unpinned(
    *,
    model_name: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    warned: bool,
    kind: str,
) -> bool:
    if requested_revision is not None or warned:
        return warned
    if resolved_revision:
        warnings.warn(
            f"{kind} {model_name!r} was requested without an immutable revision. "
            f"This operator resolved and will record Hub commit {resolved_revision!r} for reuse.",
            UserWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"{kind} {model_name!r} was requested without an immutable revision and TeAL "
            "could not recover a resolved Hub commit hash. For strict reproducibility, supply "
            "revision=<commit_sha> or use save_model=True.",
            UserWarning,
            stacklevel=3,
        )
    return True


def require_frame(value: Any, *, translator_name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"{translator_name} expected a pandas DataFrame; got {type(value).__name__}."
        )
    return value


def single_source(
    sources: Mapping[str, "BaseArtifact"], *, translator_name: str
) -> "BaseArtifact":
    if len(sources) != 1:
        raise OperatorError(
            f"{translator_name} requires exactly one source; got {list(sources)}."
        )
    return next(iter(sources.values()))


def single_input(inputs: Mapping[str, InputBatch], *, translator_name: str) -> InputBatch:
    if len(inputs) != 1:
        raise OperatorError(
            f"{translator_name} requires exactly one input packet; got {list(inputs)}."
        )
    return next(iter(inputs.values()))


def hidden_size(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "hidden_size", None)
    if value is None:
        raise TransformerResourceError("Could not determine transformer hidden size.")
    return int(value)
