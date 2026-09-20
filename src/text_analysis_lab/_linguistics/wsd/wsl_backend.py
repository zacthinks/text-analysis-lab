"""Frozen WSL reader adapter for explicit TeAL targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from text_analysis_lab._linguistics.device import resolve_devices
from text_analysis_lab._linguistics.wsd.types import WSDTarget
from text_analysis_lab._linguistics.wsd.wsl_reader import (
    DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
    DEFAULT_WSL_READER_MODEL,
    DEFAULT_WSL_READER_REVISION,
    LocalWSLReaderRuntime,
    WSLReaderScores,
)


class WSLScoringBackend(Protocol):
    def descriptor(self) -> Mapping[str, Any]: ...

    def score_target(
        self, target: WSDTarget, candidate_texts: Sequence[str]
    ) -> Mapping[str, float]: ...


class BabelscapeWSLBackend:
    """Run the released WSL reader locally for one explicit target at a time.

    TeAL loads the pretrained reader checkpoint directly through Transformers and
    Safetensors. Babelscape's unmanaged ``wsl`` Python package is not imported or installed.
    Span detection and retrieval are omitted; only the checkpoint's frozen disambiguation path
    is used.
    """

    LICENSE = "CC-BY-NC-SA-4.0"

    def __init__(
        self,
        model_name: str = DEFAULT_WSL_READER_MODEL,
        *,
        revision: str | None = DEFAULT_WSL_READER_REVISION,
        device: str = "auto",
        precision: str | int = 32,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        acknowledge_noncommercial_license: bool = False,
        model: Any | None = None,
    ) -> None:
        if not acknowledge_noncommercial_license:
            raise ValueError(
                "WSL is licensed CC BY-NC-SA 4.0. Pass "
                "acknowledge_noncommercial_license=True only for an eligible non-commercial "
                "research or education use."
            )
        self.model_name = model_name
        self.revision = revision
        self.device = resolve_devices(device).primary
        self.precision = str(precision)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        # Loading the 747 MB reader is intentionally lazy so a current persistent
        # WSD stage can be reused without touching the model runtime or network.
        self._model = model

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "type": "text-analysis-lab-local-wsl-reader",
            "model": self.model_name,
            "revision": self.revision,
            "base_config_revision": DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
            "device": self.device,
            "precision": self.precision,
            "license": self.LICENSE,
            "inference_policy": "one_target_per_call_explicit_candidates",
            "span_detection": False,
            "retrieval": False,
            "training_runtime": False,
            "upstream_python_package": False,
            "score_type": "reader_candidate_probability_joint_with_none_and_nme",
        }

    def score_target(
        self, target: WSDTarget, candidate_texts: Sequence[str]
    ) -> Mapping[str, float]:
        unique_candidates = tuple(dict.fromkeys(candidate_texts))
        if not unique_candidates:
            return {}
        if self._model is None:
            self._model = self._load_model()
        result = self._model.score_candidates(
            tokens=target.tokens,
            target_start=target.target_start,
            target_end=target.target_end,
            candidates=unique_candidates,
        )
        probabilities = _candidate_probabilities(result)
        if len(probabilities) != len(unique_candidates):
            raise RuntimeError(
                "The local WSL reader returned a different number of candidate probabilities "
                f"({len(probabilities)}) than supplied candidates ({len(unique_candidates)})."
            )
        return dict(zip(unique_candidates, probabilities, strict=True))

    def _load_model(self) -> LocalWSLReaderRuntime:
        return LocalWSLReaderRuntime(
            self.model_name,
            revision=self.revision,
            device=self.device,
            precision=self.precision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )


def _candidate_probabilities(result: Any) -> tuple[float, ...]:
    if isinstance(result, WSLReaderScores):
        return result.candidate_probabilities
    values = getattr(result, "candidate_probabilities", None)
    if values is None and isinstance(result, Mapping):
        values = result.get("candidate_probabilities")
    if values is None:
        raise RuntimeError(
            "Injected WSL reader runtime must return `candidate_probabilities`."
        )
    return tuple(float(value) for value in values)


def target_as_text_and_char_span(target: WSDTarget) -> tuple[str, tuple[int, int]]:
    """Render stored sentence tokens and recover the target's character span."""

    text = " ".join(target.tokens)
    start = (
        sum(len(token) for token in target.tokens[: target.target_start])
        + target.target_start
    )
    target_text = " ".join(target.tokens[target.target_start : target.target_end])
    return text, (start, start + len(target_text))
