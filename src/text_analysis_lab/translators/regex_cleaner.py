"""Vectorized regex-cleaning translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
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

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_FLAG_MAP: dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "VERBOSE": re.VERBOSE,
    "X": re.VERBOSE,
    "ASCII": re.ASCII,
    "A": re.ASCII,
}


@dataclass(frozen=True)
class RegexReplaceRule:
    """One ordered regular-expression replacement rule."""

    pattern: str
    replacement: str = ""
    flags: tuple[str, ...] = ()
    count: int = 0
    description: str | None = None


RuleInput = RegexReplaceRule | Mapping[str, Any]


class RegexCleaner(BaseTranslator):
    """Apply ordered regex replacements to a text column while preserving rows.

    Each replacement is applied to the full pandas Series with ``Series.str``;
    there is no Python loop over observations.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        text_field: str = "text",
        output_field: str | None = None,
        rules: Sequence[RuleInput] = (),
        strip: bool = True,
        preserve_null: bool = True,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        output = text_field if output_field is None else output_field
        if not isinstance(output, str) or not output:
            raise ValueError("output_field must be a non-empty string.")

        self.text_field = text_field
        self.output_field = output
        self.rules = tuple(_normalize_rule(rule) for rule in rules)
        self.strip = bool(strip)
        self.preserve_null = bool(preserve_null)

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = _single_source(sources)
        return OutputSpec(
            artifact_type="table",
            lineage_mode="preserved_key",
            basis_labels=DEFAULT_SOURCE_LABEL,
        )

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return mode == "translate" and route in {"sequential", "parallel"}

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        _ = sources, mode
        if params:
            raise OperatorError(
                f"RegexCleaner does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("RegexCleaner requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
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
        _ = mode, request
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        key_columns = [str(name) for name in packet.primary_key]
        missing = [
            name
            for name in [*key_columns, self.text_field]
            if name not in frame.columns
        ]
        if missing:
            raise ArtifactError(
                f"RegexCleaner source batch is missing columns {missing}."
            )

        source = frame[self.text_field]
        null_mask = source.isna()
        cleaned = source.fillna("").astype("string")
        for rule in self.rules:
            cleaned = cleaned.str.replace(
                rule.pattern,
                rule.replacement,
                n=-1 if rule.count == 0 else int(rule.count),
                flags=_compile_flags(rule.flags),
                regex=True,
            )
        if self.strip:
            cleaned = cleaned.str.strip()
        if self.preserve_null:
            cleaned = cleaned.mask(null_mask, pd.NA)

        keys = frame.loc[:, key_columns].reset_index(drop=True)
        data = pd.DataFrame({self.output_field: cleaned.reset_index(drop=True)})
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: {"keys": keys, "data": data}})

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

    def make_translate_worker(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> RegexCleaner:
        _ = mode, request
        return self.from_json_state(self.to_json_state())

    def to_json_state(self) -> dict[str, Any]:
        return {
            "text_field": self.text_field,
            "output_field": self.output_field,
            "rules": [asdict(rule) for rule in self.rules],
            "strip": self.strip,
            "preserve_null": self.preserve_null,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> RegexCleaner:
        return cls(
            text_field=str(state.get("text_field", "text")),
            output_field=str(
                state.get("output_field", state.get("text_field", "text"))
            ),
            rules=cast(Sequence[Mapping[str, Any]], state.get("rules", ())),
            strip=bool(state.get("strip", True)),
            preserve_null=bool(state.get("preserve_null", True)),
        )

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
        payload = self.to_json_state()
        payload["operator_id"] = operator_id
        (intermediate_dir / "state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> RegexCleaner:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        return obj


def _normalize_flags(flags: Any) -> tuple[str, ...]:
    if flags is None:
        return ()
    if isinstance(flags, int):
        names = [
            name
            for name, value in [
                ("IGNORECASE", re.IGNORECASE),
                ("MULTILINE", re.MULTILINE),
                ("DOTALL", re.DOTALL),
                ("VERBOSE", re.VERBOSE),
                ("ASCII", re.ASCII),
            ]
            if flags & value
        ]
        return tuple(names)
    if isinstance(flags, str):
        return (flags,)
    return tuple(str(flag) for flag in cast(Sequence[Any], flags))


def _normalize_rule(rule: RuleInput) -> RegexReplaceRule:
    if isinstance(rule, RegexReplaceRule):
        normalized = rule
    elif isinstance(rule, Mapping):
        if "pattern" not in rule:
            raise ValueError("Each RegexCleaner rule requires a 'pattern'.")
        normalized = RegexReplaceRule(
            pattern=str(rule["pattern"]),
            replacement=str(rule.get("replacement", "")),
            flags=_normalize_flags(rule.get("flags", ())),
            count=int(rule.get("count", 0)),
            description=None
            if rule.get("description") is None
            else str(rule["description"]),
        )
    else:
        raise TypeError(
            "RegexCleaner rules must be RegexReplaceRule objects or mappings."
        )

    normalized = RegexReplaceRule(
        pattern=str(normalized.pattern),
        replacement=str(normalized.replacement),
        flags=_normalize_flags(normalized.flags),
        count=int(normalized.count),
        description=normalized.description,
    )
    if normalized.count < 0:
        raise ValueError("RegexCleaner rule count must be non-negative.")
    re.compile(normalized.pattern, _compile_flags(normalized.flags))
    return normalized


def _compile_flags(flags: Sequence[str]) -> int:
    compiled = 0
    for flag in flags:
        key = str(flag).upper()
        if key not in _FLAG_MAP:
            supported = sorted(name for name in _FLAG_MAP if len(name) > 1)
            raise ValueError(
                f"Unsupported regex flag {flag!r}. Supported flags: {supported}."
            )
        compiled |= _FLAG_MAP[key]
    return compiled


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"RegexCleaner requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"RegexCleaner expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"RegexCleaner expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
