"""Vectorized regex-cleaning translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.query_scope import QueryScope
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
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL, MetadataMode

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
    frozen_runtime_fields = frozenset({"_field_namespaces", "_input_labels"})

    def __init__(
        self,
        *,
        text_field: str | None = None,
        output_field: str | None = None,
        fields: Mapping[str, str] | None = None,
        metadata_mode: MetadataMode = "none",
        rules: Sequence[RuleInput] = (),
        strip: bool = True,
        preserve_null: bool = True,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if metadata_mode not in {"none", "local", "full"}:
            raise ValueError("metadata_mode must be 'none', 'local', or 'full'.")
        if fields is not None and (text_field is not None or output_field is not None):
            raise ValueError("fields cannot be combined with text_field or output_field.")
        legacy = fields is None
        if legacy:
            if text_field is not None or output_field is not None:
                warnings.warn(
                    "RegexCleaner text_field/output_field are deprecated; use fields. "
                    "They will be removed in 1.0.0.", FutureWarning, stacklevel=2,
                )
            selected = "text" if text_field is None else text_field
            fields = {selected: selected if output_field is None else output_field}
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields must be a nonempty input-to-output mapping.")
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in fields.items()):
            raise ValueError("fields must contain nonempty string input and output labels.")
        self.fields = dict(fields)
        self.metadata_mode = metadata_mode
        self._field_namespaces: dict[str, str] = {}
        self._input_labels: dict[str, str] = {}
        # Retain legacy attributes for existing callers, without treating them
        # as a second source of truth.
        self.text_field = next(iter(self.fields))
        self.output_field = self.fields[self.text_field]
        self.rules = tuple(_normalize_rule(rule) for rule in rules)
        self.strip = bool(strip)
        self.preserve_null = bool(preserve_null)

    def translate(
        self,
        texts: str | None | Sequence[str | None] | pd.Series | pd.DataFrame | Mapping[str, pd.DataFrame],
    ) -> str | None | pd.Series | dict[str, pd.DataFrame]:
        """Clean ordinary text values without creating TeAL artifacts.

        Scalar input returns a scalar. A pandas Series returns a Series with its
        index preserved; other sequences return a new Series.
        """
        if isinstance(texts, (pd.DataFrame, Mapping)):
            if isinstance(texts, pd.DataFrame):
                data, metadata = texts, None
            else:
                data, metadata = texts.get("data"), texts.get("metadata")
                if data is not None and not isinstance(data, pd.DataFrame):
                    raise TypeError("RegexCleaner preview data must be a DataFrame.")
                if metadata is not None and not isinstance(metadata, pd.DataFrame):
                    raise TypeError("RegexCleaner preview metadata must be a DataFrame.")
                if data is None and metadata is None:
                    raise ValueError("RegexCleaner preview packet requires data or metadata.")
                if data is not None and metadata is not None and len(data) != len(metadata):
                    raise ValueError("RegexCleaner preview components must be row-aligned.")
            if data is not None and not data.columns.is_unique:
                raise ValueError("RegexCleaner preview data column names must be unique.")
            if metadata is not None and not metadata.columns.is_unique:
                raise ValueError("RegexCleaner preview metadata column names must be unique.")
            output: dict[str, pd.DataFrame] = {}
            for label, destination in self.fields.items():
                in_data = data is not None and label in data.columns
                in_metadata = metadata is not None and label in metadata.columns
                if in_data == in_metadata:
                    raise ValueError(
                        f"RegexCleaner preview field {label!r} must occur in exactly "
                        "one of the data or metadata components."
                    )
                namespace = "data" if in_data else "metadata"
                frame = data if in_data else metadata
                assert frame is not None
                if namespace not in output:
                    output[namespace] = pd.DataFrame(index=frame.index)
                if destination in output[namespace]:
                    raise ValueError(
                        f"RegexCleaner duplicate {namespace} output {destination!r}."
                    )
                output[namespace][destination] = cast(pd.Series, self.translate(frame[label]))
            return output

        scalar = isinstance(texts, str) or texts is None
        if isinstance(texts, pd.Series):
            source = texts.copy()
        elif scalar:
            source = pd.Series([texts])
        elif isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            source = pd.Series(list(texts))
        else:
            raise TypeError(
                "RegexCleaner.translate(...) expects a string, None, a sequence "
                "of text values, or a pandas Series."
            )

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

        if scalar:
            value = cleaned.iloc[0]
            return None if pd.isna(value) else str(value)
        return cleaned

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        _single_source(sources)
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
        _ = mode
        if params:
            raise OperatorError(
                f"RegexCleaner does not accept operation parameters; got {sorted(params)}."
            )
        source = _single_source(sources)
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
        scope = QueryScope(source, metadata_mode=self.metadata_mode)
        namespaces: dict[str, str] = {}
        input_labels: dict[str, str] = {}
        if callable(getattr(source, "query_columns", None)):
            choices = scope.columns()["columns"]
            for field in self.fields:
                matches = [
                    col for col in choices if col["namespace"] in {"data", "metadata"}
                    and col["output_name"] == field
                ]
                if not matches:
                    # Legacy callers may still supply an unqualified data
                    # name; QueryScope resolves it without parsing prefixes.
                    try:
                        name = scope.data_output_name(field)
                    except ValueError as exc:
                        raise OperatorError(
                            f"RegexCleaner field {field!r} is unavailable in "
                            f"metadata_mode={self.metadata_mode!r}."
                        ) from exc
                    matches = [
                        col for col in choices if col["namespace"] == "data"
                        and col["output_name"] == name
                    ]
                if len(matches) != 1:
                    raise OperatorError(
                        f"RegexCleaner field {field!r} is unavailable or ambiguous."
                    )
                namespaces[field] = str(matches[0]["namespace"])
                input_labels[field] = str(matches[0]["output_name"])
        elif self.metadata_mode == "none":
            namespaces = {field: "data" for field in self.fields}
            input_labels = {field: field for field in self.fields}
        else:
            raise OperatorError("RegexCleaner requires query_columns for metadata sources.")
        for namespace in {"data", "metadata"}:
            outputs = [self.fields[k] for k, v in namespaces.items() if v == namespace]
            if len(set(outputs)) != len(outputs):
                raise OperatorError(f"RegexCleaner duplicate {namespace} output destinations: {outputs}.")
        if any(v in source.primary_key for v in self.fields.values()):
            raise OperatorError("RegexCleaner outputs cannot collide with primary keys.")
        self._field_namespaces = namespaces
        self._input_labels = input_labels
        return scope.source_request(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(
                keys=True,
                data=[input_labels[name] for name, domain in namespaces.items() if domain == "data"] or False,
                metadata=[input_labels[name] for name, domain in namespaces.items() if domain == "metadata"] or False,
            ),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
            form="table",
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
        key_columns = list(packet.primary_key)
        missing_keys = [name for name in key_columns if name not in frame.columns]
        if missing_keys:
            raise ArtifactError(f"RegexCleaner source batch lacks keys {missing_keys}.")
        keys = frame.loc[:, key_columns].reset_index(drop=True)
        selected_labels = {
            name: self._input_labels.get(
                name, f"data.{name}" if name in key_columns else name
            )
            for name in self.fields
        }
        missing = [label for label in selected_labels.values() if label not in frame.columns]
        if missing:
            raise ArtifactError(f"RegexCleaner source batch is missing columns {missing}.")
        namespaces = self._field_namespaces
        if not namespaces:
            if len(self.fields) != 1 or self.metadata_mode != "none":
                raise OperatorError("RegexCleaner must resolve source namespaces before translating batches.")
            namespaces = {self.text_field: "data"}
        payload: dict[str, pd.DataFrame] = {
            "keys": keys,
        }
        for domain in ("data", "metadata"):
            output = {
                self.fields[name]: cast(pd.Series, self.translate(frame[selected_labels[name]])).reset_index(drop=True)
                for name, source_domain in namespaces.items() if source_domain == domain
            }
            if output:
                payload[domain] = pd.DataFrame(output)
        return BatchResult(outputs={DEFAULT_OUTPUT_LABEL: payload})

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
        worker = self.from_json_state(self.to_json_state())
        worker._field_namespaces = dict(self._field_namespaces)
        worker._input_labels = dict(self._input_labels)
        return worker

    def to_json_state(self) -> dict[str, Any]:
        return {
            "fields": dict(self.fields),
            "metadata_mode": self.metadata_mode,
            "rules": [asdict(rule) for rule in self.rules],
            "strip": self.strip,
            "preserve_null": self.preserve_null,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> RegexCleaner:
        if "fields" in state:
            return cls(
                fields=cast(Mapping[str, str], state["fields"]),
                metadata_mode=cast(MetadataMode, state.get("metadata_mode", "none")),
                rules=cast(Sequence[Mapping[str, Any]], state.get("rules", ())),
                strip=bool(state.get("strip", True)),
                preserve_null=bool(state.get("preserve_null", True)),
            )
        field = str(state.get("text_field", "text"))
        return cls(
            fields={field: str(state.get("output_field", field))},
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
        payload["field_namespaces"] = self._field_namespaces
        payload["input_labels"] = self._input_labels
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
        obj._field_namespaces = dict(state.get("field_namespaces", {}))
        obj._input_labels = dict(state.get("input_labels", {}))
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
