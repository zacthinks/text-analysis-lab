"""spaCy linguistic decomposition translator for Text Analysis Lab (TeAL).

One batched spaCy pipeline pass over source documents produces two normalized
TeAL table artifacts:

``sentences``
    The source key extended by ``sentence_id``.

``tokens``
    The sentence key extended by ``token_id``. Dependency heads are normalized
    from spaCy's document-global token indexes to sentence-local ``token_id``
    values.

The expensive NLP work is performed with ``Language.pipe``. Flattening the
resulting ``Doc``/``Span``/``Token`` object hierarchy into relational rows still
requires lightweight Python iteration over the already-processed objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
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
from text_analysis_lab.core.types import DEFAULT_SOURCE_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


SENTENCES_LABEL = "sentences"
TOKENS_LABEL = "tokens"

SENTENCE_DATA_COLUMNS: tuple[str, ...] = (
    "text",
    "char_start",
    "char_end",
)

TOKEN_DATA_COLUMNS: tuple[str, ...] = (
    "text",
    "lemma",
    "norm",
    "shape",
    "pos",
    "tag",
    "morph",
    "dep",
    "head_token_id",
    "ent_iob",
    "ent_type",
    "char_start",
    "char_end",
    "is_alpha",
    "is_ascii",
    "is_digit",
    "is_lower",
    "is_upper",
    "is_title",
    "is_punct",
    "is_left_punct",
    "is_right_punct",
    "is_space",
    "is_bracket",
    "is_quote",
    "is_currency",
    "is_stop",
    "is_oov",
    "like_url",
    "like_num",
    "like_email",
    "is_sent_start",
    "is_sent_end",
)


class SpacyTranslator(BaseTranslator):
    """Run one spaCy pipeline pass and emit normalized sentences and tokens.

    Parameters
    ----------
    model:
        spaCy pipeline package name or model directory accepted by
        :func:`spacy.load`, for example ``"en_core_web_sm"``.
    text_field:
        Data column containing source document text.
    sentence_key:
        Key column appended to the source key for the sentence output.
    token_key:
        Key column appended to the sentence key for the token output.
    spacy_batch_size:
        Internal ``Language.pipe`` batch size. This is distinct from TeAL's
        operation ``batch_size``, which controls the number of source documents
        in each durable/resumable execution unit.
    disable:
        Optional spaCy pipeline components to disable while loading the model.
        The resulting pipeline must still provide sentence boundaries.

    Notes
    -----
    TeAL owns process-level parallelism. Each worker therefore calls
    ``Language.pipe(..., n_process=1)`` rather than creating a nested spaCy
    multiprocessing pool.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        model: str = "en_core_web_sm",
        text_field: str = "text",
        sentence_key: str = "sentence_id",
        token_key: str = "token_id",
        spacy_batch_size: int = 128,
        disable: Sequence[str] = (),
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(model, str) or not model:
            raise ValueError(
                "model must be a non-empty spaCy package name or model path."
            )
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        for name, value in (("sentence_key", sentence_key), ("token_key", token_key)):
            if not isinstance(value, str) or not value or value.startswith("_"):
                raise ValueError(
                    f"{name} must be a non-empty non-structural column name."
                )
        if sentence_key == token_key:
            raise ValueError("sentence_key and token_key must be different columns.")
        if isinstance(spacy_batch_size, bool) or not isinstance(spacy_batch_size, int):
            raise TypeError("spacy_batch_size must be a positive integer.")
        if spacy_batch_size <= 0:
            raise ValueError("spacy_batch_size must be a positive integer.")

        normalized_disable = tuple(str(component) for component in disable)
        if any(not component for component in normalized_disable):
            raise ValueError("disable must contain only non-empty component names.")

        self.model = model
        self.text_field = text_field
        self.sentence_key = sentence_key
        self.token_key = token_key
        self.spacy_batch_size = int(spacy_batch_size)
        self.disable = normalized_disable
        self.resolved_spacy_version: str | None = None
        self._frozen_model_path: str | None = None

    def translate(
        self,
        texts: str | Sequence[str] | pd.Series,
    ) -> dict[str, pd.DataFrame]:
        """Run spaCy over ordinary text and return normalized sentence/token tables."""
        if isinstance(texts, pd.Series):
            values = texts.fillna("").astype(str).tolist()
        elif isinstance(texts, str):
            values = [texts]
        elif isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            values = ["" if value is None else str(value) for value in texts]
        else:
            raise TypeError(
                "SpacyTranslator.translate(...) expects a string, sequence of strings, "
                "or pandas Series."
            )

        model_source = self._frozen_model_path or self.model
        if self.is_frozen and self._frozen_model_path is None:
            raise OperatorError(
                "Frozen SpacyTranslator snapshot is missing an operator-local spaCy "
                "pipeline asset and cannot be reused under strict freeze semantics."
            )
        nlp = _load_spacy_pipeline(model_source, self.disable)
        if self.resolved_spacy_version is not None:
            current_version = _spacy_runtime_version()
            if current_version != self.resolved_spacy_version:
                raise OperatorError(
                    "SpacyTranslator requires the spaCy runtime version recorded in its "
                    f"frozen snapshot ({self.resolved_spacy_version}); found {current_version}."
                )
        docs = nlp.pipe(values, batch_size=self.spacy_batch_size, n_process=1)
        sentence_rows: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        doc_iterator = iter(docs)
        for source_position, source_text in enumerate(values):
            try:
                doc = next(doc_iterator)
            except StopIteration as exc:
                raise ArtifactError("spaCy Language.pipe returned fewer Docs than source texts.") from exc
            if doc.text != source_text:
                raise ArtifactError(
                    "spaCy changed source text during tokenization; character offsets "
                    "must refer to the original source string."
                )
            sentence_keys: list[dict[str, Any]] = []
            sentence_data: list[dict[str, Any]] = []
            token_keys: list[dict[str, Any]] = []
            token_data: list[dict[str, Any]] = []
            _append_doc_rows(
                source_key={"source_position": source_position},
                doc=doc,
                sentence_key=self.sentence_key,
                token_key=self.token_key,
                sentence_keys=sentence_keys,
                sentence_data=sentence_data,
                token_keys=token_keys,
                token_data=token_data,
            )
            for key, data in zip(sentence_keys, sentence_data, strict=True):
                sentence_rows.append({**key, **data})
            for key, data in zip(token_keys, token_data, strict=True):
                token_rows.append({**key, **data})
        sentinel = object()
        if next(doc_iterator, sentinel) is not sentinel:
            raise ArtifactError("spaCy Language.pipe returned more Docs than source texts.")
        return {
            SENTENCES_LABEL: pd.DataFrame.from_records(
                sentence_rows,
                columns=["source_position", self.sentence_key, *SENTENCE_DATA_COLUMNS],
            ),
            TOKENS_LABEL: pd.DataFrame.from_records(
                token_rows,
                columns=[
                    "source_position", self.sentence_key, self.token_key, *TOKEN_DATA_COLUMNS
                ],
            ),
        }

    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> Mapping[str, OutputSpec]:
        _ = request
        source = _single_source(sources)
        key_columns = tuple(str(name) for name in source.primary_key)
        collisions = [
            name for name in (self.sentence_key, self.token_key) if name in key_columns
        ]
        if collisions:
            raise OperatorError(
                "SpacyTranslator key columns must extend the source primary key; "
                f"already present: {collisions}."
            )

        return {
            SENTENCES_LABEL: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=DEFAULT_SOURCE_LABEL,
            ),
            TOKENS_LABEL: OutputSpec(
                artifact_type="table",
                lineage_mode="extended_key",
                basis_labels=SENTENCES_LABEL,
            ),
        }

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
                f"SpacyTranslator does not accept operation parameters; got {sorted(params)}."
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
            raise OperatorError("SpacyTranslator requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
            mode="batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 1_000,
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
        missing = [name for name in [*key_columns, self.text_field] if name not in frame.columns]
        if missing:
            raise ArtifactError(f"SpacyTranslator source batch is missing columns {missing}.")

        translated = self.translate(frame[self.text_field])
        sentence_frame = translated[SENTENCES_LABEL]
        token_frame = translated[TOKENS_LABEL]
        outputs: dict[str, Mapping[str, Any]] = {}
        if not sentence_frame.empty:
            positions = sentence_frame["source_position"].to_numpy(dtype="int64")
            keys = frame.iloc[positions].loc[:, key_columns].reset_index(drop=True)
            keys[self.sentence_key] = sentence_frame[self.sentence_key].to_numpy()
            data = sentence_frame.loc[:, list(SENTENCE_DATA_COLUMNS)].reset_index(drop=True)
            outputs[SENTENCES_LABEL] = {"keys": keys, "data": data}
        if not token_frame.empty:
            positions = token_frame["source_position"].to_numpy(dtype="int64")
            keys = frame.iloc[positions].loc[:, key_columns].reset_index(drop=True)
            keys[self.sentence_key] = token_frame[self.sentence_key].to_numpy()
            keys[self.token_key] = token_frame[self.token_key].to_numpy()
            data = token_frame.loc[:, list(TOKEN_DATA_COLUMNS)].reset_index(drop=True)
            outputs[TOKENS_LABEL] = {"keys": keys, "data": data}
        return BatchResult(outputs=outputs)

    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        _ = batch_index, mode, request
        return result.outputs or None

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
    ) -> SpacyTranslator:
        _ = mode, request
        worker = self.from_json_state(self.to_json_state())
        worker._frozen_model_path = self._frozen_model_path
        worker.is_frozen = self.is_frozen
        return worker

    def prepare_for_freeze(self) -> None:
        """Resolve the configured spaCy pipeline before durable snapshot commit."""
        _load_spacy_pipeline(self.model, self.disable)
        self.resolved_spacy_version = _spacy_runtime_version()

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        nlp = _load_spacy_pipeline(self.model, self.disable)
        model_dir = assets_dir / "spacy_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            nlp.to_disk(model_dir)
        except Exception as exc:
            raise OperatorError(
                "SpacyTranslator could not persist the configured spaCy pipeline "
                "as an operator-local frozen asset."
            ) from exc
        self._frozen_model_path = str(model_dir)
        return {
            "spacy_model_dir": model_dir.name,
            "spacy_version": self.resolved_spacy_version,
        }

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("spacy_model_dir") if manifest else None
        if not isinstance(filename, str) or not filename:
            self._frozen_model_path = None
            return
        model_dir = assets_dir / filename
        if not model_dir.is_dir():
            raise OperatorError(
                f"Frozen spaCy model asset is missing: {model_dir}."
            )
        self._frozen_model_path = str(model_dir)

    def to_json_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text_field": self.text_field,
            "sentence_key": self.sentence_key,
            "token_key": self.token_key,
            "spacy_batch_size": self.spacy_batch_size,
            "disable": list(self.disable),
            "resolved_spacy_version": self.resolved_spacy_version,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SpacyTranslator:
        obj = cls(
            model=str(state.get("model", "en_core_web_sm")),
            text_field=str(state.get("text_field", "text")),
            sentence_key=str(state.get("sentence_key", "sentence_id")),
            token_key=str(state.get("token_key", "token_id")),
            spacy_batch_size=int(state.get("spacy_batch_size", 128)),
            disable=cast(Sequence[str], state.get("disable", ())),
        )
        raw_version = state.get("resolved_spacy_version")
        obj.resolved_spacy_version = None if raw_version is None else str(raw_version)
        return obj

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
    ) -> SpacyTranslator:
        _ = mode, route
        state = json.loads(
            (intermediate_dir / "state.json").read_text(encoding="utf-8")
        )
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        obj.operator_id = operator_id
        return obj


def _append_doc_rows(
    *,
    source_key: Mapping[str, Any],
    doc: Any,
    sentence_key: str,
    token_key: str,
    sentence_keys: list[dict[str, Any]],
    sentence_data: list[dict[str, Any]],
    token_keys: list[dict[str, Any]],
    token_data: list[dict[str, Any]],
) -> None:
    try:
        sentences = tuple(doc.sents)
    except ValueError as exc:
        raise ArtifactError(
            "spaCy pipeline did not provide sentence boundaries. Enable a parser, "
            "senter, or sentencizer component before using SpacyTranslator."
        ) from exc

    for sentence_id, sentence in enumerate(sentences):
        sentence_row_key = dict(source_key)
        sentence_row_key[sentence_key] = int(sentence_id)
        sentence_keys.append(sentence_row_key)
        sentence_data.append(
            {
                "text": sentence.text,
                "char_start": int(sentence.start_char),
                "char_end": int(sentence.end_char),
            }
        )

        for token_id, token in enumerate(sentence):
            if token.i != sentence.start + token_id:
                raise ArtifactError(
                    "spaCy sentence token indexes are not contiguous; cannot normalize "
                    "document-global token indexes to TeAL sentence-local token_id values."
                )
            if token.head.i < sentence.start or token.head.i >= sentence.end:
                raise ArtifactError(
                    "spaCy dependency head crosses a sentence boundary. TeAL token "
                    "head_token_id is sentence-local and requires heads to remain "
                    "within the token's sentence."
                )

            token_row_key = dict(sentence_row_key)
            token_row_key[token_key] = int(token_id)
            token_keys.append(token_row_key)
            token_data.append(_token_record(token, sentence_start=sentence.start))


def _token_record(token: Any, *, sentence_start: int) -> dict[str, Any]:
    return {
        "text": token.text,
        "lemma": token.lemma_,
        "norm": token.norm_,
        "shape": token.shape_,
        "pos": token.pos_,
        "tag": token.tag_,
        "morph": str(token.morph),
        "dep": token.dep_,
        "head_token_id": int(token.head.i - sentence_start),
        "ent_iob": token.ent_iob_,
        "ent_type": token.ent_type_,
        "char_start": int(token.idx),
        "char_end": int(token.idx + len(token.text)),
        "is_alpha": bool(token.is_alpha),
        "is_ascii": bool(token.is_ascii),
        "is_digit": bool(token.is_digit),
        "is_lower": bool(token.is_lower),
        "is_upper": bool(token.is_upper),
        "is_title": bool(token.is_title),
        "is_punct": bool(token.is_punct),
        "is_left_punct": bool(token.is_left_punct),
        "is_right_punct": bool(token.is_right_punct),
        "is_space": bool(token.is_space),
        "is_bracket": bool(token.is_bracket),
        "is_quote": bool(token.is_quote),
        "is_currency": bool(token.is_currency),
        "is_stop": bool(token.is_stop),
        "is_oov": bool(token.is_oov),
        "like_url": bool(token.like_url),
        "like_num": bool(token.like_num),
        "like_email": bool(token.like_email),
        "is_sent_start": None
        if token.is_sent_start is None
        else bool(token.is_sent_start),
        "is_sent_end": None if token.is_sent_end is None else bool(token.is_sent_end),
    }


def _table_payload(
    keys: list[dict[str, Any]],
    data: list[dict[str, Any]],
    *,
    key_columns: Sequence[str],
    data_columns: Sequence[str],
) -> Mapping[str, Any] | None:
    if not keys:
        return None
    return {
        "keys": pd.DataFrame.from_records(keys, columns=list(key_columns)),
        "data": pd.DataFrame.from_records(data, columns=list(data_columns)),
    }


def _spacy_runtime_version() -> str:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise OperatorError(
            "SpacyTranslator requires spaCy. Install TeAL with the spaCy optional "
            "dependencies (for example `uv sync --extra spacy`)."
        ) from exc
    return str(getattr(spacy, "__version__", "unknown"))


@lru_cache(maxsize=8)
def _load_spacy_pipeline(model: str, disable: tuple[str, ...]):
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise OperatorError(
            "SpacyTranslator requires spaCy. Install TeAL with the spaCy optional "
            "dependencies (for example `uv sync --extra spacy`)."
        ) from exc

    try:
        return spacy.load(model, disable=list(disable))
    except Exception as exc:
        raise OperatorError(
            f"Could not load spaCy pipeline {model!r}. Install the requested model "
            "or provide a valid saved pipeline directory."
        ) from exc


def _single_source(sources: Mapping[str, BaseArtifact]) -> BaseArtifact:
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"SpacyTranslator requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"SpacyTranslator expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"SpacyTranslator expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
