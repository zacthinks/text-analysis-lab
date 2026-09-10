"""Scikit-learn count-vectorization translator for Text Analysis Lab (TeAL)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError, OperatorError, OperatorNotFittedError
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

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


class CountVectorizer(BaseTranslator):
    """Build a sparse count document-term matrix with scikit-learn.

    An unfitted instance requests the full source once, learns a vocabulary with
    ``sklearn.feature_extraction.text.CountVectorizer``, emits a sparse-matrix
    artifact, and is frozen with that learned vocabulary.  A frozen instance
    transforms later artifacts batch-wise and may run in parallel.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        text_field: str = "text",
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        stop_words: str | Sequence[str] | None = None,
        lowercase: bool = True,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        binary: bool = False,
        stemmer: str | None = None,
        vocabulary: Mapping[str, int] | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string.")
        if len(tuple(ngram_range)) != 2:
            raise ValueError("ngram_range must contain exactly two integers.")
        ngram_min, ngram_max = (int(value) for value in ngram_range)
        if ngram_min <= 0 or ngram_max < ngram_min:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n.")
        if max_features is not None and int(max_features) <= 0:
            raise ValueError("max_features must be positive or None.")

        self.text_field = text_field
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = _normalize_stop_words(stop_words)
        self.lowercase = bool(lowercase)
        self.token_pattern = str(token_pattern)
        self.ngram_range = (ngram_min, ngram_max)
        self.max_features = None if max_features is None else int(max_features)
        self.binary = bool(binary)
        if stemmer not in {None, "porter"}:
            raise ValueError("stemmer must be None or 'porter'.")
        self.stemmer = stemmer
        self.vocabulary_: dict[str, int] | None = (
            None if vocabulary is None else _normalize_vocabulary(vocabulary)
        )
        self._vectorizer = None
        if self.vocabulary_ is not None:
            self._vectorizer = self._make_vectorizer(vocabulary=self.vocabulary_)

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        return self.vocabulary_ is not None

    @property
    def supports_fit_translate(self) -> bool:
        return True

    @property
    def supports_parallel_translate(self) -> bool:
        return self.is_fitted

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        if mode == "fit_translate":
            return route == "sequential"
        return mode == "translate" and route in {"sequential", "parallel"}

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        _single_source(sources)
        return OutputSpec(
            artifact_type="sparse_matrix",
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
        if params:
            raise OperatorError(
                f"CountVectorizer does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("CountVectorizer requires a table artifact source.")
        return SourceRequest(
            artifact_type="table",
            mode="full_artifact" if mode == "fit_translate" else "batches",
            columns=ColumnRequest(keys=True, data=self.text_field, metadata=False),
            batch_size=(
                None
                if mode == "fit_translate"
                else request.batch_size if request.batch_size is not None else 10_000
            ),
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
        _ = request
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        key_columns = [str(name) for name in packet.primary_key]
        missing = [name for name in [*key_columns, self.text_field] if name not in frame.columns]
        if missing:
            raise ArtifactError(f"CountVectorizer source batch is missing columns {missing}.")

        texts = frame[self.text_field].fillna("").astype("string").tolist()
        keys = frame.loc[:, key_columns].reset_index(drop=True)

        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError("fit_translate received an already fitted CountVectorizer.")
            self._vectorizer = self._make_vectorizer()
            matrix = self._vectorizer.fit_transform(texts)
            self.vocabulary_ = _normalize_vocabulary(self._vectorizer.vocabulary_)
        elif mode == "translate":
            vectorizer = self._require_vectorizer()
            matrix = vectorizer.transform(texts)
        else:  # pragma: no cover - runner validates mode
            raise OperatorError(f"Unsupported CountVectorizer mode {mode!r}.")

        columns = self._feature_names()
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "data": {"values": matrix, "columns": columns},
                }
            }
        )

    def transform_external_texts(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        params: Mapping[str, Any] | None = None,
    ):
        """Vectorize new texts with the frozen vocabulary without writing artifacts."""
        _ = query, params
        values = ["" if value is None else str(value) for value in texts]
        return self._require_vectorizer().transform(values)

    def supports_external_transform(self, *, query: bool, input_kind: str) -> bool:
        _ = query
        return input_kind == "texts" and self.is_fitted

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
    ) -> "CountVectorizer":
        _ = request
        if mode != "translate" or not self.is_fitted:
            raise OperatorNotFittedError(
                "Parallel CountVectorizer workers require a fitted vocabulary."
            )
        return self.from_json_state(self.to_json_state(include_vocabulary=True))

    def to_json_state(self, *, include_vocabulary: bool = False) -> dict[str, Any]:
        state: dict[str, Any] = {
            "text_field": self.text_field,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "stop_words": self.stop_words,
            "lowercase": self.lowercase,
            "token_pattern": self.token_pattern,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "binary": self.binary,
            "stemmer": self.stemmer,
            "is_fitted": self.is_fitted,
        }
        if include_vocabulary and self.vocabulary_ is not None:
            state["vocabulary"] = self.vocabulary_
        return state

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "CountVectorizer":
        vocabulary = state.get("vocabulary")
        return cls(
            text_field=str(state.get("text_field", "text")),
            min_df=cast(int | float, state.get("min_df", 1)),
            max_df=cast(int | float, state.get("max_df", 1.0)),
            stop_words=cast(str | Sequence[str] | None, state.get("stop_words")),
            lowercase=bool(state.get("lowercase", True)),
            token_pattern=str(state.get("token_pattern", r"(?u)\b\w\w+\b")),
            ngram_range=tuple(cast(Sequence[int], state.get("ngram_range", (1, 1)))),
            max_features=cast(int | None, state.get("max_features")),
            binary=bool(state.get("binary", False)),
            stemmer=cast(str | None, state.get("stemmer")),
            vocabulary=cast(Mapping[str, int] | None, vocabulary),
        )

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if self.vocabulary_ is None:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "vocabulary.parquet"
        _vocabulary_frame(self.vocabulary_).to_parquet(path, index=False)
        return {"vocabulary_file": path.name}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        if not manifest:
            if self.is_fitted:
                return
            return
        filename = manifest.get("vocabulary_file")
        if not isinstance(filename, str) or not filename:
            raise OperatorError("CountVectorizer asset manifest is missing vocabulary_file.")
        frame = pd.read_parquet(assets_dir / filename)
        self.vocabulary_ = _vocabulary_from_frame(frame)
        self._vectorizer = self._make_vectorizer(vocabulary=self.vocabulary_)

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
        state = self.to_json_state(include_vocabulary=False)
        state["operator_id"] = operator_id
        manifest = self.save_assets(intermediate_dir)
        state["assets"] = dict(manifest)
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
    ) -> "CountVectorizer":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("CountVectorizer intermediate assets must be a mapping.")
        obj.load_assets(intermediate_dir, assets)
        obj.operator_id = operator_id
        return obj

    def _make_vectorizer(self, *, vocabulary: Mapping[str, int] | None = None):
        try:
            from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise OperatorError("CountVectorizer requires scikit-learn.") from exc

        analyzer = None
        stop_words = self.stop_words
        lowercase = self.lowercase
        token_pattern: str | None = self.token_pattern
        ngram_range = self.ngram_range
        if self.stemmer == "porter":
            analyzer = _PorterAnalyzer(
                token_pattern=self.token_pattern,
                lowercase=self.lowercase,
                stop_words=self.stop_words,
                ngram_range=self.ngram_range,
            )
            # A callable analyzer owns preprocessing/tokenization/stopword removal/ngrams.
            stop_words = None
            lowercase = False
            token_pattern = None
            ngram_range = (1, 1)

        return SklearnCountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            analyzer=analyzer if analyzer is not None else "word",
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_features=self.max_features,
            binary=self.binary,
            vocabulary=None if vocabulary is None else dict(vocabulary),
        )

    def _require_vectorizer(self):
        if self.vocabulary_ is None:
            raise OperatorNotFittedError("CountVectorizer has no fitted vocabulary.")
        if self._vectorizer is None:
            self._vectorizer = self._make_vectorizer(vocabulary=self.vocabulary_)
        return self._vectorizer

    def _feature_names(self) -> list[str]:
        if self.vocabulary_ is None:
            raise OperatorNotFittedError("CountVectorizer has no fitted vocabulary.")
        frame = _vocabulary_frame(self.vocabulary_)
        return frame["term"].astype(str).tolist()



class _PorterAnalyzer:
    """Pickle-safe word analyzer with optional stopwords and Porter stemming."""

    def __init__(
        self,
        *,
        token_pattern: str,
        lowercase: bool,
        stop_words: str | Sequence[str] | None,
        ngram_range: tuple[int, int],
    ) -> None:
        import re

        self.token_pattern = str(token_pattern)
        self.lowercase = bool(lowercase)
        self.ngram_range = tuple(int(value) for value in ngram_range)
        self._regex = re.compile(self.token_pattern)
        if stop_words == "english":
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            words = {str(value) for value in ENGLISH_STOP_WORDS}
        elif stop_words is None:
            words = set()
        elif isinstance(stop_words, str):
            raise ValueError(
                "When stemmer='porter', stop_words must be None, 'english', or a sequence."
            )
        else:
            words = {str(value) for value in stop_words}
        self.stop_words = {word.lower() for word in words} if self.lowercase else words

    def __call__(self, text: str) -> list[str]:
        from nltk.stem import PorterStemmer

        value = str(text)
        if self.lowercase:
            value = value.lower()
        raw = self._regex.findall(value)
        tokens = [str(token) for token in raw if str(token) not in self.stop_words]
        stem = PorterStemmer().stem
        stems = [stem(token) for token in tokens]
        low, high = self.ngram_range
        features: list[str] = []
        for n in range(low, high + 1):
            if n == 1:
                features.extend(stems)
            elif len(stems) >= n:
                features.extend(
                    " ".join(stems[index : index + n])
                    for index in range(len(stems) - n + 1)
                )
        return features


def _normalize_stop_words(value: str | Sequence[str] | None) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    return [str(item) for item in value]


def _normalize_vocabulary(vocabulary: Mapping[str, int]) -> dict[str, int]:
    series = pd.Series(dict(vocabulary), name="term_id", dtype="int64")
    if series.empty:
        raise ValueError("CountVectorizer vocabulary cannot be empty.")
    if series.index.astype(str).duplicated().any():
        raise ValueError("CountVectorizer vocabulary terms must be unique.")
    values = series.to_numpy(dtype="int64", copy=False)
    if not np.array_equal(np.sort(values), np.arange(len(values), dtype="int64")):
        raise ValueError("CountVectorizer vocabulary indices must be contiguous from zero.")
    series.index = series.index.astype(str)
    return cast(dict[str, int], series.to_dict())


def _vocabulary_frame(vocabulary: Mapping[str, int]) -> pd.DataFrame:
    series = pd.Series(dict(vocabulary), name="term_id", dtype="int64")
    frame = series.rename_axis("term").reset_index()
    frame["term"] = frame["term"].astype(str)
    return frame.sort_values("term_id", kind="stable").reset_index(drop=True)


def _vocabulary_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    required = {"term", "term_id"}
    if not required.issubset(frame.columns):
        raise OperatorError(
            f"CountVectorizer vocabulary asset must contain {sorted(required)}."
        )
    series = pd.Series(
        frame["term_id"].astype("int64").to_numpy(),
        index=frame["term"].astype(str).to_numpy(),
    )
    return _normalize_vocabulary(series.to_dict())


def _single_source(sources: Mapping[str, "BaseArtifact"]) -> "BaseArtifact":
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"CountVectorizer requires exactly one source under {DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"CountVectorizer expected one input under {DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            f"CountVectorizer expected a pandas DataFrame packet; got {type(value).__name__}."
        )
    return value
