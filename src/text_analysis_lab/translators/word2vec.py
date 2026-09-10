"""Train a local Gensim Word2Vec model into a named-row matrix artifact."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
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
    SourceRequest,
    TranslationMode,
    TranslationRequest,
)
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


Architecture = Literal["skipgram", "cbow"]


class Word2Vec(BaseTranslator):
    """Train static word vectors locally with Gensim.

    This is deliberately a one-shot training translator. The learned object is
    the emitted named-row dense matrix: integer primary keys preserve TeAL's
    structural contract, while unique row names hold the vocabulary. Reusing
    embeddings is therefore ordinary matrix lookup, not fitted-operator reuse.

    By default, the immediate parent key defines a sequence, so tokens keyed
    ``(..., sentence_id, token_id)`` train within sentence boundaries.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        field: str = "text",
        sequence_by: str | Sequence[str] | None = None,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        architecture: Architecture = "skipgram",
        negative_samples: int = 5,
        epochs: int = 5,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        sample: float = 0.0,
        ns_exponent: float = 0.75,
        cbow_mean: bool = True,
        shrink_windows: bool = True,
        seed: int = 42,
        training_workers: int = 1,
        training_batch_size: int = 10_000,
        word_key: str = "word_id",
        row_name: str = "word",
        drop_empty: bool = True,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(field, str) or not field:
            raise ValueError("field must be a non-empty string.")
        normalized_sequence = (
            None
            if sequence_by is None
            else _normalize_key_columns(sequence_by, name="sequence_by")
        )
        if int(vector_size) <= 0:
            raise ValueError("vector_size must be positive.")
        if int(window) <= 0:
            raise ValueError("window must be positive.")
        if int(min_count) <= 0:
            raise ValueError("min_count must be positive.")
        if architecture not in {"skipgram", "cbow"}:
            raise ValueError("architecture must be 'skipgram' or 'cbow'.")
        if int(negative_samples) <= 0:
            raise ValueError("negative_samples must be positive.")
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive.")
        if not 0.0 < float(learning_rate):
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < float(min_learning_rate) <= float(learning_rate):
            raise ValueError(
                "min_learning_rate must be positive and no larger than learning_rate."
            )
        if not 0.0 <= float(sample) < 1.0:
            raise ValueError("sample must be in [0, 1).")
        if not np.isfinite(float(ns_exponent)):
            raise ValueError("ns_exponent must be finite.")
        if int(training_workers) <= 0:
            raise ValueError("training_workers must be positive.")
        if int(training_batch_size) <= 0:
            raise ValueError("training_batch_size must be positive.")
        if not isinstance(word_key, str) or not word_key:
            raise ValueError("word_key must be a non-empty string.")
        if not isinstance(row_name, str) or not row_name or row_name.startswith("_"):
            raise ValueError("row_name must be a non-empty, non-structural string.")

        self.field = field
        self.sequence_by = normalized_sequence
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.architecture = architecture
        self.negative_samples = int(negative_samples)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.min_learning_rate = float(min_learning_rate)
        self.sample = float(sample)
        self.ns_exponent = float(ns_exponent)
        self.cbow_mean = bool(cbow_mean)
        self.shrink_windows = bool(shrink_windows)
        self.seed = int(seed)
        self.training_workers = int(training_workers)
        self.training_batch_size = int(training_batch_size)
        self.word_key = word_key
        self.row_name = row_name
        self.drop_empty = bool(drop_empty)

        # Audit diagnostics only. Learned vocabulary and vectors live exclusively
        # in the output artifact and are never persisted as operator assets.
        self.training_loss_: tuple[float, ...] = ()
        self.gensim_version_: str | None = None

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        _single_source(sources)
        return OutputSpec(
            artifact_type="dense_matrix",
            lineage_mode="new_key",
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
                f"Word2Vec does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = request
        if mode != "translate":
            raise OperatorError("Word2Vec is a one-shot translate operator, not fitted state.")
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("Word2Vec requires a table artifact source.")
        self._resolve_sequence_by(tuple(str(name) for name in source.primary_key))
        return SourceRequest(
            artifact_type="table",
            mode="full_artifact",
            columns=ColumnRequest(keys=True, data=self.field, metadata=False),
            batch_size=None,
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
        if mode != "translate":
            raise OperatorError(f"Unsupported Word2Vec mode {mode!r}.")
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        source_key = tuple(str(name) for name in packet.primary_key)
        missing = [name for name in [*source_key, self.field] if name not in frame.columns]
        if missing:
            raise ArtifactError(f"Word2Vec source is missing columns {missing}.")

        sequences = _prepare_sequences(
            frame,
            field=self.field,
            sequence_by=self._resolve_sequence_by(source_key),
            drop_empty=self.drop_empty,
        )
        words, counts, vectors, losses, gensim_version = self._train(sequences)
        self.training_loss_ = losses
        self.gensim_version_ = gensim_version
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": pd.DataFrame(
                        {self.word_key: np.arange(len(words), dtype=np.int64)}
                    ),
                    "metadata": pd.DataFrame(
                        {"count": counts.astype(np.int64, copy=False)}
                    ),
                    "data": {
                        "values": vectors,
                        "columns": _dimension_columns(self.vector_size),
                        "row_names": words,
                        "row_name": self.row_name,
                    },
                }
            }
        )

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

    def to_json_state(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "sequence_by": None if self.sequence_by is None else list(self.sequence_by),
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "architecture": self.architecture,
            "negative_samples": self.negative_samples,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "sample": self.sample,
            "ns_exponent": self.ns_exponent,
            "cbow_mean": self.cbow_mean,
            "shrink_windows": self.shrink_windows,
            "seed": self.seed,
            "training_workers": self.training_workers,
            "training_batch_size": self.training_batch_size,
            "word_key": self.word_key,
            "row_name": self.row_name,
            "drop_empty": self.drop_empty,
            "training_loss": list(self.training_loss_),
            "gensim_version": self.gensim_version_,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "Word2Vec":
        obj = cls(
            field=str(state.get("field", "text")),
            sequence_by=cast(Sequence[str] | None, state.get("sequence_by")),
            vector_size=int(state.get("vector_size", 100)),
            window=int(state.get("window", 5)),
            min_count=int(state.get("min_count", 5)),
            architecture=cast(Architecture, state.get("architecture", "skipgram")),
            negative_samples=int(state.get("negative_samples", 5)),
            epochs=int(state.get("epochs", 5)),
            learning_rate=float(state.get("learning_rate", 0.025)),
            min_learning_rate=float(state.get("min_learning_rate", 0.0001)),
            sample=float(state.get("sample", 0.0)),
            ns_exponent=float(state.get("ns_exponent", 0.75)),
            cbow_mean=bool(state.get("cbow_mean", True)),
            shrink_windows=bool(state.get("shrink_windows", True)),
            seed=int(state.get("seed", 42)),
            training_workers=int(state.get("training_workers", 1)),
            training_batch_size=int(state.get("training_batch_size", 10_000)),
            word_key=str(state.get("word_key", "word_id")),
            row_name=str(state.get("row_name", "word")),
            drop_empty=bool(state.get("drop_empty", True)),
        )
        raw_loss = state.get("training_loss", ())
        if isinstance(raw_loss, Sequence) and not isinstance(raw_loss, (str, bytes)):
            obj.training_loss_ = tuple(float(value) for value in raw_loss)
        raw_version = state.get("gensim_version")
        obj.gensim_version_ = None if raw_version is None else str(raw_version)
        return obj

    def _train(
        self, sequences: Sequence[Sequence[str]]
    ) -> tuple[list[str], np.ndarray, np.ndarray, tuple[float, ...], str]:
        try:
            import gensim
            from gensim.models import Word2Vec as GensimWord2Vec
        except (ImportError, RuntimeError) as exc:
            raise OperatorError(
                "Word2Vec training requires Gensim 4.4 or newer. Install TeAL's "
                "optional word2vec extra (for example: uv sync --extra word2vec)."
            ) from exc

        raw_version = str(getattr(gensim, "__version__", "unknown"))
        if _version_tuple(raw_version) < (4, 4):
            raise OperatorError(
                f"Word2Vec training requires Gensim>=4.4; found {raw_version}."
            )
        if self.training_workers > 1:
            warnings.warn(
                "Word2Vec training_workers>1 enables concurrent Gensim updates; "
                "exact seeded reproducibility is not guaranteed.",
                UserWarning,
                stacklevel=2,
            )
        model = GensimWord2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.training_workers,
            sg=1 if self.architecture == "skipgram" else 0,
            hs=0,
            negative=self.negative_samples,
            alpha=self.learning_rate,
            min_alpha=self.min_learning_rate,
            sample=self.sample,
            ns_exponent=self.ns_exponent,
            cbow_mean=1 if self.cbow_mean else 0,
            shrink_windows=self.shrink_windows,
            seed=self.seed,
            batch_words=self.training_batch_size,
            sorted_vocab=1,
            hashfxn=_stable_hash,
        )
        model.build_vocab(sequences)
        if len(model.wv) < 2:
            raise ArtifactError(
                "Word2Vec needs at least two vocabulary words after min_count filtering."
            )
        retained = set(model.wv.key_to_index)
        if not any(sum(word in retained for word in sequence) >= 2 for sequence in sequences):
            raise ArtifactError(
                "Word2Vec found no sequence containing at least two retained vocabulary words."
            )

        loss_recorder = _EpochLossRecorder()
        model.train(
            sequences,
            total_examples=model.corpus_count,
            epochs=self.epochs,
            start_alpha=self.learning_rate,
            end_alpha=self.min_learning_rate,
            compute_loss=True,
            callbacks=(loss_recorder,),
        )
        ranked = sorted(
            (
                (word, int(model.wv.get_vecattr(word, "count")))
                for word in model.wv.index_to_key
            ),
            key=lambda item: (-item[1], item[0]),
        )
        words = [word for word, _ in ranked]
        counts = np.asarray([count for _, count in ranked], dtype=np.int64)
        vectors = np.stack(
            [np.asarray(model.wv[word], dtype=np.float32) for word in words], axis=0
        )
        if vectors.shape != (len(words), self.vector_size):  # pragma: no cover
            raise OperatorError("Gensim returned an unexpected Word2Vec vector shape.")
        if not np.isfinite(vectors).all():  # pragma: no cover
            raise OperatorError("Gensim Word2Vec training produced non-finite vectors.")
        return words, counts, vectors, tuple(loss_recorder.losses), raw_version

    def _resolve_sequence_by(self, source_key: Sequence[str]) -> tuple[str, ...]:
        source_key = tuple(str(name) for name in source_key)
        if len(source_key) < 2:
            raise OperatorError(
                "Word2Vec training requires a hierarchical primary key so tokens can "
                "be assigned to sequences."
            )
        sequence_by = source_key[:-1] if self.sequence_by is None else self.sequence_by
        if not _is_prefix(sequence_by, source_key) or len(sequence_by) >= len(source_key):
            raise OperatorError(
                "sequence_by must be a non-empty proper prefix of the source primary "
                f"key {list(source_key)}; got {list(sequence_by)}."
            )
        return sequence_by


LocalWord2Vec = Word2Vec


class _EpochLossRecorder:
    """Collect deltas from Gensim's cumulative loss counter."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self._previous = 0.0

    def on_train_begin(self, model: Any) -> None:
        _ = model

    def on_train_end(self, model: Any) -> None:
        _ = model

    def on_epoch_begin(self, model: Any) -> None:
        _ = model

    def on_epoch_end(self, model: Any) -> None:
        cumulative = float(model.get_latest_training_loss())
        self.losses.append(cumulative - self._previous)
        self._previous = cumulative


def _prepare_sequences(
    frame: pd.DataFrame,
    *,
    field: str,
    sequence_by: Sequence[str],
    drop_empty: bool,
) -> list[list[str]]:
    if frame.empty:
        raise ArtifactError("Word2Vec cannot train on an empty source.")
    raw = frame[field]
    words = raw.astype("string")
    valid = raw.notna()
    if drop_empty:
        valid &= words.str.len().fillna(0).gt(0)
    if not bool(valid.any()):
        raise ArtifactError("Word2Vec found no trainable token values.")
    work = frame.loc[valid, [*dict.fromkeys([*sequence_by, field])]].copy()
    work["_word"] = words.loc[valid].astype(str).to_numpy()
    sequence_index = pd.MultiIndex.from_frame(work.loc[:, list(sequence_by)])
    sequence_ids, _ = pd.factorize(sequence_index, sort=False)
    sequences = [
        group["_word"].astype(str).tolist()
        for _, group in work.groupby(sequence_ids, sort=False)
        if len(group) >= 2
    ]
    if not sequences:
        raise ArtifactError("Word2Vec found no sequence containing at least two tokens.")
    return sequences


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(
        str(value).encode("utf-8"), digest_size=8, person=b"TeALW2V"
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for piece in value.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _normalize_key_columns(value: str | Sequence[str], *, name: str) -> tuple[str, ...]:
    columns = (value,) if isinstance(value, str) else tuple(value)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{name} must contain non-empty string column names.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} cannot contain duplicate columns.")
    return columns


def _is_prefix(prefix: Sequence[str], whole: Sequence[str]) -> bool:
    prefix = tuple(prefix)
    whole = tuple(whole)
    return bool(prefix) and len(prefix) <= len(whole) and whole[: len(prefix)] == prefix


def _dimension_columns(vector_size: int) -> list[str]:
    return [f"dimension_{index}" for index in range(vector_size)]


def _single_source(sources: Mapping[str, "BaseArtifact"]) -> "BaseArtifact":
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"Word2Vec expects exactly source label {DEFAULT_SOURCE_LABEL!r}; "
            f"got {sorted(sources)}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            f"Word2Vec expects exactly input label {DEFAULT_SOURCE_LABEL!r}; "
            f"got {sorted(inputs)}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError("Word2Vec expected table-form pandas data.")
    return value
