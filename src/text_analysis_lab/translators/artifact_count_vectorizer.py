"""Count arbitrary row-level artifact fields into sparse matrices."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

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


_INTERNAL_NGRAM_SEPARATOR = "\u241f"  # SYMBOL FOR UNIT SEPARATOR; reserved internally.


class ArtifactCountVectorizer(BaseTranslator):
    """Aggregate an artifact field into a sparse count matrix.

    Each source row contributes one atomic value from ``field``.  Values are
    counted at the retained primary-key prefix ``group_by``. N-grams are formed
    in canonical source-row order within ``sequence_by`` boundaries. If
    ``sequence_by`` is omitted, the immediate parent key of the source rows is
    used, so token n-grams naturally stop at sentence boundaries for a source
    keyed ``(..., sentence_id, token_id)``.

    This translator does *not* tokenize, lowercase, or apply stop-word rules.
    Those choices belong upstream (for example in a spaCy token artifact or an
    explicit subset). Null values are omitted; empty strings are omitted by
    default.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        field: str,
        group_by: str | Sequence[str],
        sequence_by: str | Sequence[str] | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        binary: bool = False,
        drop_empty: bool = True,
        ngram_separator: str = " ",
        vocabulary: Mapping[str, int] | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(field, str) or not field:
            raise ValueError("field must be a non-empty string.")
        self.field = field
        self.group_by = _normalize_key_columns(group_by, name="group_by")
        self.sequence_by = (
            None
            if sequence_by is None
            else _normalize_key_columns(sequence_by, name="sequence_by")
        )
        self.ngram_range = _normalize_ngram_range(ngram_range)
        self.min_df = _validate_df_threshold(min_df, name="min_df")
        self.max_df = _validate_df_threshold(max_df, name="max_df")
        if max_features is not None and int(max_features) <= 0:
            raise ValueError("max_features must be positive or None.")
        self.max_features = None if max_features is None else int(max_features)
        self.binary = bool(binary)
        self.drop_empty = bool(drop_empty)
        if not isinstance(ngram_separator, str) or not ngram_separator:
            raise ValueError("ngram_separator must be a non-empty string.")
        if _INTERNAL_NGRAM_SEPARATOR in ngram_separator:
            raise ValueError("ngram_separator cannot contain TeAL's reserved separator.")
        self.ngram_separator = ngram_separator
        self.vocabulary_: dict[str, int] | None = (
            None if vocabulary is None else _normalize_vocabulary(vocabulary)
        )

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
        # Aggregation can combine many finer rows into one output row and n-grams
        # may span arbitrary physical source batches. Keep one canonical full-
        # artifact pass until TeAL has group-aware partition planning.
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        return route == "sequential" and mode in {"fit_translate", "translate"}

    def output_specs(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        request: TranslationRequest,
    ) -> OutputSpec:
        _ = request
        source = _single_source(sources)
        self._validate_source_key(source.primary_key)
        return OutputSpec(
            artifact_type="sparse_matrix",
            lineage_mode="reduced_key",
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
                "ArtifactCountVectorizer does not accept operation parameters; "
                f"got {sorted(params)}."
            )
        return {}

    def input_request(
        self,
        *,
        sources: Mapping[str, "BaseArtifact"],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest:
        _ = mode, request
        source = _single_source(sources)
        if source.artifact_type.value != "table":
            raise OperatorError("ArtifactCountVectorizer requires a table artifact source.")
        self._validate_source_key(source.primary_key)
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
        packet = _single_input(inputs)
        frame = _require_frame(packet.data)
        source_key = tuple(str(name) for name in packet.primary_key)
        sequence_by = self._validate_source_key(source_key)
        required = [*source_key, self.field]
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ArtifactError(
                f"ArtifactCountVectorizer source is missing columns {missing}."
            )

        group_keys, events = _build_feature_events(
            frame,
            field=self.field,
            group_by=self.group_by,
            sequence_by=sequence_by,
            ngram_range=self.ngram_range,
            drop_empty=self.drop_empty,
            ngram_separator=self.ngram_separator,
        )

        if mode == "fit_translate":
            if self.is_fitted:
                raise OperatorError(
                    "fit_translate received an already fitted ArtifactCountVectorizer."
                )
            self.vocabulary_ = _fit_vocabulary(
                events,
                n_groups=len(group_keys),
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
            )
        elif mode == "translate":
            if not self.is_fitted:
                raise OperatorNotFittedError(
                    "ArtifactCountVectorizer has no fitted vocabulary."
                )
        else:  # pragma: no cover - runner validates mode
            raise OperatorError(f"Unsupported ArtifactCountVectorizer mode {mode!r}.")

        vocabulary = cast(dict[str, int], self.vocabulary_)
        matrix = _events_to_matrix(
            events,
            n_groups=len(group_keys),
            vocabulary=vocabulary,
            binary=self.binary,
        )
        columns = _feature_names(vocabulary)
        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": group_keys,
                    "data": {"values": matrix, "columns": columns},
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

    def to_json_state(self, *, include_vocabulary: bool = False) -> dict[str, Any]:
        state: dict[str, Any] = {
            "field": self.field,
            "group_by": list(self.group_by),
            "sequence_by": None if self.sequence_by is None else list(self.sequence_by),
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "binary": self.binary,
            "drop_empty": self.drop_empty,
            "ngram_separator": self.ngram_separator,
            "is_fitted": self.is_fitted,
        }
        if include_vocabulary and self.vocabulary_ is not None:
            state["vocabulary"] = self.vocabulary_
        return state

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "ArtifactCountVectorizer":
        return cls(
            field=str(state["field"]),
            group_by=cast(Sequence[str], state["group_by"]),
            sequence_by=cast(Sequence[str] | None, state.get("sequence_by")),
            ngram_range=tuple(cast(Sequence[int], state.get("ngram_range", (1, 1)))),
            min_df=cast(int | float, state.get("min_df", 1)),
            max_df=cast(int | float, state.get("max_df", 1.0)),
            max_features=cast(int | None, state.get("max_features")),
            binary=bool(state.get("binary", False)),
            drop_empty=bool(state.get("drop_empty", True)),
            ngram_separator=str(state.get("ngram_separator", " ")),
            vocabulary=cast(Mapping[str, int] | None, state.get("vocabulary")),
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
            return
        filename = manifest.get("vocabulary_file")
        if not isinstance(filename, str) or not filename:
            raise OperatorError(
                "ArtifactCountVectorizer asset manifest is missing vocabulary_file."
            )
        self.vocabulary_ = _vocabulary_from_frame(
            pd.read_parquet(assets_dir / filename)
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
        state = self.to_json_state(include_vocabulary=False)
        state["operator_id"] = operator_id
        state["assets"] = dict(self.save_assets(intermediate_dir))
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
    ) -> "ArtifactCountVectorizer":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        obj = cls.from_json_state(cast(Mapping[str, Any], state))
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError(
                "ArtifactCountVectorizer intermediate assets must be a mapping."
            )
        obj.load_assets(intermediate_dir, assets)
        obj.operator_id = operator_id
        return obj

    def _validate_source_key(self, source_key: Sequence[str]) -> tuple[str, ...]:
        source_key = tuple(str(name) for name in source_key)
        if len(source_key) < 2:
            raise OperatorError(
                "ArtifactCountVectorizer requires a source with a hierarchical "
                "primary key so it can reduce to a parent grain."
            )
        if not _is_prefix(self.group_by, source_key) or len(self.group_by) >= len(source_key):
            raise OperatorError(
                "group_by must be a non-empty proper retained prefix of the source "
                f"primary key {list(source_key)}; got {list(self.group_by)}."
            )
        sequence_by = (
            source_key[:-1] if self.sequence_by is None else tuple(self.sequence_by)
        )
        if not _is_prefix(self.group_by, sequence_by):
            raise OperatorError(
                "sequence_by must retain group_by as its prefix; got "
                f"group_by={list(self.group_by)}, sequence_by={list(sequence_by)}."
            )
        if not _is_prefix(sequence_by, source_key) or len(sequence_by) >= len(source_key):
            raise OperatorError(
                "sequence_by must be a proper retained prefix of the source primary "
                f"key {list(source_key)}; got {list(sequence_by)}."
            )
        return sequence_by


def _normalize_key_columns(value: str | Sequence[str], *, name: str) -> tuple[str, ...]:
    columns = (value,) if isinstance(value, str) else tuple(value)
    if not columns or any(not isinstance(col, str) or not col for col in columns):
        raise ValueError(f"{name} must contain non-empty string column names.")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} cannot contain duplicate columns.")
    return columns


def _normalize_ngram_range(value: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError("ngram_range must contain exactly two integers.")
    min_n, max_n = values
    if min_n <= 0 or max_n < min_n:
        raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n.")
    return min_n, max_n


def _validate_df_threshold(value: int | float, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer count or float proportion.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} integer values must be positive.")
        return int(value)
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} float values must be in (0, 1].")
    return value


def _is_prefix(prefix: Sequence[str], whole: Sequence[str]) -> bool:
    prefix = tuple(prefix)
    whole = tuple(whole)
    return len(prefix) <= len(whole) and whole[: len(prefix)] == prefix


def _build_feature_events(
    frame: pd.DataFrame,
    *,
    field: str,
    group_by: Sequence[str],
    sequence_by: Sequence[str],
    ngram_range: tuple[int, int],
    drop_empty: bool,
    ngram_separator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return output group keys and one row per n-gram occurrence.

    The implementation uses vectorized pandas shifts within factorized sequence
    groups. The only Python-level loops are over n-gram width and the handful of
    offsets within that width, never over source rows.
    """

    group_by = list(group_by)
    sequence_by = list(sequence_by)
    group_keys = frame.loc[:, group_by].drop_duplicates(ignore_index=True)
    if group_keys.empty:
        raise ArtifactError("ArtifactCountVectorizer cannot vectorize an empty source.")

    group_index = pd.MultiIndex.from_frame(group_keys)
    source_group_index = pd.MultiIndex.from_frame(frame.loc[:, group_by])
    group_ids = group_index.get_indexer(source_group_index)
    if np.any(group_ids < 0):  # pragma: no cover - defensive
        raise ArtifactError("Failed to map source rows to ArtifactCountVectorizer groups.")

    work = frame.loc[:, [*dict.fromkeys([*sequence_by, field])]].copy()
    work["_group_id"] = group_ids.astype("int64", copy=False)
    valid = work[field].notna()
    values = work[field].astype("string")
    if drop_empty:
        valid &= values.str.len().fillna(0).gt(0)
    work = work.loc[valid].copy()
    values = values.loc[valid].astype(str)
    if work.empty:
        return group_keys, pd.DataFrame(
            columns=["_group_id", "_feature_key", "_feature_label"]
        )
    if values.str.contains(_INTERNAL_NGRAM_SEPARATOR, regex=False).any():
        raise ArtifactError(
            "ArtifactCountVectorizer field values contain TeAL's reserved internal "
            "n-gram separator U+241F."
        )
    work["_value"] = values.to_numpy()

    sequence_index = pd.MultiIndex.from_frame(work.loc[:, sequence_by])
    sequence_ids, _ = pd.factorize(sequence_index, sort=False)
    sequence_ids = pd.Series(sequence_ids, index=work.index, dtype="int64")
    base = work["_value"].astype("string")

    event_frames: list[pd.DataFrame] = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        pieces = [base.groupby(sequence_ids, sort=False).shift(-offset) for offset in range(n)]
        valid_ngram = pd.concat(pieces, axis=1).notna().all(axis=1)
        if not bool(valid_ngram.any()):
            continue
        internal = pieces[0].loc[valid_ngram].astype("string")
        label = pieces[0].loc[valid_ngram].astype("string")
        for piece in pieces[1:]:
            component = piece.loc[valid_ngram].astype("string")
            internal = internal.str.cat(component, sep=_INTERNAL_NGRAM_SEPARATOR)
            label = label.str.cat(component, sep=ngram_separator)
        events = pd.DataFrame(
            {
                "_group_id": work.loc[valid_ngram, "_group_id"].to_numpy(dtype="int64"),
                "_feature_key": internal.astype(str).to_numpy(),
                "_feature_label": label.astype(str).to_numpy(),
            }
        )
        event_frames.append(events)

    if not event_frames:
        return group_keys, pd.DataFrame(
            columns=["_group_id", "_feature_key", "_feature_label"]
        )
    events = pd.concat(event_frames, ignore_index=True)
    mapping = events.loc[:, ["_feature_key", "_feature_label"]].drop_duplicates()
    collisions = mapping["_feature_label"].duplicated(keep=False)
    if bool(collisions.any()):
        examples = mapping.loc[collisions, "_feature_label"].drop_duplicates().head(5).tolist()
        raise ArtifactError(
            "ArtifactCountVectorizer n-gram labels are ambiguous because atomic field "
            f"values contain the display separator {ngram_separator!r}. Example(s): {examples}. "
            "Choose a different ngram_separator."
        )
    return group_keys, events


def _fit_vocabulary(
    events: pd.DataFrame,
    *,
    n_groups: int,
    min_df: int | float,
    max_df: int | float,
    max_features: int | None,
) -> dict[str, int]:
    if events.empty:
        raise ArtifactError("ArtifactCountVectorizer found no countable field values.")

    counts = (
        events.groupby(["_group_id", "_feature_key", "_feature_label"], sort=False)
        .size()
        .rename("term_count")
        .reset_index()
    )
    stats = (
        counts.groupby(["_feature_key", "_feature_label"], sort=False)
        .agg(document_frequency=("_group_id", "size"), term_frequency=("term_count", "sum"))
        .reset_index()
    )
    min_count = _resolve_min_df(min_df, n_groups)
    max_count = _resolve_max_df(max_df, n_groups)
    if max_count < min_count:
        raise OperatorError(
            f"max_df corresponds to {max_count} groups, below min_df={min_count}."
        )
    kept = stats.loc[
        stats["document_frequency"].between(min_count, max_count, inclusive="both")
    ].copy()
    if kept.empty:
        raise ArtifactError(
            "ArtifactCountVectorizer has no features after min_df/max_df filtering."
        )
    if max_features is not None and len(kept) > max_features:
        kept = kept.sort_values(
            ["term_frequency", "_feature_label"], ascending=[False, True], kind="stable"
        ).head(max_features)
    labels = sorted(kept["_feature_label"].astype(str).tolist())
    return {label: index for index, label in enumerate(labels)}


def _resolve_min_df(value: int | float, n_groups: int) -> int:
    return int(value) if isinstance(value, int) else int(math.ceil(float(value) * n_groups))


def _resolve_max_df(value: int | float, n_groups: int) -> int:
    return int(value) if isinstance(value, int) else int(math.floor(float(value) * n_groups))


def _events_to_matrix(
    events: pd.DataFrame,
    *,
    n_groups: int,
    vocabulary: Mapping[str, int],
    binary: bool,
) -> sparse.csr_matrix:
    n_features = len(vocabulary)
    if events.empty:
        return sparse.csr_matrix((n_groups, n_features), dtype=np.int64)
    counts = (
        events.groupby(["_group_id", "_feature_label"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["_feature_id"] = counts["_feature_label"].map(vocabulary)
    counts = counts.loc[counts["_feature_id"].notna()].copy()
    if counts.empty:
        return sparse.csr_matrix((n_groups, n_features), dtype=np.int64)
    rows = counts["_group_id"].to_numpy(dtype="int64")
    cols = counts["_feature_id"].to_numpy(dtype="int64")
    data = (
        np.ones(len(counts), dtype="int64")
        if binary
        else counts["count"].to_numpy(dtype="int64")
    )
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_groups, n_features), dtype=np.int64
    ).tocsr()


def _normalize_vocabulary(vocabulary: Mapping[str, int]) -> dict[str, int]:
    series = pd.Series(dict(vocabulary), name="feature_id", dtype="int64")
    if series.empty:
        raise ValueError("ArtifactCountVectorizer vocabulary cannot be empty.")
    series.index = series.index.astype(str)
    if series.index.duplicated().any():
        raise ValueError("ArtifactCountVectorizer vocabulary features must be unique.")
    values = series.to_numpy(dtype="int64", copy=False)
    if not np.array_equal(np.sort(values), np.arange(len(values), dtype="int64")):
        raise ValueError(
            "ArtifactCountVectorizer vocabulary indices must be contiguous from zero."
        )
    return cast(dict[str, int], series.to_dict())


def _vocabulary_frame(vocabulary: Mapping[str, int]) -> pd.DataFrame:
    series = pd.Series(dict(vocabulary), name="feature_id", dtype="int64")
    frame = series.rename_axis("feature").reset_index()
    frame["feature"] = frame["feature"].astype(str)
    return frame.sort_values("feature_id", kind="stable").reset_index(drop=True)


def _vocabulary_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    required = {"feature", "feature_id"}
    if not required.issubset(frame.columns):
        raise OperatorError(
            f"ArtifactCountVectorizer vocabulary asset must contain {sorted(required)}."
        )
    return _normalize_vocabulary(
        dict(zip(frame["feature"].astype(str), frame["feature_id"].astype("int64"), strict=True))
    )


def _feature_names(vocabulary: Mapping[str, int]) -> list[str]:
    return _vocabulary_frame(vocabulary)["feature"].astype(str).tolist()


def _single_source(sources: Mapping[str, "BaseArtifact"]) -> "BaseArtifact":
    if set(sources) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            "ArtifactCountVectorizer requires exactly one source under "
            f"{DEFAULT_SOURCE_LABEL!r}."
        )
    return sources[DEFAULT_SOURCE_LABEL]


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if set(inputs) != {DEFAULT_SOURCE_LABEL}:
        raise OperatorError(
            "ArtifactCountVectorizer expected exactly one input under "
            f"{DEFAULT_SOURCE_LABEL!r}."
        )
    return inputs[DEFAULT_SOURCE_LABEL]


def _require_frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            "ArtifactCountVectorizer expected a pandas DataFrame packet; got "
            f"{type(value).__name__}."
        )
    return value
