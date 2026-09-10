"""Translate lexical count matrices into count distributions over dictionary values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

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
from text_analysis_lab.dictionaries import Dictionary, PolarityDictionary, ValenceDictionary
from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.source import DictionarySource
from text_analysis_lab.dictionaries.matching import category_membership, valence_vectors

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

DictionarySpec = Dictionary | PolarityDictionary | ValenceDictionary


class DictionaryTranslator(BaseTranslator):
    """Aggregate a lexical count matrix into counts over dictionary values.

    The source must contain finite, non-negative integer counts. The output keeps
    the source row key and is always a sparse matrix whose columns represent the
    dictionary's resolved values:

    * :class:`Dictionary`: dictionary keys/categories.
    * :class:`PolarityDictionary`: ``positive``, ``negative``, and ``neutral``.
    * :class:`ValenceDictionary`: the distinct numeric scores matched in the
      selected dimension.

    Output row metadata stores ``matched``, ``unmatched``, and ``total`` counts.
    User-created dictionaries are embedded in the frozen operator. Known external
    provider dictionaries default to a provider reference plus content hash, with
    ``save_dictionary=True`` available for an operator-local offline copy. Matching
    is deterministically resolved against each source vocabulary when an operation
    starts, so the same dictionary operator can be reused with another DTM.
    """

    operation_type = "translate"

    def __init__(
        self,
        dictionary: DictionarySpec,
        *,
        dimension: str | None = None,
        save_dictionary: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not isinstance(dictionary, (Dictionary, PolarityDictionary, ValenceDictionary)):
            raise TypeError(
                "DictionaryTranslator requires Dictionary, PolarityDictionary, "
                "or ValenceDictionary."
            )
        if dimension is not None and not isinstance(dictionary, ValenceDictionary):
            raise ValueError("dimension is only valid for a ValenceDictionary.")
        self.dictionary: DictionarySpec | None = dictionary
        self.dimension = _resolve_dimension(dictionary, dimension)
        self.save_dictionary = bool(save_dictionary)
        self._external_reference: dict[str, Any] | None = _dictionary_reference(dictionary)
        self._expected_dictionary_hash = _dictionary_content_hash(dictionary)
        self._source_features: tuple[str, ...] | None = None
        self._projection: sparse.csr_matrix | None = None
        self._matched_feature_mask: np.ndarray | None = None
        self._output_columns: tuple[str, ...] | None = None

    @property
    def dictionary_kind(self) -> str:
        dictionary = self._require_dictionary()
        if isinstance(dictionary, PolarityDictionary):
            return "polarity"
        if isinstance(dictionary, ValenceDictionary):
            return "valence"
        return "categorical"

    @property
    def supports_parallel_translate(self) -> bool:
        return True

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
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
                "DictionaryTranslator does not accept operation parameters; "
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
        _ = mode
        source = _single_source(sources)
        if source.artifact_type.value not in {"sparse_matrix", "dense_matrix"}:
            raise OperatorError(
                "DictionaryTranslator requires a sparse_matrix or dense_matrix source."
            )
        self._set_source_features(source.get_data_columns())
        return SourceRequest(
            artifact_type=("sparse_matrix", "dense_matrix"),
            mode="batches",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
            batch_size=request.batch_size if request.batch_size is not None else 10_000,
            form="native",
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
            raise OperatorError(f"Unsupported DictionaryTranslator mode {mode!r}.")
        packet = _single_input(inputs)
        if not isinstance(packet.data, Mapping):
            raise ArtifactError("DictionaryTranslator expected a native matrix packet.")
        info = packet.data.get("info")
        matrix = packet.data.get("matrix")
        if not isinstance(info, pd.DataFrame):
            raise ArtifactError("DictionaryTranslator native packet is missing info rows.")
        counts = _validated_count_matrix(matrix)
        if len(info) != counts.shape[0]:
            raise ArtifactError(
                f"DictionaryTranslator key row count {len(info)} != matrix row count "
                f"{counts.shape[0]}."
            )

        projection, matched_mask, columns = self._require_resolution()
        if counts.shape[1] != projection.shape[0]:
            raise ArtifactError(
                "DictionaryTranslator source feature count changed during an operation: "
                f"matrix has {counts.shape[1]}, dictionary resolution has {projection.shape[0]}."
            )

        translated = (counts @ projection).tocsr().astype(np.int64, copy=False)
        total = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.int64, copy=False)
        matched = np.asarray(
            counts @ matched_mask.astype(np.int64, copy=False).reshape(-1, 1)
        ).reshape(-1).astype(np.int64, copy=False)
        unmatched = total - matched
        if np.any(unmatched < 0):  # pragma: no cover - defensive invariant
            raise ArtifactError("Dictionary matched count exceeded total source count.")

        key_columns = [str(name) for name in packet.primary_key]
        missing_keys = [name for name in key_columns if name not in info.columns]
        if missing_keys:
            raise ArtifactError(
                f"DictionaryTranslator source batch is missing key columns {missing_keys}."
            )
        keys = info.loc[:, key_columns].reset_index(drop=True)
        metadata = pd.DataFrame(
            {
                "matched": matched,
                "unmatched": unmatched,
                "total": total,
            }
        )

        if self.dictionary_kind == "polarity":
            pole_total = np.asarray(translated.sum(axis=1)).reshape(-1)
            if not np.array_equal(pole_total.astype(np.int64), matched):
                raise ArtifactError(
                    "Polarity translation invariant failed: positive + negative + "
                    "neutral must equal matched."
                )

        return BatchResult(
            outputs={
                DEFAULT_OUTPUT_LABEL: {
                    "keys": keys,
                    "metadata": metadata,
                    "data": {"values": translated, "columns": columns},
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

    def make_translate_worker(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> "DictionaryTranslator":
        _ = request
        if mode != "translate":
            raise OperatorError("DictionaryTranslator workers support translate mode only.")
        worker = DictionaryTranslator(
            _clone_dictionary(self._require_dictionary()),
            dimension=self.dimension,
            save_dictionary=self.save_dictionary,
        )
        worker._set_source_features(self._require_source_features())
        return worker

    def __getstate__(self) -> dict[str, Any]:
        """Make process workers independent of non-pickleable mapping proxies."""
        return {
            "dictionary": _serialize_dictionary(self._require_dictionary()),
            "dimension": self.dimension,
            "save_dictionary": self.save_dictionary,
            "source_features": (
                None if self._source_features is None else list(self._source_features)
            ),
            "operator_id": self.operator_id,
            "is_frozen": self.is_frozen,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        raw_dictionary = state.get("dictionary")
        if not isinstance(raw_dictionary, Mapping):
            raise OperatorError("DictionaryTranslator worker state is missing dictionary data.")
        restored = DictionaryTranslator(
            _deserialize_dictionary(raw_dictionary),
            dimension=cast(str | None, state.get("dimension")),
            save_dictionary=bool(state.get("save_dictionary", False)),
        )
        self.__dict__.update(restored.__dict__)
        features = state.get("source_features")
        if features is not None:
            self._set_source_features(cast(Sequence[str], features))
        self.operator_id = cast(str | None, state.get("operator_id"))
        self.is_frozen = bool(state.get("is_frozen", False))

    def to_json_state(self) -> dict[str, Any]:
        dictionary = self._require_dictionary()
        reference = _dictionary_reference(dictionary)
        if reference is None:
            stored = {"storage": "embedded", "dictionary": _serialize_dictionary(dictionary)}
        else:
            stored = {"storage": "external", "reference": reference}
        return {
            "dictionary": stored,
            "dictionary_hash": _dictionary_content_hash(dictionary),
            "dimension": self.dimension,
            "save_dictionary": self.save_dictionary,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> "DictionaryTranslator":
        raw_storage = state.get("dictionary")
        if not isinstance(raw_storage, Mapping):
            raise OperatorError("DictionaryTranslator state is missing dictionary data.")
        storage = str(raw_storage.get("storage", ""))
        dimension = cast(str | None, state.get("dimension"))
        save_dictionary = bool(state.get("save_dictionary", False))
        expected_hash = str(state.get("dictionary_hash", ""))
        if storage == "embedded":
            raw_dictionary = raw_storage.get("dictionary")
            if not isinstance(raw_dictionary, Mapping):
                raise OperatorError("Embedded DictionaryTranslator state is missing dictionary data.")
            obj = cls(
                _deserialize_dictionary(raw_dictionary),
                dimension=dimension,
                save_dictionary=save_dictionary,
            )
            if expected_hash and _dictionary_content_hash(obj._require_dictionary()) != expected_hash:
                raise OperatorError("Embedded dictionary content hash does not match operator state.")
            return obj
        if storage != "external":
            raise OperatorError(f"Unknown DictionaryTranslator dictionary storage {storage!r}.")
        reference = raw_storage.get("reference")
        if not isinstance(reference, Mapping):
            raise OperatorError("External DictionaryTranslator state is missing its provider reference.")

        # BaseOperator.load_from_dir() calls load_assets() immediately after this
        # method. Keep the external dictionary unresolved until then so an
        # operator-local saved copy can be preferred over provider/cache access.
        obj = cls.__new__(cls)
        BaseTranslator.__init__(obj, operator_id=None)
        obj.dictionary = None
        obj.dimension = dimension
        obj.save_dictionary = save_dictionary
        obj._external_reference = dict(reference)
        obj._expected_dictionary_hash = expected_hash
        obj._source_features = None
        obj._projection = None
        obj._matched_feature_mask = None
        obj._output_columns = None
        return obj

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        if not self.save_dictionary or self._external_reference is None:
            return {}
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / "dictionary.json"
        path.write_text(
            json.dumps(_serialize_dictionary(self._require_dictionary()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "dictionary_file": path.name,
            "dictionary_hash": self._expected_dictionary_hash,
        }

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        filename = manifest.get("dictionary_file") if manifest else None
        if isinstance(filename, str) and filename:
            raw = json.loads((assets_dir / filename).read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise OperatorError("Saved dictionary asset must contain a JSON object.")
            dictionary = _deserialize_dictionary(raw)
            self._install_loaded_dictionary(dictionary, source="operator-local asset")
            return
        if self.dictionary is not None:
            return
        if self._external_reference is None:
            raise OperatorError("DictionaryTranslator has neither embedded nor external dictionary state.")
        dictionary = _load_dictionary_reference(self._external_reference)
        self._install_loaded_dictionary(dictionary, source="external provider/cache")

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
        state["source_features"] = list(self._require_source_features())
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
    ) -> "DictionaryTranslator":
        _ = mode, route
        state = json.loads((intermediate_dir / "state.json").read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise OperatorError("DictionaryTranslator intermediate state must be a mapping.")
        obj = cls.from_json_state(state)
        assets = state.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("DictionaryTranslator intermediate assets must be a mapping.")
        obj.load_assets(intermediate_dir, assets)
        features = state.get("source_features")
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
            raise OperatorError(
                "DictionaryTranslator intermediate state is missing source_features."
            )
        obj._set_source_features([str(value) for value in features])
        obj.operator_id = str(operator_id)
        return obj

    def _install_loaded_dictionary(self, dictionary: DictionarySpec, *, source: str) -> None:
        expected = self._expected_dictionary_hash
        observed = _dictionary_content_hash(dictionary)
        if expected and observed != expected:
            raise OperatorError(
                "Reloaded external dictionary does not match the content hash frozen in "
                f"this operator ({source}). Reinstall the original resource or use an "
                "operator snapshot created with save_dictionary=True."
            )
        self.dictionary = dictionary
        self._external_reference = _dictionary_reference(dictionary) or self._external_reference

    def _set_source_features(self, features: Sequence[str]) -> None:
        dictionary = self._require_dictionary()
        self._source_features = tuple(str(value) for value in features)
        projection, matched_mask, columns = _resolve_dictionary_projection(
            list(self._source_features),
            dictionary,
            dimension=self.dimension,
        )
        self._projection = projection
        self._matched_feature_mask = matched_mask
        self._output_columns = columns

    def _require_dictionary(self) -> DictionarySpec:
        if self.dictionary is None:
            raise OperatorError(
                "DictionaryTranslator external dictionary has not been restored. "
                "Load the operator from its project snapshot or reinstall the provider resource."
            )
        return self.dictionary

    def _require_source_features(self) -> tuple[str, ...]:
        if self._source_features is None:
            raise OperatorError("DictionaryTranslator has no resolved source vocabulary.")
        return self._source_features

    def _require_resolution(self) -> tuple[sparse.csr_matrix, np.ndarray, tuple[str, ...]]:
        if (
            self._projection is None
            or self._matched_feature_mask is None
            or self._output_columns is None
        ):
            raise OperatorError("DictionaryTranslator source vocabulary has not been resolved.")
        return self._projection, self._matched_feature_mask, self._output_columns


def _resolve_dimension(dictionary: DictionarySpec, dimension: str | None) -> str | None:
    if not isinstance(dictionary, ValenceDictionary):
        return None
    if dimension is not None:
        value = str(dimension)
        if value not in dictionary.values:
            raise ValueError(
                f"Unknown valence dimension {value!r}; available: {list(dictionary.dimensions)}."
            )
        return value
    if len(dictionary.dimensions) != 1:
        raise ValueError(
            "dimension is required when a ValenceDictionary has multiple dimensions; "
            f"available: {list(dictionary.dimensions)}."
        )
    return dictionary.dimensions[0]


def _resolve_dictionary_projection(
    features: list[str],
    dictionary: DictionarySpec,
    *,
    dimension: str | None,
) -> tuple[sparse.csr_matrix, np.ndarray, tuple[str, ...]]:
    if isinstance(dictionary, PolarityDictionary):
        return _resolve_polarity_projection(features, dictionary)
    if isinstance(dictionary, ValenceDictionary):
        assert dimension is not None
        return _resolve_valence_projection(features, dictionary, dimension=dimension)

    keys, membership = category_membership(features, dictionary)
    matched = np.asarray(membership.sum(axis=1)).reshape(-1) > 0
    return membership.astype(np.int64), matched, tuple(str(key) for key in keys)


def _resolve_polarity_projection(
    features: list[str], dictionary: PolarityDictionary
) -> tuple[sparse.csr_matrix, np.ndarray, tuple[str, ...]]:
    keys, membership = category_membership(features, dictionary.dictionary)
    key_index = {key: index for index, key in enumerate(keys)}

    positive = _pole_mask(membership, [key_index[key] for key in dictionary.positive])
    negative = _pole_mask(membership, [key_index[key] for key in dictionary.negative])
    neutral = _pole_mask(membership, [key_index[key] for key in dictionary.neutral])
    assignment_count = positive.astype(np.int8) + negative.astype(np.int8) + neutral.astype(np.int8)
    conflict_indices = np.flatnonzero(assignment_count > 1)
    if conflict_indices.size:
        examples = [features[int(index)] for index in conflict_indices[:10]]
        raise OperatorError(
            "Polarity dictionary assigns source features to multiple different poles; "
            f"examples: {examples}."
        )

    masks = (positive, negative, neutral)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for column_index, mask in enumerate(masks):
        indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if indices.size:
            rows.append(indices)
            cols.append(np.full(indices.size, column_index, dtype=np.int64))
    if rows:
        row_index = np.concatenate(rows)
        col_index = np.concatenate(cols)
        data = np.ones(row_index.size, dtype=np.int64)
        projection = sparse.coo_matrix(
            (data, (row_index, col_index)), shape=(len(features), 3)
        ).tocsr()
    else:
        projection = sparse.csr_matrix((len(features), 3), dtype=np.int64)
    matched = assignment_count > 0
    return projection, matched, ("positive", "negative", "neutral")


def _resolve_valence_projection(
    features: list[str],
    dictionary: ValenceDictionary,
    *,
    dimension: str,
) -> tuple[sparse.csr_matrix, np.ndarray, tuple[str, ...]]:
    scores, matched = valence_vectors(features, dictionary)[dimension]
    if not np.any(matched):
        return (
            sparse.csr_matrix((len(features), 0), dtype=np.int64),
            matched,
            (),
        )

    normalized_scores = scores.copy()
    normalized_scores[np.isclose(normalized_scores, 0.0)] = 0.0
    values = np.unique(normalized_scores[matched])
    values.sort()
    value_to_column = {float(value): index for index, value in enumerate(values.tolist())}
    rows = np.flatnonzero(matched).astype(np.int64, copy=False)
    cols = np.asarray(
        [value_to_column[float(normalized_scores[int(index)])] for index in rows],
        dtype=np.int64,
    )
    data = np.ones(rows.size, dtype=np.int64)
    projection = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(features), len(values))
    ).tocsr()
    columns = tuple(_format_score(float(value)) for value in values.tolist())
    if len(set(columns)) != len(columns):  # pragma: no cover - defensive
        raise OperatorError("Distinct valence values collapsed to duplicate column labels.")
    return projection, matched, columns


def _pole_mask(membership: sparse.csr_matrix, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(membership.shape[0], dtype=bool)
    return np.asarray(membership[:, indices].sum(axis=1)).reshape(-1) > 0


def _format_score(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("Valence dictionary scores must be finite for translation.")
    if value == 0.0:
        value = 0.0
    return format(float(value), ".17g")


def _validated_count_matrix(value: Any) -> sparse.csr_matrix:
    if sparse.issparse(value):
        matrix = value.tocsr(copy=True)
        observed = np.asarray(matrix.data, dtype=float)
        shape = matrix.shape
    else:
        dense = np.asarray(value)
        if dense.ndim != 2:
            raise ArtifactError("DictionaryTranslator source matrix must be two-dimensional.")
        observed = np.asarray(dense, dtype=float).reshape(-1)
        matrix = sparse.csr_matrix(dense)
        shape = dense.shape

    if len(shape) != 2:
        raise ArtifactError("DictionaryTranslator source matrix must be two-dimensional.")
    if observed.size:
        if not np.all(np.isfinite(observed)):
            raise ArtifactError("DictionaryTranslator requires finite count values.")
        if np.any(observed < 0):
            raise ArtifactError("DictionaryTranslator requires non-negative count values.")
        rounded = np.rint(observed)
        if not np.allclose(observed, rounded, rtol=0.0, atol=1e-9):
            raise ArtifactError(
                "DictionaryTranslator requires integer-valued counts; weighted matrices "
                "such as TF-IDF are not accepted."
            )
    matrix.data = np.rint(np.asarray(matrix.data, dtype=float)).astype(np.int64)
    matrix.eliminate_zeros()
    return matrix.astype(np.int64, copy=False)


def _serialize_dictionary(dictionary: DictionarySpec) -> dict[str, Any]:
    if isinstance(dictionary, PolarityDictionary):
        return {
            "kind": "polarity",
            "dictionary": _serialize_dictionary(dictionary.dictionary),
            "positive": list(dictionary.positive),
            "negative": list(dictionary.negative),
            "neutral": list(dictionary.neutral),
        }
    if isinstance(dictionary, ValenceDictionary):
        return {
            "kind": "valence",
            "values": {
                dimension: {pattern: float(value) for pattern, value in scores.items()}
                for dimension, scores in dictionary.values.items()
            },
            "valuetype": dictionary.valuetype,
            "case_sensitive": dictionary.case_sensitive,
            "name": dictionary.name,
            "provenance": _serialize_provenance(dictionary.provenance),
            "source": _serialize_dictionary_source(dictionary.source),
        }
    return {
        "kind": "categorical",
        "entries": {key: list(patterns) for key, patterns in dictionary.entries.items()},
        "valuetype": dictionary.valuetype,
        "case_sensitive": dictionary.case_sensitive,
        "name": dictionary.name,
        "provenance": _serialize_provenance(dictionary.provenance),
        "source": _serialize_dictionary_source(dictionary.source),
    }


def _deserialize_valuetype(state: Mapping[str, Any], *, default: str) -> str:
    """Read current valuetype while accepting Round 27.4 serialized state."""
    valuetype = str(state.get("valuetype", default))
    if valuetype == "glob" and bool(state.get("non_whitespace_glob", False)):
        return "non_whitespace_glob"
    return valuetype


def _deserialize_dictionary(state: Mapping[str, Any]) -> DictionarySpec:
    kind = str(state.get("kind", ""))
    if kind == "categorical":
        entries = state.get("entries")
        if not isinstance(entries, Mapping):
            raise OperatorError("Serialized categorical dictionary is missing entries.")
        return Dictionary(
            cast(Mapping[str, Sequence[str]], entries),
            valuetype=_deserialize_valuetype(state, default="glob"),
            case_sensitive=bool(state.get("case_sensitive", False)),
            name=cast(str | None, state.get("name")),
            provenance=_deserialize_provenance(state.get("provenance")),
            source=_deserialize_dictionary_source(state.get("source")),
        )
    if kind == "polarity":
        base_state = state.get("dictionary")
        if not isinstance(base_state, Mapping):
            raise OperatorError("Serialized polarity dictionary is missing its base dictionary.")
        base = _deserialize_dictionary(base_state)
        if not isinstance(base, Dictionary):
            raise OperatorError("Serialized polarity dictionary base must be categorical.")
        return PolarityDictionary(
            base,
            positive=cast(Sequence[str], state.get("positive", ())),
            negative=cast(Sequence[str], state.get("negative", ())),
            neutral=cast(Sequence[str], state.get("neutral", ())) or None,
        )
    if kind == "valence":
        values = state.get("values")
        if not isinstance(values, Mapping):
            raise OperatorError("Serialized valence dictionary is missing values.")
        return ValenceDictionary(
            cast(Mapping[str, Any], values),
            valuetype=_deserialize_valuetype(state, default="fixed"),
            case_sensitive=bool(state.get("case_sensitive", False)),
            name=cast(str | None, state.get("name")),
            provenance=_deserialize_provenance(state.get("provenance")),
            source=_deserialize_dictionary_source(state.get("source")),
        )
    raise OperatorError(f"Unknown serialized dictionary kind {kind!r}.")


def _dictionary_reference(dictionary: DictionarySpec) -> dict[str, Any] | None:
    source = dictionary.source
    if source is None:
        return None
    reference: dict[str, Any] = {
        "source": source.to_dict(),
        "kind": (
            "polarity"
            if isinstance(dictionary, PolarityDictionary)
            else "valence"
            if isinstance(dictionary, ValenceDictionary)
            else "categorical"
        ),
    }
    if isinstance(dictionary, PolarityDictionary):
        reference.update(
            {
                "positive": list(dictionary.positive),
                "negative": list(dictionary.negative),
                "neutral": list(dictionary.neutral),
            }
        )
    return reference


def _load_dictionary_reference(reference: Mapping[str, Any]) -> DictionarySpec:
    raw_source = reference.get("source")
    source = DictionarySource.from_value(
        cast(Mapping[str, Any] | None, raw_source) if isinstance(raw_source, Mapping) else None
    )
    if source is None:
        raise OperatorError("External dictionary reference is missing provider/source metadata.")
    from text_analysis_lab.dictionaries.providers.registry import load_dictionary_source

    loaded = load_dictionary_source(source)
    kind = str(reference.get("kind", ""))
    if kind == "polarity":
        if isinstance(loaded, PolarityDictionary):
            base = loaded.dictionary
        elif isinstance(loaded, Dictionary):
            base = loaded
        else:
            raise OperatorError(
                f"External source {source.provider}/{source.resource} did not reload "
                "as a categorical/polarity dictionary."
            )
        return PolarityDictionary(
            base,
            positive=cast(Sequence[str], reference.get("positive", ())),
            negative=cast(Sequence[str], reference.get("negative", ())),
            neutral=cast(Sequence[str], reference.get("neutral", ())) or None,
        )
    if kind == "valence":
        if not isinstance(loaded, ValenceDictionary):
            raise OperatorError(
                f"External source {source.provider}/{source.resource} did not reload "
                "as a valence dictionary."
            )
        return loaded
    if kind == "categorical":
        if isinstance(loaded, PolarityDictionary):
            return loaded.dictionary
        if isinstance(loaded, Dictionary):
            return loaded
        raise OperatorError(
            f"External source {source.provider}/{source.resource} did not reload "
            "as a categorical dictionary."
        )
    raise OperatorError(f"Unknown external dictionary reference kind {kind!r}.")


def _dictionary_content_hash(dictionary: DictionarySpec) -> str:
    payload = _dictionary_semantic_payload(dictionary)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dictionary_semantic_payload(dictionary: DictionarySpec) -> dict[str, Any]:
    if isinstance(dictionary, PolarityDictionary):
        return {
            "kind": "polarity",
            "dictionary": _dictionary_semantic_payload(dictionary.dictionary),
            "positive": list(dictionary.positive),
            "negative": list(dictionary.negative),
            "neutral": list(dictionary.neutral),
        }
    if isinstance(dictionary, ValenceDictionary):
        return {
            "kind": "valence",
            "values": {
                dimension: {pattern: float(value) for pattern, value in scores.items()}
                for dimension, scores in dictionary.values.items()
            },
            "valuetype": dictionary.valuetype,
            "case_sensitive": dictionary.case_sensitive,
        }
    return {
        "kind": "categorical",
        "entries": {key: list(patterns) for key, patterns in dictionary.entries.items()},
        "valuetype": dictionary.valuetype,
        "case_sensitive": dictionary.case_sensitive,
    }


def _clone_dictionary(dictionary: DictionarySpec) -> DictionarySpec:
    return _deserialize_dictionary(_serialize_dictionary(dictionary))


def _serialize_dictionary_source(
    source: DictionarySource | None,
) -> dict[str, Any] | None:
    return None if source is None else source.to_dict()


def _deserialize_dictionary_source(value: Any) -> DictionarySource | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise OperatorError("Serialized dictionary source must be a mapping or null.")
    return DictionarySource.from_value(cast(Mapping[str, Any], value))


def _serialize_provenance(
    provenance: DictionaryProvenance | None,
) -> dict[str, str | None] | None:
    return None if provenance is None else provenance.to_dict()


def _deserialize_provenance(value: Any) -> DictionaryProvenance | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise OperatorError("Serialized dictionary provenance must be a mapping or null.")
    return DictionaryProvenance.from_value(cast(Mapping[str, Any], value))


def _single_source(sources: Mapping[str, "BaseArtifact"]) -> "BaseArtifact":
    if len(sources) != 1:
        raise OperatorError(
            f"DictionaryTranslator requires exactly one source; got {list(sources)}."
        )
    return next(iter(sources.values()))


def _single_input(inputs: Mapping[str, InputBatch]) -> InputBatch:
    if len(inputs) != 1:
        raise OperatorError(
            f"DictionaryTranslator requires exactly one input packet; got {list(inputs)}."
        )
    return next(iter(inputs.values()))
