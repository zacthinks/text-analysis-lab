"""Ephemeral row-wise rankings of features in sparse or dense matrix artifacts.

FREX follows the unshrunk empirical-CDF frequency/exclusivity definition used
by stm::calcfrex(logbeta, wordcounts=NULL). It deliberately does not claim
equivalence to STM's James-Stein-smoothed labelTopics() output.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from text_analysis_lab.analysis._matrix_utils import (
    is_sparse_matrix,
    require_matrix_artifact,
)
from text_analysis_lab.analysis.feature_summary import feature_summary

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

FeatureScorer = Callable[..., Any]
BUILTIN_METHODS = frozenset({
    "value", "absolute", "probability", "term_frequency",
    "document_frequency", "lift", "frex",
})
CORPUS_METHODS = frozenset({"term_frequency", "document_frequency", "lift"})


def _feature_axis(artifact: BaseArtifact, *, label_field: str) -> tuple[list[str], list[Any]]:
    frame = artifact.get_feature_metadata()
    if not isinstance(frame, pd.DataFrame) or "column" not in frame:
        raise ValueError("top_features() requires Feature Metadata with a canonical 'column' field.")
    names = frame["column"].tolist()
    n = len(artifact.get_data_columns())
    if len(names) != n or "column_index" not in frame or frame["column_index"].tolist() != list(range(n)):
        raise ValueError("Matrix Feature Metadata must match the ordered data-column axis.")
    if any(pd.isna(value) or not isinstance(value, str) or not value for value in names):
        raise ValueError("Canonical feature names must be nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Canonical feature names must be unique for reference alignment.")
    if label_field not in frame:
        raise ValueError(f"Feature Metadata has no label_field={label_field!r}; available: {list(frame)}.")
    labels = frame[label_field].tolist()
    if any(pd.isna(label) or str(label) == "" for label in labels) or len(set(map(str, labels))) != len(labels):
        raise ValueError(f"Feature labels in {label_field!r} must be nonmissing and unique.")
    return names, labels


def _valid_probability(values: np.ndarray, *, position: Any, method: str) -> np.ndarray:
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{method} requires finite nonnegative values at matrix row {position}.")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{method} requires positive row mass at matrix row {position}.")
    return values / total


def _dense_row(matrix: Any, row_index: int, width: int) -> np.ndarray:
    if is_sparse_matrix(matrix):
        values = matrix.getrow(row_index).toarray().ravel() if hasattr(matrix, "getrow") else matrix[row_index].toarray().ravel()
    else:
        values = np.asarray(matrix[row_index]).ravel()
    values = np.asarray(values, dtype=float)
    if values.size != width or not np.all(np.isfinite(values)):
        raise ValueError("Matrix row has wrong width or contains non-finite values.")
    return values


def _percentile(values: np.ndarray) -> np.ndarray:
    """Empirical CDF within a row: ties receive the fraction <= the value."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    end = np.searchsorted(sorted_values, sorted_values, side="right")
    result = np.empty(len(values), dtype=float)
    result[order] = end / float(len(values))
    return result


def _custom_call(scorer: FeatureScorer, context: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Any:
    available = {**context, **kwargs}
    try:
        signature = inspect.signature(scorer)
    except (TypeError, ValueError):
        return scorer(**available)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return scorer(**available)
    args: list[Any] = []
    named: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if parameter.name in available:
            value = available[parameter.name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise ValueError(
                f"Custom top_features scorer requests unavailable parameter {parameter.name!r}; "
                f"available: {sorted(available)}."
            )
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            named[parameter.name] = value
    return scorer(*args, **named)


def _named_custom_parameters(custom: Mapping[str, FeatureScorer]) -> set[str]:
    """Find explicitly requested context fields without triggering computation.

    A bare **kwargs parameter does not opt into potentially expensive corpus
    aggregation. Scorers must name corpus statistics to request them.
    """
    named: set[str] = set()
    for scorer in custom.values():
        try:
            signature = inspect.signature(scorer)
        except (TypeError, ValueError):
            continue
        named.update(
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        )
    return named


def top_features(
    artifact: BaseArtifact,
    *,
    n: int = 10,
    methods: Sequence[str] | str = ("value",),
    reference: BaseArtifact | str | None = None,
    frex_weight: float = 0.7,
    label_field: str = "column",
    custom: Mapping[str, FeatureScorer] | None = None,
    batch_size: int = 1_000,
    **custom_kwargs: Any,
) -> pd.DataFrame:
    """Return a wide per-row/per-rank DataFrame of independently ranked features.

    The weight in FREX is assigned to exclusivity, not frequency:
    1 / (frex_weight / E + (1-frex_weight) / F).
    F and E are empirical CDF percentiles of row-normalized frequencies and
    cross-row exclusivity. No rare-word James-Stein shrinkage is applied.
    Across DTM rows this compares *documents*, not inferred topics.

    Corpus statistics use the explicitly provided reference matrix. When
    omitted, term_frequency/document_frequency treat the subject itself as
    the corpus, which is appropriate only if its rows are observations.
    Lift always requires a reference. Canonical columns, never display
    labels, determine reference alignment. A zero-background feature with
    positive subject mass raises instead of returning infinite lift.

    Named custom scorers can request values, probabilities, feature_names,
    feature_indices, row_key, reference, term_frequency, document_frequency,
    background_probability and caller-supplied keyword arguments. A scorer
    returns one finite numeric score for each feature in the current row.
    Valid row probabilities are available even for custom-only rankings.
    A named probabilities parameter rejects invalid distributions.

    Corpus statistics are computed only when a built-in method or an explicitly
    named custom parameter requests them. A scorer accepting only **kwargs
    receives None for unrequested corpus statistics; name those parameters
    explicitly to request the scan. Custom background_probability needs
    an explicit reference matrix (as does built-in lift).
    """
    require_matrix_artifact(artifact, "top_features()")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not np.isfinite(frex_weight) or not 0 <= frex_weight <= 1:
        raise ValueError("frex_weight must be between 0 and 1 (exclusivity weight).")
    if isinstance(methods, str):
        selected = [methods]
    else:
        selected = list(methods)
    if not selected and not custom:
        raise ValueError("Choose at least one built-in or custom scoring method.")
    if len(selected) != len(set(selected)) or any(method not in BUILTIN_METHODS for method in selected):
        raise ValueError(f"methods must be unique members of {sorted(BUILTIN_METHODS)}.")
    custom = {} if custom is None else dict(custom)
    for name, scorer in custom.items():
        if not isinstance(name, str) or not name or not callable(scorer):
            raise ValueError("Custom scorers require nonempty string names and callable values.")
    custom_fields = _named_custom_parameters(custom)
    names, labels = _feature_axis(artifact, label_field=label_field)
    width = len(names)
    primary_key = [str(value) for value in artifact.primary_key]
    requested = [*selected, *custom]
    result_columns = [*primary_key, "rank"]
    for method in requested:
        result_columns.extend((f"{method}_ranking", f"{method}_score"))
    if len(set(result_columns)) != len(result_columns) or set(requested) & BUILTIN_METHODS - set(selected):
        raise ValueError("Built-in/custom names or generated ranking columns collide with result columns.")
    if not width or n == 0:
        return pd.DataFrame(columns=result_columns)

    if isinstance(reference, str):
        reference = artifact.project.get_artifact(reference)
    if reference is not None:
        require_matrix_artifact(reference, "top_features(reference)")
        if reference.project != artifact.project:
            raise ValueError("reference must belong to the same Project as the subject.")
        reference_names, _ = _feature_axis(reference, label_field="column")
        if len(reference_names) != width or set(reference_names) != set(names):
            raise ValueError("reference must have the exact same canonical feature set.")
    if "lift" in selected and reference is None:
        raise ValueError("lift requires an explicit corpus count matrix via reference=.")
    if "background_probability" in custom_fields and reference is None:
        raise ValueError(
            "Custom background_probability requires an explicit corpus count "
            "matrix via reference=."
        )
    needs_background = "lift" in selected or "background_probability" in custom_fields
    needs_counts = (
        bool(CORPUS_METHODS & set(selected))
        or bool({"term_frequency", "document_frequency"} & custom_fields)
        or needs_background
    )
    counts_source = reference if reference is not None else artifact
    tf: np.ndarray | None = None
    df: np.ndarray | None = None
    background: np.ndarray | None = None
    if needs_counts:
        summary = feature_summary(counts_source, batch_size=int(batch_size))
        if summary.empty or counts_source.n_rows == 0:
            raise ValueError("Corpus count matrix must contain at least one row and feature.")
        if not np.all(np.isfinite(summary[["sum", "min", "max"]].to_numpy(dtype=float))):
            raise ValueError("Corpus count matrix must contain finite counts.")
        if np.any(summary["min"].to_numpy(dtype=float) < 0):
            raise ValueError("Corpus counts must be nonnegative.")
        by_name = {name: index for index, name in enumerate(counts_source.get_data_columns())}
        aligned = [by_name[name] for name in names]
        tf = summary["sum"].to_numpy(dtype=float)[aligned]
        df = summary["nonzero_rows"].to_numpy(dtype=float)[aligned]
        if needs_background:
            total = float(np.sum(tf))
            if not np.isfinite(total) or total <= 0:
                raise ValueError(
                    "Background prevalence requires positive total corpus term frequency."
                )
            background = tf / total

    # FREX exclusivity requires a cross-row denominator: one streaming pass,
    # followed by the independent row ranking pass. Memory is O(features).
    exclusivity_denominator: np.ndarray | None = None
    if "frex" in selected:
        exclusivity_denominator = np.zeros(width, dtype=float)
        for batch in artifact.iter_batches(
            batch_size=int(batch_size), key_columns=False, data_columns=True,
            metadata_columns=False, metadata_mode="none", form="native", include_position=True,
        ):
            matrix = batch["matrix"]
            for i in range(matrix.shape[0]):
                position = batch["info"]["_position"].iloc[i]
                values = _dense_row(matrix, i, width)
                exclusivity_denominator += _valid_probability(values, position=position, method="frex")
        if not np.all(np.isfinite(exclusivity_denominator)):
            raise ValueError("FREX cross-row word mass is non-finite.")

    rows: list[dict[str, Any]] = []
    top_n = min(int(n), width)
    for batch in artifact.iter_batches(
        batch_size=int(batch_size), key_columns=True, data_columns=True,
        metadata_columns=False, metadata_mode="none", form="native", include_position=True,
    ):
        info = batch["info"].reset_index(drop=True)
        matrix = batch["matrix"]
        if matrix.shape[1] != width or len(info) != matrix.shape[0]:
            raise ValueError("Matrix batch width or row-key count differs from Feature Metadata.")
        for i in range(matrix.shape[0]):
            record = {key: info.iloc[i][key] for key in primary_key}
            position = info["_position"].iloc[i]
            values = _dense_row(matrix, i, width)
            probability: np.ndarray | None = None
            if set(selected) & {"probability", "lift", "frex"} or "probabilities" in custom_fields:
                probability = _valid_probability(
                    values, position=position, method="probability/lift/frex/custom"
                )
            elif custom and np.all(values >= 0):
                # Cheap optional context for **kwargs scorers; signed/zero rows
                # remain valid for custom formulas using raw values only.
                row_sum = float(np.sum(values))
                if np.isfinite(row_sum) and row_sum > 0:
                    probability = values / row_sum
            scores: dict[str, np.ndarray] = {}
            if "value" in selected:
                scores["value"] = values
            if "absolute" in selected:
                scores["absolute"] = np.abs(values)
            if "probability" in selected:
                scores["probability"] = probability
            if "term_frequency" in selected:
                scores["term_frequency"] = tf
            if "document_frequency" in selected:
                scores["document_frequency"] = df
            if "lift" in selected:
                assert probability is not None and background is not None
                if np.any((background == 0) & (probability > 0)):
                    raise ValueError(f"lift: positive feature mass absent from reference at row {position}.")
                scores["lift"] = np.divide(
                    probability, background, out=np.zeros(width, dtype=float), where=background > 0
                )
            if "frex" in selected:
                assert probability is not None and exclusivity_denominator is not None
                exclusivity = np.divide(
                    probability, exclusivity_denominator,
                    out=np.zeros(width, dtype=float), where=exclusivity_denominator > 0,
                )
                frequency_rank = _percentile(probability)
                exclusivity_rank = _percentile(exclusivity)
                scores["frex"] = 1.0 / (
                    frex_weight / exclusivity_rank + (1.0 - frex_weight) / frequency_rank
                )
            context = {
                "values": values, "probabilities": probability,
                "feature_names": tuple(names),
                "feature_indices": np.arange(width, dtype=int),
                "row_key": dict(record), "reference": reference,
                "term_frequency": tf, "document_frequency": df,
                "background_probability": background,
            }
            for name, scorer in custom.items():
                try:
                    raw = _custom_call(scorer, context, custom_kwargs)
                    score = np.asarray(raw, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Custom scorer {name!r} failed at row {position}: {exc}") from exc
                if score.ndim != 1 or score.size != width or not np.all(np.isfinite(score)):
                    raise ValueError(f"Custom scorer {name!r} must return {width} finite numeric scores.")
                scores[name] = score
            rankings = {
                name: np.lexsort((np.arange(width), -scores[name]))[:top_n]
                for name in requested
            }
            for rank in range(top_n):
                out = {**record, "rank": rank + 1}
                for method in requested:
                    index = int(rankings[method][rank])
                    out[f"{method}_ranking"] = labels[index]
                    out[f"{method}_score"] = float(scores[method][index])
                rows.append(out)
    return pd.DataFrame(rows, columns=result_columns)
