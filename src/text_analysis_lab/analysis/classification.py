"""Design-aware binary classification evaluation over stable TeAL keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from text_analysis_lab.core.errors import ArtifactError

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "prevalence")


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Ephemeral raw and optional design-weighted audit evaluation."""

    raw_metrics: dict[str, float]
    raw_confusion: pd.DataFrame
    weighted_metrics: dict[str, float] | None = None
    weighted_confusion: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return tidy metric rows for display or downstream plotting."""
        rows = [
            {"scope": "raw_audit", "metric": metric, "value": self.raw_metrics[metric]}
            for metric in _METRICS
        ]
        if self.weighted_metrics is not None:
            rows.extend(
                {
                    "scope": "design_weighted",
                    "metric": metric,
                    "value": self.weighted_metrics[metric],
                }
                for metric in _METRICS
            )
        return pd.DataFrame(rows, columns=["scope", "metric", "value"])

    def confusion_matrix(self, *, weighted: bool = False) -> pd.DataFrame:
        """Return raw counts or inverse-probability weighted estimated totals."""
        if weighted:
            if self.weighted_confusion is None:
                raise ValueError(
                    "This evaluation has no design-weighted confusion matrix."
                )
            return self.weighted_confusion.copy()
        return self.raw_confusion.copy()


def classification(
    predictions: BaseArtifact,
    *,
    gold: BaseArtifact | str,
    pi: BaseArtifact | str | None = None,
    prediction_field: str = "prediction",
    gold_field: str = "label",
    pi_field: str = "pi",
) -> ClassificationEvaluation:
    """Evaluate binary predictions against human audit labels by stable key.

    ``predictions`` may cover a superset of the audit keys. ``gold`` defines the
    audit evaluation set. Field names are configurable so one multi-code Focus
    Coder export can be evaluated directly against separate prediction artifacts.
    If ``pi`` is supplied its key set must exactly equal ``gold`` and
    inverse-probability weights ``1/pi_i`` are used for finite-corpus estimated
    totals and ratios.
    """
    project = predictions.project
    gold_artifact = project.get_artifact(gold)
    pi_artifact = None if pi is None else project.get_artifact(pi)

    _require_fields(predictions, prediction_field, role="predictions")
    _require_fields(gold_artifact, gold_field, role="gold")
    keys = tuple(str(value) for value in gold_artifact.primary_key)
    if not keys:
        raise ArtifactError("Gold artifact has no primary key.")
    if tuple(str(value) for value in predictions.primary_key) != keys:
        raise ArtifactError(
            "Prediction and gold artifacts must use the same primary-key columns."
        )

    pred_col = "__teal_prediction__"
    gold_col = "__teal_gold__"
    pi_col = "__teal_pi__"
    pred = _frame(predictions, prediction_field).rename(
        columns={prediction_field: pred_col}
    )
    gold_frame = _frame(gold_artifact, gold_field).rename(
        columns={gold_field: gold_col}
    )
    _validate_unique(pred, keys, role="predictions")
    _validate_unique(gold_frame, keys, role="gold")

    merged = gold_frame.merge(
        pred,
        on=list(keys),
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if (merged["_merge"] != "both").any():
        missing = int((merged["_merge"] != "both").sum())
        raise ArtifactError(f"Predictions are missing {missing} gold/audit key(s).")
    merged = merged.drop(columns=["_merge"])
    y_true = _binary(merged[gold_col], name=f"gold field {gold_field!r}")
    y_pred = _binary(merged[pred_col], name=f"prediction field {prediction_field!r}")

    raw_conf = _confusion(y_true, y_pred, weights=None)
    raw_metrics = _metrics(raw_conf)

    if pi_artifact is None:
        return ClassificationEvaluation(raw_metrics=raw_metrics, raw_confusion=raw_conf)

    _require_fields(pi_artifact, pi_field, role="pi")
    if tuple(str(value) for value in pi_artifact.primary_key) != keys:
        raise ArtifactError(
            "pi and gold artifacts must use the same primary-key columns."
        )
    pi_frame = _frame(pi_artifact, pi_field).rename(columns={pi_field: pi_col})
    _validate_unique(pi_frame, keys, role="pi")
    if set(_key_tuples(pi_frame, keys)) != set(_key_tuples(gold_frame, keys)):
        raise ArtifactError("pi key set must exactly equal the gold/audit key set.")

    merged_pi = merged.merge(
        pi_frame,
        on=list(keys),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    inclusion = pd.to_numeric(merged_pi[pi_col], errors="coerce").to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(inclusion))
        or np.any(inclusion <= 0)
        or np.any(inclusion > 1)
    ):
        raise ArtifactError("pi values must be finite and satisfy 0 < pi <= 1.")
    weights = 1.0 / inclusion
    weighted_conf = _confusion(y_true, y_pred, weights=weights)
    weighted_metrics = _metrics(weighted_conf)
    return ClassificationEvaluation(
        raw_metrics=raw_metrics,
        raw_confusion=raw_conf,
        weighted_metrics=weighted_metrics,
        weighted_confusion=weighted_conf,
    )


def _frame(artifact: BaseArtifact, field: str) -> pd.DataFrame:
    value = artifact.query(
        key_columns=True,
        data_columns=[field],
        metadata_columns=False,
        form="table",
        include_position=False,
    )
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(f"Could not materialize {field!r} as a table.")
    return value.reset_index(drop=True)


def _require_fields(artifact: BaseArtifact, field: str, *, role: str) -> None:
    columns = [str(value) for value in artifact.get_data_columns()]
    if field not in columns:
        raise ArtifactError(
            f"{role} artifact must expose data field {field!r}; available fields are {columns}."
        )


def _validate_unique(frame: pd.DataFrame, keys: tuple[str, ...], *, role: str) -> None:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ArtifactError(f"{role} frame is missing primary-key column(s) {missing}.")
    if frame.duplicated(subset=list(keys)).any():
        raise ArtifactError(f"{role} frame contains duplicate primary keys.")


def _key_tuples(frame: pd.DataFrame, keys: tuple[str, ...]) -> list[tuple[int, ...]]:
    return [
        tuple(int(v) for v in row)
        for row in frame.loc[:, list(keys)].itertuples(index=False, name=None)
    ]


def _binary(series: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)) or np.any(~np.isin(numeric, [0.0, 1.0])):
        raise ArtifactError(f"{name} values must all be binary 0/1.")
    return numeric.astype(int)


def _confusion(
    y_true: np.ndarray, y_pred: np.ndarray, *, weights: np.ndarray | None
) -> pd.DataFrame:
    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
        integer = True
    else:
        weights = np.asarray(weights, dtype=float)
        integer = False
    values = np.zeros((2, 2), dtype=float)
    for actual in (0, 1):
        for predicted in (0, 1):
            mask = (y_true == actual) & (y_pred == predicted)
            values[actual, predicted] = float(weights[mask].sum())
    if integer:
        values = values.astype(int)
    return pd.DataFrame(
        values,
        index=pd.Index([0, 1], name="actual"),
        columns=pd.Index([0, 1], name="predicted"),
    )


def _metrics(confusion: pd.DataFrame) -> dict[str, float]:
    tn = float(confusion.loc[0, 0])
    fp = float(confusion.loc[0, 1])
    fn = float(confusion.loc[1, 0])
    tp = float(confusion.loc[1, 1])
    total = tn + fp + fn + tp
    accuracy = _ratio(tp + tn, total)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    prevalence = _ratio(tp + fn, total)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "prevalence": prevalence,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
