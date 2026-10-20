"""Key-aligned, reproducible supervised model search over TeAL matrices.

ClassifierSearch is an in-memory sklearn training utility, not a BaseTranslator.
Only source rows selected by development-label keys are retrieved. Neither
search candidates nor fold estimators are persisted as TeAL artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from random import Random
from math import isfinite
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, ParameterGrid, StratifiedKFold,
    train_test_split,
)
from sklearn.svm import LinearSVC

from text_analysis_lab.translators.fitted_predictor import FittedPredictor


def _describe_search_value(value: Any) -> Any:
    """Keep provenance JSON-safe without confusing parameter objects with scalars.

    The actual estimator and estimator-valued hyperparameters remain preserved
    in the fitted model asset. This is a descriptive, stable selection record.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else {
            "__kind__": "nonfinite_float", "value": str(value),
        }
    if isinstance(value, BaseEstimator):
        return {
            "__kind__": "sklearn_estimator",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "parameters": {
                str(k): _describe_search_value(v)
                for k, v in sorted(value.get_params(deep=False).items())
            },
        }
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "numpy_array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _describe_search_value(value.tolist()),
        }
    if isinstance(value, Mapping):
        return {
            str(k): _describe_search_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_describe_search_value(v) for v in value]
    return {
        "__kind__": "object",
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _describe_search_params(params: Mapping[str, Any]) -> dict[str, Any]:
    description = {str(k): _describe_search_value(v) for k, v in params.items()}
    # Enforce portable strict JSON before creating the saved operator.
    json.dumps(description, sort_keys=True, allow_nan=False)
    return description


@dataclass(frozen=True)
class FinalizedClassifier:
    """Refitted predictor and its selected representation (not a stored artifact)."""

    predictor: FittedPredictor
    source: Any
    provenance: dict[str, Any]


def _field_request(artifact: Any, name: str) -> dict[str, Any]:
    inventory = artifact.query_columns(metadata_mode="full")
    matches = [
        item for item in inventory["columns"]
        if item["namespace"] in {"key", "data", "metadata"}
        and name in (item["base_name"], item["output_name"], item["qualified_name"])
    ]
    unique = {(m["namespace"], m["qualified_name"]): m for m in matches}
    if len(unique) != 1:
        raise ValueError(
            f"Expected one accessible label/group field {name!r}; "
            f"found {len(unique)}. Select an unambiguous field name."
        )
    item = next(iter(unique.values()))
    namespace = item["namespace"]
    if namespace == "key":
        # Keys are always loaded to preserve stable identity.
        return {"namespace": namespace, "output_name": item["output_name"]}
    return {"namespace": namespace, "output_name": item["output_name"]}


def _load_development_labels(
    artifact: Any, target: str, group: str | None, batch_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray | None]:
    target_spec = _field_request(artifact, target)
    group_spec = _field_request(artifact, group) if group is not None else None
    needs = [target_spec] + ([group_spec] if group_spec else [])
    data = list(dict.fromkeys(
        s["output_name"] for s in needs if s["namespace"] == "data"
    ))
    metadata = list(dict.fromkeys(
        s["output_name"] for s in needs if s["namespace"] == "metadata"
    ))
    primary_key = list(artifact.primary_key)
    keys: list[dict[str, Any]] = []
    targets: list[Any] = []
    groups: list[Any] = []
    batches = artifact.query(
        key_columns=True, data_columns=data if data else False,
        metadata_columns=metadata if metadata else False,
        metadata_mode="full" if metadata else "none",
        form="table", iter_batches=True, batch_size=batch_size,
    )
    for frame in batches:
        if frame.empty:
            continue
        if frame[target_spec["output_name"]].isna().any():
            raise ValueError("Development labels must be complete; exclude unlabeled rows.")
        if group_spec and frame[group_spec["output_name"]].isna().any():
            raise ValueError("Cross-validation groups must be complete.")
        keys.extend(frame[primary_key].to_dict(orient="records"))
        targets.extend(frame[target_spec["output_name"]].tolist())
        if group_spec:
            groups.extend(frame[group_spec["output_name"]].tolist())
    if len(keys) < 2:
        raise ValueError("ClassifierSearch requires at least two development labels.")
    if len({tuple(record[col] for col in primary_key) for record in keys}) != len(keys):
        raise ValueError("Development-label artifact contains duplicate primary keys.")
    return keys, np.asarray(targets), np.asarray(groups) if group_spec else None


def _extract_representation(
    representation: Any,
    keys: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> Any:
    """Fetch only labeled positions with indexed key lookup, preserving key order."""
    matrices = []
    try:
        for start in range(0, len(keys), batch_size):
            request = keys[start:start + batch_size]
            positions = representation.project.query.positions_by_keys(
                representation, request,
            )
            result = representation.get_matrix(positions=positions)
            if result.shape[0] != len(request):
                raise ValueError("Feature retrieval returned the wrong row count.")
            matrices.append(result)
    except KeyError as exc:
        raise ValueError(
            "Representation is missing development keys; all labeled keys "
            "must be present in each searched representation."
        ) from exc
    if not matrices:
        raise ValueError("Representation had no matching labeled rows.")
    if any(sparse.issparse(batch) for batch in matrices):
        return sparse.vstack(matrices, format="csr")
    return np.vstack(matrices)


def _model_families(
    config: Mapping[str, Any], random_state: int | None,
) -> list[tuple[str, Any, dict[str, Any]]]:
    if not config:
        raise ValueError("ClassifierSearch requires at least one model family.")
    families = []
    for name, details in config.items():
        if isinstance(name, str) and name in {"logistic", "logistic_regression"}:
            estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            params = details
        elif isinstance(name, str) and name in {"linear_svm", "linear_svc"}:
            estimator = LinearSVC(random_state=random_state)
            params = details
        elif isinstance(details, (tuple, list)) and len(details) == 2:
            estimator, params = details
        elif isinstance(details, Mapping) and "estimator" in details:
            estimator = details["estimator"]
            params = details.get("param_grid", {})
        elif hasattr(name, "fit") and hasattr(name, "get_params"):
            estimator, params = name, details
        else:
            raise ValueError(f"Unknown classifier family {name!r}.")
        if not callable(getattr(estimator, "fit", None)) or not callable(
            getattr(estimator, "predict", None)
        ):
            raise TypeError(f"Model family {name!r} must have fit() and predict().")
        if not isinstance(params, Mapping):
            raise TypeError(f"Parameter grid for {name!r} must be a mapping.")
        clone(estimator)
        families.append((str(name), estimator, dict(params)))
    return families


class ClassifierSearch:
    """Search representation and sklearn-model combinations on development labels.

    All candidates use the same fixed CV or holdout folds. A selected candidate
    is *freshly refitted* on all development labels by finalize().
    """

    def __init__(
        self, *,
        representations: Sequence[Any],
        models: Mapping[str, Any],
        cv: int | Any | None = 5,
        holdout: float | None = None,
        scoring: str = "f1",
        n_iter: int | None = None,
        random_state: int | None = 42,
        batch_size: int = 10_000,
    ) -> None:
        self.representations = list(representations)
        if not self.representations:
            raise ValueError("Provide at least one representation.")
        if (cv is None) == (holdout is None):
            raise ValueError("Specify exactly one of cv or holdout.")
        if holdout is not None and not (0 < holdout < 1):
            raise ValueError("holdout must be between 0 and 1.")
        if isinstance(cv, int) and (isinstance(cv, bool) or cv < 2):
            raise ValueError("cv must be an integer at least 2 or a sklearn splitter.")
        if n_iter is not None and (
            isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0
        ):
            raise ValueError("n_iter must be a positive integer or None.")
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.cv, self.holdout, self.scoring = cv, holdout, scoring
        self.n_iter, self.random_state, self.batch_size = n_iter, random_state, int(batch_size)
        self._families = _model_families(models, random_state)
        self._validate_representations()
        self._candidates: list[dict[str, Any]] = []
        self._keys: list[dict[str, Any]] | None = None
        self._y: np.ndarray | None = None
        self._groups: np.ndarray | None = None
        self._folds: list[tuple[np.ndarray, np.ndarray]] = []
        self._label_artifact: Any | None = None
        self._target: str | None = None
        self._results: pd.DataFrame | None = None
        self._matrices: dict[int, Any] = {}

    def _validate_representations(self) -> None:
        first = self.representations[0]
        project = first.project
        key_schema = tuple(first.primary_key)
        for item in self.representations:
            if item.artifact_type.value not in {"dense_matrix", "sparse_matrix"}:
                raise TypeError("Search representations must be matrix artifacts.")
            if tuple(item.primary_key) != key_schema:
                raise ValueError("All representations must share ordered primary-key schema.")
            if item.project is not project:
                raise ValueError("Search representations must be in the same Project.")

    def _candidate_configs(self) -> list[dict[str, Any]]:
        rows = []
        for rep_index, representation in enumerate(self.representations):
            for family_name, estimator, grid in self._families:
                combinations = ParameterGrid(grid)
                if self.n_iter is None:
                    configurations = iter(combinations)
                else:
                    # ParameterGrid supports O(1) indexing and O(number of
                    # hyperparameters) cardinality, without materializing all
                    # combinations. Random.sample(range(...), k) uses bounded
                    # memory for large finite grids.
                    budget = min(self.n_iter, len(combinations))
                    picked = Random(self.random_state).sample(
                        range(len(combinations)), budget,
                    ) if budget < len(combinations) else range(len(combinations))
                    configurations = (
                        combinations[int(index)] for index in picked
                    )
                for params in configurations:
                    rows.append({
                        "candidate": len(rows),
                        "representation_index": rep_index,
                        "representation_id": representation.artifact_id,
                        "model": family_name,
                        "params": dict(params),
                        "_estimator": estimator,
                    })
        return rows

    def _make_folds(
        self, y: np.ndarray, groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(y))
        if self.holdout is not None:
            if groups is not None:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=self.holdout,
                    random_state=self.random_state,
                )
                return [
                    (np.asarray(training), np.asarray(validation))
                    for training, validation in splitter.split(indices, y, groups)
                ]
            training, validation = train_test_split(
                indices, test_size=self.holdout,
                random_state=self.random_state, stratify=y,
            )
            return [(np.asarray(training), np.asarray(validation))]
        if isinstance(self.cv, int):
            splitter = GroupKFold(n_splits=self.cv) if groups is not None else (
                StratifiedKFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state,
                )
            )
        else:
            splitter = self.cv
        folds = [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in splitter.split(indices, y, groups)
        ]
        for training, validation in folds:
            if len(set(training) & set(validation)):
                raise ValueError("Cross-validation training and validation rows overlap.")
            if (
                groups is not None
                and len(set(groups[training]) & set(groups[validation]))
            ):
                raise ValueError("Grouped CV has overlapping development groups.")
        return folds

    def plan(
        self, labels: Any, *, target: str, group: str | None = None,
    ) -> pd.DataFrame:
        if tuple(labels.primary_key) != tuple(self.representations[0].primary_key):
            raise ValueError("Labels and representations require the same key schema.")
        if labels.project is not self.representations[0].project:
            raise ValueError("Development labels must be in the representation Project.")
        keys, y, groups = _load_development_labels(
            labels, target, group, self.batch_size,
        )
        folds = self._make_folds(y, groups)
        if not folds or any(not len(t) or not len(v) for t, v in folds):
            raise ValueError("Cross-validation folds must have nonempty training and validation.")
        configs = self._candidate_configs()
        if not configs:
            raise ValueError("No candidate configurations.")
        self._keys, self._y, self._groups, self._folds = keys, y, groups, folds
        self._label_artifact, self._target = labels, str(target)
        self._candidates = configs
        self._results = None
        self._matrices = {}
        return pd.DataFrame([
            {k: v for k, v in row.items() if not k.startswith("_")} | {
                "n_folds": len(folds), "planned_fits": len(folds),
                "development_rows": len(keys),
            }
            for row in configs
        ])

    def fit(
        self, labels: Any, *, target: str, group: str | None = None,
    ) -> ClassifierSearch:
        self.plan(labels, target=target, group=group)
        assert self._keys is not None and self._y is not None
        scorer = get_scorer(self.scoring)
        outcomes: list[dict[str, Any]] = []
        for row in self._candidates:
            index = row["representation_index"]
            if index not in self._matrices:
                try:
                    self._matrices[index] = _extract_representation(
                        self.representations[index], self._keys,
                        batch_size=self.batch_size,
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    self._matrices[index] = exc
            matrix = self._matrices[index]
            scores = []
            times = []
            error = None
            if isinstance(matrix, Exception):
                error = str(matrix)
            else:
                for training, validation in self._folds:
                    started = perf_counter()
                    try:
                        estimator = clone(row["_estimator"]).set_params(**row["params"])
                        estimator.fit(matrix[training], self._y[training])
                        scores.append(float(scorer(estimator, matrix[validation], self._y[validation])))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        break
                    finally:
                        times.append(perf_counter() - started)
            outcomes.append({
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "mean_score": float(np.mean(scores)) if error is None else np.nan,
                "std_score": float(np.std(scores)) if error is None else np.nan,
                "fold_scores": list(scores),
                "fit_seconds": float(sum(times)),
                "status": "failed" if error is not None else "ok",
                "error": error,
            })
        self._results = pd.DataFrame(outcomes)
        return self

    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call fit(...) before results().")
        return self._results.copy()

    def finalize(self, *, candidate: int) -> FinalizedClassifier:
        if self._results is None or self._y is None or self._label_artifact is None:
            raise RuntimeError("Call fit(...) before finalize().")
        index = int(candidate)
        if index < 0 or index >= len(self._candidates):
            raise IndexError(f"Unknown candidate {candidate}.")
        chosen = self._results.loc[self._results["candidate"] == index].iloc[0]
        if chosen["status"] != "ok":
            raise ValueError("Cannot finalize a failed candidate.")
        row = self._candidates[index]
        matrix = self._matrices[row["representation_index"]]
        fitted = clone(row["_estimator"]).set_params(**row["params"])
        fitted.fit(matrix, self._y)
        source = self.representations[row["representation_index"]]
        assert self._keys is not None
        key_identity = sha256(
            json.dumps(
                [[record[col] for col in source.primary_key] for record in self._keys],
                separators=(",", ":"), default=str,
            ).encode("utf-8")
        ).hexdigest()
        provenance = {
            "development_key_fingerprint": key_identity,
            "development_primary_key": list(source.primary_key),
            "representation_artifact_id": source.artifact_id,
            "development_labels_artifact_id": self._label_artifact.artifact_id,
            "target": self._target,
            "candidate": index,
            "model": row["model"],
            "params": _describe_search_params(row["params"]),
            "development_rows": len(self._y),
        }
        return FinalizedClassifier(
            predictor=FittedPredictor(
                fitted, save_model=True, training_provenance=provenance,
            ),
            source=source,
            provenance=provenance,
        )
