"""Key-aligned, reproducible supervised model search over TeAL matrices.

ClassifierSearch is an in-memory sklearn training utility, not a BaseTranslator.
Only source rows selected by development-label keys are retrieved. Neither
search candidates nor fold estimators are persisted as TeAL artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.callback import ProgressBar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit, ParameterGrid,
    ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split,
)
# Successive-halving search is explicitly experimental in sklearn.
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.svm import LinearSVC
from tqdm import tqdm

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
        and name == item["output_name"]
    ]
    if len(matches) != 1:
        available = sorted(item["output_name"] for item in inventory["columns"]
                           if item["namespace"] in {"key", "data", "metadata"})
        raise ValueError(
            f"Expected an exact accessible label/group field {name!r}; "
            f"available fields: {available}."
        )
    item = matches[0]
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


class _SearchFitCounter:
    """Non-propagating SearchCV callback for individual completed CV fits."""

    def __init__(self, bar: Any) -> None:
        self.bar = bar

    def setup(self, estimator: Any, context: Any) -> None:
        pass

    def teardown(self, estimator: Any, context: Any) -> None:
        pass

    def on_fit_task_begin(self, estimator: Any, context: Any) -> None:
        pass

    def on_fit_task_end(self, estimator: Any, context: Any) -> bool:
        if context.task_name == "candidate-split-evaluation":
            self.bar.update(1)
        return False


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
        strategy: str | None = None,
        search_kwargs: Mapping[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.representations = list(representations)
        self._initial_representations = tuple(self.representations)
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
        self.strategy = strategy or ("random" if n_iter is not None else "grid")
        strategies = {"grid", "random", "halving_grid", "halving_random"}
        if self.strategy not in strategies:
            raise ValueError(f"Unknown sklearn search strategy {self.strategy!r}.")
        if self.strategy == "random" and n_iter is None:
            raise ValueError("Set n_iter to a positive candidate budget for strategy='random'.")
        if self.strategy in {"grid", "halving_grid"} and n_iter is not None:
            raise ValueError("n_iter is only applicable to randomized search strategies.")
        if self.strategy == "halving_random" and n_iter is None:
            raise ValueError("Set n_iter to an initial candidate budget for halving_random.")
        if self.strategy.startswith("halving") and holdout is not None:
            raise ValueError("Successive halving needs cross-validation, not one holdout fold.")
        self.search_kwargs = dict(search_kwargs or {})
        reserved = {
            "estimator", "param_grid", "param_distributions", "cv", "scoring",
            "refit", "n_iter", "n_candidates", "random_state", "n_jobs",
            "return_train_score", "error_score",
        }
        overlap = reserved.intersection(self.search_kwargs)
        if overlap:
            raise ValueError(f"Set {sorted(overlap)} through ClassifierSearch, not search_kwargs.")
        self.n_jobs = n_jobs
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
        self._progress_initial = 0

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
        """Build the initial search plan using sklearn's own samplers."""
        rows = []
        for rep_index, representation in enumerate(self.representations):
            for family_name, estimator, grid in self._families:
                if self.strategy in {"grid", "halving_grid"}:
                    configurations = ParameterGrid(grid)
                else:
                    # ParameterSampler uses bounded indexing for finite grids;
                    # random searches also support scipy parameter distributions.
                    configurations = ParameterSampler(
                        grid, n_iter=self.n_iter, random_state=self.random_state,
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

    def _sklearn_search(self, estimator: Any, grid: Mapping[str, Any]) -> Any:
        """Let sklearn own enumeration, folding, fitting and score aggregation."""
        common = dict(
            estimator=estimator, scoring=self.scoring, cv=self._folds,
            refit=False, error_score=np.nan, return_train_score=False,
            n_jobs=self.n_jobs, **self.search_kwargs,
        )
        if self.strategy == "grid":
            return GridSearchCV(param_grid=grid, **common)
        if self.strategy == "random":
            return RandomizedSearchCV(
                param_distributions=grid, n_iter=self.n_iter,
                random_state=self.random_state, **common,
            )
        if self.strategy == "halving_grid":
            return HalvingGridSearchCV(
                param_grid=grid, random_state=self.random_state, **common,
            )
        return HalvingRandomSearchCV(
            param_distributions=grid, n_candidates=self.n_iter,
            random_state=self.random_state, **common,
        )

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
        # A new plan/fit begins with the constructor's search space, not
        # any additional representations registered by earlier fit_more rounds.
        self.representations = list(self._initial_representations)
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
        self._group = group
        self._candidates = configs
        self._results = None
        self._matrices = {}
        # Successive halving may refit survivors in later rounds. Reporting
        # only the initial candidates * folds as a total is misleading.
        # sklearn determines the subsequent stages when its search runs.
        adaptive = self.strategy.startswith("halving")
        return pd.DataFrame([
            {k: v for k, v in row.items() if not k.startswith("_")} | {
                "n_folds": len(folds),
                "initial_round_fits": len(folds),
                "planned_fits": None if adaptive else len(folds),
                "development_rows": len(keys),
            }
            for row in configs
        ])

    def fit(
        self, labels: Any, *, target: str, group: str | None = None,
        progress: bool | str = False,
    ) -> ClassifierSearch:
        """Fit candidates with one compact CV-fit bar or verbose sklearn output.

        progress=True counts completed CV fits in place. progress="verbose"
        restores sklearn's detailed nested callbacks. With n_jobs>1 compact
        progress advances after whole searches because child workers cannot
        safely update one in-process counter. Successive-halving budgets are
        adaptive and the percentage is deliberately omitted.
        """
        if progress is not True and progress is not False and progress != "verbose":
            raise ValueError("progress must be False, True, or 'verbose'.")
        self.plan(labels, target=target, group=group)
        assert self._keys is not None and self._y is not None
        planned = list(self._candidates)
        outcomes: list[dict[str, Any]] = []
        self._candidates = []
        n_folds = len(self._folds)
        total = (
            None if self.strategy.startswith("halving")
            else self._progress_initial + len(planned) * n_folds
        )
        bar = (
            tqdm(
                total=total, initial=self._progress_initial,
                desc="Classifier search", unit="fit", dynamic_ncols=True,
            )
            if progress is True else None
        )
        try:
            for rep_index, representation in enumerate(self.representations):
                for model_name, estimator, grid in self._families:
                    group_plan = [
                        row for row in planned
                        if row["representation_index"] == rep_index
                        and row["model"] == model_name
                    ]
                    if progress == "verbose":
                        print(
                            f"Representation {rep_index + 1}/{len(self.representations)}"
                            f" ({representation.artifact_id}), {model_name}: "
                            f"{len(group_plan)} initial candidates x {n_folds} folds",
                            flush=True,
                        )
                    if bar is not None:
                        bar.set_postfix_str(
                            f"{representation.artifact_id} / {model_name}", refresh=False,
                        )
                    failure: str | None = None
                    outcome_data: list[dict[str, Any]] = []
                    # No matrix => no model fits. If sklearn raises before
                    # cv_results_, the number attempted is unknown.
                    attempted_fits: int | None = 0
                    try:
                        if rep_index not in self._matrices:
                            self._matrices[rep_index] = _extract_representation(
                                representation, self._keys, batch_size=self.batch_size,
                            )
                        matrix = self._matrices[rep_index]
                        search = self._sklearn_search(estimator, grid)
                        if progress == "verbose":
                            search.set_callbacks(ProgressBar(max_propagation_depth=None))
                        elif bar is not None and self.n_jobs in (None, 1):
                            search.set_callbacks(_SearchFitCounter(bar))
                        attempted_fits = None
                        search.fit(matrix, self._y)
                        cv_results = search.cv_results_
                        if bar is not None and self.n_jobs not in (None, 1):
                            # Worker-local callbacks cannot update the parent process.
                            bar.update(len(cv_results["params"]) * n_folds)
                        last_round = (
                            int(np.max(cv_results["iter"]))
                            if "iter" in cv_results else None
                        )
                        for j, params in enumerate(cv_results["params"]):
                            mean = float(cv_results["mean_test_score"][j])
                            folds = [
                                float(cv_results[f"split{k}_test_score"][j])
                                for k in range(n_folds)
                            ]
                            valid = bool(np.isfinite(mean) and all(
                                np.isfinite(score) for score in folds
                            ))
                            stage = (
                                int(cv_results["iter"][j])
                                if "iter" in cv_results else None
                            )
                            outcome_data.append({
                                "params": dict(params),
                                "mean_score": mean if valid else np.nan,
                                "std_score": (
                                    float(cv_results["std_test_score"][j])
                                    if valid else np.nan
                                ),
                                "fold_scores": folds,
                                "fit_seconds": float(
                                    cv_results["mean_fit_time"][j] * n_folds
                                ),
                                # Each cv_results_ row is one candidate
                                # evaluated over every shared CV fold. Halving
                                # lists survivors again at higher resources.
                                "executed_fits": n_folds,
                                "status": "ok" if valid else "failed",
                                "error": None if valid else (
                                    "One or more sklearn CV fits failed; "
                                    "see FitFailedWarning for details."
                                ),
                                "search_iteration": stage,
                                "n_resources": (
                                    int(cv_results["n_resources"][j])
                                    if "n_resources" in cv_results else None
                                ),
                                "eligible_for_selection": (
                                    last_round is None or stage == last_round
                                ),
                            })
                    except Exception as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                    if failure is not None:
                        # A broken representation or all-failed sklearn search
                        # does not prevent other representation/model searches.
                        outcome_data = [{
                            "params": row["params"],
                            "mean_score": np.nan,
                            "std_score": np.nan,
                            "fold_scores": [],
                            "fit_seconds": 0.0,
                            "executed_fits": attempted_fits,
                            "status": "failed",
                            "error": failure,
                            "search_iteration": None,
                            "n_resources": None,
                            "eligible_for_selection": True,
                        } for row in group_plan]
                    for result in outcome_data:
                        candidate = len(self._candidates)
                        row = {
                            "candidate": candidate,
                            "representation_index": rep_index,
                            "representation_id": representation.artifact_id,
                            "model": model_name,
                            "params": result["params"],
                            "_estimator": estimator,
                        }
                        self._candidates.append(row)
                        outcomes.append({
                            **{k: v for k, v in row.items() if not k.startswith("_")},
                            **{k: v for k, v in result.items() if k != "params"},
                            "strategy": self.strategy,
                            "search_round": 0,
                        })
        finally:
            if bar is not None:
                bar.close()
        self._results = pd.DataFrame(outcomes)
        return self

    def fit_more(
        self, *,
        representations: Sequence[Any],
        models: Mapping[str, Any],
        progress: bool | str = False,
    ) -> ClassifierSearch:
        """Append searches with identical development labels, CV folds and scoring.

        Earlier candidates retain their IDs and fitted results. New search
        rounds use the existing key-aligned development selection exclusively.
        """
        if self._results is None or self._label_artifact is None or self._target is None:
            raise RuntimeError("Call fit(...) before fit_more(...).")
        requested = list(representations)
        if not requested:
            raise ValueError("Provide at least one representation to fit_more.")
        if not models:
            raise ValueError("Provide at least one model family to fit_more.")
        additional = ClassifierSearch(
            representations=requested, models=models,
            cv=self.cv, holdout=self.holdout, scoring=self.scoring,
            n_iter=self.n_iter, random_state=self.random_state,
            batch_size=self.batch_size, strategy=self.strategy,
            search_kwargs=self.search_kwargs, n_jobs=self.n_jobs,
        )
        # Inherit the exact folds, including a custom or stochastic splitter.
        additional._make_folds = lambda y, groups: self._folds
        # Start the appended round's bar at the count already evaluated.
        previous = self._results["executed_fits"]
        additional._progress_initial = int(previous.dropna().sum())
        additional.fit(
            self._label_artifact, target=self._target,
            group=self._group, progress=progress,
        )
        if not np.array_equal(additional._y, self._y):
            raise ValueError("Development labels changed since the initial fit.")
        if additional._keys != self._keys:
            raise ValueError("Development keys changed since the initial fit.")
        existing = {rep.artifact_id: i for i, rep in enumerate(self.representations)}
        for local_index, rep in enumerate(additional.representations):
            if rep.artifact_id not in existing:
                existing[rep.artifact_id] = len(self.representations)
                self.representations.append(rep)
            mapped = existing[rep.artifact_id]
            if mapped not in self._matrices and local_index in additional._matrices:
                self._matrices[mapped] = additional._matrices[local_index]
        round_number = int(self._results["search_round"].max()) + 1
        offset = len(self._candidates)
        for row in additional._candidates:
            new_row = dict(row)
            new_row["candidate"] += offset
            new_row["representation_index"] = existing[
                row["representation_id"]
            ]
            self._candidates.append(new_row)
        new_results = additional.results()
        new_results["candidate"] += offset
        new_results["representation_index"] = new_results[
            "representation_id"
        ].map(existing)
        new_results["search_round"] = round_number
        self._results = pd.concat(
            [self._results, new_results], ignore_index=True,
        )
        return self

    def results(self) -> pd.DataFrame:
        """Return one row per sklearn CV evaluation.

        Under halving, a configuration may occur in multiple rounds at
        different resource levels. executed_fits counts attempted CV fits
        in each reported evaluation. The total is known only if all
        searches successfully returned cv_results_.
        """
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
        if not chosen["eligible_for_selection"]:
            raise ValueError("Cannot finalize an intermediate successive-halving candidate.")
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
            "search_strategy": self.strategy,
            "search_iteration": (
                int(chosen["search_iteration"])
                if pd.notna(chosen["search_iteration"]) else None
            ),
            "development_rows": len(self._y),
        }
        return FinalizedClassifier(
            predictor=FittedPredictor(
                fitted, save_model=True, training_provenance=provenance,
            ),
            source=source,
            provenance=provenance,
        )
