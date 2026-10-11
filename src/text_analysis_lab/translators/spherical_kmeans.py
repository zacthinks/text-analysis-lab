"""Spherical K-means: directional Lloyd updates on dense or CSR feature matrices.

The scientific algorithm follows the directional K-means formulation described
by Banerjee et al. (JMLR, 2005), and uses spherecluster's SphericalKMeans
implementation as a design reference (github.com/jasonlaska/spherecluster).
This is a separate, narrow NumPy/SciPy implementation: neither spherecluster
nor sklearn's private K-means internals are imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy
from scipy import sparse

from text_analysis_lab.core.errors import OperatorError, OperatorNotFittedError
from text_analysis_lab.translators.clustering import KMeans, _one_hot


def _unit_rows(matrix: Any) -> Any:
    """Normalize observations without densifying a CSR source matrix."""
    if sparse.issparse(matrix):
        unit = matrix.tocsr().astype(np.float64)
        lengths = np.sqrt(np.asarray(unit.multiply(unit).sum(axis=1)).ravel())
    else:
        unit = np.asarray(matrix, dtype=np.float64)
        lengths = np.linalg.norm(unit, axis=1)
    if not np.all(np.isfinite(lengths)):
        raise OperatorError("SphericalKMeans requires finite observation norms.")
    if np.any(lengths == 0):
        raise OperatorError(
            "SphericalKMeans rejects zero-length observations: cosine "
            "direction is undefined."
        )
    if sparse.issparse(unit):
        return unit.multiply((1.0 / lengths)[:, None]).tocsr()
    return unit / lengths[:, None]


def _dense_row(unit: Any, position: int) -> np.ndarray:
    if sparse.issparse(unit):
        return np.asarray(unit.getrow(position).toarray()).ravel()
    return np.asarray(unit[position]).ravel()


def _assign(
    unit: Any, centers: np.ndarray, *, batch_size: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Max-cosine assignment with bounded row-batch similarity matrices."""
    n_rows = unit.shape[0]
    labels = np.empty(n_rows, dtype=np.int64)
    scores = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        similarities = np.asarray(unit[start:end] @ centers.T, dtype=np.float64)
        similarities = np.clip(similarities, -1.0, 1.0)
        nearest = similarities.argmax(axis=1)
        labels[start:end] = nearest
        scores[start:end] = similarities[np.arange(end - start), nearest]
    return labels, scores


def _initialize(unit: Any, k: int, *, init: str, rng: np.random.Generator) -> np.ndarray:
    """Pick unit observations, optionally with cosine-distance++ probabilities."""
    n_rows = unit.shape[0]
    if init == "random":
        chosen = list(map(int, rng.choice(n_rows, size=k, replace=False)))
    else:
        chosen = [int(rng.integers(n_rows))]
        closest_costs = np.full(n_rows, np.inf, dtype=np.float64)
        while len(chosen) < k:
            sim = np.asarray(unit @ _dense_row(unit, chosen[-1])).ravel()
            # On the sphere, squared Euclidean separation = 2*(1-cosine).
            # The constant factor cancels from these seeding probabilities.
            closest_costs = np.minimum(
                closest_costs, np.maximum(0.0, 1.0 - np.clip(sim, -1.0, 1.0))
            )
            closest_costs[chosen] = 0.0
            total = float(closest_costs.sum())
            if total > 0:
                chosen.append(int(rng.choice(n_rows, p=closest_costs / total)))
            else:
                available = np.setdiff1d(
                    np.arange(n_rows), np.asarray(chosen), assume_unique=True
                )
                chosen.append(int(rng.choice(available)))
    return np.vstack([_dense_row(unit, i) for i in chosen])


def _spherical_run(
    unit: Any, k: int, *, init: str, max_iter: int, tol: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """One directional Lloyd optimization with an explicit empty-cluster policy."""
    centers = _initialize(unit, k, init=init, rng=rng)
    n_rows = unit.shape[0]
    iteration = 0
    for iteration in range(1, max_iter + 1):
        labels, scores = _assign(unit, centers)
        updated = np.empty_like(centers)
        needs_reseed: list[int] = []
        for cluster in range(k):
            selected = np.flatnonzero(labels == cluster)
            if selected.size == 0:
                needs_reseed.append(cluster)
                continue
            # Unlike Euclidean K-means, the resultant must be renormalized
            # after *every* centroid update. This is spherecluster's central
            # scientific distinction.
            resultant = np.asarray(unit[selected].sum(axis=0)).ravel()
            length = np.linalg.norm(resultant)
            if length <= 1e-12:
                # Opposing observations can cancel. Keep the previous valid
                # unit center instead of dividing by zero or producing NaNs.
                updated[cluster] = centers[cluster]
            else:
                updated[cluster] = resultant / length

        if needs_reseed:
            # Deterministic farthest-observation rescue. This is not sampling
            # a smaller training corpus: the whole source has been assigned.
            # Distinct directions cannot be guaranteed when data are repeated.
            remaining = set(range(n_rows))
            for cluster in needs_reseed:
                candidate = min(
                    remaining, key=lambda i: (scores[i], i)
                )
                remaining.remove(candidate)
                updated[cluster] = _dense_row(unit, candidate)

        # Stop only after an actual center update. Always reassign to the
        # saved final centers so artifact labels match frozen prediction.
        shift = float(np.sum((updated - centers) ** 2))
        centers = updated
        if shift <= tol:
            break

    final_labels, final_scores = _assign(unit, centers)
    directional_loss = float(np.maximum(0.0, 1.0 - final_scores).sum())
    return final_labels, centers, directional_loss, iteration


class SphericalKMeans(KMeans):
    """Directional K-means with unit centroids, sparse input, frozen reuse.

    The scientific fit consumes all rows of the chosen TeAL input artifact.
    Prediction is independent per row and follows the existing batched
    KMeans translation contract. Zero-length observations are rejected.
    """

    batched_translate = True

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters, init=init, n_init=n_init,
            max_iter=max_iter, tol=tol, random_state=random_state,
            operator_id=operator_id,
        )
        if isinstance(n_init, bool) or not isinstance(n_init, (int, np.integer)) or n_init < 1:
            raise ValueError("SphericalKMeans n_init must be a positive integer.")
        if max_iter < 1:
            raise ValueError("SphericalKMeans max_iter must be positive.")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("SphericalKMeans tol must be finite and nonnegative.")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
            or int(random_state) < 0
        ):
            raise ValueError("SphericalKMeans random_state must be nonnegative or None.")
        self.n_init = int(n_init)
        self.random_state = None if random_state is None else int(random_state)
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.numpy_version_: str | None = None
        self.scipy_version_: str | None = None

    def fit_transform(self, input_matrix: Any) -> Any:
        if self.is_fitted:
            raise OperatorError("SphericalKMeans is already fitted; create a new operator.")
        matrix, source, structured = self._standalone_input(input_matrix)
        n_rows = matrix.shape[0]
        if n_rows < self.n_clusters:
            raise ValueError("SphericalKMeans requires n_samples >= n_clusters.")
        unit = _unit_rows(matrix)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            result = _spherical_run(
                unit, self.n_clusters, init=self.init,
                max_iter=self.max_iter, tol=self.tol, rng=rng,
            )
            if best is None or result[2] < best[2]:
                best = result
        assert best is not None
        labels, centers, loss, iterations = best
        self._centers = centers.copy()
        self.n_clusters_ = self.n_clusters
        self.inertia_ = loss  # sum of 1 - cosine similarities, not squared L2
        self.n_iter_ = iterations
        self.numpy_version_ = np.__version__
        self.scipy_version_ = scipy.__version__
        return self._output(
            _one_hot(labels, n_clusters=self.n_clusters), source, structured
        )

    def translate(self, input_matrix: Any) -> Any:
        if self._centers is None:
            raise OperatorNotFittedError(
                "SphericalKMeans must be fitted before prediction."
            )
        matrix, source, structured = self._standalone_input(input_matrix)
        if matrix.shape[1] != self._centers.shape[1]:
            raise ValueError("SphericalKMeans requires the fitted feature width.")
        labels, _ = _assign(_unit_rows(matrix), self._centers)
        return self._output(
            _one_hot(labels, n_clusters=self.n_clusters), source, structured
        )

    def to_json_state(self) -> dict[str, Any]:
        state = super().to_json_state()
        state.pop("sklearn_version", None)
        state.update({
            "backend": "numpy-scipy",
            "numpy_version": self.numpy_version_,
            "scipy_version": self.scipy_version_,
            "inertia": self.inertia_,
            "n_iter": self.n_iter_,
        })
        return state

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> SphericalKMeans:
        result = cls(
            n_clusters=int(state.get("n_clusters", 8)),
            init=str(state.get("init", "k-means++")),
            n_init=int(state.get("n_init", 10)),
            max_iter=int(state.get("max_iter", 300)),
            tol=float(state.get("tol", 1e-4)),
            random_state=state.get("random_state", 42),
        )
        features = state.get("source_features")
        if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
            result.source_features_ = tuple(map(str, features))
        inertia = state.get("inertia")
        result.inertia_ = None if inertia is None else float(inertia)
        iterations = state.get("n_iter")
        result.n_iter_ = None if iterations is None else int(iterations)
        numpy_version = state.get("numpy_version")
        scipy_version = state.get("scipy_version")
        result.numpy_version_ = None if numpy_version is None else str(numpy_version)
        result.scipy_version_ = None if scipy_version is None else str(scipy_version)
        return result

    def load_assets(self, assets_dir: Any, manifest: Mapping[str, Any]) -> None:
        super().load_assets(assets_dir, manifest)
        if self._centers is not None and (
            not np.all(np.isfinite(self._centers))
            or not np.allclose(np.linalg.norm(self._centers, axis=1), 1.0, atol=1e-7)
        ):
            raise OperatorError("SphericalKMeans frozen centers must be unit-normalized.")
