# Spherical K-means: scientific design and spherecluster reference

Phase 5B adds `text_analysis_lab.translators.SphericalKMeans` to TeAL.
It is an independently written, small NumPy/SciPy implementation of
directional Lloyd clustering. It **does not** depend on the old
`spherecluster` package or on scikit-learn's internal Cython/private
K-means interfaces.

## Primary methodological and implementation references

- **Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005).**
  "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions."
  *Journal of Machine Learning Research*, 6, 1345–1382.
  https://www.jmlr.org/papers/v6/banerjee05a.html
- **spherecluster**, by Jason Laska and contributors:
  https://github.com/jasonlaska/spherecluster
  In particular, see
  [`spherecluster/spherical_kmeans.py`](https://github.com/jasonlaska/spherecluster/blob/master/spherecluster/spherical_kmeans.py)
  and the [repository README](https://github.com/jasonlaska/spherecluster).
  Its license is MIT (copyright Clara Labs, 2016).
  The TeAL implementation is independently authored, not copied from the
  legacy `spherecluster` implementation.

## What TeAL takes from spherecluster

- **Unit-length input rows:** spherical K-means is a directional algorithm.
  Normalize raw numerical feature vectors before fitting and at prediction
  time, retaining CSR sparsity where provided.
- **Cosine-based assignment:** choose the largest dot product between
  normalized observations and normalized cluster centers.
- **Re-normalize the mean direction after every centroid update:**
  this is the defining difference from normalizing observations once and
  using ordinary Euclidean K-means. It is emphasized explicitly in
  spherecluster's `_spherical_kmeans_single_lloyd`.
- **Multiple initializations:** fit independent candidate directions and
  select the best fit using the spherical objective. TeAL uses
  `sum(1 - max cosine similarity)` over input rows; spherecluster's
  historical implementation instead reports a squared-Euclidean inertia,
  related by a factor of 2 when the vectors are unit-normalized.
- **Dense and CSR support:** as in spherecluster, both inputs are accepted,
  but frozen centers are stored as a compact dense K×M array.
- **New-data prediction:** reuse the *fixed unit centers* rather than
  re-optimizing on the new observations.

## Explicit differences and modernizations

- spherecluster's original code imports private, version-specific sklearn
  modules (`sklearn.cluster._k_means_fast`,
  `sklearn.cluster.k_means_`). TeAL instead uses a public SciPy/NumPy
  matrix multiplication and independent centroid updates.
- TeAL uses a cosine-distance++ seeding rule (proportional to
  `1 - cos(x, c)` with the nearest existing center) or seeded random
  sample initialization. These may produce different starting centers and
  cluster labels than historical spherecluster/sklearn seeding.
- Zero-vector inputs are **rejected**: cosine direction does not exist
  for those rows. We do not silently create a fake direction or unassigned
  membership by default.
- Empty clusters are deterministically reseeded with low-affinity
  observations. A cluster with an exactly zero resultant direction retains
  its prior unit direction. If the data have fewer distinguishable directions
  than K, some output columns may remain empty; this is not a fabricated
  scientific cluster.
- Instead of relying on legacy estimator labels, the final assigned
  memberships are recomputed against **the exact frozen final centers**;
  this enforces training-output/reloaded-prediction parity.
- TeAL fits on all observations in the **researcher-selected source
  artifact**, with no implicit sampling, and can batch independent
  frozen-centroid prediction. This is not minibatch spherical fitting.
- Output uses TeAL's existing preserved-key, sparse hard N×K artifact
  contract, stable ordered cluster columns, snapshot metadata and
  `centers.npy` numerical assets. Record fitted NumPy/SciPy versions
  rather than incorrectly attributing the implementation to sklearn.
- Core fitting computes row-to-center similarities in bounded row batches.
  Its K×M centroid storage is dense. Users should still account for
  O(K×M) center memory and iteration costs when K and M are large.

## Numerical acceptance

Tests cover unit center norms after optimization, the directional objective,
a case where cosine assignment differs from Euclidean assignment against
nonunit centers, repeatable seeds, CSR/dense parity, scale-invariant
prediction, zero vectors, sparse high-dimensional examples, degenerate
clusters, preserved source row identities, Project artifact reopening and
the strict frozen operator schema/batching behavior.

This scientific implementation does not reproduce every historical
`spherecluster` initialization or tie-breaking rule exactly and should not
claim bit-for-bit label agreement. Reference equivalence concerns the
directional objective and the normalization in each Lloyd update.
