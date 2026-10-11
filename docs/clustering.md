# Clustering translators

**Status:** ten translators implemented on the clustering feature branch;
release/version integration requires the final consolidation review. These
are transformations of **numeric matrix feature artifacts**, not automatic
topic interpretation or supervised classifiers.

```python
from text_analysis_lab import translators as tr

# Standalone: returns a CSR hard-membership matrix with K cluster columns.
membership = tr.KMeans(n_clusters=3, random_state=42).fit_transform(X)

# With TeAL Project and an existing numeric matrix artifact:
# result = project.translate(tr.DBSCAN(eps=0.4, min_samples=3), matrix_artifact)["output"]
# result is a persistent, keyed sparse matrix Artifact.
```

## Available models and exact output/reuse contracts

| Translator | Fitting backend | Dense input | CSR input | Output | New-row reuse |
| --- | --- | --- | --- | --- | --- |
| `KMeans` | sklearn | Yes | Yes | sparse hard N×K | Yes; frozen centers |
| `MiniBatchKMeans` | sklearn | Yes | Yes | sparse hard N×K | Yes; frozen centers |
| `KMedoids` | optional Rust `kmedoids` FasterPAM | Yes | Yes | sparse hard N×K | Yes; frozen medoid vectors |
| `SphericalKMeans` | NumPy/SciPy (see spherecluster reference) | Yes | Yes | sparse hard N×K | Yes; frozen unit centers |
| `DBSCAN` | sklearn | Yes | Yes | sparse hard N×K | No; fit-only |
| `HDBSCAN` | sklearn | Yes | Yes, supported metric/backend combinations | sparse hard N×K | No; fit-only |
| `OPTICS` | sklearn | Yes | Metric-dependent; e.g. cosine | sparse hard N×K | No; fit-only |
| `AgglomerativeClustering` | sklearn | Yes | No | sparse hard N×K | No; fit-only |
| `GaussianMixture` | sklearn | Yes | No | **dense posterior N×K** | Yes; frozen mixture parameters |
| `SpectralClustering` | sklearn | Yes | Supported feature-based affinities | sparse hard N×K | No; fit-only |

Sparse support depends on the underlying metric and scientific
configuration; accepting a sparse *type* does not guarantee every
metric/algorithm combination can operate on it. The wrappers reject
precomputed distance and affinity matrices: these are **feature
translators**, not arbitrary square matrices with hidden row identity.

### Membership meaning

The output always preserves the input primary keys and row order through
TeAL's `preserved_key` lineage. Each column is named `cluster_0`,
`cluster_1`, etc., in the fitted label/order scheme.

- **Hard clustering:** one value of 1 per assigned row; all others are zero.
  Density methods may designate an observation as noise (`-1`), encoded
  as an entirely **zero** row, not an extra `noise` cluster column.
  All-noise results are valid sparse **N×0** Artifacts.
- **GaussianMixture:** posterior responsibilities are real probabilities
  between zero and one, generally summing to one across components in
  each row. Do not treat them as hard labels without an explicit separate
  research decision. Mixture components need not represent substantive
  research topics.
- **K is a representation, not a guarantee:** requested K may yield
  unused columns in some algorithms. DBSCAN/HDBSCAN/OPTICS discover their
  own cluster counts. Column indices are local to a fit and have no
  meaningful alignment with other independent fits.

### Frozen scientific reuse

A new instance fits over **the complete explicitly selected input
Artifact**. TeAL does not introduce implicit sampling, a
`training_frac` parameter or a second undocumented fit corpus. If
you need a smaller scientifically chosen training set, build and select
that source Artifact yourself first.

Reusable clusterers save numerical fitted assets and ordered input feature
schema. `Project.translate(frozen_operator, new_artifact, batch_size=...)`
may apply to a different keyed source with the **same ordered feature
columns**. Batched reuse does not refit. In MiniBatch K-means, the
constructor's `batch_size` is an **algorithmic fitting** parameter,
different from TeAL's operation-level `batch_size` for row-batched
prediction.

Fit-only models retain configuration/provenance and the original keyed
membership Artifact, **not** a predictive hierarchy or out-of-sample
assigner. Calling `translate` on a fitted density, agglomerative or
spectral clusterer is not a valid scientific extension to new data.
If out-of-sample labels are needed, train an independent inductive
classifier from intentionally aligned X and cluster-membership Y.
That classifier is another scientific procedure and is not part
of this unsupervised translator family.

### Parameter and backend boundaries

- K-means and MiniBatch K-means use Euclidean distance. MiniBatch fit is
  stochastic/approximate relative to batch K-means; set and record
  `random_state` for reproducibility.
- K-medoids supports feature-based Euclidean, Manhattan or cosine
  distance, choosing actual source rows as medoids. Install
  `text-analysis-lab[kmedoids]` for fitting. Reusing a saved
  fitted snapshot needs only NumPy/scikit-learn, not the Rust backend.
  The optional Rust package is GPL-3.0-or-later; see
  [backend/attribution and license notes](clustering-kmedoids-backend.md).
- Spherical K-means uses cosine similarity on nonzero vectors, normalizes
  both inputs and cluster centers, and normalizes centers **after every
  Lloyd update**. Zero-direction input rows are rejected.
  See the [spherecluster comparison and credits](clustering-spherical-kmeans.md).
- Agglomerative Ward linkage requires Euclidean metric.
  The user can choose either a fixed K or a distance-threshold cut.
  Sparse inputs are explicitly rejected rather than silently densified.
- GaussianMixture fits **dense** feature matrices and supports
  `full`, `tied`, `diag` and `spherical` covariance.
  `predict_proba` yields real-valued responsibilities.
- SpectralClustering accepts **feature-based** RBF or nearest-neighbor
  affinity, not a precomputed affinity graph; it is fit-only.
- HDBSCAN uses sklearn's convention that `min_samples` includes the
  observation itself, unlike some alternative libraries.
  No `approximate_predict` or stored condensed hierarchy is exposed.
- OPTICS can use Xi or DBSCAN-style extraction; with the latter,
  `eps` must be positive and no larger than `max_eps`.
  Its sparse metric restrictions reflect sklearn's backend.

### Memory and execution costs

Every initial cluster fit is a **global** scientific operation over the
chosen source. Because an algorithm is passed a sparse matrix does not
mean the algorithm will have sparse internal working state:

- **K-medoids:** full float64 N×N dissimilarity matrix consumes at least
  8N² bytes (800 MB when N=10,000), plus working memory; no implicit CLARA
  approximation. This backend may dominate laptop RAM.
- **OPTICS:** sklearn's implementation has quadratic-time behavior;
  neighbors/reachability computation can become expensive.
- **SpectralClustering:** graph construction and eigendecomposition can
  require significant working memory, especially with RBF affinity;
  a nearest-neighbor affinity does not eliminate all solver costs.
- **AgglomerativeClustering / density clustering:** full-data fits can be
  expensive, depending on the number of rows, feature geometry, and metric.
- **SphericalKMeans:** CSR feature matrices remain sparse for input
  normalization; cluster centers are dense K×M; similarity assignments
  are computed in bounded batches without materializing an N×N matrix.
- **Reusable prediction:** center/medoid/Gaussian component assignment
  works independently on row batches, limiting operational memory
  without changing the original fit.

No performance claims are benchmarks. For large intended datasets,
record N, M, K, input sparsity, metric, fit duration and peak RAM in
the user's own workflow before selecting a heavy method.

### Provenance, compatibility and downstream validity

All methods store scientific parameters, source feature order, discovered
cluster count and relevant fitted backend/version metadata in the
operator snapshot. sklearn wrappers record the sklearn fit version;
K-medoids records Rust `kmedoids` and sklearn distance versions;
SphericalKMeans records NumPy/SciPy versions. Versions are provenance,
**not** a promise of bit-for-bit scientific reproduction in a different
environment.

Persisted fitted operators are strict snapshots. Some are intentionally
provenance-only; only explicitly reusable operators may assign new
observations. The derived clusters are exploratory measurements, not
validated substantive categories. Revisit cluster stability, noise,
interpretability and source evidence separately from API acceptance.

## Verification and release

The repository's `tests/test_clustering_*.py`,
`tests/test_phase4_clustering.py` and `tests/test_spherical_kmeans.py`
cover individual scientific parity, persistence and error paths.
`tests/test_clustering_family_acceptance.py` provides integrated
synthetic cross-family Project/reopen/frozen-operation checks. The full
test suite and additional real-Project release smoke tests should be
verified at the **final merge candidate SHA**, not assumed from
successful individual phase PRs.

TeAL course-specific notebooks and private AERA/S-062 fixtures belong to
their own course workspace and do not enter the software repository CI.
Bundled release/version/tag changes are a distinct final review after
the clustering family is integrated with `main` and other pending
features are accounted for.
