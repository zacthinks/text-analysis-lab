# HDBSCAN course checkpoint (synthetic or course data)

This is an **instructor-facing workflow sketch** for the Phase 2 clustering
milestone. It requires a real TeAL `Project` and a keyed document-term matrix
artifact `dtm` already built by the ordinary course import/vectorization cells.
The AERA corpus is **not** included in this repository or CI.

## Experiment: DTM → SVD → optional UMAP → HDBSCAN

```python
import numpy as np
from text_analysis_lab.translators import HDBSCAN, KMeans, SVD, UMAP

# Existing variables from your notebook:
# project: TeAL Project; dtm: keyed sparse_matrix DTM Artifact.
rows, terms = dtm.get_matrix().shape
rank = min(30, rows - 1, terms - 1)
if rank < 2:
    raise ValueError("The selected corpus is too small for this demonstration")

svd = project.translate(
    SVD(n_components=rank, random_state=42), dtm, alias="clustering_svd"
)["output"]

# Compare direct SVD clustering with UMAP-aided clustering on the SAME
# stable-key observation universe.
direct = project.translate(
    HDBSCAN(min_cluster_size=5, min_samples=5),
    svd, alias="hdbscan_svd",
)["output"]

umap_dims = min(10, rank - 1)
umap = project.translate(
    UMAP(n_components=umap_dims, metric="cosine", random_state=42),
    svd, alias="clustering_umap",
)["output"]
via_umap = project.translate(
    HDBSCAN(min_cluster_size=5, min_samples=5),
    umap, alias="hdbscan_umap",
)["output"]
baseline = project.translate(
    KMeans(n_clusters=5, random_state=42), svd, alias="clustering_kmeans"
)["output"]

def describe_clusters(artifact):
    values = artifact.get_matrix().tocsr()
    sizes = np.asarray(values.sum(axis=0)).ravel().astype(int)
    noise = int(np.count_nonzero(np.diff(values.indptr) == 0))
    print("K =", values.shape[1], "sizes =", sizes.tolist())
    print("Unassigned/noise:", noise, "/", values.shape[0],
          f"({noise / values.shape[0]:.1%})")

for label, artifact in [
    ("Direct SVD", direct),
    ("UMAP → HDBSCAN", via_umap),
    ("KMeans baseline", baseline),
]:
    print(label)
    describe_clusters(artifact)
```

**Inspect examples, not just counts.** Query each membership Artifact with
`key_columns=True, data_columns=True` and join the selected stable keys back
to source documents via TeAL's ordinary key/lineage APIs. Read several source
texts per discovered cluster and noise group. Re-run with alternative
`min_cluster_size`, `min_samples`, SVD ranks and UMAP dimensions, without
claiming that cluster indices are validated constructs.

**Cautions:** Use a corpus large enough for the chosen SVD rank, UMAP
`n_neighbors`, and density thresholds. HDBSCAN uses sklearn's own
`min_samples` convention (includes the observation). An N x 0 all-noise
output is a valid outcome and should be reported as such, never coerced to a
fake cluster. Fit is full-corpus for the chosen subset; to work on a sample,
explicitly create a subset Artifact first. The HDBSCAN operator is fit-only and
cannot predict new rows; later separately trained classifiers from keyed X,Y
are a different scientific analysis. UMAP's input metric and HDBSCAN's
embedding-space metric measure different geometries.

This example is exploratory teaching material, **not** an assertion about
actual AERA results. The actual notebook smoke test remains a private course
acceptance task.
