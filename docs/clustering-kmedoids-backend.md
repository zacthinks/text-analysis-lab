# Rust FasterPAM backend used by KMedoids

TeAL's optional `KMedoids` translator delegates fitting to the Python
`kmedoids` package, which wraps the FasterPAM Rust implementation by
Erich Schubert and contributors:
https://github.com/kno10/python-kmedoids

The backend is **GPL-3.0-or-later**. TeAL's own source code is MIT-licensed.
Noncommercial use does **not** automatically remove GPL obligations.
Attribution is necessary for scholarship, but cannot replace a licensing
review where GPL-covered components are distributed or integrated. Consult
the upstream LICENSE before packaging or redistributing a combined product.
Users opt into this dependency with `text-analysis-lab[kmedoids]`; the
`all` extra also includes it for integration testing.

Scientific references to credit in related papers:

- Schubert, E., & Lenssen, L. (2022). *Fast k-medoids Clustering in Rust and
  Python*. Journal of Open Source Software, 7(75), 4183.
  https://doi.org/10.21105/joss.04183
- Schubert, E., & Rousseeuw, P. J. (2021). *Fast and Eager k-Medoids
  Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS
  Algorithms*. Information Systems, 101, 101804.
  https://doi.org/10.1016/j.is.2021.101804

## Scientific and computational contract

`KMedoids` accepts numeric dense or sparse feature matrices with the
`euclidean`, `manhattan`, or `cosine` metric. Training materializes a
full float64 N×N distance matrix for the explicitly chosen rows and runs
Rust FasterPAM. No samples or clusters are fabricated automatically.
The distance matrix alone occupies 8×N² bytes before backend overhead,
about 800 MB for 10,000 rows. Prepare any smaller training artifacts
explicitly in TeAL if the research design permits.

The fitted clusterer outputs a keyed sparse N×K hard-membership artifact.
For later rows, it assigns each observation to the nearest frozen medoid
feature vector. It does not refit medoids on new data. The frozen operator
retains scientific parameters, source feature order, fitted medoid vectors,
and upstream backend versions; it discards the N×N training distances.

Unlike sklearn's ordinary KMeans, medoids are original training rows,
not newly averaged prototype coordinates. Indices stored in provenance
are source-row positions at fit time; original row identity is available
through the retained input artifact's lineage.
