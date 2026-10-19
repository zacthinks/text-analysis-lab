# Changelog

All notable changes to TeAL are documented here.

TeAL follows Semantic Versioning. Before 1.0, backward-incompatible public API changes may occur in a minor release; patch releases are reserved for backward-compatible fixes and maintenance.

## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated

### Removed

## [0.8.0] - 2026-10-19

### Added

- `ArgMax()` translator for dense or sparse named matrices, producing a preserved-key table of each row's `max_feature` and `max_value`; deterministic feature-name ties, unassigned all-zero rows, and sparse implicit-zero handling (PR #48).
- Ephemeral `svd_spectrum()` matrix diagnostics and reusable standalone `visualization.scree(...)` plots for component energy and explained variance (PRs #44–#45).
- Query-time `with_artifacts` / `with_columns` composition of explicitly selected keyed table, JSONL, and matrix features without publishing a joined Artifact, including native and streaming output support (PRs #43, #46, #47).
- Scatter-plot color and hover attachments via `with_artifacts` and `with_columns`, including opt-in categorical missing-color labeling (PR #49).
- Bounded histogram and crosstab aggregation over attached relational and selected matrix fields, including qualified collisions (PR #50).
- Explicit `with_artifacts` and `with_columns` pass-through for the curated table, records, preview, sample, and streaming row/batch convenience wrappers.

### Changed

### Fixed

### Deprecated

### Removed

## [0.7.0] - 2026-10-13

### Added

- Opt-in two-pass fitting for the scikit-learn-backed `CountVectorizer` when an unfitted translator receives `Project.translate(..., batch_size=N)`. Each batch uses sklearn preprocessing and unrestricted vocabulary learning; global TF/DF filtering and a frozen vocabulary are applied before sparse matrix output is replayed.
- Opt-in complete-group, two-pass fitting for `ArtifactCountVectorizer` through `Project.translate(..., batch_by="<key>", batch_size=N)`. Aggregation respects retained primary-key prefixes and n-gram sequence boundaries.
- Translator-specific regression coverage for whole-versus-batched vocabulary and keyed matrix parity, stop words, stemming, character analyzers, DF thresholds, `max_features` ties, empty batches, checkpoint/resume, and persisted operator reuse.

### Changed

- Both count vectorizers now default to underscore-separated displayed word n-gram labels (for example, `new_york`) and support a configurable `ngram_separator`. The sklearn-backed vectorizer keeps its internal sklearn vocabulary strings unchanged; previously saved operators retain their earlier separator convention.
- Whole-artifact fitting remains the default when no batching opt-in is supplied. Fitted `CountVectorizer` reuse continues to transform in one pass; `ArtifactCountVectorizer` grouped batches are constrained not to split their output groups.

### Fixed

- Prevent ambiguous artifact-token n-gram labels from silently combining distinct features across batches, including fitted one-pass reuse. Operation-local collision tracking is cleared after successful finalization.

### Notes

- Both batched fitting modes retain corpus-wide feature statistics in memory and checkpoint them for recovery. A single large batch or an exceptionally large distinct-feature vocabulary can still require substantial memory.

## [0.6.0] - 2026-10-12

### Added

- Optional multi-pass `BaseTranslator.translation_passes(...)` negotiation with a default of one, a nonterminal `complete_translation_pass(...)` callback, and `TranslationRequest.pass_index/pass_count` execution context.
- Repeat traversal of one saved translation plan for sequential and supported parallel routes. Each pass must fully drain its canonical results and checkpoint before subsequent workers are created.
- Durable pass/unit checkpoint reconciliation and terminal callback checkpointing for resumable repeated translation; compatibility with old one-pass operation descriptors.
- Regression tests for pass barriers, repeated grouped/multi-source plans, existing one-pass behavior, interrupted resume, and multi-output publication.

### Changed

- Parallel repeated-pass execution creates new worker translators from the updated canonical state after each pass barrier.
- The release changes the public translator execution protocol without changing default one-pass semantics or frozen scientific-state restrictions.

### Notes

- Repeated passes reread source artifacts and use the same stored source-unit assignments; no all-batches memory cache is introduced. A very large individual batch can still exceed memory.
- Crash recovery does not provide transactional exactly-once side effects outside TeAL's checkpoint/writer guarantees.
- `CountVectorizer` and `ArtifactCountVectorizer` have not yet been converted to a batched multi-pass fitting algorithm.

## [0.5.0] - 2026-10-12

### Added

- Optional `Project.translate(..., batch_by="<primary-key level>", batch_size=N)` for complete hierarchical key-group batching. `batch_by` names one actual primary-key column; TeAL resolves the inclusive prefix so repeated child IDs under different parent keys remain separate.
- Optional `SourceRequest.min_batch_by` for a translator to declare the finest acceptable complete-group batch level. Requests finer than the requirement are rejected rather than silently changing the caller's batching policy.

### Changed

- Grouped translation plans persist chosen key groups and processing order in the operation's SQLite execution plan for durable retry and resume. Legacy row-based batching remains the default, with unchanged positional request constructors.
- Grouped batches follow first-seen group order and original source position *within* each group. Rows may therefore appear in a different physical order from interleaved row batching.

### Fixed

- Reject grouped `form="single"` requests that could discard records. Only one full-primary-key row per grouped batch is accepted with that form.
- Preserve grouped execution with arbitrary valid key names that overlap SQL planning labels, and migrate previously persisted positional plan tables.

### Notes

- The first grouped batching implementation applies to single-source `mode="batches"` requests. Translators requesting `mode="full_artifact"` still use their existing behavior; grouping requests against them are rejected. A single exceptionally large group may still require substantial memory.
- `ArtifactCountVectorizer` does **not** yet fit or transform in multiple batches; its two-pass refactor is deferred.
- This is a focused **minor release** under TeAL's existing policy because it adds a new optional public execution mode and translator constraint; release numbers indicate API evolution, not the magnitude of the implementation. No existing translator scientific semantics or positional request constructors are intentionally changed.

## [0.4.0] - 2026-10-11

### Added

- Generic matrix-row `top_features()` analysis with value, absolute, probability, term/document frequency, corpus-adjusted lift, empirical unshrunk FREX, and custom scorers, including canonical Feature Metadata alignment and corpus references.
- Persisted-Project DTM/LDA/SVD ranking acceptance checks across reopening, plus cross-family clustering acceptance.

- Ten unsupervised clustering translators with keyed, provenance-aware matrix outputs: KMeans, MiniBatchKMeans, KMedoids, SphericalKMeans, DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering, GaussianMixture, and SpectralClustering. Explicit fit-only and reusable frozen-model boundaries, sparse hard assignments, density-noise semantics, and dense Gaussian-mixture posterior responsibilities.
- Optional Rust FasterPAM `kmedoids` backend (`[kmedoids]` extra), upstream scientific credits, and NumPy/SciPy spherical K-means referenced to spherecluster.
- Backend-version fit provenance, clustering-specific scientific documentation, and integrated ten-method persistence/reuse tests.

### Changed

### Fixed

### Deprecated

### Removed

## [0.3.0] - 2026-10-09

### Added

- Standalone translation contracts for reusable translators.
- Transform-only `Pipeline` composition and reconstruction from explicit artifact provenance intervals.
- Explicit GeCo `documents`, `text`, and `display_text` roles with Pipeline-backed geometry replay.
- Strict frozen-operator snapshot semantics, schema v2, and an isolated legacy migration path.
- `Project.upgrade_legacy_operators(...)` for explicit, exact, auditable upgrades of eligible pre-strict projects without rerunning analyses.
- Runtime package version reporting through `text_analysis_lab.__version__`.
- A documented SemVer/release workflow with version consistency checks and tag-triggered GitHub releases.

### Changed

- Frozen translators now distinguish scientific state from execution-only controls more strictly.
- Reconstructable external model/resource references can serve as frozen scientific state when exact identity is pinned.
- GeCo Semantic Search and arbitrary new-text transforms now use the same recovered frozen Pipeline.
- SentenceTransformer encoders follow the selected model's normal default prompt behavior unless an explicit prompt override is frozen.
- Column-wise matrix normalization is executable but not reusable for arbitrary new-row replay.
- Legacy project upgrades rebind only completed operations to exact reusable strict replacements; historical-only and incomplete state remains untouched.

### Fixed

- Matrix SQL queries now retain filtering/sorting on relational key, position, and metadata columns while returning an actionable error only when a matrix feature is incorrectly referenced as a SQL column.
- Legacy project upgrades recover interrupted operation-descriptor synchronization and refuse ambiguous prior strict migrations rather than guessing.

### Deprecated

- Historical SentenceTransformer `task` state remains loadable for pre-1.0 compatibility, with a warning; new encoders no longer expose `task` as the preferred public abstraction.

### Removed

- The private GeCo-specific representation replay engine, replaced by general Pipeline reconstruction.

## [0.2.0] - 2026-09-30

Development checkpoint preceding the standalone translator, strict-freeze, Pipeline reconstruction, GeCo replay, and legacy-project upgrade refactors.

## [0.1.0] - 2026-09-10

Initial development release.
