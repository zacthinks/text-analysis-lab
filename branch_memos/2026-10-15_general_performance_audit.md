# TeAL general performance audit — translation pipeline, corpus operations and model fitting

**Date:** 2026-10-15
**Branch:** `perf/translation-pipeline-audit`
**Status:** Scope + static source audit; no optimization patches or reproducible benchmarks yet.
**Priority:** After Unit 6 AERA notebook correctness and classroom run; maintain separate repo/branch.
**Companion memo:** [subset-specific investigation](2026-10-15_subset_performance.md).

## Why broaden scope?

The instructor reported these **end-to-end wall times on a local AERA course project**:

| Operation | Observed | Configuration | Suspected work |
|---|---:|---|---|
| `project.subset` | ~20 s | spaCy `is_alpha & ~is_stop`, keys + 2 flags, `iter_batches=True`, `batch_size=20_000`, `workers=1` | Source materialization, batched key filtering, writing and checkpoints |
| `ArtifactCountVectorizer` | ~50 s | `field="lemma"`, `group_by="row_id"`, unigram, `min_df=5`, grouped `batch_by="row_id"`, `batch_size=150`, `workers=1` | Two complete passes, global vocabulary statistics, group query/plan, event construction, sparse writes, checkpoint |

**These timings are observations, not controlled benchmarks.** We do not have the exact restricted token count, hardware/OS details, project file location, cache state, installed TeAL source revision or run-level timing traces. In particular, ~50s for a globally fitted count representation may involve real work; do not classify all of it as overhead.

The broader hypothesis is that recurring **generic infrastructure costs** affect multiple operations, with additional **translator-specific** costs. Do not optimize subset and vectorizer independently until their shared bottlenecks are measured.

## Static code findings (TeAL `main`, inspected 2026-10-15)

### A. Shared durable translation runtime
- `src/text_analysis_lab/core/translate.py` creates a durable SQLite unit plan, queries each source batch, dispatches a translator, writes output parts, commits progress, and creates a resumable checkpoint for every completed unit.
- Sequential units visit `_materialize_plan_unit` → `translate_batch` → `_commit_outputs` → `_save_intermediate_state` → `mark_complete`. Checkpointing writes operator state, writer JSON, and pass metadata then atomically swaps staging/previous directories.
- On grouped batches, `_iter_group_source_plans` computes document grouping and `_query_group_source` builds a group-to-position SQL join followed by an artifact query using returned positions. Examine redundant scans/materialization across batches and repeated passes.
- The writer emits durable key/data Parquet part files and performs sealing validations. Preserve lineage, stable key identity/order, replay/resume correctness and persistence; these are intentional capabilities, not inherently defects.

### B. Simple `subset`
- `src/text_analysis_lab/core/subset.py` builds a keys-only preserved-key artifact. Each batch copies/validates keys, applies a vectorized mask, writes retained keys and checkpoints the frozen Python predicate.
- The predicate alone is likely much cheaper than the full end-to-end operation, but not proven.
- See [focused subset memo](2026-10-15_subset_performance.md) for detailed hypotheses, tests and possible optimizations.

### C. `ArtifactCountVectorizer` (newly observed ~50s)
- `src/text_analysis_lab/translators/artifact_count_vectorizer.py::translation_passes()` returns **two** for grouped batched `fit_translate`: pass 0 gathers global term/document statistics and finalizes the vocabulary; pass 1 revisits tokens and builds output sparse matrix. This is correct for global `min_df`/`max_df` semantics.
- `_build_feature_events()` uses Pandas MultiIndex/group identification, copies/normalizes strings, factorizes sentence boundaries, constructs feature events and validates collisions; even for unigrams this may be measurable compute/allocation overhead.
- `_accumulate_fit_events()` groups token occurrences into per-document term counts then updates Python `Counter` objects. Check profiling before revising scientific logic.
- **Notable checkpoint candidate:** `save_intermediate_state()` serializes the cumulative `_fit_tf`/`_fit_df` vocabulary statistics into a full `fit_statistics.parquet` at **each first-pass checkpoint**; it serializes the fitted `vocabulary.parquet` during later pass checkpoints. This repeats growing state, potentially causing superlinear aggregate bytes/time over many batches. Instrument bytes/elapsed checkpoint duration vs batch number.
- Query cost is incurred twice over the input, and the same global execution plan is reused across passes.
- For roughly 7,000 documents at 150 documents per batch, expect **about 47 units per pass / 94 total** if all input document groups are present; actual count must come from the run. A smaller input doc set changes this estimate.

### D. Other operations worth sampling
- `Project.restrict` and key-domain operations; `Project.aggregate` and grouped matrix aggregation; common `Project.query` / feature summaries; ordinary `CountVectorizer`; and one representative persisted matrix translation.
- Separate genuinely expensive numeric models (`svd_spectrum`, SVD, LDA, UMAP, HDBSCAN) from infrastructure costs. Algorithm runtime is expected and must be explained, not confused with an I/O regression. Audit them in a later tier as time allows.
- Check Python vs DuckDB/Arrow/Pandas conversion costs, especially query pushdown on preserved-key artifacts.

## Audit principle

**First make costs visible, then optimize the largest shared causes.** Preserve correct outputs, frozen operator provenance, resumability/interrupt recovery, deterministic keys/lineage and low peak memory. Do not change the public API just to make a benchmark fast.

## Phase 0 — reproducible measurement harness and timing breakdown (first implementation PR)

Build a benchmark suite outside the CI hot path, with a short smoke benchmark in CI only if stable.

**Datasets**
- Hierarchical spaCy-like tokens keyed by `(row_id, sentence_id, token_id)`, with `text`, `lemma`, `is_alpha`, `is_stop`.
- Source sizes: 10k, 100k, 500k, ~1–1.5m tokens. Corpus sizes ~1k and ~7k docs; realistic skew/long tails, repeated lemmas, 20–80% filter retention, sparse vocabularies.
- Direct source table and restricted/preserved-key source; include source with noncontiguous/interleaved group keys.
- Real AERA project only as instructor-run acceptance benchmark **not committed to repo**; no corpus upload or data leakage.

**Operations**
- `subset` (Boolean flags), `ArtifactCountVectorizer` unigram fit, `restrict`, `aggregate`, a bounded matrix `query`, and one representative model translation.
- For vectorizer: compare `batch_size` 50/150/500/1000; for subset 5k/20k/100k, with one process; compare one-shot in-memory predicate or count transform separately for useful lower bounds.
- Separately compare whole-artifact `ArtifactCountVectorizer` (small datasets only) to batched scientific equivalence; compare fitted-vocabulary streaming and two-pass fitting to isolate fit costs. Do not demand whole-artifact memory use at full scale.

**Instrumentation, stages, and measures**
1. Alias/catalog/project resolution, source-view build and durable plan construction.
2. Group discovery and per-batch source read/position lookup, including inherited-view scans and conversion.
3. Python filter / feature-event creation and TF/DF accumulation.
4. Operator scientific state serialization; writer checkpoint serialization + directory swap; checkpoint bytes and time per batch; track size growth.
5. Output Parquet fragment writing/aggregation; artifact finalization, global key uniqueness/row-order checks, lineage validation.
6. Total wall time, CPU time, peak RSS, I/O bytes/part counts, throughput (input tokens/sec), retries and batch count.
7. Benchmarks on Linux and Windows when available, with TeAL commit, DuckDB/Pandas/PyArrow version, CPU, storage type and cold/warm cache recorded.

Do not leave noisy profiler prints on the public hot path. Use opt-in structured timers/instrumentation or external profiling, with minimal uninstrumented measurement overhead. Keep warmup/repetitions and report distribution (e.g., median and range).

## Phase 1 — shared infrastructure improvements, only with evidence

Potentially: cache/reuse compatible source query plans, reduce duplicate grouped-key position scans, avoid unnecessary full DataFrame conversions/key copies, minimize writer fragmentation, and cut checkpoint copying when equivalent crash safety can be proven.

Checkpoint redesign is especially sensitive. **Do not merely skip checkpointing** or lengthen intervals and silently weaken durability. Consider immutable operator assets plus compact/delta/recoverable fit state, and versioned/append-only checkpoint strategies; test failure at every relevant write/swap boundary.

Keep fast path separate from arbitrary-function safety: do not silently reinterpret `Project.subset(function)` as SQL. SQL expression pushdown might be a separately specified feature.

## Phase 2 — translator-level improvements

- If vectorizer compute dominates: optimize unigram-only feature-event path; avoid needless construction of duplicate feature-label/key string representations when no n-grams; consider streaming aggregate/count implementation maintaining **exact** document frequency, min_df/max_df semantics, deterministic vocab order and keyed output.
- If vectorizer checkpoint dominates: optimize cumulative TF/DF state persistence without affecting resumed outcomes.
- If UMAP/SVD/LDA dominate: benchmark numerical parameter and backend behavior separately, not as generic translation overhead.
- Refactor shared mechanisms once rather than adding many ad hoc special cases to each translator.

## Regression and acceptance gates

- Bitwise/exact keyed contents where appropriate: same document-term feature names and order, same count matrix incl. min_df, max_df and TF/DF; same subset selected keys/order/lineage; no accidental drops, duplicate group counting or cross-sentence bigrams.
- Compare full vs batched matrix building; multiple batch sizes; empty groups and masks; sparse/dense artifacts; preserved-key descendants; interleaved keys.
- Interrupt/restart during both vectorizer fitting and output passes; checkpoint publishing (Windows filesystem semantics included); frozen operators still reusable; aliases/idempotence, deletion and historical provenance intact.
- Existing suites include `tests/test_artifact_count_vectorizer_batched.py`, `tests/test_artifact_count_vectorizer_external.py`, `tests/test_operation_parallel_resume.py`, `tests/test_teal_external_integrations.py`.
- Publish **measured baseline vs optimized timings**, data sizes, hardware, memory cost, comparison methodology and failure/recovery results before declaring an improvement.
- Focus first on common infrastructure. An optimization to the two observed ops only is not enough to call the effort a general performance audit.

## Scope management / handoff

- This is a **planning branch** initially; avoid modifying library source before Stage 0 instruments a useful baseline.
- The earlier `perf/subset-predicate-throughput` branch is the origin of this broader branch. The old branch/memo is preserved for provenance; **use `perf/translation-pipeline-audit` as the active planning branch**.
- The course notebook work is separately tracked in `zacthinks/s062_dev`. Keep performance changes out of that repository.
