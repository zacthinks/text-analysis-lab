# Subset throughput: profile and optimize keyed predicate filtering

**Date:** 2026-10-15  
**Branch:** `perf/subset-predicate-throughput`  
**Status:** Investigation memo only. No runtime benchmarks or implementation changes yet.  
**Priority:** Record now; schedule after the S-062 Unit 6 notebook is usable.

## Motivation / concrete case

The S-062 Unit 6 application calls:

```python
def keep_alpha_nonstop(frame):
    return (
        frame["is_alpha"].fillna(False).astype(bool)
        & ~frame["is_stop"].fillna(False).astype(bool)
    )

content_tokens = project.subset(
    exploratory_tokens,
    keep_alpha_nonstop,
    data_columns=["is_alpha", "is_stop"],
    key_columns=True,
    form="table",
    iter_batches=True,
    batch_size=20_000,
    workers=1,
    alias="u6_lemma_alpha_nonstop_tokens",
)["output"]
```

**User-observed end-to-end wall time: approximately 20 seconds.** The exact row count of the restricted input, retained fraction, hardware, filesystem, hot/cold state, and TeAL commit used in that observation are not yet established. Earlier full AERA spaCy token output included approximately 1.45 million rows, but do not assume the restricted 50+ word token artifact is identical. The predicate is a trivial vectorized Boolean conjunction; the 20 s measurement covers *all* of TeAL's I/O, keys, lineage and durability, not the Pandas expression alone.

**Goal:** Make simple keyed row selection materially faster without weakening TeAL's guarantees: persistent keyed child artifact; exact subset contents and order; frozen predicate provenance; safely resumable failure handling where promised; metadata/data inherited through preserved-key lineage; matrix/table compatibility and all current query formats.

## Read-only source tracing (TeAL `main`, Oct. 15)

1. **`src/text_analysis_lab/core/subset.py`:** `Project.subset()` wraps `FunctionSubsetTranslator` and calls the generic `Project.translate()` runner. Source request is `mode="batches"` when `iter_batches=True` (default 10,000 if unspecified); only keys + selected data/metadata are requested. User's 20,000-row setting requests roughly `ceil(n_rows/20_000)` execution units.
2. **`subset.py::translate_batch()`:** `_extract_key_frame()` copies full primary keys and `reset_index()`, then `isna().any().any()`; `_predicate_packet()` selects visibility of keys; user function returns mask; `_normalize_mask()` checks shape/NA; `keys.loc[mask].reset_index(drop=True)` allocates retained keys. This is vectorized but performs multiple allocations/validations per batch.
3. **`src/text_analysis_lab/core/translate.py::_build_plan_units()`:** Every batch gets a durable SQLite plan record with a physical-position range. `_run_sequential()` performs `mark_running`, `_materialize_plan_unit` (artifact query with `_position` range), function execution, `_commit_outputs`, `_save_intermediate_state`, and `mark_complete`. Both progress updates are SQLite transactions per unit.
4. **`translate.py::_save_intermediate_state()`:** For resumable subset, **each batch** writes a translator snapshot (including cloudpickled user callable), writer checkpoint JSON, pass metadata, and atomically replaces a staging directory, with directory renames/deletion of previous generation. This may be relatively expensive on Windows and can dominate trivial predicates. It preserves an explicit crash-recovery contract; do not drop it casually.
5. **`src/text_analysis_lab/core/writer.py`:** The selected keys are persisted through `ArtifactWriter.write` and sealed with artifact-wide key validations at `finalize()`. Per-batch Parquet files and writer descriptors incur I/O independent of the predicate.
6. **`include_position=True` default:** The subset input query may carry `_position` in addition to keys/flags; check whether it is needed for semantics or can be skipped in this specific path. For the user's call `key_columns=True` exposes the full three-part keys although the function accesses only two flags; internal key preservation is nevertheless required.
7. **Inherited view:** The input is `project.restrict(spaCy_tokens, to=documents)`, a preserved-key artifact. Repeated queries can traverse inherited key/data relationships; characterize query construction, actual SQL, projection pushdown and per-batch scan cost before proposing a new path.

**Hypotheses, not measurements:** (H1) repeated position-range query + materialization; (H2) per-unit checkpoint file/directory churn; (H3) many small Parquet key fragments + catalog/writer updates; (H4) key frame copies and mask validation; (H5) inherited-view joins. We do **not** know which dominates the 20 s run yet.

## Phase 0: reproducible benchmarks and instrumentation (first PR)

Build a **repeatable end-to-end performance benchmark** that runs on a generated hierarchical-key table and a preserved-key/restricted derivative analogous to spaCy tokens. Keep the test local/synthetic and don't require the private AERA corpus.

- Sizes: 10k, 100k, 500k, 1m+ tokens; realistic three-level `(row_id, sentence_id, token_id)` keys, Boolean `is_alpha`, `is_stop`, `lemma`, mixed retained selectivities (10%, 50%, 80%). Compare source table vs `restrict` derivative.
- Batch sizes: 1,000 / 10,000 / 20,000 / 100,000; `iter_batches=False` only on small datasets to compare fixed overhead; `workers=1` as the baseline. Include fresh project copies or unique aliases to avoid cached-alias reuse; warm/cold OS caches and filesystem platform are recorded.
- Independent timings: raw Pandas predicate on an already-loaded frame; artifact query/materialization over same flags + primary keys; `_extract_key_frame` and `_normalize_mask`; key-write alone; end-to-end `project.subset` including seal/finalize; total bytes/Parquet part count and elapsed checkpoints.
- Instrument monotonic timings in representative stages: plan creation, per-batch `artifact.query`, predicate + key selection, Parquet writer, per-batch checkpoint, and finalization. Avoid noisy ad hoc `print` in the hot path; use an opt-in profiler/benchmark hook or local cProfile.
- Record: baseline absolute times, p50/p95 batch latency, throughput (tokens/sec), retained count and peak RSS. Windows (actual teaching environment if available) and Linux if accessible.
- Verify that **the output key set and order** match an in-memory reference exactly; source keys may be interleaved and have empty batches.

## Phase 1: low-risk improvements, only after measurements

Potential candidate changes, gated by benchmarks:

1. Reduce unnecessary table-frame allocations/key null scans where existing artifact constraints already prove them redundant; ensure direct-output corruption still raises.
2. Avoid unused `_position` materialization by default only when it does not alter a documented predicate input contract; otherwise suggest a user-facing explicit `include_position=False` tuning tip.
3. Reduce repeated SQL/view construction and improve position-range query pushdown. Consider one streaming query/cursor rather than one `artifact.query` per batch, **but** maintain canonical source order, controlled memory and resumability.
4. Optimize output writer buffering/fragments/finalization if measured expensive; avoid giant in-memory accumulation.
5. Make subset's *stateless operator snapshot* cheaper to checkpoint (no repeated heavy callable serialization if immutable), while retaining compatibility with resume and source-independent, frozen predicate behavior.
6. Consider increasing the recommended batch size for simple flags if faster and memory permits. Document measured trade-offs rather than silently changing an API default.

Do **not** add SQL-only optimized subset behavior for arbitrary Python functions by guessing their semantics. A separate opt-in **predicate expression** API could push Boolean filtering down to DuckDB, but would be an explicit API/behavior change requiring design review, query-safety tests, and a clear frozen-provenance representation.

## Phase 2: if benchmarks justify deeper changes

Explore an exact lazy key-domain filtering mode or SQL predicate pushdown **as a distinct capability**, with preservation of key lineage and deterministic scan/order. Compare against ordinary `subset(function)`, and establish its contract for predicates on inherited metadata/data, SQL NULLs, and expression safety. No unreviewed changes to core translation durability.

## Nonnegotiable correctness and regression tests

- Equality of key identity **and order**, `n_rows`, inherited data/metadata, and lineage, for flat and hierarchical keys; empty, all-true, all-false, nullable mask and malformed masks; ambiguous/qualified column names.
- Predicate sees exactly requested `key_columns`, `data_columns`, `metadata_columns`, `include_position`, all supported `form` values.
- Failure on missing columns, non-bool/wrong-length/NA mask as documented; no silent coercion behavior change.
- Frozen/persisted predicate survives source-code deletion or mutation (file-backed and callable versions) and reopens with equivalent semantics.
- Sequential and parallel paths; no duplicate/drop after interruption, before/after checkpoint swap and after writer commit; resume on Linux and Windows.
- `overwrite` and alias idempotence, failed operation cleanup, and persistence across `Project.close/open`.
- Keep existing tests, especially `tests/test_operation_parallel_resume.py`, `tests/test_teal_external_integrations.py`, `tests/test_wikinews_corpus_integration.py`, `tests/test_artifact_count_vectorizer_external.py`.

## Exit criteria and priority

- Demonstrable end-to-end speedup on an approximately one-million-token, two-flag subset with identical keys and all current tests passing; report absolute baseline/new time, ratio and environment.
- No regression for tiny inputs, empty subsets, reopens or restart/resume.
- Treat 20 seconds as the original observed timing, **not** a baseline we can reproduce until profiled on the same project/environment.
- Implementation deferred until the Unit 6 teaching notebook is stable. Keep these changes on this isolated TeAL performance branch, independent of `s062_dev`.
