# Hierarchy-aware batching in `Project.translate()`

**Introduced in TeAL 0.4.1.** This document describes the translation *execution* batch policy. It does not change any translator's scientific aggregation unit or fitting algorithm.

## User interface

Given a keyed token artifact whose ordered primary key is `("corpus_id", "collection_id", "document_id", "sentence_id", "token_id")`, supply **one key name**:

```python
outputs = project.translate(
    translator,
    tokens,
    batch_by="document_id",
    batch_size=10,
)
```

TeAL resolves `"document_id"` into the inclusive prefix `("corpus_id", "collection_id", "document_id")`; `batch_size=10` counts **ten complete prefix-key groups**, not ten rows/tokens. Duplicate `document_id` values under distinct ancestors are not the same group. The names themselves have no linguistic meaning to TeAL; its only authority is the bound artifact's **ordered** key.

With `batch_by=None` (the default), legacy positional/row batching is retained. The engine will **not** infer or silently promote a batching level based on a translator.

## Translator requirements

A translator requesting batched source input can specify the finest complete-group boundary it accepts:

```python
from text_analysis_lab.core.operator import SourceRequest, ColumnRequest

return SourceRequest(
    artifact_type="table",
    mode="batches",
    batch_size=request.batch_size or 100,
    columns=ColumnRequest(keys=True, data="text", metadata=False),
    form="table",
    min_batch_by="document_id",
)
```

For the example key, a caller may use `batch_by="corpus_id"`, `"collection_id"`, or `"document_id"`. `"sentence_id"` or `"token_id"` would split required document groups and raises `OperatorError`. Omitting `batch_by` also raises when the minimum level is coarser than the full/finest primary key. A leaf-level minimum allows legacy ordinary row batching.

`min_batch_by` is a translator constraint; it is not an execution preference and is not an aggregation level. Existing positional `SourceRequest` and `TranslationRequest` constructor arguments keep their previous positions.

## Ordering and durability

Batches are planned in first-appearance order of each distinct prefix key, with source rows ordered by original artifact position within each group. Physically interleaved source records may therefore be reordered **across groups**. This is an explicit operational behavior; a translator whose science depends on arbitrary interleaved global row order may not treat all `batch_by` settings as interchangeable.

The engine records selected groups and units durably in SQLite and reuses that saved plan when recovering a failed operation, instead of rediscovering different groups. Both sequential and supported parallel processing use the same source `InputBatch` contract; parallel output commits are restored to planned order. The legacy positional SQLite plan layout remains readable.

## Scope and safety

- Grouped execution in 0.4.1 supports **one source** requesting `mode="batches"`. Multi-source group synchronization is not inferred or supported. Non-grouped multi-source and `mode="full_artifact"` behavior remain available as before.
- `form="single"` is permitted with explicit grouped execution only when `batch_by` is the **finest** primary-key name **and** the effective `batch_size` is **1**. Other combinations may have multiple rows in a single packet and are rejected up front rather than silently losing records.
- A batch contains all rows in the selected groups. In pathological cases, even one giant group can exceed RAM; this feature guarantees **group integrity**, not a fixed byte/row memory ceiling.
- Some translators fit globally and continue requesting `mode="full_artifact"`. Specifically, `ArtifactCountVectorizer` is **not** migrated to batched vocabulary fitting or transformation in 0.4.1. That is a subsequent two-pass architecture effort, not a capability introduced by this release.

## Release version

Version `0.4.1` is an explicit exception to the project's usual pre-1.0 policy that reserves minor releases for new public features: this is a backward-compatible execution-only extension without a planned scientific-semantics change. The exception does not alter the default release policy.
