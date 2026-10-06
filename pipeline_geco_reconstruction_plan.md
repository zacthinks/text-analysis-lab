# Pipeline Reconstruction and GeCo Integration Memo

## Status

This memo defines the next TeAL pipeline work after PR #6 ("Add transform-only Pipeline infrastructure") merged to `main`.

The purpose is to align three pieces of the architecture that are currently only partially connected:

1. frozen translator semantics,
2. reconstruction of reusable Pipelines from recorded artifact provenance, and
3. GeCo's transformation of new/query text into existing TeAL representation spaces.

The work should proceed conservatively. The core principle is that Pipeline should remain wiring/orchestration over ordinary frozen TeAL translators rather than becoming a second scientific data model or a second translation engine.

---

## 1. First invariant to verify: what does "frozen" mean?

Before implementing provenance-derived Pipelines, audit TeAL's freezing mechanics.

A frozen translator should represent a fixed scientific transformation. Subject to stochasticity intrinsic to the underlying method, the same document passed through the same frozen translator should produce the same scientific output.

That means parameters fall into two categories.

### Scientific / output-determining parameters

These must be captured by the frozen operator state and must not remain freely overridable after freezing.

Examples include:

- model identity and revision,
- fitted vocabulary,
- retained feature mask,
- fitted TF-IDF state,
- fitted dimensionality-reduction state,
- text normalization choices,
- prompts/prefixes that alter model input,
- embedding task/role when it changes the representation,
- normalization choices that change vector values,
- any other option that can change the scientific result for the same input.

If a frozen translator can still receive a semantic parameter at call time that changes the output, that is a violation of the intended architecture unless the variation is explicitly modeled as a different frozen transformation.

### Execution-only parameters

These may vary because they affect how computation is carried out rather than what transformation is defined.

Examples may include:

- batch size,
- worker/process count,
- device placement,
- other scheduling/resource controls that should not alter scientific semantics.

The audit must not assume that an argument is execution-only merely because it currently appears as an operation or `translate(...)` parameter. It must verify whether changing it can alter output semantics.

### Immediate audit target: SentenceTransformerEncoder

Current `SentenceTransformerEncoder` is especially important because it has persisted configuration such as `task`, `prompt_name`, and `prompt`, but its standalone/artifact execution also permits a call-time `task` override.

That is exactly the kind of case the freeze audit must adjudicate.

The old GeCo replay code currently invokes semantic-query encoding with a call-time override such as:

```python
operator.translate(texts, task="query", ...)
```

We should not preserve this behavior automatically merely because the old replay code did it. First decide whether query/document role is part of one frozen translator, two related frozen transformations, or some other explicit representation contract.

### Audit acceptance criterion

After the audit, for every reusable/frozen translator used by Pipeline:

- all scientific/output-determining state is frozen and persisted;
- ordinary standalone Pipeline execution does not need extra semantic kwargs to reproduce the recorded transformation;
- any remaining runtime arguments are demonstrably execution-only;
- save/load round trips preserve the same transformation;
- GeCo or another integration should not need private semantic overrides to make a frozen translator mean something different.

If this criterion fails, fix freezing/translator contracts before relying on artifact-derived Pipelines.

---

## 2. General artifact-interval Pipeline reconstruction

Implement a general constructor that recovers a transform-only Pipeline from TeAL's recorded provenance between an explicit start artifact and end artifact.

Conceptually:

```python
pipeline = project.pipeline(
    start=start_artifact,
    end=end_artifact,
)
```

or an equivalent API such as:

```python
pipeline = Pipeline.from_artifacts(
    project,
    start=start_artifact,
    end=end_artifact,
)
```

The exact public spelling can be decided during implementation. The important contract is the explicit interval.

### Why both start and end are required

Do not infer "the natural beginning" of an end artifact by convention.

The user must identify:

- the artifact whose ordinary Python-equivalent values will become Pipeline input, and
- the artifact whose representation should be reproduced at Pipeline output.

This avoids guessing whether an upstream cleaning, normalization, filtering, or other operation is scientifically part of the representation.

### Simple motivating case

A common representation lineage may be:

```text
documents
  -> CountVectorizer
  -> FeatureTrimmer
  -> TfidfTransformer
  -> SVD
  -> geometry
```

Given `start=documents` and `end=geometry`, TeAL should be able to reconstruct the recorded frozen transformation and execute it over new ordinary Python inputs.

This is the primary initial target.

### Provenance, not just row-lineage

Pipeline reconstruction should primarily follow operation provenance:

- which operation produced an artifact,
- which source artifacts the operation consumed,
- source labels,
- output labels,
- the persisted frozen operator,
- the recorded output chosen from a multi-output operation.

Artifact lineage metadata remains important for validation, but row-lineage rules and computational replay are not identical questions.

In particular, a `new_key` lineage boundary may stop metadata/row-identity ancestry while still corresponding to a perfectly replayable computational operation. Do not mechanically reuse metadata-inheritance traversal rules as the Pipeline reconstruction algorithm.

### Conservative ambiguity policy

The constructor should succeed only when the requested start-to-end computational subgraph is unambiguous and transform-only.

It should fail with a useful diagnostic rather than guess.

Initial support should prioritize:

- one-input / one-output linear chains;
- preserved frozen translator state;
- explicitly recorded source/output labels;
- straightforward DAGs where every required source is itself reachable from the declared start input(s).

For example, this should be straightforward:

```text
A -> B -> C -> D
```

A DAG may also be valid if provenance completely determines it:

```text
      -> B ->
A             D
      -> C ->
```

But this should initially fail for a single declared start:

```text
A -> B --\
         -> D
X -> C --/
```

because `D` also depends on `X`, which is not derivable from `A`.

A later extension may support multiple explicit starts, e.g.:

```python
project.pipeline(
    start={"documents": A, "dictionary": X},
    end=D,
)
```

Do not invent implicit fixed/captured external inputs in the first version.

### Reconstruction should preserve exact frozen operators

The constructor should load the frozen operator snapshots that actually produced the recorded artifacts. It should not create fresh operators from guessed configuration.

The recovered Pipeline is a reusable in-memory execution graph of those frozen transformations.

---

## 3. GeCo currently duplicates Pipeline-like replay logic

Current GeCo integration still uses the private module:

```text
text_analysis_lab.integrations._representation_replay
```

The provider methods:

```python
transform_texts(...)
transform_query(...)
```

walk artifact provenance, load frozen operators, and execute them over ordinary Python/scientific-Python values.

Functionally, this is a private mini-Pipeline engine.

Once general artifact-interval Pipeline reconstruction exists, GeCo should stop owning this separate replay mechanism.

Desired architecture:

```text
TeAL artifact provenance
        |
 start + end
        v
recovered frozen Pipeline
        |
        +--> ordinary Python use
        |
        +--> GeCo transform_texts / semantic search
```

GeCo should become a consumer of the same Pipeline semantics that users can exercise directly.

---

## 4. Revise the GeCo creation contract: representation text vs display text

There is a specific provenance problem in the current GeCo API.

Today, a linked GeCo project can be given:

- a text artifact for display/use in the workspace, and
- one or more geometry artifacts.

Those artifacts need not lie on a single directed computational path.

Example:

```text
                    -> RegexCleaner -> clean_text   (display)
original_text
                    -> CountVectorizer -> LDA/SVD   (geometry)
```

If GeCo only knows `clean_text` as "the text artifact," there is no directed path from `clean_text` to the geometry. Walking backward from the geometry reaches `original_text`, not `clean_text`.

It would be unsafe to guess that `clean_text` is an acceptable representation input merely because it contains text. The cleaning could be cosmetic, or it could materially change the representation.

### Proposed contract

Separate:

1. **representation text**: the artifact that is actually the start point for reconstructing transforms into the registered geometries; and
2. **display text**: an optional artifact/field used for what humans see in GeCo.

If display text is omitted, it defaults to representation text.

Conceptually:

```python
project.geco.create(
    ...,
    text=representation_text,
    display_text=clean_text,  # optional
    geometries=[...],
)
```

Exact naming can be refined, but the semantics should be explicit.

### Validation rule

For each geometry registered in a TeAL-backed GeCo workspace, TeAL must be able to recover a valid transform-only Pipeline from the declared representation-text start artifact to that geometry.

If a geometry is not reachable/replayable from that declared start, workspace creation or geometry registration should fail clearly.

The display artifact is not used to infer the representation transformation.

This eliminates guesswork and makes GeCo's semantic-search/new-text transformation reproducible.

---

## 5. Query vs document embedding semantics require a separate audit

Do not assume all embedding models use query/document prefixes, prompts, or asymmetric task routing.

Possible model contracts include:

- symmetric models where query and document encoding are identical;
- retrieval models trained with distinct query/document prompts or routes;
- models exposing a native task/prompt API;
- models for which adding an arbitrary prefix would be incorrect.

Therefore TeAL must not implement a universal rule such as "prepend `query: ` for semantic search."

The audit should determine, model-by-model / backend-by-backend, how SentenceTransformers and other supported embedding translators define:

- document encoding,
- query encoding,
- prompts/prefixes,
- Router/task semantics,
- normalization behavior.

Whenever query transformation differs scientifically from document transformation, that distinction must be represented explicitly in the frozen architecture rather than smuggled in as an arbitrary runtime override.

This audit is connected to, but logically distinct from, artifact-Pipeline reconstruction.

---

## 6. Proposed workstreams

The work should be split so developers can proceed independently where possible.

### Workstream A - Frozen translator semantics audit

Goal: verify the foundational invariant that a frozen translator fully determines scientific output for a given input.

Tasks:

- inventory reusable translators and standalone `translate(...)` signatures;
- classify call-time parameters as scientific vs execution-only;
- compare constructor state, fitted state, `to_json_state()`, saved assets, and operation parameters;
- test save/load replay;
- identify semantic call-time overrides;
- fix violations before Pipeline reconstruction depends on them.

High-priority target: `SentenceTransformerEncoder.task` and prompt behavior.

Deliverable: tests + fixes + a concise audit summary.

### Workstream B - Artifact interval -> Pipeline constructor

Goal: reconstruct a reusable transform-only Pipeline from explicit start and end artifacts.

Tasks:

- traverse operation provenance between the requested endpoints;
- preserve source labels and output labels;
- load exact frozen operators;
- reject fitting/training stages;
- reject ambiguous/external dependencies not reachable from declared starts;
- return useful diagnostics;
- test linear chains first;
- then test unambiguous DAG cases.

Core acceptance example:

```text
documents -> CountVectorizer -> FeatureTrimmer -> TF-IDF -> SVD
```

Recovered Pipeline output on held-out text should match manual standalone composition and the recorded representation schema.

### Workstream C - GeCo text/display contract

Goal: make the start artifact for representation replay explicit.

Tasks:

- revise linked-workspace creation/geometry registration API as needed;
- distinguish representation text from optional display text;
- default display text to representation text;
- validate every geometry against the declared representation start;
- migrate tests and documentation.

### Workstream D - GeCo uses recovered Pipelines

Goal: remove duplicate private replay orchestration.

Tasks:

- replace most/all of `_representation_replay` traversal/execution with the general constructor;
- make `TeALGeCoProvider.transform_texts()` run the recovered Pipeline;
- handle semantic-query transformation only after Workstream A/E establishes the correct frozen contract;
- retain GeCo-specific error wrapping and stable-key/resource responsibilities, but not a second transformation engine.

### Workstream E - Query/document embedding audit

Goal: establish correct semantic-search encoding behavior.

Tasks:

- inspect supported embedding backends/models;
- determine when query/document transforms are symmetric vs asymmetric;
- determine whether prompts come from model configuration, TeAL configuration, or explicit paired transforms;
- remove unjustified universal prefix assumptions;
- add tests for representative symmetric and asymmetric models.

---

## 7. Recommended sequencing

The dependency order is:

```text
A. freezing audit
        |
        v
B. artifact -> Pipeline constructor
        |
        +------------------+
        |                  |
        v                  v
C. GeCo text/display   E. query/document
   contract               embedding audit
        |                  |
        +--------+---------+
                 v
D. GeCo migration to recovered Pipelines
```

Workstreams C and E can be investigated in parallel once the basic freeze expectations are clear.

Do not make GeCo migration depend on new private replay special cases. Any generally useful transformation behavior should live in frozen translator/Pipeline contracts.

---

## 8. Architectural invariants to preserve

1. **Pipeline is orchestration, not a second data model.**
2. **Frozen translators own scientific transformation semantics.**
3. **Runtime execution controls may vary only when they do not redefine the transformation.**
4. **Artifact-derived Pipelines require explicit start and end points.**
5. **Reconstruction follows recorded operation provenance and exact frozen operators; it does not guess scientific intent.**
6. **Ambiguity should fail loudly rather than be resolved heuristically.**
7. **GeCo should consume general TeAL Pipeline behavior rather than maintain a parallel transformation engine.**
8. **Display text and representation-input text are conceptually different and must be allowed to differ explicitly.**
9. **Query/document embedding asymmetry must be grounded in the model's actual contract, not a universal prefix convention.**
10. **Existing stable-key and artifact provenance guarantees remain authoritative.**

---

## 9. Definition of done for this phase

This phase is complete when:

- freezing semantics have been audited and semantic overrides are either eliminated or explicitly modeled;
- TeAL can reconstruct a transform-only Pipeline between explicit artifact endpoints for the supported unambiguous cases;
- a standard CountVectorizer -> feature subset/trimming -> TF-IDF -> dimensionality-reduction representation can be reconstructed and run on new Python text values;
- GeCo distinguishes representation input text from optional display text;
- registered GeCo geometries are validated as reachable/replayable from the declared representation start;
- GeCo new-text and semantic-query transformation uses the general Pipeline mechanism rather than bespoke lineage replay;
- embedding query/document behavior has explicit, tested semantics;
- failures for ambiguous or unsupported provenance graphs are clear and intentional.
