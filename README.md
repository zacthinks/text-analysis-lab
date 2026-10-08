# Text Analysis Lab (TeAL)

**Current development release: 0.2.0 · Python 3.11+ · MIT**

Text Analysis Lab (TeAL) is an end-to-end project environment for computer-assisted text analysis. It is designed to support transparent, intentional, rigorous, and reproducible multi-stage work: importing and restructuring corpora, building representations, exploring patterns, developing and validating measures, and carrying results into downstream analysis while preserving the relationships among those stages.

A TeAL `Project` owns durable, keyed `Artifacts` and records the lineage, operations, operator state, metadata, and methodological notes that connect them. The aim is to make complex text-analysis workflows easier to inspect, extend, rerun, and report without manually reconstructing how each intermediate object was produced.

For the broader motivation, architecture, and current project overview, see the [TeAL website](https://zacthinks.github.io/teal/).

> **Development status:** TeAL is still pre-1.0. The public API is usable and broadly tested, but interfaces may continue to change.

## What TeAL supports

TeAL currently includes:

- CSV, JSONL, Parquet, Excel, folder, TXT, and PDF ingestion;
- persistent keyed Artifacts with explicit lineage and operation provenance;
- filtering, sampling, splitting, restriction, key selection, merging, joining, aggregation, run collapsing, and primary-key restructuring;
- alias-backed reuse and safe overwrite semantics;
- corpus inspection, context retrieval, KWIC, summaries, diagnostics, and visualization;
- sparse and dense matrix analytics, including nearest neighbors, distances, row summaries, feature summaries, and matrix summaries;
- spaCy sentence/token decomposition with POS tags, dependencies, entities, and lexical flags;
- count matrices, TF-IDF, feature trimming, normalization, SVD/LSA, LDA, UMAP, and Word2Vec;
- sentence-transformer and contextual-transformer representations;
- researcher-defined and provider-backed dictionaries and lexicons;
- classical classification, frozen fitted predictors, and external-result registration;
- coreference resolution, semantic-role labeling, and word-sense disambiguation;
- generalized-difference estimation for audited machine-coded measurements;
- GeCo integration for interactive exploration, human coding, and classifier development;
- optional OpenAI Responses API translation with durable provenance;
- versioned project, artifact, operation, operator, and standalone memos;
- the localhost-only **TeAL Project Center** for visual project inspection and memoing.

TeAL keeps corpus-scale transformations inside the project whenever possible. Small tables can still be materialized for inspection, plotting, or bounded analysis.

## Installation

TeAL is currently installed directly from GitHub.

### With `uv`

For TeAL with GeCo human coding and exploration:

```bash
uv add "text-analysis-lab[geco] @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

For TeAL with all optional features:

```bash
uv add "text-analysis-lab[all] @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

For only the core dependencies:

```bash
uv add "text-analysis-lab @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

### With `pip`

For TeAL with GeCo human coding and exploration:

```bash
pip install "text-analysis-lab[geco] @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

For TeAL with all optional features:

```bash
pip install "text-analysis-lab[all] @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

For only the core dependencies:

```bash
pip install "text-analysis-lab @ git+https://github.com/zacthinks/text-analysis-lab.git"
```

Many optional integrations are imported lazily. A minimal installation therefore remains usable, and TeAL will raise a targeted error when a requested feature needs an additional dependency. Use `[geco]` if you want TeAL's interactive exploration and human-coding workflows; use `[all]` for the broadest supported feature set.

Some integrations also require external model or data resources. For example:

```bash
python -m spacy download en_core_web_sm
python -m wn download 'oewn:2025+'
```

### GPU-enabled PyTorch

TeAL does not choose a CUDA, ROCm, or CPU PyTorch build for downstream projects. If your workflow uses GPU-backed transformer models, let the research project that owns the environment specify the appropriate PyTorch build in its own `pyproject.toml` / lockfile. See the [`uv` PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for current backend options.

## Quick start

Suppose `documents.csv` contains a `text` column and a `group` metadata column:

```python
import text_analysis_lab as teal
from text_analysis_lab import translators as tr

project = teal.Project.create("demo.teal", name="demo")

documents = project.read_csv(
    "documents.csv",
    text_fields="text",
    metadata_fields=["group"],
    alias="documents",
)

lengths = project.translate(
    tr.TextLength({"text": ["words", "characters"]}),
    documents,
    alias="document_lengths",
)["output"]

preview = lengths.query(
    data_columns=["text"],
    metadata_columns=["group", "text_words", "text_characters"],
    metadata_mode="full",
    limit=10,
)

print(preview)
```

Artifacts are written directly into the project. Reopen the project later and recover important Artifacts by alias:

```python
project = teal.Project.open("demo.teal")
documents = project.get_artifact("documents")
```

## Core project model

A TeAL project is a graph, not a single linear pipeline.

- **Artifacts** are durable keyed objects stored in the project.
- **Stable keys**, not physical row order, align related Artifacts.
- **Lineage** records structural Artifact-to-Artifact relationships.
- **Provenance** records the operations and operators that produced Artifacts.
- **Aliases** provide stable human-readable references and support idempotent reuse on creation/operation calls.
- **Queries and analyses** can materialize ordinary in-memory results without extending the Artifact graph.

Structural operations such as `subset`, `sample`, `split`, `probability_split`, `select_keys`, `restrict`, `merge`, `join`, `set_primary_keys`, `collapse_runs`, and `aggregate` create lineage-aware Artifacts while preserving stable identity.

## Translators

Most substantive transformations are implemented as translators. A translator consumes one or more Artifacts and produces one or more new Artifacts while recording its configuration and provenance.

```python
outputs = project.translate(
    tr.SomeTranslator(...),
    documents,
    alias="result",
)

result = outputs["output"]
```

`project.translate(...)` always returns a mapping from output label to Artifact, including for single-output translators.

Built-in translators include text cleaning and diagnostics, spaCy decomposition, count vectorization, TF-IDF, feature trimming, normalization, SVD/LSA, LDA, UMAP, Word2Vec, sentence/contextual transformer encoding, dictionary coding, fitted prediction, linguistic models, function mapping, and external API/model paths.

## Project Center

The Project Center provides a visual interface to the same TeAL project used from Python. It supports project graph inspection, lineage/provenance views, artifact previews, storage inspection, and versioned memos for projects, artifacts, operations, and operators.

Launch it from Python:

```python
project.launch_project_center()
```

or from the command line:

```bash
teal gui path/to/project
```

## GeCo integration

TeAL can create linked [GeCo](https://github.com/zacthinks/GeCo) workspaces for interactive exploration, qualitative coding, classifier development, and finite human-labeling tasks. Stable TeAL keys are preserved across the boundary so human judgments and frozen predictors can return to the TeAL project.

An exploratory workspace can combine documents with one or more geometries and projections:

```python
geco = project.geco.create(
    "coding",
    documents=documents,
    text_field="text",
    geometry=tfidf,
    projections={"umap": umap},
)

geco.launch()
```

Selected human codes and fitted classifiers can be exported back into TeAL:

```python
labels = geco.export_codes("code name")
predictor = geco.export_classifier("classifier_name")
```

For finite human-labeling tasks, TeAL can create a focused coding workspace with `project.geco.create_focus(...)`.

## External results

Analyses performed outside TeAL can still be registered as project Artifacts.

Use `Project.from_keyed_frame(...)` when external measurements attach new columns to an existing key universe. Use `Project.register_external(...)` when you need more explicit control over provenance and structural basis.

## Upgrading older projects

Projects created before TeAL's strict frozen-operator format may need an explicit compatibility upgrade before their historical transformations can participate in current Pipeline replay. TeAL only upgrades operators it can prove are exact migrations and leaves ambiguous historical state untouched.

Use `project.upgrade_legacy_operators(dry_run=True)` to inspect a project before applying changes. See [docs/UPGRADING.md](docs/UPGRADING.md) for the full procedure and guarantees.

## Development and testing

For contributors working on TeAL itself:

```bash
git clone https://github.com/zacthinks/text-analysis-lab.git
cd text-analysis-lab
uv sync --all-extras
uv run ruff check .
uv run pytest -q -rs
```

## License

TeAL is released under the MIT License. See [LICENSE](LICENSE).

Third-party models, datasets, lexicons, and external integrations may have their own licenses and usage restrictions; those terms apply independently of TeAL's MIT license.
