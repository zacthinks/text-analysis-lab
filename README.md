# Text Analysis Lab (TeAL)

**Current development release: 0.2.0**

Text Analysis Lab (TeAL) is a Python-first toolkit for reproducible, inspectable text analysis. It is designed for research settings where text moves through many transformations—import, cleaning, decomposition, feature construction, representation learning, coding, aggregation, validation, and analysis—and where researchers need to preserve what every derived object means and how it came to exist.

TeAL organizes work around a persistent `Project` and a graph of durable, keyed Artifacts. Rather than treating every intermediate result as a disposable DataFrame, TeAL records stable keys, lineage, operation provenance, operator state, and project metadata so results can be traced back to earlier representations and source text.

TeAL is still pre-1.0. The public API is usable and broadly tested, but interfaces may continue to change as the workflow/task layer and Project Center mature.

## What TeAL supports

TeAL 0.2.0 currently includes:

- CSV, JSONL, Parquet, Excel, folder, TXT, and PDF ingestion;
- persistent keyed Artifacts with explicit lineage and provenance;
- filtering, sampling, splitting, restriction, key selection, merging, joining, aggregation, run collapsing, and primary-key restructuring;
- alias-backed artifact reuse and safe overwrite semantics;
- text cleaning, text-length metadata, corpus inspection, KWIC, and nearest-neighbor analysis;
- spaCy sentence/token decomposition with POS tags, dependencies, entities, and lexical flags;
- count matrices, TF-IDF, feature trimming, normalization, and sparse/dense matrix workflows;
- researcher-defined and provider-backed dictionaries and lexicons;
- SVD/LSA, LDA, UMAP, and local Word2Vec;
- sentence-transformer and contextual-transformer representations;
- classical classification, frozen fitted predictors, and external-result registration;
- coreference resolution, semantic-role labeling, and word-sense disambiguation;
- generalized-difference estimation for audited machine-coded measurements;
- optional GeCo integration for interactive coding and geometric exploration;
- optional OpenAI Responses API translation with durable provenance;
- versioned project, artifact, operation, operator, and standalone memos;
- a localhost-only **TeAL Project Center** with an Artifact Map and Memo Center.

A central design goal is to keep corpus-scale transformations inside the project rather than requiring repeated full-corpus round trips through pandas. Small tables can still be materialized for inspection, plotting, or bounded analysis.

## Installation

TeAL requires Python 3.10 or newer.

### Developing TeAL with `uv`

Clone the repository, enter it, and run:

```bash
uv sync
```

To install every optional feature used by the full integration suite:

```bash
uv sync --all-extras
```

Then run Python or Jupyter inside the environment:

```bash
uv run python
uv run jupyter lab
```

### Installing with `pip`

A minimal local install is:

```bash
pip install .
```

Optional feature groups can be installed as needed:

```bash
pip install ".[pdf]"
pip install ".[spacy]"
pip install ".[word2vec]"
pip install ".[transformers]"
pip install ".[linguistics]"
pip install ".[llm]"
```

Or install all optional TeAL features:

```bash
pip install ".[all]"
```

Some integrations require external model/data resources. For example:

```bash
python -m spacy download en_core_web_sm
python -m wn download 'oewn:2025+'
```

## GPU-enabled PyTorch in downstream research projects

TeAL does **not** choose a CUDA/ROCm/CPU PyTorch build for downstream projects. The research project that owns the environment should make that hardware-specific choice and commit its own `pyproject.toml` and `uv.lock`.

On a new machine, `uv` can be used once to discover an appropriate PyTorch backend. For example, in a disposable environment:

```bash
uv venv .torch-probe
uv pip install --python .torch-probe/Scripts/python.exe torch --torch-backend=auto
```

On POSIX systems use `.torch-probe/bin/python` instead. Inspect the resolved build:

```bash
.torch-probe/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If the machine resolves to a backend such as `cu130`, encode that choice in the **downstream research project's** `pyproject.toml` rather than overriding Torch after `uv sync`:

```toml
[project]
dependencies = [
    "text-analysis-lab[all]",
    "torch",
]

[tool.uv.sources]
text-analysis-lab = { git = "https://github.com/zacthinks/text-analysis-lab.git" }
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

`explicit = true` keeps the PyTorch index from becoming a general package source for unrelated dependencies. Once the backend is encoded, ordinary `uv lock`, `uv sync`, and `uv run` reproduce that project environment without a fragile post-sync Torch override. CPU execution remains available when CUDA is unavailable, although exact numerical results need not be bit-for-bit identical across hardware backends.

See the current [`uv` PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for available backend/index options.

## Quick start

Suppose `documents.csv` contains a `text` column and a `group` metadata column:

```python
import text_analysis_lab as teal
from text_analysis_lab.translators import TextLength

project = teal.Project.create("demo.teal", name="demo")

documents = project.read_csv(
    "documents.csv",
    text_fields="text",
    metadata_fields=["group"],
    alias="documents",
)

lengths = project.translate(
    TextLength({"text": ["words", "characters"]}),
    documents,
    alias="document_lengths",
)["output"]

preview = lengths.query(
    data_columns=["text"],
    metadata_columns=["group", "text_words", "text_characters"],
    metadata_mode="full",
    limit=10,
    form="table",
)

print(preview)
```

An existing project can be reopened later:

```python
project = teal.Project.open("demo.teal")
documents = project.get_artifact("documents")
```

## Project Center

TeAL 0.2.0 introduces a local browser interface backed directly by the project's catalog. It is intentionally a view/controller over TeAL state, not a second source of truth.

Open the general Project Center:

```python
project.launch_project_center()
```

Open directly on a specific tab:

```python
project.launch_artifact_map()
project.launch_memo_center()
```

All three methods launch the same localhost-only application. The current tabs are:

- **Artifacts** — browse the project artifact graph, switch between lineage and operation-provenance edges, inspect disk footprint and matrix dimensions, lazily load `artifact.json`, preview table artifacts page-by-page with selectable key/data/metadata components, and create/edit artifact memos;
- **Memos** — browse/search project, standalone, artifact, operation, and operator memos; create standalone memos; edit Markdown; preview rendered Markdown; and inspect/restore version history.

Memo saves are append-only at the storage layer: editing feels in-place in the interface, but every explicit save creates a new version. Memos are intentionally not deletable through the Project Center.

The Artifact Map remains an initial project-inspection interface. Table preview is deliberately paged and queries only the displayed rows/components; richer representation-specific previews and additional project/workflow views remain planned.

## Linguistic decomposition

With the `spacy` extra and an installed spaCy model, TeAL can decompose documents into sentence and token Artifacts:

```python
from text_analysis_lab.translators import SpacyTranslator

outputs = project.translate(
    SpacyTranslator(model="en_core_web_sm"),
    documents,
)

sentences = outputs["sentences"]
tokens = outputs["tokens"]
```

Sentence/token Artifacts extend the source primary key, preserving stable links back to source records.

## Representations and translation

Most substantive transformations are implemented as translators. A translator consumes one or more Artifacts and produces one or more new Artifacts while recording configuration and lineage.

Built-in translators include count vectorization, TF-IDF, feature trimming, matrix normalization, SVD/LSA, LDA, UMAP, Word2Vec, sentence/contextual transformer encoding, linguistic models, dictionary coding, fitted prediction, and external model/API paths.

UMAP persistence is explicit because fitted nearest-neighbor search state can be much larger than the resulting embedding. Visualization-only use should normally keep the default `reuse="none"`. Use `reuse="recompute"` when later transformation may refit from the immutable fitting Artifact, or `reuse="stored"` to persist transform-capable fitted state. Stored UMAPs use TeAL's exact compact format by default; `storage="native"` preserves upstream serialization and is generally discouraged for large sparse text matrices.

Structural project operations such as `subset`, `sample`, `split`, `probability_split`, `select_keys`, `restrict`, `merge`, `join`, `set_primary_keys`, `collapse_runs`, and `aggregate` create new keyed Artifacts without requiring users to manage row alignment manually.

For analyses computed outside TeAL, `Project.register_external(...)` can register completed results as normal lineage-aware Artifacts while keeping external execution/checkpointing responsibilities outside TeAL.

## GeCo integration

TeAL can link compatible Artifacts to [GeCo](https://github.com/zacthinks/GeCo) for interactive coding, geometric exploration, classifier development, and focused qualitative/computational workflows. Stable identity is preserved across the boundary so labels and frozen predictors can return to TeAL without relying on physical row order.

## Development and testing

Normal clean-clone check:

```bash
uv sync
uv run ruff check .
uv run pytest -q -rs
```

Full optional environment:

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -q -rs
```

Two external integration groups are opt-in because they may download real model resources:

```bash
# PowerShell
$env:TEAL_RUN_HF_EXTERNAL = "1"
$env:TEAL_RUN_LINGUISTICS_EXTERNAL = "1"
uv run pytest -q -rs
```

```bash
# bash/zsh
export TEAL_RUN_HF_EXTERNAL=1
export TEAL_RUN_LINGUISTICS_EXTERNAL=1
uv run pytest -q -rs
```

The 0.2.0 release-preparation checkpoint passed the complete externally enabled suite in the maintainer environment before the Project Center documentation/launcher polish; rerun the full suite locally before publishing a release or major checkpoint.

## Roadmap

The next major architectural layer is not another collection of text algorithms. TeAL's longer-term plan includes:

- persistent methodological **Workflows** composed of durable steps;
- human **Tasks** that can block/unblock workflows and aggregate into a project research to-do list;
- bounded/background execution with explicit resource admission;
- richer Project Center views for tasks, workflows, artifact previews, and project status;
- literature-grounded methodological workflows, including measurement-development/audit designs.

DBYS (Develop Before You Scale) is a natural future methodological workflow, but TeAL does not currently impose DBYS-specific batch selection. Existing `split`, `subset`, `sample`, `select_keys`, `restrict`, and related primitives already support representative, purposive, challenge-seeking, and adaptive Development case selection. Future workflow support should orchestrate and document those choices rather than privilege one sampling rule.

## License

TeAL is released under the MIT License. See [LICENSE](LICENSE).

Third-party models, datasets, lexicons, and external integrations may have their own licenses and usage restrictions; those terms apply independently of TeAL's MIT license.
