# Text Analysis Lab (TeAL)

Text Analysis Lab (TeAL) is a Python toolkit for building reproducible, inspectable text-analysis workflows. It is designed for research settings where text moves through many transformations—cleaning, decomposition, feature construction, representation learning, coding, aggregation, validation, and analysis—and where it matters to preserve a clear relationship between each derived artifact and the text it came from.

TeAL organizes work around a persistent project and a graph of durable artifacts. Instead of treating every intermediate result as a disposable DataFrame, TeAL records transformations, stable keys, lineage, and provenance so that later results can be traced back to earlier representations and source text.

TeAL is under active development. The public API is usable, but some interfaces may still change before a stable 1.0 release.

## What TeAL supports

TeAL currently includes tools for:

- importing CSV, JSONL, Parquet, text files, and PDFs;
- persistent, keyed text artifacts with lineage and provenance;
- filtering, sampling, splitting, restricting, merging, joining, and aggregation;
- text cleaning, text-length diagnostics, and corpus inspection;
- spaCy sentence and token decomposition, including POS tags, dependencies, entities, and lexical flags;
- count matrices, TF-IDF, feature trimming, normalization, and sparse-matrix workflows;
- dictionaries and lexicons, including researcher-defined and provider-backed dictionaries;
- latent semantic analysis/SVD, LDA, and UMAP;
- locally trained Word2Vec models;
- pretrained sentence-transformer and contextual-transformer representations;
- classification and frozen fitted predictors;
- coreference resolution, semantic-role labeling, and word-sense disambiguation;
- generalized-difference estimation for audited machine-coded measurements;
- optional integration with [GeCo](https://github.com/zacthinks/GeCo) for interactive coding and geometric exploration;
- optional OpenAI Responses API translation with provenance and resumable batching.

A central design goal is to keep corpus-scale transformations inside the project rather than requiring repeated full-corpus round trips through pandas. Small tables can still be materialized when they are useful for inspection, plotting, or bounded analysis.

## Installation

TeAL requires Python 3.10 or newer.

### With `uv`

Clone the repository, enter the repository directory, and run:

```bash
uv sync
```

This installs TeAL and the repository's development tools. To install every optional TeAL feature as well, run:

```bash
uv sync --all-extras
```

To run Python or Jupyter inside the environment:

```bash
uv run python
uv run jupyter lab
```

### With `pip`

A minimal local install is:

```bash
pip install .
```

Optional feature groups can be installed as needed, for example:

```bash
pip install ".[spacy]"
pip install ".[word2vec]"
pip install ".[transformers]"
pip install ".[linguistics]"
pip install ".[llm]"
pip install ".[pdf]"
```

To install all optional TeAL features:

```bash
pip install ".[all]"
```

Some integrations also require external model or data resources. For example, a spaCy English pipeline can be installed with:

```bash
python -m spacy download en_core_web_sm
```

Word-sense disambiguation uses Open English WordNet (OEWN) as its lexical ontology. TeAL does not bundle a separate WordNet extension or fallback lexicon. Download the OEWN data separately:

```bash
python -m wn download 'oewn:2025+'
```

## Quick start

Suppose `documents.csv` contains a `text` column and a `group` metadata column.

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
project.close()
```

The resulting artifact stores the new length measurements while inheriting source text and earlier metadata through preserved-key lineage.

An existing project can be reopened later:

```python
project = teal.Project.open("demo.teal")
documents = project.get_artifact("documents")
```

## Linguistic decomposition

With the `spacy` extra and an installed spaCy model, TeAL can decompose documents into sentence and token artifacts:

```python
from text_analysis_lab.translators import SpacyTranslator

outputs = project.translate(
    SpacyTranslator(model="en_core_web_sm"),
    documents,
)

sentences = outputs["sentences"]
tokens = outputs["tokens"]

print(
    tokens.query(
        data_columns=["text", "lemma", "pos", "dep", "ent_type", "is_stop", "like_num"],
        limit=20,
        form="table",
    )
)
```

Sentence and token artifacts extend the source primary key, which makes it possible to move between document-, sentence-, and token-level analyses while retaining stable links to source records.

## Representations and translation

Most substantive transformations in TeAL are implemented as translators. A translator consumes one or more artifacts and produces one or more new artifacts while recording its configuration and lineage.

Built-in translators include count vectorization, TF-IDF, feature trimming, matrix normalization, SVD/LSA, LDA, UMAP, Word2Vec, sentence-transformer encoding, linguistic models, dictionary coding, and fitted prediction.

Structural project operations such as `subset`, `sample`, `split`, `restrict`, `merge`, `join`, and `aggregate` create new keyed artifacts without requiring users to manage row alignment manually.

## Dictionaries

TeAL supports researcher-defined dictionaries as well as selected external dictionary providers. Provider metadata and provenance travel with dictionary objects and downstream dictionary translations.

```python
import text_analysis_lab as teal

teal.dictionaries.catalog()
teal.dictionaries.download_nltk()  # explicit one-time NLTK data download

hu_liu = teal.dictionaries.hu_liu()
vader = teal.dictionaries.vader()
sentiwordnet = teal.dictionaries.sentiwordnet()
```

The optional `lexicons` extra adds AFINN support.

## Model resources and caches

Large external model assets are cached outside individual TeAL project artifacts so they can be reused across projects. Set the `TEAL_CACHE_DIR` environment variable to override TeAL's default cache location.

Some model-backed translators have additional licensing or resource requirements. In particular, the current WSD reader model has a CC BY-NC-SA 4.0 non-commercial license, so `WordSenseDisambiguator` requires explicit acknowledgement of that license before use.

## GeCo integration

TeAL can link compatible artifacts to GeCo for interactive coding, geometric exploration, classifier development, and focused coding workflows. GeCo is maintained separately at [zacthinks/GeCo](https://github.com/zacthinks/GeCo).

TeAL preserves stable key alignment at the integration boundary so labels and frozen predictors can move between the two systems without relying on row order alone.

## Development

Install the repository environment and run the test suite with:

```bash
uv sync
uv run pytest -q
```

Some tests require optional model stacks or external resources and may be skipped when those dependencies are unavailable.

## License

TeAL is released under the MIT License. See [LICENSE](LICENSE).

Third-party models, datasets, and lexicons may have their own licenses and usage restrictions. Those licenses continue to apply independently of TeAL's MIT license.
