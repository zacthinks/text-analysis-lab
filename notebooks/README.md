# Real-world TeAL notebook acceptance

These notebooks complement low-level pytest tests. They use the same **frozen, licensed Wikinews March 2025 fixture** at `tests/fixtures/wikinews_2025_small/articles.jsonl`, so the demonstrations run offline and their source texts can be audited. The binary politics/nonpolitics target in the supervised notebooks is derived from TeAL's *local fixture annotation*, not a claim about Wikinews editorial labeling.

| Notebook | Feature family | Full workflow |
| --- | --- | --- |
| [Standalone real-world acceptance](standalone_translator_real_world_acceptance.ipynb) | Counts → feature trimming → TF-IDF → SVD; structured text | Fit and freeze translators, reopen, compare manual standalone and pipeline replay, inspect paragraphs and learned terms. |
| [Supervised lexical features](supervised_wikinews_lexical_features.ipynb) | Sparse counts, TF-IDF, augmented TF-IDF + text-length fields | Import → split development/audit keys → construct features → CV classifier search → fresh refit → audit inspection → reopen and replay. |
| [Supervised latent features](supervised_wikinews_latent_features.ipynb) | Dense SVD, LDA document-topic mixtures, augmented dense SVD | Import → development/audit split → construct and inspect latent representations → CV classifier search → fresh refit → audit inspection → reopen and replay. |
| [**Full 20 Newsgroups (local)**](supervised_20newsgroups_full_local.ipynb) | **20 topics; full counts, TF-IDF, SVD, LDA, augmented SVD** | **11,314 training and 7,532 independent test texts; multi-class macro-F1 CV, fresh refit, frozen train-only transforms, held-out evaluation and replay. Manual execution, not CI.** |

Run the two supervised notebooks inside a checked-out TeAL repository with the project development dependencies installed. To automatically execute both as actual Jupyter notebooks and preserve their cell outputs:

```bash
uv run python scripts/run_real_world_notebooks.py
```

The dedicated GitHub Actions **Real-world notebooks** workflow also executes them and uploads finished `*.executed.ipynb` artifacts so reviewers can inspect *actual values*, not just a pass/fail summary.

**Methodological limits:** This is a deliberately small corpus (**15 excerpts**), not a performance benchmark or independent population sample. The unsupervised representations are fitted to all **unlabeled** texts before splitting annotations: the exercise is transductive, not a prospective inductive holdout evaluation. Audit labels are not used for classifier search. Do not compare scores as generalizable evidence about representation superiority. For prospective classification, fit learned vocabulary/IDF/latent representations on development texts only, then transform the held-out texts with the frozen pipeline.

Before approving a change, inspect the executed notebook tables and qualitative sections, not just the GitHub green check.

## Full-corpus local benchmark

Open `supervised_20newsgroups_full_local.ipynb` in Jupyter and run all cells. It downloads the scikit-learn **20 Newsgroups** dataset at first use, with the official train/test split. By default it uses all 20 categories and both complete partitions. LDA, SVD and classifier search can take significant time and memory; change `INCLUDE_LDA`, `MAX_VOCAB`, `SVD_COMPONENTS`, `CV_FOLDS`, or `MAX_DEVELOPMENT` only if you want an explicitly reduced debug run. It is deliberately **excluded** from the GitHub real-world notebook runner.

The large notebook learns text representations exclusively on **training documents**, selects classifiers using only training labels, then transforms untouched test documents through frozen TeAL operators. Test labels never enter TeAL's feature pipelines. The small Wikinews examples remain CI smoke checks, not a substitute for this large independent-test demonstration.
