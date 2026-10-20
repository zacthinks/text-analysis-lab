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

The large notebook learns all text representations exclusively on development documents and evaluates on the official independent test partition. A single `ClassifierSearch` now supports **cumulative trials**: its initial `fit()` compares L2/L-BFGS logistic regression on every available representation (two `C` values per representation), followed by `fit_more()` appending TF-IDF-only L1 and 50/50 Elastic Net configurations using SAGA (two `C` values per penalty). Prior candidates retain their IDs and scores without refitting; the same development keys, CV folds and macro-F1 criterion govern both rounds. With five representations, the notebook runs **14 candidates / 42 CV fits**, sequentially (`n_jobs=1`) with sklearn 1.9's native nested progress callbacks. No SAGA search is run on raw unnormalized counts, which proved prohibitively slow in local diagnostics. Once a winner is selected across both rounds, a fresh final fit and frozen operator produce test predictions. Test labels never enter TeAL's representation training or hyperparameter search. The small Wikinews examples remain CI smoke checks, not a substitute for this large independent-test demonstration.
