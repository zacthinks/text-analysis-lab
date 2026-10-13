# Releasing TeAL

TeAL uses Semantic Versioning.

Before 1.0:

- **patch** (`0.x.Y`) — backward-compatible fixes, tests, documentation, and maintenance;
- **minor** (`0.X.0`) — new public features or meaningful public/API/serialization changes, including breaking pre-1.0 changes;
- **major** (`1.0.0`) — the first stability commitment. After 1.0, ordinary SemVer major/minor/patch rules apply.

## Source of truth

The authoritative package version is:

```toml
[project]
version = "X.Y.Z"
```

in `pyproject.toml`.

Do not maintain a second hard-coded Python version constant. `text_analysis_lab.__version__` is derived from installed package metadata.

The README release label and a matching release heading in `CHANGELOG.md` must agree with `pyproject.toml`; CI checks this.

## Normal development

Do **not** bump the version for every PR.

During a larger multi-PR effort, intermediate phase PRs merge without a version bump. Add user-visible changes under `## [Unreleased]` when appropriate.

When a PR is the **final release-worthy PR for a coherent feature, refactor, or fix**, that PR should normally close the release bookkeeping at the same time:

1. choose the next SemVer version;
2. move the relevant `[Unreleased]` entries into a dated `[X.Y.Z]` section;
3. update `pyproject.toml`;
4. update the README release label;
5. leave a fresh empty `[Unreleased]` section;
6. merge only with normal Tests and Acceptance green.

Examples:

- a standalone backward-compatible bug fix may close as `0.3.1`;
- the final PR in a substantial pre-1.0 feature/refactor may close as `0.4.0`;
- intermediate PRs inside that larger effort do not increment the version.

A dedicated release-preparation PR is **optional**, not the default. Use one when a release boundary is being cut from work that has already merged without the version bookkeeping—for example this initial 0.3.0 versioning bootstrap.

## Tag the released merge commit

After the release-bearing PR merges, tag that exact merge commit as `vX.Y.Z` and push the tag.

Example:

```bash
git checkout main
git pull --ff-only
git tag -a v0.3.0 -m "TeAL 0.3.0"
git push origin v0.3.0
```

The tag is the permanent statement that one exact commit is TeAL version X.Y.Z. It does not modify source files.

## Tag-triggered release checks

Pushing `vX.Y.Z` runs `.github/workflows/release.yml`.

The workflow refuses to create a release unless:

- the tag version exactly matches `pyproject.toml`;
- README and changelog version metadata are consistent;
- the test suite passes;
- Acceptance passes;
- the package builds successfully.

If those checks pass, the workflow creates a GitHub Release and attaches the built package artifacts from that exact tagged commit.

TeAL is currently installed from GitHub, so the workflow does not publish to PyPI. PyPI publication can be added later as a separate explicit decision.
