# Changelog

All notable changes to TeAL are documented here.

TeAL follows Semantic Versioning. Before 1.0, backward-incompatible public API changes may occur in a minor release; patch releases are reserved for backward-compatible fixes and maintenance.

## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated

### Removed

## [0.3.0] - 2026-10-09

### Added

- Standalone translation contracts for reusable translators.
- Transform-only `Pipeline` composition and reconstruction from explicit artifact provenance intervals.
- Explicit GeCo `documents`, `text`, and `display_text` roles with Pipeline-backed geometry replay.
- Strict frozen-operator snapshot semantics, schema v2, and an isolated legacy migration path.
- `Project.upgrade_legacy_operators(...)` for explicit, exact, auditable upgrades of eligible pre-strict projects without rerunning analyses.
- Runtime package version reporting through `text_analysis_lab.__version__`.
- A documented SemVer/release workflow with version consistency checks and tag-triggered GitHub releases.

### Changed

- Frozen translators now distinguish scientific state from execution-only controls more strictly.
- Reconstructable external model/resource references can serve as frozen scientific state when exact identity is pinned.
- GeCo Semantic Search and arbitrary new-text transforms now use the same recovered frozen Pipeline.
- SentenceTransformer encoders follow the selected model's normal default prompt behavior unless an explicit prompt override is frozen.
- Column-wise matrix normalization is executable but not reusable for arbitrary new-row replay.
- Legacy project upgrades rebind only completed operations to exact reusable strict replacements; historical-only and incomplete state remains untouched.

### Fixed

- Matrix SQL queries now retain filtering/sorting on relational key, position, and metadata columns while returning an actionable error only when a matrix feature is incorrectly referenced as a SQL column.
- Legacy project upgrades recover interrupted operation-descriptor synchronization and refuse ambiguous prior strict migrations rather than guessing.

### Deprecated

- Historical SentenceTransformer `task` state remains loadable for pre-1.0 compatibility, with a warning; new encoders no longer expose `task` as the preferred public abstraction.

### Removed

- The private GeCo-specific representation replay engine, replaced by general Pipeline reconstruction.

## [0.2.0] - 2026-09-30

Development checkpoint preceding the standalone translator, strict-freeze, Pipeline reconstruction, GeCo replay, and legacy-project upgrade refactors.

## [0.1.0] - 2026-09-10

Initial development release.
