# Wikinews 2025 small TeAL integration fixture

This fixture contains 15 short verbatim excerpts from English Wikinews articles published in March 2025. It is intentionally small enough for ordinary integration tests but varied enough to exercise real text, metadata, hierarchy, context, split/subset behavior, and query logic.

- Source: Wikinews
- Attribution: Wikinews
- License: Creative Commons Attribution 4.0 (CC BY 4.0), applicable to Wikinews text published after December 16, 2024.
- The stored texts are verbatim excerpts, not full articles.
- `topic_group` and `region_group` are local TeAL test annotations, not claimed as Wikinews metadata.
- Each JSONL row retains the source article URL for attribution and inspection.

The fixture is frozen: tests should not fetch the live web or resample articles. If the fixture changes, update `manifest.json` and the golden assertions deliberately.
