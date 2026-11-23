"""Generate genuine persisted scientific history with the pre-R0 TeAL baseline.

Run this file with the *baseline* interpreter, not the candidate TeAL checkout.
It deliberately leaves one durable CountVectorizer operation failed mid-plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from text_analysis_lab import Project
from text_analysis_lab.core.translate import _ExecutionPlan
from text_analysis_lab.translators import CountVectorizer, TextLength


def main(root: Path) -> None:
    if root.exists():
        raise RuntimeError("Refusing to overwrite the pre-R0 fixture")
    csv_path = root.parent / "pre-r0-documents.csv"
    pd.DataFrame({
        "text": [
            "alpha cat runs", "alpha dog jumps", "beta cat sleeps",
            "beta dog runs", "gamma cat jumps", "gamma dog sleeps",
        ]
    }).to_csv(csv_path, index=False)

    with Project.create(root, "pre_r0_real_fixture") as project:
        source = project.read_csv(csv_path, text_fields="text", metadata_fields=None)
        project.add_artifact_alias(source, "legacy_source")
        words_op = TextLength({"text": "words"})
        words = project.translate(words_op, source)["output"]
        project.add_artifact_alias(words, "legacy_words")
        project.add_operator_alias(words_op.operator_id, "legacy_words_operator")
        counts = project.translate(
            CountVectorizer(text_field="text", min_df=1), source
        )["output"]
        project.add_artifact_alias(counts, "legacy_counts")
        linked = project.geco.create(
            "legacy_coding", documents=source, text_field="text", geometry=counts
        )
        legacy_geometry_name = linked.coder.geometries()[0]["name"]
        linked.close()

        # Inject failure after the original TeAL has persisted its durable plan,
        # so the candidate release must restore a genuinely old checkpoint.
        original_mark = _ExecutionPlan.mark_complete
        tripped = False

        def fail_once(plan, unit_index):
            nonlocal tripped
            if not tripped:
                tripped = True
                raise RuntimeError("intentional pre-R0 interruption")
            return original_mark(plan, unit_index)

        _ExecutionPlan.mark_complete = fail_once
        try:
            try:
                project.translate(
                    CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=4, binary=True),
                    source,
                    batch_size=2,
                )
            except RuntimeError as exc:
                if "intentional pre-R0 interruption" not in str(exc):
                    raise
            else:
                raise AssertionError("Expected a persisted failed operation")
        finally:
            _ExecutionPlan.mark_complete = original_mark
        assert tripped

        operations = project.list_operations()
        failed = [row for row in operations if row["status"] == "failed"]
        if len(failed) != 1:
            raise AssertionError(f"Expected exactly one failed operation, found {failed!r}")
        payload = {
            "source_id": source.artifact_id,
            "words_id": words.artifact_id,
            "counts_id": counts.artifact_id,
            "words_operator_id": words_op.operator_id,
            "failed_operation_id": failed[0]["operation_id"],
            "historical_operation_ids": [row["operation_id"] for row in operations],
            "counts_columns": counts.get_data_columns(),
            "counts_matrix": counts.get_matrix().toarray().tolist(),
            "legacy_geometry_name": legacy_geometry_name,
        }
    (root.parent / "pre-r0-expected.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: legacy_pre_r0_fixture.py PROJECT_PATH")
    main(Path(sys.argv[1]).resolve())
