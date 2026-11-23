"""Acceptance of real pre-R0 scientific history under the integration branch.

The fixture is made by running tests/fixtures/legacy_pre_r0_fixture.py with
TeAL installed from commit 19e98f4ee1d18630157625495725578d09e81030.
No modern TeAL code is imported in that subprocess.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from text_analysis_lab import Project
from text_analysis_lab.translators import TextLength

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.skipif(
        os.getenv("TEAL_PRE_R0_BASELINE_CHECKOUT") is None,
        reason="Set TEAL_PRE_R0_BASELINE_CHECKOUT to a separate pre-R0 checkout.",
    ),
]


def _run(args: list[str], *, timeout: int = 1200) -> None:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"Failed {args!r}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )


def test_pre_r0_project_science_survives_environment_upgrade(tmp_path: Path):
    baseline = Path(os.environ["TEAL_PRE_R0_BASELINE_CHECKOUT"]).resolve()
    assert (baseline / "pyproject.toml").is_file()
    python_env = tmp_path / "pre-r0-env"
    _run(["uv", "venv", str(python_env), "--python", sys.executable])
    old_python = python_env / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    _run(["uv", "pip", "install", "--python", str(old_python), "-e", f"{baseline}[geco]"])
    _run([
        str(old_python), "-I",
        str(Path(__file__).parent / "fixtures" / "legacy_pre_r0_fixture.py"),
        str(tmp_path / "old-project"),
    ])
    root = tmp_path / "old-project"
    expected = json.loads((tmp_path / "pre-r0-expected.json").read_text(encoding="utf-8"))

    with Project.open(root) as project:
        historical = {
            str(row["operation_id"]): dict(row)
            for row in project.list_operations()
        }
        assert set(expected["historical_operation_ids"]) == set(historical)
        for row in historical.values():
            assert row.get("environment_id") is None
            assert row.get("platform_id") is None
        source = project.get_artifact("legacy_source")
        words = project.get_artifact("legacy_words")
        counts = project.get_artifact("legacy_counts")
        assert source.artifact_id == expected["source_id"]
        assert words.artifact_id == expected["words_id"]
        assert counts.artifact_id == expected["counts_id"]
        assert counts.get_data_columns() == expected["counts_columns"]
        assert counts.get_matrix().toarray().tolist() == expected["counts_matrix"]
        linked = project.geco.open("legacy_coding")
        assert linked.external_provider is not None
        assert linked.coder.geometry_matrix(expected["legacy_geometry_name"]).shape[0] == source.n_rows
        linked.close()
        assert project.resolve_operator_id("legacy_words_operator") == expected["words_operator_id"]
        restored = project.get_operator(expected["words_operator_id"])
        assert isinstance(restored, TextLength)
        replay = project.pipeline(start=source, end=words)
        assert len(replay.stages) == 1
        assert replay.stages[0].translator.operator_id == expected["words_operator_id"]

        resumed = project.resume_operation(expected["failed_operation_id"])["output"]
        assert resumed.status == "complete"
        assert resumed.n_rows == source.n_rows
        # The original event remains historically unattributed even after a
        # modern resume attempt. New attempts may record their real context.
        old = project.get_operation(expected["failed_operation_id"])
        assert old["environment_id"] is None
        assert old["platform_id"] is None
        assert project.execution_attempts(expected["failed_operation_id"])
        fresh = project.translate(TextLength({"text": "words"}), source)["output"]
        assert fresh.n_rows == source.n_rows
        fresh_operation = project.operation_for_artifact(fresh)
        assert fresh_operation["environment_id"] == project.environments.opening_id
        assert fresh_operation["platform_id"] == project.platforms.opening_id

    with Project.open(root) as reopened:
        assert reopened.get_artifact("legacy_counts").artifact_id == expected["counts_id"]
        assert reopened.get_operation(expected["failed_operation_id"])["status"] == "complete"
        assert reopened.resolve_operator_id("legacy_words_operator") == expected["words_operator_id"]
