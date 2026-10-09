from __future__ import annotations

from importlib.metadata import version

import text_analysis_lab as teal


def test_runtime_version_matches_installed_distribution() -> None:
    assert teal.__version__ == version("text-analysis-lab")
