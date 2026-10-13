"""Keep README discovery lists synchronized with public translator/analysis APIs."""
from __future__ import annotations

import re
from pathlib import Path

from text_analysis_lab import analysis, translators
from text_analysis_lab.analysis import generalized_difference_by, text_diagnostics


README = Path(__file__).resolve().parents[1] / "README.md"


def _directory(header: str, next_header: str) -> dict[str, list[str]]:
    source = README.read_text(encoding="utf-8")
    body = source.split(f"## {header}\n", 1)[1].split(f"## {next_header}\n", 1)[0]
    groups = re.findall(
        r"<details>\s*<summary>([^<]+)</summary>(.*?)</details>",
        body, flags=re.DOTALL,
    )
    assert groups, f"Missing collapsible directory for {header}"
    return {
        name: re.findall(r"^- `([^\x60]+)`$", entries, flags=re.MULTILINE)
        for name, entries in groups
    }


def test_translator_directory_includes_public_classes() -> None:
    groups = _directory("Translators", "Analyses")
    listed = {name for names in groups.values() for name in names}
    exported = {
        name for name in translators.__all__
        if isinstance(getattr(translators, name), type)
        and issubclass(getattr(translators, name), translators.CountVectorizer.__bases__[0])
        and getattr(translators, name).__module__.startswith("text_analysis_lab.translators.")
    }
    # LSA/LatentSemanticAnalysis and LocalWord2Vec alias their canonical classes.
    aliases = {"LSA", "LatentSemanticAnalysis", "LocalWord2Vec"}
    assert exported - aliases <= listed
    assert all(names for names in groups.values())


def test_analysis_directory_covers_artifact_bound_methods() -> None:
    from text_analysis_lab.analysis.accessor import ArtifactAnalysis
    groups = _directory("Analyses", "Project Center")
    listed = {name for names in groups.values() for name in names}
    bound = {
        name for name, value in vars(ArtifactAnalysis).items()
        if not name.startswith("_") and callable(value)
    }
    assert bound <= listed
    assert all(callable(getattr(analysis, name)) for name in listed)
    assert analysis.text_diagnostics is text_diagnostics
    assert analysis.generalized_difference_by is generalized_difference_by
