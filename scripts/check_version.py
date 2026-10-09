from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "pyproject.toml").open("rb") as handle:
    version = str(tomllib.load(handle)["project"]["version"])

readme = (ROOT / "README.md").read_text(encoding="utf-8")
changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

errors: list[str] = []

readme_match = re.search(r"Current development release:\s*([^·\s]+)", readme)
if readme_match is None:
    errors.append("README.md has no 'Current development release' version.")
elif readme_match.group(1) != version:
    errors.append(
        f"README.md reports {readme_match.group(1)!r}, but pyproject.toml is {version!r}."
    )

if f"## [{version}]" not in changelog:
    errors.append(
        f"CHANGELOG.md has no release heading for pyproject.toml version {version!r}."
    )

if errors:
    for error in errors:
        print(f"version-check: {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"Version metadata is consistent: {version}")
