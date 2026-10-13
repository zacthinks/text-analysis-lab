from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with (ROOT / "pyproject.toml").open("rb") as handle:
    version = str(tomllib.load(handle)["project"]["version"])

changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
if f"## [{version}]" not in changelog:
    print(
        f"version-check: CHANGELOG.md has no release heading for "
        f"pyproject.toml version {version!r}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"Version metadata is consistent: {version}")
