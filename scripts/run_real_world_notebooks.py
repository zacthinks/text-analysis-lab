"""Execute actual real-text notebooks as notebook cells, not pytest unit tests.

Usage: python scripts/run_real_world_notebooks.py
Output: notebook-runs/*.executed.ipynb (uploaded as CI artifacts).
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "supervised_wikinews_lexical_features.ipynb",
    "supervised_wikinews_latent_features.ipynb",
)


def main() -> int:
    output_dir = ROOT / "notebook-runs"
    output_dir.mkdir(exist_ok=True)
    failures = []
    for filename in NOTEBOOKS:
        source = ROOT / "notebooks" / filename
        notebook = nbformat.read(source, as_version=4)
        print(f"Executing {filename}", flush=True)
        try:
            finished = NotebookClient(
                notebook, timeout=240, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            nbformat.write(
                finished, output_dir / filename.replace(".ipynb", ".executed.ipynb")
            )
            print(f"PASS {filename}", flush=True)
        except Exception:
            failures.append(filename)
            print(f"FAILED {filename}", file=sys.stderr, flush=True)
            traceback.print_exc()
    if failures:
        print("Notebook failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
