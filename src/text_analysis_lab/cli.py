"""Command-line interface for Text Analysis Lab."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from text_analysis_lab.core.project import Project


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teal",
        description="Text Analysis Lab command-line interface.",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    gui = commands.add_parser(
        "gui",
        help="Launch the local TeAL Project Center.",
    )
    gui.add_argument(
        "path",
        type=Path,
        help="Path to a TeAL project.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the TeAL command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "gui":
        project_path = args.path.expanduser().resolve()

        if not project_path.is_dir():
            parser.error(f"Project directory does not exist: {project_path}")

        project = Project.open(project_path)

        try:
            project.launch_project_center()
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            project.close()

        return 0

    return 0
