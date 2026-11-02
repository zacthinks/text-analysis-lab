"""Command-line interface for Text Analysis Lab.

The model commands are project-independent and do not import optional ML runtimes
until verification or explicit installation is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from text_analysis_lab.core.project import Project


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teal",
        description="Text Analysis Lab command-line interface.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gui = commands.add_parser("gui", help="Launch the local TeAL Project Center.")
    gui.add_argument("path", type=Path, help="Path to a TeAL project.")

    models = commands.add_parser("models", help="Manage user-level model resources.")
    models.add_argument(
        "--cache-root", type=Path, default=None,
        help="Override TeAL's user cache root (also supports TEAL_CACHE_DIR).",
    )
    models.add_argument(
        "--hf-cache-dir", type=Path, default=None,
        help="Override the Hugging Face hub cache for inventory inspection.",
    )
    actions = models.add_subparsers(dest="model_command", required=True)
    for command, help_text in (
        ("list", "List recognized resources and candidate locations."),
        ("info", "Show read-only inventory details for one model."),
        ("verify", "Verify a spaCy model can be resolved in this interpreter."),
        ("install", "Resolve or explicitly download an approved spaCy model."),
        ("cache-path", "Show user-level cache paths without creating them."),
    ):
        action = actions.add_parser(command, help=help_text)
        action.add_argument(
            "--json", action="store_true",
            help="Print machine-readable JSON output.",
        )
        if command in {"info", "verify", "install"}:
            action.add_argument("model", help="Model resource ID, e.g. spacy:en_core_web_sm.")
        if command == "install":
            action.add_argument(
                "--allow-download", action="store_true",
                help="Permit a network download of a checksum-pinned official model.",
            )
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _show(value: Any, *, json_output: bool) -> None:
    normalized = _jsonable(value)
    if json_output:
        print(json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if isinstance(normalized, dict):
        for name, item in normalized.items():
            if isinstance(item, (dict, list)):
                print(f"{name}: {json.dumps(item, ensure_ascii=False)}")
            else:
                print(f"{name}: {item}")
        return
    if isinstance(normalized, list):
        for item in normalized:
            if isinstance(item, dict):
                descriptor = item.get("descriptor", {})
                key = descriptor.get("key") or (
                    f"{descriptor.get('provider')}:{descriptor.get('model_id')}"
                )
                state = "present (unverified)" if item.get("present") else "not present"
                print(f"{key}\t{state}")
            else:
                print(item)
        return
    print(normalized)


def _run_models(args: argparse.Namespace) -> int:
    from text_analysis_lab.models import ModelManager
    from text_analysis_lab.models.spacy import SpacyResourceError, resolve_spacy

    manager = ModelManager(cache_root=args.cache_root, hf_cache_dir=args.hf_cache_dir)
    action = args.model_command
    try:
        if action == "cache-path":
            value = manager.cache_paths()
        elif action == "list":
            value = manager.list_models()
        elif action == "info":
            value = manager.inspect(args.model)
        elif action == "verify":
            if not args.model.startswith("spacy:"):
                raise NotImplementedError(
                    "Verification is currently supported for spaCy models only."
                )
            value = resolve_spacy(
                manager.cache_paths()["models"], args.model.removeprefix("spacy:"),
                allow_download=False,
            )
        elif action == "install":
            value = manager.install(
                args.model, allow_download=args.allow_download,
            )
        else:
            raise ValueError(f"Unknown models command: {action}")
    except (SpacyResourceError, KeyError, NotImplementedError, ValueError, OSError) as exc:
        print(f"teal models: {exc}", file=sys.stderr)
        return 2
    _show(value, json_output=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run TeAL's command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        return _run_models(args)

    if args.command == "gui":
        project_path = args.path.expanduser().resolve()
        if not project_path.is_dir():
            parser.error(f"Project directory does not exist: {project_path}")

        project = Project.open(project_path)
        project_center = project.launch_project_center()
        try:
            project_center.wait()
        except KeyboardInterrupt:
            project_center.close()
        finally:
            project.close()
        return 0

    return 0
