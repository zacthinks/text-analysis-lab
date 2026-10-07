"""Dump real TeAL linguistic outputs for BoI annotation-rule design.

This is an exploratory script, not production architecture. It runs one shared
spaCy parse over a deliberately varied probe corpus, then reuses those sentence
and token tables for SRL, semantic-head resolution, WSD/sense selection, and
coreference. The result is a plain-text report intended for joint inspection
before a semantic-annotation rule DSL is designed.

Example:

    uv run --extra linguistics python notebooks/boi_annotation_rule_probe.py \
        --output boi_annotation_probe.txt

The script continues when an optional model stage fails so that locally
available outputs are still captured. Use --skip-srl, --skip-wsd, or
--skip-coref when you deliberately do not want to run a model-heavy stage.
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback
from pathlib import Path

import pandas as pd

from text_analysis_lab.translators import (
    CoreferenceResolver,
    SemanticRoleHeadResolver,
    SemanticRoleLabeler,
    SenseSelector,
    SpacyTranslator,
    WordSenseDisambiguator,
)


PROBE_DOCUMENTS = [
    "Three balls rolled.",
    "More than two balls rolled.",
    "At least three balls rolled.",
    "Fewer than five balls rolled.",
    "At most six balls rolled.",
    "Between three and five balls rolled.",
    "Many balls rolled.",
    "A few balls rolled.",
    "Those three red balls rolled quickly.",
    "The very green ball bounced.",
    "John likes all of the apples.",
    "John and Mary ate bread and apples.",
    "John fell thanks to the rain.",
    "Alice saw the book on the mat.",
    "Alice read the book on number theory.",
    "She picked up an apple.",
    "The dog with the green hat barked.",
    "A taller dog stood beside the small dog.",
    "Alice dropped the book. She picked it up.",
    "The company hired two engineers. It promoted one of them later.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boi_annotation_probe.txt"),
        help="Plain-text report path.",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--coref-model",
        default="lingmess",
        choices=("lingmess", "fcoref"),
    )
    parser.add_argument("--skip-srl", action="store_true")
    parser.add_argument("--skip-wsd", action="store_true")
    parser.add_argument("--skip-coref", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Prevent WSD model download; useful once the model is cached.",
    )
    return parser.parse_args()


def frame_text(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "<not produced>"
    if frame.empty:
        return "<empty>"
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        260,
        "display.max_colwidth",
        100,
    ):
        return frame.to_string(index=False)


def section(lines: list[str], title: str, frame: pd.DataFrame | None) -> None:
    lines.extend(
        [
            "",
            "=" * 100,
            title,
            "=" * 100,
            frame_text(frame),
        ]
    )


def stage_error(lines: list[str], title: str, exc: BaseException) -> None:
    lines.extend(
        [
            "",
            "=" * 100,
            f"{title} - ERROR",
            "=" * 100,
            f"{type(exc).__name__}: {exc}",
            "",
            traceback.format_exc(),
        ]
    )


def main() -> int:
    args = parse_args()
    lines: list[str] = [
        "BOI ANNOTATION-RULE PROBE",
        "=" * 100,
        "Exploratory dump of actual TeAL linguistic outputs.",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"spaCy model: {args.spacy_model}",
        f"Device: {args.device}",
        f"Coreference model: {args.coref_model}",
        "",
        "Probe documents:",
    ]
    for i, text in enumerate(PROBE_DOCUMENTS):
        lines.append(f"[{i}] {text}")

    # One canonical parse. Every downstream stage reuses these exact rows.
    try:
        parsed = SpacyTranslator(model=args.spacy_model).translate(PROBE_DOCUMENTS)
        sentences = parsed["sentences"]
        tokens = parsed["tokens"]
    except Exception as exc:
        stage_error(lines, "SPACY", exc)
        args.output.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote partial report to {args.output}")
        return 1

    section(lines, "SPACY - SENTENCES", sentences)
    section(lines, "SPACY - TOKENS / DEPENDENCIES / MORPHOLOGY / NER", tokens)

    srl_outputs: dict[str, pd.DataFrame] = {}
    role_heads: pd.DataFrame | None = None
    if not args.skip_srl:
        try:
            srl_outputs = SemanticRoleLabeler(device=args.device).translate(
                sentences,
                tokens,
                sentence_keys=["source_position", "sentence_id"],
            )
            for label in ("predicates", "role_spans", "failures"):
                section(lines, f"SRL - {label.upper()}", srl_outputs.get(label))

            role_spans = srl_outputs.get("role_spans")
            if role_spans is not None:
                role_heads = SemanticRoleHeadResolver().translate(
                    role_spans,
                    tokens,
                    sentence_keys=["source_position", "sentence_id"],
                )["role_heads"]
                section(lines, "SRL - RESOLVED ROLE HEADS", role_heads)
        except Exception as exc:
            stage_error(lines, "SRL / SEMANTIC HEAD RESOLUTION", exc)
    else:
        lines.append("\nSRL skipped by command-line flag.")

    wsd_outputs: dict[str, pd.DataFrame] = {}
    if not args.skip_wsd:
        try:
            wsd_outputs = WordSenseDisambiguator(
                device=args.device,
                local_files_only=args.local_files_only,
            ).translate(
                tokens,
                token_keys=["source_position", "sentence_id", "token_id"],
            )
            for label in ("candidates", "senses", "unresolved"):
                section(lines, f"WSD - {label.upper()}", wsd_outputs.get(label))

            candidates = wsd_outputs.get("candidates")
            if candidates is not None and not candidates.empty:
                selected = SenseSelector().translate(candidates)["senses"]
                section(lines, "WSD - DEFAULT SENSESELECTOR OUTPUT", selected)
        except Exception as exc:
            stage_error(lines, "WSD / SENSE SELECTION", exc)
    else:
        lines.append("\nWSD skipped by command-line flag.")

    if not args.skip_coref:
        try:
            documents = pd.DataFrame(
                {
                    "source_position": range(len(PROBE_DOCUMENTS)),
                    "text": PROBE_DOCUMENTS,
                }
            )
            coref_outputs = CoreferenceResolver(
                model=args.coref_model,
                device=args.device,
            ).translate(
                documents,
                tokens,
                document_keys=["source_position"],
            )
            for label in ("mentions", "failures"):
                section(lines, f"COREFERENCE - {label.upper()}", coref_outputs.get(label))
        except Exception as exc:
            stage_error(lines, "COREFERENCE", exc)
    else:
        lines.append("\nCoreference skipped by command-line flag.")

    # A compact per-document view at the end makes rule-design inspection easier.
    lines.extend(
        [
            "",
            "#" * 100,
            "PER-DOCUMENT COMPACT VIEWS",
            "#" * 100,
        ]
    )
    for source_position, text in enumerate(PROBE_DOCUMENTS):
        lines.extend(
            [
                "",
                "-" * 100,
                f"[{source_position}] {text}",
                "-" * 100,
            ]
        )

        doc_tokens = tokens.loc[tokens["source_position"] == source_position]
        keep_token_cols = [
            col
            for col in (
                "sentence_id",
                "token_id",
                "text",
                "lemma",
                "pos",
                "tag",
                "morph",
                "dep",
                "head_token_id",
                "ent_iob",
                "ent_type",
            )
            if col in doc_tokens.columns
        ]
        lines.append("\nTOKENS")
        lines.append(frame_text(doc_tokens.loc[:, keep_token_cols]))

        for label in ("predicates", "role_spans"):
            frame = srl_outputs.get(label)
            if frame is not None and "source_position" in frame.columns:
                lines.append(f"\n{label.upper()}")
                lines.append(
                    frame_text(
                        frame.loc[frame["source_position"] == source_position]
                    )
                )

        if role_heads is not None and "source_position" in role_heads.columns:
            lines.append("\nROLE_HEADS")
            lines.append(
                frame_text(
                    role_heads.loc[
                        role_heads["source_position"] == source_position
                    ]
                )
            )

        for label in ("senses", "unresolved"):
            frame = wsd_outputs.get(label)
            if frame is not None and "source_position" in frame.columns:
                lines.append(f"\nWSD_{label.upper()}")
                lines.append(
                    frame_text(
                        frame.loc[frame["source_position"] == source_position]
                    )
                )

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote BoI annotation-rule probe report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
