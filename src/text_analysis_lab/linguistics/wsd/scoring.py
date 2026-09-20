"""Reusable frozen WSL experiment orchestration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from text_analysis_lab.linguistics.hashing import canonical_json, fingerprint
from text_analysis_lab.linguistics.wsd.cache import GlossTextCache
from text_analysis_lab.linguistics.wsd.ontology import OntologyProvider
from text_analysis_lab.linguistics.wsd.types import (
    CacheStats,
    ExperimentSummary,
    GlossRenderConfig,
    ScoredSenseCandidate,
    SenseCandidate,
    UnresolvedWSDTarget,
    WSDTarget,
    WSDTargetSpanPolicy,
)
from text_analysis_lab.linguistics.wsd.wsl_backend import WSLScoringBackend


def score_wsd_targets(
    targets: Sequence[WSDTarget],
    *,
    ontology: OntologyProvider,
    backend: WSLScoringBackend,
    cache_dir: str | Path,
    render_config: GlossRenderConfig | None = None,
    target_span_policy: WSDTargetSpanPolicy = "carrier_only",
) -> tuple[
    list[ScoredSenseCandidate],
    list[UnresolvedWSDTarget],
    CacheStats,
]:
    """Rank explicit targets and return auditable candidate and unresolved records."""

    targets = tuple(targets)
    if target_span_policy not in {"carrier_only", "unique_mwe_envelope"}:
        raise ValueError(
            "target_span_policy must be 'carrier_only' or 'unique_mwe_envelope'"
        )
    render_config = render_config or GlossRenderConfig()
    candidates_by_target: dict[str, tuple[SenseCandidate, ...]] = {}
    all_candidates_by_id: dict[str, SenseCandidate] = {}
    unresolved_targets: list[UnresolvedWSDTarget] = []
    target_candidate_lookup = getattr(ontology, "candidates_for_target", None)
    for target in targets:
        candidates = tuple(
            target_candidate_lookup(target)
            if callable(target_candidate_lookup)
            else ontology.candidates(target.lookup_form, target.pos)
        )
        candidates_by_target[target.target_id] = candidates
        if not candidates:
            unresolved_targets.append(_unresolved(target, "no_candidates"))
        for candidate in candidates:
            existing = all_candidates_by_id.get(candidate.sense_id)
            if existing is not None and existing.gloss != candidate.gloss:
                raise RuntimeError(
                    f"sense ID {candidate.sense_id!r} has multiple gloss payloads in one run"
                )
            all_candidates_by_id.setdefault(candidate.sense_id, candidate)

    cache = GlossTextCache(cache_dir, render_config=render_config)
    cached_glosses, cache_stats = cache.resolve_candidates(
        tuple(all_candidates_by_id.values())
    )

    rows: list[ScoredSenseCandidate] = []
    for target in targets:
        candidates = candidates_by_target[target.target_id]
        if not candidates:
            continue
        reader_target, reader_heading, applied_span_policy = _reader_target_selection(
            target,
            candidates,
            target_span_policy=target_span_policy,
        )
        candidate_texts = {
            candidate.sense_id: _wsl_candidate_text(
                heading=reader_heading,
                gloss_text=cached_glosses[candidate.sense_id].gloss_text,
            )
            for candidate in candidates
        }
        unique_texts = tuple(dict.fromkeys(candidate_texts.values()))
        try:
            scores_by_text = dict(backend.score_target(reader_target, unique_texts))
        except ValueError as exc:
            # Target-specific input limitations (for example, too many candidates or an
            # overlong joint reader input) are ordinary unresolved cases. Systemic model,
            # tokenizer, checkpoint, CUDA, and implementation errors must fail fast rather
            # than silently turning an entire stage into unresolved rows.
            unresolved_targets.append(
                _unresolved(target, f"backend_input_error:{type(exc).__name__}:{exc}")
            )
            continue
        raw_scores = [
            float(scores_by_text[candidate_texts[c.sense_id]]) for c in candidates
        ]
        score_total = sum(max(score, 0.0) for score in raw_scores)
        if score_total <= 0:
            unresolved_targets.append(
                _unresolved(target, "nonpositive_candidate_scores")
            )
            continue
        normalized = [max(score, 0.0) / score_total for score in raw_scores]
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-normalized[i], candidates[i].sense_id),
        )
        ranks = [0] * len(candidates)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        margin = normalized[order[0]] - normalized[order[1]] if len(order) > 1 else None

        for index, candidate in enumerate(candidates):
            cached = cached_glosses[candidate.sense_id]
            model_input_text = candidate_texts[candidate.sense_id]
            rows.append(
                ScoredSenseCandidate(
                    target_id=target.target_id,
                    source_type=target.source_type,
                    document_index=target.document_index,
                    sentence_id=target.sentence_id,
                    sense_id=candidate.sense_id,
                    synset_id=candidate.synset_id,
                    sense_label=candidate.sense_label,
                    aliases=candidate.aliases,
                    ili=candidate.ili,
                    ontology_id=candidate.ontology_id,
                    ontology_version=candidate.ontology_version,
                    lemma=candidate.lemma,
                    pos=target.pos,
                    gloss_text=cached.gloss_text,
                    gloss_hash=cached.gloss_hash,
                    model_input_text=model_input_text,
                    model_input_hash=fingerprint(
                        {"language": "en", "text": model_input_text}
                    ),
                    embedding_cache_key=None,
                    raw_score=raw_scores[index],
                    score_type="wsl_reader_candidate_probability_joint_with_none_and_nme",
                    normalized_score=normalized[index],
                    rank=ranks[index],
                    selected=ranks[index] == 1,
                    top1_margin=margin,
                    source=candidate.source,
                    candidate_kind=str(
                        candidate.metadata.get("candidate_kind") or "singleword"
                    ),
                    candidate_components=tuple(
                        str(item)
                        for item in candidate.metadata.get("mwe_components", ())
                    ),
                    candidate_trigger_lemmas=tuple(
                        str(item)
                        for item in candidate.metadata.get("mwe_trigger_lemmas", ())
                    ),
                    candidate_component_token_indices=tuple(
                        int(item)
                        for item in candidate.metadata.get(
                            "mwe_component_token_indices", ()
                        )
                    ),
                    reader_target_start=reader_target.target_start,
                    reader_target_end=reader_target.target_end,
                    reader_target_text=" ".join(
                        reader_target.tokens[
                            reader_target.target_start : reader_target.target_end
                        ]
                    ),
                    reader_candidate_heading=reader_heading,
                    reader_target_span_policy=applied_span_policy,
                    source_ids=target.source_ids,
                )
            )

    rows.sort(key=lambda row: (row.target_id, row.rank, row.sense_id))
    unresolved_targets.sort(key=lambda row: row.target_id)
    return rows, unresolved_targets, cache_stats


def run_frozen_wsl_experiment(
    targets: Sequence[WSDTarget],
    *,
    ontology: OntologyProvider,
    backend: WSLScoringBackend,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    render_config: GlossRenderConfig | None = None,
    target_span_policy: WSDTargetSpanPolicy = "carrier_only",
) -> tuple[list[ScoredSenseCandidate], ExperimentSummary]:
    """Compatibility wrapper for the standalone frozen WSL experiment."""

    targets = tuple(targets)
    rows, unresolved_targets, cache_stats = score_wsd_targets(
        targets,
        ontology=ontology,
        backend=backend,
        cache_dir=cache_dir,
        render_config=render_config,
        target_span_policy=target_span_policy,
    )
    summary = ExperimentSummary.from_results(
        backend=backend.descriptor(),
        ontology=ontology.descriptor(),
        target_count=len(targets),
        unresolved_targets=unresolved_targets,
        cache=cache_stats,
        rows=rows,
    )
    if output_dir is not None:
        write_experiment_outputs(
            output_dir,
            rows=rows,
            unresolved_targets=unresolved_targets,
            summary=summary,
        )
    return rows, summary


def _reader_target_selection(
    target: WSDTarget,
    candidates: Sequence[SenseCandidate],
    *,
    target_span_policy: WSDTargetSpanPolicy,
) -> tuple[WSDTarget, str, str]:
    """Choose the exact target span and uniform candidate heading seen by WSL.

    ``unique_mwe_envelope`` is intentionally conservative. It expands only when every
    licensed MWE candidate points to the same distinct-token construction match. If no
    MWE is licensed, or multiple construction matches compete, it falls back to the
    original carrier span rather than making an arbitrary pre-disambiguation choice.
    """

    if target_span_policy not in {"carrier_only", "unique_mwe_envelope"}:
        raise ValueError(
            "target_span_policy must be 'carrier_only' or 'unique_mwe_envelope'"
        )

    carrier_heading = " ".join(target.tokens[target.target_start : target.target_end])
    if target_span_policy == "carrier_only":
        return target, carrier_heading, "carrier_only"

    matches = tuple(
        dict.fromkeys(
            tuple(int(index) for index in match)
            for candidate in candidates
            for match in candidate.metadata.get("mwe_component_token_matches", ())
            if match
        )
    )
    if len(matches) != 1:
        fallback = (
            "carrier_only_no_licensed_mwe"
            if not matches
            else "carrier_only_ambiguous_mwe_matches"
        )
        return target, carrier_heading, fallback

    matched_indices = matches[0]
    if len(set(matched_indices)) != len(matched_indices):
        return target, carrier_heading, "carrier_only_invalid_mwe_match"
    if any(index < 0 or index >= len(target.tokens) for index in matched_indices):
        return target, carrier_heading, "carrier_only_invalid_mwe_match"
    ordered_indices = tuple(sorted(matched_indices))
    reader_start = ordered_indices[0]
    reader_end = ordered_indices[-1] + 1
    if not (reader_start <= target.target_start and reader_end >= target.target_end):
        return target, carrier_heading, "carrier_only_unanchored_mwe_match"

    reader_target = target.model_copy(
        update={"target_start": reader_start, "target_end": reader_end}
    )
    heading = " ".join(target.tokens[index] for index in matched_indices)
    return reader_target, heading, "unique_mwe_envelope"


def _wsl_candidate_text(*, heading: str, gloss_text: str) -> str:
    """Render every candidate with one reader-target heading."""

    normalized = " ".join(heading.replace("_", " ").split())
    return f"{normalized}: {gloss_text}"


def _unresolved(target: WSDTarget, reason: str) -> UnresolvedWSDTarget:
    return UnresolvedWSDTarget(
        target_id=target.target_id,
        source_type=target.source_type,
        document_index=target.document_index,
        sentence_id=target.sentence_id,
        lemma=target.lookup_form,
        pos=target.pos,
        reason=reason,
        surface_form=target.surface_form,
        parser_lemma=target.parser_lemma,
        entity_type=target.entity_type,
        source_ids=target.source_ids,
    )


def write_experiment_outputs(
    output_dir: str | Path,
    *,
    rows: Sequence[ScoredSenseCandidate],
    summary: ExperimentSummary,
    unresolved_targets: Sequence[UnresolvedWSDTarget] = (),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_dir / "sense_candidates.jsonl",
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows),
    )
    _write_text_atomic(
        output_dir / "unresolved_targets.jsonl",
        "".join(
            canonical_json(row.model_dump(mode="json")) + "\n"
            for row in unresolved_targets
        ),
    )
    _write_text_atomic(
        output_dir / "summary.json",
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
