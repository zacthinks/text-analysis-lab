"""Typed records used by TeAL word-sense disambiguation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

WSDTargetSpanPolicy = Literal["carrier_only", "unique_mwe_envelope"]

from pydantic import BaseModel, ConfigDict, Field, model_validator

from text_analysis_lab.linguistics.hashing import fingerprint


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GlossPayload(FrozenRecord):
    """The exact human-readable lexical material available for one candidate sense."""

    definition: str
    examples: tuple[str, ...] = ()
    language: str = "en"

    @model_validator(mode="after")
    def validate_definition(self) -> GlossPayload:
        if not self.definition.strip():
            raise ValueError("gloss definitions cannot be empty")
        return self


class GlossRenderConfig(FrozenRecord):
    """Controls the exact text sent to the gloss encoder."""

    include_examples: bool = False
    example_prefix: str = "Example: "
    separator: str = " "

    def render(self, payload: GlossPayload) -> str:
        definition = " ".join(payload.definition.split())
        if not self.include_examples or not payload.examples:
            return definition
        examples = [
            f"{self.example_prefix}{' '.join(example.split())}"
            for example in payload.examples
            if example.strip()
        ]
        return self.separator.join([definition, *examples])


class SenseCandidate(FrozenRecord):
    """One explicit candidate supplied by the lexical ontology."""

    sense_id: str
    synset_id: str
    ontology_id: str
    ontology_version: str
    lemma: str
    pos: Literal["n", "v", "a", "r", "s"]
    gloss: GlossPayload
    sense_label: str | None = None
    aliases: tuple[str, ...] = ()
    ili: str | None = None
    source: Literal["ontology"] = "ontology"
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class WSDTarget(FrozenRecord):
    """One explicit target occurrence to disambiguate.

    ``lemma`` is retained as the lookup-form field for compatibility with the
    WSD scoring pipeline. Production targets set it from the observed
    surface form, never from the parser lemma. ``parser_lemma`` is provenance
    only and may be overridden downstream by the selected ontology sense.
    """

    target_id: str
    source_type: str
    document_index: int = Field(ge=0)
    sentence_id: int = Field(ge=0)
    tokens: tuple[str, ...]
    target_start: int = Field(ge=0)
    target_end: int = Field(ge=1)
    lemma: str
    pos: Literal["n", "v", "a", "r", "s"]
    surface_form: str | None = None
    parser_lemma: str | None = None
    entity_type: str | None = None
    source_ids: Mapping[str, int | str | None] = Field(default_factory=dict)

    @property
    def lookup_form(self) -> str:
        return self.surface_form or self.lemma

    @model_validator(mode="after")
    def validate_target_span(self) -> WSDTarget:
        if self.target_end <= self.target_start:
            raise ValueError("target_end must be greater than target_start")
        if self.target_end > len(self.tokens):
            raise ValueError("target span exceeds the supplied context tokens")
        return self


class ScoredSenseCandidate(FrozenRecord):
    """Auditable model output for one target-candidate pair.

    ``lemma`` is the ontology lemma attached to this candidate. On the selected
    row it is the resolved lemma that overrides, but never overwrites, the
    parser lemma for downstream lexical and ontology features.
    """

    target_id: str
    source_type: str
    document_index: int
    sentence_id: int
    sense_id: str
    synset_id: str
    sense_label: str | None = None
    aliases: tuple[str, ...] = ()
    ili: str | None = None
    ontology_id: str
    ontology_version: str
    lemma: str
    pos: str
    gloss_text: str
    gloss_hash: str
    model_input_text: str
    model_input_hash: str
    embedding_cache_key: str | None = None
    raw_score: float
    score_type: str = "model_score"
    normalized_score: float
    rank: int = Field(ge=1)
    selected: bool
    top1_margin: float | None = None
    source: str
    candidate_kind: str = "singleword"
    candidate_components: tuple[str, ...] = ()
    candidate_trigger_lemmas: tuple[str, ...] = ()
    candidate_component_token_indices: tuple[int, ...] = ()
    reader_target_start: int = Field(default=0, ge=0)
    reader_target_end: int = Field(default=1, ge=1)
    reader_target_text: str = ""
    reader_candidate_heading: str = ""
    reader_target_span_policy: str = "carrier_only"
    source_ids: Mapping[str, int | str | None] = Field(default_factory=dict)


class UnresolvedWSDTarget(FrozenRecord):
    target_id: str
    source_type: str
    document_index: int
    sentence_id: int
    lemma: str
    pos: str
    reason: str
    surface_form: str | None = None
    parser_lemma: str | None = None
    entity_type: str | None = None
    source_ids: Mapping[str, int | str | None] = Field(default_factory=dict)


class NamedEntityConcept(FrozenRecord):
    """One deterministic ontology concept supplied by a named-entity type."""

    target_id: str
    entity_type: str
    mapping_label: str
    candidate: SenseCandidate


class CacheStats(FrozenRecord):
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    unique_glosses: int = Field(default=0, ge=0)


class ExperimentSummary(FrozenRecord):
    backend: Mapping[str, Any]
    ontology: Mapping[str, Any]
    target_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    unresolved_target_count: int = Field(ge=0)
    unresolved_reason_counts: Mapping[str, int] = Field(default_factory=dict)
    cache: CacheStats
    result_fingerprint: str

    @classmethod
    def from_results(
        cls,
        *,
        backend: Mapping[str, Any],
        ontology: Mapping[str, Any],
        target_count: int,
        unresolved_targets: Sequence[UnresolvedWSDTarget],
        cache: CacheStats,
        rows: list[ScoredSenseCandidate],
    ) -> ExperimentSummary:
        compact_rows = [
            {
                "target_id": row.target_id,
                "sense_id": row.sense_id,
                "gloss_hash": row.gloss_hash,
                "raw_score": row.raw_score,
                "rank": row.rank,
                "selected": row.selected,
            }
            for row in rows
        ]
        reason_counts = Counter(row.reason for row in unresolved_targets)
        return cls(
            backend=dict(backend),
            ontology=dict(ontology),
            target_count=target_count,
            candidate_count=len(rows),
            unresolved_target_count=len(unresolved_targets),
            unresolved_reason_counts=dict(sorted(reason_counts.items())),
            cache=cache,
            result_fingerprint=fingerprint(compact_rows),
        )
