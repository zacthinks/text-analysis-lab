"""Open English WordNet candidate generation for TeAL WSD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from text_analysis_lab.linguistics.cache import user_cache_paths
from text_analysis_lab.linguistics.wsd.mwe import (
    MWE_COMPONENT_NORMALIZATION,
    MWE_INDEX_SCHEMA_VERSION,
    MultiwordLemmaEntry,
    MultiwordLemmaIndex,
    lexical_components,
    load_or_build_multiword_lemma_index,
)
from text_analysis_lab.linguistics.wsd.types import (
    GlossPayload,
    SenseCandidate,
    WSDTarget,
)

POS_MAP = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "AUX": "v",
    "ADJ": "a",
    "ADV": "r",
    "n": "n",
    "v": "v",
    "a": "a",
    "s": "s",
    "r": "r",
}

_WN_RUNTIME_CACHE: dict[tuple[int, str], tuple[Any, Any]] = {}


def normalize_wordnet_pos(pos: str | None) -> str | None:
    if pos is None:
        return None
    return POS_MAP.get(pos.strip()) or POS_MAP.get(pos.strip().upper())


class OntologyProvider(Protocol):
    def descriptor(self) -> Mapping[str, Any]: ...

    def candidates(self, form: str, pos: str) -> Sequence[SenseCandidate]: ...


class OpenEnglishWordNetProvider:
    """Open English WordNet provider with cached token and multiword candidates.

    Ordinary candidate generation follows Morphy analyses of the observed surface form.
    For a single-token target, OEWN multiword lexical entries are first recovered by exact
    component and POS. Before scoring a concrete target, every component of the MWE must
    also occur as a whole sentence token or as a Morphy analysis of a whole sentence token.
    No adjacency, order, distance, or dependency filter is applied.
    """

    def __init__(
        self,
        lexicon: str = "oewn:2025+",
        *,
        mwe_cache_dir: str | Path | None = None,
        include_multiword_candidates: bool = True,
        share_runtime: bool = True,
    ) -> None:
        try:
            import wn
            from wn.morphy import Morphy
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "Open English WordNet support requires the optional `wn` package. "
                "Install TeAL with its NLP dependencies."
            ) from exc
        self.lexicon = lexicon
        self.include_multiword_candidates = bool(include_multiword_candidates)
        self.mwe_cache_dir = Path(
            mwe_cache_dir
            if mwe_cache_dir is not None
            else user_cache_paths().wordnet_mwe_indices
        )
        runtime_key = (id(wn), lexicon)
        runtime = _WN_RUNTIME_CACHE.get(runtime_key) if share_runtime else None
        if runtime is None:
            try:
                wordnet = wn.Wordnet(lexicon)
            except Exception as exc:  # pragma: no cover - local data state
                raise RuntimeError(
                    f"The {lexicon!r} lexicon is not installed. "
                    f"Run `python -m wn download {lexicon}`."
                ) from exc
            morphy = Morphy(wordnet)
            runtime = (wordnet, morphy)
            if share_runtime:
                _WN_RUNTIME_CACHE[runtime_key] = runtime
        self.wordnet, self.morphy = runtime
        self._lemma_cache: dict[
            tuple[str, str], tuple[tuple[str, tuple[str, ...]], ...]
        ] = {}
        self._direct_candidate_cache: dict[
            tuple[str, str], tuple[SenseCandidate, ...]
        ] = {}
        self._candidate_cache: dict[tuple[str, str], tuple[SenseCandidate, ...]] = {}
        self._context_component_cache: dict[
            tuple[str, ...], tuple[frozenset[str], ...]
        ] = {}
        self._mwe_index: MultiwordLemmaIndex | None = None
        self._mwe_index_cache_hit: bool | None = None
        self._mwe_index_path: Path | None = None

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "type": "wn",
            "lexicon": self.lexicon,
            "candidate_lookup": "wn_morphy_surface_plus_distinct_sentence_token_mwe_union",
            "multiword_candidates": self.include_multiword_candidates,
            "multiword_index_schema_version": MWE_INDEX_SCHEMA_VERSION,
            "multiword_component_normalization": MWE_COMPONENT_NORMALIZATION,
            "multiword_filter": "target_anchored_distinct_tokens_raw_casefold_or_morphy",
            "multiword_order_filter": "none",
            "multiword_distance_filter": "none",
            "multiword_dependency_filter": "none",
        }

    def ensure_multiword_index(
        self, *, force_rebuild: bool = False
    ) -> MultiwordLemmaIndex:
        if self._mwe_index is not None and not force_rebuild:
            return self._mwe_index
        index, cache_hit, path = load_or_build_multiword_lemma_index(
            self.wordnet,
            lexicon=self.lexicon,
            cache_dir=self.mwe_cache_dir,
            force_rebuild=force_rebuild,
        )
        self._mwe_index = index
        self._mwe_index_cache_hit = cache_hit
        self._mwe_index_path = path
        if force_rebuild:
            self._candidate_cache.clear()
        return index

    def multiword_index_status(self) -> Mapping[str, Any]:
        index = self.ensure_multiword_index()
        return {
            **index.descriptor(),
            "cache_hit": self._mwe_index_cache_hit,
            "cache_path": str(self._mwe_index_path),
        }

    def lookup_lemma_sources(self, form: str, pos: str) -> dict[str, tuple[str, ...]]:
        """Return each POS-compatible lemma and the Morphy inputs that licensed it."""

        normalized_pos = normalize_wordnet_pos(pos)
        if normalized_pos is None:
            return {}
        surface_form = " ".join(form.replace("_", " ").split())
        casefold_form = surface_form.casefold()
        cache_key = (surface_form, normalized_pos)
        cached = self._lemma_cache.get(cache_key)
        if cached is not None:
            return {lemma: sources for lemma, sources in cached}

        sources_by_lemma: dict[str, set[str]] = {}
        morphy_inputs = (
            (surface_form,)
            if casefold_form == surface_form
            else (surface_form, casefold_form)
        )
        for morphy_input in morphy_inputs:
            analyses = self.morphy(morphy_input, pos=normalized_pos)
            source = "morphy_raw" if morphy_input == surface_form else "morphy_casefold"
            for lemma in analyses.get(normalized_pos, ()):
                sources_by_lemma.setdefault(str(lemma), set()).add(source)

        frozen = tuple(
            (lemma, tuple(sorted(sources)))
            for lemma, sources in sorted(sources_by_lemma.items())
        )
        self._lemma_cache[cache_key] = frozen
        return {lemma: sources for lemma, sources in frozen}

    def lookup_lemmas(self, form: str, pos: str) -> tuple[str, ...]:
        return tuple(self.lookup_lemma_sources(form, pos))

    def multiword_entries(self, form: str, pos: str) -> tuple[MultiwordLemmaEntry, ...]:
        """Return all component- and POS-compatible MWE entries for one token form."""

        normalized_pos = normalize_wordnet_pos(pos)
        if normalized_pos is None or not self.include_multiword_candidates:
            return ()
        surface_form = " ".join(form.replace("_", " ").split())
        if len(lexical_components(surface_form)) != 1:
            return ()
        trigger_lemmas = tuple(self.lookup_lemma_sources(surface_form, normalized_pos))
        if not trigger_lemmas:
            trigger_lemmas = lexical_components(surface_form)
        index = self.ensure_multiword_index()
        by_word_id: dict[str, MultiwordLemmaEntry] = {}
        for trigger in trigger_lemmas:
            for entry in index.lookup(trigger, normalized_pos):
                by_word_id.setdefault(entry.word_id, entry)
        return tuple(
            sorted(
                by_word_id.values(),
                key=lambda item: (item.lemma.casefold(), item.word_id),
            )
        )

    def context_token_component_forms(
        self, tokens: Sequence[str]
    ) -> tuple[frozenset[str], ...]:
        """Return one lexical-analysis bundle per whole sentence token.

        Each bundle contains the token's case-folded raw lexical form plus every
        single-token Morphy analysis recovered from the observed surface form across
        WordNet's noun, verb, adjective, and adverb POS categories. Parser lemmas are
        deliberately excluded. Keeping bundles token-specific lets the MWE licensing
        check enforce that repeated components consume distinct sentence tokens.
        """

        cache_key = tuple(str(token) for token in tokens)
        cached = self._context_component_cache.get(cache_key)
        if cached is not None:
            return cached

        bundles: list[frozenset[str]] = []
        for raw_token in cache_key:
            forms: set[str] = set()
            raw_parts = lexical_components(raw_token)
            if len(raw_parts) == 1:
                forms.add(raw_parts[0])
            for pos in ("n", "v", "a", "r"):
                for lemma in self.lookup_lemmas(raw_token, pos):
                    lemma_parts = lexical_components(lemma)
                    if len(lemma_parts) == 1:
                        forms.add(lemma_parts[0])
            bundles.append(frozenset(forms))

        result = tuple(bundles)
        self._context_component_cache[cache_key] = result
        return result

    def candidates_for_target(self, target: WSDTarget) -> Sequence[SenseCandidate]:
        """Build a target-specific inventory after sentence licensing MWE entries.

        Licensing happens before sense construction and synset deduplication. This avoids
        materializing irrelevant gloss candidates and preserves a licensed lexical carrier
        when several MWE lemmas share one synset.
        """

        normalized_pos = normalize_wordnet_pos(target.pos)
        if normalized_pos is None:
            return ()
        surface_form = " ".join(target.lookup_form.replace("_", " ").split())
        candidates = list(self._direct_candidates(surface_form, normalized_pos))
        if self.include_multiword_candidates:
            token_forms = self.context_token_component_forms(target.tokens)
            trigger_lemmas = tuple(
                self.lookup_lemma_sources(surface_form, normalized_pos)
            )
            for entry in self.multiword_entries(surface_form, normalized_pos):
                component_token_indices = _mwe_entry_token_assignment(
                    entry,
                    trigger_lemmas=trigger_lemmas,
                    target=target,
                    token_forms=token_forms,
                )
                if component_token_indices is None:
                    continue
                candidates.extend(
                    self._candidates_for_multiword_entry(
                        entry,
                        surface_form=surface_form,
                        trigger_lemmas=trigger_lemmas,
                        component_token_indices=component_token_indices,
                    )
                )

        deduplicated: dict[str, SenseCandidate] = {}
        for candidate in candidates:
            existing = deduplicated.get(candidate.synset_id)
            deduplicated[candidate.synset_id] = (
                candidate
                if existing is None
                else _merge_synset_candidates(existing, candidate)
            )
        return tuple(deduplicated.values())

    def candidates(self, form: str, pos: str) -> Sequence[SenseCandidate]:
        """Return ordinary and component/POS-compatible multiword OEWN senses."""

        normalized_pos = normalize_wordnet_pos(pos)
        if normalized_pos is None:
            return ()
        surface_form = " ".join(form.replace("_", " ").split())
        cache_key = (surface_form, normalized_pos)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = list(self._direct_candidates(surface_form, normalized_pos))
        trigger_lemmas = tuple(self.lookup_lemma_sources(surface_form, normalized_pos))
        for entry in self.multiword_entries(surface_form, normalized_pos):
            candidates.extend(
                self._candidates_for_multiword_entry(
                    entry,
                    surface_form=surface_form,
                    trigger_lemmas=trigger_lemmas,
                )
            )

        deduplicated: dict[str, SenseCandidate] = {}
        for candidate in candidates:
            existing = deduplicated.get(candidate.synset_id)
            if existing is None:
                deduplicated[candidate.synset_id] = candidate
            else:
                deduplicated[candidate.synset_id] = _merge_synset_candidates(
                    existing, candidate
                )
        result = tuple(deduplicated.values())
        self._candidate_cache[cache_key] = result
        return result

    def _direct_candidates(self, form: str, pos: str) -> tuple[SenseCandidate, ...]:
        normalized_pos = normalize_wordnet_pos(pos)
        if normalized_pos is None:
            return ()
        surface_form = " ".join(form.replace("_", " ").split())
        cache_key = (surface_form, normalized_pos)
        cached = self._direct_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        casefold_form = surface_form.casefold()
        morphy_inputs = (
            (surface_form,)
            if casefold_form == surface_form
            else (surface_form, casefold_form)
        )
        lemma_sources = self.lookup_lemma_sources(surface_form, normalized_pos)
        lookup_lemmas = tuple(lemma_sources)
        by_synset: dict[str, SenseCandidate] = {}
        for lookup_lemma in lookup_lemmas:
            words = self._matching_words(lookup_lemma, normalized_pos)
            if words:
                for word in words:
                    for candidate in self._candidates_from_word(
                        word,
                        lookup_form=surface_form,
                        lookup_lemmas=lookup_lemmas,
                        lookup_form_sources=lemma_sources.get(lookup_lemma, ()),
                        morphy_inputs=morphy_inputs,
                        candidate_kind="singleword",
                    ):
                        existing = by_synset.get(candidate.synset_id)
                        by_synset[candidate.synset_id] = (
                            candidate
                            if existing is None
                            else _merge_synset_candidates(existing, candidate)
                        )
                continue

            for rank, synset in enumerate(
                self.wordnet.synsets(lookup_lemma, pos=normalized_pos), start=1
            ):
                label = _sense_alias(lookup_lemma, normalized_pos, rank)
                candidate = self._candidate_from_synset(
                    synset,
                    lemma=lookup_lemma,
                    pos=normalized_pos,
                    sense_id=str(_call_or_value(synset, "id")),
                    sense_label=label,
                    native_rank=rank,
                    word_id=None,
                )
                candidate = candidate.model_copy(
                    update={
                        "metadata": {
                            **dict(candidate.metadata),
                            "candidate_kind": "singleword",
                            "lookup_form": surface_form,
                            "morphy_inputs": morphy_inputs,
                            "lookup_forms": lookup_lemmas,
                            "lookup_form_sources": lemma_sources.get(lookup_lemma, ()),
                            "lookup_method": "wn_morphy_raw_union_casefold_surface",
                            "matched_lemmas": (lookup_lemma,),
                            "matched_sense_ids": (candidate.sense_id,),
                            "sense_order_source": "wordnet_synsets_fallback",
                        }
                    }
                )
                existing = by_synset.get(candidate.synset_id)
                by_synset[candidate.synset_id] = (
                    candidate
                    if existing is None
                    else _merge_synset_candidates(existing, candidate)
                )

        result = tuple(by_synset.values())
        self._direct_candidate_cache[cache_key] = result
        return result

    def _matching_words(self, lemma: str, pos: str) -> list[Any]:
        try:
            words = list(self.wordnet.words(lemma, pos=pos))
        except TypeError:
            words = list(self.wordnet.words(lemma))
        words = [word for word in words if str(_call_or_value(word, "pos")) == pos]
        exact = [
            word
            for word in words
            if " ".join(str(_call_or_value(word, "lemma")).replace("_", " ").split())
            == lemma
        ]
        if exact:
            return exact
        casefold = [
            word
            for word in words
            if " ".join(
                str(_call_or_value(word, "lemma")).replace("_", " ").split()
            ).casefold()
            == lemma.casefold()
        ]
        return casefold or words

    def _candidates_from_word(
        self,
        word: Any,
        *,
        lookup_form: str,
        lookup_lemmas: tuple[str, ...],
        lookup_form_sources: tuple[str, ...],
        morphy_inputs: tuple[str, ...],
        candidate_kind: str,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[SenseCandidate, ...]:
        word_id = str(_call_or_value(word, "id"))
        canonical_lemma = " ".join(
            str(_call_or_value(word, "lemma")).replace("_", " ").split()
        )
        pos = str(_call_or_value(word, "pos"))
        result: list[SenseCandidate] = []
        for rank, sense in enumerate(word.senses(), start=1):
            sense_id = str(_call_or_value(sense, "id"))
            synset = sense.synset()
            label = _sense_alias(canonical_lemma, pos, rank)
            candidate = self._candidate_from_synset(
                synset,
                lemma=canonical_lemma,
                pos=pos,
                sense_id=sense_id,
                sense_label=label,
                native_rank=rank,
                word_id=word_id,
            )
            result.append(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **dict(candidate.metadata),
                            "candidate_kind": candidate_kind,
                            "lookup_form": lookup_form,
                            "morphy_inputs": morphy_inputs,
                            "lookup_forms": lookup_lemmas,
                            "lookup_form_sources": lookup_form_sources,
                            "lookup_method": "wn_morphy_surface_plus_sentence_licensed_component_pos_mwe_union",
                            "matched_lemmas": (canonical_lemma,),
                            "matched_sense_ids": (sense_id,),
                            **dict(extra_metadata or {}),
                        }
                    }
                )
            )
        return tuple(result)

    def _candidates_for_multiword_entry(
        self,
        entry: MultiwordLemmaEntry,
        *,
        surface_form: str,
        trigger_lemmas: tuple[str, ...],
        component_token_indices: tuple[int, ...] | None = None,
    ) -> tuple[SenseCandidate, ...]:
        words = self._matching_words(entry.lemma, entry.pos)
        exact_id_words = [
            word for word in words if str(_call_or_value(word, "id")) == entry.word_id
        ]
        if exact_id_words:
            words = exact_id_words
        result: list[SenseCandidate] = []
        for word in words:
            result.extend(
                self._candidates_from_word(
                    word,
                    lookup_form=surface_form,
                    lookup_lemmas=(entry.lemma,),
                    lookup_form_sources=("mwe_component_reverse_index",),
                    morphy_inputs=(surface_form,),
                    candidate_kind="multiword",
                    extra_metadata={
                        "mwe_lemma": entry.lemma,
                        "mwe_word_id": entry.word_id,
                        "mwe_components": entry.components,
                        "mwe_trigger_lemmas": trigger_lemmas,
                        "mwe_component_token_indices": tuple(
                            component_token_indices or ()
                        ),
                        "mwe_component_token_matches": (
                            ()
                            if component_token_indices is None
                            else (tuple(component_token_indices),)
                        ),
                        "mwe_index_schema_version": MWE_INDEX_SCHEMA_VERSION,
                    },
                )
            )
        return tuple(result)

    def resolve_reference(self, reference: str) -> SenseCandidate | None:
        try:
            lemma, pos, _ = reference.rsplit(".", 2)
        except ValueError:
            return None
        for candidate in self._direct_candidates(lemma.replace("_", " "), pos):
            if reference == candidate.sense_label or reference in candidate.aliases:
                return candidate
        return None

    def _candidate_from_synset(
        self,
        synset: Any,
        *,
        lemma: str,
        pos: str,
        sense_id: str,
        sense_label: str,
        native_rank: int,
        word_id: str | None,
    ) -> SenseCandidate:
        ili = _ili_string(synset)
        return SenseCandidate(
            sense_id=sense_id,
            synset_id=str(_call_or_value(synset, "id")),
            ontology_id="Open English WordNet",
            ontology_version=self.lexicon,
            lemma=lemma,
            pos=pos,
            gloss=GlossPayload(
                definition=synset.definition(),
                examples=tuple(synset.examples() or ()),
            ),
            sense_label=sense_label,
            aliases=(sense_label,),
            ili=ili,
            metadata={
                "ili": ili,
                "native_rank": native_rank,
                "word_id": word_id,
                "sense_order_source": "exact_lexical_entry",
            },
        )


def _mwe_entry_token_assignment(
    entry: MultiwordLemmaEntry,
    *,
    trigger_lemmas: Sequence[str],
    target: WSDTarget,
    token_forms: Sequence[frozenset[str]],
) -> tuple[int, ...] | None:
    """Return one deterministic distinct-token assignment for an MWE entry.

    The returned tuple is aligned with ``entry.components``. At least one component
    that triggered reverse-index retrieval must consume a token in the original target
    span. Remaining components may occur in any order and at any distance. Returning
    the assignment, rather than only a Boolean, lets the reader later test an expanded
    construction span without rerunning or approximating lexical matching.
    """

    components = tuple(str(item).casefold() for item in entry.components)
    if len(components) <= 1:
        return tuple(range(target.target_start, target.target_end))[: len(components)]
    normalized_triggers = frozenset(
        str(item).casefold() for item in trigger_lemmas if str(item).strip()
    )
    if not normalized_triggers:
        return None

    choices: list[tuple[int, ...]] = []
    for component in components:
        matches = tuple(
            token_index
            for token_index, forms in enumerate(token_forms)
            if component in forms
        )
        if not matches:
            return None
        choices.append(matches)

    target_indices = frozenset(range(target.target_start, target.target_end))
    anchors = sorted(
        (
            (component_index, token_index)
            for component_index, component in enumerate(components)
            if component in normalized_triggers
            for token_index in choices[component_index]
            if token_index in target_indices
        ),
        key=lambda item: (item[1], item[0]),
    )
    for anchor_component, anchor_token in anchors:
        assignment: list[int | None] = [None] * len(components)
        assignment[anchor_component] = anchor_token
        remaining = [
            index for index in range(len(components)) if index != anchor_component
        ]
        remaining.sort(key=lambda index: (len(choices[index]), index))
        if _assign_distinct_component_tokens(
            remaining,
            choices=choices,
            used={anchor_token},
            assignment=assignment,
        ):
            return tuple(int(token_index) for token_index in assignment)
    return None


def _mwe_entry_is_licensed(
    entry: MultiwordLemmaEntry,
    *,
    trigger_lemmas: Sequence[str],
    target: WSDTarget,
    token_forms: Sequence[frozenset[str]],
) -> bool:
    """Compatibility Boolean wrapper around deterministic assignment."""

    return (
        _mwe_entry_token_assignment(
            entry,
            trigger_lemmas=trigger_lemmas,
            target=target,
            token_forms=token_forms,
        )
        is not None
    )


def _assign_distinct_component_tokens(
    component_indices: Sequence[int],
    *,
    choices: Sequence[Sequence[int]],
    used: set[int],
    assignment: list[int | None],
) -> bool:
    if not component_indices:
        return True
    component_index = component_indices[0]
    for token_index in choices[component_index]:
        if token_index in used:
            continue
        used.add(token_index)
        assignment[component_index] = token_index
        if _assign_distinct_component_tokens(
            component_indices[1:],
            choices=choices,
            used=used,
            assignment=assignment,
        ):
            return True
        assignment[component_index] = None
        used.remove(token_index)
    return False


def _merge_synset_candidates(
    first: SenseCandidate, second: SenseCandidate
) -> SenseCandidate:
    """Merge lexical paths to one concept while preserving an MWE carrier when present."""

    first_kind = str(first.metadata.get("candidate_kind") or "singleword")
    second_kind = str(second.metadata.get("candidate_kind") or "singleword")
    if second_kind == "multiword" and first_kind != "multiword":
        primary, other = second, first
    else:
        primary, other = first, second

    metadata = dict(primary.metadata)
    metadata["matched_lemmas"] = tuple(
        dict.fromkeys(
            (
                *tuple(primary.metadata.get("matched_lemmas", (primary.lemma,))),
                *tuple(other.metadata.get("matched_lemmas", (other.lemma,))),
            )
        )
    )
    metadata["matched_sense_ids"] = tuple(
        dict.fromkeys(
            (
                *tuple(primary.metadata.get("matched_sense_ids", (primary.sense_id,))),
                *tuple(other.metadata.get("matched_sense_ids", (other.sense_id,))),
            )
        )
    )
    metadata["candidate_kinds"] = tuple(dict.fromkeys((first_kind, second_kind)))
    mwe_lemmas = tuple(
        dict.fromkeys(
            str(value)
            for value in (
                first.metadata.get("mwe_lemma"),
                second.metadata.get("mwe_lemma"),
            )
            if value
        )
    )
    if mwe_lemmas:
        metadata["mwe_lemmas"] = mwe_lemmas

    component_token_matches = tuple(
        dict.fromkeys(
            tuple(int(index) for index in match)
            for candidate in (first, second)
            for match in candidate.metadata.get("mwe_component_token_matches", ())
            if match
        )
    )
    if component_token_matches:
        metadata["mwe_component_token_matches"] = component_token_matches
        if len(component_token_matches) == 1:
            metadata["mwe_component_token_indices"] = component_token_matches[0]

    return primary.model_copy(
        update={
            "aliases": tuple(
                dict.fromkeys(
                    (
                        *primary.aliases,
                        *other.aliases,
                        *(
                            ()
                            if primary.sense_label is None
                            else (primary.sense_label,)
                        ),
                        *(() if other.sense_label is None else (other.sense_label,)),
                    )
                )
            ),
            "metadata": metadata,
        }
    )


def _call_or_value(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return value() if callable(value) else value


def _ili_string(synset: Any) -> str | None:
    ili = _call_or_value(synset, "ili")
    if ili is None:
        return None
    value = _call_or_value(ili, "id")
    return str(value if value is not None else ili)


def _sense_alias(lemma: str, pos: str, rank: int) -> str:
    normalized = "_".join(lemma.split())
    return f"{normalized}.{pos}.{rank:02d}"
