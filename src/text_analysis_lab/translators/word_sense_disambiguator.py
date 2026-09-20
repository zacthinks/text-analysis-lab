"""Teaching-scale token word-sense disambiguation for TeAL."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from text_analysis_lab._linguistics.cache import user_cache_paths

DEFAULT_WSL_READER_MODEL = "Babelscape/wsl-reader-deberta-v3-base"
DEFAULT_WSL_READER_REVISION = "809d05bd12f261d26b42e28dc2b31db430c1585c"
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact

TOKENS = "tokens"
SENSES = "senses"
CANDIDATES = "candidates"
UNRESOLVED = "unresolved"

SENSE_DATA_COLUMNS = (
    "surface_form",
    "parser_lemma",
    "resolved_lemma",
    "lemma_overrides_parser",
    "pos",
    "sense_id",
    "synset_id",
    "sense_label",
    "aliases",
    "ili",
    "ontology_id",
    "ontology_version",
    "gloss_text",
    "normalized_score",
    "top1_margin",
    "candidate_kind",
    "model_name",
    "model_revision",
)
CANDIDATE_DATA_COLUMNS = (
    "sense_id",
    "synset_id",
    "sense_label",
    "candidate_lemma",
    "aliases",
    "ili",
    "ontology_id",
    "ontology_version",
    "gloss_text",
    "gloss_hash",
    "model_input_text",
    "model_input_hash",
    "raw_score",
    "score_type",
    "normalized_score",
    "rank",
    "selected",
    "top1_margin",
    "candidate_source",
    "candidate_kind",
    "candidate_components",
    "candidate_trigger_lemmas",
    "candidate_component_token_indices",
    "reader_target_start",
    "reader_target_end",
    "reader_target_text",
    "reader_candidate_heading",
    "reader_target_span_policy",
    "model_name",
    "model_revision",
)
UNRESOLVED_DATA_COLUMNS = ("surface_form", "parser_lemma", "pos", "reason")


class WordSenseDisambiguator(BaseTranslator):
    """Disambiguate every WordNet-eligible token in a selected TeAL token artifact.

    This translator intentionally has no artificial token cap.  Unit 10 should run it on a
    sentence or another tiny subset so students can see the computational cost of modern
    lexical-semantic models.  The same artifact contract can later support the TeAL-native
    Bag-of-Ideas rewrite, including semantic-head targets.
    """

    operation_type = "translate"

    def __init__(
        self,
        *,
        lexicon: str = "oewn:2025+",
        model_name: str = DEFAULT_WSL_READER_MODEL,
        model_revision: str | None = DEFAULT_WSL_READER_REVISION,
        sentence_key: str = "sentence_id",
        token_key: str = "token_id",
        device: str = "auto",
        precision: str | int = 32,
        include_multiword_candidates: bool = True,
        target_span_policy: str = "carrier_only",
        local_files_only: bool = False,
        acknowledge_noncommercial_license: bool = False,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(operator_id=operator_id)
        if not lexicon:
            raise ValueError("lexicon must be non-empty.")
        if not model_name:
            raise ValueError("model_name must be non-empty.")
        if not sentence_key or not token_key or sentence_key == token_key:
            raise ValueError(
                "sentence_key and token_key must be distinct non-empty names."
            )
        if target_span_policy not in {"carrier_only", "unique_mwe_envelope"}:
            raise ValueError(
                "target_span_policy must be 'carrier_only' or 'unique_mwe_envelope'."
            )
        if not acknowledge_noncommercial_license:
            raise ValueError(
                "The Babelscape WSL reader is CC BY-NC-SA 4.0. Pass "
                "acknowledge_noncommercial_license=True only for eligible non-commercial "
                "research or education use."
            )
        self.lexicon = str(lexicon)
        self.model_name = str(model_name)
        self.model_revision = None if model_revision is None else str(model_revision)
        self.sentence_key = str(sentence_key)
        self.token_key = str(token_key)
        self.device = str(device)
        self.precision = str(precision)
        self.include_multiword_candidates = bool(include_multiword_candidates)
        self.target_span_policy = str(target_span_policy)
        self.local_files_only = bool(local_files_only)
        self.acknowledge_noncommercial_license = True

    def output_specs(
        self, *, sources: Mapping[str, BaseArtifact], request: TranslationRequest
    ):
        _ = request
        tokens = _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        if "candidate_id" in tokens.primary_key:
            raise OperatorError(
                "WordSenseDisambiguator candidate_id collides with token keys."
            )
        return {
            SENSES: OutputSpec(
                artifact_type="table", lineage_mode="preserved_key", basis_labels=TOKENS
            ),
            CANDIDATES: OutputSpec(
                artifact_type="table", lineage_mode="extended_key", basis_labels=TOKENS
            ),
            UNRESOLVED: OutputSpec(
                artifact_type="table", lineage_mode="preserved_key", basis_labels=TOKENS
            ),
        }

    def validate_operation_params(self, params, *, sources, mode):
        _ = sources, mode
        if params:
            raise OperatorError(
                f"WordSenseDisambiguator does not accept operation parameters; got {sorted(params)}."
            )
        return {}

    def input_request(self, *, sources, mode, request):
        _ = mode, request
        _validate_sources(
            sources, sentence_key=self.sentence_key, token_key=self.token_key
        )
        return {
            TOKENS: SourceRequest(
                artifact_type="table",
                mode="full_artifact",
                columns=ColumnRequest(
                    keys=True,
                    data=("text", "lemma", "pos", "ent_type"),
                    metadata=False,
                ),
                form="table",
                metadata_mode="none",
                include_position=False,
            )
        }

    def translate_batch(self, inputs, *, mode, request):
        _ = mode, request
        if set(inputs) != {TOKENS}:
            raise OperatorError(
                f"WordSenseDisambiguator expects one input under {TOKENS!r}."
            )
        packet = inputs[TOKENS]
        tokens = _frame(packet.data)
        token_keys = list(packet.primary_key)
        if len(token_keys) < 2 or token_keys[-2:] != [
            self.sentence_key,
            self.token_key,
        ]:
            raise ArtifactError(
                "WordSenseDisambiguator expects token primary keys to end in "
                f"{self.sentence_key!r}, {self.token_key!r}."
            )
        required = [*token_keys, "text", "lemma", "pos", "ent_type"]
        missing = [name for name in required if name not in tokens]
        if missing:
            raise ArtifactError(
                f"WordSenseDisambiguator token source is missing {missing}."
            )

        document_keys = token_keys[:-2]
        sentence_group_keys = [*document_keys, self.sentence_key]
        sentence_tokens: dict[tuple[int, ...], pd.DataFrame] = {}
        for raw_key, group in tokens.groupby(
            sentence_group_keys, sort=False, dropna=False
        ):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            ordered = group.sort_values(self.token_key).reset_index(drop=True)
            # spaCy legitimately emits SPACE tokens for control/whitespace runs
            # (for example a form-feed separating title and abstract).  Such
            # tokens carry no lexical content and some WordPiece/SentencePiece
            # tokenizers encode them as zero subwords.  They must not enter the
            # WSL reader context, but their TeAL token rows remain untouched.
            reader_mask = ordered["text"].fillna("").astype(str).str.strip().ne("")
            sentence_tokens[tuple(int(v) for v in key)] = ordered.loc[
                reader_mask
            ].reset_index(drop=True)

        document_ordinals: dict[tuple[int, ...], int] = {}
        target_keys: dict[str, dict[str, int]] = {}
        target_context: dict[str, tuple[str, str | None, str]] = {}
        targets: list[WSDTarget] = []
        for _, row in tokens.sort_values(token_keys).iterrows():
            raw_pos = None if pd.isna(row["pos"]) else str(row["pos"])
            wn_pos = _normalize_wordnet_pos(raw_pos)
            if wn_pos is None:
                continue
            document_key = tuple(int(row[name]) for name in document_keys)
            if document_key not in document_ordinals:
                document_ordinals[document_key] = len(document_ordinals)
            sentence_id = int(row[self.sentence_key])
            token_id = int(row[self.token_key])
            sentence_key = (*document_key, sentence_id)
            context = sentence_tokens.get(sentence_key)
            if context is None or context.empty:
                continue
            local_matches = context.index[
                context[self.token_key].astype(int) == token_id
            ].tolist()
            if len(local_matches) != 1:
                raise ArtifactError(
                    f"Token {sentence_key + (token_id,)} does not align uniquely within its sentence."
                )
            local_index = int(local_matches[0])
            surface = str(row["text"])
            parser_lemma = surface if pd.isna(row["lemma"]) else str(row["lemma"])
            target_id = (
                f"tokens:{document_ordinals[document_key]}:{sentence_id}:{token_id}"
            )
            key_record = {name: int(row[name]) for name in token_keys}
            target_keys[target_id] = key_record
            target_context[target_id] = (surface, parser_lemma, str(raw_pos or ""))
            from text_analysis_lab._linguistics.wsd.types import WSDTarget

            targets.append(
                WSDTarget(
                    target_id=target_id,
                    source_type="tokens",
                    document_index=document_ordinals[document_key],
                    sentence_id=sentence_id,
                    tokens=tuple(str(value) for value in context["text"]),
                    target_start=local_index,
                    target_end=local_index + 1,
                    lemma=" ".join(surface.replace("_", " ").split()),
                    pos=wn_pos,
                    surface_form=surface,
                    parser_lemma=" ".join(parser_lemma.replace("_", " ").split()),
                    entity_type=None
                    if pd.isna(row["ent_type"])
                    else str(row["ent_type"]),
                    source_ids={"token_id": token_id},
                )
            )

        if not targets:
            return BatchResult(
                outputs={
                    SENSES: {
                        "keys": pd.DataFrame(columns=token_keys),
                        "data": pd.DataFrame(columns=list(SENSE_DATA_COLUMNS)),
                    },
                    CANDIDATES: {
                        "keys": pd.DataFrame(columns=[*token_keys, "candidate_id"]),
                        "data": pd.DataFrame(columns=list(CANDIDATE_DATA_COLUMNS)),
                    },
                    UNRESOLVED: {
                        "keys": pd.DataFrame(columns=token_keys),
                        "data": pd.DataFrame(columns=list(UNRESOLVED_DATA_COLUMNS)),
                    },
                }
            )

        cache = user_cache_paths()
        ontology = _make_ontology(
            lexicon=self.lexicon,
            include_multiword_candidates=self.include_multiword_candidates,
        )
        backend = _make_backend(
            model_name=self.model_name,
            model_revision=self.model_revision,
            device=self.device,
            precision=self.precision,
            local_files_only=self.local_files_only,
        )
        from text_analysis_lab._linguistics.wsd.experiment import score_wsd_targets

        rows, unresolved, _cache_stats = score_wsd_targets(
            targets,
            ontology=ontology,
            backend=backend,
            cache_dir=cache.wsd_gloss_text,
            target_span_policy=cast(Any, self.target_span_policy),
        )

        sense_keys: list[dict[str, Any]] = []
        sense_data: list[dict[str, Any]] = []
        candidate_keys: list[dict[str, Any]] = []
        candidate_data: list[dict[str, Any]] = []
        grouped_rows: dict[str, list[Any]] = {}
        for row in rows:
            grouped_rows.setdefault(row.target_id, []).append(row)

        for target_id, target_rows in grouped_rows.items():
            key_record = target_keys[target_id]
            ordered = sorted(
                target_rows, key=lambda item: (int(item.rank), item.sense_id)
            )
            for candidate_id, row in enumerate(ordered):
                candidate_keys.append({**key_record, "candidate_id": int(candidate_id)})
                candidate_data.append(_candidate_record(self, row))
            selected = next((row for row in ordered if bool(row.selected)), None)
            if selected is not None:
                surface, parser_lemma, source_pos = target_context[target_id]
                resolved_lemma = str(selected.lemma)
                sense_keys.append(dict(key_record))
                sense_data.append(
                    {
                        "surface_form": surface,
                        "parser_lemma": parser_lemma,
                        "resolved_lemma": resolved_lemma,
                        "lemma_overrides_parser": _normalized(parser_lemma)
                        != _normalized(resolved_lemma),
                        "pos": source_pos,
                        "sense_id": str(selected.sense_id),
                        "synset_id": str(selected.synset_id),
                        "sense_label": selected.sense_label,
                        "aliases": json.dumps(
                            list(selected.aliases), ensure_ascii=False
                        ),
                        "ili": selected.ili,
                        "ontology_id": str(selected.ontology_id),
                        "ontology_version": str(selected.ontology_version),
                        "gloss_text": str(selected.gloss_text),
                        "normalized_score": float(selected.normalized_score),
                        "top1_margin": None
                        if selected.top1_margin is None
                        else float(selected.top1_margin),
                        "candidate_kind": str(selected.candidate_kind),
                        "model_name": self.model_name,
                        "model_revision": self.model_revision,
                    }
                )

        unresolved_keys: list[dict[str, Any]] = []
        unresolved_data: list[dict[str, Any]] = []
        for row in unresolved:
            key_record = target_keys[row.target_id]
            surface, parser_lemma, source_pos = target_context[row.target_id]
            unresolved_keys.append(dict(key_record))
            unresolved_data.append(
                {
                    "surface_form": surface,
                    "parser_lemma": parser_lemma,
                    "pos": source_pos,
                    "reason": str(row.reason),
                }
            )

        outputs: dict[str, Mapping[str, Any]] = {
            SENSES: {
                "keys": pd.DataFrame.from_records(sense_keys, columns=token_keys),
                "data": pd.DataFrame.from_records(
                    sense_data, columns=list(SENSE_DATA_COLUMNS)
                ),
            },
            CANDIDATES: {
                "keys": pd.DataFrame.from_records(
                    candidate_keys, columns=[*token_keys, "candidate_id"]
                ),
                "data": pd.DataFrame.from_records(
                    candidate_data, columns=list(CANDIDATE_DATA_COLUMNS)
                ),
            },
            UNRESOLVED: {
                "keys": pd.DataFrame.from_records(unresolved_keys, columns=token_keys),
                "data": pd.DataFrame.from_records(
                    unresolved_data, columns=list(UNRESOLVED_DATA_COLUMNS)
                ),
            },
        }
        return BatchResult(outputs=outputs)

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs or None

    def finalize_translation(self, *, mode, request):
        _ = mode, request

    def to_json_state(self) -> dict[str, Any]:
        return {
            "lexicon": self.lexicon,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "sentence_key": self.sentence_key,
            "token_key": self.token_key,
            "device": self.device,
            "precision": self.precision,
            "include_multiword_candidates": self.include_multiword_candidates,
            "target_span_policy": self.target_span_policy,
            "local_files_only": self.local_files_only,
            "acknowledge_noncommercial_license": True,
        }

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> WordSenseDisambiguator:
        return cls(**cast(dict[str, Any], dict(state)))


def _make_ontology(*, lexicon: str, include_multiword_candidates: bool):
    from text_analysis_lab._linguistics.wsd.ontology import WnOntologyProvider

    return WnOntologyProvider(
        lexicon,
        include_multiword_candidates=include_multiword_candidates,
        mwe_cache_dir=user_cache_paths().wordnet_mwe_indices,
    )


def _make_backend(*, model_name, model_revision, device, precision, local_files_only):
    from text_analysis_lab._linguistics.wsd.wsl_backend import BabelscapeWSLBackend

    return BabelscapeWSLBackend(
        model_name,
        revision=model_revision,
        device=device,
        precision=precision,
        cache_dir=user_cache_paths().huggingface_hub,
        local_files_only=local_files_only,
        acknowledge_noncommercial_license=True,
    )


def _candidate_record(translator: WordSenseDisambiguator, row: Any) -> dict[str, Any]:
    return {
        "sense_id": str(row.sense_id),
        "synset_id": str(row.synset_id),
        "sense_label": row.sense_label,
        "candidate_lemma": str(row.lemma),
        "aliases": json.dumps(list(row.aliases), ensure_ascii=False),
        "ili": row.ili,
        "ontology_id": str(row.ontology_id),
        "ontology_version": str(row.ontology_version),
        "gloss_text": str(row.gloss_text),
        "gloss_hash": str(row.gloss_hash),
        "model_input_text": str(row.model_input_text),
        "model_input_hash": str(row.model_input_hash),
        "raw_score": float(row.raw_score),
        "score_type": str(row.score_type),
        "normalized_score": float(row.normalized_score),
        "rank": int(row.rank),
        "selected": bool(row.selected),
        "top1_margin": None if row.top1_margin is None else float(row.top1_margin),
        "candidate_source": str(row.source),
        "candidate_kind": str(row.candidate_kind),
        "candidate_components": json.dumps(
            list(row.candidate_components), ensure_ascii=False
        ),
        "candidate_trigger_lemmas": json.dumps(
            list(row.candidate_trigger_lemmas), ensure_ascii=False
        ),
        "candidate_component_token_indices": json.dumps(
            list(row.candidate_component_token_indices), ensure_ascii=False
        ),
        "reader_target_start": int(row.reader_target_start),
        "reader_target_end": int(row.reader_target_end),
        "reader_target_text": str(row.reader_target_text),
        "reader_candidate_heading": str(row.reader_candidate_heading),
        "reader_target_span_policy": str(row.reader_target_span_policy),
        "model_name": translator.model_name,
        "model_revision": translator.model_revision,
    }


def _validate_sources(sources, *, sentence_key: str, token_key: str):
    if set(sources) != {TOKENS}:
        raise OperatorError(
            f"WordSenseDisambiguator expects source label {TOKENS!r}; got {sorted(sources)}."
        )
    tokens = sources[TOKENS]
    if tokens.artifact_type.value != "table":
        raise OperatorError("WordSenseDisambiguator requires a table token artifact.")
    if len(tokens.primary_key) < 2 or list(tokens.primary_key[-2:]) != [
        sentence_key,
        token_key,
    ]:
        raise OperatorError(
            "WordSenseDisambiguator expects token primary keys to end in "
            f"{sentence_key!r}, {token_key!r}."
        )
    return tokens


def _frame(value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ArtifactError(
            "WordSenseDisambiguator expected a pandas DataFrame token source."
        )
    return value


def _normalized(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split()).casefold()


def _normalize_wordnet_pos(pos: str | None) -> str | None:
    mapping = {
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
    if pos is None:
        return None
    stripped = pos.strip()
    return mapping.get(stripped) or mapping.get(stripped.upper())
