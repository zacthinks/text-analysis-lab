"""Shared runtime for the published AllenNLP BERT semantic-role labeler.

This is the single implementation used by both the standalone diagnostic command and the
persistent Parquet SRL stage. Keeping model loading, legacy WordPiece tokenization, predicate
conditioning, and BIO decoding here prevents the validated path from diverging from the
production pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from text_analysis_lab._linguistics.device import resolve_devices
from text_analysis_lab._linguistics.srl.decoding import viterbi_decode_bio
from text_analysis_lab._linguistics.srl.structures import (
    BioRepair,
    BioSpan,
    bio_spans,
    repair_projected_bio_tags,
)
from text_analysis_lab._linguistics.srl.wordpiece import (
    LegacyBertVocabulary,
    legacy_wordpiece_tokenize,
)
from text_analysis_lab._linguistics.srl_bundle import (
    METADATA_NAME,
    WEIGHTS_NAME,
    discard_legacy_bert_buffers,
)


@dataclass(frozen=True, slots=True)
class EncodedSrlSentence:
    """One pre-tokenized sentence encoded exactly as AllenNLP's SRL reader encoded it."""

    tokens: tuple[str, ...]
    wordpieces: tuple[str, ...]
    input_ids: tuple[int, ...]
    wordpiece_starts: tuple[int, ...]
    wordpiece_ends: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SrlTokenPrediction:
    """Inspectable SRL output for one sentence and one explicitly designated predicate."""

    tokens: tuple[str, ...]
    predicate_index: int
    predicate: str
    wordpieces: tuple[str, ...]
    input_ids: tuple[int, ...]
    predicate_indicator: tuple[int, ...]
    wordpiece_offsets: tuple[int, ...]
    wordpiece_tags: tuple[str, ...]
    raw_tags: tuple[str, ...]
    tags: tuple[str, ...]
    bio_repairs: tuple[BioRepair, ...]
    word_scores: tuple[float, ...]
    spans: tuple[BioSpan, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["spans"] = [asdict(span) for span in self.spans]
        return value


def format_srl_description(tokens: Sequence[str], spans: Sequence[BioSpan]) -> str:
    """Format BIO spans using AllenNLP's familiar bracketed frame display."""

    by_start = {span.start: span for span in spans}
    pieces: list[str] = []
    index = 0
    while index < len(tokens):
        span = by_start.get(index)
        if span is None:
            pieces.append(tokens[index])
            index += 1
            continue
        text = " ".join(tokens[span.start : span.end])
        pieces.append(f"[{span.label}: {text}]")
        index = span.end
    description = " ".join(pieces)
    for mark in (" .", " ,", " ;", " :", " !", " ?"):
        description = description.replace(mark, mark.strip())
    return description


class AllenNlpSrlRuntime:
    """Load and run the converted AllenNLP BERT SRL checkpoint.

    The runtime reproduces the released model rather than merely approximating its interface:
    each predicate indicator is passed into BERT as ``token_type_ids``; decoding occurs over
    WordPieces; and the first WordPiece prediction is selected for each original token.
    """

    def __init__(
        self,
        runtime_dir: str | Path,
        *,
        device: str | tuple[str, ...] = "auto",
        strict_device: bool = False,
        mixed_precision: bool = True,
        max_length: int | None = None,
        show_progress: bool = False,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.requested_device = device
        self.strict_device = strict_device
        self.mixed_precision = mixed_precision
        self.requested_max_length = max_length
        self.show_progress = show_progress
        self._loaded: (
            tuple[Any, Any, Any, tuple[str, ...], dict[str, Any], str, int] | None
        ) = None

    @staticmethod
    def _require_runtime() -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "AllenNLP SRL requires PyTorch plus the TeAL `nlp` dependencies "
                "(Transformers and safetensors)."
            ) from exc
        return torch, load_file, BertConfig, BertModel

    def _load(self) -> tuple[Any, Any, Any, tuple[str, ...], dict[str, Any], str, int]:
        if self._loaded is not None:
            return self._loaded

        torch, load_file, BertConfig, BertModel = self._require_runtime()
        metadata_path = self.runtime_dir / METADATA_NAME
        weights_path = self.runtime_dir / WEIGHTS_NAME
        vocab_path = self.runtime_dir / "tokenizer" / "vocab.txt"
        if (
            not metadata_path.exists()
            or not weights_path.exists()
            or not vocab_path.exists()
        ):
            raise FileNotFoundError(
                f"Incomplete SRL runtime at {self.runtime_dir}. The runtime must contain "
                f"{METADATA_NAME}, {WEIGHTS_NAME}, and tokenizer/vocab.txt."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = tuple(str(label) for label in metadata["labels"])
        if not labels:
            raise RuntimeError("The converted SRL runtime contains no labels.")
        bert_config = BertConfig.from_dict(metadata["bert_config"])
        vocabulary = LegacyBertVocabulary.from_file(vocab_path)

        class SrlBert(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bert_model = BertModel(bert_config)
                self.tag_projection_layer = torch.nn.Linear(
                    bert_config.hidden_size, len(labels)
                )
                self.embedding_dropout = torch.nn.Dropout(
                    float(metadata.get("embedding_dropout", 0.0))
                )

            def forward(self, input_ids, token_type_ids, attention_mask):
                bert_embeddings = self.bert_model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )[0]
                return self.tag_projection_layer(
                    self.embedding_dropout(bert_embeddings)
                )

        resolved = resolve_devices(self.requested_device, strict=self.strict_device)
        device = resolved.primary
        if self.show_progress:
            print(f"Loading converted SRL weights on {device}...", flush=True)
        model = SrlBert()
        state = discard_legacy_bert_buffers(load_file(str(weights_path), device="cpu"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        material_missing = [
            key
            for key in missing
            if not key.endswith(("position_ids", "token_type_ids"))
        ]
        material_unexpected = [
            key
            for key in unexpected
            if not key.endswith(("position_ids", "token_type_ids"))
        ]
        if material_missing or material_unexpected:
            raise RuntimeError(
                "Converted AllenNLP weights do not match the reconstructed model. "
                f"Missing: {material_missing}; unexpected: {material_unexpected}"
            )
        model.to(device)
        model.eval()
        maximum = int(self.requested_max_length or bert_config.max_position_embeddings)
        if self.show_progress:
            print("SRL model loaded.", flush=True)
        self._loaded = (torch, vocabulary, model, labels, metadata, device, maximum)
        return self._loaded

    @property
    def max_length(self) -> int:
        return self._load()[-1]

    @property
    def device(self) -> str:
        return self._load()[-2]

    def metadata(self) -> dict[str, Any]:
        return dict(self._load()[4])

    def encode_tokens(self, tokens: Sequence[str]) -> EncodedSrlSentence:
        """Apply AllenNLP's per-token lowercasing and legacy greedy WordPiece algorithm."""

        _torch, vocabulary, _model, _labels, metadata, _device, _maximum = self._load()
        normalized_tokens = tuple(str(token) for token in tokens)
        if not normalized_tokens:
            raise ValueError("SRL requires at least one token.")
        lowercase = bool(
            metadata.get("lowercase_input", "uncased" in metadata["tokenizer_name"])
        )
        vocab = vocabulary.token_to_id
        wordpieces: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for token in normalized_tokens:
            value = token.lower() if lowercase else token
            pieces = legacy_wordpiece_tokenize(
                value,
                vocab,
                unk_token=vocabulary.unk_token,
            )
            if not pieces:
                raise ValueError("SRL input tokens must contain non-whitespace text.")
            starts.append(len(wordpieces) + 1)
            wordpieces.extend(pieces)
            ends.append(len(wordpieces) + 1)

        pieces_with_specials = (
            vocabulary.cls_token,
            *wordpieces,
            vocabulary.sep_token,
        )
        input_ids = tuple(
            int(value)
            for value in vocabulary.convert_tokens_to_ids(pieces_with_specials)
        )
        encoded = EncodedSrlSentence(
            tokens=normalized_tokens,
            wordpieces=tuple(pieces_with_specials),
            input_ids=input_ids,
            wordpiece_starts=tuple(starts),
            wordpiece_ends=tuple(ends),
        )
        if len(encoded.input_ids) > self.max_length:
            raise ValueError(
                f"Sentence has {len(encoded.input_ids)} WordPieces, exceeding the model "
                f"maximum of {self.max_length}."
            )
        return encoded

    def predict_encoded_batch(
        self,
        instances: Sequence[tuple[EncodedSrlSentence, int]],
    ) -> tuple[SrlTokenPrediction, ...]:
        """Run a batch of predicate-conditioned instances through the validated model."""

        if not instances:
            return ()
        torch, vocabulary, model, labels, _metadata, device, maximum = self._load()
        for encoded, predicate_index in instances:
            if not 0 <= predicate_index < len(encoded.tokens):
                raise IndexError(
                    f"predicate_index {predicate_index} is outside a sentence with "
                    f"{len(encoded.tokens)} tokens"
                )
            if len(encoded.input_ids) > maximum:
                raise ValueError(
                    f"Sentence has {len(encoded.input_ids)} WordPieces, exceeding the model "
                    f"maximum of {maximum}."
                )

        lengths = [len(encoded.input_ids) for encoded, _ in instances]
        padded_length = max(lengths)
        input_ids = torch.full(
            (len(instances), padded_length),
            int(vocabulary.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(instances), padded_length), dtype=torch.long, device=device
        )
        predicate_indicators = torch.zeros(
            (len(instances), padded_length), dtype=torch.long, device=device
        )

        for row_index, (encoded, predicate_index) in enumerate(instances):
            length = len(encoded.input_ids)
            input_ids[row_index, :length] = torch.tensor(
                encoded.input_ids, dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1
            predicate_start = encoded.wordpiece_starts[predicate_index]
            predicate_end = encoded.wordpiece_ends[predicate_index]
            predicate_indicators[row_index, predicate_start:predicate_end] = 1

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.mixed_precision and str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            logits = model(input_ids, predicate_indicators, attention_mask)
            probabilities = torch.softmax(logits.float(), dim=-1)

        predictions: list[SrlTokenPrediction] = []
        for row_index, (encoded, predicate_index) in enumerate(instances):
            length = lengths[row_index]
            emissions = probabilities[row_index, :length]
            path = viterbi_decode_bio(emissions, labels)
            word_label_ids = tuple(path[offset] for offset in encoded.wordpiece_starts)
            raw_tags = tuple(labels[label_id] for label_id in word_label_ids)
            tags, bio_repairs = repair_projected_bio_tags(raw_tags)
            word_scores = tuple(
                float(emissions[offset, label_id].item())
                for offset, label_id in zip(
                    encoded.wordpiece_starts, word_label_ids, strict=True
                )
            )
            wordpiece_tags = tuple(labels[label_id] for label_id in path)
            indicator = [0] * length
            predicate_start = encoded.wordpiece_starts[predicate_index]
            predicate_end = encoded.wordpiece_ends[predicate_index]
            indicator[predicate_start:predicate_end] = [1] * (
                predicate_end - predicate_start
            )
            spans = tuple(bio_spans(tags))
            predictions.append(
                SrlTokenPrediction(
                    tokens=encoded.tokens,
                    predicate_index=predicate_index,
                    predicate=encoded.tokens[predicate_index],
                    wordpieces=encoded.wordpieces,
                    input_ids=encoded.input_ids,
                    predicate_indicator=tuple(indicator),
                    wordpiece_offsets=encoded.wordpiece_starts,
                    wordpiece_tags=wordpiece_tags,
                    raw_tags=raw_tags,
                    tags=tags,
                    bio_repairs=bio_repairs,
                    word_scores=word_scores,
                    spans=spans,
                    description=format_srl_description(encoded.tokens, spans),
                )
            )
        return tuple(predictions)

    def predict_tokens(
        self,
        tokens: Sequence[str],
        *,
        predicate_index: int,
    ) -> SrlTokenPrediction:
        encoded = self.encode_tokens(tokens)
        return self.predict_encoded_batch(((encoded, predicate_index),))[0]
