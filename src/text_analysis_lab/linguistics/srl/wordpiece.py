"""Compatibility helpers for AllenNLP's legacy BERT SRL tokenization.

AllenNLP's SRL reader did not call the full BERT tokenizer on the sentence. It lowercased
one already-tokenized word at a time and called the old ``WordpieceTokenizer`` directly.
Modern Transformers no longer exposes that private helper, so we preserve the small,
deterministic greedy WordPiece algorithm and the original vocabulary indexing ourselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LegacyBertVocabulary:
    """Exact token-to-index mapping from a BERT ``vocab.txt`` file.

    Keeping this tiny resource independent of ``transformers.BertTokenizer`` avoids API
    drift across Transformers releases. The SRL model only needs the original token IDs
    and special-token IDs; it does not need a modern tokenizer object.
    """

    token_to_id: Mapping[str, int]
    unk_token: str = "[UNK]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    cls_token: str = "[CLS]"

    @classmethod
    def from_file(cls, path: str | Path) -> LegacyBertVocabulary:
        path = Path(path)
        tokens = path.read_text(encoding="utf-8").splitlines()
        if not tokens:
            raise RuntimeError(f"BERT vocabulary is empty: {path}")
        token_to_id = {token: index for index, token in enumerate(tokens)}
        vocabulary = cls(token_to_id=token_to_id)
        missing = [
            token
            for token in (
                vocabulary.pad_token,
                vocabulary.unk_token,
                vocabulary.cls_token,
                vocabulary.sep_token,
            )
            if token not in token_to_id
        ]
        if missing:
            raise RuntimeError(
                f"BERT vocabulary at {path} is missing required tokens: {missing}"
            )
        return vocabulary

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id[self.sep_token]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id[self.cls_token]

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> tuple[int, ...]:
        unknown = self.unk_token_id
        return tuple(self.token_to_id.get(str(token), unknown) for token in tokens)


def tokenizer_vocab(tokenizer: Any) -> Mapping[str, int]:
    """Return a BERT tokenizer vocabulary across Transformers versions.

    This remains for compatibility with external experiments. Production SRL inference
    reads ``vocab.txt`` through :class:`LegacyBertVocabulary` instead.
    """

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if vocab:
            return vocab

    for attribute in ("vocab", "_vocab"):
        vocab = getattr(tokenizer, attribute, None)
        if vocab:
            return vocab

    raise RuntimeError("Could not obtain the BERT tokenizer vocabulary.")


def legacy_wordpiece_tokenize(
    text: str,
    vocab: Mapping[str, int],
    *,
    unk_token: str = "[UNK]",
    max_input_chars_per_word: int = 100,
) -> list[str]:
    """Reproduce the old Hugging Face ``WordpieceTokenizer.tokenize`` method.

    This is the exact tokenization level used by AllenNLP's BERT SRL reader after it
    optionally lowercased each externally supplied token. It uses greedy
    longest-match-first segmentation and does not run BERT's basic tokenizer.
    """

    output_tokens: list[str] = []
    stripped = text.strip()
    if not stripped:
        return output_tokens

    for token in stripped.split():
        characters = list(token)
        if len(characters) > max_input_chars_per_word:
            output_tokens.append(unk_token)
            continue

        is_bad = False
        start = 0
        sub_tokens: list[str] = []
        while start < len(characters):
            end = len(characters)
            current_substring: str | None = None
            while start < end:
                substring = "".join(characters[start:end])
                if start > 0:
                    substring = "##" + substring
                if substring in vocab:
                    current_substring = substring
                    break
                end -= 1

            if current_substring is None:
                is_bad = True
                break

            sub_tokens.append(current_substring)
            start = end

        if is_bad:
            output_tokens.append(unk_token)
        else:
            output_tokens.extend(sub_tokens)

    return output_tokens
