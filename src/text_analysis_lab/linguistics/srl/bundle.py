#!/usr/bin/env python3
"""Convert the public AllenNLP BERT SRL archive into a TeAL-compatible model bundle.

The converter imports PyTorch, Transformers, and safetensors, but never imports AllenNLP.
It deserializes the archive's ``weights.th`` with ``weights_only=True`` and writes a
portable ``model.safetensors`` plus explicit JSON metadata and a local BERT tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from text_analysis_lab.linguistics.model_resources import download_file_atomic

METADATA_NAME = "teal_srl.json"
WEIGHTS_NAME = "model.safetensors"
FORMAT_VERSION = "3"

# Older Transformers checkpoints serialized this deterministic buffer. Current
# Transformers registers it with persistent=False, so it is correctly absent from the
# reconstructed model's state dict.
LEGACY_NONPERSISTENT_BERT_BUFFERS = {
    "bert_model.embeddings.position_ids",
    "bert_model.embeddings.token_type_ids",
}

BERT_BASE_UNCASED_VOCAB_URL = (
    "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt"
)


def _status(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def _progress_bar(*, total: int | None, description: str, enabled: bool):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return None
    return tqdm(
        total=total,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not enabled,
    )


def find_member(archive: tarfile.TarFile, names: set[str]) -> tarfile.TarInfo:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile() and PurePosixPath(member.name).name in names
    ]
    if not matches:
        raise RuntimeError(f"Archive does not contain any of: {sorted(names)}")
    matches.sort(
        key=lambda member: (len(PurePosixPath(member.name).parts), member.name)
    )
    return matches[0]


def read_json_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> dict[str, Any]:
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Could not read {member.name}")
    return json.load(handle)


def read_labels(archive: tarfile.TarFile) -> list[str]:
    candidates = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and PurePosixPath(member.name).name == "labels.txt"
        and "vocabulary" in PurePosixPath(member.name).parts
    ]
    if not candidates:
        raise RuntimeError("Archive does not contain vocabulary/labels.txt")
    candidates.sort(
        key=lambda member: (len(PurePosixPath(member.name).parts), member.name)
    )
    handle = archive.extractfile(candidates[0])
    if handle is None:
        raise RuntimeError(f"Could not read {candidates[0].name}")
    return [line.decode("utf-8").rstrip("\n") for line in handle if line.strip()]


def nested_get(value: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current = value
        try:
            for key in path:
                current = current[key]
        except (KeyError, TypeError):
            continue
        return current
    return None


def discard_legacy_bert_buffers(state: dict[str, Any]) -> dict[str, Any]:
    """Remove deterministic buffers no longer persisted by modern Transformers."""

    return {
        key: value
        for key, value in state.items()
        if key not in LEGACY_NONPERSISTENT_BERT_BUFFERS
    }


def normalize_state_dict(state: Any) -> dict[str, Any]:
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("weights.th did not contain a state dictionary")
    normalized: dict[str, Any] = {}
    for key, tensor in state.items():
        new_key = str(key)
        new_key = new_key.removeprefix("module.")
        if new_key in LEGACY_NONPERSISTENT_BERT_BUFFERS:
            continue
        normalized[new_key] = tensor.detach().cpu().contiguous()
    required = {
        "tag_projection_layer.weight",
        "tag_projection_layer.bias",
        "bert_model.embeddings.word_embeddings.weight",
    }
    missing = sorted(required - normalized.keys())
    if missing:
        raise RuntimeError(
            f"Archive weights are missing expected SRL parameters: {missing}"
        )
    return normalized


def _bert_config(
    tokenizer_name: str, configured: Any, BertConfig: Any
) -> dict[str, Any]:
    if isinstance(configured, dict):
        return configured
    if tokenizer_name == "bert-base-uncased":
        # BertConfig defaults are the original BERT-base architecture. The complete
        # fine-tuned weights come from the AllenNLP archive; no HF model weights are used.
        return BertConfig().to_dict()
    return BertConfig.from_pretrained(tokenizer_name).to_dict()


def _save_tokenizer_vocabulary(
    tokenizer_name: str,
    output_dir: Path,
    *,
    show_progress: bool,
) -> None:
    """Preserve the exact WordPiece vocabulary needed by the SRL reader.

    AllenNLP indexed already-tokenized WordPieces directly from ``vocab.txt``. We therefore
    store that file rather than constructing a modern ``BertTokenizer`` object, whose
    constructor changed incompatibly between Transformers 4 and 5.
    """

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    vocab_output = tokenizer_dir / "vocab.txt"
    if tokenizer_name == "bert-base-uncased":
        download_file_atomic(
            BERT_BASE_UNCASED_VOCAB_URL,
            vocab_output,
            description="BERT vocabulary",
            show_progress=show_progress,
        )
        return

    _status(
        f"Downloading tokenizer vocabulary '{tokenizer_name}' through Transformers...",
        enabled=show_progress,
    )
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(tokenizer_dir)
    if not vocab_output.exists():
        raise RuntimeError(
            f"Tokenizer '{tokenizer_name}' did not produce tokenizer/vocab.txt."
        )


def convert_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    tokenizer_name: str | None = None,
    include_tokenizer: bool = False,
    overwrite: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import BertConfig
    except ImportError as exc:
        raise RuntimeError(
            "Conversion requires the development PyTorch runtime and `install TeAL with the linguistic dependencies`."
        ) from exc

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    _status(f"Reading AllenNLP archive: {archive_path}", enabled=show_progress)
    with tarfile.open(archive_path, "r:*") as archive:
        config_member = find_member(archive, {"config.json"})
        weight_member = find_member(archive, {"weights.th", "best.th"})
        config = read_json_member(archive, config_member)
        labels = read_labels(archive)
        configured_tokenizer = nested_get(
            config,
            ("dataset_reader", "bert_model_name"),
            ("validation_dataset_reader", "bert_model_name"),
            ("model", "bert_model", "_name_or_path"),
        )
        tokenizer_name = tokenizer_name or configured_tokenizer or "bert-base-uncased"
        configured_bert = nested_get(config, ("model", "bert_model"))
        bert_config = _bert_config(tokenizer_name, configured_bert, BertConfig)

        extracted = archive.extractfile(weight_member)
        if extracted is None:
            raise RuntimeError(f"Could not read {weight_member.name}")
        descriptor, temporary_name = tempfile.mkstemp(suffix=".th")
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        progress = _progress_bar(
            total=weight_member.size,
            description="Extracting checkpoint",
            enabled=show_progress,
        )
        try:
            with temporary_path.open("wb") as temporary:
                while chunk := extracted.read(1024 * 1024):
                    temporary.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
            if progress is not None:
                progress.close()
            _status(
                "Loading the legacy checkpoint into CPU memory (this can take a while)...",
                enabled=show_progress,
            )
            state = torch.load(temporary_path, map_location="cpu", weights_only=True)
        finally:
            if progress is not None:
                progress.close()
            temporary_path.unlink(missing_ok=True)

    normalized = normalize_state_dict(state)
    weights_output = output_dir / WEIGHTS_NAME
    _status("Writing converted weights as safetensors...", enabled=show_progress)
    save_file(normalized, str(weights_output))

    if include_tokenizer:
        _status(f"Preparing tokenizer: {tokenizer_name}", enabled=show_progress)
        _save_tokenizer_vocabulary(
            tokenizer_name,
            output_dir,
            show_progress=show_progress,
        )

    metadata = {
        "format_version": FORMAT_VERSION,
        "architecture": "allennlp_srl_bert",
        "source_archive": archive_path.name,
        "tokenizer_name": tokenizer_name,
        "lowercase_input": "uncased" in tokenizer_name.lower(),
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "bert_config": bert_config,
        "labels": labels,
        "embedding_dropout": float(
            nested_get(config, ("model", "embedding_dropout")) or 0.0
        ),
        "source_config": {
            "model_type": nested_get(config, ("model", "type")),
            "dataset_reader_type": nested_get(config, ("dataset_reader", "type")),
        },
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _status(
        f"SRL runtime prepared in {time.perf_counter() - started:.1f}s: {output_dir}",
        enabled=show_progress,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="Local AllenNLP SRL model.tar.gz archive"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory for the converted bundle"
    )
    parser.add_argument(
        "--tokenizer-name",
        help="Override the tokenizer recorded in the AllenNLP configuration",
    )
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Download and save the tokenizer into the bundle for offline inference",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = convert_archive(
        args.archive,
        args.output_dir,
        tokenizer_name=args.tokenizer_name,
        include_tokenizer=args.include_tokenizer,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
