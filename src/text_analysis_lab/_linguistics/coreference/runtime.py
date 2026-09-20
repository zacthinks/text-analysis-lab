"""Shared FastCoref/LingMess model preparation and inference runtime.

The runtime resolves Hugging Face snapshots from a reusable user-level cache, runs bounded
batched inference, normalizes exact character-offset clusters, and is shared by both the
production Parquet backend and the lower-level validation notebook.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CorefModelName = Literal["fcoref", "lingmess"]


@dataclass(frozen=True)
class FastCorefModelSpec:
    name: CorefModelName
    repository: str
    directory_name: str
    class_name: str
    description: str


_MODEL_SPECS: dict[str, FastCorefModelSpec] = {
    "fcoref": FastCorefModelSpec(
        name="fcoref",
        repository="biu-nlp/f-coref",
        directory_name="fcoref",
        class_name="FCoref",
        description="Fast distilled coreference model (high-throughput option).",
    ),
    "fastcoref": FastCorefModelSpec(
        name="fcoref",
        repository="biu-nlp/f-coref",
        directory_name="fcoref",
        class_name="FCoref",
        description="Fast distilled coreference model (high-throughput option).",
    ),
    "lingmess": FastCorefModelSpec(
        name="lingmess",
        repository="biu-nlp/lingmess-coref",
        directory_name="lingmess",
        class_name="LingMessCoref",
        description="Larger, more accurate LingMess model (production default).",
    ),
    "lingmesscoref": FastCorefModelSpec(
        name="lingmess",
        repository="biu-nlp/lingmess-coref",
        directory_name="lingmess",
        class_name="LingMessCoref",
        description="Larger, more accurate LingMess model (production default).",
    ),
}


@dataclass(frozen=True)
class CorefMention:
    cluster_id: int
    mention_id: int
    start_char: int
    end_char: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "mention_id": self.mention_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
        }


@dataclass(frozen=True)
class CorefNormalizationIssue:
    reason: str
    detail: str


@dataclass(frozen=True)
class CorefPrediction:
    document_index: int
    text: str
    mentions: tuple[CorefMention, ...]
    normalization_issue: CorefNormalizationIssue | None = None

    @property
    def clusters(self) -> tuple[tuple[CorefMention, ...], ...]:
        grouped: dict[int, list[CorefMention]] = {}
        for mention in self.mentions:
            grouped.setdefault(mention.cluster_id, []).append(mention)
        return tuple(tuple(grouped[index]) for index in sorted(grouped))

    def cluster_strings(self) -> list[list[str]]:
        return [[mention.text for mention in cluster] for cluster in self.clusters]

    def rows(self) -> list[dict[str, object]]:
        return [
            {"document_index": self.document_index, **mention.to_dict()}
            for mention in self.mentions
        ]


@dataclass(frozen=True)
class CorefBatchPrediction:
    model: CorefModelName
    model_repository: str
    device: str
    elapsed_seconds: float
    documents: tuple[CorefPrediction, ...]

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for prediction in self.documents:
            rows.extend(prediction.rows())
        return rows


def get_fastcoref_model_spec(model: str) -> FastCorefModelSpec:
    key = model.strip().lower().replace("-", "").replace("_", "")
    if key not in _MODEL_SPECS:
        choices = "fcoref or lingmess"
        raise ValueError(f"unknown coreference model {model!r}; choose {choices}")
    return _MODEL_SPECS[key]


def _model_is_present(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((model_dir / name).is_file() for name in candidates)


def _snapshot_download(
    *,
    repository: str,
    cache_dir: Path,
    local_dir: Path,
    force_download: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency is supplied by transformers
        raise RuntimeError(
            "Hugging Face model download support is unavailable. Run `uv sync --all-extras`."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repository,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            force_download=force_download,
        )
    )


def _set_eager_attention(config: Any) -> Any:
    """Force eager attention on a Transformers config.

    LingMess uses Longformer. Recent Transformers releases may select SDPA by default,
    but Longformer does not implement that backend. The published FastCoref release
    constructs the config internally, so the experimental wrapper patches that single
    construction step rather than modifying the downloaded model snapshot.
    """

    config._attn_implementation = "eager"
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = "eager"
    return config


@contextmanager
def _fastcoref_eager_attention_patch(enabled: bool) -> Iterator[None]:
    """Temporarily make FastCoref's internal AutoConfig request eager attention."""

    if not enabled:
        yield
        return

    modeling = importlib.import_module("fastcoref.modeling")
    original = modeling.AutoConfig.from_pretrained

    def from_pretrained_eager(*args: Any, **kwargs: Any) -> Any:
        config = original(*args, **kwargs)
        return _set_eager_attention(config)

    modeling.AutoConfig.from_pretrained = from_pretrained_eager
    try:
        yield
    finally:
        modeling.AutoConfig.from_pretrained = original


def prepare_fastcoref_model(
    workspace: str | Path,
    *,
    model: str = "fcoref",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    show_progress: bool = True,
) -> Path:
    """Resolve a FastCoref snapshot from a reusable Hugging Face cache.

    ``workspace`` is retained for backward compatibility. If ``cache_dir`` is omitted, the
    historical workspace-local cache root is used. Production orchestration supplies the
    shared TeAL Hugging Face cache.
    """

    spec = get_fastcoref_model_spec(model)
    effective_cache = (
        Path(workspace) / "huggingface" / "hub"
        if cache_dir is None
        else Path(cache_dir)
    )
    effective_cache.mkdir(parents=True, exist_ok=True)
    materialized_dir = effective_cache.parent / "materialized" / spec.directory_name

    if not force_download and _model_is_present(materialized_dir):
        if show_progress:
            print(
                f"Coreference model ready: {materialized_dir}",
                flush=True,
            )
        return materialized_dir

    if show_progress:
        action = "Refreshing" if force_download else "Resolving"
        print(
            f"{action} {spec.repository} from shared cache {effective_cache}...",
            flush=True,
        )
    model_dir = _snapshot_download(
        repository=spec.repository,
        cache_dir=effective_cache,
        local_dir=materialized_dir,
        force_download=force_download,
    )
    if not _model_is_present(model_dir):
        raise RuntimeError(
            f"The resolved snapshot at {model_dir} does not contain "
            "a recognized model weight file."
        )
    if show_progress:
        print(f"Coreference model ready: {model_dir}", flush=True)
    return model_dir


def _malformed_coref_prediction(
    *,
    document_index: int,
    original_text: str,
    detail: str,
) -> CorefPrediction:
    """Return an atomic document-level failure rather than partial coreference output."""

    return CorefPrediction(
        document_index=document_index,
        text=original_text,
        mentions=(),
        normalization_issue=CorefNormalizationIssue(
            reason="malformed_model_output",
            detail=detail,
        ),
    )


def normalize_fastcoref_result(
    result: Any,
    *,
    document_index: int,
    original_text: str,
) -> CorefPrediction:
    """Convert a FastCoref result to stable character-offset records.

    FastCoref can occasionally return a malformed offset such as ``None`` inside an
    otherwise valid result. Coreference clusters are document-level objects, so retaining
    only the other mentions would create a silently partial annotation. The normalization
    policy is therefore atomic per document: any malformed cluster/span discards every
    mention for that document and returns one auditable issue. Other documents in the same
    model batch remain usable.
    """

    try:
        offset_clusters = result.get_clusters(as_strings=False)
    except Exception as exc:
        return _malformed_coref_prediction(
            document_index=document_index,
            original_text=original_text,
            detail=f"get_clusters(as_strings=False) failed: {type(exc).__name__}: {exc}",
        )

    if offset_clusters is None or isinstance(offset_clusters, (str, bytes)):
        return _malformed_coref_prediction(
            document_index=document_index,
            original_text=original_text,
            detail=f"unexpected clusters container: {offset_clusters!r}",
        )

    mentions: list[CorefMention] = []
    try:
        clusters = tuple(offset_clusters)
    except TypeError:
        return _malformed_coref_prediction(
            document_index=document_index,
            original_text=original_text,
            detail=f"clusters are not iterable: {offset_clusters!r}",
        )

    for cluster_id, cluster in enumerate(clusters):
        if cluster is None or isinstance(cluster, (str, bytes)):
            return _malformed_coref_prediction(
                document_index=document_index,
                original_text=original_text,
                detail=f"cluster_id={cluster_id}: unexpected cluster {cluster!r}",
            )
        try:
            spans = tuple(cluster)
        except TypeError:
            return _malformed_coref_prediction(
                document_index=document_index,
                original_text=original_text,
                detail=f"cluster_id={cluster_id}: cluster is not iterable: {cluster!r}",
            )

        for mention_id, span in enumerate(spans):
            if span is None or isinstance(span, (str, bytes)):
                return _malformed_coref_prediction(
                    document_index=document_index,
                    original_text=original_text,
                    detail=(
                        f"cluster_id={cluster_id}, mention_id={mention_id}: "
                        f"unexpected span {span!r}"
                    ),
                )
            try:
                coordinates = tuple(span)
            except TypeError:
                return _malformed_coref_prediction(
                    document_index=document_index,
                    original_text=original_text,
                    detail=(
                        f"cluster_id={cluster_id}, mention_id={mention_id}: "
                        f"span is not iterable: {span!r}"
                    ),
                )
            if len(coordinates) != 2:
                return _malformed_coref_prediction(
                    document_index=document_index,
                    original_text=original_text,
                    detail=(
                        f"cluster_id={cluster_id}, mention_id={mention_id}: "
                        f"expected two offsets, found {coordinates!r}"
                    ),
                )
            try:
                start_char, end_char = (int(coordinates[0]), int(coordinates[1]))
            except (TypeError, ValueError, OverflowError) as exc:
                return _malformed_coref_prediction(
                    document_index=document_index,
                    original_text=original_text,
                    detail=(
                        f"cluster_id={cluster_id}, mention_id={mention_id}: "
                        f"non-integer offsets {coordinates!r}: {type(exc).__name__}: {exc}"
                    ),
                )
            if not (0 <= start_char <= end_char <= len(original_text)):
                return _malformed_coref_prediction(
                    document_index=document_index,
                    original_text=original_text,
                    detail=(
                        f"cluster_id={cluster_id}, mention_id={mention_id}: "
                        f"invalid character span {(start_char, end_char)} for document "
                        f"length {len(original_text)}"
                    ),
                )
            mentions.append(
                CorefMention(
                    cluster_id=cluster_id,
                    mention_id=mention_id,
                    start_char=start_char,
                    end_char=end_char,
                    text=original_text[start_char:end_char],
                )
            )
    return CorefPrediction(
        document_index=document_index,
        text=original_text,
        mentions=tuple(mentions),
    )


class FastCorefRuntime:
    """Small experimental wrapper around FCoref and LingMessCoref."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        model: str = "fcoref",
        device: str = "cpu",
        compile_model: bool = False,
        show_progress: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.spec = get_fastcoref_model_spec(model)
        self.device = device
        self.compile_model = bool(compile_model)
        self.show_progress = bool(show_progress)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._model: Any | None = None
        self._model_dir: Path | None = None

    def prepare_model(self, *, force_download: bool = False) -> Path:
        self._model_dir = prepare_fastcoref_model(
            self.workspace,
            model=self.spec.name,
            cache_dir=self.cache_dir,
            force_download=force_download,
            show_progress=self.show_progress,
        )
        return self._model_dir

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_dir = self.prepare_model()
        try:
            import fastcoref
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "The experimental coreference dependency is unavailable. "
                "Run `uv sync --all-extras` from the project root."
            ) from exc
        except AttributeError as exc:
            if "PyExtensionType" in str(exc):
                raise RuntimeError(
                    "FastCoref 2.1.6 / datasets 2.14.4 is incompatible with "
                    "PyArrow 21 or newer. Run `uv sync --all-extras` with the "
                    "TeAL dependency constraint `pyarrow<21`."
                ) from exc
            raise

        model_class = getattr(fastcoref, self.spec.class_name)
        parameters: dict[str, object] = {
            "model_name_or_path": str(model_dir),
            "device": self.device,
            # FastCoref only needs tokenization for raw text. A blank English tokenizer avoids
            # downloading another spaCy pipeline and keeps this experiment independent.
            "nlp": spacy.blank("en"),
            # FastCoref's nested tqdm bars do not update reliably for long documents and
            # can make completed GPU work look stalled. TeAL reports the outer
            # scope batches and authoritative wall-clock timings instead.
            "enable_progress_bar": False,
        }
        if "compile_model" in inspect.signature(model_class).parameters:
            parameters["compile_model"] = self.compile_model
        elif self.compile_model:
            raise RuntimeError(
                "This installed FastCoref release does not support compile_model=True."
            )

        if self.show_progress:
            print(
                f"Loading {self.spec.name} on {self.device} from {model_dir}...",
                flush=True,
            )
            if self.spec.name == "lingmess":
                print(
                    "Using eager attention for LingMess/Longformer compatibility.",
                    flush=True,
                )
        with _fastcoref_eager_attention_patch(self.spec.name == "lingmess"):
            self._model = model_class(**parameters)
        if self.show_progress:
            print("Coreference model loaded.", flush=True)
        return self._model

    def configured_max_document_tokens(self) -> int | None:
        """Return FastCoref's explicit document limit, if the model defines one.

        FastCoref 2.1.6 sets ``FCoref.max_doc_len`` to ``None``. That is not an
        invalid value: F-Coref segments complete documents internally and therefore
        exposes no wrapper-level cutoff. LingMess normally supplies an integer limit.
        """

        model = self._load_model()
        raw_limit = getattr(model, "max_doc_len", None)
        if raw_limit is None:
            return None
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Unexpected {self.spec.name} max_doc_len value: {raw_limit!r}"
            ) from exc
        if limit < 1:
            raise RuntimeError(
                f"Unexpected {self.spec.name} max_doc_len value: {raw_limit!r}"
            )
        return limit

    def document_token_counts(
        self, texts: Iterable[str]
    ) -> tuple[list[int], int | None]:
        """Return model-token counts and any explicit per-document limit."""

        documents = list(texts)
        model = self._load_model()
        tokenizer = model.tokenizer
        try:
            encoded = tokenizer(
                documents,
                add_special_tokens=False,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            if documents and input_ids and isinstance(input_ids[0], int):
                input_ids = [input_ids]
            counts = [len(values) for values in input_ids]
        except Exception:
            counts = [
                len(
                    tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                )
                for text in documents
            ]
        return counts, self.configured_max_document_tokens()

    def predict_texts(
        self,
        texts: Iterable[str],
        *,
        max_tokens_in_batch: int = 10_000,
        release_logits: bool = True,
    ) -> CorefBatchPrediction:
        documents = list(texts)
        if not documents:
            raise ValueError("at least one document is required")
        if any(not isinstance(text, str) for text in documents):
            raise TypeError("all coreference inputs must be strings")
        if max_tokens_in_batch < 1:
            raise ValueError("max_tokens_in_batch must be at least 1")

        model = self._load_model()
        if self.show_progress:
            print(
                f"Running {self.spec.name} on {len(documents)} document(s) "
                f"with max_tokens_in_batch={max_tokens_in_batch}...",
                flush=True,
            )
        started = time.perf_counter()
        raw_results = model.predict(
            texts=documents,
            is_split_into_words=False,
            max_tokens_in_batch=max_tokens_in_batch,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
        if len(raw_results) != len(documents):
            raise RuntimeError(
                f"FastCoref returned {len(raw_results)} results for {len(documents)} documents"
            )

        predictions: list[CorefPrediction] = []
        for document_index, (text, result) in enumerate(
            zip(documents, raw_results, strict=True)
        ):
            try:
                predictions.append(
                    normalize_fastcoref_result(
                        result,
                        document_index=document_index,
                        original_text=text,
                    )
                )
            finally:
                if release_logits and hasattr(result, "release_logits"):
                    result.release_logits()

        if self.show_progress:
            print(f"Coreference inference finished in {elapsed:.2f}s.", flush=True)
        return CorefBatchPrediction(
            model=self.spec.name,
            model_repository=self.spec.repository,
            device=self.device,
            elapsed_seconds=elapsed,
            documents=tuple(predictions),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--model", choices=("fcoref", "lingmess"), default="fcoref")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens-in-batch", type=int, default=2_000)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Document to analyze; may be supplied more than once",
    )
    return parser.parse_args()


def main() -> None:
    from text_analysis_lab._linguistics.cache import user_cache_paths

    args = parse_args()
    runtime = FastCorefRuntime(
        args.workspace,
        model=args.model,
        device=args.device,
        compile_model=args.compile_model,
        show_progress=True,
        cache_dir=user_cache_paths().huggingface_hub,
    )
    runtime.prepare_model(force_download=args.force_download)
    if args.download_only:
        return
    texts = args.texts or [
        "Alice called Mary after she arrived at the office.",
        "The agency published its report. It later withdrew the document.",
    ]
    prediction = runtime.predict_texts(
        texts,
        max_tokens_in_batch=args.max_tokens_in_batch,
    )
    print(f"Model: {prediction.model} ({prediction.model_repository})")
    print(f"Device: {prediction.device}")
    print(f"Elapsed: {prediction.elapsed_seconds:.2f}s")
    for document in prediction.documents:
        print(f"\nDocument {document.document_index}: {document.text}")
        print(f"Clusters: {document.cluster_strings()}")
        for row in document.rows():
            print(row)


if __name__ == "__main__":
    main()
