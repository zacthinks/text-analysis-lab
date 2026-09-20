"""Storage and download helpers for reusable external model artifacts.

Callers choose the storage root. Production orchestration supplies TeAL' user-level
cache, while tests and low-level callers may use an isolated temporary root. Downloads are
written atomically so interrupted requests do not leave partial archives.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ModelResource:
    """Description of one upstream model artifact."""

    model_id: str
    source_url: str
    filename: str
    architecture: str
    base_model: str | None = None


ALLENNLP_SRL_BERT = ModelResource(
    model_id="allennlp-srl-bert-2020-12-15",
    source_url=(
        "https://storage.googleapis.com/allennlp-public-models/"
        "structured-prediction-srl-bert.2020.12.15.tar.gz"
    ),
    filename="structured-prediction-srl-bert.2020.12.15.tar.gz",
    architecture="allennlp_srl_bert",
    base_model="bert-base-uncased",
)


@dataclass(frozen=True, slots=True)
class ProjectModelPaths:
    """Canonical paths for a model under a caller-supplied storage root."""

    root: Path
    source_dir: Path
    archive_path: Path
    source_manifest_path: Path
    runtime_dir: Path


def project_model_paths(
    project_dir: str | Path,
    resource: ModelResource = ALLENNLP_SRL_BERT,
) -> ProjectModelPaths:
    root = Path(project_dir) / "models" / resource.model_id
    source_dir = root / "source"
    return ProjectModelPaths(
        root=root,
        source_dir=source_dir,
        archive_path=source_dir / resource.filename,
        source_manifest_path=source_dir / "source.json",
        runtime_dir=root / "runtime",
    )


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _response_metadata(response: BinaryIO) -> dict[str, object | None]:
    headers = getattr(response, "headers", None)
    get = headers.get if headers is not None else lambda _name: None
    content_length = get("Content-Length")
    return {
        "content_length_header": int(content_length) if content_length else None,
        "etag": get("ETag"),
        "last_modified": get("Last-Modified"),
        "content_type": get("Content-Type"),
    }


def _progress_bar(*, total: int | None, description: str, enabled: bool):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is explicit in the NLP extra
        return None
    return tqdm(
        total=total,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not enabled,
    )


def download_file_atomic(
    source_url: str,
    destination: str | Path,
    *,
    description: str,
    show_progress: bool = False,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> dict[str, object | None]:
    """Download one file with visible progress and atomically publish it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    downloaded_bytes = 0
    response_details: dict[str, object | None] = {}

    try:
        if show_progress:
            print(f"Connecting: {source_url}", flush=True)
        request = Request(
            source_url,
            headers={"User-Agent": "text-analysis-lab-model-downloader/0.3"},
        )
        with (
            urlopen(request, timeout=timeout) as response,
            os.fdopen(descriptor, "wb") as output,
        ):
            response_details = _response_metadata(response)
            expected_length = response_details.get("content_length_header")
            total = expected_length if isinstance(expected_length, int) else None
            progress = _progress_bar(
                total=total, description=description, enabled=show_progress
            )
            try:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
            finally:
                if progress is not None:
                    progress.close()
            output.flush()
            os.fsync(output.fileno())

        expected_length = response_details.get("content_length_header")
        if isinstance(expected_length, int) and downloaded_bytes != expected_length:
            raise OSError(
                f"Incomplete download: expected {expected_length} bytes, "
                f"received {downloaded_bytes}."
            )

        os.replace(temporary_path, destination)
        return {**response_details, "size_bytes": downloaded_bytes}
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_model_archive(
    project_dir: str | Path,
    resource: ModelResource = ALLENNLP_SRL_BERT,
    *,
    force_download: bool = False,
    show_progress: bool = False,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> ProjectModelPaths:
    """Ensure that the upstream archive exists under the supplied storage root.

    Existing files are trusted and reused. TeAL does not repeatedly hash or police
    cache internals. An interrupted download is still safe because the final archive path
    is only published after the request completes.
    """

    paths = project_model_paths(project_dir, resource)
    paths.source_dir.mkdir(parents=True, exist_ok=True)

    if force_download:
        paths.archive_path.unlink(missing_ok=True)
        paths.source_manifest_path.unlink(missing_ok=True)

    if paths.archive_path.exists():
        if show_progress:
            print(f"Using existing AllenNLP archive: {paths.archive_path}", flush=True)
        if not paths.source_manifest_path.exists():
            _atomic_json_write(
                paths.source_manifest_path,
                {
                    "resource": asdict(resource),
                    "source_url": resource.source_url,
                    "filename": resource.filename,
                    "size_bytes": paths.archive_path.stat().st_size,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "download": None,
                },
            )
        return paths

    details = download_file_atomic(
        resource.source_url,
        paths.archive_path,
        description="AllenNLP SRL archive",
        show_progress=show_progress,
        timeout=timeout,
        chunk_size=chunk_size,
    )
    _atomic_json_write(
        paths.source_manifest_path,
        {
            "resource": asdict(resource),
            "source_url": resource.source_url,
            "filename": resource.filename,
            "size_bytes": paths.archive_path.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "download": details,
        },
    )
    return paths
