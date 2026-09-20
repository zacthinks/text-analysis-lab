"""Preparation of the published AllenNLP SRL model in a reusable cache."""

from __future__ import annotations

import json
from pathlib import Path

from text_analysis_lab.linguistics.model_resources import (
    ALLENNLP_SRL_BERT,
    ProjectModelPaths,
    ensure_model_archive,
    project_model_paths,
)
from text_analysis_lab.linguistics.srl.bundle import (
    FORMAT_VERSION,
    METADATA_NAME,
    WEIGHTS_NAME,
    convert_archive,
)


def _runtime_is_ready(paths: ProjectModelPaths) -> bool:
    """Return whether a compatible converted runtime already exists."""

    metadata_path = paths.runtime_dir / METADATA_NAME
    if not (
        metadata_path.exists()
        and (paths.runtime_dir / WEIGHTS_NAME).exists()
        and (paths.runtime_dir / "tokenizer" / "vocab.txt").exists()
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return str(metadata.get("format_version")) == FORMAT_VERSION


def prepare_project_srl_runtime(
    project_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    force_download: bool = False,
    reconvert: bool = False,
    show_progress: bool = False,
) -> ProjectModelPaths:
    """Ensure the published SRL runtime exists in ``cache_dir``.

    ``project_dir`` is retained for backward compatibility. When ``cache_dir`` is omitted,
    the historical project-local layout is used. Production orchestration passes the shared
    user cache explicitly so all projects reuse one converted runtime.
    """

    storage_root = Path(project_dir) if cache_dir is None else Path(cache_dir)
    paths = project_model_paths(storage_root, ALLENNLP_SRL_BERT)
    runtime_ready = _runtime_is_ready(paths)
    if runtime_ready and not reconvert and not force_download:
        if show_progress:
            print(
                f"Using existing converted SRL runtime: {paths.runtime_dir}", flush=True
            )
        return paths

    if download:
        paths = ensure_model_archive(
            storage_root,
            ALLENNLP_SRL_BERT,
            force_download=force_download,
            show_progress=show_progress,
        )
    elif not paths.archive_path.exists():
        raise FileNotFoundError(
            f"Expected the AllenNLP archive at {paths.archive_path}. "
            "Allow downloading or place the archive there."
        )

    if reconvert or force_download or not runtime_ready:
        convert_archive(
            paths.archive_path,
            paths.runtime_dir,
            tokenizer_name=ALLENNLP_SRL_BERT.base_model,
            include_tokenizer=True,
            overwrite=True,
            show_progress=show_progress,
        )
    return paths
