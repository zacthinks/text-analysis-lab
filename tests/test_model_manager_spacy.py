"""Managed spaCy model validation without network or package installation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from text_analysis_lab.models.spacy import (
    DIGEST_ALGORITHM,
    SpacyResourceError,
    available_managed_spacy,
    directory_sha256,
    resolve_spacy,
    verify_managed_spacy,
)


def _entry(tmp_path: Path, *, name: str = "en_core_web_sm", version: str = "3.8.0"):
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    path = tmp_path / "source"
    nlp.to_disk(path)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({"lang": "en", "name": "core_web_sm", "version": version, "spacy_version": ">=3.0,<4.0"})
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    digest = directory_sha256(path)
    entry = tmp_path / "models" / "spacy" / name / version / digest
    entry.mkdir(parents=True)
    import shutil
    shutil.copytree(path, entry / "pipeline")
    (entry / "manifest.json").write_text(json.dumps({
        "provider": "spacy", "model_id": name, "model_version": version,
        "digest_algorithm": DIGEST_ALGORITHM, "content_sha256": digest,
    }), encoding="utf-8")
    return entry, digest


def test_verified_managed_pipeline_loads_without_network(tmp_path: Path) -> None:
    import spacy
    entry, digest = _entry(tmp_path)
    root = tmp_path / "models"
    resolution = resolve_spacy(root, "en_core_web_sm", spacy_version=spacy.__version__)
    assert resolution.source == "managed_cache"
    assert resolution.digest == digest
    assert resolution.frozen_identity()["content_sha256"] == digest
    assert [sent.text for sent in spacy.load(resolution.load_target)("First. Second.").sents] == [
        "First.", "Second."
    ]
    assert available_managed_spacy(root, "en_core_web_sm") == (("3.8.0", digest),)


def test_modified_bytes_rejected_for_frozen_identity(tmp_path: Path) -> None:
    entry, digest = _entry(tmp_path)
    (entry / "pipeline" / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SpacyResourceError, match="Invalid or incompatible|fingerprint"):
        verify_managed_spacy(tmp_path / "models", "en_core_web_sm", "3.8.0", digest, spacy_version="3.8.0")


def test_incompatible_spacy_runtime_rejected(tmp_path: Path) -> None:
    _, digest = _entry(tmp_path)
    with pytest.raises(SpacyResourceError, match="requires spaCy"):
        verify_managed_spacy(tmp_path / "models", "en_core_web_sm", "3.8.0", digest, spacy_version="2.3.0")


def test_no_cache_mutation_and_unknown_model_rejected(tmp_path: Path) -> None:
    root = tmp_path / "nonexistent"
    with pytest.raises(SpacyResourceError, match="unavailable offline"):
        resolve_spacy(root, "en_core_web_sm")
    with pytest.raises(SpacyResourceError, match="Unknown or unsupported"):
        resolve_spacy(root, "custom_model")
    assert not root.exists()


def test_symlink_and_manifest_identity_rejected(tmp_path: Path) -> None:
    entry, digest = _entry(tmp_path)
    (entry / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SpacyResourceError, match="manifest"):
        verify_managed_spacy(tmp_path / "models", "en_core_web_sm", "3.8.0", digest, spacy_version="3.8.0")

    src = tmp_path / "other"
    src.mkdir()
    (src / "data").write_text("bytes", encoding="utf-8")
    (src / "link").symlink_to(src / "data")
    with pytest.raises(SpacyResourceError, match="Symlink"):
        directory_sha256(src)
