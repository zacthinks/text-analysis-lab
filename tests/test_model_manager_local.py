"""User-registered local model assets are references, not executable models."""
from __future__ import annotations

import pytest

from text_analysis_lab.models import ModelManager


def test_register_local_file_and_directory_without_copying(tmp_path):
    cache = tmp_path / "cache"
    file = tmp_path / "weights.ckpt"
    file.write_bytes(b"checkpoint")
    folder = tmp_path / "spacy-or-other-resource"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")

    manager = ModelManager(cache_root=cache)
    item = manager.register_local("my checkpoint", file)
    assert item.descriptor.key == "local:my checkpoint"
    assert item.present and item.ready is None and item.verified is None

    manager.register_local("shared directory", folder)
    other = ModelManager(cache_root=cache)
    assert other.inspect("local:shared directory").present
    assert other.verify("local:my checkpoint").ready is None
    with pytest.raises(FileExistsError):
        other.register_local("my checkpoint", file)

    other.unregister_local("my checkpoint")
    assert file.read_bytes() == b"checkpoint"
    with pytest.raises(KeyError):
        other.inspect("local:my checkpoint")


def test_missing_local_resource_remains_visible_as_missing(tmp_path):
    file = tmp_path / "model.bin"
    file.write_bytes(b"data")
    manager = ModelManager(cache_root=tmp_path / "cache")
    manager.register_local("model", file)
    file.unlink()
    status = manager.inspect("local:model")
    assert status.present is False
    assert status.ready is None
    assert status.locations[0].path == file


def test_registration_rejects_invalid_paths_and_names(tmp_path):
    manager = ModelManager(cache_root=tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        manager.register_local("missing", tmp_path / "missing")
    file = tmp_path / "model.bin"
    file.write_bytes(b"data")
    for name in ("", "../invalid", "folder/name", "C:invalid"):
        with pytest.raises(ValueError):
            manager.register_local(name, file)


def test_discovers_unlisted_huggingface_snapshot_in_existing_cache(tmp_path):
    cache = tmp_path / "hub"
    snapshot = (
        cache / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots" / "revision123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    manager = ModelManager(cache_root=tmp_path / "teal", hf_cache_dir=cache)
    status = manager.inspect("huggingface:sentence-transformers/all-MiniLM-L6-v2")
    assert status.present
    assert status.ready is None
    assert status.locations[0].revision == "revision123"
    assert manager.verify(status.descriptor.key).verified is None
