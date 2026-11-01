"""Read-only model-manager contract tests."""
from __future__ import annotations

from pathlib import Path

from text_analysis_lab.models import ModelManager, standard_hf_cache_path


def test_hf_cache_precedence(tmp_path: Path) -> None:
    env = {"XDG_CACHE_HOME": str(tmp_path / "xdg")}
    assert standard_hf_cache_path(environ=env) == tmp_path / "xdg" / "huggingface" / "hub"
    env["HF_HOME"] = str(tmp_path / "hf-home")
    assert standard_hf_cache_path(environ=env) == tmp_path / "hf-home" / "hub"
    env["HUGGINGFACE_HUB_CACHE"] = str(tmp_path / "old-hub")
    assert standard_hf_cache_path(environ=env) == tmp_path / "old-hub"
    env["HF_HUB_CACHE"] = str(tmp_path / "new-hub")
    assert standard_hf_cache_path(environ=env) == tmp_path / "new-hub"


def test_listing_never_creates_cache_or_assumes_verified(tmp_path: Path) -> None:
    teal = tmp_path / "teal"
    hub = tmp_path / "hub"
    manager = ModelManager(cache_root=teal, hf_cache_dir=hub)
    paths = manager.cache_paths()
    assert paths["teal"] == teal
    assert paths["huggingface_hub"] == hub
    items = manager.list_models()
    assert len(items) >= 4
    assert len({item.descriptor.key for item in items}) == len(items)
    assert all(not item.verified and not item.compatible and not item.ready for item in items)
    assert not teal.exists()
    assert not hub.exists()


def test_provisional_srl_presence_is_not_readiness(tmp_path: Path) -> None:
    manager = ModelManager(cache_root=tmp_path, hf_cache_dir=tmp_path / "hf")
    (tmp_path / "models" / "allennlp-srl-bert-2020-12-15").mkdir(parents=True)
    srl = next(item for item in manager.list_models() if item.descriptor.provider == "allennlp")
    assert srl.present
    assert not srl.ready
    assert not srl.verified
