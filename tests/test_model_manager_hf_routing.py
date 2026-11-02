"""Ensure native HF inference cache and historic TeAL assets agree."""
from __future__ import annotations

from text_analysis_lab.linguistics.cache import active_hf_cache_path, standard_hf_cache_path


def test_default_native_cache_and_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert standard_hf_cache_path() == tmp_path / "home" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert standard_hf_cache_path() == tmp_path / "explicit"
    assert active_hf_cache_path("biu-nlp/f-coref", teal_root=tmp_path / "teal") == tmp_path / "explicit"
    assert not (tmp_path / "teal").exists()


def test_existing_legacy_hub_snapshot_is_reused_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "new"))
    teal = tmp_path / "teal"
    legacy = teal / "huggingface" / "hub"
    snapshot = legacy / "models--biu-nlp--f-coref" / "snapshots" / "sha-old"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert active_hf_cache_path("biu-nlp/f-coref", teal_root=teal) == legacy
    assert active_hf_cache_path("Babelscape/wsl-reader-deberta-v3-base", teal_root=teal) == tmp_path / "new"
    assert active_hf_cache_path("biu-nlp/f-coref", revision="other", teal_root=teal) == tmp_path / "new"
