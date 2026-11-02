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
    assert active_hf_cache_path("biu-nlp/f-coref", teal_root=teal) == tmp_path / "new"
    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert active_hf_cache_path("biu-nlp/f-coref", teal_root=teal) == legacy
    assert active_hf_cache_path("Babelscape/wsl-reader-deberta-v3-base", teal_root=teal) == tmp_path / "new"
    assert active_hf_cache_path("biu-nlp/f-coref", revision="other", teal_root=teal) == tmp_path / "new"


def test_explicit_hf_cache_override_used_by_manager_install(tmp_path, monkeypatch):
    import sys
    import types
    from text_analysis_lab.models import ModelManager
    root = tmp_path / "teal"
    explicit = tmp_path / "selected-hub"
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(explicit / "models--biu-nlp--f-coref" / "snapshots" / "abc")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=fake_snapshot_download
    ))
    manager = ModelManager(cache_root=root, hf_cache_dir=explicit)
    manager.install("huggingface:biu-nlp/f-coref", allow_download=False)
    assert captured["cache_dir"] == str(explicit)
    assert captured["local_files_only"] is True
    assert not root.exists()


def test_prefetched_coref_snapshot_is_used_without_second_materialized_copy(tmp_path, monkeypatch):
    from text_analysis_lab.linguistics.coreference import runtime
    import sys
    import types

    cache = tmp_path / "hub"
    snapshot = cache / "models--biu-nlp--f-coref" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"test")
    seen = []

    def snapshot_download(**kwargs):
        seen.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    model = runtime.prepare_fastcoref_model(
        tmp_path / "workspace", model="fcoref", cache_dir=cache, show_progress=False,
    )
    assert model == snapshot
    assert len(seen) == 1 and seen[0]["cache_dir"] == str(cache)
    assert "local_dir" not in seen[0]
    assert not (tmp_path / "materialized").exists()
