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


def test_native_hf_revision_removal_uses_provider_plan_and_preserves_other_revisions(
    tmp_path, monkeypatch,
):
    import sys
    import types
    from text_analysis_lab.models import ModelManager

    cache = tmp_path / "hub"
    cache.mkdir()
    selected_sha = "a" * 40
    other_sha = "b" * 40
    selected = types.SimpleNamespace(
        commit_hash=selected_sha, snapshot_path=cache / "selected"
    )
    other = types.SimpleNamespace(commit_hash=other_sha, snapshot_path=cache / "other")
    repository = types.SimpleNamespace(
        repo_id="biu-nlp/f-coref", repo_type="model",
        revisions=(selected, other),
    )
    executed = []

    class Strategy:
        expected_freed_size_str = "12.0 MB"
        def execute(self):
            executed.append(True)

    class Info:
        repos = (repository,)
        def delete_revisions(self, revision):
            assert revision == selected_sha
            return Strategy()

    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(scan_cache_dir=lambda cache_dir: Info()),
    )
    manager = ModelManager(cache_root=tmp_path / "teal", hf_cache_dir=cache)
    key = "huggingface:biu-nlp/f-coref"
    preview = manager.remove(key, revision=selected_sha)
    assert preview["removed"] is False
    assert preview["expected_freed_size"] == "12.0 MB"
    assert not executed
    result = manager.remove(key, revision=selected_sha, confirm=True)
    assert result["removed"] is True
    assert executed == [True]
    assert other.commit_hash == other_sha
    assert not (tmp_path / "teal").exists()


def test_teal_owned_cache_removal_does_not_touch_other_directories(tmp_path):
    from text_analysis_lab.models import ModelManager

    root = tmp_path / "teal"
    mwe = root / "wordnet" / "mwe_indices"
    mwe.mkdir(parents=True)
    (mwe / "index.json").write_text("{}", encoding="utf-8")
    wsd = root / "wsd" / "gloss_text"
    wsd.mkdir(parents=True)
    (wsd / "keep.txt").write_text("saved", encoding="utf-8")
    manager = ModelManager(cache_root=root, hf_cache_dir=tmp_path / "hub")
    assert manager.remove("teal-cache:wordnet-mwe-indices")["removed"] is False
    assert mwe.exists()
    assert manager.remove("teal-cache:wordnet-mwe-indices", confirm=True)["removed"] is True
    assert not mwe.exists()
    assert (wsd / "keep.txt").read_text(encoding="utf-8") == "saved"



def test_native_hub_cache_offline_prefetch_reuse_and_revision_eviction(tmp_path, monkeypatch):
    """Exercise the real Hub cache scanner and download resolver, not mocks."""
    import os
    import pytest
    from text_analysis_lab.models import ModelManager
    from text_analysis_lab.linguistics.coreference.runtime import prepare_fastcoref_model

    pytest.importorskip("huggingface_hub")
    if os.name == "nt":
        pytest.skip("Native Hub symlink fixtures require platform-specific permissions")

    cache = tmp_path / "hub"
    repository = cache / "models--biu-nlp--f-coref"
    blobs = repository / "blobs"
    blobs.mkdir(parents=True)
    revisions = ("a" * 40, "b" * 40)
    for revision in revisions:
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for name, content in (
            ("config.json", b"{}"),
            ("model.safetensors", b"test-weights"),
        ):
            blob_name = revision + "-" + name.replace(".", "-")
            (blobs / blob_name).write_bytes(content)
            (snapshot / name).symlink_to(
                ("../../blobs/" + blob_name)
            )
    refs = repository / "refs"
    refs.mkdir()
    (refs / "main").write_text(revisions[1], encoding="utf-8")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    manager = ModelManager(cache_root=tmp_path / "teal", hf_cache_dir=cache)
    key = "huggingface:biu-nlp/f-coref"
    selected = manager.install(key, allow_download=False)
    assert selected == repository / "snapshots" / revisions[1]
    assert manager.verify(key).ready is None  # Fake weights do not prove loadability.
    prepared = prepare_fastcoref_model(
        tmp_path / "unrelated-project", cache_dir=cache, show_progress=False
    )
    assert prepared == selected
    assert not (tmp_path / "materialized").exists()
    assert not (tmp_path / "unrelated-project").exists()

    preview = manager.remove(key, revision=revisions[0])
    assert preview["removed"] is False
    assert (repository / "snapshots" / revisions[0]).is_dir()
    manager.remove(key, revision=revisions[0], confirm=True)
    assert not (repository / "snapshots" / revisions[0]).exists()
    assert selected.is_dir()
    assert manager.install(key, allow_download=False) == selected
    assert manager.verify(key).ready is None
