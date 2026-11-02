"""Unit tests for non-mutating resource inventory."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_listing_does_not_create_directories_or_claim_readiness(tmp_path: Path) -> None:
    teal, hub = tmp_path / "teal", tmp_path / "hub"
    manager = ModelManager(cache_root=teal, hf_cache_dir=hub)
    assert manager.cache_paths()["teal"] == teal
    assert manager.cache_paths()["huggingface_hub"] == hub
    models = manager.list_models()
    assert len(models) >= 7
    assert len({item.descriptor.key for item in models}) == len(models)
    assert all(item.verified is None and item.compatible is None and item.ready is None for item in models)
    assert not teal.exists()
    assert not hub.exists()


def test_exact_revision_and_legacy_hub_inventory(tmp_path: Path) -> None:
    teal, hub = tmp_path / "teal", tmp_path / "hub"
    manager = ModelManager(cache_root=teal, hf_cache_dir=hub)
    wsl_key = "huggingface:Babelscape/wsl-reader-deberta-v3-base"
    descriptor = manager.inspect(wsl_key).descriptor
    repo_dir = hub / "models--Babelscape--wsl-reader-deberta-v3-base" / "snapshots"
    wrong = repo_dir / "0123456789012345678901234567890123456789"
    wrong.mkdir(parents=True)
    (wrong / "model.safetensors").write_bytes(b"fake")
    assert not manager.inspect(wsl_key).present  # Must not confuse revisions.
    pinned = repo_dir / descriptor.revision
    pinned.mkdir()
    (pinned / "model.safetensors").write_bytes(b"fake")
    status = manager.inspect(wsl_key)
    assert status.present
    assert status.candidate_path == pinned
    assert [x.revision for x in status.locations] == [descriptor.revision]
    assert status.verified is None and status.ready is None

    legacy = teal / "huggingface" / "hub" / "models--biu-nlp--f-coref" / "snapshots" / "commitA"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}", encoding="utf-8")
    coref = manager.inspect("huggingface:biu-nlp/f-coref")
    assert coref.present
    assert coref.locations[0].source == "legacy_teal_hf"


def test_srl_archive_and_converted_runtime_are_distinct(tmp_path: Path) -> None:
    from text_analysis_lab.linguistics.model_resources import ALLENNLP_SRL_BERT, project_model_paths
    manager = ModelManager(cache_root=tmp_path, hf_cache_dir=tmp_path / "hub")
    paths = project_model_paths(tmp_path, ALLENNLP_SRL_BERT)
    assert paths.root == tmp_path / "models" / ALLENNLP_SRL_BERT.model_id
    paths.source_dir.mkdir(parents=True)
    paths.archive_path.write_bytes(b"fake archive")
    srl = manager.inspect(f"allennlp:{ALLENNLP_SRL_BERT.model_id}")
    assert srl.present
    assert not srl.prepared
    assert [loc.source for loc in srl.locations] == ["srl_archive"]

    paths.runtime_dir.mkdir()
    (paths.runtime_dir / "teal_srl.json").write_text('{"format_version": "3"}')
    (paths.runtime_dir / "model.safetensors").write_bytes(b"fake")
    (paths.runtime_dir / "tokenizer").mkdir()
    (paths.runtime_dir / "tokenizer" / "vocab.txt").write_text("hi")
    srl = manager.inspect(f"allennlp:{ALLENNLP_SRL_BERT.model_id}")
    assert srl.prepared
    assert [loc.source for loc in srl.locations] == ["srl_archive", "srl_runtime"]
    assert srl.verified is None and srl.ready is None


def test_spacy_package_presence_does_not_imply_ready(tmp_path: Path, monkeypatch) -> None:
    import text_analysis_lab.models as module
    def fake_version(name: str) -> str:
        if name == "en_core_web_sm":
            return "3.8.0"
        raise module.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(module.importlib.metadata, "version", fake_version)
    manager = ModelManager(cache_root=tmp_path, hf_cache_dir=tmp_path / "hub")
    model = manager.inspect("spacy:en_core_web_sm")
    assert model.present
    assert model.locations[0].revision == "3.8.0"
    assert model.candidate_path is None  # Distribution presence is not a directory path.
    assert model.compatible is None and model.verified is None and model.ready is None


def test_unknown_model_is_explicit(tmp_path: Path) -> None:
    manager = ModelManager(cache_root=tmp_path, hf_cache_dir=tmp_path / "hub")
    with pytest.raises(KeyError, match="Unknown TeAL model"):
        manager.inspect("spacy:unapproved_model")


def test_lexical_auxiliary_caches_are_discovered_without_creating_them(tmp_path: Path) -> None:
    manager = ModelManager(cache_root=tmp_path / "teal", hf_cache_dir=tmp_path / "hub")
    assert not manager.inspect("teal-cache:wordnet").present
    assert not manager.inspect("teal-cache:wordnet-mwe-indices").present
    assert not manager.inspect("teal-cache:wsd-gloss-text").present
    assert not (tmp_path / "teal").exists()
    auxiliary = tmp_path / "teal" / "wsd" / "gloss_text"
    auxiliary.mkdir(parents=True)
    (auxiliary / "cache.json").write_text("{}", encoding="utf-8")
    status = manager.inspect("teal-cache:wsd-gloss-text")
    assert status.present
    assert status.candidate_path == auxiliary
    assert status.ready is None and status.verified is None



def test_wordnet_aggregate_is_not_independently_removable(tmp_path: Path) -> None:
    from text_analysis_lab.models import ModelManager

    root = tmp_path / "teal"
    wordnet = root / "wordnet"
    mwe = wordnet / "mwe_indices"
    mwe.mkdir(parents=True)
    (wordnet / "other-owned-data.txt").write_text("keep", encoding="utf-8")
    (mwe / "index.json").write_text("{}", encoding="utf-8")
    manager = ModelManager(cache_root=root, hf_cache_dir=tmp_path / "hub")
    assert manager.inspect("teal-cache:wordnet").present
    assert manager.inspect("teal-cache:wordnet-mwe-indices").present
    with pytest.raises(NotImplementedError, match="separately managed"):
        manager.remove("teal-cache:wordnet", confirm=True)
    assert (wordnet / "other-owned-data.txt").is_file()
    assert (mwe / "index.json").is_file()
    preview = manager.remove("teal-cache:wordnet-mwe-indices")
    assert preview["removed"] is False
    assert "mwe_indices" in preview["path"]
    assert (mwe / "index.json").is_file()
    manager.remove("teal-cache:wordnet-mwe-indices", confirm=True)
    assert not mwe.exists()
    assert (wordnet / "other-owned-data.txt").read_text(encoding="utf-8") == "keep"


def test_huggingface_structural_presence_does_not_claim_verified_or_ready(tmp_path: Path) -> None:
    from text_analysis_lab.models import ModelManager

    hub = tmp_path / "hub"
    fake = hub / "models--biu-nlp--f-coref" / "snapshots" / ("c" * 40)
    fake.mkdir(parents=True)
    (fake / "config.json").write_text("{}", encoding="utf-8")
    (fake / "model.safetensors").write_bytes(b"not-a-real-model")
    manager = ModelManager(cache_root=tmp_path / "teal", hf_cache_dir=hub)
    status = manager.verify("huggingface:biu-nlp/f-coref")
    assert status.present
    assert status.candidate_path == fake
    assert status.verified is None
    assert status.compatible is None
    assert status.ready is None
    assert "not been validated" in status.detail
