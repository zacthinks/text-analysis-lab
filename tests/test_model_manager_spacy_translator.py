"""Regression tests for the spaCy manager/translator integration."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from text_analysis_lab.core.operator import BaseOperator, TranslationRequest
from text_analysis_lab.linguistics.cache import user_cache_paths
from text_analysis_lab.models.spacy import DIGEST_ALGORITHM, directory_sha256
from text_analysis_lab.translators.spacy_translator import SpacyTranslator


def _managed(tmp_path: Path, monkeypatch):
    spacy = pytest.importorskip("spacy")
    import text_analysis_lab.translators.spacy_translator as module

    monkeypatch.setenv("TEAL_CACHE_DIR", str(tmp_path / "shared"))
    source = tmp_path / "source"
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.to_disk(source)
    meta_file = source / "meta.json"
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    metadata.update({
        "lang": "en",
        "name": "core_web_sm",
        "version": "3.8.0",
        "spacy_version": ">=3.8,<3.9",
    })
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")
    digest = directory_sha256(source)
    root = user_cache_paths(create=False).models
    entry = root / "spacy" / "en_core_web_sm" / "3.8.0" / digest
    entry.mkdir(parents=True)
    shutil.copytree(source, entry / "pipeline")
    (entry / "manifest.json").write_text(json.dumps({
        "provider": "spacy",
        "model_id": "en_core_web_sm",
        "model_version": "3.8.0",
        "digest_algorithm": DIGEST_ALGORITHM,
        "content_sha256": digest,
    }), encoding="utf-8")
    module._load_spacy_pipeline.cache_clear()
    return entry, digest


def test_managed_pipeline_freezes_and_roundtrips(tmp_path: Path, monkeypatch) -> None:
    entry, digest = _managed(tmp_path, monkeypatch)
    translator = SpacyTranslator(model="en_core_web_sm")
    assert len(translator.translate("One. Two.")["sentences"]) == 2
    path = tmp_path / "frozen"
    translator.save_to_dir(path, operator_id="op_spacy_managed")
    assert translator.resource_identity is not None
    assert translator.resource_identity["kind"] == "managed_cache"
    assert translator.resource_identity["content_sha256"] == digest
    assert not (path / "assets" / "spacy_model").exists()

    restored = BaseOperator.load_from_dir(path)
    assert isinstance(restored, SpacyTranslator)
    assert len(restored.translate("Again.")["sentences"]) == 1
    worker = restored.make_translate_worker(mode="translate", request=TranslationRequest())
    assert len(worker.translate("Worker.")["sentences"]) == 1

    (entry / "pipeline" / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="Frozen managed spaCy resource failed verification"):
        BaseOperator.load_from_dir(path).translate("Tampered.")


def test_frozen_managed_resource_never_falls_back_to_package(
    tmp_path: Path, monkeypatch
) -> None:
    entry, _ = _managed(tmp_path, monkeypatch)
    translator = SpacyTranslator(model="en_core_web_sm")
    path = tmp_path / "frozen"
    translator.save_to_dir(path, operator_id="op_spacy_managed")
    shutil.rmtree(entry)
    with pytest.raises(Exception, match="Frozen managed spaCy resource failed verification"):
        BaseOperator.load_from_dir(path).translate("No fallback.")


def test_explicit_local_directory_still_loads(tmp_path: Path) -> None:
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model = tmp_path / "custom"
    nlp.to_disk(model)
    assert len(SpacyTranslator(model=str(model)).translate("First. Second.")["sentences"]) == 2

def test_ordinary_first_use_calls_shared_installer_once(tmp_path, monkeypatch):
    entry, digest = _managed(tmp_path, monkeypatch)
    import text_analysis_lab.models.spacy_install as installer

    calls = []
    original = installer.install_official_spacy

    def recording_install(root, name, *, allow_download=False, **kwargs):
        calls.append((name, allow_download))
        return original(root, name, allow_download=False)

    monkeypatch.setattr(installer, "install_official_spacy", recording_install)
    translator = SpacyTranslator(model="en_core_web_sm")
    translator.translate("First sentence.")
    translator.translate("Second sentence.")
    assert calls == [("en_core_web_sm", True)]
    assert translator.resource_identity is None


def test_explicit_offline_never_attempts_download(tmp_path, monkeypatch):
    import text_analysis_lab.models.spacy_install as installer

    monkeypatch.setenv("TEAL_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setenv("TEAL_OFFLINE", "1")
    def no_network(*args, **kwargs):
        assert kwargs["allow_download"] is False
        from text_analysis_lab.models.spacy import SpacyResourceError
        raise SpacyResourceError("unavailable offline")
    monkeypatch.setattr(installer, "install_official_spacy", no_network)
    with pytest.raises(Exception, match="unavailable offline"):
        SpacyTranslator(model="en_core_web_sm").translate("No download.")
    assert not (tmp_path / "empty-cache").exists()
