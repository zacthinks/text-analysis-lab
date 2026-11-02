"""Smoke tests for the project-independent TeAL model commands."""
from __future__ import annotations

import json

from text_analysis_lab.cli import main


def test_models_cache_paths_do_not_create_directories(tmp_path, capsys):
    root = tmp_path / "cache"
    assert main(["models", "--cache-root", str(root), "cache-path", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["teal"] == str(root)
    assert not root.exists()


def test_models_list_and_info_are_read_only(tmp_path, capsys):
    root = tmp_path / "cache"
    assert main(["models", "--cache-root", str(root), "list", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert any(x["descriptor"]["model_id"] == "en_core_web_sm" for x in entries)
    assert main([
        "models", "--cache-root", str(root), "info",
        "spacy:en_core_web_sm", "--json",
    ]) == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["descriptor"]["provider"] == "spacy"
    assert entry["ready"] is False
    assert not root.exists()


def test_models_verify_offline_fails_without_mutation(tmp_path, capsys, monkeypatch):
    root = tmp_path / "cache"
    import text_analysis_lab.models.spacy as provider
    monkeypatch.setattr(
        provider.importlib.metadata, "version",
        lambda name: (_ for _ in ()).throw(
            provider.importlib.metadata.PackageNotFoundError(name)
        ),
    )
    assert main([
        "models", "--cache-root", str(root), "verify",
        "spacy:en_core_web_sm",
    ]) == 2
    assert "unavailable offline" in capsys.readouterr().err
    assert not root.exists()


def test_models_reject_unknown_without_download(tmp_path, capsys):
    root = tmp_path / "cache"
    assert main([
        "models", "--cache-root", str(root), "install",
        "spacy:unapproved", "--allow-download",
    ]) == 2
    assert "No approved spaCy release" in capsys.readouterr().err
    assert not root.exists()

def test_models_install_http_failure_is_reported_without_traceback(tmp_path, capsys, monkeypatch):
    from urllib.error import URLError
    import text_analysis_lab.models.spacy_install as installer
    import text_analysis_lab.models.spacy as provider

    monkeypatch.setattr(provider.importlib.metadata, "version",
                        lambda name: (_ for _ in ()).throw(
                            provider.importlib.metadata.PackageNotFoundError(name)
                        ) if name == "en_core_web_sm" else __import__("importlib").metadata.version(name))

    class FailedOpener:
        def open(self, *args, **kwargs):
            raise URLError("simulated HTTP failure")
    monkeypatch.setattr(installer, "build_opener", lambda *args: FailedOpener())
    result = main([
        "models", "--cache-root", str(tmp_path / "cache"), "install",
        "spacy:en_core_web_sm", "--allow-download",
    ])
    assert result == 2
    error = capsys.readouterr().err
    assert "Check your network connection" in error
    assert "Traceback" not in error
