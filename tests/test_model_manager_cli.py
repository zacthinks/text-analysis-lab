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
    assert entry["ready"] is None  # Read-only inventory has not validated readiness.
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

    original_version = provider.importlib.metadata.version
    monkeypatch.setattr(provider.importlib.metadata, "version",
                        lambda name: (_ for _ in ()).throw(
                            provider.importlib.metadata.PackageNotFoundError(name)
                        ) if name == "en_core_web_sm" else original_version(name))

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


def test_verify_hf_reports_missing_and_teal_cache_remove_is_preview(tmp_path, capsys):
    root = tmp_path / "teal"
    assert main(["models", "--cache-root", str(root), "verify",
                 "huggingface:biu-nlp/f-coref"]) == 2
    assert "No structurally complete" in capsys.readouterr().err
    assert main(["models", "--cache-root", str(root), "remove",
                 "teal-cache:wordnet-mwe-indices", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["removed"] is False
    assert not root.exists()


def test_models_list_json_keys_can_roundtrip_into_info(tmp_path, capsys):
    root = tmp_path / "cache"
    assert main(["models", "--cache-root", str(root), "list", "--json"]) == 0
    models = json.loads(capsys.readouterr().out)
    chosen = next(x for x in models if x["descriptor"]["provider"] == "huggingface")
    key = chosen["key"]
    assert key == chosen["descriptor"]["key"]
    assert key == "huggingface:" + chosen["descriptor"]["model_id"]
    assert main(["models", "--cache-root", str(root), "info", key, "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["key"] == key
    assert not root.exists()


def test_cli_install_provider_success_envelopes(tmp_path, capsys, monkeypatch):
    from dataclasses import dataclass
    from pathlib import Path
    from text_analysis_lab.models import ModelManager
    from text_analysis_lab.linguistics.model_resources import (
        ALLENNLP_SRL_BERT, project_model_paths,
    )

    @dataclass
    class SpaCyFixture:
        load_target: str
        source: str

    def fake_install(self, model_key, *, allow_download=False, **kwargs):
        if model_key.startswith("spacy:"):
            return SpaCyFixture("en_core_web_sm", "installed_package")
        if model_key.startswith("huggingface:"):
            return tmp_path / "hub" / "snapshots" / "abcdef"
        if model_key.startswith("allennlp:"):
            return project_model_paths(tmp_path / "teal", ALLENNLP_SRL_BERT)
        raise AssertionError(model_key)

    monkeypatch.setattr(ModelManager, "install", fake_install)
    keys = [
        "spacy:en_core_web_sm",
        "huggingface:biu-nlp/f-coref",
        f"allennlp:{ALLENNLP_SRL_BERT.model_id}",
    ]
    for key in keys:
        assert main(["models", "--cache-root", str(tmp_path / "teal"),
                     "install", key, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["key"] == key
        assert payload["provider"] == key.split(":", 1)[0]
        assert isinstance(payload["location"], str) and payload["location"]
        assert payload["location_type"] in {"directory", "import_name"}
        assert payload["source"]
        if key.startswith("spacy:"):
            assert payload["location_type"] == "import_name"
            assert payload["source"] == "installed_package"
        else:
            assert payload["location_type"] == "directory"
        assert payload["details"]
    assert not (tmp_path / "teal").exists()


def test_verify_unknown_hf_is_not_claimed_ready_by_exit_zero(tmp_path, capsys):
    root = tmp_path / "teal"
    hub = tmp_path / "hub"
    snap = hub / "models--biu-nlp--f-coref" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"invalid-weights")
    code = main(["models", "--cache-root", str(root),
                 "--hf-cache-dir", str(hub),
                 "verify", "huggingface:biu-nlp/f-coref", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0  # Command completed; runtime readiness remains unknown.
    assert data["present"] is True
    assert data["ready"] is None and data["verified"] is None
    assert data["key"] == "huggingface:biu-nlp/f-coref"



def test_models_json_failures_are_structured_for_desktop(tmp_path, capsys, monkeypatch):
    from text_analysis_lab.models import ModelManager
    root = tmp_path / "teal"
    hub = tmp_path / "hub"

    cases = (
        (["info", "spacy:unknown", "--json"], "unknown_resource"),
        (["remove", "huggingface:biu-nlp/f-coref", "--confirm", "--json"],
         "invalid_request"),
        (["remove", "teal-cache:wordnet", "--json"], "unsupported_operation"),
        (["verify", "huggingface:biu-nlp/f-coref", "--json"], "not_ready"),
    )
    for command, category in cases:
        result = main([
            "models", "--cache-root", str(root), "--hf-cache-dir", str(hub),
            *command,
        ])
        out = capsys.readouterr()
        assert result == 2
        assert out.err == ""
        payload = json.loads(out.out)
        assert payload["ok"] is False
        assert payload["error"]["category"] == category
        assert isinstance(payload["error"]["message"], str)
        assert payload["error"]["message"]
        assert "Traceback" not in out.out
    assert not root.exists()
    assert not hub.exists()


def test_models_json_offline_provider_error_is_machine_readable(tmp_path, capsys, monkeypatch):
    from text_analysis_lab.models import ModelManager

    def fail_install(self, model_key, *, allow_download=False, **kwargs):
        raise FileNotFoundError("Prefetch is unavailable in offline mode")

    monkeypatch.setattr(ModelManager, "install", fail_install)
    result = main([
        "models", "--cache-root", str(tmp_path / "teal"),
        "install", "huggingface:biu-nlp/f-coref", "--json",
    ])
    out = capsys.readouterr()
    assert result == 2 and out.err == ""
    value = json.loads(out.out)
    assert value["key"] == "huggingface:biu-nlp/f-coref"
    assert value["error"]["category"] == "missing_resource"
    assert "offline" in value["error"]["message"]
