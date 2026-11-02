"""Network opt-in acceptance for an actual official spaCy model release.

Run with TEAL_RUN_REAL_SPACY_DOWNLOAD=1. This test intentionally exercises the
ordinary translator path, not a separate prefetch command.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TEAL_RUN_REAL_SPACY_DOWNLOAD") != "1",
    reason="Real official spaCy download is opt-in.",
)


def _documents(project):
    import pandas as pd

    from text_analysis_lab.core.writer import create_artifact_writer

    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_documents"),
        artifact_id="art_documents",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({
        "keys": pd.DataFrame({"document_id": [1, 2]}),
        "data": pd.DataFrame({"text": [
            "Alice sees Bob. They wave.",
            "Two sentences here. Another here.",
        ]}),
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_documents",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_documents")


def test_real_official_first_use_shared_cache_and_frozen_project(tmp_path, monkeypatch):
    import text_analysis_lab as teal
    import text_analysis_lab.models.spacy_install as installer

    from text_analysis_lab.core.operator import BaseOperator
    from text_analysis_lab.linguistics.cache import user_cache_paths
    from text_analysis_lab.models.spacy import verify_managed_spacy
    from text_analysis_lab.translators import SpacyTranslator

    pytest.importorskip("spacy")
    try:
        installed_version = importlib.metadata.version("en_core_web_sm")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    assert installed_version is None, (
        "Real first-use smoke requires no preinstalled en_core_web_sm package."
    )

    monkeypatch.setenv("TEAL_CACHE_DIR", str(tmp_path / "shared-cache"))
    monkeypatch.delenv("TEAL_OFFLINE", raising=False)
    calls = []
    original = installer._download_wheel

    def record_download(*args, **kwargs):
        calls.append(args[0].name)
        return original(*args, **kwargs)

    monkeypatch.setattr(installer, "_download_wheel", record_download)
    project_path = tmp_path / "cloud-synced-project"

    # Ordinary translator use must perform the first download without prefetch.
    translator = SpacyTranslator(model="en_core_web_sm")
    parsed = translator.translate("Alice sees Bob. They wave.")
    assert len(parsed["sentences"]) == 2
    assert calls == ["en_core_web_sm"]
    assert not project_path.exists()

    snapshot = tmp_path / "frozen-operator"
    translator.save_to_dir(snapshot, operator_id="op_spacy_real")
    identity = translator.resource_identity
    assert identity is not None and identity["kind"] == "managed_cache"
    assert not (snapshot / "assets" / "spacy_model").exists()

    cached = verify_managed_spacy(
        user_cache_paths(create=False).models,
        identity["name"], identity["model_version"], identity["content_sha256"],
    )
    assert Path(cached.load_target).is_dir()
    assert str(cached.load_target).startswith(str(tmp_path / "shared-cache"))

    # A separate translator, with network disabled, must reuse precisely this data.
    monkeypatch.setenv("TEAL_OFFLINE", "1")
    assert len(SpacyTranslator(model="en_core_web_sm").translate("Another sentence.")["sentences"]) == 1
    assert len(BaseOperator.load_from_dir(snapshot).translate("Frozen sentence.")["sentences"]) == 1
    assert calls == ["en_core_web_sm"]

    project = teal.Project.create(project_path, name="real_spacy")
    try:
        source = _documents(project)
        outputs = project.translate(
            SpacyTranslator(model="en_core_web_sm", download_model=False),
            source,
            workers=1, batch_size=1,
        )
        assert set(outputs) == {"sentences", "tokens"}
        assert all(x.status == "complete" for x in outputs.values())
        ids = {name: x.artifact_id for name, x in outputs.items()}
    finally:
        project.close()

    project = teal.Project.open(project_path)
    try:
        assert all(project.get_artifact(k).status == "complete" for k in ids.values())
    finally:
        project.close()

    # Shared user cache is separate from the research project.
    assert not (project_path / ".venv").exists()
    assert not (project_path / "models").exists()
    assert calls == ["en_core_web_sm"]
