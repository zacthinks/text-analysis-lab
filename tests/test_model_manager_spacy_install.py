"""Deterministic installer acceptance tests using a locally generated wheel."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from text_analysis_lab.models import spacy_install as installer
from text_analysis_lab.models.spacy import SpacyResourceError, verify_managed_spacy


def _wheel(tmp_path: Path, *, extra=()):
    spacy = pytest.importorskip("spacy")
    source = tmp_path / "source"
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.to_disk(source)
    metadata = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    metadata.update({
        "lang": "en", "name": "core_web_sm", "version": "3.8.0",
        "spacy_version": ">=3.8,<3.9",
    })
    (source / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("en_core_web_sm/__init__.py", "raise RuntimeError('must not execute')")
        for file in sorted(source.rglob("*")):
            if file.is_file():
                archive.write(
                    file, "en_core_web_sm/en_core_web_sm-3.8.0/"
                    + file.relative_to(source).as_posix()
                )
        for name, body in extra:
            archive.writestr(name, body)
    release = installer.OfficialSpacyRelease(
        "en_core_web_sm", "3.8.0",
        hashlib.sha256(wheel.read_bytes()).hexdigest(),
        20 * 1024**2, 100 * 1024**2,
    )
    return wheel, release


def test_pinned_wheel_only_extracts_data_and_publishes_exactly_once(tmp_path):
    spacy = pytest.importorskip("spacy")
    wheel, release = _wheel(tmp_path)
    root = tmp_path / "cache"
    with installer._install_lock(root / "spacy" / "en_core_web_sm" / ".install.lock"):
        first = installer._publish_pinned_wheel(root, release, wheel)
        second = installer._publish_pinned_wheel(root, release, wheel)
    assert first == second
    assert first.source == "managed_cache"
    assert not list(Path(first.load_target).rglob("*.py"))
    assert [s.text for s in spacy.load(first.load_target)("First sentence. Second sentence.").sents] == ["First sentence.", "Second sentence."]
    assert verify_managed_spacy(root, "en_core_web_sm", "3.8.0", first.digest) == first


@pytest.mark.parametrize("bad_path", (
    "../escape",
    "/absolute.txt",
    "en_core_web_sm/en_core_web_sm-3.8.0/../../escape",
    "en_core_web_sm/en_core_web_sm-3.8.0/setup.py",
    "en_core_web_sm\\en_core_web_sm-3.8.0\\file",
))
def test_unsafe_archive_entries_rejected(tmp_path, bad_path):
    wheel, release = _wheel(tmp_path, extra=((bad_path, b"bad"),))
    with pytest.raises(SpacyResourceError):
        installer._publish_pinned_wheel(tmp_path / "cache", release, wheel)
    assert not (tmp_path / "escape").exists()


def test_modified_wheel_checksum_rejected(tmp_path):
    wheel, release = _wheel(tmp_path)
    bad = installer.OfficialSpacyRelease(
        release.name, release.version, "0" * 64,
        release.max_wheel_bytes, release.max_unpacked_bytes,
    )
    with pytest.raises(SpacyResourceError, match="checksum"):
        installer._publish_pinned_wheel(tmp_path / "cache", bad, wheel)


def test_offline_requires_no_model_directory_creation(tmp_path):
    root = tmp_path / "absent"
    with pytest.raises(SpacyResourceError):
        installer.install_official_spacy(root, "en_core_web_sm", allow_download=False)
    assert not root.exists()


def test_explicit_download_is_idempotent(tmp_path, monkeypatch):
    wheel, release = _wheel(tmp_path)
    monkeypatch.setitem(installer.OFFICIAL_SPACY_RELEASES, "en_core_web_sm", release)
    calls = []

    def fake_download(model, target, **kwargs):
        calls.append(model.name)
        shutil.copyfile(wheel, target)

    monkeypatch.setattr(installer, "_download_wheel", fake_download)
    root = tmp_path / "cache"
    a = installer.install_official_spacy(root, "en_core_web_sm", allow_download=True)
    b = installer.install_official_spacy(root, "en_core_web_sm", allow_download=True)
    assert a == b
    assert calls == ["en_core_web_sm"]

def test_standard_zip_directory_records_are_accepted(tmp_path):
    wheel, _ = _wheel(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("en_core_web_sm/", "")
        archive.writestr("en_core_web_sm/en_core_web_sm-3.8.0/", "")
    release = installer.OfficialSpacyRelease(
        "en_core_web_sm", "3.8.0",
        hashlib.sha256(wheel.read_bytes()).hexdigest(), 20 * 1024**2, 100 * 1024**2,
    )
    resolved = installer._publish_pinned_wheel(tmp_path / "cache", release, wheel)
    assert resolved.source == "managed_cache"

def test_official_download_transport_failure_has_actionable_message(tmp_path, monkeypatch):
    from urllib.error import URLError

    class BrokenOpener:
        def open(self, *args, **kwargs):
            raise URLError("simulated connection failure")

    monkeypatch.setattr(installer, "build_opener", lambda *args: BrokenOpener())
    release = installer.OFFICIAL_SPACY_RELEASES["en_core_web_sm"]
    with pytest.raises(SpacyResourceError, match="Check your network connection") as caught:
        installer._download_wheel(release, tmp_path / "failed.whl")
    assert isinstance(caught.value.__cause__, URLError)
    assert not (tmp_path / "failed.whl").exists()


def test_official_download_stream_failure_has_actionable_message(tmp_path, monkeypatch):
    from urllib.error import URLError

    class BrokenResponse:
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, n):
            raise URLError("simulated interrupted transfer")

    class BrokenOpener:
        def open(self, *args, **kwargs):
            return BrokenResponse()

    monkeypatch.setattr(installer, "build_opener", lambda *args: BrokenOpener())
    release = installer.OFFICIAL_SPACY_RELEASES["en_core_web_sm"]
    with pytest.raises(SpacyResourceError, match="Check your network connection") as caught:
        installer._download_wheel(release, tmp_path / "partial.whl")
    assert isinstance(caught.value.__cause__, URLError)
