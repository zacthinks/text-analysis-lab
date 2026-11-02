"""Opt-in installation of checksum-pinned official spaCy model data.

No pip invocation, package-code execution, or downloads on import/discovery.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from text_analysis_lab.models.spacy import (
    SpacyResourceError, _metadata, directory_sha256,
    resolve_spacy, spacy_cache_root, verify_managed_spacy,
)


@dataclass(frozen=True)
class OfficialSpacyRelease:
    name: str
    version: str
    wheel_sha256: str
    max_wheel_bytes: int
    max_unpacked_bytes: int

    @property
    def url(self) -> str:
        stem = f"{self.name}-{self.version}"
        return f"https://github.com/explosion/spacy-models/releases/download/{stem}/{stem}-py3-none-any.whl"


# Exact official wheel SHA-256 values from explosion/spacy-models release metadata.
OFFICIAL_SPACY_RELEASES = {
    "en_core_web_sm": OfficialSpacyRelease("en_core_web_sm", "3.8.0", "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85", 80 * 1024**2, 350 * 1024**2),
    "en_core_web_md": OfficialSpacyRelease("en_core_web_md", "3.8.0", "5e6329fe3fecedb1d1a02c3ea2172ee0fede6cea6e4aefb6a02d832dba78a310", 250 * 1024**2, 800 * 1024**2),
    "en_core_web_lg": OfficialSpacyRelease("en_core_web_lg", "3.8.0", "293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63", 850 * 1024**2, 3 * 1024**3),
}


@contextmanager
def _install_lock(path: Path, *, timeout: float = 120.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SpacyResourceError("Refusing symlinked spaCy install lock.")
    with path.open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        start = time.monotonic()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() - start >= timeout:
                    raise SpacyResourceError("Timed out waiting for spaCy installation lock.") from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class _HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme.lower() != "https":
            raise SpacyResourceError("Refusing non-HTTPS model redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_wheel(release, destination, *, progress=None, cancelled=None, timeout=60.0):
    if urlsplit(release.url).hostname != "github.com" or urlsplit(release.url).scheme != "https":
        raise SpacyResourceError("Unapproved model URL.")
    opener = build_opener(_HTTPSOnlyRedirect())
    digest = hashlib.sha256()
    count = 0
    with opener.open(Request(release.url, headers={"User-Agent": "TeAL-model-manager/1"}), timeout=timeout) as response:
        header = response.headers.get("Content-Length")
        total = int(header) if header and header.isdecimal() else None
        if total is not None and total > release.max_wheel_bytes:
            raise SpacyResourceError("Model wheel too large.")
        with destination.open("wb") as output:
            while True:
                if cancelled is not None and cancelled():
                    raise SpacyResourceError("Download cancelled.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > release.max_wheel_bytes:
                    raise SpacyResourceError("Model wheel too large.")
                digest.update(chunk)
                output.write(chunk)
                if progress is not None:
                    progress(count, total)
            output.flush()
            os.fsync(output.fileno())
    if total is not None and count != total:
        raise SpacyResourceError("Incomplete model download.")
    if digest.hexdigest() != release.wheel_sha256:
        raise SpacyResourceError("Official wheel SHA-256 mismatch.")


def _check_wheel_sha256(wheel, release):
    if wheel.stat().st_size > release.max_wheel_bytes:
        raise SpacyResourceError("Model wheel too large.")
    digest = hashlib.sha256()
    with wheel.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != release.wheel_sha256:
        raise SpacyResourceError("Model wheel checksum mismatch.")


def _extract_pipeline(wheel, destination, release):
    prefix = (release.name, f"{release.name}-{release.version}")
    total = 0
    seen = set()
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            raw = info.filename
            parts = PurePosixPath(raw).parts
            if (not raw or raw.startswith("/") or "\\" in raw or ":" in raw
                or "//" in raw or any(p in ("", ".", "..") for p in raw.split("/"))):
                raise SpacyResourceError("Unsafe path in spaCy wheel.")
            mode = (info.external_attr >> 16) & 0xffff
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise SpacyResourceError("Symlink or special archive member.")
            if parts[:2] != prefix or len(parts) <= 2 or info.is_dir():
                continue
            relative = parts[2:]
            if any(p in ("__pycache__", ".git") for p in relative):
                raise SpacyResourceError("Unexpected model data.")
            if Path(relative[-1]).suffix.lower() in {".py", ".pyc", ".pth", ".so", ".dll", ".dylib", ".pyd", ".exe", ".sh", ".bat", ".ps1"}:
                raise SpacyResourceError("Executable model payload rejected.")
            key = "/".join(relative).casefold()
            if key in seen:
                raise SpacyResourceError("Duplicate model file.")
            seen.add(key)
            total += info.file_size
            if len(seen) > 5000 or total > release.max_unpacked_bytes:
                raise SpacyResourceError("Model extraction limits exceeded.")
            if info.compress_size and info.file_size > 500 * info.compress_size:
                raise SpacyResourceError("Suspicious archive compression ratio.")
            target = destination.joinpath(*relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if not seen:
        raise SpacyResourceError("No spaCy pipeline data in official wheel.")


def _publish_pinned_wheel(models_root, release, wheel):
    _check_wheel_sha256(wheel, release)
    root = spacy_cache_root(models_root)
    parent = root / release.name / release.version
    if any(p.is_symlink() for p in (root, parent.parent, parent)):
        raise SpacyResourceError("Symlinked model cache path.")
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teal-stage-", dir=parent) as temp:
        entry = Path(temp) / "entry"
        pipeline = entry / "pipeline"
        pipeline.mkdir(parents=True)
        _extract_pipeline(wheel, pipeline, release)
        import spacy
        _metadata(pipeline, release.name, release.version, spacy.__version__)
        try:
            spacy.load(pipeline)
        except Exception as exc:
            raise SpacyResourceError("Model cannot load in this Python environment.") from exc
        digest = directory_sha256(pipeline)
        destination = parent / digest
        manifest = {
            "provider": "spacy", "model_id": release.name,
            "model_version": release.version, "digest_algorithm": "sha256-tree-v1",
            "content_sha256": digest, "source_url": release.url,
            "source_wheel_sha256": release.wheel_sha256,
        }
        (entry / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if destination.exists() or destination.is_symlink():
            return verify_managed_spacy(models_root, release.name, release.version, digest)
        entry.rename(destination)
        return verify_managed_spacy(models_root, release.name, release.version, digest)


def install_official_spacy(models_root, model_name, *, allow_download=False, progress=None, cancelled=None):
    if model_name not in OFFICIAL_SPACY_RELEASES:
        raise SpacyResourceError(f"No approved spaCy release for {model_name!r}.")
    try:
        return resolve_spacy(models_root, model_name, allow_download=False)
    except SpacyResourceError:
        if not allow_download:
            raise
    import spacy
    from packaging.specifiers import SpecifierSet
    if spacy.__version__ not in SpecifierSet(">=3.8,<3.9"):
        raise SpacyResourceError("No pinned model release for this spaCy version.")
    release = OFFICIAL_SPACY_RELEASES[model_name]
    root = spacy_cache_root(models_root)
    model_root = root / model_name
    if root.is_symlink() or model_root.is_symlink():
        raise SpacyResourceError("Symlinked model cache path.")
    with _install_lock(model_root / ".install.lock"):
        try:
            return resolve_spacy(models_root, model_name, allow_download=False)
        except SpacyResourceError:
            pass
        with tempfile.TemporaryDirectory(prefix=".teal-download-", dir=model_root) as temp:
            wheel = Path(temp) / "official.whl"
            _download_wheel(release, wheel, progress=progress, cancelled=cancelled)
            return _publish_pinned_wheel(models_root, release, wheel)
