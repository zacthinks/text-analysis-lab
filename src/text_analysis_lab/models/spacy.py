"""Offline resolution of immutable spaCy pipeline data in the user-level cache.

The managed layout is models/spacy/<name>/<version>/<tree-sha256>/{manifest.json,pipeline/}.
No package installs or network operations occur here. Hashes identify bytes, not trust:
only a trusted installer may publish entries from approved official sources.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path

APPROVED_SPACY_MODELS = frozenset(("en_core_web_sm", "en_core_web_md", "en_core_web_lg"))
DIGEST_ALGORITHM = "sha256-tree-v1"
_HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class SpacyResourceError(ValueError):
    """Unavailable, invalid, or incompatible spaCy resource."""


@dataclass(frozen=True, slots=True)
class SpacyResolution:
    model_id: str
    version: str
    source: str
    load_target: str
    digest: str | None = None

    def frozen_identity(self) -> dict[str, str]:
        if self.source != "managed_cache" or self.digest is None:
            raise SpacyResourceError("Only managed spaCy entries have this identity.")
        return {
            "kind": "managed_cache",
            "name": self.model_id,
            "model_version": self.version,
            "digest_algorithm": DIGEST_ALGORITHM,
            "content_sha256": self.digest,
        }


def spacy_cache_root(models_root: Path) -> Path:
    return Path(models_root) / "spacy"


def directory_sha256(directory: Path) -> str:
    """Versioned, root-independent digest of sorted relative paths and file bytes.

    Reject symlinks and non-regular entries; do not follow out-of-tree links.
    """
    import stat

    if not directory.is_dir() or directory.is_symlink():
        raise SpacyResourceError(f"Not a regular spaCy pipeline directory: {directory}")
    digest = hashlib.sha256()
    seen = False
    for entry in sorted(directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()):
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SpacyResourceError(f"Symlink in managed spaCy data: {entry}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise SpacyResourceError(f"Non-file in managed spaCy data: {entry}")
        relative = entry.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with entry.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        seen = True
    if not seen:
        raise SpacyResourceError("Managed spaCy pipeline is empty.")
    return digest.hexdigest()


def _metadata(directory: Path, name: str, version: str, spacy_version: str) -> None:
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not an object")
        if (str(meta.get("lang", "")) + "_" + str(meta.get("name", ""))) != name:
            # spaCy meta files normally store "core_web_sm", not "en_core_web_sm".
            if meta.get("name") != name:
                raise ValueError("model name mismatch")
        if str(meta.get("version", "")) != version:
            raise ValueError("model version mismatch")
        from packaging.specifiers import SpecifierSet

        constraint = str(meta["spacy_version"])
        if not constraint or spacy_version not in SpecifierSet(constraint):
            raise ValueError(f"requires spaCy {constraint}; found {spacy_version}")
        if not (directory / "config.cfg").is_file():
            raise ValueError("missing config.cfg")
        if not (directory / "vocab").is_dir() or not (directory / "tokenizer").is_file():
            raise ValueError("missing standard spaCy pipeline data")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SpacyResourceError(f"Invalid or incompatible spaCy data at {directory}: {exc}") from exc


def verify_managed_spacy(
    models_root: Path,
    name: str,
    version: str,
    digest: str,
    *,
    spacy_version: str | None = None,
) -> SpacyResolution:
    """Verify an exact cache identity without consulting packages or another version."""
    if name not in APPROVED_SPACY_MODELS or not _VERSION.fullmatch(version) or not _HEX_SHA256.fullmatch(digest):
        raise SpacyResourceError("Invalid managed spaCy identity.")
    entry = spacy_cache_root(models_root) / name / version / digest
    pipeline = entry / "pipeline"
    if any(path.is_symlink() for path in (entry.parent.parent, entry.parent, entry, pipeline)):
        raise SpacyResourceError("Symlink in managed spaCy cache entry.")
    try:
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "provider": "spacy",
            "model_id": name,
            "model_version": version,
            "digest_algorithm": DIGEST_ALGORITHM,
            "content_sha256": digest,
        }
        if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("manifest identity mismatch")
        if (entry / "manifest.json").is_symlink():
            raise ValueError("symlinked manifest")
    except (OSError, ValueError, TypeError) as exc:
        raise SpacyResourceError(f"Invalid managed spaCy manifest: {exc}") from exc

    if spacy_version is None:
        try:
            spacy_version = importlib.metadata.version("spacy")
        except importlib.metadata.PackageNotFoundError as exc:
            raise SpacyResourceError("spaCy is not installed in the selected environment.") from exc
    _metadata(pipeline, name, version, spacy_version)
    observed = directory_sha256(pipeline)
    if observed != digest:
        raise SpacyResourceError(f"Managed spaCy content fingerprint mismatch: {name} {version}.")
    return SpacyResolution(name, version, "managed_cache", str(pipeline), digest)


def available_managed_spacy(models_root: Path, name: str) -> tuple[tuple[str, str], ...]:
    """Enumerate *unverified* identities without imports or filesystem mutation."""
    if name not in APPROVED_SPACY_MODELS:
        return ()
    root = spacy_cache_root(models_root) / name
    if not root.is_dir() or root.is_symlink():
        return ()
    found = []
    for versions in root.iterdir():
        if not versions.is_dir() or versions.is_symlink() or not _VERSION.fullmatch(versions.name):
            continue
        for entry in versions.iterdir():
            if (entry.is_dir() and not entry.is_symlink()
                    and _HEX_SHA256.fullmatch(entry.name)):
                found.append((versions.name, entry.name))
    return tuple(sorted(found, reverse=True))


def resolve_spacy(
    models_root: Path,
    name: str,
    *,
    allow_download: bool = False,
    spacy_version: str | None = None,
) -> SpacyResolution:
    """Prefer a verified compatible managed entry, then a loadable installed package.

    Does not install or download. For frozen replays use verify_managed_spacy with
    the recorded identity rather than this *unfrozen* preference resolver.
    """
    if name not in APPROVED_SPACY_MODELS:
        raise SpacyResourceError(f"Unknown or unsupported spaCy model {name!r}.")
    failures: list[str] = []
    for version, digest in available_managed_spacy(models_root, name):
        try:
            return verify_managed_spacy(
                models_root, name, version, digest, spacy_version=spacy_version
            )
        except SpacyResourceError as exc:
            failures.append(str(exc))
    try:
        package_version = importlib.metadata.version(name)
        import spacy

        spacy.load(name)
    except (importlib.metadata.PackageNotFoundError, ImportError, OSError, ValueError) as exc:
        failures.append(f"Installed spaCy package unavailable or incompatible: {exc}")
    else:
        return SpacyResolution(name, package_version, "package", name)
    if allow_download:
        raise SpacyResourceError(
            "Automatic spaCy download is not implemented yet; explicitly install a "
            "compatible package or supply a verified managed entry. " + "; ".join(failures)
        )
    raise SpacyResourceError(
        f"spaCy model {name!r} is unavailable offline. " + "; ".join(failures)
    )
