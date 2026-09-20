"""Conventional device resolution with explicit, auditable fallback behavior."""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedDevices:
    requested: str | tuple[str, ...]
    devices: tuple[str, ...]
    fell_back: bool

    @property
    def primary(self) -> str:
        return self.devices[0]


def _torch_capabilities() -> tuple[bool, int, bool]:
    try:
        import torch
    except ImportError:
        return False, 0, False

    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    return cuda_available, cuda_count, mps_available


def resolve_devices(
    requested: str | tuple[str, ...],
    *,
    strict: bool = False,
) -> ResolvedDevices:
    """Resolve a user device request to concrete runtime devices.

    Supported requests include ``auto``, ``cpu``, ``cuda``, ``cuda:0``, ``mps``, or a
    tuple of concrete device strings. Multiple GPUs are represented as a tuple; model
    backends decide how to shard work over them.
    """

    cuda_available, cuda_count, mps_available = _torch_capabilities()
    requests = (requested,) if isinstance(requested, str) else requested
    resolved: list[str] = []
    fell_back = False

    for item in requests:
        normalized = item.strip().lower()
        if normalized == "auto":
            if cuda_available:
                resolved.append("cuda:0")
            elif mps_available:
                resolved.append("mps")
            else:
                resolved.append("cpu")
            continue

        if normalized == "cpu":
            resolved.append("cpu")
            continue

        if normalized == "mps":
            if mps_available:
                resolved.append("mps")
                continue
            if strict:
                raise RuntimeError("MPS was requested but is unavailable.")
            warnings.warn("MPS is unavailable; falling back to CPU.", stacklevel=2)
            resolved.append("cpu")
            fell_back = True
            continue

        if normalized == "cuda":
            normalized = "cuda:0"

        if normalized.startswith("cuda:"):
            try:
                index = int(normalized.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {item!r}") from exc

            if cuda_available and 0 <= index < cuda_count:
                resolved.append(f"cuda:{index}")
                continue

            if strict:
                raise RuntimeError(
                    f"{normalized} was requested but CUDA is unavailable or the index is invalid."
                )
            warnings.warn(
                f"{normalized} is unavailable; falling back to CPU.",
                stacklevel=2,
            )
            resolved.append("cpu")
            fell_back = True
            continue

        raise ValueError(f"Unsupported device request: {item!r}")

    # Preserve order while removing duplicates, which avoids accidentally loading a model
    # twice on the same device.
    unique = tuple(dict.fromkeys(resolved))
    return ResolvedDevices(requested=requested, devices=unique, fell_back=fell_back)
