"""BIO-constrained decoding used by the converted AllenNLP SRL model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def allowed_bio_transitions(labels: Sequence[str]) -> tuple[list[list[float]], list[float]]:
    """Build AllenNLP-compatible zero/-infinity BIO transition potentials."""

    size = len(labels)
    transitions = [[0.0 for _ in range(size)] for _ in range(size)]
    starts = [0.0 for _ in range(size)]
    for current_index, current in enumerate(labels):
        if current.startswith("I-"):
            starts[current_index] = float("-inf")
        for previous_index, previous in enumerate(labels):
            if current.startswith("I-") and previous not in {current, "B-" + current[2:]}:
                transitions[previous_index][current_index] = float("-inf")
    return transitions, starts


def viterbi_decode_bio(emissions: Any, labels: Sequence[str]) -> list[int]:
    """Decode one ``[sequence, labels]`` tensor with valid-BIO constraints.

    ``emissions`` may be a torch tensor or a NumPy-like object. The implementation imports
    torch lazily so the package remains importable before the user installs PyTorch.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
        raise RuntimeError("BIO Viterbi decoding requires PyTorch for the SRL backend.") from exc

    if emissions.ndim != 2:
        raise ValueError("emissions must have shape [sequence_length, number_of_labels]")
    if emissions.shape[0] == 0:
        return []
    if emissions.shape[1] != len(labels):
        raise ValueError("emission label dimension does not match the label vocabulary")

    transition_values, start_values = allowed_bio_transitions(labels)
    transitions = torch.tensor(transition_values, dtype=emissions.dtype, device=emissions.device)
    scores = emissions[0] + torch.tensor(
        start_values, dtype=emissions.dtype, device=emissions.device
    )
    history: list[Any] = []
    for timestep in range(1, emissions.shape[0]):
        candidate_scores = scores[:, None] + transitions
        best_scores, best_previous = candidate_scores.max(dim=0)
        scores = best_scores + emissions[timestep]
        history.append(best_previous)

    best_last = int(scores.argmax().item())
    path = [best_last]
    for backpointers in reversed(history):
        best_last = int(backpointers[best_last].item())
        path.append(best_last)
    path.reverse()
    return path
