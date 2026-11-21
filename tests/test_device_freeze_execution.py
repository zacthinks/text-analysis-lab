"""Device portability, opt-in durable locks, and schema-v2 legacy compatibility."""

from __future__ import annotations

import json

import pytest

from text_analysis_lab.core.errors import OperatorError
from text_analysis_lab.core.operator import BaseOperator
from text_analysis_lab.linguistics import device as device_module
from text_analysis_lab.translators.coreference_resolver import CoreferenceResolver
from text_analysis_lab.translators.semantic_role_labeler import SemanticRoleLabeler
from text_analysis_lab.translators.word_sense_disambiguator import WordSenseDisambiguator
from text_analysis_lab.translators.sentence_transformer_encoder import SentenceTransformerEncoder
from text_analysis_lab.translators.contextual_transformer import ContextualTransformer


COMMIT = "0123456789abcdef0123456789abcdef01234567"
LINGUISTIC = (CoreferenceResolver, SemanticRoleLabeler, WordSenseDisambiguator)


@pytest.mark.parametrize("cls", LINGUISTIC)
def test_new_freeze_omits_placement_and_replays_gpu_origin_on_cpu(cls, tmp_path, monkeypatch):
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (True, 1, False))
    operator = cls(device="cuda:0")
    directory = operator.save_to_dir(tmp_path / "frozen", operator_id="operator-1")
    descriptor = json.loads((directory / "operator.json").read_text())
    assert "device" not in descriptor["json_state"]
    assert "strict_device" not in descriptor["json_state"]
    assert "device_lock" not in descriptor
    frozen = BaseOperator.load_from_dir(directory)
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (False, 0, False))
    assert frozen.resolve_execution_device("cpu") == "cpu"
    assert frozen.to_json_state() == descriptor["json_state"]
    with pytest.raises(Exception):
        frozen.model_name = "different"


@pytest.mark.parametrize("cls", LINGUISTIC)
def test_explicit_freeze_device_is_concrete_strict_and_separate(cls, tmp_path, monkeypatch):
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (True, 1, False))
    operator = cls(device="auto")
    directory = operator.save_to_dir(
        tmp_path / "frozen", operator_id="operator-1", freeze_device=True
    )
    descriptor = json.loads((directory / "operator.json").read_text())
    assert descriptor["device_lock"] == {"device": "cuda:0"}
    assert "device" not in descriptor["json_state"]
    frozen = BaseOperator.load_from_dir(directory)
    assert frozen.resolve_execution_device(None) == "cuda:0"
    with pytest.raises(OperatorError, match="Frozen device lock"):
        frozen.resolve_execution_device("cpu")
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (False, 0, False))
    with pytest.raises(OperatorError, match="unavailable"):
        frozen.resolve_execution_device(None)


@pytest.mark.parametrize("cls", LINGUISTIC)
def test_legacy_v2_device_fields_remain_unchanged_and_preserve_fallback(
    cls, tmp_path, monkeypatch
):
    directory = cls().save_to_dir(tmp_path / "old", operator_id="legacy")
    descriptor_path = directory / "operator.json"
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["json_state"]["device"] = "cuda:0"
    if cls is not WordSenseDisambiguator:
        descriptor["json_state"]["strict_device"] = False
    descriptor_path.write_text(json.dumps(descriptor))
    previous = descriptor_path.read_bytes()
    loaded = BaseOperator.load_from_dir(directory)
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (False, 0, False))
    assert loaded.resolve_execution_device(None) is None
    assert loaded.resolve_execution_device("cpu") == "cpu"
    assert loaded.to_json_state()["device"] == "cuda:0"
    assert descriptor_path.read_bytes() == previous
    if cls is not WordSenseDisambiguator:
        descriptor["json_state"]["strict_device"] = True
        descriptor_path.write_text(json.dumps(descriptor))
        strict = BaseOperator.load_from_dir(directory)
        with pytest.raises(RuntimeError, match="CUDA"):
            strict.resolve_execution_device("cpu")


@pytest.mark.parametrize("cls", (SentenceTransformerEncoder, ContextualTransformer))
def test_hf_freeze_default_portable_and_explicit_cpu_lock(cls, tmp_path, monkeypatch):
    monkeypatch.setattr(cls, "prepare_for_freeze", lambda self: None)
    operator = cls("example/model", revision=COMMIT)
    directory = operator.save_to_dir(tmp_path / "portable", operator_id="portable")
    descriptor = json.loads((directory / "operator.json").read_text())
    assert "device_lock" not in descriptor
    assert "device" not in descriptor["json_state"]
    locked = cls("example/model", revision=COMMIT)
    directory2 = locked.save_to_dir(
        tmp_path / "locked", operator_id="locked", freeze_device=True, device="cpu"
    )
    recovered = BaseOperator.load_from_dir(directory2)
    assert recovered.resolve_execution_device(None) == "cpu"
    with pytest.raises(OperatorError, match="Frozen device lock"):
        recovered.resolve_execution_device("mps")


def test_explicit_device_lock_rejects_unsupported_operator(tmp_path):
    from text_analysis_lab.translators.regex_cleaner import RegexCleaner

    with pytest.raises(OperatorError, match="does not support"):
        RegexCleaner().save_to_dir(
            tmp_path / "invalid", operator_id="invalid", freeze_device=True
        )


def test_wsd_fp16_refuses_cpu_before_loading_any_ontology_or_model(monkeypatch):
    """Explicit CPU cannot silently downgrade a frozen FP16 scientific choice."""
    import pandas as pd
    import text_analysis_lab.translators.word_sense_disambiguator as module

    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (False, 0, False))

    def unwanted(**kwargs):
        pytest.fail("CPU-incompatible FP16 must fail before resource preparation")

    monkeypatch.setattr(module, "_make_ontology", unwanted)
    monkeypatch.setattr(module, "_make_backend", unwanted)
    tokens = pd.DataFrame({
        "row_id": [1], "sentence_id": [0], "token_id": [0],
        "text": ["Dogs"], "lemma": ["dog"], "pos": ["NOUN"], "ent_type": [""],
    })
    frozen = WordSenseDisambiguator(device="cpu", precision=16)
    with pytest.raises(ValueError, match="float16.*CUDA"):
        frozen.translate(
            tokens, token_keys=["row_id", "sentence_id", "token_id"], device="cpu"
        )


def test_srl_autocast_only_on_cuda_and_retains_mixed_precision_choice(monkeypatch, tmp_path):
    """Exercise the actual SRL conditional autocast path with CPU-backed fake tensors."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from text_analysis_lab.linguistics.srl.runtime import (
        AllenNlpSrlRuntime, EncodedSrlSentence,
    )

    observed = []
    old_full, old_zeros, old_tensor = torch.full, torch.zeros, torch.tensor

    def wrap(creator):
        def create(*args, **kwargs):
            kwargs.pop("device", None)
            return creator(*args, **kwargs)
        return create

    def fake_autocast(*, device_type, dtype):
        observed.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch, "full", wrap(old_full))
    monkeypatch.setattr(torch, "zeros", wrap(old_zeros))
    monkeypatch.setattr(torch, "tensor", wrap(old_tensor))
    monkeypatch.setattr(torch, "autocast", fake_autocast)
    encoded = EncodedSrlSentence(
        tokens=("Alice", "runs"),
        wordpieces=("Alice", "runs"),
        input_ids=(101, 102),
        wordpiece_starts=(0, 1),
        wordpiece_ends=(1, 2),
    )

    for selected in ("cpu", "cuda:0"):
        runtime = AllenNlpSrlRuntime(
            tmp_path, device=selected, mixed_precision=True
        )
        def fake_model(input_ids, predicate_indicators, attention_mask):
            return old_zeros((*input_ids.shape, 1), dtype=torch.float32)
        monkeypatch.setattr(
            runtime, "_load",
            lambda selected=selected: (
                torch, SimpleNamespace(pad_token_id=0), fake_model,
                ("O",), {}, selected, 8,
            ),
        )
        result = runtime.predict_encoded_batch(((encoded, 1),))
        assert len(result) == 1
        assert result[0].tokens == encoded.tokens
        assert runtime.mixed_precision is True

    assert observed == [("cuda", torch.float16)]


def test_unlocked_hf_explicit_cuda_index_checked_before_model_load(monkeypatch):
    from types import SimpleNamespace
    import sys

    from text_analysis_lab.translators._hf_utils import (
        TransformerResourceError, resolve_device,
    )

    mock_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(TransformerResourceError, match="index 99.*unavailable"):
        resolve_device("cuda:99")
    mock_torch.cuda.is_available = lambda: False
    assert resolve_device("auto") == "cpu"
    with pytest.raises(TransformerResourceError, match="index 0.*unavailable"):
        resolve_device("cuda:0")


@pytest.mark.parametrize("cls", LINGUISTIC)
def test_implicit_linguistic_device_provenance_distinguishes_fallback_from_binding(
    cls, monkeypatch
):
    # Fallback remains a run-specific observation, never a new frozen lock.
    monkeypatch.setattr(device_module, "_torch_capabilities", lambda: (False, 0, False))
    operator = cls(device="cuda:0")
    assert operator.execution_device_provenance() == {
        "requested": "cuda:0", "resolved": "cpu"
    }
    assert operator.execution_device_provenance("cpu") == {
        "requested": "cpu", "resolved": "cpu"
    }
    assert "device" not in operator.to_json_state()
    assert operator._frozen_device_lock is None
