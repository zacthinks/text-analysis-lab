"""Manual, expensive acceptance of actual TeAL model-manager resources.

This is deliberately in scripts/, not tests/: ordinary pytest does not run it.
Use the workflow_dispatch-only GitHub workflow or invoke it explicitly with
TEAL_REAL_PROVIDER_PHASE=prefetch|offline and shared cache environment settings.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

from text_analysis_lab.models import ModelManager
from text_analysis_lab.linguistics.cache import active_hf_cache_path
from text_analysis_lab.linguistics.model_resources import ALLENNLP_SRL_BERT
from text_analysis_lab.linguistics.coreference.runtime import FastCorefRuntime
from text_analysis_lab.linguistics.srl.runtime import AllenNlpSrlRuntime
from text_analysis_lab.linguistics.wsd.wsl_reader import (
    DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
    DEFAULT_WSL_READER_MODEL,
    LocalWSLReaderRuntime,
)


def main() -> None:
    hf_cache = Path(os.environ["HF_HUB_CACHE"])
    teal_cache = Path(os.environ["TEAL_CACHE_DIR"])
    phase = os.environ["TEAL_REAL_PROVIDER_PHASE"]
    manager = ModelManager(cache_root=teal_cache, hf_cache_dir=hf_cache)
    coref_key = "huggingface:biu-nlp/f-coref"
    wsl_key = "huggingface:" + DEFAULT_WSL_READER_MODEL
    srl_key = "allennlp:" + ALLENNLP_SRL_BERT.model_id

    assert active_hf_cache_path("biu-nlp/f-coref") == hf_cache
    assert manager.cache_paths()["huggingface_hub"] == hf_cache

    if phase == "prefetch":
        from transformers import AutoConfig

        manager.install(coref_key, allow_download=True)
        manager.install(wsl_key, allow_download=True)
        # WSL loads a separate pinned DeBERTa base config.
        AutoConfig.from_pretrained(
            "microsoft/deberta-v3-base",
            revision=DEFAULT_DEBERTA_V3_BASE_CONFIG_REVISION,
            cache_dir=str(hf_cache),
        )
        manager.install(srl_key, allow_download=True)
        print("Prefetched real FastCoref, pinned WSL and converted SRL.")
        return

    if phase != "offline":
        raise ValueError(f"Invalid phase {phase!r}; select prefetch or offline.")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TEAL_OFFLINE") == "1"

    assert manager.install(coref_key, allow_download=False).is_dir()
    assert manager.install(wsl_key, allow_download=False).is_dir()
    srl_paths = manager.install(srl_key, allow_download=False)
    assert manager.verify(srl_key).prepared

    coref = FastCorefRuntime(
        teal_cache, model="fcoref", cache_dir=hf_cache,
        device="cpu", show_progress=False,
    )
    result = coref.predict_texts(["Alice met Bob. She greeted him."])
    assert len(result.documents) == 1
    assert coref._model_dir and coref._model_dir.is_dir()
    assert str(coref._model_dir).startswith(str(hf_cache))
    del coref
    gc.collect()

    srl = AllenNlpSrlRuntime(
        srl_paths.runtime_dir, device="cpu", mixed_precision=False,
    )
    encoded = srl.encode_tokens(["Alice", "sees", "Bob", "."])
    assert srl.predict_encoded_batch([(encoded, 1)])
    del srl
    gc.collect()

    reader = LocalWSLReaderRuntime(
        DEFAULT_WSL_READER_MODEL,
        device="cpu",
        cache_dir=hf_cache,
        local_files_only=True,
    )
    scores = reader.score_candidates(
        tokens=["Alice", "runs", "quickly", "."],
        target_start=1,
        target_end=2,
        candidates=["move swiftly", "operate"],
    )
    assert len(scores.candidate_probabilities) == 2
    assert not (hf_cache.parent / "materialized" / "fcoref").exists()
    print("Real offline provider inference and shared-cache reuse succeeded.")


if __name__ == "__main__":
    main()
