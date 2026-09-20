from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.translators import (
    ContextualTransformer,
    ContextWindowExceededError,
    SentenceTransformerEncoder,
)

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeTokenizer:
    model_max_length = 8
    model_input_names: ClassVar[list[str]] = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    init_kwargs: ClassVar[dict[str, str]] = {"_commit_hash": _COMMIT}
    is_fast = True

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
        **kwargs,
    ):
        _ = kwargs
        torch = pytest.importorskip("torch")
        if isinstance(texts, str):
            texts = [texts]
        id_rows = []
        offset_rows = []
        special_rows = []
        for text in texts:
            words = str(text).split()
            ids = [101] if add_special_tokens else []
            offsets = [(0, 0)] if add_special_tokens else []
            specials = [1] if add_special_tokens else []
            cursor = 0
            for word in words:
                start = str(text).find(word, cursor)
                end = start + len(word)
                cursor = end
                ids.append(1000 + len(word))
                offsets.append((start, end))
                specials.append(0)
            if add_special_tokens:
                ids.append(102)
                offsets.append((0, 0))
                specials.append(1)
            if truncation and max_length is not None and len(ids) > int(max_length):
                ids = ids[: int(max_length)]
                offsets = offsets[: int(max_length)]
                specials = specials[: int(max_length)]
                if add_special_tokens:
                    ids[-1] = 102
                    offsets[-1] = (0, 0)
                    specials[-1] = 1
            id_rows.append(ids)
            offset_rows.append(offsets)
            special_rows.append(specials)
        if return_tensors is None:
            return {"input_ids": id_rows}
        width = max((len(row) for row in id_rows), default=0)
        padded_ids, masks, padded_offsets, padded_specials, type_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        for ids, offsets, specials in zip(
            id_rows, offset_rows, special_rows, strict=True
        ):
            pad = width - len(ids)
            padded_ids.append(ids + [0] * pad)
            masks.append([1] * len(ids) + [0] * pad)
            padded_offsets.append(offsets + [(0, 0)] * pad)
            padded_specials.append(specials + [1] * pad)
            type_ids.append([0] * width)
        result = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "token_type_ids": torch.tensor(type_ids, dtype=torch.long),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(padded_offsets, dtype=torch.long)
        if return_special_tokens_mask:
            result["special_tokens_mask"] = torch.tensor(
                padded_specials, dtype=torch.long
            )
        return result

    def convert_ids_to_tokens(self, ids):
        values = []
        for value in ids:
            if value == 101:
                values.append("[CLS]")
            elif value == 102:
                values.append("[SEP]")
            else:
                values.append(f"tok_{int(value)}")
        return values

    def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
        _ = already_has_special_tokens
        return [1 if value in {101, 102} else 0 for value in ids]

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeAutoModelInstance:
    def __init__(self):
        self.config = SimpleNamespace(
            hidden_size=3, max_position_embeddings=6, _commit_hash=_COMMIT
        )
        self.device = "cpu"

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        torch = pytest.importorskip("torch")
        ids = encoded["input_ids"].to(dtype=torch.float32)
        hidden = torch.stack([ids, ids * 2.0, torch.ones_like(ids)], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, path, **kwargs):
        _ = kwargs
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"fake")


def _install_fake_transformers(monkeypatch, *, fail_cache=False):
    calls = []

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            local = bool(kwargs.get("local_files_only", False))
            calls.append(("tokenizer", str(source), local))
            if fail_cache and local and not Path(str(source)).exists():
                raise OSError("cache miss")
            return FakeTokenizer()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            local = bool(kwargs.get("local_files_only", False))
            calls.append(("model", str(source), local))
            if fail_cache and local and not Path(str(source)).exists():
                raise OSError("cache miss")
            return FakeAutoModelInstance()

    module = types.ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer
    module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def _install_fake_sentence_transformers(monkeypatch, *, fail_cache=False):
    calls = []

    class SentenceTransformer:
        def __init__(self, source, **kwargs):
            local = bool(kwargs.get("local_files_only", False))
            calls.append(("load", str(source), local))
            if fail_cache and local and not Path(str(source)).exists():
                raise OSError("cache miss")
            self.tokenizer = FakeTokenizer()
            self.prompts = {"document": "doc: ", "query": "query: "}
            self.default_prompt_name = None
            self.max_seq_length = 6
            self.first = SimpleNamespace(
                auto_model=FakeAutoModelInstance(),
                tokenizer=self.tokenizer,
            )
            self.encode_calls = []

        def __getitem__(self, index):
            assert index == 0
            return self.first

        def _encode(self, texts, *, kind, normalize_embeddings=False, **kwargs):
            self.encode_calls.append((kind, dict(kwargs)))
            rows = np.asarray(
                [
                    [float(len(text)), float(index + 1), 1.0]
                    for index, text in enumerate(texts)
                ],
                dtype=np.float32,
            )
            if normalize_embeddings:
                rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            return rows

        def encode_document(self, texts, **kwargs):
            return self._encode(texts, kind="document", **kwargs)

        def encode_query(self, texts, **kwargs):
            return self._encode(texts, kind="query", **kwargs)

        def encode(self, texts, **kwargs):
            return self._encode(texts, kind="generic", **kwargs)

        def save_pretrained(self, path, **kwargs):
            _ = kwargs
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "modules.json").write_text("[]", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"fake")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def _packet(texts: list[str]) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_documents",
        primary_key=("doc_id",),
        data=pd.DataFrame({"doc_id": list(range(len(texts))), "text": texts}),
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _request(*, model_batch_size=2):
    return TranslationRequest(
        workers=1,
        batch_size=4,
        params={"device": "cpu", "model_batch_size": model_batch_size},
    )


def test_contextual_transformer_emits_model_tokens_and_aligned_embeddings(monkeypatch):
    _install_fake_transformers(monkeypatch)
    translator = ContextualTransformer("example/model", revision=_COMMIT)
    result = translator.translate_batch(
        {"source": _packet(["one two"])}, mode="translate", request=_request()
    )
    tokens = result.outputs["tokens"]
    contextual = result.outputs["contextual_embeddings"]
    assert tokens["keys"].to_dict("records") == [
        {"doc_id": 0, "token_id": 0},
        {"doc_id": 0, "token_id": 1},
        {"doc_id": 0, "token_id": 2},
        {"doc_id": 0, "token_id": 3},
    ]
    assert tokens["data"]["token"].tolist()[0] == "[CLS]"
    assert tokens["data"]["token"].tolist()[-1] == "[SEP]"
    assert tokens["data"]["char_start"].tolist() == [pd.NA, 0, 4, pd.NA]
    assert tokens["data"]["char_end"].tolist() == [pd.NA, 3, 7, pd.NA]
    assert tokens["data"]["is_special"].tolist() == [True, False, False, True]
    assert tokens["metadata"].iloc[0].to_dict() == {
        "source_token_count": 4,
        "embedded_token_count": 4,
        "truncated": False,
    }
    values = np.asarray(contextual["data"]["values"])
    assert values.shape == (4, 3)
    assert contextual["keys"].equals(tokens["keys"])


def test_contextual_transformer_refuses_and_audits_truncation(monkeypatch):
    _install_fake_transformers(monkeypatch)
    strict = ContextualTransformer("example/model", revision=_COMMIT)
    with pytest.raises(ContextWindowExceededError, match="7 tokens"):
        strict.translate_batch(
            {"source": _packet(["one two three four five"])},
            mode="translate",
            request=_request(),
        )

    truncating = ContextualTransformer(
        "example/model", revision=_COMMIT, truncation="truncate"
    )
    with pytest.warns(UserWarning, match="explicitly truncating"):
        result = truncating.translate_batch(
            {"source": _packet(["one two three four five"])},
            mode="translate",
            request=_request(),
        )
    tokens = result.outputs["tokens"]
    assert len(tokens["keys"]) == 6
    assert tokens["metadata"]["source_token_count"].unique().tolist() == [7]
    assert tokens["metadata"]["embedded_token_count"].unique().tolist() == [6]
    assert tokens["metadata"]["truncated"].unique().tolist() == [True]
    assert len(result.outputs["contextual_embeddings"]["data"]["values"]) == 6


def test_contextual_transformer_output_lineage_uses_token_basis(monkeypatch):
    _install_fake_transformers(monkeypatch)
    translator = ContextualTransformer("example/model", revision=_COMMIT)
    source = SimpleNamespace(primary_key=["doc_id"])
    specs = translator.output_specs(sources={"source": source}, request=_request())
    assert specs["tokens"].lineage_mode == "extended_key"
    assert specs["tokens"].basis_labels == ("source",)
    assert specs["contextual_embeddings"].lineage_mode == "preserved_key"
    assert specs["contextual_embeddings"].basis_labels == ("tokens",)


def test_sentence_transformer_uses_document_recipe_and_counts_prompt(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch)
    encoder = SentenceTransformerEncoder(
        "example/sbert", revision=_COMMIT, task="document", normalize=True
    )
    result = encoder.translate_batch(
        {"source": _packet(["one two", "three"])},
        mode="translate",
        request=_request(model_batch_size=1),
    )
    payload = result.outputs["output"]
    values = np.asarray(payload["data"]["values"])
    assert values.shape == (2, 3)
    assert np.linalg.norm(values, axis=1).tolist() == pytest.approx([1.0, 1.0])
    # document prompt "doc: " adds one model token to each source before specials
    assert payload["metadata"]["token_count"].tolist() == [5, 4]
    model = encoder._runtime_model
    assert model.encode_calls[0][0] == "document"
    assert model.encode_calls[0][1]["prompt_name"] == "document"


def test_sentence_transformer_context_guard_includes_model_prompt(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch)
    encoder = SentenceTransformerEncoder("example/sbert", revision=_COMMIT)
    # 4 source words + 1 prompt word + 2 special tokens = 7 > model limit 6
    with pytest.raises(ContextWindowExceededError) as excinfo:
        encoder.translate_batch(
            {"source": _packet(["one two three four"])},
            mode="translate",
            request=_request(),
        )
    assert "7 tokens" in str(excinfo.value)
    assert "including any model prompt" in str(excinfo.value)


def test_both_transformers_save_operator_local_models(monkeypatch, tmp_path: Path):
    t_calls = _install_fake_transformers(monkeypatch)
    contextual = ContextualTransformer(
        "example/model", revision=_COMMIT, save_model=True
    )
    contextual.download()
    cdir = tmp_path / "contextual"
    contextual.save_to_dir(cdir, operator_id="op_contextual")
    assert (cdir / "assets" / "model" / "model.safetensors").exists()
    t_calls.clear()
    restored_contextual = ContextualTransformer.load_from_dir(cdir)
    restored_contextual.download()
    assert t_calls[0][1] == str(cdir / "assets" / "model")
    assert t_calls[0][2] is True

    s_calls = _install_fake_sentence_transformers(monkeypatch)
    sentence = SentenceTransformerEncoder(
        "example/sbert", revision=_COMMIT, save_model=True
    )
    sentence.download()
    sdir = tmp_path / "sentence"
    sentence.save_to_dir(sdir, operator_id="op_sentence")
    assert (sdir / "assets" / "model" / "modules.json").exists()
    s_calls.clear()
    restored_sentence = SentenceTransformerEncoder.load_from_dir(sdir)
    restored_sentence.download()
    assert s_calls == [("load", str(sdir / "assets" / "model"), True)]


def _seed_documents(project: teal.Project):
    from text_analysis_lab.core.writer import create_artifact_writer

    rows = pd.DataFrame({"doc_id": [1, 2], "text": ["one two", "three"]})
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_transformer_docs"),
        artifact_id="art_transformer_docs",
        label="documents",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["doc_id"]], "data": rows[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_transformer_docs",
        artifact_type="table",
        label="documents",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_transformer_docs")


def test_contextual_transformer_real_teal_two_output_round_trip(
    monkeypatch, tmp_path: Path
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    _install_fake_transformers(monkeypatch)
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="contextual_round_trip")
    try:
        source = _seed_documents(project)
        outputs = project.translate(
            ContextualTransformer("example/model", revision=_COMMIT),
            source,
            batch_size=2,
            device="cpu",
            model_batch_size=2,
        )
        tokens = outputs["tokens"]
        embeddings = outputs["contextual_embeddings"]
        assert tokens.primary_key == ["doc_id", "token_id"]
        assert embeddings.primary_key == ["doc_id", "token_id"]
        assert embeddings.descriptor["lineage"]["basis_artifact_ids"] == [
            tokens.artifact_id
        ]
        assert embeddings.get_matrix().shape == (7, 3)
        token_frame = tokens.query(
            key_columns=True,
            data_columns=True,
            metadata_columns=True,
            metadata_mode="local",
            form="table",
        )
        assert token_frame.groupby("doc_id").size().to_dict() == {1: 4, 2: 3}
        token_id = tokens.artifact_id
        embedding_id = embeddings.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        reopened_tokens = reopened.get_artifact(token_id)
        reopened_embeddings = reopened.get_artifact(embedding_id)
        assert reopened_tokens.primary_key == ["doc_id", "token_id"]
        assert reopened_embeddings.get_matrix().shape == (7, 3)
        assert reopened_embeddings.descriptor["lineage"]["basis_artifact_ids"] == [
            token_id
        ]
    finally:
        reopened.close()


def test_sentence_transformer_real_teal_dense_round_trip(monkeypatch, tmp_path: Path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    _install_fake_sentence_transformers(monkeypatch)
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="sentence_round_trip")
    try:
        source = _seed_documents(project)
        embedded = project.translate(
            SentenceTransformerEncoder("example/sbert", revision=_COMMIT),
            source,
            batch_size=2,
            device="cpu",
            model_batch_size=2,
        )["output"]
        assert embedded.primary_key == ["doc_id"]
        assert embedded.get_matrix().shape == (2, 3)
        artifact_id = embedded.artifact_id
    finally:
        project.close()
    reopened = teal.Project.open(project_path)
    try:
        assert reopened.get_artifact(artifact_id).get_matrix().shape == (2, 3)
    finally:
        reopened.close()
