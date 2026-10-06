from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from text_analysis_lab.core.errors import (
    ArtifactError,
    OperatorNotFittedError,
    StandaloneTranslationNotSupportedError,
)
from text_analysis_lab.core.operator import (
    BaseTranslator,
    BatchResult,
    ColumnRequest,
    InputBatch,
    OutputSpec,
    SourceRequest,
    TranslationRequest,
)
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab import dictionaries
import text_analysis_lab.translators as translators_module
from text_analysis_lab.translators import (
    ArtifactCountVectorizer,
    ContextualTransformer,
    CoreferenceResolver,
    CountVectorizer,
    DelimiterDecomposer,
    DictionaryTranslator,
    EmbeddingLookup,
    FeatureTrimmer,
    FittedPredictor,
    FunctionMapper,
    GeCoPredictor,
    LDA,
    MatrixNormalizer,
    MatrixRowAggregator,
    MatrixTranspose,
    OpenAIResponsesTranslator,
    RegexCleaner,
    RegexReplaceRule,
    SVD,
    SemanticRoleLabeler,
    SentenceTransformerEncoder,
    SpacyTranslator,
    TextFileExtractor,
    PdfTextExtractor,
    TextLength,
    TfidfTransformer,
    UMAP,
    Word2Vec,
    WordSenseDisambiguator,
)


def _matrix_packet(matrix) -> InputBatch:
    return InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": list(range(int(matrix.shape[0])))}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _matrix_source(kind: str, columns: list[str]):
    return SimpleNamespace(
        artifact_type=ArtifactType(kind),
        primary_key=["doc_id"],
        get_data_columns=lambda: list(columns),
    )


class _InternalOnlyTranslator(BaseTranslator):
    """Minimal translator proving standalone support is not required by inheritance."""

    def output_specs(self, *, sources, request):
        _ = sources, request
        return OutputSpec(artifact_type="table", lineage_mode="preserved_key")

    def input_request(self, *, sources, mode, request):
        _ = sources, mode, request
        return SourceRequest(
            artifact_type="table",
            columns=ColumnRequest(keys=True, data=True, metadata=False),
        )

    def translate_batch(self, inputs, *, mode, request):
        _ = inputs, mode, request
        return BatchResult()

    def handle_batch_result(self, result, *, batch_index, mode, request):
        _ = batch_index, mode, request
        return result.outputs

    def finalize_translation(self, *, mode, request):
        _ = mode, request
        return None


def test_every_public_translator_overrides_standalone_translate() -> None:
    public_translators = {
        value
        for name in translators_module.__all__
        if inspect.isclass(value := getattr(translators_module, name))
        and issubclass(value, BaseTranslator)
        and value is not BaseTranslator
    }
    missing = sorted(
        cls.__name__
        for cls in public_translators
        if cls.translate is BaseTranslator.translate
    )
    assert missing == []


def test_base_translator_allows_internal_execution_only_subclasses() -> None:
    with pytest.raises(
        StandaloneTranslationNotSupportedError,
        match="does not expose a standalone",
    ):
        _InternalOnlyTranslator().translate(["not", "a", "public", "translator"])


def test_regex_cleaner_standalone_matches_teal_batch_semantics() -> None:
    translator = RegexCleaner(
        rules=[
            RegexReplaceRule(r"hello", "hi", flags=("IGNORECASE",)),
            {"pattern": r"[!-]+", "replacement": " "},
            {"pattern": r"\s+", "replacement": " "},
        ]
    )
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": ["  HELLO   world!!! ", None, "Hello---TeAL"],
        }
    )

    direct = translator.translate(frame["text"])
    assert isinstance(direct, pd.Series)
    assert direct.tolist()[0] == "hi world"
    assert pd.isna(direct.tolist()[1])
    assert direct.tolist()[2] == "hi TeAL"
    assert translator.translate("  HELLO!!! ") == "hi"
    assert translator.translate(None) is None

    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["text"]
    pd.testing.assert_series_equal(
        batch.reset_index(drop=True),
        direct.reset_index(drop=True),
        check_names=False,
    )


def test_tfidf_standalone_matches_teal_batch_and_validates_state() -> None:
    train = sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    new_rows = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    translator = TfidfTransformer(norm=None)
    translator.source_features_ = ("alpha", "beta", "gamma")
    translator._transformer = translator._make_transformer().fit(train)

    direct = translator.translate(new_rows)
    assert sparse.isspmatrix_csr(direct)

    batch = translator.translate_batch(
        {"source": _matrix_packet(new_rows)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_allclose(batch.toarray(), direct.toarray())

    with pytest.raises(ValueError, match="fitted feature width"):
        translator.translate(sparse.csr_matrix([[1.0, 2.0]]))
    with pytest.raises(OperatorNotFittedError, match="fitted IDF state"):
        TfidfTransformer().translate(new_rows)


def test_feature_trimmer_standalone_is_semantically_equivalent_to_lazy_teal_view() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )
    translator = FeatureTrimmer()
    translator.source_width_ = 4
    translator.kept_indices_ = (0, 2)

    direct = translator.translate(matrix)
    np.testing.assert_array_equal(direct.toarray(), matrix[:, [0, 2]].toarray())

    key_packet = InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id",),
        data=pd.DataFrame({"doc_id": [0, 1]}),
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    lazy_payload = translator.translate_batch(
        {"source": key_packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert lazy_payload["feature_indices"] == [0, 2]
    np.testing.assert_array_equal(
        direct.toarray(),
        matrix[:, lazy_payload["feature_indices"]].toarray(),
    )

    dense = matrix.toarray()
    dense_direct = translator.translate(dense)
    assert isinstance(dense_direct, np.ndarray)
    np.testing.assert_array_equal(dense_direct, dense[:, [0, 2]])


@pytest.mark.parametrize(
    ("axis", "norm", "kind"),
    [
        ("rows", "l2", "sparse_matrix"),
        ("columns", "l1", "dense_matrix"),
    ],
)
def test_matrix_normalizer_standalone_matches_teal_batch(
    axis: str,
    norm: str,
    kind: str,
) -> None:
    dense = np.array(
        [
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    matrix = sparse.csr_matrix(dense) if kind == "sparse_matrix" else dense
    columns = ["alpha", "beta", "gamma"]
    translator = MatrixNormalizer(axis=axis, norm=norm)
    translator.input_request(
        sources={"source": _matrix_source(kind, columns)},
        mode="translate",
        request=TranslationRequest(batch_size=2),
    )

    direct = translator.translate(matrix)
    batch = translator.translate_batch(
        {"source": _matrix_packet(matrix)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]

    if sparse.issparse(direct):
        assert sparse.isspmatrix_csr(batch)
        np.testing.assert_allclose(batch.toarray(), direct.toarray())
    else:
        assert isinstance(batch, np.ndarray)
        np.testing.assert_allclose(batch, direct)

    with pytest.raises(ValueError, match="two-dimensional"):
        translator.translate(np.array([1.0, 2.0, 3.0]))



def test_count_vectorizer_standalone_matches_teal_batch() -> None:
    translator = CountVectorizer(vocabulary={"alpha": 0, "beta": 1})
    texts = ["alpha beta alpha", None, "beta"]

    direct = translator.translate(texts)
    assert sparse.isspmatrix_csr(direct["values"])
    assert direct["feature_metadata"]["column"].tolist() == ["alpha", "beta"]

    frame = pd.DataFrame({"doc_id": [0, 1, 2], "text": texts})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_array_equal(batch.toarray(), direct["values"].toarray())

    with pytest.raises(OperatorNotFittedError, match="fitted vocabulary"):
        CountVectorizer().translate(["alpha"])


def test_standalone_matrix_chain_preserves_and_projects_feature_metadata() -> None:
    count = CountVectorizer(
        vocabulary={"alpha": 0, "beta": 1, "gamma": 2}
    )
    counts = count.translate(
        ["alpha beta gamma", "alpha alpha gamma", "beta gamma"]
    )
    counts["feature_metadata"]["family"] = ["left", "middle", "right"]
    counts["feature_metadata"]["custom"] = [10, 20, 30]

    trimmer = FeatureTrimmer()
    trimmer.source_width_ = 3
    trimmer.kept_indices_ = (0, 2)
    trimmed = trimmer.translate(counts)
    assert trimmed["feature_metadata"].to_dict("list") == {
        "column_index": [0, 1],
        "column": ["alpha", "gamma"],
        "family": ["left", "right"],
        "custom": [10, 30],
    }

    dictionary = dictionaries.Dictionary(
        {"alpha_code": ["alpha"], "gamma_code": ["gamma"]},
        valuetype="fixed",
        case_sensitive=True,
    )
    translated = DictionaryTranslator(dictionary).translate(trimmed)
    assert translated["feature_metadata"]["column"].tolist() == [
        "alpha_code",
        "gamma_code",
    ]

    tfidf = TfidfTransformer(norm=None)
    tfidf.source_features_ = ("alpha", "gamma")
    tfidf._transformer = tfidf._make_transformer().fit(trimmed["values"])
    weighted = tfidf.translate(trimmed)
    pd.testing.assert_frame_equal(
        weighted["feature_metadata"],
        trimmed["feature_metadata"],
    )

    normalized = MatrixNormalizer(norm="l2").translate(weighted)
    pd.testing.assert_frame_equal(
        normalized["feature_metadata"],
        trimmed["feature_metadata"],
    )

    svd = SVD(n_components=1, random_state=0)
    svd.source_features_ = ("alpha", "gamma")
    svd._estimator = svd._make_estimator().fit(normalized["values"])
    reduced = svd.translate(normalized)
    assert reduced["values"].shape == (3, 1)
    assert reduced["feature_metadata"].to_dict("list") == {
        "column_index": [0],
        "column": ["component_0"],
    }


def test_svd_standalone_matches_teal_batch_and_validates_width() -> None:
    train = sparse.csr_matrix(
        [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [1.0, 1.0, 0.0]]
    )
    new_rows = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    translator = SVD(n_components=2, random_state=0)
    translator.source_features_ = ("alpha", "beta", "gamma")
    translator._estimator = translator._make_estimator().fit(train)

    direct = translator.translate(new_rows)
    batch = translator.translate_batch(
        {"source": _matrix_packet(new_rows)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_allclose(batch, direct)

    with pytest.raises(ValueError, match="fitted feature width"):
        translator.translate(np.array([[1.0, 2.0]]))
    with pytest.raises(OperatorNotFittedError, match="fitted decomposition"):
        SVD().translate(new_rows)


def test_lda_standalone_matches_teal_batch_and_validates_counts() -> None:
    train = sparse.csr_matrix(
        [[2, 0, 1], [0, 3, 1], [1, 1, 0], [4, 0, 2]],
        dtype=float,
    )
    new_rows = sparse.csr_matrix([[1, 0, 2], [0, 1, 1]], dtype=float)
    translator = LDA(n_components=2, max_iter=3, random_state=0)
    translator.source_features_ = ("alpha", "beta", "gamma")
    translator._estimator = translator._make_estimator().fit(train)

    direct = translator.translate(new_rows)
    batch = translator.translate_batch(
        {"source": _matrix_packet(new_rows)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_allclose(batch, direct)

    with pytest.raises(ArtifactError, match="non-negative"):
        translator.translate(np.array([[1.0, -1.0, 0.0]]))
    with pytest.raises(OperatorNotFittedError, match="fitted topic model"):
        LDA().translate(new_rows)


def test_fitted_predictor_standalone_matches_teal_batch() -> None:
    from sklearn.linear_model import LogisticRegression

    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0).fit(train_x, train_y)
    values = np.array([[0.5], [2.5]])
    translator = FittedPredictor(model, probability_class=1)

    direct = translator.translate(values)
    assert list(direct.columns) == ["prediction", "probability"]

    translator._source_type = "dense_matrix"
    batch = translator.translate_batch(
        {"source": _matrix_packet(values)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]
    pd.testing.assert_frame_equal(batch, direct)

    with pytest.raises(OperatorNotFittedError, match="unavailable"):
        FittedPredictor(None).translate(values)


def test_text_length_standalone_supports_text_records_and_frames() -> None:
    translator = TextLength(
        {
            "text": ("characters", "words"),
            "title": "log_characters",
        }
    )
    frame = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "text": ["hello world", None],
            "title": ["abc", "longer"],
        },
        index=[10, 20],
    )

    direct = translator.translate(frame[["text", "title"]])
    assert isinstance(direct, pd.DataFrame)
    assert direct.index.tolist() == [10, 20]
    assert direct.loc[10, "text_characters"] == 11
    assert direct.loc[10, "text_words"] == 2

    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame.reset_index(drop=True),
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["metadata"]
    pd.testing.assert_frame_equal(batch, direct.reset_index(drop=True))

    single = TextLength({"text": "words"})
    assert single.translate("one two") == {"text_words": 2}
    assert single.translate({"text": "one two three"}) == {"text_words": 3}


def test_transformer_standalone_runtime_defaults_match_teal_contract() -> None:
    contextual_signature = inspect.signature(ContextualTransformer.translate)
    sentence_signature = inspect.signature(SentenceTransformerEncoder.translate)

    assert contextual_signature.parameters["device"].default == "auto"
    assert contextual_signature.parameters["model_batch_size"].default == 16
    assert sentence_signature.parameters["device"].default == "auto"
    assert sentence_signature.parameters["model_batch_size"].default == 32


def test_sentence_transformer_standalone_matches_teal_batch_with_fixed_task(
    monkeypatch,
) -> None:
    import text_analysis_lab.translators.sentence_transformer_encoder as module

    class FakeModel:
        tokenizer = object()
        prompts = {}

        def encode_document(self, texts, **kwargs):
            _ = kwargs
            return np.asarray([[len(text), 1.0] for text in texts], dtype=float)

        def encode_query(self, texts, **kwargs):
            _ = kwargs
            return np.asarray([[len(text), 2.0] for text in texts], dtype=float)

        def encode(self, texts, **kwargs):
            _ = kwargs
            return np.asarray([[len(text), 0.0] for text in texts], dtype=float)

    document = SentenceTransformerEncoder("fake-model", task="document")
    query = SentenceTransformerEncoder("fake-model", task="query")
    fake_model = FakeModel()
    monkeypatch.setattr(module, "resolve_device", lambda value: "cpu")
    monkeypatch.setattr(
        module,
        "count_tokens",
        lambda tokenizer, texts: [len(text.split()) for text in texts],
    )
    for translator in (document, query):
        monkeypatch.setattr(
            translator,
            "_runtime_component",
            lambda *, device: fake_model,
        )
        monkeypatch.setattr(
            translator,
            "_context_limit",
            lambda *, model, tokenizer: 100,
        )

    texts = ["hello world", "TeAL"]
    direct = document.translate(texts)
    query_values = query.translate(texts)
    series_query = query.translate(pd.Series(texts))
    assert direct["values"].dtype == np.float32
    assert query_values["values"][:, 1].tolist() == [2.0, 2.0]
    np.testing.assert_array_equal(series_query["values"], query_values["values"])
    assert query_values["feature_metadata"]["column"].tolist() == ["dim_0", "dim_1"]

    frame = pd.DataFrame({"doc_id": [1, 2], "text": texts})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = query.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(
            params={"device": "cpu", "model_batch_size": 32}
        ),
    ).outputs["output"]
    np.testing.assert_allclose(batch["data"]["values"], query_values["values"])
    pd.testing.assert_frame_equal(batch["metadata"], query_values["metadata"])



def test_matrix_row_aggregator_standalone_matches_teal_batch() -> None:
    matrix = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=float,
    )
    info = pd.DataFrame(
        {
            "doc_id": [1, 1, 2],
            "sentence_id": [0, 0, 0],
            "token_id": [0, 1, 0],
        }
    )
    translator = MatrixRowAggregator(group_by=["doc_id"], pooling="mean")

    direct = translator.translate(matrix, info)
    np.testing.assert_allclose(direct["values"], [[2.0, 3.0], [5.0, 6.0]])
    assert direct["groups"].to_dict("list") == {"doc_id": [1, 2]}
    assert direct["counts"].tolist() == [2, 1]

    packet = InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id", "sentence_id", "token_id"),
        data={"info": info, "matrix": matrix, "columns": ["d0", "d1"]},
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    np.testing.assert_allclose(batch["data"]["values"], direct["values"])
    pd.testing.assert_frame_equal(batch["keys"], direct["groups"])
    assert batch["metadata"]["n_rows"].tolist() == direct["counts"].tolist()


def test_function_mapper_standalone_matches_teal_batch() -> None:
    translator = FunctionMapper(
        lambda packet: {
            "data": pd.DataFrame(
                {"doubled": packet["data"]["value"].to_numpy() * 2}
            ),
            "metadata": pd.DataFrame(
                {"label": packet["metadata"]["group"].astype(str).str.lower()}
            ),
        }
    )
    data = pd.DataFrame({"value": [1, 2, 3]})
    metadata = pd.DataFrame({"group": ["A", "B", "A"]})

    direct = translator.translate({"data": data, "metadata": metadata})
    assert direct["data"]["doubled"].tolist() == [2, 4, 6]
    assert direct["metadata"]["label"].tolist() == ["a", "b", "a"]

    translator._data_columns = ("value",)
    translator._metadata_columns = ("group",)
    frame = pd.DataFrame(
        {
            "doc_id": [10, 11, 12],
            "value": [1, 2, 3],
            "group": ["A", "B", "A"],
        }
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="art_table",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["data"], direct["data"])
    pd.testing.assert_frame_equal(batch["metadata"], direct["metadata"])
    assert batch["keys"]["doc_id"].tolist() == [10, 11, 12]


@pytest.mark.parametrize("sparse_embeddings", [False, True])
def test_embedding_lookup_standalone_matches_teal_batch(
    sparse_embeddings: bool,
) -> None:
    dense = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    embeddings = sparse.csr_matrix(dense) if sparse_embeddings else dense
    row_names = ["a", "b"]
    tokens = pd.DataFrame(
        {
            "doc_id": [1, 1, 1],
            "token_id": [0, 1, 2],
            "lemma": ["b", "missing", "a"],
        }
    )
    translator = EmbeddingLookup(field="lemma")

    direct = translator.translate(
        tokens,
        {
            "values": embeddings,
            "columns": ["d0", "d1"],
            "metadata": pd.DataFrame({"word": row_names}),
            "row_name_column": "word",
        },
    )
    direct_values = (
        direct["values"].toarray()
        if sparse.issparse(direct["values"])
        else direct["values"]
    )
    np.testing.assert_array_equal(direct_values, [[3, 4], [0, 0], [1, 2]])
    assert direct["feature_metadata"]["column"].tolist() == ["d0", "d1"]

    token_packet = InputBatch(
        source_label="tokens",
        artifact_id="art_tokens",
        primary_key=("doc_id", "token_id"),
        data=tokens,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    embedding_packet = InputBatch(
        source_label="embeddings",
        artifact_id="art_embeddings",
        primary_key=("word_id",),
        data={
            "info": pd.DataFrame({"_position": [0, 1]}),
            "matrix": embeddings,
            "columns": ["d0", "d1"],
            "row_names": row_names,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"tokens": token_packet, "embeddings": embedding_packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]

    batch_values = batch["data"]["values"]
    batch_values = (
        batch_values.toarray() if sparse.issparse(batch_values) else batch_values
    )
    np.testing.assert_array_equal(batch_values, direct_values)
    pd.testing.assert_frame_equal(batch["metadata"], direct["metadata"])



def test_delimiter_decomposer_standalone_matches_teal_batch() -> None:
    translator = DelimiterDecomposer(delimiter="\n", new_key="line_id")
    texts = pd.Series(["first\n\n second ", None, "third\nfourth"])

    direct = translator.translate(texts)
    assert direct["source_positions"].tolist() == [0, 0, 2, 2]
    assert direct["segment_ids"].tolist() == [0, 1, 0, 1]
    assert direct["texts"].tolist() == ["first", "second", "third", "fourth"]

    frame = pd.DataFrame({"doc_id": [10, 11, 12], "text": texts})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_text",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    assert batch["keys"].to_dict("records") == [
        {"doc_id": 10, "line_id": 0},
        {"doc_id": 10, "line_id": 1},
        {"doc_id": 12, "line_id": 0},
        {"doc_id": 12, "line_id": 1},
    ]
    assert batch["data"]["text"].tolist() == direct["texts"].tolist()


def test_matrix_transpose_standalone_matches_teal_batch() -> None:
    matrix = sparse.csr_matrix(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    features = ["alpha", "beta", "gamma"]
    row_labels = ["doc_id=10", "doc_id=11"]
    feature_metadata = pd.DataFrame(
        {
            "column_index": [0, 1, 2],
            "column": features,
            "family": ["a", "b", "b"],
        }
    )
    translator = MatrixTranspose()

    direct = translator.translate(
        matrix,
        features=features,
        row_labels=row_labels,
        feature_metadata=feature_metadata,
    )
    assert sparse.isspmatrix_csr(direct["values"])
    np.testing.assert_array_equal(
        direct["values"].toarray(),
        matrix.toarray().T,
    )
    assert direct["feature_metadata"]["column"].tolist() == row_labels
    assert direct["row_name_column"] == "feature"
    assert direct["metadata"].columns.tolist() == ["feature", "family"]
    assert direct["metadata"]["feature"].tolist() == features

    source = SimpleNamespace(
        artifact_type=ArtifactType.SPARSE_MATRIX,
        primary_key=["doc_id"],
        get_data_columns=lambda: list(features),
        get_feature_frame=lambda: feature_metadata.copy(),
        row_name=None,
    )
    translator.input_request(
        sources={"source": source},
        mode="translate",
        request=TranslationRequest(),
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="art_matrix",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": [10, 11]}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    np.testing.assert_array_equal(
        batch["data"]["values"].toarray(),
        direct["values"].toarray(),
    )
    assert batch["data"]["columns"] == direct["feature_metadata"]["column"].tolist()
    assert batch["row_name_column"] == direct["row_name_column"]
    pd.testing.assert_frame_equal(batch["metadata"], direct["metadata"])



def test_text_file_extractor_standalone_matches_teal_batch(tmp_path) -> None:
    text_path = tmp_path / "document.txt"
    text_path.write_bytes(b"alpha\r\nbeta\rgamma\n")
    missing = tmp_path / "missing.txt"
    translator = TextFileExtractor()

    direct = translator.translate([text_path, missing, None])
    assert direct["text"].tolist() == ["alpha\nbeta\ngamma\n", "", ""]
    assert direct["extraction_status"].tolist() == ["ok", "failed", "failed"]
    assert str(direct["text"].dtype) == "string"
    assert str(direct["page_count"].dtype) == "Int64"

    frame = pd.DataFrame(
        {"file_id": [1, 2, 3], "path": [str(text_path), str(missing), None]}
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="art_files",
        primary_key=("file_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["data"], direct)
    assert batch["keys"]["file_id"].tolist() == [1, 2, 3]


def test_pdf_text_extractor_standalone_matches_teal_batch_for_failures() -> None:
    pytest.importorskip("pdfplumber")
    translator = PdfTextExtractor()
    direct = translator.translate([None, None])
    assert direct["extraction_status"].tolist() == ["failed", "failed"]
    assert direct["pages_extracted"].astype(int).tolist() == [0, 0]

    frame = pd.DataFrame({"file_id": [10, 11], "path": [None, None]})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_pdfs",
        primary_key=("file_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["data"], direct)
    assert batch["keys"]["file_id"].tolist() == [10, 11]


def test_artifact_count_vectorizer_standalone_matches_teal_batch() -> None:
    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 1, 2, 2],
            "sentence_id": [0, 0, 1, 0, 0],
            "token_id": [0, 1, 0, 0, 1],
            "lemma": ["alpha", "beta", "alpha", "beta", "beta"],
        }
    )
    translator = ArtifactCountVectorizer(
        field="lemma",
        group_by=("doc_id",),
        vocabulary={"alpha": 0, "beta": 1},
    )
    source_key = ("doc_id", "sentence_id", "token_id")

    direct = translator.translate(frame, source_key=source_key)
    assert direct["groups"].to_dict("list") == {"doc_id": [1, 2]}
    assert direct["feature_metadata"]["column"].tolist() == ["alpha", "beta"]
    assert "columns" not in direct
    assert direct["values"].toarray().tolist() == [[2, 1], [0, 2]]

    packet = InputBatch(
        source_label="source",
        artifact_id="art_tokens",
        primary_key=source_key,
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["keys"], direct["groups"])
    np.testing.assert_array_equal(
        batch["data"]["values"].toarray(),
        direct["values"].toarray(),
    )
    assert batch["data"]["columns"] == direct["feature_metadata"]["column"].tolist()


def test_dictionary_translator_standalone_matches_teal_batch() -> None:
    features = ["Good", "bad", "economy", "neutral"]
    matrix = sparse.csr_matrix(
        np.asarray([[2, 1, 3, 0], [0, 2, 0, 1]], dtype=float)
    )
    dictionary = dictionaries.Dictionary(
        {
            "positive": ["good"],
            "negative": ["bad"],
            "neutral": ["neutral"],
            "economy": ["econom*"],
        },
        valuetype="glob",
        case_sensitive=False,
    )
    translator = DictionaryTranslator(dictionary)

    direct = translator.translate(matrix, features=features)
    assert direct["metadata"].to_dict("list") == {
        "matched": [6, 3],
        "unmatched": [0, 0],
        "total": [6, 3],
    }

    packet = InputBatch(
        source_label="source",
        artifact_id="art_counts",
        primary_key=("doc_id",),
        data={
            "info": pd.DataFrame({"doc_id": [10, 11]}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    np.testing.assert_array_equal(
        batch["data"]["values"].toarray(),
        direct["values"].toarray(),
    )
    assert batch["data"]["columns"] == tuple(
        direct["feature_metadata"]["column"].tolist()
    )
    pd.testing.assert_frame_equal(batch["metadata"], direct["metadata"])


def test_geco_predictor_standalone_matches_teal_batch() -> None:
    class ColumnProbability:
        def __init__(self) -> None:
            self.classes_ = np.asarray([0, 1])

        def predict_proba(self, values):
            if sparse.issparse(values):
                p = np.asarray(values[:, 0].toarray()).reshape(-1)
            else:
                p = np.asarray(values)[:, 0]
            p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
            return np.column_stack([1.0 - p, p])

    predictor = GeCoPredictor(
        [ColumnProbability(), ColumnProbability()],
        source_specs=[
            {"source_index": 0, "n_features": 1},
            {"source_index": 1, "n_features": 1},
        ],
        member_specs=[
            {"source_index": 0, "positive_class": 1},
            {"source_index": 1, "positive_class": 1},
        ],
        aggregation="mean",
    )
    first = sparse.csr_matrix([[0.2], [0.8], [0.4]])
    second = np.asarray([[0.6], [0.1], [0.9]])

    direct = predictor.translate(
        {
            "values": first,
            "feature_metadata": pd.DataFrame(
                {"column_index": [0], "column": ["x0"]}
            ),
        },
        {
            "values": second,
            "feature_metadata": pd.DataFrame(
                {"column_index": [0], "column": ["y0"]}
            ),
        },
    )
    np.testing.assert_allclose(direct["probability"], [0.4, 0.45, 0.65])
    assert direct["prediction"].tolist() == [0, 0, 1]

    packets = {
        "source_0": InputBatch(
            source_label="source_0",
            artifact_id="art_0",
            primary_key=("row_id",),
            data={
                "info": pd.DataFrame({"row_id": [1, 2, 3]}),
                "matrix": first,
            },
            batch_index=0,
            batch_count=1,
            is_first=True,
            is_last=True,
        ),
        "source_1": InputBatch(
            source_label="source_1",
            artifact_id="art_1",
            primary_key=("row_id",),
            data={
                "info": pd.DataFrame({"row_id": [1, 2, 3]}),
                "matrix": second,
            },
            batch_index=0,
            batch_count=1,
            is_first=True,
            is_last=True,
        ),
    }
    batch = predictor.translate_batch(
        packets,
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["data"], direct)


def test_word2vec_standalone_matches_teal_batch(monkeypatch) -> None:
    translator = Word2Vec(
        field="lemma",
        vector_size=3,
        min_count=1,
        epochs=2,
    )
    vectors = np.arange(9, dtype=np.float32).reshape(3, 3)
    monkeypatch.setattr(
        translator,
        "_train",
        lambda sequences: (
            ["alpha", "beta", "gamma"],
            np.asarray([3, 2, 1], dtype=np.int64),
            vectors,
            (4.0, 2.0),
            "4.4.0",
        ),
    )
    sequences = [["alpha", "beta"], ["alpha", "gamma"]]

    direct = translator.translate(sequences)
    assert direct["words"] == ["alpha", "beta", "gamma"]
    assert direct["metadata"]["word"].tolist() == ["alpha", "beta", "gamma"]
    assert direct["row_name_column"] == "word"
    assert direct["counts"].tolist() == [3, 2, 1]
    np.testing.assert_array_equal(direct["values"], vectors)
    assert direct["feature_metadata"]["column"].tolist() == ["dimension_0", "dimension_1", "dimension_2"]
    assert direct["training_loss"] == (4.0, 2.0)

    frame = pd.DataFrame(
        {
            "doc_id": [1, 1, 2, 2],
            "sentence_id": [0, 0, 0, 0],
            "token_id": [0, 1, 0, 1],
            "lemma": ["alpha", "beta", "alpha", "gamma"],
        }
    )
    packet = InputBatch(
        source_label="source",
        artifact_id="art_tokens",
        primary_key=("doc_id", "sentence_id", "token_id"),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    np.testing.assert_array_equal(batch["data"]["values"], direct["values"])
    assert batch["row_name_column"] == direct["row_name_column"]
    assert batch["metadata"]["word"].tolist() == direct["words"]
    assert batch["metadata"]["count"].tolist() == direct["counts"].tolist()


def test_spacy_standalone_matches_teal_batch(monkeypatch) -> None:
    pytest.importorskip("spacy")
    import spacy as spacy_module
    import text_analysis_lab.translators.spacy_translator as module
    from spacy.tokens import Doc

    nlp = spacy_module.blank("en")
    doc = Doc(
        nlp.vocab,
        words=["Hello", "world", "."],
        spaces=[True, False, False],
        heads=[1, 1, 1],
        deps=["dep", "ROOT", "punct"],
        sent_starts=[True, False, False],
    )

    class FakeNLP:
        def pipe(self, texts, *, batch_size, n_process):
            assert list(texts) == [doc.text]
            assert batch_size == 8
            assert n_process == 1
            yield doc

    monkeypatch.setattr(module, "_load_spacy_pipeline", lambda model, disable: FakeNLP())
    translator = SpacyTranslator(model="fake", spacy_batch_size=8)
    direct = translator.translate([doc.text])
    assert direct["sentences"]["source_position"].tolist() == [0]
    assert direct["tokens"]["token_id"].tolist() == [0, 1, 2]

    frame = pd.DataFrame({"doc_id": [7], "text": [doc.text]})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_docs",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    assert batch["sentences"]["keys"].to_dict("records") == [
        {"doc_id": 7, "sentence_id": 0}
    ]
    assert batch["tokens"]["keys"]["token_id"].tolist() == [0, 1, 2]
    pd.testing.assert_frame_equal(
        batch["tokens"]["data"],
        direct["tokens"].loc[:, list(module.TOKEN_DATA_COLUMNS)].reset_index(drop=True),
    )


def test_umap_standalone_requires_available_fitted_state() -> None:
    class FakeEstimator:
        def transform(self, matrix):
            dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
            return dense[:, :2] + 1.0

    translator = UMAP(n_components=2, reuse="stored")
    translator.source_features_ = ("a", "b", "c")
    translator._estimator = FakeEstimator()
    matrix = sparse.csr_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    direct = translator.translate(matrix)
    np.testing.assert_allclose(direct, [[2.0, 3.0], [5.0, 6.0]])

    batch = translator.translate_batch(
        {"source": _matrix_packet(matrix)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]["data"]["values"]
    np.testing.assert_allclose(batch, direct)

    recompute = UMAP(n_components=2, reuse="recompute")
    recompute._fit_completed = True
    with pytest.raises(OperatorNotFittedError, match="no fitted reducer"):
        recompute.translate(matrix)


def test_openai_responses_standalone_matches_teal_batch(monkeypatch) -> None:
    translator = OpenAIResponsesTranslator(
        model="fake-model",
        prompt="Classify: {text}",
        output_field="response",
    )

    class Usage:
        input_tokens = 5
        output_tokens = 2
        total_tokens = 7

    class Response:
        id = "resp_1"
        model = "fake-model"
        output_text = "yes"
        usage = Usage()

    monkeypatch.setattr(translator, "_request", lambda prompt: Response())
    direct = translator.translate({"text": "hello"})
    assert direct["data"]["response"].tolist() == ["yes"]
    assert direct["metadata"]["openai_total_tokens"].tolist() == [7]

    frame = pd.DataFrame({"doc_id": [1], "text": ["hello"]})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_docs",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(),
    ).outputs["output"]
    pd.testing.assert_frame_equal(batch["data"], direct["data"])
    pd.testing.assert_frame_equal(batch["metadata"], direct["metadata"])


def test_contextual_transformer_standalone_matches_teal_batch(monkeypatch) -> None:
    import text_analysis_lab.translators.contextual_transformer as module

    class FakeModel:
        pass

    class FakeTokenizer:
        pass

    translator = ContextualTransformer("fake-model")
    monkeypatch.setattr(module, "resolve_device", lambda value: "cpu")
    monkeypatch.setattr(module, "count_tokens", lambda tokenizer, texts: [4 for _ in texts])
    monkeypatch.setattr(translator, "_runtime_components", lambda *, device: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(translator, "_context_limit", lambda *, tokenizer, model: 8)
    monkeypatch.setattr(
        translator,
        "_encode_block",
        lambda *, model, tokenizer, texts, device, context_limit: [
            {
                "tokens": ["[CLS]", "tok", "[SEP]"],
                "input_ids": [101, 1001, 102],
                "offsets": [(0, 0), (0, len(text)), (0, 0)],
                "special_tokens_mask": [1, 0, 1],
                "token_type_ids": [0, 0, 0],
                "embeddings": np.asarray(
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
                ),
            }
            for text in texts
        ],
    )

    direct = translator.translate(["hello"], device="cpu", model_batch_size=2)
    assert direct["tokens"]["source_position"].tolist() == [0, 0, 0]
    assert direct["tokens"]["token_id"].tolist() == [0, 1, 2]
    assert direct["values"].shape == (3, 2)
    assert direct["feature_metadata"]["column"].tolist() == ["dim_0", "dim_1"]

    frame = pd.DataFrame({"doc_id": [9], "text": ["hello"]})
    packet = InputBatch(
        source_label="source",
        artifact_id="art_docs",
        primary_key=("doc_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )
    batch = translator.translate_batch(
        {"source": packet},
        mode="translate",
        request=TranslationRequest(params={"device": "cpu", "model_batch_size": 2}),
    ).outputs
    assert batch["tokens"]["keys"].to_dict("records") == [
        {"doc_id": 9, "token_id": 0},
        {"doc_id": 9, "token_id": 1},
        {"doc_id": 9, "token_id": 2},
    ]
    np.testing.assert_array_equal(
        batch["contextual_embeddings"]["data"]["values"],
        direct["values"],
    )


def test_coreference_standalone_matches_teal_batch(monkeypatch) -> None:
    import text_analysis_lab.translators.coreference_resolver as module
    from text_analysis_lab.linguistics.coreference.runtime import (
        CorefBatchPrediction,
        CorefMention,
        CorefPrediction,
    )

    class FakeRuntime:
        def document_token_counts(self, texts):
            return [len(texts[0].split())], None

        def predict_texts(self, texts, **kwargs):
            _ = kwargs
            return CorefBatchPrediction(
                model="lingmess",
                model_repository="fake",
                device="cpu",
                elapsed_seconds=0.0,
                documents=(
                    CorefPrediction(
                        0,
                        texts[0],
                        (
                            CorefMention(0, 0, 0, 5, "Alice"),
                            CorefMention(0, 1, 12, 15, "She"),
                        ),
                    ),
                ),
            )

    monkeypatch.setattr(module, "_make_runtime", lambda **kwargs: FakeRuntime())
    documents = pd.DataFrame(
        {"row_id": [7], "text": ["Alice left. She slept."]}
    )
    tokens = pd.DataFrame(
        {
            "row_id": [7, 7, 7, 7, 7, 7],
            "sentence_id": [0, 0, 0, 1, 1, 1],
            "token_id": [0, 1, 2, 0, 1, 2],
            "text": ["Alice", "left", ".", "She", "slept", "."],
            "lemma": ["Alice", "leave", ".", "she", "sleep", "."],
            "pos": ["PROPN", "VERB", "PUNCT", "PRON", "VERB", "PUNCT"],
            "dep": ["nsubj", "ROOT", "punct", "nsubj", "ROOT", "punct"],
            "head_token_id": [1, 1, 1, 1, 1, 1],
            "ent_type": ["PERSON", "", "", "", "", ""],
            "char_start": [0, 6, 10, 12, 16, 21],
            "char_end": [5, 10, 11, 15, 21, 22],
        }
    )
    translator = CoreferenceResolver(device="cpu")
    direct = translator.translate(
        documents,
        tokens,
        document_keys=["row_id"],
    )
    assert direct["mentions"]["text"].tolist() == ["Alice", "She"]
    assert direct["mentions"]["cluster_id"].tolist() == [0, 0]

    batch = translator.translate_batch(
        {
            "documents": InputBatch(
                "documents",
                "art_documents",
                ("row_id",),
                documents,
                0,
                1,
                True,
                True,
            ),
            "tokens": InputBatch(
                "tokens",
                "art_tokens",
                ("row_id", "sentence_id", "token_id"),
                tokens,
                0,
                1,
                True,
                True,
            ),
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    batch_mentions = pd.concat(
        [batch["mentions"]["keys"], batch["mentions"]["data"]], axis=1
    )
    pd.testing.assert_frame_equal(batch_mentions, direct["mentions"])


def test_semantic_role_labeler_standalone_matches_teal_batch(monkeypatch) -> None:
    import text_analysis_lab.translators.semantic_role_labeler as module
    from text_analysis_lab.linguistics.srl.runtime import SrlTokenPrediction
    from text_analysis_lab.linguistics.srl.structures import BioSpan

    class FakeRuntime:
        def encode_tokens(self, tokens):
            return tuple(tokens)

        def predict_encoded_batch(self, instances):
            tokens = tuple(instances[0][0])
            return (
                SrlTokenPrediction(
                    tokens=tokens,
                    predicate_index=1,
                    predicate="runs",
                    wordpieces=tokens,
                    input_ids=(1, 2, 3, 4),
                    predicate_indicator=(0, 1, 0, 0),
                    wordpiece_offsets=(0, 1, 2, 3),
                    wordpiece_tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    raw_tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    tags=("B-ARG0", "B-V", "B-ARGM-MNR", "O"),
                    bio_repairs=(),
                    word_scores=(0.9, 0.95, 0.8, 0.99),
                    spans=(
                        BioSpan("ARG0", 0, 1),
                        BioSpan("V", 1, 2),
                        BioSpan("ARGM-MNR", 2, 3),
                    ),
                    description="",
                ),
            )

    monkeypatch.setattr(module, "_make_runtime", lambda **kwargs: FakeRuntime())
    sentences = pd.DataFrame(
        {
            "row_id": [2],
            "sentence_id": [0],
            "text": ["Alice runs quickly."],
            "char_start": [0],
            "char_end": [19],
        }
    )
    tokens = pd.DataFrame(
        {
            "row_id": [2, 2, 2, 2],
            "sentence_id": [0, 0, 0, 0],
            "token_id": [0, 1, 2, 3],
            "text": ["Alice", "runs", "quickly", "."],
            "lemma": ["Alice", "run", "quickly", "."],
            "pos": ["PROPN", "VERB", "ADV", "PUNCT"],
            "tag": ["NNP", "VBZ", "RB", "."],
            "dep": ["nsubj", "ROOT", "advmod", "punct"],
            "head_token_id": [1, 1, 1, 1],
            "ent_type": ["PERSON", "", "", ""],
            "char_start": [0, 6, 11, 18],
            "char_end": [5, 10, 18, 19],
        }
    )
    translator = SemanticRoleLabeler(device="cpu")
    direct = translator.translate(
        sentences,
        tokens,
        sentence_keys=["row_id", "sentence_id"],
    )
    assert direct["predicates"]["text"].tolist() == ["runs"]
    assert direct["roles"]["role"].tolist() == ["ARG0", "V", "ARGM-MNR"]

    batch = translator.translate_batch(
        {
            "sentences": InputBatch(
                "sentences",
                "art_sentences",
                ("row_id", "sentence_id"),
                sentences,
                0,
                1,
                True,
                True,
            ),
            "tokens": InputBatch(
                "tokens",
                "art_tokens",
                ("row_id", "sentence_id", "token_id"),
                tokens,
                0,
                1,
                True,
                True,
            ),
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    for label in ("predicates", "roles", "failures"):
        combined = pd.concat(
            [batch[label]["keys"], batch[label]["data"]],
            axis=1,
        )
        pd.testing.assert_frame_equal(combined, direct[label])


def test_word_sense_disambiguator_standalone_matches_teal_batch_without_targets() -> None:
    tokens = pd.DataFrame(
        {
            "row_id": [1, 1],
            "sentence_id": [0, 0],
            "token_id": [0, 1],
            "text": [".", "!"],
            "lemma": [".", "!"],
            "pos": ["PUNCT", "PUNCT"],
            "ent_type": ["", ""],
        }
    )
    translator = WordSenseDisambiguator(
        device="cpu",
    )
    token_keys = ["row_id", "sentence_id", "token_id"]
    direct = translator.translate(tokens, token_keys=token_keys)
    assert direct["senses"].empty
    assert direct["candidates"].empty
    assert direct["unresolved"].empty

    batch = translator.translate_batch(
        {
            "tokens": InputBatch(
                "tokens",
                "art_tokens",
                tuple(token_keys),
                tokens,
                0,
                1,
                True,
                True,
            )
        },
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    for label in ("senses", "candidates", "unresolved"):
        combined = pd.concat(
            [batch[label]["keys"], batch[label]["data"]],
            axis=1,
        )
        pd.testing.assert_frame_equal(combined, direct[label])


def test_srl_translate_from_text_composes_spacy_and_structured_translate(
    monkeypatch,
) -> None:
    parsed = {
        "sentences": pd.DataFrame(
            {
                "source_position": [0],
                "sentence_id": [0],
                "text": ["Alice runs."],
                "char_start": [0],
                "char_end": [11],
            }
        ),
        "tokens": pd.DataFrame({"source_position": [0], "sentence_id": [0], "token_id": [0]}),
    }
    parser_calls = []

    def fake_parse(self, texts):
        parser_calls.append((self.model, self.spacy_batch_size, tuple(self.disable), texts))
        return parsed

    monkeypatch.setattr(SpacyTranslator, "translate", fake_parse)
    translator = SemanticRoleLabeler(sentence_key="sentence_id", token_key="token_id")
    structured_calls = []

    def fake_translate(sentences, tokens, *, sentence_keys):
        structured_calls.append((sentences, tokens, sentence_keys))
        return {"predicates": pd.DataFrame(), "roles": pd.DataFrame(), "failures": pd.DataFrame()}

    monkeypatch.setattr(translator, "translate", fake_translate)
    result = translator.translate_from_text(
        ["Alice runs."],
        spacy_model="fake-spacy",
        spacy_batch_size=7,
        spacy_disable=("ner",),
    )

    assert parser_calls == [("fake-spacy", 7, ("ner",), ["Alice runs."])]
    assert structured_calls[0][0] is parsed["sentences"]
    assert structured_calls[0][1] is parsed["tokens"]
    assert structured_calls[0][2] == ["source_position", "sentence_id"]
    assert set(result) == {"predicates", "roles", "failures"}


def test_wsd_translate_from_text_composes_spacy_and_structured_translate(
    monkeypatch,
) -> None:
    parsed = {
        "sentences": pd.DataFrame(),
        "tokens": pd.DataFrame(
            {
                "source_position": [0],
                "sentence_id": [0],
                "token_id": [0],
                "text": ["bank"],
            }
        ),
    }

    monkeypatch.setattr(SpacyTranslator, "translate", lambda self, texts: parsed)
    translator = WordSenseDisambiguator(sentence_key="sentence_id", token_key="token_id")
    seen = {}

    def fake_translate(tokens, *, token_keys):
        seen["tokens"] = tokens
        seen["token_keys"] = token_keys
        return {"senses": pd.DataFrame(), "candidates": pd.DataFrame(), "unresolved": pd.DataFrame()}

    monkeypatch.setattr(translator, "translate", fake_translate)
    result = translator.translate_from_text("The bank approved the loan.")

    assert seen["tokens"] is parsed["tokens"]
    assert seen["token_keys"] == ["source_position", "sentence_id", "token_id"]
    assert set(result) == {"senses", "candidates", "unresolved"}


def test_coreference_translate_from_text_composes_documents_spacy_and_translate(
    monkeypatch,
) -> None:
    parsed = {
        "sentences": pd.DataFrame(),
        "tokens": pd.DataFrame(
            {
                "source_position": [0, 1],
                "sentence_id": [0, 0],
                "token_id": [0, 0],
            }
        ),
    }
    monkeypatch.setattr(SpacyTranslator, "translate", lambda self, texts: parsed)
    translator = CoreferenceResolver(text_field="body")
    seen = {}

    def fake_translate(documents, tokens, *, document_keys):
        seen["documents"] = documents
        seen["tokens"] = tokens
        seen["document_keys"] = document_keys
        return {"mentions": pd.DataFrame(), "failures": pd.DataFrame()}

    monkeypatch.setattr(translator, "translate", fake_translate)
    result = translator.translate_from_text(["Alice left.", "She returned."])

    assert seen["documents"].to_dict("list") == {
        "source_position": [0, 1],
        "body": ["Alice left.", "She returned."],
    }
    assert seen["tokens"] is parsed["tokens"]
    assert seen["document_keys"] == ["source_position"]
    assert set(result) == {"mentions", "failures"}


