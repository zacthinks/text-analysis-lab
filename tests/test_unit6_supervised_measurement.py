from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from text_analysis_lab.analysis.classification import classification
from text_analysis_lab.core.errors import ArtifactError, OperatorError
from text_analysis_lab.core.operator import BaseOperator, InputBatch, TranslationRequest
from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator
from text_analysis_lab.core.types import ArtifactType
from text_analysis_lab.translators import FittedPredictor, FunctionMapper


def _source(
    kind: str = "table",
    columns: tuple[str, ...] = ("x", "z"),
    metadata: tuple[str, ...] = (),
):
    def query_columns(*, metadata_mode="none"):
        metadata_cols = list(metadata) if metadata_mode != "none" else []
        return {
            "columns": [
                *[
                    {
                        "namespace": "key", "base_name": "row_id",
                        "qualified_name": "key.row_id", "output_name": "row_id",
                        "source_artifact_id": "art_source",
                    }
                ],
                *[
                    {
                        "namespace": "data", "base_name": c,
                        "qualified_name": f"data.{c}", "output_name": c,
                        "source_artifact_id": "art_source",
                    }
                    for c in columns
                ],
                *[
                    {
                        "namespace": "metadata", "base_name": c,
                        "qualified_name": f"metadata.art_source.{c}", "output_name": c,
                        "source_artifact_id": "art_source",
                    }
                    for c in metadata_cols
                ],
            ]
        }

    return SimpleNamespace(
        artifact_type=ArtifactType(kind),
        primary_key=["row_id"],
        get_data_columns=lambda: list(columns),
        query_columns=query_columns,
    )


def _table_packet(frame: pd.DataFrame, *, label: str = "source") -> InputBatch:
    return InputBatch(
        source_label=label,
        artifact_id=f"art_{label}",
        primary_key=("row_id",),
        data=frame,
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def _matrix_packet(matrix, *, label: str = "source") -> InputBatch:
    # FittedPredictor still consumes the ordinary native matrix packet.
    return InputBatch(
        source_label=label,
        artifact_id=f"art_{label}",
        primary_key=("row_id",),
        data={
            "info": pd.DataFrame({"row_id": np.arange(matrix.shape[0], dtype=int)}),
            "matrix": matrix,
        },
        batch_index=0,
        batch_count=1,
        is_first=True,
        is_last=True,
    )


def test_function_mapper_packet_contract_data_and_metadata() -> None:
    seen: list[tuple[list[str], list[str]]] = []

    def make_h(packet):
        data = packet["data"]
        metadata = packet["metadata"]
        seen.append((list(data.columns), list(metadata.columns)))
        return {
            "data": pd.DataFrame(
                {"H": ((data["x"] > 0) | (data["z"] > 0)).astype(int)}
            ),
            "metadata": pd.DataFrame(
                {"group_upper": metadata["group"].str.upper()}
            ),
            # FunctionMapper consumes only data/metadata packet components;
            # unrelated callable metadata is ignored.
            "notes": {"source": "unit-test"},
        }

    mapper = FunctionMapper(make_h)
    source = _source(metadata=("group",))
    params = mapper.validate_operation_params(
        {
            "data_columns": ["x", "z"],
            "metadata_columns": ["group"],
            "metadata_mode": "local",
        },
        sources={"source": source},
        mode="translate",
    )
    request = TranslationRequest(params=params)
    input_request = mapper.input_request(
        sources={"source": source}, mode="translate", request=request
    )
    assert input_request.columns.keys is True
    assert input_request.columns.data == ["x", "z"]
    assert input_request.columns.metadata == ["group"]
    assert input_request.metadata_mode == "local"
    assert input_request.form == "table"

    frame = pd.DataFrame(
        {
            "row_id": [5, 7, 9],
            "x": [0, 1, 0],
            "z": [0, 0, 2],
            "group": ["a", "b", "a"],
        }
    )
    result = mapper.translate_batch(
        {"source": _table_packet(frame)}, mode="translate", request=request
    ).outputs["output"]
    assert seen == [(["x", "z"], ["group"])]
    assert result["keys"].to_dict("list") == {"row_id": [5, 7, 9]}
    assert result["data"].to_dict("list") == {"H": [0, 1, 1]}
    assert result["metadata"].to_dict("list") == {"group_upper": ["A", "B", "A"]}


def test_function_mapper_matrix_columns_resolve_from_matrix_schema_not_query_view() -> None:
    """Matrix features live outside the relational query column inventory."""
    source = _source("sparse_matrix", ("qualitative", "quantitative"))

    def query_columns(*, metadata_mode="none"):
        _ = metadata_mode
        return {
            "columns": [
                {
                    "namespace": "key",
                    "base_name": "row_id",
                    "qualified_name": "key.row_id",
                    "output_name": "row_id",
                    "source_artifact_id": "art_source",
                }
            ]
        }

    source.query_columns = query_columns
    mapper = FunctionMapper(
        lambda packet: {
            "data": pd.DataFrame(
                {
                    "qual": (packet["data"]["qualitative"] > 0).astype("int8"),
                    "quant": (packet["data"]["quantitative"] > 0).astype("int8"),
                }
            )
        }
    )
    params = mapper.validate_operation_params(
        {"data_columns": ["qualitative", "quantitative"]},
        sources={"source": source},
        mode="translate",
    )
    request = TranslationRequest(params=params)
    input_request = mapper.input_request(
        sources={"source": source}, mode="translate", request=request
    )
    assert mapper._data_columns == ("qualitative", "quantitative")
    assert input_request.columns.data == ["qualitative", "quantitative"]


def test_function_mapper_sparse_matrix_dataframe_packet_and_snapshot(tmp_path) -> None:
    matrix = sparse.csr_matrix([[0, 1], [0, 0], [2, 0]], dtype=float)
    mapper = FunctionMapper(
        lambda packet: {
            "data": pd.DataFrame(
                {
                    "H": (
                        packet["data"].sparse.to_coo().tocsr().sum(axis=1).A1 > 0
                    ).astype(int)
                }
            )
        }
    )
    source = _source("sparse_matrix", ("a", "b"))
    params = mapper.validate_operation_params(
        {"data_columns": ["a", "b"]}, sources={"source": source}, mode="translate"
    )
    request = TranslationRequest(params=params)
    mapper.input_request(sources={"source": source}, mode="translate", request=request)
    sparse_frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=["a", "b"])
    sparse_frame.insert(0, "row_id", [0, 1, 2])
    output = mapper.translate_batch(
        {"source": _table_packet(sparse_frame)}, mode="translate", request=request
    ).outputs["output"]
    assert output["data"]["H"].tolist() == [1, 0, 1]

    snapshot = tmp_path / "mapper"
    mapper.save_to_dir(snapshot, operator_id="optr_mapper")
    restored = BaseOperator.load_from_dir(snapshot)
    assert isinstance(restored, FunctionMapper)
    restored._data_columns = ("a", "b")
    smaller = pd.DataFrame.sparse.from_spmatrix(matrix[:2], columns=["a", "b"])
    smaller.insert(0, "row_id", [0, 1])
    result = restored.translate_batch(
        {"source": _table_packet(smaller)}, mode="translate", request=TranslationRequest()
    )
    assert result.outputs["output"]["data"]["H"].tolist() == [1, 0]


def test_function_mapper_rejects_row_change_and_key_authority() -> None:
    source = _source()
    frame = pd.DataFrame({"row_id": [0, 1], "x": [1, 2], "z": [0, 0]})

    wrong = FunctionMapper(lambda packet: {"data": pd.DataFrame({"value": [1]})})
    params = wrong.validate_operation_params(
        {"data_columns": True}, sources={"source": source}, mode="translate"
    )
    request = TranslationRequest(params=params)
    wrong.input_request(sources={"source": source}, mode="translate", request=request)
    with pytest.raises(ArtifactError, match="returned 1 data rows"):
        wrong.translate_batch({"source": _table_packet(frame)}, mode="translate", request=request)

    keys = FunctionMapper(
        lambda packet: {
            "keys": pd.DataFrame({"row_id": [10, 11]}),
            "data": pd.DataFrame({"value": [1, 2]}),
        }
    )
    params = keys.validate_operation_params(
        {"data_columns": True}, sources={"source": source}, mode="translate"
    )
    request = TranslationRequest(params=params)
    keys.input_request(sources={"source": source}, mode="translate", request=request)
    with pytest.raises(ArtifactError, match="may not return 'keys'"):
        keys.translate_batch({"source": _table_packet(frame)}, mode="translate", request=request)

    non_packet = FunctionMapper(lambda packet: pd.DataFrame({"value": [1, 2]}))
    params = non_packet.validate_operation_params(
        {"data_columns": True}, sources={"source": source}, mode="translate"
    )
    request = TranslationRequest(params=params)
    non_packet.input_request(sources={"source": source}, mode="translate", request=request)
    with pytest.raises(ArtifactError, match="must return a mapping"):
        non_packet.translate_batch(
            {"source": _table_packet(frame)}, mode="translate", request=request
        )


def test_fitted_predictor_sparse_prediction_probability_and_snapshot(tmp_path) -> None:
    X = sparse.csr_matrix(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(random_state=0).fit(X, y)
    predictor = FittedPredictor(model, probability_class=1)
    source = _source("sparse_matrix", ("a", "b"))
    predictor.input_request(sources={"source": source}, mode="translate", request=TranslationRequest())
    output = predictor.translate_batch(
        {"source": _matrix_packet(X)}, mode="translate", request=TranslationRequest()
    ).outputs["output"]
    assert output["keys"]["row_id"].tolist() == list(range(6))
    assert output["data"]["prediction"].tolist() == model.predict(X).tolist()
    assert output["data"]["probability"].tolist() == pytest.approx(
        model.predict_proba(X)[:, 1].tolist()
    )

    snapshot = tmp_path / "predictor"
    predictor.save_to_dir(snapshot, operator_id="optr_predictor")
    restored = BaseOperator.load_from_dir(snapshot)
    assert isinstance(restored, FittedPredictor)
    restored._source_type = "sparse_matrix"
    restored_output = restored.translate_batch(
        {"source": _matrix_packet(X[:2])}, mode="translate", request=TranslationRequest()
    ).outputs["output"]["data"]
    assert restored_output["prediction"].tolist() == model.predict(X[:2]).tolist()
    assert restored_output["probability"].tolist() == pytest.approx(
        model.predict_proba(X[:2])[:, 1].tolist()
    )


def test_fitted_predictor_dense_and_probability_unsupported_failure() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0).fit(X, y)
    predictor = FittedPredictor(model)
    source = _source("dense_matrix", ("x",))
    predictor.input_request(sources={"source": source}, mode="translate", request=TranslationRequest())
    output = predictor.translate_batch(
        {"source": _matrix_packet(X)}, mode="translate", request=TranslationRequest()
    ).outputs["output"]["data"]
    assert list(output.columns) == ["prediction"]

    svc = LinearSVC().fit(X, y)
    unsupported = FittedPredictor(svc, probability_class=1)
    unsupported.input_request(
        sources={"source": source}, mode="translate", request=TranslationRequest()
    )
    with pytest.raises(OperatorError, match="does not expose predict_proba"):
        unsupported.translate_batch(
            {"source": _matrix_packet(X)}, mode="translate", request=TranslationRequest()
        )


def _prob_documents(n: int = 10) -> InputBatch:
    # Deliberately scrambled physical rows: _position defines canonical source order.
    order = [4, 0, 9, 2, 6, 1, 8, 3, 7, 5][:n]
    frame = pd.DataFrame({"row_id": order, "_position": order})
    return _table_packet(frame, label="documents")


def _prob_strata(values: dict[int, int]) -> InputBatch:
    keys = list(reversed(list(values)))
    frame = pd.DataFrame({"row_id": keys, "H": [values[key] for key in keys]})
    return _table_packet(frame, label="strata")


def test_probability_split_srs_exact_n_disjoint_exhaustive_and_constant_pi() -> None:
    translator = ProbabilitySplitTranslator(n=4, random_state=12)
    result = translator.translate_batch(
        {"documents": _prob_documents()}, mode="translate", request=TranslationRequest()
    ).outputs
    sample = result["sample"]["keys"]["row_id"].tolist()
    remainder = result["remainder"]["keys"]["row_id"].tolist()
    assert len(sample) == 4
    assert set(sample).isdisjoint(remainder)
    assert set(sample) | set(remainder) == set(range(10))
    assert result["pi"]["keys"]["row_id"].tolist() == sample
    assert result["pi"]["data"]["pi"].tolist() == pytest.approx([0.4] * 4)

    again = ProbabilitySplitTranslator(n=4, random_state=12).translate_batch(
        {"documents": _prob_documents()}, mode="translate", request=TranslationRequest()
    ).outputs
    assert again["sample"]["keys"]["row_id"].tolist() == sample


def test_probability_split_disproportionate_alignment_and_realized_pi() -> None:
    strata = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    translator = ProbabilitySplitTranslator(
        n=4, allocation={0: 1, 1: 1}, random_state=3
    )
    result = translator.translate_batch(
        {"documents": _prob_documents(), "strata": _prob_strata(strata)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    sample = result["sample"]["keys"]["row_id"].tolist()
    sampled_h = [strata[key] for key in sample]
    assert sampled_h.count(0) == 2
    assert sampled_h.count(1) == 2
    expected_pi = [2 / 6 if strata[key] == 0 else 2 / 4 for key in sample]
    assert result["pi"]["data"]["pi"].tolist() == pytest.approx(expected_pi)

    # Relative weights rather than literal counts.
    result_scaled = ProbabilitySplitTranslator(
        n=4, allocation={0: 20, 1: 20}, random_state=3
    ).translate_batch(
        {"documents": _prob_documents(), "strata": _prob_strata(strata)},
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    assert result_scaled["sample"]["keys"]["row_id"].tolist() == sample


def test_probability_split_strata_contract_failures() -> None:
    strata = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1}
    documents = _prob_documents()

    missing_frame = pd.DataFrame({"row_id": list(range(9)), "H": [strata[i] for i in range(9)]})
    with pytest.raises(ArtifactError, match="key set must exactly equal"):
        ProbabilitySplitTranslator(n=4, allocation={0: 1, 1: 1}).translate_batch(
            {"documents": documents, "strata": _table_packet(missing_frame, label="strata")},
            mode="translate",
            request=TranslationRequest(),
        )

    duplicate_frame = pd.DataFrame({"row_id": [*range(9), 8], "H": [0] * 10})
    with pytest.raises(ArtifactError, match="duplicate"):
        ProbabilitySplitTranslator(n=4, allocation={0: 1}).translate_batch(
            {"documents": documents, "strata": _table_packet(duplicate_frame, label="strata")},
            mode="translate",
            request=TranslationRequest(),
        )

    with pytest.raises(ValueError, match="missing observed"):
        ProbabilitySplitTranslator(n=4, allocation={0: 1}).translate_batch(
            {"documents": documents, "strata": _prob_strata(strata)},
            mode="translate",
            request=TranslationRequest(),
        )
    with pytest.raises(ValueError, match="unknown"):
        ProbabilitySplitTranslator(n=4, allocation={0: 1, 1: 1, 2: 1}).translate_batch(
            {"documents": documents, "strata": _prob_strata(strata)},
            mode="translate",
            request=TranslationRequest(),
        )
    with pytest.raises(ValueError, match="only N_h"):
        ProbabilitySplitTranslator(n=8, allocation={0: 1, 1: 9}).translate_batch(
            {"documents": documents, "strata": _prob_strata(strata)},
            mode="translate",
            request=TranslationRequest(),
        )


def test_probability_split_specs_make_pi_depend_on_sample_output() -> None:
    source = _source("table", ("text",))
    strata = _source("table", ("H",))
    translator = ProbabilitySplitTranslator(n=2, allocation={0: 1, 1: 1})
    specs = translator.output_specs(
        sources={"documents": source, "strata": strata}, request=TranslationRequest()
    )
    assert specs["remainder"].basis_labels == ("documents",)
    assert specs["sample"].basis_labels == ("documents",)
    assert specs["pi"].basis_labels == ("sample",)
    assert specs["pi"].artifact_type == ArtifactType.TABLE


class _FakeProject:
    def get_artifact(self, value):
        return value


class _FakeArtifact:
    def __init__(self, frame: pd.DataFrame, data_columns: list[str], project: _FakeProject):
        self._frame = frame.copy()
        self._data_columns = list(data_columns)
        self.project = project
        self.primary_key = ["row_id"]

    def get_data_columns(self):
        return list(self._data_columns)

    def query(self, **kwargs):
        fields = kwargs.get("data_columns")
        selected = ["row_id", *fields]
        return self._frame.loc[:, selected].copy()


def test_classification_aligns_by_key_and_computes_design_weighted_metrics() -> None:
    project = _FakeProject()
    predictions = _FakeArtifact(
        pd.DataFrame({"row_id": [4, 1, 3, 2], "prediction": [1, 1, 0, 0]}),
        ["prediction"],
        project,
    )
    gold = _FakeArtifact(
        pd.DataFrame({"row_id": [3, 1, 4], "label": [1, 1, 0]}),
        ["label"],
        project,
    )
    pi = _FakeArtifact(
        pd.DataFrame({"row_id": [1, 4, 3], "pi": [1.0, 0.5, 0.25]}),
        ["pi"],
        project,
    )
    result = classification(predictions, gold=gold, pi=pi)
    # row 3: FN, row 1: TP, row 4: FP
    assert result.raw_confusion.loc[0, 1] == 1
    assert result.raw_confusion.loc[1, 0] == 1
    assert result.raw_confusion.loc[1, 1] == 1
    assert result.raw_metrics["accuracy"] == pytest.approx(1 / 3)
    assert result.raw_metrics["precision"] == pytest.approx(1 / 2)
    assert result.raw_metrics["recall"] == pytest.approx(1 / 2)
    assert result.raw_metrics["prevalence"] == pytest.approx(2 / 3)

    weighted = result.confusion_matrix(weighted=True)
    assert weighted.loc[0, 1] == pytest.approx(2.0)  # pi=.5 for row 4
    assert weighted.loc[1, 0] == pytest.approx(4.0)  # pi=.25 for row 3
    assert weighted.loc[1, 1] == pytest.approx(1.0)
    assert result.weighted_metrics["accuracy"] == pytest.approx(1 / 7)
    assert result.weighted_metrics["prevalence"] == pytest.approx(5 / 7)
    tidy = result.to_frame()
    assert set(tidy["scope"]) == {"raw_audit", "design_weighted"}
    assert set(tidy["metric"]) == {
        "accuracy", "precision", "recall", "specificity", "f1", "prevalence"
    }


def test_classification_accepts_named_focus_coder_fields() -> None:
    project = _FakeProject()
    predictions = _FakeArtifact(
        pd.DataFrame({"row_id": [0, 1, 2, 3], "qual_prediction": [0, 1, 1, 0]}),
        ["qual_prediction"],
        project,
    )
    focus_labels = _FakeArtifact(
        pd.DataFrame(
            {
                "row_id": [3, 1, 0, 2],
                "qualitative": [1, 1, 0, 0],
                "quantitative": [0, 1, 0, 1],
            }
        ),
        ["qualitative", "quantitative"],
        project,
    )
    pi = _FakeArtifact(
        pd.DataFrame({"row_id": [2, 0, 3, 1], "audit_probability": [0.25] * 4}),
        ["audit_probability"],
        project,
    )

    result = classification(
        predictions,
        gold=focus_labels,
        pi=pi,
        prediction_field="qual_prediction",
        gold_field="qualitative",
        pi_field="audit_probability",
    )

    assert result.raw_confusion.loc[0, 0] == 1
    assert result.raw_confusion.loc[0, 1] == 1
    assert result.raw_confusion.loc[1, 0] == 1
    assert result.raw_confusion.loc[1, 1] == 1
    assert result.weighted_metrics == pytest.approx(result.raw_metrics)


def test_classification_constant_pi_matches_raw_and_key_contracts_fail() -> None:
    project = _FakeProject()
    predictions = _FakeArtifact(
        pd.DataFrame({"row_id": [0, 1, 2, 3], "prediction": [0, 1, 1, 0]}),
        ["prediction"], project,
    )
    gold = _FakeArtifact(
        pd.DataFrame({"row_id": [3, 1, 0, 2], "label": [1, 1, 0, 0]}),
        ["label"], project,
    )
    pi = _FakeArtifact(
        pd.DataFrame({"row_id": [2, 0, 3, 1], "pi": [0.25] * 4}),
        ["pi"], project,
    )
    result = classification(predictions, gold=gold, pi=pi)
    assert result.weighted_metrics == pytest.approx(result.raw_metrics)

    missing_pi = _FakeArtifact(pd.DataFrame({"row_id": [0, 1, 2], "pi": [0.5] * 3}), ["pi"], project)
    with pytest.raises(ArtifactError, match="key set must exactly equal"):
        classification(predictions, gold=gold, pi=missing_pi)

    duplicate_gold = _FakeArtifact(
        pd.DataFrame({"row_id": [0, 0], "label": [0, 1]}), ["label"], project
    )
    with pytest.raises(ArtifactError, match="duplicate"):
        classification(predictions, gold=duplicate_gold)

class _FakeCodingArtifact:
    def __init__(self, *, artifact_id: str, frame: pd.DataFrame, data_columns: list[str], project):
        self.artifact_id = artifact_id
        self._frame = frame.copy().reset_index(drop=True)
        self._data_columns = list(data_columns)
        self.project = project
        self.primary_key = ["row_id"]
        self.n_rows = len(frame)

    def require_complete(self):
        return None

    def get_data_columns(self):
        return list(self._data_columns)

    def position_by_key(self, key):
        key_value = key[0] if isinstance(key, tuple) else key
        matches = self._frame.index[self._frame["row_id"] == key_value].tolist()
        if not matches:
            raise KeyError(key_value)
        return int(matches[0])

    def query(self, **kwargs):
        frame = self._frame.copy()
        if "_position" not in frame.columns:
            frame.insert(0, "_position", np.arange(len(frame), dtype=int))
        positions = kwargs.get("positions")
        if positions is not None:
            frame = frame[frame["_position"].isin(positions)].copy()
        columns: list[str] = []
        if kwargs.get("include_position"):
            columns.append("_position")
        if kwargs.get("key_columns") is True:
            columns.extend(["row_id"])
        data_columns = kwargs.get("data_columns")
        if isinstance(data_columns, list):
            columns.extend(data_columns)
        # Preserve order while removing accidental duplicates.
        columns = list(dict.fromkeys(columns))
        return frame.loc[:, columns].reset_index(drop=True)


class _FakeCodingProject:
    def __init__(self, root):
        self.storage = SimpleNamespace(teal_dir=root)
        self.returned_frames: list[pd.DataFrame] = []
        self.result = object()
        self.artifacts: dict[str, _FakeCodingArtifact] = {}

    def get_artifact(self, value):
        if isinstance(value, str):
            return self.artifacts[value]
        return value

    def from_keyed_frame(self, source, frame, **kwargs):
        self.returned_frames.append(frame.copy())
        assert kwargs["data_fields"] == ["label"]
        assert kwargs["require_complete"] is True
        return self.result


def test_binary_code_resumes_after_interrupt_and_commits_complete_labels(tmp_path, monkeypatch) -> None:
    from text_analysis_lab.core.binary_code import binary_code

    project = _FakeCodingProject(tmp_path / ".teal")
    text = _FakeCodingArtifact(
        artifact_id="art_F",
        frame=pd.DataFrame(
            {"row_id": [0, 1, 2, 3], "text": ["zero", "one", "two", "three"]}
        ),
        data_columns=["text"],
        project=project,
    )
    # Audit has its own ordering, independent of the text-source position.
    audit = _FakeCodingArtifact(
        artifact_id="art_A",
        frame=pd.DataFrame({"row_id": [3, 1], "_position": [0, 1]}),
        data_columns=[],
        project=project,
    )
    project.artifacts = {"A": audit, "F": text}

    first_answers = iter(["1"])

    def first_input(prompt):
        try:
            return next(first_answers)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", first_input)
    with pytest.raises(KeyboardInterrupt):
        binary_code(
            project,
            "A",
            text_source="F",
            text_field="text",
            instructions="Mark the construct.",
            context_before=1,
            context_after=1,
        )
    state_files = list((project.storage.teal_dir / "coding").glob("*.sqlite"))
    assert len(state_files) == 1
    assert project.returned_frames == []

    # Only the still-uncoded second audit row prompts after resume.
    monkeypatch.setattr("builtins.input", lambda prompt: "0")
    result = binary_code(
        project,
        "A",
        text_source="F",
        text_field="text",
        instructions="Mark the construct.",
        context_before=1,
        context_after=1,
    )
    assert result is project.result
    assert len(project.returned_frames) == 1
    committed = project.returned_frames[0]
    assert committed.to_dict("list") == {"row_id": [3, 1], "label": [1, 0]}
    assert list((project.storage.teal_dir / "coding").glob("*.sqlite")) == []


def test_probability_split_numeric_strata_match_bool_int_and_integral_float() -> None:
    documents = _prob_documents()
    bool_frame = pd.DataFrame(
        {"row_id": list(reversed(range(10))), "H": [key >= 6 for key in reversed(range(10))]}
    )
    result = ProbabilitySplitTranslator(
        n=4, allocation={0: 1, 1: 1}, random_state=5
    ).translate_batch(
        {"documents": documents, "strata": _table_packet(bool_frame, label="strata")},
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    assert len(result["sample"]["keys"]) == 4

    float_frame = bool_frame.copy()
    float_frame["H"] = float_frame["H"].astype(float)
    result_float = ProbabilitySplitTranslator(
        n=4, allocation={False: 1, True: 1}, random_state=5
    ).translate_batch(
        {"documents": documents, "strata": _table_packet(float_frame, label="strata")},
        mode="translate",
        request=TranslationRequest(),
    ).outputs
    assert result_float["sample"]["keys"]["row_id"].tolist() == result["sample"]["keys"]["row_id"].tolist()
