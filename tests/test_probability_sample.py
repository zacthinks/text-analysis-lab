"""Regression tests for the additive probability_sample interface."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.operator import InputBatch, TranslationRequest
from text_analysis_lab.core.probability_sample import ProbabilitySampleTranslator
from text_analysis_lab.core.probability_split import ProbabilitySplitTranslator
from text_analysis_lab.core.types import ArtifactType


def _packet(values=None, *, label="documents", field="H"):
    frame = pd.DataFrame({"row_id": range(10)})
    if values is not None:
        frame[field] = values
    return InputBatch(
        source_label=label,
        artifact_id="art_" + label,
        primary_key=("row_id",),
        data=frame,
        batch_index=0, batch_count=1, is_first=True, is_last=True,
    )


def _run(translator, *, strata=None, probabilities=None):
    sources = {"documents": _packet()}
    if strata is not None:
        sources["strata"] = _packet(strata, label="strata")
    if probabilities is not None:
        sources["probabilities"] = _packet(
            probabilities, label="probabilities", field="probability"
        )
    return translator.translate_batch(
        sources, mode="translate", request=TranslationRequest()
    ).outputs


def _ids(result, name):
    return result[name]["keys"]["row_id"].tolist()


def test_srs_matches_legacy_seeded_draw_and_pi():
    modern = _run(ProbabilitySampleTranslator(n=4, random_state=17))
    legacy = _run(ProbabilitySplitTranslator(n=4, random_state=17))
    assert _ids(modern, "sample") == _ids(legacy, "sample")
    assert _ids(modern, "remainder") == _ids(legacy, "remainder")
    assert modern["pi"]["data"]["pi"].tolist() == pytest.approx([0.4] * 4)


def test_fixed_fraction_rounds_half_up_and_keeps_fixed_size():
    result = _run(ProbabilitySampleTranslator(fraction=0.25, random_state=3))
    assert len(_ids(result, "sample")) == 3
    assert result["pi"]["data"]["pi"].tolist() == pytest.approx([0.3] * 3)


def test_stratified_proportional_equal_and_exact_counts():
    h = [0] * 6 + [1] * 4
    proportional = _run(
        ProbabilitySampleTranslator(n=5, has_strata=True, random_state=2),
        strata=h,
    )
    sampled = _ids(proportional, "sample")
    assert sum(h[i] == 0 for i in sampled) == 3
    assert sum(h[i] == 1 for i in sampled) == 2
    assert proportional["pi"]["data"]["pi"].tolist() == pytest.approx([0.5] * 5)

    equal = _run(
        ProbabilitySampleTranslator(
            n=4, has_strata=True, allocation="equal", random_state=2
        ),
        strata=h,
    )
    assert sum(h[i] == 0 for i in _ids(equal, "sample")) == 2
    assert sum(h[i] == 1 for i in _ids(equal, "sample")) == 2

    explicit = _run(
        ProbabilitySampleTranslator(
            n_by_stratum={0: 1, 1: 3}, has_strata=True, random_state=2
        ),
        strata=h,
    )
    sample_ids = _ids(explicit, "sample")
    assert sum(h[i] == 0 for i in sample_ids) == 1
    assert sum(h[i] == 1 for i in sample_ids) == 3


def test_stratum_probabilities_and_keyed_artifact_align():
    h = [0] * 6 + [1] * 4
    result = _run(
        ProbabilitySampleTranslator(
            probability={0: 1.0, 1: 0.2}, has_strata=True, random_state=42
        ),
        strata=h,
    )
    assert set(range(6)).issubset(_ids(result, "sample"))
    pis = result["pi"]["data"]["pi"].tolist()
    assert 1.0 in pis

    keyed = _run(
        ProbabilitySampleTranslator(
            probability_source=True, random_state=42,
        ),
        probabilities=[1.0] * 6 + [0.2] * 4,
    )
    assert _ids(result, "sample") == _ids(keyed, "sample")
    assert result["pi"]["data"]["pi"].tolist() == keyed["pi"]["data"]["pi"].tolist()


def test_bernoulli_is_not_fixed_size_and_conflicting_args_fail():
    bernoulli = ProbabilitySampleTranslator(probability=0.4, random_state=4)
    assert bernoulli.design == "bernoulli"
    result = _run(bernoulli)
    assert set(_ids(result, "sample")).isdisjoint(_ids(result, "remainder"))
    assert len(_ids(result, "sample")) + len(_ids(result, "remainder")) == 10
    assert result["pi"]["data"]["pi"].tolist() == pytest.approx(
        [0.4] * len(_ids(result, "sample"))
    )
    with pytest.raises(ValueError, match="exactly one"):
        ProbabilitySampleTranslator(n=4, probability=0.4)
    with pytest.raises(ValueError, match="imply"):
        ProbabilitySampleTranslator(n=4, design="bernoulli")
    with pytest.raises(ValueError, match="strata"):
        ProbabilitySampleTranslator(probability={0: 0.4})


def test_bernoulli_all_selected_and_none_selected_are_valid_draws():
    all_selected = _run(
        ProbabilitySampleTranslator(probability=1.0, random_state=1)
    )
    assert _ids(all_selected, "sample") == list(range(10))
    assert _ids(all_selected, "remainder") == []
    assert _ids(all_selected, "pi") == list(range(10))
    assert all_selected["pi"]["data"]["pi"].tolist() == [1.0] * 10

    none_selected = _run(
        ProbabilitySampleTranslator(probability=1e-12, random_state=1)
    )
    assert _ids(none_selected, "sample") == []
    assert _ids(none_selected, "pi") == []
    assert none_selected["pi"]["data"]["pi"].empty
    assert _ids(none_selected, "remainder") == list(range(10))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"n": 3}, "srswor"),
        ({"n": 3, "has_strata": True}, "stratified_srswor"),
        ({"probability": 0.4}, "bernoulli"),
        ({"probability": {0: 0.1, 1: 0.9}, "has_strata": True}, "bernoulli"),
        ({"n_by_stratum": {0: 1, 1: 3}, "has_strata": True}, "stratified_srswor"),
        ({"probability_source": True}, "bernoulli"),
    ],
)
def test_saved_design_roundtrip(kwargs, expected):
    translator = ProbabilitySampleTranslator(**kwargs)
    assert translator.design == expected
    restored = ProbabilitySampleTranslator.from_json_state(translator.to_json_state())
    assert restored.to_json_state() == translator.to_json_state()


def test_alignment_mismatch_does_not_silently_accept_missing_keys():
    wrong = _packet([0] * 10, label="strata")
    wrong.data.loc[0, "row_id"] = 999
    with pytest.raises(ArtifactError, match="key set"):
        ProbabilitySampleTranslator(n=3, has_strata=True).translate_batch(
            {"documents": _packet(), "strata": wrong},
            mode="translate", request=TranslationRequest(),
        )


def test_source_contract_pi_lineage():
    src = SimpleNamespace(
        artifact_type=ArtifactType.TABLE,
        primary_key=("row_id",),
    )
    op = ProbabilitySampleTranslator(n=4)
    outputs = op.output_specs(sources={"documents": src}, request=TranslationRequest())
    assert outputs["pi"].basis_labels == ("sample",)
    assert outputs["sample"].basis_labels == ("documents",)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 4, "has_strata": True, "allocation": "equal"},
        {"n": 4, "has_strata": True, "allocation": {0: 2, 1: 1}},
        {"fraction": 0.4, "has_strata": True},
        {"probability": 0.2, "has_strata": True},
    ],
)
def test_explicit_allocation_modes_roundtrip(kwargs):
    original = ProbabilitySampleTranslator(**kwargs)
    saved = original.to_json_state()
    restored = ProbabilitySampleTranslator.from_json_state(saved)
    assert restored.to_json_state() == saved


def test_project_probability_sample_artifacts_and_reopen(tmp_path):
    import text_analysis_lab as teal

    root = tmp_path / "probability_sample_project"
    project = teal.Project.create(root, name="sample_project")
    try:
        frame = pd.DataFrame({
            "row_id": list(range(20)),
            "text": [f"document {i}" for i in range(20)],
        })
        source = project.register_external(
            frame, primary_key="row_id", data_fields="text"
        )
        strata = project.register_external(
            pd.DataFrame({
                "row_id": list(range(20)),
                "H": [0] * 12 + [1] * 8,
            }),
            primary_key="row_id", data_fields="H",
        )
        outcome = project.probability_sample(
            source, n=8, strata=strata, allocation="equal", random_state=14
        )
        sample, remainder, pi = (
            outcome["sample"], outcome["remainder"], outcome["pi"]
        )
        def keys(artifact):
            return set(artifact.query(
                key_columns=True, data_columns=False,
                metadata_columns=False, form="table",
            )["row_id"].tolist())

        assert len(keys(sample)) == 8
        assert keys(sample).isdisjoint(keys(remainder))
        assert keys(sample) | keys(remainder) == set(range(20))
        assert keys(pi) == keys(sample)
        assert pi.descriptor["lineage"]["basis_artifact_ids"] == [
            sample.artifact_id
        ]
        operation = project.get_operation(pi.operation_id)
        frozen = project.get_operator(operation["operator_id"])
        assert isinstance(frozen, ProbabilitySampleTranslator)
        assert frozen.design == "stratified_srswor"
        saved_state = frozen.to_json_state()
        sample_id, pi_id = sample.artifact_id, pi.artifact_id
    finally:
        project.close()

    reopened = teal.Project.open(root)
    try:
        reloaded = reopened.get_artifact(pi_id)
        assert keys(reloaded) == keys(reopened.get_artifact(sample_id))
        operation = reopened.get_operation(reloaded.operation_id)
        frozen = reopened.get_operator(operation["operator_id"])
        assert isinstance(frozen, ProbabilitySampleTranslator)
        assert frozen.to_json_state() == saved_state
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("probability", "expected_selected"),
    [(1.0, 20), (1e-12, 0)],
)
def test_project_bernoulli_extreme_draw_persists_empty_table_and_reopens(
    tmp_path, probability, expected_selected
):
    """Valid Bernoulli edge draws remain committed without conditioning/redraw."""
    import text_analysis_lab as teal

    root = tmp_path / "bernoulli_edge"
    project = teal.Project.create(root, name="bernoulli_edge")
    expected_rows = {
        "sample": expected_selected,
        "remainder": 20 - expected_selected,
        "pi": expected_selected,
    }
    try:
        source = project.register_external(
            pd.DataFrame({
                "row_id": range(20),
                "text": [f"text {i}" for i in range(20)],
            }),
            primary_key="row_id", data_fields="text",
        )
        result = project.probability_sample(
            source, probability=probability, random_state=1
        )
        artifact_ids = {label: art.artifact_id for label, art in result.items()}
        assert set(artifact_ids) == {"sample", "remainder", "pi"}
        for label, expected_n in expected_rows.items():
            artifact = result[label]
            assert artifact.n_rows == expected_n
            assert artifact.descriptor["status"] == "complete"
            queried = artifact.query(
                key_columns=True,
                data_columns=["pi"] if label == "pi" else False,
                metadata_columns=False,
                form="table",
            )
            assert len(queried) == expected_n
            assert list(queried.columns) == (
                ["row_id", "pi"] if label == "pi" else ["row_id"]
            )
            if label == "pi":
                assert queried["pi"].tolist() == [probability] * expected_n
        assert result["pi"].descriptor["lineage"]["basis_artifact_ids"] == [
            result["sample"].artifact_id
        ]
        operator_row = project.get_operation(result["pi"].operation_id)
        frozen = project.get_operator(operator_row["operator_id"])
        assert isinstance(frozen, ProbabilitySampleTranslator)
        assert frozen.design == "bernoulli"
    finally:
        project.close()

    reopened = teal.Project.open(root)
    try:
        for label, artifact_id in artifact_ids.items():
            artifact = reopened.get_artifact(artifact_id)
            assert artifact.n_rows == expected_rows[label]
            queried = artifact.query(
                key_columns=True,
                data_columns=["pi"] if label == "pi" else False,
                metadata_columns=False,
                form="table",
            )
            assert len(queried) == expected_rows[label]
        pi_artifact = reopened.get_artifact(artifact_ids["pi"])
        assert pi_artifact.descriptor["lineage"]["basis_artifact_ids"] == [
            artifact_ids["sample"]
        ]
        op = reopened.get_operation(pi_artifact.operation_id)
        restored = reopened.get_operator(op["operator_id"])
        assert isinstance(restored, ProbabilitySampleTranslator)
        assert restored.design == "bernoulli"
        assert restored.scalar_probability == probability
    finally:
        reopened.close()



def test_keyed_probabilities_reject_duplicate_source_keys():
    """The probability side input must not silently overwrite duplicate IDs."""
    probability_input = _packet(
        [0.15 + 0.01 * i for i in range(10)],
        label="probabilities",
        field="probability",
    )
    probability_input.data.loc[5, "row_id"] = 4
    with pytest.raises(ArtifactError, match="duplicate primary keys"):
        ProbabilitySampleTranslator(
            probability_source=True, random_state=37
        ).translate_batch(
            {"documents": _packet(), "probabilities": probability_input},
            mode="translate",
            request=TranslationRequest(),
        )


def test_project_shuffled_keyed_probabilities_align_and_reopen(tmp_path):
    """Different per-unit probabilities must join by keys, not physical order."""
    import text_analysis_lab as teal

    root = tmp_path / "keyed_bernoulli"
    probability_by_key = {
        key: 0.07 + key * 0.043 for key in range(20)
    }
    expected_draw = np.random.default_rng(37).random(20)
    expected_sample = {
        key for key, u in enumerate(expected_draw)
        if u < probability_by_key[key]
    }
    # This test should detect both an incorrect draw and incorrect stored pi.
    assert 0 < len(expected_sample) < 20

    probability_rows = pd.DataFrame({
        "row_id": list(probability_by_key),
        "probability": list(probability_by_key.values()),
    }).sample(frac=1, random_state=203).reset_index(drop=True)
    assert probability_rows["row_id"].tolist() != list(range(20))
    assert probability_rows["probability"].nunique() == 20

    project = teal.Project.create(root, name="keyed_bernoulli")
    try:
        source = project.register_external(
            pd.DataFrame({
                "row_id": range(20),
                "text": [f"document {key}" for key in range(20)],
            }),
            primary_key="row_id",
            data_fields="text",
        )
        probability_artifact = project.register_external(
            probability_rows,
            primary_key="row_id",
            data_fields="probability",
        )
        # Confirm that the probability artifact itself retains the shuffled
        # physical row order after registration.
        source_probabilities = probability_artifact.query(
            key_columns=True,
            data_columns=["probability"],
            metadata_columns=False,
            include_position=True,
            order_by="_position",
            form="table",
        )
        assert source_probabilities["row_id"].tolist() == (
            probability_rows["row_id"].tolist()
        )

        result = project.probability_sample(
            source,
            probability=probability_artifact,
            design="bernoulli",
            random_state=37,
        )
        ids = {label: artifact.artifact_id for label, artifact in result.items()}
        sample_keys = set(result["sample"].query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].tolist())
        remainder_keys = set(result["remainder"].query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].tolist())
        stored_pi = result["pi"].query(
            key_columns=True, data_columns=["pi"], metadata_columns=False,
            form="table",
        )
        assert sample_keys == expected_sample
        assert remainder_keys == set(range(20)) - expected_sample
        assert set(stored_pi["row_id"].tolist()) == expected_sample
        for row in stored_pi.itertuples(index=False):
            assert row.pi == pytest.approx(probability_by_key[row.row_id])
        assert result["pi"].descriptor["lineage"]["basis_artifact_ids"] == [
            ids["sample"]
        ]
        operation = project.get_operation(result["pi"].operation_id)
        frozen = project.get_operator(operation["operator_id"])
        assert isinstance(frozen, ProbabilitySampleTranslator)
        assert frozen.design == "bernoulli"
        assert frozen.probability_kind == "artifact"
    finally:
        project.close()

    reopened = teal.Project.open(root)
    try:
        sample = reopened.get_artifact(ids["sample"])
        remainder = reopened.get_artifact(ids["remainder"])
        pi = reopened.get_artifact(ids["pi"])
        sampled_keys = set(sample.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].tolist())
        remaining_keys = set(remainder.query(
            key_columns=True, data_columns=False, metadata_columns=False,
            form="table",
        )["row_id"].tolist())
        assert sampled_keys == expected_sample
        assert remaining_keys == set(range(20)) - expected_sample
        reloaded_pi = pi.query(
            key_columns=True, data_columns=["pi"], metadata_columns=False,
            form="table",
        )
        assert set(reloaded_pi["row_id"].tolist()) == expected_sample
        for row in reloaded_pi.itertuples(index=False):
            assert row.pi == pytest.approx(probability_by_key[row.row_id])
        assert pi.descriptor["lineage"]["basis_artifact_ids"] == [
            ids["sample"]
        ]
        operator_row = reopened.get_operation(pi.operation_id)
        frozen = reopened.get_operator(operator_row["operator_id"])
        assert isinstance(frozen, ProbabilitySampleTranslator)
        assert frozen.design == "bernoulli"
        assert frozen.probability_kind == "artifact"
    finally:
        reopened.close()
