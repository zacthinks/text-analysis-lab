from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

pyarrow = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.errors import ArtifactError
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import FittedPredictor, FunctionMapper


def _seed_documents(project: teal.Project):
    rows = pd.DataFrame(
        {
            "row_id": list(range(12)),
            "text": [
                "room is cold",
                "ordinary office",
                "temperature low",
                "warm room",
                "neutral hallway",
                "cold air",
                "conference room",
                "ambient temperature",
                "plain document",
                "room temperature cold",
                "another ordinary text",
                "cold temperature",
            ],
        }
    )
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("art_F"),
        artifact_id="art_F",
        label="focal_texts",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({"keys": rows[["row_id"]], "data": rows[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="art_F",
        artifact_type="table",
        label="focal_texts",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("art_F")


def _query(artifact, fields=True):
    return artifact.query(
        key_columns=True,
        data_columns=fields,
        metadata_columns=False,
        include_position=True,
        order_by="_position",
        form="table",
    )


def test_unit6_teal_only_vertical_close_reopen(tmp_path: Path, monkeypatch) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="unit6_vertical")
    try:
        F = _seed_documents(project)

        # Pre-label proxy H is same-key data, produced through an ordinary Translator.
        H = project.translate(
            FunctionMapper(
                lambda packet: {
                    "data": pd.DataFrame(
                        {
                            "H": packet["data"]["text"]
                            .str.contains(r"cold|temperature", case=False, regex=True)
                            .astype(int)
                        }
                    )
                }
            ),
            F,
            data_columns=["text"],
            batch_size=4,
        )["output"]
        assert H.descriptor["lineage"]["lineage_mode"] == "preserved_key"

        split = project.probability_split(
            F,
            n=6,
            remainder_label="train",
            sample_label="audit",
            strata=H,
            allocation={0: 1, 1: 2},
            random_state=17,
        )
        T, A, Pi_A = split["train"], split["audit"], split["pi"]
        t_keys = set(_query(T, False)["row_id"].astype(int))
        a_keys = set(_query(A, False)["row_id"].astype(int))
        assert t_keys.isdisjoint(a_keys)
        assert t_keys | a_keys == set(range(12))
        assert set(_query(Pi_A, ["pi"])["row_id"].astype(int)) == a_keys
        assert Pi_A.descriptor["lineage"]["basis_artifact_ids"] == [A.artifact_id]

        # External keyed measurements align by identity, not physical row order.
        t_rows = list(reversed(sorted(t_keys)))
        L_T = project.from_keyed_frame(
            T,
            pd.DataFrame({"row_id": t_rows[:3], "label": [1, 0, 1]}),
            data_fields=["label"],
        )
        assert _query(L_T, ["label"])["row_id"].astype(int).tolist() == sorted(
            t_rows[:3]
        )
        with pytest.raises(ArtifactError, match="not present in source"):
            project.from_keyed_frame(
                T,
                pd.DataFrame({"row_id": [999], "label": [1]}),
                data_fields=["label"],
            )
        duplicate_key = t_rows[0]
        with pytest.raises(ArtifactError, match="duplicate primary keys"):
            project.from_keyed_frame(
                T,
                pd.DataFrame(
                    {"row_id": [duplicate_key, duplicate_key], "label": [0, 1]}
                ),
                data_fields=["label"],
            )
        with pytest.raises(ArtifactError, match="exactly equal source"):
            project.from_keyed_frame(
                A,
                pd.DataFrame({"row_id": sorted(a_keys)[:-1], "label": 0}),
                data_fields=["label"],
                require_complete=True,
            )

        # Generic metadata attachment aligns by stable key, owns no data, and
        # leaves the source representation resolvable through preserved-key lineage.
        metadata_rows = pd.DataFrame(
            {
                "row_id": list(reversed(range(12))),
                "party": ["D" if row % 2 == 0 else "R" for row in reversed(range(12))],
            }
        )
        F_meta = project.attach_metadata(
            F,
            metadata_rows,
            metadata_fields=["party"],
            require_complete=True,
        )
        # get_data_columns() reports representation data available through
        # lineage, so the attached artifact still exposes F's text.  The
        # physical contract is that this artifact owns no local data component.
        assert "data" not in F_meta.components
        assert F_meta.get_data_columns() == ["text"]
        assert "party" in F_meta.get_metadata_columns()
        resolved_meta = F_meta.query(
            key_columns=True,
            data_columns=["text"],
            metadata_columns=["party"],
            metadata_mode="full",
            form="table",
        )
        assert len(resolved_meta) == 12
        assert resolved_meta.sort_values("row_id")["party"].tolist() == [
            "D" if row % 2 == 0 else "R" for row in range(12)
        ]

        # A simple readable/numeric representation for a frozen fitted predictor.
        features = project.translate(
            FunctionMapper(
                lambda packet: {
                    "data": pd.DataFrame(
                        {
                            "length": packet["data"]["text"].str.len().astype(float),
                            "cold": packet["data"]["text"]
                            .str.contains("cold", case=False)
                            .astype(float),
                            "temperature": packet["data"]["text"]
                            .str.contains("temperature", case=False)
                            .astype(float),
                        }
                    )
                }
            ),
            F,
            data_columns=["text"],
        )["output"]
        feature_frame = _query(features, ["length", "cold", "temperature"])
        X = feature_frame[["length", "cold", "temperature"]]
        y = ((feature_frame["cold"] + feature_frame["temperature"]) > 0).astype(int)
        model = LogisticRegression(random_state=0).fit(X, y)
        C = FittedPredictor(model, probability_class=1)
        L_F = project.translate(C, features, batch_size=5)["output"]
        lf_frame = _query(L_F, ["prediction", "probability"])
        assert len(lf_frame) == 12
        assert lf_frame["probability"].between(0, 1).all()

        # Independently code A; answers are based only on A's text, not predictions.
        a_order = _query(A, False)["row_id"].astype(int).tolist()
        text_by_key = dict(
            zip(_query(F, ["text"])["row_id"], _query(F, ["text"])["text"], strict=True)
        )
        answers = iter(
            [
                "1"
                if (
                    "cold" in text_by_key[key].lower()
                    or "temperature" in text_by_key[key].lower()
                )
                else "0"
                for key in a_order
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
        L_A = project.binary_code(
            A,
            text_source=F,
            text_field="text",
            instructions="Code 1 when the text mentions cold or temperature; otherwise 0.",
            context_before=1,
            context_after=1,
        )
        assert set(_query(L_A, ["label"])["row_id"].astype(int)) == a_keys
        assert not list((project.storage.teal_dir / "coding").glob("*.sqlite"))

        evaluation = L_F.analysis.classification(gold=L_A, pi=Pi_A)
        assert evaluation.weighted_confusion is not None
        assert set(evaluation.to_frame()["scope"]) == {"raw_audit", "design_weighted"}

        gd = L_F.analysis.generalized_difference(gold=L_A, pi=Pi_A)
        assert gd.variance_design == "stratified_srswor"
        assert gd.standard_error is not None
        assert np.isfinite(gd.estimated_mean)
        assert np.isfinite(gd.standard_error)

        ids = {
            "F": F.artifact_id,
            "H": H.artifact_id,
            "T": T.artifact_id,
            "A": A.artifact_id,
            "Pi_A": Pi_A.artifact_id,
            "L_T": L_T.artifact_id,
            "features": features.artifact_id,
            "L_F": L_F.artifact_id,
            "L_A": L_A.artifact_id,
        }
        predictor_operator_id = C.operator_id
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        for artifact_id in ids.values():
            artifact = reopened.get_artifact(artifact_id)
            assert artifact.status == "complete"

        reopened_predictor = reopened.get_operator(predictor_operator_id)
        assert isinstance(reopened_predictor, FittedPredictor)
        features = reopened.get_artifact(ids["features"])
        reused = reopened.translate(reopened_predictor, features)["output"]
        assert len(_query(reused, ["prediction", "probability"])) == 12

        evaluation = reopened.get_artifact(ids["L_F"]).analysis.classification(
            gold=reopened.get_artifact(ids["L_A"]),
            pi=reopened.get_artifact(ids["Pi_A"]),
        )
        assert evaluation.weighted_metrics is not None

        gd = reopened.get_artifact(ids["L_F"]).analysis.generalized_difference(
            gold=reopened.get_artifact(ids["L_A"]),
            pi=reopened.get_artifact(ids["Pi_A"]),
        )
        assert gd.variance_design == "stratified_srswor"
        assert gd.standard_error is not None
    finally:
        reopened.close()


def test_binary_code_progress_survives_real_project_close_reopen(
    tmp_path: Path, monkeypatch
) -> None:
    project_path = tmp_path / "binary_resume_project"
    project = teal.Project.create(project_path, name="binary_resume")
    try:
        F = _seed_documents(project)
        A = project.from_keyed_frame(
            F,
            pd.DataFrame({"row_id": [2, 7, 10], "audit_member": [1, 1, 1]}),
            data_fields=["audit_member"],
        )
        a_id = A.artifact_id
        f_id = F.artifact_id

        answers = iter(["1"])

        def interrupt_after_one(prompt):
            try:
                return next(answers)
            except StopIteration:
                raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupt_after_one)
        with pytest.raises(KeyboardInterrupt):
            project.binary_code(
                A,
                text_source=F,
                text_field="text",
                instructions="Binary audit resume test.",
                context_before=0,
                context_after=0,
            )
        assert len(list((project.storage.teal_dir / "coding").glob("*.sqlite"))) == 1
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        remaining = iter(["0", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(remaining))
        L_A = reopened.binary_code(
            reopened.get_artifact(a_id),
            text_source=reopened.get_artifact(f_id),
            text_field="text",
            instructions="Binary audit resume test.",
            context_before=0,
            context_after=0,
        )
        frame = _query(L_A, ["label"])
        assert frame["row_id"].astype(int).tolist() == [2, 7, 10]
        assert frame["label"].astype(int).tolist() == [1, 0, 1]
        assert not list((reopened.storage.teal_dir / "coding").glob("*.sqlite"))
    finally:
        reopened.close()


def test_function_mapper_real_sparse_matrix_to_table_preserves_keys(
    tmp_path: Path,
) -> None:
    """FunctionMapper must expose matrix batches as named DataFrames, not densify the corpus."""
    from scipy import sparse

    from text_analysis_lab.core.writer import create_artifact_writer

    project = teal.Project.create(tmp_path / "function_mapper_sparse", name="fm_sparse")
    try:
        artifact_id = "art_sparse_source"
        writer = create_artifact_writer(
            artifact_type="sparse_matrix",
            artifact_dir=project.storage.artifact_dir(artifact_id),
            artifact_id=artifact_id,
            label="scores",
            lineage_mode="new_key",
            basis_artifact_ids=(),
        )
        writer.write(
            {
                "keys": pd.DataFrame({"row_id": [4, 7, 9]}),
                "data": {
                    "values": sparse.csr_matrix([[0, 2], [0, 0], [3, 1]]),
                    "columns": ["qualitative", "quantitative"],
                },
            }
        )
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id=artifact_id,
            artifact_type="sparse_matrix",
            label="scores",
            lineage_mode="new_key",
            status="complete",
            basis_artifact_ids=(),
        )
        source = project.get_artifact(artifact_id)

        mapped = project.translate(
            FunctionMapper(
                lambda packet: {
                    "data": pd.DataFrame(
                        {
                            "qual": (packet["data"]["qualitative"] > 0).astype("int8"),
                            "quant": (packet["data"]["quantitative"] > 0).astype(
                                "int8"
                            ),
                        }
                    )
                }
            ),
            source,
            data_columns=["qualitative", "quantitative"],
            batch_size=2,
        )["output"]
        frame = mapped.query(
            key_columns=True,
            data_columns=True,
            form="table",
            order_by="_position",
        )
        assert frame["row_id"].astype(int).tolist() == [4, 7, 9]
        assert frame["qual"].astype(int).tolist() == [0, 0, 1]
        assert frame["quant"].astype(int).tolist() == [1, 0, 1]
    finally:
        project.close()
