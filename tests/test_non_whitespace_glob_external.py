from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import DictionaryTranslator


FEATURES = [
    "survey",
    "surveys",
    "survey data",
    "surveys were",
    "national survey",
    "case study",
    "case studies",
    "case study design",
]


def _register_identity_dtm(project: teal.Project):
    artifact_id = "art_non_whitespace_glob_dtm"
    writer = create_artifact_writer(
        artifact_type="sparse_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label="non_whitespace_glob_dtm",
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write(
        {
            "keys": pd.DataFrame({"row_id": list(range(len(FEATURES)))}),
            "data": {
                "values": sparse.identity(len(FEATURES), dtype=float, format="csr"),
                "columns": FEATURES,
            },
        }
    )
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix",
        label="non_whitespace_glob_dtm",
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


def test_non_whitespace_glob_real_project_audit_and_translation_agree(tmp_path: Path) -> None:
    project = teal.Project.create(tmp_path / "non_whitespace_glob", name="nwg")
    try:
        dtm = _register_identity_dtm(project)
        dictionary = teal.dictionaries.Dictionary(
            {"method": ["survey*", "case stud*"]},
            valuetype="non_whitespace_glob",
            case_sensitive=False,
        )

        matches = dtm.analysis.dictionary_matches(dictionary)
        assert matches["feature"].tolist() == [
            "survey",
            "surveys",
            "case study",
            "case studies",
        ]

        translated = project.translate(
            DictionaryTranslator(dictionary), dtm, batch_size=3
        )["output"]
        values = translated.get_matrix().toarray().reshape(-1).tolist()
        assert values == [1, 1, 0, 0, 0, 1, 1, 0]

        operation_id = translated.operation_id
        assert operation_id is not None
        operator_id = project.catalog.get_operation(operation_id)["operator_id"]
    finally:
        project.close()

    reopened = teal.Project.open(tmp_path / "non_whitespace_glob")
    try:
        operator = reopened.get_operator(str(operator_id))
        assert isinstance(operator, DictionaryTranslator)
        assert operator.dictionary is not None
        assert operator.dictionary.valuetype == "non_whitespace_glob"
    finally:
        reopened.close()


def test_all_pattern_valuetypes_use_python_regex_semantics_with_pyarrow_strings(tmp_path: Path) -> None:
    """Arrow-backed pandas strings must not switch dictionary matching to RE2."""
    project = teal.Project.create(tmp_path / "glob_regex_backend", name="glob-regex")
    try:
        dtm = _register_identity_dtm(project)

        ordinary_glob = teal.dictionaries.Dictionary(
            {"method": ["survey*"]},
            valuetype="glob",
            case_sensitive=False,
        )
        assert dtm.analysis.dictionary_matches(ordinary_glob)["feature"].tolist() == [
            "survey",
            "surveys",
            "survey data",
            "surveys were",
        ]

        # Python ``re`` supports lookbehind; Arrow/RE2 does not. Regex dictionary
        # behavior should therefore remain independent of pandas' string backend.
        regex = teal.dictionaries.Dictionary(
            {"method": [r"(?<!national )survey$"]},
            valuetype="regex",
            case_sensitive=False,
        )
        assert dtm.analysis.dictionary_matches(regex)["feature"].tolist() == ["survey"]
    finally:
        project.close()
