from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import text_analysis_lab as teal
from text_analysis_lab.core.errors import PipelineError
from text_analysis_lab.translators import CountVectorizer, RegexCleaner, Word2Vec


def _source(project: teal.Project, tmp_path: Path):
    path = tmp_path / "documents.csv"
    pd.DataFrame(
        {
            "text": ["  alpha   beta  ", " gamma   alpha "],
        }
    ).to_csv(path, index=False)
    return project.read_csv(
        path,
        text_fields="text",
        metadata_fields=None,
        batch_size=1,
    )


def test_pipeline_native_linear_execution_uses_translator_contracts() -> None:
    pipeline = teal.Pipeline()
    compact = pipeline.add(
        "compact",
        RegexCleaner(rules=({"pattern": r"\s+", "replacement": " "},)),
        source=pipeline.input,
    )
    renamed = pipeline.add(
        "renamed",
        RegexCleaner(rules=({"pattern": "alpha", "replacement": "A"},)),
        source=compact["output"],
    )
    pipeline.output("clean", renamed["output"])

    result = pipeline.translate(
        pd.Series(["  alpha   beta  ", " gamma   alpha "]),
        outputs=["clean"],
    )

    assert result["clean"].tolist() == ["A beta", "gamma A"]
    caps = pipeline.execution_capabilities(outputs=["clean"])
    assert caps.reusable
    assert caps.native
    assert caps.artifact
    assert caps.portable


def test_requested_output_closure_ignores_unrelated_nonreusable_branch() -> None:
    pipeline = teal.Pipeline()
    clean = pipeline.add(
        "clean",
        RegexCleaner(),
        source=pipeline.input,
    )
    training = pipeline.add(
        "training",
        Word2Vec(min_count=1),
        source=pipeline.input,
    )
    pipeline.output("clean", clean["output"])
    pipeline.output("training", training["output"])

    clean_caps = pipeline.execution_capabilities(outputs=["clean"])
    all_caps = pipeline.execution_capabilities()
    assert clean_caps.native
    assert clean_caps.reusable
    assert not all_caps.native
    assert not all_caps.reusable

    result = pipeline.translate(pd.Series([" alpha "]), outputs=["clean"])
    assert result["clean"].tolist() == ["alpha"]


def test_pipeline_blocks_unfitted_translator_before_project_operation(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project",
        name="pipeline_unfitted",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        pipeline = teal.Pipeline()
        count = pipeline.add(
            "count",
            CountVectorizer(text_field="text"),
            source=pipeline.input,
        )
        pipeline.output("counts", count["output"])
        before = len(project.list_operations())

        with pytest.raises(PipelineError, match="requires fitted state|cannot execute"):
            project.run_pipeline(pipeline, source)

        assert len(project.list_operations()) == before
    finally:
        project.close()


def test_project_pipeline_executes_each_stage_as_normal_translation(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "project",
        name="pipeline_artifacts",
        delete_existing=True,
    )
    try:
        source = _source(project, tmp_path)
        pipeline = teal.Pipeline()
        compact = pipeline.add(
            "compact",
            RegexCleaner(
                text_field="text",
                output_field="compact_text",
                rules=({"pattern": r"\s+", "replacement": " "},),
            ),
            source=pipeline.input,
        )
        renamed = pipeline.add(
            "renamed",
            RegexCleaner(
                text_field="compact_text",
                output_field="clean_text",
                rules=({"pattern": "alpha", "replacement": "A"},),
            ),
            source=compact["output"],
        )
        pipeline.output("clean", renamed["output"])

        before = len(project.list_operations())
        outputs = project.run_pipeline(pipeline, source)
        after = len(project.list_operations())

        assert after - before == 2
        frame = outputs["clean"].query(
            key_columns=False,
            data_columns=True,
            metadata_columns=False,
            include_position=False,
            form="table",
        )
        assert frame["clean_text"].tolist() == ["A beta", "gamma A"]
    finally:
        project.close()
