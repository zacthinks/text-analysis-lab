from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import text_analysis_lab as teal
from text_analysis_lab.integrations._representation_replay import _replay_texts_like
from text_analysis_lab.translators import (
    CountVectorizer,
    DelimiterDecomposer,
    SVD,
    TfidfTransformer,
)


pytestmark = pytest.mark.acceptance


_DOCUMENTS = [
    "alpha beta gamma",
    "alpha alpha delta",
    "beta gamma epsilon",
    "gamma delta epsilon",
    "alpha epsilon beta",
    "delta gamma alpha",
    "epsilon beta beta",
    "gamma alpha beta",
]


def _source(project: teal.Project, tmp_path: Path):
    source_path = tmp_path / "documents.csv"
    pd.DataFrame(
        {
            "text": _DOCUMENTS,
            "group": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    ).to_csv(source_path, index=False)
    return project.read_csv(
        source_path,
        text_fields="text",
        metadata_fields="group",
        batch_size=3,
    )


def _fit_representation(project: teal.Project, source):
    count = CountVectorizer(text_field="text", min_df=1)
    counts = project.translate(count, source)["output"]

    tfidf = TfidfTransformer(norm=None)
    weighted = project.translate(tfidf, counts)["output"]

    svd = SVD(n_components=2, random_state=7)
    reduced = project.translate(svd, weighted)["output"]

    return {
        "count": count,
        "counts": counts,
        "tfidf": tfidf,
        "weighted": weighted,
        "svd": svd,
        "reduced": reduced,
    }


def _dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def _ordered_keys(artifact) -> pd.DataFrame:
    return artifact.query(
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        include_position=False,
        order_by="_position",
        form="table",
    ).reset_index(drop=True)


def test_frozen_pipeline_survives_fresh_interpreter_and_matches_geco_replay(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    project = teal.Project.create(project_path, name="standalone_acceptance")
    try:
        source = _source(project, tmp_path)
        fitted = _fit_representation(project, source)
        payload = {
            "project_path": str(project_path),
            "count_operator_id": str(fitted["count"].operator_id),
            "tfidf_operator_id": str(fitted["tfidf"].operator_id),
            "svd_operator_id": str(fitted["svd"].operator_id),
            "target_artifact_id": str(fitted["reduced"].artifact_id),
            "texts": [
                "alpha gamma gamma",
                "epsilon delta alpha",
                "beta beta epsilon",
            ],
        }
    finally:
        project.close()

    payload_path = tmp_path / "subprocess-input.json"
    output_path = tmp_path / "subprocess-output.json"
    payload["output_path"] = str(output_path)
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    child = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import sys

        import numpy as np
        import text_analysis_lab as teal

        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        project = teal.Project.open(payload["project_path"])
        try:
            count = project.get_operator(payload["count_operator_id"])
            tfidf = project.get_operator(payload["tfidf_operator_id"])
            svd = project.get_operator(payload["svd_operator_id"])
        finally:
            project.close()

        values = svd.translate(tfidf.translate(count.translate(payload["texts"])))
        Path(payload["output_path"]).write_text(
            json.dumps(np.asarray(values).tolist()),
            encoding="utf-8",
        )
        """
    )
    subprocess.run(
        [sys.executable, "-c", child, str(payload_path)],
        check=True,
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    child_values = np.asarray(json.loads(output_path.read_text(encoding="utf-8")))

    reopened = teal.Project.open(project_path)
    try:
        before_operations = len(reopened.list_operations())
        before_artifacts = len(reopened.list_artifacts())
        replayed = _replay_texts_like(
            reopened,
            payload["target_artifact_id"],
            payload["texts"],
            query=False,
        )
        assert len(reopened.list_operations()) == before_operations
        assert len(reopened.list_artifacts()) == before_artifacts
    finally:
        reopened.close()

    np.testing.assert_allclose(child_values, _dense(replayed), rtol=1e-12, atol=1e-12)


def test_frozen_representation_is_invariant_to_teal_batch_size(
    tmp_path: Path,
) -> None:
    project = teal.Project.create(
        tmp_path / "batch-project",
        name="batch_acceptance",
    )
    try:
        source = _source(project, tmp_path)
        fitted = _fit_representation(project, source)

        count_small = project.translate(
            project.get_operator(str(fitted["count"].operator_id)),
            source,
            batch_size=1,
            workers=1,
        )["output"]
        count_large = project.translate(
            project.get_operator(str(fitted["count"].operator_id)),
            source,
            batch_size=5,
            workers=1,
        )["output"]

        tfidf_small = project.translate(
            project.get_operator(str(fitted["tfidf"].operator_id)),
            count_small,
            batch_size=1,
            workers=1,
        )["output"]
        tfidf_large = project.translate(
            project.get_operator(str(fitted["tfidf"].operator_id)),
            count_large,
            batch_size=5,
            workers=1,
        )["output"]

        svd_small = project.translate(
            project.get_operator(str(fitted["svd"].operator_id)),
            tfidf_small,
            batch_size=1,
            workers=1,
        )["output"]
        svd_large = project.translate(
            project.get_operator(str(fitted["svd"].operator_id)),
            tfidf_large,
            batch_size=5,
            workers=1,
        )["output"]

        for left, right in (
            (count_small, count_large),
            (tfidf_small, tfidf_large),
            (svd_small, svd_large),
        ):
            pd.testing.assert_frame_equal(_ordered_keys(left), _ordered_keys(right))
            np.testing.assert_allclose(
                _dense(left.get_matrix()),
                _dense(right.get_matrix()),
                rtol=1e-12,
                atol=1e-12,
            )
    finally:
        project.close()


def test_cardinality_changing_translation_is_invariant_to_batch_size(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "segments.csv"
    pd.DataFrame(
        {
            "text": [
                "alpha|beta|gamma",
                "delta",
                "epsilon|zeta",
                "",
                "eta|theta|iota|kappa",
            ],
            "group": [0, 0, 1, 1, 2],
        }
    ).to_csv(source_path, index=False)

    project = teal.Project.create(
        tmp_path / "segments-project",
        name="segments_acceptance",
    )
    try:
        source = project.read_csv(
            source_path,
            text_fields="text",
            metadata_fields="group",
            batch_size=2,
        )
        small = project.translate(
            DelimiterDecomposer(delimiter="|", new_key="segment_id"),
            source,
            batch_size=1,
            workers=1,
        )["output"]
        large = project.translate(
            DelimiterDecomposer(delimiter="|", new_key="segment_id"),
            source,
            batch_size=4,
            workers=1,
        )["output"]

        def frame(artifact):
            return artifact.query(
                key_columns=True,
                data_columns=True,
                metadata_columns=False,
                include_position=False,
                order_by="_position",
                form="table",
            ).reset_index(drop=True)

        pd.testing.assert_frame_equal(frame(small), frame(large))
    finally:
        project.close()


def test_built_wheel_imports_and_standalone_translation_works_outside_checkout(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheel_dir),
            str(repo_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(wheel_dir.glob("text_analysis_lab-*.whl"))
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    subprocess.run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            str(venv_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    venv_python = (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python"
    )

    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            str(wheels[0]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    smoke = textwrap.dedent(
        """
        import json
        from pathlib import Path

        import text_analysis_lab
        from text_analysis_lab.translators import CountVectorizer

        translator = CountVectorizer(vocabulary={"alpha": 0, "beta": 1})
        values = translator.translate(["alpha beta alpha", "beta"]).toarray()
        Path("result.json").write_text(
            json.dumps(
                {
                    "values": values.tolist(),
                    "module": text_analysis_lab.__file__,
                }
            ),
            encoding="utf-8",
        )
        """
    )
    outside = tmp_path / "outside-checkout"
    outside.mkdir()
    smoke_run = subprocess.run(
        [str(venv_python), "-c", smoke],
        check=False,
        cwd=outside,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert smoke_run.returncode == 0, (
        "wheel-installed smoke process failed\n"
        f"stdout:\n{smoke_run.stdout}\n"
        f"stderr:\n{smoke_run.stderr}"
    )
    result = json.loads((outside / "result.json").read_text(encoding="utf-8"))
    assert result["values"] == [[2, 1], [0, 1]]
    assert str(repo_root / "src") not in str(Path(result["module"]).resolve())
