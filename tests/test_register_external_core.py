from __future__ import annotations

import pandas as pd
import pytest

from text_analysis_lab.core.register_external import _normalize_external
from text_analysis_lab.core.types import ArtifactType


def test_dataframe_adapter_splits_into_writer_batches() -> None:
    frame = pd.DataFrame(
        {"row_id": [0, 1, 2], "score": [1.0, 2.0, 3.0], "group": ["a", "b", "c"]}
    )
    kind, descriptor, payloads = _normalize_external(
        frame,
        artifact_type=ArtifactType.TABLE,
        primary_key=("row_id",),
        data_fields=("score",),
        metadata_fields=("group",),
        format=None,
        batch_size=2,
        duckdb_options=None,
    )
    assert kind == "dataframe"
    assert descriptor == {"kind": "dataframe", "rows": 3}
    batches = list(payloads)
    assert [len(batch["keys"]) for batch in batches] == [2, 1]
    assert batches[0]["data"]["score"].tolist() == [1.0, 2.0]
    assert batches[1]["metadata"]["group"].tolist() == ["c"]


def test_payload_iterable_retains_caller_batch_boundaries() -> None:
    supplied = [
        {"keys": pd.DataFrame({"row_id": [0, 1]})},
        {"keys": pd.DataFrame({"row_id": [2]})},
    ]
    kind, _, payloads = _normalize_external(
        supplied,
        artifact_type=ArtifactType.TABLE,
        primary_key=("row_id",),
        data_fields=(),
        metadata_fields=(),
        format=None,
        batch_size=1,
        duckdb_options=None,
    )
    assert kind == "payload_iterable"
    assert [len(batch["keys"]) for batch in payloads] == [2, 1]


def test_matrix_registration_requires_explicit_payload_shape() -> None:
    with pytest.raises(TypeError, match="DataFrame registration"):
        _normalize_external(
            pd.DataFrame({"row_id": [0]}),
            artifact_type=ArtifactType.DENSE_MATRIX,
            primary_key=("row_id",),
            data_fields=(),
            metadata_fields=(),
            format=None,
            batch_size=10,
            duckdb_options=None,
        )
