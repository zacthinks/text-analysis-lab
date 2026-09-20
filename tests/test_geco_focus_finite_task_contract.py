from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

geometric_coder = pytest.importorskip("geometric_coder")
from geometric_coder import GeometricCoder

from text_analysis_lab.integrations.geco import GeCoManager

_FINITE_FOCUS_API = all(
    callable(getattr(GeometricCoder, name, None))
    for name in ("configure_focus", "focus_task", "focus_progress")
)


pytestmark = pytest.mark.skipif(
    not _FINITE_FOCUS_API,
    reason="Installed GeCo does not expose the finite Focus task API.",
)


class FakeArtifact:
    def __init__(
        self, artifact_id: str, frame: pd.DataFrame, *, keys=("row_id",)
    ) -> None:
        self.artifact_id = artifact_id
        self.primary_key = list(keys)
        self._frame = frame.reset_index(drop=True).copy()
        self.n_rows = len(frame)

    def query(
        self,
        *,
        key_columns=True,
        data_columns=False,
        metadata_columns=False,
        order_by=None,
        include_position=False,
        form="table",
        **kwargs,
    ):
        del order_by, form, kwargs
        columns = list(self.primary_key) if key_columns else []
        if data_columns not in (False, None):
            columns.extend(list(data_columns))
        if metadata_columns not in (False, None):
            columns.extend(list(metadata_columns))
        result = self._frame.loc[:, columns].copy()
        if include_position:
            result["_position"] = range(len(result))
        return result


class FakeProject:
    def __init__(self, root: Path, artifacts: dict[str, FakeArtifact]) -> None:
        self.storage = SimpleNamespace(
            teal_dir=root / ".teal", touch_manifest=lambda: None
        )
        self.storage.teal_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts = artifacts
        self.created_frame = None

    def get_artifact(self, ref):
        if isinstance(ref, FakeArtifact):
            return ref
        return self._artifacts[str(ref)]

    def from_keyed_frame(self, documents, frame, **kwargs):
        self.created_frame = frame.copy()
        return {"documents": documents, "frame": frame.copy(), **kwargs}


def test_teal_focus_bridge_matches_installed_geco_finite_task_contract(
    tmp_path: Path, monkeypatch
) -> None:
    import text_analysis_lab.integrations.geco as bridge

    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: GeometricCoder)
    monkeypatch.setattr(
        bridge,
        "_installed_geco_version",
        lambda: getattr(geometric_coder, "__version__", "unknown"),
    )

    audit = FakeArtifact("audit", pd.DataFrame({"row_id": [3, 1]}))
    full = FakeArtifact(
        "full",
        pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3],
                "text": ["zero", "one", "two", "three"],
                "group": ["A", "B", "C", "D"],
            }
        ),
    )
    project = FakeProject(tmp_path, {"audit": audit, "full": full})
    linked = GeCoManager(project).create_focus(
        "methods_audit",
        documents=audit,
        text_source=full,
        text_field="text",
        metadata_fields=["group"],
        codes={"qualitative": "Q", "quantitative": "N"},
        allow_unsure=False,
    )

    session_id = int(linked.manifest["focus_session_id"])
    task = linked.coder.focus_task(session_id)
    assert task["required_user_keys"] == [{"row_id": 3}, {"row_id": 1}]
    assert [row["code_id"] for row in task["required_codes"]] == [1, 2]
    assert task["allow_unsure"] is False

    # The finite Focus protocol treats Unsure as unresolved when forbidden.
    first = linked.coder.units()[0]
    linked.coder.annotate(
        int(first["observation_id"]), 1, "unsure", origin="human_focus_coder"
    )
    progress = linked.coder.focus_progress(session_id)
    assert progress["unsure_judgments"] == 1
    assert progress["resolved_judgments"] == 0
    assert not progress["complete"]
