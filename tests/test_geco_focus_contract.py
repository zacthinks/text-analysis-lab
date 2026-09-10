from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from text_analysis_lab.integrations.geco import GeCoIntegrationError, GeCoManager, _normalize_focus_codes


class FakeArtifact:
    def __init__(self, artifact_id: str, frame: pd.DataFrame, *, keys=("row_id",)) -> None:
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


class FakeCoder:
    registry: dict[str, "FakeCoder"] = {}

    def __init__(self, path: Path, data: pd.DataFrame) -> None:
        self.path = path
        self.data = data.copy()
        self.codes: list[dict[str, object]] = []
        self.exports: dict[int, pd.DataFrame] = {}
        self.launch_kwargs: dict[str, object] | None = None
        self.session_state: dict[str, object] = {}
        self.focus_config: dict[str, object] | None = None
        self.closed = False

    @classmethod
    def create(cls, *, project_dir, data, geometries=None, overwrite=False, **kwargs):
        assert geometries is None
        assert overwrite is False
        path = Path(project_dir)
        path.mkdir(parents=True, exist_ok=False)
        coder = cls(path, data)
        cls.registry[str(path.resolve())] = coder
        return coder

    @classmethod
    def open(cls, project_dir, **kwargs):
        assert "external_provider" not in kwargs
        coder = cls.registry[str(Path(project_dir).resolve())]
        coder.closed = False
        return coder

    def create_code(self, name, definition=""):
        code_id = len(self.codes) + 1
        self.codes.append({"code_id": code_id, "name": name, "description": definition})
        return code_id

    def sessions(self):
        return [{"session_id": 1, "title": "Default session", "state": dict(self.session_state)}]

    def patch_session_state(self, session_id, patch):
        assert int(session_id) == 1
        self.session_state.update(dict(patch))

    def configure_focus(self, **kwargs):
        self.focus_config = dict(kwargs)
        return 1

    def launch_focus_coder(self, **kwargs):
        self.launch_kwargs = dict(kwargs)
        return "focus-launched"

    def launch(self, **kwargs):
        self.launch_kwargs = dict(kwargs)
        return "launched"

    def export_codes(self, code_id):
        return self.exports[int(code_id)].copy()

    def close(self):
        self.closed = True


class FakeProject:
    def __init__(self, root: Path, artifacts: dict[str, FakeArtifact]) -> None:
        self.storage = SimpleNamespace(
            teal_dir=root / ".teal",
            touch_manifest=lambda: None,
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


def test_focus_code_normalization_preserves_order_and_descriptions() -> None:
    assert _normalize_focus_codes(
        {"qualitative": "Qualitative methods", "quantitative": "Quantitative methods"}
    ) == [
        {"name": "qualitative", "description": "Qualitative methods"},
        {"name": "quantitative", "description": "Quantitative methods"},
    ]
    with pytest.raises(GeCoIntegrationError, match="Duplicate focus code"):
        _normalize_focus_codes([{"name": "x"}, {"name": "x"}])


def test_create_focus_uses_audit_keys_but_larger_text_source(tmp_path: Path, monkeypatch) -> None:
    import text_analysis_lab.integrations.geco as bridge

    FakeCoder.registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: FakeCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.8.7-test")

    audit = FakeArtifact("audit", pd.DataFrame({"row_id": [3, 1]}))
    full = FakeArtifact(
        "full",
        pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3],
                "text": ["zero", "one", "two", "three"],
                "year": [2020, 2021, 2022, 2023],
            }
        ),
    )
    project = FakeProject(tmp_path, {"audit": audit, "full": full})
    manager = GeCoManager(project)

    linked = manager.create_focus(
        "audit_methods",
        documents=audit,
        text_source=full,
        text_field="text",
        metadata_fields=["year"],
        codes={
            "qualitative": "Uses qualitative methods",
            "quantitative": "Uses quantitative methods",
        },
        allow_unsure=False,
    )

    assert linked.mode == "focus_coder"
    assert linked.external_provider is None
    assert linked.coder.data["row_id"].tolist() == [3, 1]
    assert linked.coder.data["text"].tolist() == ["three", "one"]
    assert linked.coder.data["year"].tolist() == [2023, 2021]
    assert [row["name"] for row in linked.manifest["codes"]] == [
        "qualitative",
        "quantitative",
    ]
    assert linked.manifest["focus_session_id"] == 1
    assert linked.coder.focus_config == {
        "codes": [1, 2],
        "allow_unsure": False,
        "user_keys": [{"row_id": 3}, {"row_id": 1}],
        "title": "audit_methods",
    }
    assert linked.launch_focus() == "focus-launched"
    assert linked.coder.launch_kwargs == {"session_id": 1}

    linked.close()
    reopened = manager.open("audit_methods")
    assert reopened.is_focus_coder
    assert reopened.external_provider is None



def test_launch_focus_keeps_geco_0814_integrated_ui_fallback(tmp_path: Path, monkeypatch) -> None:
    import text_analysis_lab.integrations.geco as bridge

    class IntegratedUICoder(FakeCoder):
        configure_focus = None
        launch_focus_coder = None

    IntegratedUICoder.registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: IntegratedUICoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.8.14-test")

    audit = FakeArtifact("audit", pd.DataFrame({"row_id": [3, 1]}))
    full = FakeArtifact(
        "full",
        pd.DataFrame({"row_id": [0, 1, 2, 3], "text": ["zero", "one", "two", "three"]}),
    )
    project = FakeProject(tmp_path, {"audit": audit, "full": full})
    linked = GeCoManager(project).create_focus(
        "audit_methods",
        documents=audit,
        text_source=full,
        text_field="text",
        codes={"qualitative": "Q", "quantitative": "N"},
        allow_unsure=False,
    )

    assert linked.manifest["focus_session_id"] is None
    assert linked.coder.session_state["explore_code_palette"] == [1, 2]
    assert linked.launch_focus() == "launched"


def test_focus_export_requires_binary_completion_and_combines_codes(tmp_path: Path, monkeypatch) -> None:
    import text_analysis_lab.integrations.geco as bridge

    FakeCoder.registry.clear()
    monkeypatch.setattr(bridge, "_load_geometric_coder", lambda: FakeCoder)
    monkeypatch.setattr(bridge, "_installed_geco_version", lambda: "0.8.7-test")

    audit = FakeArtifact("audit", pd.DataFrame({"row_id": [3, 1]}))
    full = FakeArtifact(
        "full",
        pd.DataFrame({"row_id": [0, 1, 2, 3], "text": ["zero", "one", "two", "three"]}),
    )
    project = FakeProject(tmp_path, {"audit": audit, "full": full})
    linked = GeCoManager(project).create_focus(
        "audit_methods",
        documents=audit,
        text_source=full,
        text_field="text",
        codes={"qualitative": "Q", "quantitative": "N"},
        allow_unsure=True,
    )
    linked.coder.exports[1] = pd.DataFrame({"row_id": [3, 1], "label": [1, 0]})
    linked.coder.exports[2] = pd.DataFrame({"row_id": [3], "label": [1]})

    with pytest.raises(GeCoIntegrationError, match="unlabeled or Unsure"):
        linked.export_focus_labels(
            fields={"qualitative": "qual", "quantitative": "quant"}
        )

    linked.coder.exports[2] = pd.DataFrame({"row_id": [3, 1], "label": [1, 0]})
    result = linked.export_focus_labels(
        fields={"qualitative": "qual", "quantitative": "quant"},
        output_label="audit_labels",
        alias="methods_audit_labels",
        overwrite=True,
    )
    assert result["data_fields"] == ["qual", "quant"]
    assert result["require_complete"] is True
    assert result["alias"] == "methods_audit_labels"
    assert result["overwrite"] is True
    assert project.created_frame[["row_id", "qual", "quant"]].to_dict("records") == [
        {"row_id": 3, "qual": 1, "quant": 1},
        {"row_id": 1, "qual": 0, "quant": 0},
    ]
