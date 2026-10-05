"""Concrete artifact subclasses and loading registry for TeAL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, get_args

import numpy as np
import pandas as pd

from text_analysis_lab.core.artifact_base import (
    BaseArtifact,
    _merge_info_and_data_frames,
    _merge_info_and_data_records,
    _read_json,
)
from text_analysis_lab.core.errors import (
    ArtifactError,
    UnsupportedArtifactOperationError,
    UnsupportedArtifactTypeError,
)
from text_analysis_lab.core.lineage import artifact_lineage, basis_artifact_ids
from text_analysis_lab.core.types import (
    ArtifactType,
    ColumnSelect,
    MetadataMode,
    StreamingMode,
)

try:
    from text_analysis_lab.core.types import StructuralColumn
except ImportError:  # Compatibility with the pre-StructuralColumn version.
    StructuralColumn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from scipy import sparse as scipy_sparse

    from text_analysis_lab.core.kwic import KWICResult
    from text_analysis_lab.core.project import Project


if StructuralColumn is None:
    STRUCTURAL_COLUMNS: frozenset[str] = frozenset(
        {"_position", "_batch", "_row_offset"}
    )
else:
    STRUCTURAL_COLUMNS = frozenset(str(value) for value in get_args(StructuralColumn))


def _positions_as_list(positions: Sequence[int]) -> list[int]:
    return [int(position) for position in positions]


def _requested_data_columns(
    available: Sequence[str],
    data_columns: ColumnSelect,
) -> list[str]:
    """Resolve a data-column request against known data columns."""
    available = [str(col) for col in available]

    if data_columns is True:
        return available
    if data_columns is False:
        return []
    if isinstance(data_columns, str):
        requested = [data_columns]
    else:
        requested = [str(col) for col in data_columns]

    available_set = set(available)
    missing = [col for col in requested if col not in available_set]
    if missing:
        raise ArtifactError(f"Requested data column(s) are not available: {missing}.")
    return requested


def _normalize_records(
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Project records onto columns, filling missing values with None."""
    columns = [str(col) for col in columns]
    return [{col: record.get(col, None) for col in columns} for record in records]


def _records_frame(
    records: Sequence[dict[str, Any]], columns: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(records, columns=[str(col) for col in columns]).reset_index(
        drop=True
    )


def _matrix_column_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    if "column" in frame.columns:
        return [str(value) for value in frame["column"].tolist()]
    return [str(value) for value in frame.iloc[:, -1].tolist()]


def _first_part(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    return next(iter(sorted(path.glob(f"part-*.{suffix}"))), None)


def _locations_for_positions(
    artifact: BaseArtifact, positions: Sequence[int]
) -> list[Any]:
    if not positions:
        return []
    return artifact.project.query.locations_by_positions(artifact, positions)


def _group_offsets_by_batch(locations: Sequence[Any]) -> dict[int, dict[int, int]]:
    """Return {batch: {row_offset: position}} for Location-like records."""
    wanted: dict[int, dict[int, int]] = {}
    for location in locations:
        batch = int(location.batch)
        offset = int(location.row_offset)
        position = int(location.position)
        wanted.setdefault(batch, {})[offset] = position
    return wanted


def _validated_feature_indices(
    raw_indices: Any,
    *,
    source_width: int,
    artifact_id: str,
) -> list[int]:
    if isinstance(raw_indices, np.ndarray):
        values = raw_indices.tolist()
    elif isinstance(raw_indices, Sequence) and not isinstance(
        raw_indices, (str, bytes)
    ):
        values = list(raw_indices)
    else:
        raise ArtifactError(
            f"Matrix artifact {artifact_id} records invalid feature_indices."
        )
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ArtifactError(
                f"Matrix artifact {artifact_id} feature_indices must be integers."
            )
        index = int(value)
        if index < 0 or index >= int(source_width):
            raise ArtifactError(
                f"Matrix artifact {artifact_id} feature index {index} is outside "
                f"source width {source_width}."
            )
        indices.append(index)
    if not indices:
        raise ArtifactError(
            f"Matrix artifact {artifact_id} feature_indices may not be empty."
        )
    if any(right <= left for left, right in pairwise(indices)):
        raise ArtifactError(
            f"Matrix artifact {artifact_id} feature_indices must be unique and "
            "strictly increasing."
        )
    return indices


class TableArtifact(BaseArtifact):
    """Row-addressable tablular artifact."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.TABLE

    def get_data_columns(self) -> list[str]:
        return list(self.query_columns(metadata_mode="none")["data"])

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        return self.query(
            key_columns=False,
            data_columns=data_columns,
            metadata_columns=False,
            metadata_mode="none",
            positions=_positions_as_list(positions),
            form="table",
            include_position=False,
        )

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        records = self.query(
            key_columns=False,
            data_columns=data_columns,
            metadata_columns=False,
            metadata_mode="none",
            positions=_positions_as_list(positions),
            form="records",
            include_position=False,
        )
        return [dict(record) for record in records]

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        return self._own_data_frame_for_positions(positions, data_columns=data_columns)

    def _native_from_info_and_data(self, info: pd.DataFrame, data: Any) -> pd.DataFrame:
        data_frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return _merge_info_and_data_frames(
            info.reset_index(drop=True),
            data_frame.reset_index(drop=True),
            include_position=True,
        )


class JsonlArtifact(BaseArtifact):
    """Row-addressable artifact with JSONL representation data."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.JSONL

    def get_data_columns(self) -> list[str]:
        columns = self.components.get("data", {}).get("columns")
        if columns is None:
            raise ArtifactError(
                f"JSON artifact {self.artifact_id} does not record data columns."
            )
        return [str(col) for col in columns]

    def _jsonl_records_for_locations(
        self, locations: Sequence[Any]
    ) -> dict[int, dict[str, Any]]:
        wanted_by_batch = _group_offsets_by_batch(locations)
        records_by_position: dict[int, dict[str, Any]] = {}

        for batch, offset_to_position in wanted_by_batch.items():
            path = self.storage.data_part_path(batch, "jsonl")
            if not path.exists():
                continue

            remaining = set(offset_to_position)
            with path.open("r", encoding="utf-8") as handle:
                for offset, line in enumerate(handle):
                    if offset not in remaining:
                        continue
                    raw = json.loads(line)
                    position = offset_to_position[offset]
                    records_by_position[position] = {
                        str(key): value
                        for key, value in dict(raw).items()
                        if str(key) not in STRUCTURAL_COLUMNS
                    }
                    remaining.remove(offset)
                    if not remaining:
                        break

        return records_by_position

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        positions = _positions_as_list(positions)
        columns = _requested_data_columns(self.get_data_columns(), data_columns)
        if not positions:
            return []
        if not columns:
            return [{} for _ in positions]

        locations = _locations_for_positions(self, positions)
        records_by_position = self._jsonl_records_for_locations(locations)
        records = [records_by_position.get(position, {}) for position in positions]
        return _normalize_records(records, columns)

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        columns = _requested_data_columns(self.get_data_columns(), data_columns)
        records = self._own_data_records_for_positions(positions, data_columns=columns)
        return _records_frame(records, columns)

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        return self._own_data_records_for_positions(
            positions, data_columns=data_columns
        )

    def _native_from_info_and_data(
        self, info: pd.DataFrame, data: Any
    ) -> list[dict[str, Any]]:
        data_records = [dict(record) for record in data]
        data_columns = list(data_records[0]) if data_records else []
        return _merge_info_and_data_records(
            info.reset_index(drop=True),
            data_records,
            data_columns=data_columns,
            include_position=True,
        )


class _MatrixArtifact(BaseArtifact):
    """Shared matrix representation helpers.

    Matrix feature axes are positional. get_feature_metadata() annotates those
    positions, while lazy feature views store only source-relative integer
    indices in their lineage. Matrix-value ownership and Feature Metadata
    ownership are independent: a transform may own new values while inheriting
    the exact feature axis from a basis matrix.
    """

    value_suffix: ClassVar[str]

    def _own_feature_metadata(self) -> pd.DataFrame:
        """Return Feature Metadata physically owned by this matrix artifact."""
        path = self.storage.feature_metadata_path
        if path.exists():
            frame = pd.read_parquet(path).reset_index(drop=True)
            if "column_index" not in frame.columns:
                frame.insert(0, "column_index", np.arange(len(frame), dtype="int64"))
            return frame
        labels = [str(i) for i in range(self._infer_n_columns())]
        return pd.DataFrame(
            {
                "column_index": np.arange(len(labels), dtype="int64"),
                "column": labels,
            }
        )

    def _feature_metadata_basis(self) -> _MatrixArtifact:
        """Return the single matrix basis for inherited Feature Metadata."""
        lineage = artifact_lineage(self)
        explicit_basis_id = lineage.get("feature_metadata_basis_artifact_id")
        if explicit_basis_id is not None:
            basis_ids = [str(explicit_basis_id)]
        else:
            basis_ids = basis_artifact_ids(self)
        if len(basis_ids) != 1:
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} requires exactly one basis "
                "artifact for inherited Feature Metadata."
            )
        basis = self.project.get_artifact(basis_ids[0])
        if not isinstance(basis, _MatrixArtifact):
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} cannot inherit Feature Metadata "
                f"from non-matrix artifact {basis.artifact_id}."
            )
        return basis

    def get_feature_metadata(self) -> pd.DataFrame:
        """Return Feature Metadata in this artifact's local feature order.

        New artifacts record whether they own or inherit Feature Metadata.
        Older artifacts remain readable: value-owning matrices are treated as
        metadata owners, while keys-only/lazy matrix views inherit from their
        single basis artifact. Positional feature projections are applied by
        index, never by matching feature labels.
        """
        lineage = artifact_lineage(self)
        mode = lineage.get("feature_metadata_mode")
        if mode is None:
            inherit = not self.has_own_data()
        elif mode == "own":
            inherit = False
        elif mode == "inherit":
            inherit = True
        else:
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} records invalid "
                f"feature_metadata_mode={mode!r}."
            )

        if inherit:
            basis = self._feature_metadata_basis()
            frame = basis.get_feature_metadata()
            raw_indices = lineage.get("feature_indices")
            if raw_indices is not None:
                indices = _validated_feature_indices(
                    raw_indices,
                    source_width=len(frame),
                    artifact_id=self.artifact_id,
                )
                frame = frame.iloc[indices].reset_index(drop=True)
            else:
                frame = frame.reset_index(drop=True)
        else:
            if not self.has_own_data():
                raise ArtifactError(
                    f"Matrix artifact {self.artifact_id} declares owned Feature "
                    "Metadata but has no owned matrix data."
                )
            frame = self._own_feature_metadata()

        frame = frame.copy()
        if "column_index" in frame.columns:
            frame["column_index"] = np.arange(len(frame), dtype="int64")
        else:
            frame.insert(0, "column_index", np.arange(len(frame), dtype="int64"))

        if self.has_own_data():
            width = self._infer_n_columns()
            if len(frame) != width:
                raise ArtifactError(
                    f"Matrix artifact {self.artifact_id} Feature Metadata width "
                    f"{len(frame)} does not match matrix width {width}."
                )
        return frame

    def get_feature_frame(self) -> pd.DataFrame:
        """Backward-compatible alias for get_feature_metadata()."""
        return self.get_feature_metadata()

    def get_data_columns(self) -> list[str]:
        frame = self.get_feature_metadata()
        if frame.empty:
            return []
        if "column" in frame.columns:
            return [str(value) for value in frame["column"].tolist()]
        return [str(value) for value in frame.iloc[:, -1].tolist()]

    def _feature_indices_to_data_artifact(self) -> list[int]:
        """Map this local feature axis onto the physical data owner's columns."""
        if self.has_own_data():
            return list(range(self._infer_n_columns()))

        lineage = artifact_lineage(self)
        if str(lineage.get("lineage_mode")) not in {"preserved_key", "rekeyed_key"}:
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} cannot inherit matrix features "
                "through this lineage mode."
            )
        basis_ids = basis_artifact_ids(self)
        if len(basis_ids) != 1:
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} requires exactly one basis artifact "
                "for inherited matrix data."
            )
        basis = self.project.get_artifact(basis_ids[0])
        if not isinstance(basis, _MatrixArtifact):
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} cannot inherit matrix features "
                f"from non-matrix artifact {basis.artifact_id}."
            )
        basis_indices = basis._feature_indices_to_data_artifact()
        raw_indices = lineage.get("feature_indices")
        if raw_indices is None:
            return basis_indices
        local = _validated_feature_indices(
            raw_indices,
            source_width=len(basis_indices),
            artifact_id=self.artifact_id,
        )
        return [basis_indices[index] for index in local]

    def _local_legacy_row_name_component(self) -> Mapping[str, Any] | None:
        """Return the pre-row_name_column named-row descriptor, if present."""
        data = self.components.get("data", {})
        if not isinstance(data, Mapping):
            return None
        component = data.get("row_names")
        return component if isinstance(component, Mapping) else None

    def _row_name_source(self) -> tuple["_MatrixArtifact", str, bool] | None:
        """Resolve the artifact and storage scheme that own named-row semantics."""
        value = self.descriptor.get("row_name_column")
        if isinstance(value, str) and value:
            return self, value, False

        legacy = self._local_legacy_row_name_component()
        if legacy is not None:
            name = legacy.get("name")
            if isinstance(name, str) and name:
                return self, name, True

        lineage = artifact_lineage(self)
        if str(lineage.get("lineage_mode")) not in {"preserved_key", "rekeyed_key"}:
            return None
        basis_ids = basis_artifact_ids(self)
        if len(basis_ids) != 1:
            return None
        basis = self.project.get_artifact(basis_ids[0])
        if not isinstance(basis, _MatrixArtifact):
            return None
        return basis._row_name_source()

    def _row_name_column(self) -> str | None:
        """Resolve the privileged row-name field through row lineage."""
        source = self._row_name_source()
        return None if source is None else source[1]

    @property
    def has_row_names(self) -> bool:
        """Whether this matrix defines a unique human-readable row axis."""
        return self._row_name_source() is not None

    @property
    def row_name(self) -> str | None:
        """Return the privileged row-name field, such as ``"word"``."""
        return self._row_name_column()

    def _legacy_row_names_for_positions(
        self, positions: Sequence[int]
    ) -> list[str]:
        """Read named rows from the legacy data/row_names parquet dataset."""
        legacy = self._local_legacy_row_name_component()
        if legacy is None:
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} has no legacy row-name component."
            )
        raw_path = legacy.get("path", "data/row_names/")
        legacy_dir = self.storage.artifact_dir / str(raw_path)
        locations = _locations_for_positions(self, positions)
        wanted_by_batch = _group_offsets_by_batch(locations)
        names_by_position: dict[int, str] = {}
        for batch, offset_to_position in wanted_by_batch.items():
            path = legacy_dir / self.storage.part_name(batch, "parquet")
            if not path.exists():
                raise ArtifactError(f"Missing legacy matrix row_names part: {path}")
            frame = pd.read_parquet(path, columns=["row_name"])
            for offset, position in offset_to_position.items():
                names_by_position[position] = str(frame.iloc[int(offset)]["row_name"])
        return [names_by_position[int(position)] for position in positions]

    def get_row_names(self, *, positions: Sequence[int] | None = None) -> list[str]:
        """Return privileged row names, including legacy named-row artifacts."""
        source = self._row_name_source()
        if source is None:
            raise UnsupportedArtifactOperationError(
                f"Matrix artifact {self.artifact_id} does not define row names."
            )
        if positions is None:
            positions = list(range(int(self.n_rows or 0)))
        resolved = _positions_as_list(positions)
        if not resolved:
            return []

        source_artifact, row_name_column, legacy = source
        if legacy:
            source_positions = (
                resolved
                if source_artifact is self
                else self.project.query.map_descendant_positions_to_ancestor_positions(
                    self,
                    source_artifact,
                    resolved,
                )
            )
            return source_artifact._legacy_row_names_for_positions(source_positions)

        query_info = self.query_columns(metadata_mode="full")
        matches = [
            column
            for column in query_info.get("columns", ())
            if str(column.get("namespace")) == "metadata"
            and str(column.get("base_name")) == row_name_column
            and str(column.get("source_artifact_id"))
            == str(source_artifact.artifact_id)
        ]
        if len(matches) != 1:
            raise ArtifactError(
                f"Matrix row-name metadata column {row_name_column!r} does not "
                f"resolve uniquely for artifact {self.artifact_id}."
            )
        output_name = str(matches[0]["output_name"])
        frame = self.query(
            key_columns=False,
            data_columns=False,
            metadata_columns=[output_name],
            metadata_mode="full",
            positions=resolved,
            form="table",
            include_position=False,
        )
        if not isinstance(frame, pd.DataFrame) or output_name not in frame.columns:
            raise ArtifactError(
                f"Matrix row-name metadata column {row_name_column!r} could not be resolved."
            )
        return [str(value) for value in frame[output_name].tolist()]

    def positions_by_row_names(self, row_names: Sequence[str]) -> list[int]:
        """Resolve unique row names to artifact-local integer positions."""
        requested = [str(value) for value in row_names]
        if not requested:
            return []
        index = getattr(self, "_row_name_position_index", None)
        if index is None:
            all_names = self.get_row_names()
            index = {name: position for position, name in enumerate(all_names)}
            self._row_name_position_index = index
        missing = [name for name in requested if name not in index]
        if missing:
            examples = missing[:5]
            raise KeyError(
                f"Unknown matrix row name(s) for {self.artifact_id}: {examples}."
            )
        return [int(index[name]) for name in requested]

    def position_by_row_name(self, row_name: str) -> int:
        """Resolve one unique row name to its artifact-local position."""
        return self.positions_by_row_names([str(row_name)])[0]

    def get_rows_by_name(
        self,
        row_names: Sequence[str],
        *,
        data_columns: ColumnSelect = True,
    ) -> Any:
        """Return named rows in requested order using the matrix's native form."""
        return self.get_matrix(
            data_columns=data_columns,
            positions=self.positions_by_row_names(row_names),
        )

    def get_row_by_name(
        self, row_name: str, *, data_columns: ColumnSelect = True
    ) -> Any:
        """Return one named row as a one-row native matrix."""
        return self.get_rows_by_name([str(row_name)], data_columns=data_columns)

    def _infer_n_columns(self) -> int:
        raise NotImplementedError

    def _column_indices(
        self, data_columns: ColumnSelect
    ) -> tuple[list[str], list[int]]:
        available = self.get_data_columns()
        selected = _requested_data_columns(available, data_columns)
        index_by_name = {name: index for index, name in enumerate(available)}
        return selected, [index_by_name[name] for name in selected]

    def _physical_column_indices(
        self, data_columns: ColumnSelect
    ) -> tuple[list[str], list[int]]:
        """Resolve a local column request to physical data-owner positions."""
        selected, local_indices = self._column_indices(data_columns)
        physical_map = self._feature_indices_to_data_artifact()
        if len(physical_map) != len(self.get_data_columns()):
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} feature projection width is inconsistent."
            )
        return selected, [physical_map[index] for index in local_indices]

    def _resolve_external_data_columns(self, data_columns: ColumnSelect) -> list[str]:
        if data_columns is False:
            return []
        return _requested_data_columns(self.get_data_columns(), data_columns)

    def _data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> Any:
        if data_columns is False:
            return None
        _, physical_indices = self._physical_column_indices(data_columns)
        data_positions = self._resolve_data_positions(positions)
        data_artifact = self._require_data_artifact()
        if not isinstance(data_artifact, _MatrixArtifact):
            raise ArtifactError(
                f"Matrix artifact {self.artifact_id} resolved a non-matrix data owner."
            )
        return data_artifact._own_data_native_for_positions_by_indices(
            data_positions,
            column_indices=physical_indices,
        )

    def _data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        selected, _ = self._physical_column_indices(data_columns)
        matrix = self._data_native_for_positions(positions, data_columns=data_columns)
        if self.artifact_type == ArtifactType.SPARSE_MATRIX:
            return pd.DataFrame.sparse.from_spmatrix(
                matrix,
                columns=selected,
            ).reset_index(drop=True)
        return pd.DataFrame(matrix, columns=selected).reset_index(drop=True)

    def _data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        selected, _ = self._physical_column_indices(data_columns)
        matrix = self._data_native_for_positions(positions, data_columns=data_columns)
        if self.artifact_type == ArtifactType.SPARSE_MATRIX:
            csr = matrix.tocsr()
            records: list[dict[str, Any]] = []
            for row_index in range(int(csr.shape[0])):
                row = csr.getrow(row_index)
                records.append(
                    {
                        selected[int(col_index)]: (
                            value.item() if hasattr(value, "item") else value
                        )
                        for col_index, value in zip(row.indices, row.data, strict=True)
                    }
                )
            return records
        dense = np.asarray(matrix)
        return [
            {selected[index]: row[index] for index in range(len(selected))}
            for row in dense
        ]

    def _native_from_info_and_data(
        self, info: pd.DataFrame, data: Any
    ) -> dict[str, Any]:
        result = {"info": info.reset_index(drop=True), "matrix": data}
        columns = self.get_data_columns()
        shape = getattr(data, "shape", ())
        if len(shape) == 2 and int(shape[1]) == len(columns):
            result["columns"] = columns
        if self.has_row_names and "_position" in info.columns:
            result["row_names"] = self.get_row_names(
                positions=info["_position"].astype(int).tolist()
            )
        return result

    def kwic(
        self,
        pattern: str,
        *,
        window: int = 5,
        before: int | None = None,
        after: int | None = None,
        valuetype: Literal["fixed", "regex"] = "fixed",
        case_sensitive: bool = False,
        enforce_word_boundary: bool = True,
        key_columns: ColumnSelect = True,
        data_columns: ColumnSelect = True,
        metadata_columns: ColumnSelect = False,
        metadata_mode: MetadataMode = "none",
        where: str | None = None,
        order_by: str | Sequence[str] | None = None,
        positions: Sequence[int] | None = None,
        sample_n: int | None = None,
        sample_frac: float | None = None,
        random_state: int | None = None,
        limit: int | None = None,
        batch_size: int = 100_000,
        streaming_mode: StreamingMode = "auto",
        search_columns: ColumnSelect | None = None,
        target_matches: int | None = None,
    ) -> KWICResult:
        raise UnsupportedArtifactOperationError(
            f"KWIC is not supported for {self.__class__.__name__}."
        )


class DenseMatrixArtifact(_MatrixArtifact):
    """Row-addressable dense matrix artifact."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.DENSE_MATRIX
    value_suffix: ClassVar[str] = "npy"

    def _infer_n_columns(self) -> int:
        path = _first_part(self.storage.data_values_dir, self.value_suffix)
        if path is None:
            return 0
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2:
            raise ArtifactError(f"Dense matrix part {path} is not two-dimensional.")
        return int(values.shape[1])

    def _load_batch_rows(self, batch: int, row_offsets: Sequence[int]) -> np.ndarray:
        path = self.storage.data_value_part_path(batch, self.value_suffix)
        if not path.exists():
            raise ArtifactError(f"Missing dense matrix data part: {path}")
        values = np.load(path, mmap_mode="r")
        return np.asarray(values[[int(offset) for offset in row_offsets], :])

    def _own_data_native_for_positions_by_indices(
        self,
        positions: Sequence[int],
        *,
        column_indices: Sequence[int],
    ) -> np.ndarray:
        positions = _positions_as_list(positions)
        resolved_columns = [int(index) for index in column_indices]
        if not positions:
            return np.empty((0, len(resolved_columns)))

        locations = _locations_for_positions(self, positions)
        wanted_by_batch = _group_offsets_by_batch(locations)
        rows_by_position: dict[int, np.ndarray] = {}

        for batch, offset_to_position in wanted_by_batch.items():
            offsets = list(offset_to_position)
            rows = self._load_batch_rows(batch, offsets)
            rows = rows[:, resolved_columns] if resolved_columns else rows[:, []]
            for row_index, offset in enumerate(offsets):
                rows_by_position[offset_to_position[offset]] = np.asarray(
                    rows[row_index]
                )

        if not rows_by_position:
            return np.empty((0, len(resolved_columns)))
        return np.vstack([rows_by_position[position] for position in positions])

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> np.ndarray:
        _, column_indices = self._column_indices(data_columns)
        return self._own_data_native_for_positions_by_indices(
            positions,
            column_indices=column_indices,
        )

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        selected_columns, _ = self._column_indices(data_columns)
        matrix = self._own_data_native_for_positions(
            positions, data_columns=selected_columns
        )
        return pd.DataFrame(matrix, columns=selected_columns).reset_index(drop=True)

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        frame = self._own_data_frame_for_positions(positions, data_columns=data_columns)
        return [
            {str(key): value for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]

    def get_matrix(
        self,
        *,
        data_columns: ColumnSelect = True,
        positions: Sequence[int] | None = None,
    ) -> np.ndarray:
        result = self.query(
            key_columns=False,
            data_columns=data_columns,
            metadata_columns=False,
            metadata_mode="none",
            positions=positions,
            form="native",
            include_position=False,
        )
        return result["matrix"]


class SparseMatrixArtifact(_MatrixArtifact):
    """Row-addressable sparse matrix artifact."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.SPARSE_MATRIX
    value_suffix: ClassVar[str] = "npz"

    def _sparse_module(self):
        try:
            from scipy import sparse
        except ImportError as exc:  # pragma: no cover
            raise UnsupportedArtifactOperationError(
                "SparseMatrixArtifact requires scipy."
            ) from exc
        return sparse

    def _infer_n_columns(self) -> int:
        path = _first_part(self.storage.data_values_dir, self.value_suffix)
        if path is None:
            return 0
        matrix = self._sparse_module().load_npz(path).tocsr()
        return int(matrix.shape[1])

    def _load_batch_rows(
        self, batch: int, row_offsets: Sequence[int]
    ) -> scipy_sparse.csr_matrix:
        path = self.storage.data_value_part_path(batch, self.value_suffix)
        if not path.exists():
            raise ArtifactError(f"Missing sparse matrix data part: {path}")
        matrix = self._sparse_module().load_npz(path).tocsr()
        return matrix[[int(offset) for offset in row_offsets], :]

    def _own_data_native_for_positions_by_indices(
        self,
        positions: Sequence[int],
        *,
        column_indices: Sequence[int],
    ) -> scipy_sparse.csr_matrix:
        sparse = self._sparse_module()
        positions = _positions_as_list(positions)
        resolved_columns = [int(index) for index in column_indices]
        if not positions:
            return sparse.csr_matrix((0, len(resolved_columns)))

        locations = _locations_for_positions(self, positions)
        wanted_by_batch = _group_offsets_by_batch(locations)
        rows_by_position: dict[int, Any] = {}

        for batch, offset_to_position in wanted_by_batch.items():
            offsets = list(offset_to_position)
            rows = self._load_batch_rows(batch, offsets)
            rows = rows[:, resolved_columns] if resolved_columns else rows[:, []]
            for row_index, offset in enumerate(offsets):
                rows_by_position[offset_to_position[offset]] = rows[row_index]

        if not rows_by_position:
            return sparse.csr_matrix((0, len(resolved_columns)))

        return cast(
            "scipy_sparse.csr_matrix",
            sparse.vstack(
                [rows_by_position[position] for position in positions],
                format="csr",
            ),
        )

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> scipy_sparse.csr_matrix:
        _, column_indices = self._column_indices(data_columns)
        return self._own_data_native_for_positions_by_indices(
            positions,
            column_indices=column_indices,
        )

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        selected_columns, _ = self._column_indices(data_columns)
        matrix = self._own_data_native_for_positions(
            positions, data_columns=selected_columns
        )
        return pd.DataFrame.sparse.from_spmatrix(
            matrix,
            columns=selected_columns,
        ).reset_index(drop=True)

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        selected_columns, _ = self._column_indices(data_columns)
        matrix = self._own_data_native_for_positions(
            positions,
            data_columns=selected_columns,
        ).tocsr()
        shape = cast("tuple[int, ...]", matrix.shape)
        records: list[dict[str, Any]] = []
        for row_index in range(shape[0]):
            row = matrix.getrow(row_index)
            records.append(
                {
                    selected_columns[int(col_index)]: (
                        value.item() if hasattr(value, "item") else value
                    )
                    for col_index, value in zip(row.indices, row.data, strict=True)
                }
            )
        return records

    def get_matrix(
        self,
        *,
        data_columns: ColumnSelect = True,
        positions: Sequence[int] | None = None,
    ) -> scipy_sparse.csr_matrix:
        result = self.query(
            key_columns=False,
            data_columns=data_columns,
            metadata_columns=False,
            metadata_mode="none",
            positions=positions,
            form="native",
            include_position=False,
        )
        return result["matrix"]


class OtherArtifact(BaseArtifact):
    """Placeholder artifact for unsupported/custom representation data."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.OTHER

    def get_data_columns(self) -> list[str]:
        raise UnsupportedArtifactOperationError(
            "OtherArtifact does not expose standardized data columns."
        )

    def _own_data_records_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> list[dict[str, Any]]:
        raise UnsupportedArtifactOperationError(
            "OtherArtifact does not expose standardized data records."
        )

    def _own_data_frame_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> pd.DataFrame:
        raise UnsupportedArtifactOperationError(
            "OtherArtifact does not expose standardized data frames."
        )

    def _own_data_native_for_positions(
        self,
        positions: Sequence[int],
        *,
        data_columns: ColumnSelect = True,
    ) -> Any:
        raise UnsupportedArtifactOperationError(
            "OtherArtifact does not expose standardized native data."
        )

    def _native_from_info_and_data(self, info: pd.DataFrame, data: Any) -> Any:
        raise UnsupportedArtifactOperationError(
            "OtherArtifact does not expose standardized native data."
        )


def _artifact_registry(
    classes: Iterable[type[BaseArtifact]],
) -> dict[ArtifactType, type[BaseArtifact]]:
    registry: dict[ArtifactType, type[BaseArtifact]] = {}
    for cls in classes:
        artifact_type = cls.artifact_type
        if artifact_type in registry:
            raise RuntimeError(
                f"Duplicate artifact type {artifact_type!r}: "
                f"{registry[artifact_type].__name__} and {cls.__name__}."
            )
        registry[artifact_type] = cls
    return registry


ARTIFACT_CLASSES: tuple[type[BaseArtifact], ...] = (
    TableArtifact,
    JsonlArtifact,
    SparseMatrixArtifact,
    DenseMatrixArtifact,
    OtherArtifact,
)

ARTIFACT_TYPE_REGISTRY: dict[ArtifactType, type[BaseArtifact]] = _artifact_registry(
    ARTIFACT_CLASSES
)


def load_artifact(project: Project, artifact_dir: str | Path) -> BaseArtifact:
    """Load an artifact handle from an artifact directory."""
    artifact_path = Path(artifact_dir)
    descriptor = _read_json(artifact_path / "artifact.json")

    try:
        artifact_type = ArtifactType(descriptor["artifact_type"])
    except KeyError as exc:
        raise UnsupportedArtifactTypeError(
            f"Artifact descriptor at {artifact_path} does not include artifact_type."
        ) from exc
    except ValueError as exc:
        raise UnsupportedArtifactTypeError(
            f"Unsupported artifact type: {descriptor.get('artifact_type')!r}."
        ) from exc

    cls = ARTIFACT_TYPE_REGISTRY.get(artifact_type)
    if cls is None:
        raise UnsupportedArtifactTypeError(
            f"Unsupported artifact type: {artifact_type!r}."
        )

    artifact = cls(project, artifact_path)
    if artifact.artifact_type != artifact_type:
        raise UnsupportedArtifactTypeError(
            f"Descriptor artifact_type {artifact_type!r} does not match "
            f"{cls.__name__}.artifact_type {artifact.artifact_type!r}."
        )
    return artifact
