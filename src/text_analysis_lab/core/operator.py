"""Base operator contracts for TextAnalysisLab (TeAL).

An operator is a reusable-after-freezing execution specification. One operation
is one execution event of an operator; operation records live in the project
catalog.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, get_args

from text_analysis_lab.core.errors import (
    FrozenOperatorError,
    OperatorError,
    OutputSpecError,
)
from text_analysis_lab.core.lineage import validate_lineage_mode
from text_analysis_lab.core.types import (
    DEFAULT_OUTPUT_LABEL,
    ArtifactType,
    ColumnSelect,
    LineageMode,
    MetadataMode,
    OperationType,
    QueryForm,
)

if TYPE_CHECKING:
    from text_analysis_lab.core.artifact_base import BaseArtifact


OtherDataSerializer = Callable[[Any, Path, int], Mapping[str, Any] | None]

TranslationMode = Literal["translate", "fit_translate"]
RunRoute = Literal["sequential", "parallel"]
SourceMode = Literal["batches", "full_artifact"]
WriterPayload = Mapping[str, Any]
OutputMap = Mapping[str, WriterPayload]


@dataclass(frozen=True)
class ColumnRequest:
    """Column roles to request from a source artifact."""

    keys: ColumnSelect = True
    metadata: ColumnSelect = False
    data: ColumnSelect = True


@dataclass(frozen=True)
class SourceRequest:
    """Accepted source artifact type and materialization request."""

    artifact_type: ArtifactType | str | Sequence[ArtifactType | str]
    mode: SourceMode = "batches"
    columns: ColumnRequest = field(default_factory=ColumnRequest)
    batch_size: int | None = None
    form: QueryForm = "native"
    metadata_mode: MetadataMode = "none"
    include_position: bool = False


@dataclass(frozen=True)
class TranslationRequest:
    """User execution requests passed through Project.translate(...).

    ``params`` stores operation-specific run parameters supplied to
    ``Project.translate(...)``. These are not frozen operator state; they describe
    this operation's binding of the operator to concrete source columns/options.
    """

    workers: int = 1
    batch_size: int | None = None
    max_outstanding_units: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputBatch:
    """One materialized source packet passed to a translator.

    ``batch_index`` and ``batch_count`` are source-local: they describe this
    packet's position in the planned stream for ``source_label``. A full-artifact
    source is represented as the single packet for that source, so both
    ``is_first`` and ``is_last`` are true.
    """

    source_label: str
    artifact_id: str
    primary_key: tuple[str, ...]
    data: Any
    batch_index: int
    batch_count: int
    is_first: bool
    is_last: bool


@dataclass(frozen=True)
class BatchResult:
    """Return value from translator.translate_batch(...)."""

    outputs: OutputMap = field(default_factory=dict)
    value: Any = None


InputRequest = Mapping[str, SourceRequest]


OPERATION_TYPES: tuple[str, ...] = get_args(OperationType)


@dataclass(frozen=True)
class OutputSpec:
    """Declare one artifact produced by an operator.

    ``basis_labels`` name another output from the same operation. When it is
    ``None``, execution resolves the output basis from the operation's external
    source artifacts.
    """

    artifact_type: ArtifactType | str = ArtifactType.TABLE
    lineage_mode: LineageMode = "new_key"
    basis_labels: str | Sequence[str] | None = None
    data_serializer: OtherDataSerializer | None = None
    data_serializer_ref: Mapping[str, str] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.basis_labels is None:
            basis_labels: tuple[str, ...] = ()
        elif isinstance(self.basis_labels, str):
            if not self.basis_labels:
                raise OutputSpecError(
                    "OutputSpec.basis_labels cannot contain empty labels."
                )
            basis_labels = (self.basis_labels,)
        elif isinstance(self.basis_labels, Sequence):
            basis_labels = tuple(self.basis_labels)
            if any(not isinstance(label, str) or not label for label in basis_labels):
                raise OutputSpecError(
                    "OutputSpec.basis_labels must contain only non-empty strings."
                )

        object.__setattr__(self, "basis_labels", basis_labels)

        try:
            artifact_type = ArtifactType(self.artifact_type)
        except ValueError as exc:
            supported = ", ".join(item.value for item in ArtifactType)
            raise OutputSpecError(
                f"Unsupported output artifact_type={self.artifact_type!r}. "
                f"Supported: {supported}."
            ) from exc

        if artifact_type == ArtifactType.OTHER:
            if self.data_serializer is None:
                raise OutputSpecError(
                    "ArtifactType.OTHER outputs require OutputSpec.data_serializer."
                )
            serializer_ref = _callable_ref(
                self.data_serializer, name="OutputSpec.data_serializer"
            )
        else:
            # Serializers are meaningful only for OTHER artifacts. For ordinary
            # TeAL artifact types, the writer knows how to serialize the payload.
            serializer_ref = None
            if self.data_serializer is not None:
                object.__setattr__(self, "data_serializer", None)

        object.__setattr__(self, "data_serializer_ref", serializer_ref)

        lineage_mode = validate_lineage_mode(str(self.lineage_mode))
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "lineage_mode", lineage_mode)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe descriptor form of this output spec.

        ``data_serializer`` is executable Python state. For OTHER outputs,
        ``__post_init__`` validates it once and stores its importable
        module/qualname reference in ``data_serializer_ref`` for descriptors,
        writer construction, and resume.
        """
        artifact_type = ArtifactType(self.artifact_type)
        lineage_mode = validate_lineage_mode(str(self.lineage_mode))
        return {
            "artifact_type": artifact_type.value,
            "lineage_mode": lineage_mode,
            "basis_labels": self.basis_labels,
            "data_serializer": self.data_serializer_ref,
        }


OutputSpecDeclaration = OutputSpec | Mapping[str, OutputSpec]


def validate_operation_type(operation_type: OperationType | str) -> OperationType:
    """Validate and return an operation type value."""
    value = str(operation_type)
    if value not in OPERATION_TYPES:
        raise OperatorError(
            f"Invalid operation_type {value!r}. Expected one of {list(OPERATION_TYPES)}."
        )
    return cast(OperationType, value)


def validate_output_label(label: str) -> str:
    """Validate and return an output label."""
    if not isinstance(label, str) or not label:
        raise OutputSpecError("Output labels must be non-empty strings.")
    return label


def normalize_output_specs(
    output_specs: OutputSpecDeclaration,
) -> dict[str, OutputSpec]:
    """Normalize one spec or a label->spec mapping to a dictionary.

    A single ``OutputSpec`` normalizes to ``{"output": spec}``.
    """
    if isinstance(output_specs, OutputSpec):
        return {DEFAULT_OUTPUT_LABEL: output_specs}

    if not isinstance(output_specs, Mapping):
        raise OutputSpecError(
            "output_specs() must return an OutputSpec or a mapping of label to OutputSpec."
        )

    specs: dict[str, OutputSpec] = {}
    for raw_label, spec in output_specs.items():
        label = validate_output_label(raw_label)
        if not isinstance(spec, OutputSpec):
            raise OutputSpecError(
                f"Output {label!r} must be an OutputSpec; got {type(spec).__name__}."
            )
        specs[label] = spec

    if not specs:
        raise OutputSpecError("An operator must declare at least one OutputSpec.")

    return specs


def validate_output_specs(output_specs: OutputSpecDeclaration) -> dict[str, OutputSpec]:
    """Validate output declarations and return normalized label->spec pairs.

    Validation checks declaration-level structure and output-to-output lineage
    relationships. ``basis_labels`` may also name input source labels, which are
    known only by the execution runner after ``input_request(...)`` is resolved.
    Execution therefore performs the final cross-namespace basis validation.
    """
    specs = normalize_output_specs(output_specs)

    for label, spec in specs.items():
        basis_labels = cast(tuple[str, ...], spec.basis_labels)
        if label in basis_labels:
            raise OutputSpecError(f"Output {label!r} cannot use itself as basis_label.")

    visited: set[str] = set()

    def visit(label: str, stack: list[str]) -> None:
        if label in visited:
            return
        if label in stack:
            cycle = stack[stack.index(label) :] + [label]
            raise OutputSpecError(
                f"Cycle in output basis_labels dependencies: {' -> '.join(cycle)}."
            )

        stack.append(label)
        for basis_label in cast(tuple[str, ...], specs[label].basis_labels):
            if basis_label in specs:
                visit(basis_label, stack)
        stack.pop()
        visited.add(label)

    for label in specs:
        visit(label, [])

    return specs


def output_specs_in_dependency_order(
    output_specs: OutputSpecDeclaration,
) -> tuple[tuple[str, OutputSpec], ...]:
    """Return output specs with basis outputs before dependent outputs.

    Basis labels that are not output labels are treated as source labels and do
    not participate in output dependency ordering.
    """
    specs = validate_output_specs(output_specs)
    ordered: list[tuple[str, OutputSpec]] = []
    added: set[str] = set()

    def add(label: str) -> None:
        if label in added:
            return

        spec = specs[label]
        for basis_label in cast(tuple[str, ...], spec.basis_labels):
            if basis_label in specs:
                add(basis_label)

        ordered.append((label, spec))
        added.add(label)

    for label in specs:
        add(label)

    return tuple(ordered)


def _require_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    data = dict(value)
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise OperatorError(f"{name} must be JSON-serializable.") from exc
    return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically publish a human-facing operator descriptor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _callable_ref(value: Callable[..., Any], *, name: str) -> dict[str, str]:
    if not callable(value):
        raise OutputSpecError(f"{name} must be callable.")

    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module_name, str) or not module_name:
        raise OutputSpecError(f"{name} must expose a non-empty __module__.")
    if not isinstance(qualname, str) or not qualname:
        raise OutputSpecError(f"{name} must expose a non-empty __qualname__.")
    if module_name == "__main__" or "<locals>" in qualname or qualname == "<lambda>":
        raise OutputSpecError(
            f"{name} must be an importable top-level callable; got "
            f"{module_name}:{qualname}."
        )

    try:
        module = importlib.import_module(module_name)
        resolved: Any = module
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except Exception as exc:
        raise OutputSpecError(
            f"{name} must be importable by module and qualname; could not "
            f"resolve {module_name}:{qualname}."
        ) from exc

    if resolved is not value:
        raise OutputSpecError(
            f"{name} must resolve to the same callable by module and qualname; "
            f"{module_name}:{qualname} resolved to a different object."
        )

    return {"module": module_name, "qualname": qualname}


def _resolve_qualname(module_name: str, qualname: str) -> type[Any]:
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise OperatorError(f"{module_name}:{qualname} did not resolve to a class.")
    return obj


class BaseOperator(ABC):
    """Base class for reusable TeAL operator specifications."""

    operation_type: ClassVar[OperationType]

    def __init__(self, *, operator_id: str | None = None) -> None:
        self.operator_id = None if operator_id is None else str(operator_id)
        self.is_frozen = False

    @property
    def requires_source(self) -> bool:
        """Return whether this operator type consumes source artifacts."""
        return validate_operation_type(self.operation_type) != "import"

    @abstractmethod
    def output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> OutputSpecDeclaration:
        """Return output declarations resolved for these sources and this run."""

    def validated_output_specs(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> dict[str, OutputSpec]:
        """Return this operator's validated output declarations for one run."""
        return validate_output_specs(
            self.output_specs(sources=sources, request=request)
        )

    def output_specs_in_dependency_order(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        request: TranslationRequest,
    ) -> tuple[tuple[str, OutputSpec], ...]:
        """Return run-resolved outputs in basis-before-dependent order."""
        return output_specs_in_dependency_order(
            self.output_specs(sources=sources, request=request)
        )

    def _ensure_mutable(self) -> None:
        if self.is_frozen:
            raise FrozenOperatorError(
                f"Cannot modify frozen operator {self.operator_id or self.__class__.__name__}. "
                "Create a new operator instead."
            )

    def assign_operator_id(self, operator_id: str) -> None:
        """Assign the project operator ID before freezing."""
        self._ensure_mutable()
        value = str(operator_id)
        if self.operator_id is not None and self.operator_id != value:
            raise FrozenOperatorError(
                f"Operator already has operator_id={self.operator_id!r}; "
                f"cannot assign {value!r}."
            )
        self.operator_id = value

    def freeze(self, *, operator_id: str | None = None) -> None:
        """Freeze this operator specification."""
        if operator_id is not None:
            self.assign_operator_id(operator_id)
        self.is_frozen = True

    def ensure_ready(self, **context: Any) -> None:
        """Validate execution readiness for a specific execution context.

        Execution code calls this with mode/source context before running an
        operation. Serialization does not call this method: an operator may be
        serializable even when it is not ready for every execution mode.
        """
        _ = context

    def to_json_state(self) -> dict[str, Any]:
        """Return JSON-serializable state needed to reconstruct this operator."""
        return dict()

    @classmethod
    def from_json_state(cls, state: Mapping[str, Any]) -> Self:
        """Reconstruct an operator from ``to_json_state()`` output.

        Stateless operators can rely on this default only when their serialized
        JSON state is empty. Operators with durable configuration or fitted state
        must override this method explicitly.
        """
        if state:
            raise OperatorError(
                f"{cls.__name__} has non-empty json_state but does not implement "
                "from_json_state()."
            )
        return cls()

    def save_assets(self, assets_dir: Path) -> Mapping[str, Any]:
        """Save non-JSON assets and return a JSON-serializable manifest."""
        _ = assets_dir
        return {}

    def load_assets(self, assets_dir: Path, manifest: Mapping[str, Any]) -> None:
        """Load non-JSON assets previously saved by ``save_assets``.

        Stateless/no-asset operators can rely on this default only when the
        serialized asset manifest is empty. Operators with assets must override
        this method explicitly.
        """
        _ = assets_dir
        if manifest:
            raise OperatorError(
                f"{self.__class__.__name__} has assets but does not implement "
                "load_assets()."
            )

    def to_descriptor(self) -> dict[str, Any]:
        """Return the JSON descriptor for this operator snapshot."""
        if self.operator_id is None:
            raise OperatorError("Cannot serialize operator without operator_id.")
        operation_type = validate_operation_type(self.operation_type)
        json_state = _require_json_mapping(
            self.to_json_state(), name="operator json_state"
        )
        return {
            "schema_version": 1,
            "operator_id": self.operator_id,
            "operation_type": operation_type,
            "class": {
                "module": self.__class__.__module__,
                "qualname": self.__class__.__qualname__,
            },
            "json_state": json_state,
        }

    def save_to_dir(
        self,
        operator_dir: str | Path,
        *,
        operator_id: str,
    ) -> Path:
        """Save this operator snapshot to an operator directory."""
        if not self.is_frozen:
            self.assign_operator_id(operator_id)
        elif self.operator_id != str(operator_id):
            raise FrozenOperatorError(
                f"Frozen operator has operator_id={self.operator_id!r}; "
                f"cannot save as {operator_id!r}."
            )

        path = Path(operator_dir)
        path.mkdir(parents=True, exist_ok=True)
        assets_dir = path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        manifest = _require_json_mapping(
            self.save_assets(assets_dir),
            name="operator assets manifest",
        )
        descriptor = self.to_descriptor()
        descriptor["assets"] = manifest
        _write_json(path / "operator.json", descriptor)
        self.is_frozen = True
        return path

    @classmethod
    def load_from_dir(cls, operator_dir: str | Path) -> BaseOperator:
        """Load an operator snapshot from an operator directory."""
        path = Path(operator_dir)
        descriptor = json.loads((path / "operator.json").read_text(encoding="utf-8"))
        if not isinstance(descriptor, dict):
            raise OperatorError("operator.json must contain a JSON object.")

        class_info = descriptor.get("class")
        if not isinstance(class_info, Mapping):
            raise OperatorError("operator.json is missing class metadata.")
        module_name = class_info.get("module")
        qualname = class_info.get("qualname")
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            raise OperatorError(
                "operator class metadata must include module and qualname strings."
            )

        operator_cls = _resolve_qualname(module_name, qualname)
        if not issubclass(operator_cls, BaseOperator):
            raise OperatorError(
                f"Loaded class {module_name}:{qualname} is not a BaseOperator subclass."
            )

        state = descriptor.get("json_state", {})
        if not isinstance(state, Mapping):
            raise OperatorError("operator json_state must be a mapping.")

        obj = operator_cls.from_json_state(state)
        if not isinstance(obj, BaseOperator):
            raise OperatorError(
                "from_json_state() must return a BaseOperator instance."
            )

        recorded_operation_type = descriptor.get("operation_type")
        if recorded_operation_type is not None:
            recorded = validate_operation_type(str(recorded_operation_type))
            actual = validate_operation_type(obj.operation_type)
            if recorded != actual:
                raise OperatorError(
                    f"operator.json records operation_type={recorded!r}, "
                    f"but loaded class declares {actual!r}."
                )

        operator_id = descriptor.get("operator_id")
        if not isinstance(operator_id, str) or not operator_id:
            raise OperatorError(
                "operator.json must include a non-empty operator_id string."
            )
        obj.operator_id = operator_id

        assets = descriptor.get("assets", {})
        if not isinstance(assets, Mapping):
            raise OperatorError("operator assets manifest must be a mapping.")
        obj.load_assets(path / "assets", assets)
        obj.is_frozen = True
        return obj


class BaseTranslator(BaseOperator):
    """Base class for user-extensible artifact translators.

    The concrete translation runner lives outside this module. This class defines
    the developer-facing translation protocol; ``translate.py`` implements the
    project-side runner for that protocol. User-facing translators may also expose
    a typed ``translate(project, ...)`` convenience method that delegates to
    ``project.translate(...)``; internal translators used by core operations do
    not need to provide one.
    """

    operation_type: ClassVar[OperationType] = "translate"

    @property
    def requires_fit(self) -> bool:
        """Return whether this translator must learn fitted state before translate."""
        return False

    @property
    def is_fitted(self) -> bool:
        """Return whether required fitted state is already available."""
        return True

    @property
    def supports_fit_translate(self) -> bool:
        """Return whether this translator can fit and translate in one operation."""
        return False

    @property
    def supports_parallel_translate(self) -> bool:
        """Return whether translate mode can use translator-created workers."""
        return False

    @property
    def supports_parallel_fit_translate(self) -> bool:
        """Return whether fit_translate mode can use translator-created workers."""
        return False

    def supports_resume(self, *, mode: TranslationMode, route: RunRoute) -> bool:
        """Return whether this translator can resume this mode/route safely."""
        _ = mode, route
        return False

    def validate_operation_params(
        self,
        params: Mapping[str, Any],
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
    ) -> Mapping[str, Any]:
        """Validate and normalize operation-specific parameters for one run.

        These parameters are persisted with the operation descriptor, not the
        frozen operator snapshot. Translators that need run-time options such as
        source column names should override this hook and return their preferred
        normalized runtime mapping. JSON persistence is a separate boundary,
        handled through ``serialize_operation_params(...)`` when TeAL creates
        the durable operation record.
        """
        _ = sources, mode
        return dict(params)

    def serialize_operation_params(
        self,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return the JSON-ready durable form of normalized operation params.

        TeAL owns the operation descriptor. Translators override only this hook
        when their normalized runtime parameters need a custom JSON form.
        """
        return dict(params)

    def deserialize_operation_params(
        self,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Restore normalized runtime params from their durable JSON form.

        Translators only need to override this when
        ``serialize_operation_params(...)`` changes runtime value types.
        """
        return dict(params)

    @abstractmethod
    def input_request(
        self,
        *,
        sources: Mapping[str, BaseArtifact],
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> SourceRequest | Mapping[str, SourceRequest]:
        """Return the effective source/materialization request for this run."""

    def initialize_translation(
        self,
        *,
        mode: TranslationMode,
        route: RunRoute,
        request: TranslationRequest,
    ) -> None:
        """Initialize canonical translator state for a new operation.

        Resume paths do not call this hook; ``load_intermediate_state(...)`` is
        responsible for restoring an already-initialized canonical translator.
        """
        _ = mode, route, request

    @abstractmethod
    def translate_batch(
        self,
        inputs: Mapping[str, InputBatch],
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BatchResult:
        """Translate one materialized planned input unit.

        In parallel routes this hook is called on worker translators. The
        returned packet is opaque to TeAL and is passed back to the canonical
        translator via ``handle_batch_result(...)``.
        """

    @abstractmethod
    def handle_batch_result(
        self,
        result: BatchResult,
        *,
        batch_index: int,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        """Accept one batch result and return writer-ready outputs, if any.

        TeAL always calls this hook on the canonical translator in durable plan
        order, even when parallel workers finish their units out of order.
        Any internal fitting, accumulation, or checkpointable state is owned by
        the canonical translator itself. TeAL only receives output payloads to
        write, or ``None`` to move on without writing for this unit.
        """

    @abstractmethod
    def finalize_translation(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> OutputMap | None:
        """Finalize the operation and return any final writer-ready outputs."""

    def make_translate_worker(
        self,
        *,
        mode: TranslationMode,
        request: TranslationRequest,
    ) -> BaseTranslator:
        """Return a worker translator for parallel execution."""
        _ = mode, request
        raise OperatorError(
            f"{self.__class__.__name__} does not implement make_translate_worker()."
        )

    def save_intermediate_state(
        self,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> None:
        """Persist operation-local state needed to resume this translation.

        This is not the final reusable operator snapshot written by
        ``save_to_dir(...)``. It is a recovery hook for translators that opt
        into ``supports_resume(...)``. The canonical translator owns whatever
        intermediate state it needs to preserve.
        """
        _ = intermediate_dir, operator_id, mode, route
        raise OperatorError(
            f"{self.__class__.__name__} does not implement intermediate state saving."
        )

    @classmethod
    def load_intermediate_state(
        cls,
        intermediate_dir: Path,
        *,
        operator_id: str,
        mode: TranslationMode,
        route: RunRoute,
    ) -> BaseTranslator:
        """Load a translator from operation-local intermediate state."""
        _ = intermediate_dir, operator_id, mode, route
        raise OperatorError(
            f"{cls.__name__} does not implement intermediate state loading."
        )
