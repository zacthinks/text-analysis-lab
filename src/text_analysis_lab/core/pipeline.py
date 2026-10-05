"""Lightweight translator composition for Text Analysis Lab (TeAL).

A Pipeline is wiring and orchestration only. It never fits translators and never
introduces pipeline-specific scientific data representations.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from text_analysis_lab.core.artifact_base import BaseArtifact
from text_analysis_lab.core.errors import PipelineError
from text_analysis_lab.core.operator import BaseTranslator, ExecutionCapabilities
from text_analysis_lab.core.types import DEFAULT_OUTPUT_LABEL, DEFAULT_SOURCE_LABEL

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project


PipelinePortKind = Literal["input", "stage"]


@dataclass(frozen=True)
class PipelinePort:
    """Reference to one external pipeline input or one labeled stage output."""

    kind: PipelinePortKind
    name: str
    output_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PipelineError("Pipeline port names must be non-empty strings.")
        if self.kind == "input":
            if self.output_label is not None:
                raise PipelineError("Pipeline input ports cannot have output labels.")
            return
        if self.kind != "stage":
            raise PipelineError(f"Unsupported pipeline port kind {self.kind!r}.")
        if not isinstance(self.output_label, str) or not self.output_label:
            raise PipelineError("Stage output ports require a non-empty output label.")


class PipelineInputRef:
    """Accessor for the public inputs of a Pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def __getitem__(self, name: str) -> PipelinePort:
        value = str(name)
        if value not in self._pipeline.input_names:
            raise PipelineError(
                f"Unknown pipeline input {value!r}. "
                f"Declared inputs: {list(self._pipeline.input_names)!r}."
            )
        return PipelinePort("input", value)

    @property
    def port(self) -> PipelinePort:
        if len(self._pipeline.input_names) != 1:
            raise PipelineError(
                "pipeline.input is ambiguous because this Pipeline has multiple inputs. "
                "Use pipeline.input['name']."
            )
        return PipelinePort("input", self._pipeline.input_names[0])


class PipelineStageRef:
    """Reference returned by Pipeline.add(...)."""

    def __init__(self, pipeline: Pipeline, stage_name: str) -> None:
        self._pipeline = pipeline
        self.stage_name = stage_name

    def __getitem__(self, output_label: str) -> PipelinePort:
        label = str(output_label)
        if not label:
            raise PipelineError("Pipeline output labels must be non-empty strings.")
        return PipelinePort("stage", self.stage_name, label)

    @property
    def output(self) -> PipelinePort:
        return self[DEFAULT_OUTPUT_LABEL]


@dataclass(frozen=True)
class PipelineStage:
    """One translator and its labeled source bindings."""

    name: str
    translator: BaseTranslator
    sources: tuple[tuple[str, PipelinePort], ...]

    def source_map(self) -> dict[str, PipelinePort]:
        return dict(self.sources)


@dataclass(frozen=True)
class PipelineCapabilityIssue:
    """One stage-level reason a pipeline execution mode is unavailable."""

    stage: str
    translator: str
    reason: str


@dataclass(frozen=True)
class PipelineCapabilities:
    """Capabilities derived from the stages required by selected outputs."""

    reusable: bool
    artifact: bool
    native: bool
    portable: bool
    requirements: tuple[str, ...] = ()
    issues: tuple[PipelineCapabilityIssue, ...] = ()


class Pipeline:
    """A user-authored DAG of configured translators and labeled connections."""

    def __init__(self, *, inputs: Sequence[str] | None = None) -> None:
        raw_inputs = (DEFAULT_SOURCE_LABEL,) if inputs is None else tuple(inputs)
        if not raw_inputs:
            raise PipelineError("A Pipeline must declare at least one input.")
        names = tuple(str(name) for name in raw_inputs)
        if any(not name for name in names):
            raise PipelineError("Pipeline input names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise PipelineError("Pipeline input names must be unique.")
        self._input_names = names
        self._stages: dict[str, PipelineStage] = {}
        self._outputs: dict[str, PipelinePort] = {}
        self._input_accessor = PipelineInputRef(self)

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def input(self) -> PipelinePort | PipelineInputRef:
        """Return the sole input port, or a labeled accessor for multiple inputs."""
        if len(self._input_names) == 1:
            return self._input_accessor.port
        return self._input_accessor

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return tuple(self._stages.values())

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._effective_outputs())

    def add(
        self,
        name: str,
        translator: BaseTranslator,
        *,
        source: PipelinePort | None = None,
        sources: Mapping[str, PipelinePort] | None = None,
    ) -> PipelineStageRef:
        """Add one translator stage with explicit labeled source bindings."""
        stage_name = str(name)
        if not stage_name:
            raise PipelineError("Pipeline stage names must be non-empty strings.")
        if stage_name in self._stages:
            raise PipelineError(f"Duplicate pipeline stage name {stage_name!r}.")
        if not isinstance(translator, BaseTranslator):
            raise PipelineError(
                f"Pipeline stage {stage_name!r} requires a BaseTranslator; "
                f"got {type(translator).__name__}."
            )
        if source is not None and sources is not None:
            raise PipelineError("Pass source= or sources=, not both.")

        if sources is None:
            if source is None:
                if not self._stages:
                    if len(self._input_names) != 1:
                        raise PipelineError(
                            f"Stage {stage_name!r} needs explicit sources because "
                            "the Pipeline has multiple public inputs."
                        )
                    source = self._input_accessor.port
                else:
                    raise PipelineError(
                        f"Stage {stage_name!r} needs an explicit source binding."
                    )
            source_bindings = {DEFAULT_SOURCE_LABEL: source}
        else:
            source_bindings = dict(sources)
            if not source_bindings:
                raise PipelineError(
                    f"Stage {stage_name!r} must bind at least one source."
                )

        normalized: list[tuple[str, PipelinePort]] = []
        for raw_label, port in source_bindings.items():
            label = str(raw_label)
            if not label:
                raise PipelineError("Stage source labels must be non-empty strings.")
            if not isinstance(port, PipelinePort):
                raise PipelineError(
                    f"Stage {stage_name!r} source {label!r} must reference "
                    "a pipeline input or stage output."
                )
            normalized.append((label, port))

        self._stages[stage_name] = PipelineStage(
            name=stage_name,
            translator=translator,
            sources=tuple(normalized),
        )
        try:
            self._validate_structure()
        except Exception:
            del self._stages[stage_name]
            raise
        return PipelineStageRef(self, stage_name)

    def output(self, name: str, port: PipelinePort) -> Pipeline:
        """Expose a stage output or public input under a pipeline output name."""
        output_name = str(name)
        if not output_name:
            raise PipelineError("Pipeline output names must be non-empty strings.")
        if output_name in self._outputs:
            raise PipelineError(f"Duplicate pipeline output name {output_name!r}.")
        if not isinstance(port, PipelinePort):
            raise PipelineError("Pipeline outputs must reference PipelinePort values.")
        self._outputs[output_name] = port
        try:
            self._validate_structure()
        except Exception:
            del self._outputs[output_name]
            raise
        return self

    def validate(
        self,
        *,
        mode: Literal["native", "artifact"] | None = None,
        project: Project | None = None,
        outputs: Sequence[str] | None = None,
    ) -> PipelineCapabilities:
        """Validate graph structure and optionally one execution mode."""
        self._validate_structure()
        caps = self.execution_capabilities(project=project, outputs=outputs)
        if mode == "native" and not caps.native:
            raise PipelineError(self._capability_message("native", caps))
        if mode == "artifact" and not caps.artifact:
            raise PipelineError(self._capability_message("artifact", caps))
        if mode not in {None, "native", "artifact"}:
            raise PipelineError(f"Unsupported Pipeline validation mode {mode!r}.")
        return caps

    def execution_capabilities(
        self,
        *,
        project: Project | None = None,
        outputs: Sequence[str] | None = None,
    ) -> PipelineCapabilities:
        """Derive capabilities over the dependency closure of selected outputs."""
        selected = self._select_outputs(outputs)
        required = self._required_stages(selected.values())
        reusable = True
        artifact = True
        native = True
        portable = True
        requirements: list[str] = []
        issues: list[PipelineCapabilityIssue] = []

        for stage_name in self._topological_order(required):
            stage = self._stages[stage_name]
            caps = stage.translator.execution_capabilities(project=project)
            reusable = reusable and caps.reusable
            artifact = artifact and caps.artifact
            native = native and caps.native
            portable = portable and caps.portable
            for requirement in caps.requirements:
                if requirement not in requirements:
                    requirements.append(requirement)
            for reason in caps.reasons:
                issues.append(
                    PipelineCapabilityIssue(
                        stage=stage_name,
                        translator=stage.translator.__class__.__name__,
                        reason=reason,
                    )
                )

        return PipelineCapabilities(
            reusable=reusable,
            artifact=artifact,
            native=native,
            portable=portable,
            requirements=tuple(requirements),
            issues=tuple(issues),
        )

    def translate(
        self,
        values: Any = None,
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Sequence[str] | None = None,
        project: Project | None = None,
    ) -> Mapping[str, Any]:
        """Execute selected outputs through standalone translator.translate(...)."""
        external = self._normalize_external_values(values=values, inputs=inputs)
        selected = self._select_outputs(outputs)
        required = self._required_stages(selected.values())
        self.validate(mode="native", project=project, outputs=tuple(selected))

        stage_values: dict[tuple[str, str], Any] = {}
        needed_labels = self._needed_output_labels(required, selected.values())

        for stage_name in self._topological_order(required):
            stage = self._stages[stage_name]
            if project is not None:
                # Project-backed translators such as UMAP(reuse="recompute") may
                # rebuild transient transform state here. Standalone values have
                # no TeAL source artifacts, so the source mapping is intentionally
                # empty.
                stage.translator.prepare_for_translation(project=project, sources={})
                refreshed = stage.translator.execution_capabilities(project=project)
                if not refreshed.native:
                    raise PipelineError(
                        self._stage_capability_message(stage_name, "native", refreshed)
                    )

            bound = {
                label: self._resolve_value(port, external, stage_values)
                for label, port in stage.sources
            }
            result = self._call_standalone(stage.translator, bound)
            labels = needed_labels.get(stage_name, {DEFAULT_OUTPUT_LABEL})
            if labels == {DEFAULT_OUTPUT_LABEL}:
                stage_values[(stage_name, DEFAULT_OUTPUT_LABEL)] = result
            else:
                if not isinstance(result, Mapping):
                    raise PipelineError(
                        f"Stage {stage_name!r} is used as a multi-output stage but "
                        f"{stage.translator.__class__.__name__}.translate(...) "
                        f"returned {type(result).__name__}, not a mapping."
                    )
                for label in labels:
                    if label not in result:
                        raise PipelineError(
                            f"Stage {stage_name!r} did not return required output "
                            f"label {label!r}. Available labels: {list(result)!r}."
                        )
                    stage_values[(stage_name, label)] = result[label]

        return {
            name: self._resolve_value(port, external, stage_values)
            for name, port in selected.items()
        }

    def run_artifacts(
        self,
        project: Project,
        sources: BaseArtifact
        | str
        | Mapping[str, BaseArtifact | str],
        *,
        outputs: Sequence[str] | None = None,
    ) -> Mapping[str, BaseArtifact]:
        """Execute selected outputs through ordinary Project.translate(...) calls."""
        external = self._normalize_artifact_sources(project, sources)
        selected = self._select_outputs(outputs)
        required = self._required_stages(selected.values())
        self.validate(mode="artifact", project=project, outputs=tuple(selected))
        stage_values: dict[tuple[str, str], BaseArtifact] = {}

        for stage_name in self._topological_order(required):
            stage = self._stages[stage_name]
            bound = {
                label: self._resolve_value(port, external, stage_values)
                for label, port in stage.sources
            }
            stage.translator.prepare_for_translation(project=project, sources=bound)
            caps = stage.translator.execution_capabilities(project=project)
            if not caps.artifact:
                raise PipelineError(
                    self._stage_capability_message(stage_name, "artifact", caps)
                )
            if stage.translator.requires_fit and not stage.translator.is_fitted:
                raise PipelineError(
                    f"Pipeline stage {stage_name!r} would require fitting "
                    f"{stage.translator.__class__.__name__}; Pipelines are transform-only."
                )

            produced = project.translate(stage.translator, bound)
            for label, artifact in produced.items():
                stage_values[(stage_name, label)] = artifact

        return {
            name: self._resolve_value(port, external, stage_values)
            for name, port in selected.items()
        }

    def describe(self) -> dict[str, Any]:
        """Return a compact derived description of the in-memory graph."""
        outputs = self._effective_outputs()
        return {
            "inputs": self._input_names,
            "stages": tuple(
                {
                    "name": stage.name,
                    "translator": stage.translator.__class__.__name__,
                    "operator_id": stage.translator.operator_id,
                    "sources": {
                        label: self._format_port(port)
                        for label, port in stage.sources
                    },
                }
                for stage in self._stages.values()
            ),
            "outputs": {
                name: self._format_port(port) for name, port in outputs.items()
            },
        }

    def _effective_outputs(self) -> dict[str, PipelinePort]:
        if self._outputs:
            return dict(self._outputs)
        terminal = self._terminal_stages()
        if len(terminal) == 1:
            return {
                DEFAULT_OUTPUT_LABEL: PipelinePort(
                    "stage", terminal[0], DEFAULT_OUTPUT_LABEL
                )
            }
        if not self._stages and len(self._input_names) == 1:
            return {
                DEFAULT_OUTPUT_LABEL: PipelinePort("input", self._input_names[0])
            }
        raise PipelineError(
            "Pipeline outputs are ambiguous. Declare them explicitly with "
            "pipeline.output(...)."
        )

    def _terminal_stages(self) -> list[str]:
        consumed = {
            port.name
            for stage in self._stages.values()
            for _, port in stage.sources
            if port.kind == "stage"
        }
        return [name for name in self._stages if name not in consumed]

    def _select_outputs(
        self, outputs: Sequence[str] | None
    ) -> dict[str, PipelinePort]:
        available = self._effective_outputs()
        if outputs is None:
            return available
        names = tuple(str(name) for name in outputs)
        if len(set(names)) != len(names):
            raise PipelineError("Requested pipeline output names must be unique.")
        missing = [name for name in names if name not in available]
        if missing:
            raise PipelineError(
                f"Unknown pipeline outputs {missing!r}. "
                f"Available outputs: {list(available)!r}."
            )
        return {name: available[name] for name in names}

    def _validate_structure(self) -> None:
        for stage in self._stages.values():
            labels = [label for label, _ in stage.sources]
            if len(set(labels)) != len(labels):
                raise PipelineError(
                    f"Stage {stage.name!r} has duplicate source labels."
                )
            for _, port in stage.sources:
                self._validate_port(port)
                if port.kind == "stage" and port.name == stage.name:
                    raise PipelineError(
                        f"Stage {stage.name!r} cannot consume its own output."
                    )
        for port in self._outputs.values():
            self._validate_port(port)
        self._topological_order(set(self._stages))

    def _validate_port(self, port: PipelinePort) -> None:
        if port.kind == "input":
            if port.name not in self._input_names:
                raise PipelineError(
                    f"Pipeline reference names unknown input {port.name!r}."
                )
            return
        if port.name not in self._stages:
            raise PipelineError(
                f"Pipeline reference names unknown stage {port.name!r}."
            )

    def _topological_order(self, required: set[str]) -> tuple[str, ...]:
        visiting: set[str] = set()
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str, stack: list[str]) -> None:
            if name not in required or name in visited:
                return
            if name in visiting:
                cycle = stack[stack.index(name) :] + [name]
                raise PipelineError(
                    f"Cycle in Pipeline graph: {' -> '.join(cycle)}."
                )
            visiting.add(name)
            stack.append(name)
            for _, port in self._stages[name].sources:
                if port.kind == "stage":
                    visit(port.name, stack)
            stack.pop()
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self._stages:
            visit(name, [])
        return tuple(order)

    def _required_stages(self, ports: Any) -> set[str]:
        required: set[str] = set()

        def add(port: PipelinePort) -> None:
            if port.kind != "stage" or port.name in required:
                return
            required.add(port.name)
            for _, upstream in self._stages[port.name].sources:
                add(upstream)

        for port in ports:
            add(port)
        return required

    def _needed_output_labels(
        self,
        required: set[str],
        selected_ports: Any,
    ) -> dict[str, set[str]]:
        labels: dict[str, set[str]] = {name: set() for name in required}
        for stage_name in required:
            for _, port in self._stages[stage_name].sources:
                if port.kind == "stage" and port.name in required:
                    labels[port.name].add(str(port.output_label))
        for port in selected_ports:
            if port.kind == "stage":
                labels[port.name].add(str(port.output_label))
        return labels

    def _normalize_external_values(
        self,
        *,
        values: Any,
        inputs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if inputs is not None:
            if values is not None:
                raise PipelineError("Pass values= or inputs=, not both.")
            supplied = {str(name): value for name, value in inputs.items()}
        else:
            if len(self._input_names) != 1:
                raise PipelineError(
                    "This Pipeline has multiple inputs; pass inputs={...}."
                )
            supplied = {self._input_names[0]: values}
        self._validate_external_names(supplied)
        return supplied

    def _normalize_artifact_sources(
        self,
        project: Project,
        sources: BaseArtifact | str | Mapping[str, BaseArtifact | str],
    ) -> dict[str, BaseArtifact]:
        if isinstance(sources, Mapping):
            supplied = {str(name): value for name, value in sources.items()}
        else:
            if len(self._input_names) != 1:
                raise PipelineError(
                    "This Pipeline has multiple inputs; pass a mapping of input names "
                    "to artifacts."
                )
            supplied = {self._input_names[0]: sources}
        self._validate_external_names(supplied)
        return {
            name: value
            if isinstance(value, BaseArtifact)
            else project.get_artifact(str(value))
            for name, value in supplied.items()
        }

    def _validate_external_names(self, supplied: Mapping[str, Any]) -> None:
        missing = [name for name in self._input_names if name not in supplied]
        extra = [name for name in supplied if name not in self._input_names]
        if missing or extra:
            raise PipelineError(
                "Pipeline input binding mismatch. "
                f"Missing={missing!r}, extra={extra!r}."
            )

    @staticmethod
    def _resolve_value(
        port: PipelinePort,
        external: Mapping[str, Any],
        stage_values: Mapping[tuple[str, str], Any],
    ) -> Any:
        if port.kind == "input":
            return external[port.name]
        key = (port.name, str(port.output_label))
        if key not in stage_values:
            raise PipelineError(
                f"Pipeline stage output {port.name!r}[{port.output_label!r}] "
                "was not produced."
            )
        return stage_values[key]

    @staticmethod
    def _call_standalone(
        translator: BaseTranslator,
        bound: Mapping[str, Any],
    ) -> Any:
        values = tuple(bound.values())
        if len(values) == 1:
            return translator.translate(values[0])

        signature = inspect.signature(translator.translate)
        params = signature.parameters
        can_use_keywords = all(
            label in params
            and params[label].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            for label in bound
        )
        if can_use_keywords:
            return translator.translate(**bound)
        return translator.translate(*values)

    @staticmethod
    def _format_port(port: PipelinePort) -> str:
        if port.kind == "input":
            return f"input:{port.name}"
        return f"{port.name}:{port.output_label}"

    @staticmethod
    def _stage_capability_message(
        stage_name: str,
        mode: str,
        caps: ExecutionCapabilities,
    ) -> str:
        reasons = "; ".join(caps.reasons) or "translator reports mode unavailable"
        return (
            f"Pipeline stage {stage_name!r} cannot execute in {mode} mode: {reasons}"
        )

    @staticmethod
    def _capability_message(mode: str, caps: PipelineCapabilities) -> str:
        relevant = [
            issue
            for issue in caps.issues
            if (
                mode == "native"
                or mode == "artifact"
            )
        ]
        if not relevant:
            return f"Pipeline cannot execute in {mode} mode."
        details = "; ".join(
            f"{issue.stage} ({issue.translator}): {issue.reason}"
            for issue in relevant
        )
        return f"Pipeline cannot execute in {mode} mode. {details}"
