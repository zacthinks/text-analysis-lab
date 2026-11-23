"""R1 single-operation, data-only secondary Python coordinator.

This is deliberately not a general RPC endpoint. Only the official TextLength
translator, its JSON scientific state, and narrow execution settings cross the
interpreter boundary. Never import a callable named by a request.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from text_analysis_lab.core.environments import (
    EnvironmentRegistryError,
    fingerprint_observation,
    normalize_environment,
    observe_environment,
)
from text_analysis_lab.core.errors import OperatorError

if TYPE_CHECKING:
    from text_analysis_lab.core.project import Project

PROTOCOL_VERSION = 2
TRANSLATOR_TYPE = "TextLength"
TRANSLATOR_MODULE = "text_analysis_lab.translators.text_length"


class SecondaryCoordinatorError(OperatorError):
    """Pilot coordinator rejected a request or could not verify the child."""


class SecondaryOperationError(SecondaryCoordinatorError):
    """The child failed, or exited with an uncertain persisted operation outcome."""

    def __init__(
        self, message: str, *, request_id: str, operation_id: str | None = None
    ) -> None:
        self.request_id = request_id
        self.operation_id = operation_id
        super().__init__(message)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SecondaryCoordinatorError("Invalid secondary coordinator JSON.") from exc
    if not isinstance(value, dict):
        raise SecondaryCoordinatorError("Secondary coordinator message must be a JSON object.")
    return value


def _keys(message: Mapping[str, Any], expected: set[str]) -> None:
    if set(message) != expected:
        raise SecondaryCoordinatorError(
            f"Unexpected secondary protocol keys: {sorted(set(message) ^ expected)}."
        )


def _version(message: Mapping[str, Any]) -> None:
    if type(message.get("protocol_version")) is not int or message["protocol_version"] != PROTOCOL_VERSION:
        raise SecondaryCoordinatorError(
            f"Incompatible secondary protocol version {message.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION}."
        )


def _pilot_code_digest() -> str:
    """Require the code actually implementing this narrow protocol to match.

    TeAL distribution versions alone can hide locally modified installations.
    This is a conservative R1 pilot compatibility check, not a hash of the
    parent's scientific dependencies or a generic package-verification scheme.
    """
    digest = hashlib.sha256()
    for name in (
        "text_analysis_lab.core.secondary_coordinator",
        "text_analysis_lab.core.catalog",
        "text_analysis_lab.core.environments",
        "text_analysis_lab.core.project",
        "text_analysis_lab.core.translate",
        "text_analysis_lab.translators.text_length",
    ):
        module = importlib.import_module(name)
        source = getattr(module, "__file__", None)
        if not source or not str(source).endswith(".py"):
            raise SecondaryCoordinatorError(
                f"Cannot verify source of secondary protocol module {name}."
            )
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(Path(source).read_bytes()).digest())
    return digest.hexdigest()


def _probe_payload() -> dict[str, Any]:
    observation = normalize_environment(observe_environment())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "probe",
        "environment_id": "env_" + fingerprint_observation(observation),
        "observation": observation,
        "pilot_code_sha256": _pilot_code_digest(),
        "teal_version": metadata.version("text-analysis-lab"),
        "observation_schema_version": observation["schema_version"],
        "python_version": list(sys.version_info[:3]),
    }


def _check_probe(project: Project, selected: str, payload: dict[str, Any]) -> None:
    _keys(
        payload,
        {
            "protocol_version", "kind", "environment_id", "observation",
            "pilot_code_sha256", "teal_version",
            "observation_schema_version", "python_version",
        },
    )
    _version(payload)
    if payload["kind"] != "probe":
        raise SecondaryCoordinatorError("Child returned an unexpected probe response.")

    # Do not trust the logical identity claimed by a child: reconstruct its
    # installed metadata fingerprint and validate against the selected record.
    actual = normalize_environment(payload["observation"])
    if actual["kind"] != "observed_unlocked":
        raise SecondaryCoordinatorError("Child must probe its actual installed software.")
    actual_id = "env_" + fingerprint_observation(actual)
    if payload["environment_id"] != actual_id:
        raise EnvironmentRegistryError("Child's claimed environment identity disagrees with its software inventory.")
    target = project.environments.get(selected)["observation"]
    if target["kind"] == "observed_unlocked":
        if actual_id != selected:
            raise EnvironmentRegistryError(
                "Installed packages or Python changed in bound secondary interpreter."
            )
    elif target["kind"] == "portable_locked":
        if not project.environments._matches_portable(selected, actual):
            raise EnvironmentRegistryError(
                "Child's installed packages, Python or immutable TeAL source do not "
                "match a declared portable lock realization."
            )
    else:
        raise SecondaryCoordinatorError("Unsupported secondary environment kind.")

    current = project.environments.get(project.environments.opening_id)["observation"]
    if (payload["teal_version"] != actual["teal"]["version"]
            or payload["teal_version"] != current["teal"]["version"]
            or payload["teal_version"] != target["teal"]["version"]):
        raise SecondaryCoordinatorError(
            "Incompatible TeAL distribution version in secondary interpreter."
        )
    if (payload["observation_schema_version"] != actual["schema_version"]
            or target["schema_version"] != actual["schema_version"]):
        raise SecondaryCoordinatorError("Incompatible environment observation schema.")
    py = payload["python_version"]
    if (
        not isinstance(py, list) or len(py) != 3
        or any(type(v) is not int for v in py)
        or (py[0], py[1]) < (3, 11)
        or ".".join(map(str, py)) != actual["python"]["version"]
    ):
        raise SecondaryCoordinatorError("Secondary interpreter has incompatible Python version.")
    if (not isinstance(payload["pilot_code_sha256"], str)
            or payload["pilot_code_sha256"] != _pilot_code_digest()):
        raise SecondaryCoordinatorError(
            "Incompatible TeAL secondary protocol implementation/source; "
            "matching package versions alone are insufficient."
        )


def _run(
    executable: Path,
    action: str,
    data: Mapping[str, Any] | None = None,
    *,
    timeout: float | None = None,
) -> tuple[int, dict[str, Any] | None, str]:
    args = [str(executable), "-m", "text_analysis_lab.core.secondary_coordinator", action]
    try:
        result = subprocess.run(
            args, input=None if data is None else _json(data), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SecondaryCoordinatorError(
            f"Unable to launch or complete secondary coordinator {action}: {exc}"
        ) from exc
    if not result.stdout.strip():
        return result.returncode, None, result.stderr[-2000:]
    try:
        return result.returncode, _read_json(result.stdout.strip()), result.stderr[-2000:]
    except SecondaryCoordinatorError:
        return result.returncode, None, result.stderr[-2000:]


def _verified_executable(project: Project, selected: str) -> Path:
    executable = project.environments.bound_interpreter(selected)
    exit_code, payload, stderr = _run(executable, "--probe", timeout=60)
    if exit_code != 0 or payload is None:
        raise SecondaryCoordinatorError(
            f"Bound Python interpreter could not import or probe compatible TeAL "
            f"(exit={exit_code}): {stderr}"
        )
    _check_probe(project, selected, payload)
    return executable


def _execution_settings(
    *,
    workers: int, batch_size: int | None, max_outstanding_units: int | None,
    memo: str | None, alias: Any, overwrite: bool, batch_by: str | None,
    freeze_device: bool, params: Mapping[str, Any],
) -> dict[str, Any]:
    if type(workers) is not int or workers < 1:
        raise SecondaryCoordinatorError("Pilot requires workers >=1.")
    for name, value in (
        ("batch_size", batch_size),
        ("max_outstanding_units", max_outstanding_units),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise SecondaryCoordinatorError(f"Pilot requires positive {name}.")
    if batch_by is not None or freeze_device or params:
        raise SecondaryCoordinatorError(
            "Secondary TextLength pilot does not support batch_by, "
            "freeze_device, or translator execution parameters."
        )
    if memo is not None and not isinstance(memo, str):
        raise SecondaryCoordinatorError("Pilot memo must be a string or None.")
    if type(overwrite) is not bool:
        raise SecondaryCoordinatorError("Pilot overwrite must be a bool.")
    # TextLength has a single output. Preserve the existing public alias
    # contract rather than accepting mapping forms its local engine rejects.
    if alias is not None and (not isinstance(alias, str) or not alias):
        raise SecondaryCoordinatorError(
            "Secondary TextLength pilot requires a nonempty string alias."
        )
    return {
        "workers": workers, "batch_size": batch_size,
        "max_outstanding_units": max_outstanding_units, "memo": memo,
        "alias": alias, "overwrite": overwrite,
    }


def _source_id(project: Project, sources: Any) -> str:
    from text_analysis_lab.core.artifact_base import BaseArtifact

    if isinstance(sources, Mapping):
        if set(sources) != {"source"}:
            raise SecondaryCoordinatorError("Pilot requires one source labeled 'source'.")
        source = sources["source"]
    elif isinstance(sources, (list, tuple)):
        raise SecondaryCoordinatorError("Pilot requires a single artifact, not a sequence.")
    elif isinstance(sources, (BaseArtifact, str)):
        source = sources
    else:
        raise SecondaryCoordinatorError("Pilot requires one TeAL artifact or artifact ID.")
    return project.resolve_artifact_id(source)


def _validate_translator(translator: Any) -> dict[str, Any]:
    from text_analysis_lab.translators.text_length import TextLength

    if type(translator) is not TextLength or translator.operator_id is not None:
        raise SecondaryCoordinatorError(
            "Secondary pilot supports only a new built-in TextLength translator; "
            "custom, frozen, and other translators require local execution."
        )
    state = translator.to_json_state()
    if not isinstance(state, dict):
        raise SecondaryCoordinatorError("TextLength state must be a JSON object.")
    try:
        encoded = _json(state)
        rebuilt = TextLength.from_json_state(json.loads(encoded))
        if rebuilt.to_json_state() != state:
            raise SecondaryCoordinatorError("TextLength state did not round-trip.")
    except (ValueError, TypeError, KeyError) as exc:
        raise SecondaryCoordinatorError("Invalid TextLength scientific configuration.") from exc
    return state


def _validate_request(message: dict[str, Any]) -> None:
    _keys(message, {
        "protocol_version", "kind", "request_id", "project_path",
        "environment_id", "translator", "scientific_state", "source_ids",
        "settings", "operation_id",
    })
    _version(message)
    if message["kind"] not in ("translate", "resume"):
        raise SecondaryCoordinatorError("Unsupported secondary operation kind.")
    if message["translator"] != TRANSLATOR_TYPE:
        raise SecondaryCoordinatorError("Only built-in TextLength is supported.")
    if not isinstance(message["request_id"], str) or not message["request_id"]:
        raise SecondaryCoordinatorError("Missing secondary request identifier.")
    if not isinstance(message["project_path"], str) or not message["project_path"]:
        raise SecondaryCoordinatorError("Missing TeAL project path.")
    if not isinstance(message["environment_id"], str) or not message["environment_id"]:
        raise SecondaryCoordinatorError("Missing selected environment ID.")
    if message["kind"] == "translate":
        if message["operation_id"] is not None:
            raise SecondaryCoordinatorError("New translation cannot specify an operation ID.")
        if not isinstance(message["scientific_state"], dict):
            raise SecondaryCoordinatorError("Missing TextLength state.")
        if not isinstance(message["source_ids"], dict) or (
            set(message["source_ids"]) != {"source"}
            or not isinstance(message["source_ids"]["source"], str)
        ):
            raise SecondaryCoordinatorError("Pilot requires one source artifact ID.")
        if not isinstance(message["settings"], dict):
            raise SecondaryCoordinatorError("Missing translation settings.")
        _keys(message["settings"], {
            "workers", "batch_size", "max_outstanding_units",
            "memo", "alias", "overwrite",
        })
        _execution_settings(
            **message["settings"], batch_by=None, freeze_device=False, params={}
        )
    else:
        if (
            not isinstance(message["operation_id"], str)
            or not message["operation_id"]
            or message["settings"] is not None
            or message["scientific_state"] is not None
            or message["source_ids"] is not None
        ):
            raise SecondaryCoordinatorError("Invalid secondary resume request.")


def _refresh_parent(project: Project) -> None:
    # A separate Python process has committed new catalog rows and data files.
    # Reset connections/views before resolving any output artifact on this Project.
    project.query.clear_cache()
    project.catalog.close()


def _return_artifacts(
    project: Project, selected: str, response: dict[str, Any],
    request_id: str, *, expected_operation_id: str | None = None,
    requested_alias: str | None = None, overwrite: bool = False,
) -> Mapping[str, Any]:
    _keys(response, {
        "protocol_version", "kind", "request_id", "status", "operation_id",
        "output_artifact_ids", "environment_id", "platform_id", "result_kind",
    })
    _version(response)
    if response["kind"] != "result" or response["request_id"] != request_id:
        raise SecondaryCoordinatorError("Mismatched secondary response.")
    if response["status"] != "complete" or response["environment_id"] != selected:
        raise SecondaryCoordinatorError("Secondary returned incompatible status or environment.")
    result_kind = response["result_kind"]
    if result_kind not in {"executed", "reused"}:
        raise SecondaryCoordinatorError("Invalid secondary result kind.")
    opid = response["operation_id"]
    if not isinstance(opid, str) or not opid:
        raise SecondaryCoordinatorError("Secondary response omitted operation ID.")
    if expected_operation_id is not None and opid != expected_operation_id:
        raise SecondaryCoordinatorError("Secondary resumed the wrong operation.")

    operation = project.catalog.get_operation(opid)
    if operation["status"] != "complete":
        raise SecondaryCoordinatorError("Secondary result references an incomplete operation.")
    if result_kind == "executed":
        # The executor's actual platform, including a different resume host,
        # comes from the child attempt. Never substitute the parent platform.
        attempts = project.catalog.list_execution_attempts(opid)
        if not attempts or attempts[-1]["outcome"] != "complete":
            raise SecondaryCoordinatorError("Missing completed child execution attempt.")
        if (attempts[-1]["environment_id"] != selected
                or attempts[-1]["platform_id"] != response["platform_id"]
                or project.platforms.get(response["platform_id"]) is None):
            raise SecondaryCoordinatorError("Child execution attempt provenance mismatch.")
        # Only the selected child may create an operation for this request.
        # A resume executes an existing B operation, not a new one.
        if operation["environment_id"] != selected:
            raise SecondaryCoordinatorError(
                "Executed operation has incorrect environment provenance."
            )
        if expected_operation_id is None and operation["coordinator_request_id"] != request_id:
            raise SecondaryCoordinatorError("Secondary operation request correlation mismatch.")
    else:
        if response["platform_id"] != operation["platform_id"]:
            raise SecondaryCoordinatorError("Reused alias lost its original platform provenance.")
        # Alias reuse legitimately returns an earlier artifact: there is *no*
        # operation belonging to this request, and its historical executor need
        # not be B. Check the actual alias and durable output edges instead.
        if (
            expected_operation_id is not None or not requested_alias or overwrite
            or project.catalog.operation_for_coordinator_request(request_id) is not None
        ):
            raise SecondaryCoordinatorError("Invalid secondary alias-reuse claim.")

    outputs = response["output_artifact_ids"]
    if not isinstance(outputs, dict) or not outputs or not all(
        isinstance(label, str) and label and isinstance(aid, str) and aid
        for label, aid in outputs.items()
    ):
        raise SecondaryCoordinatorError("Invalid secondary artifact IDs.")
    recorded = {
        row["output_label"]: row["artifact_id"]
        for row in project.catalog.operation_outputs(opid)
    }
    if result_kind == "executed":
        if outputs != recorded:
            raise SecondaryCoordinatorError("Returned output artifact IDs disagree with catalog.")
    else:
        # Idempotence can reuse any completed aliased artifact, including one
        # whose original producer labeled it differently from TextLength's
        # "output". Validate its durable producer edge and alias independently.
        if (
            set(outputs) != {"output"}
            or outputs["output"] not in recorded.values()
            or project.resolve_artifact_id(requested_alias) != outputs["output"]
        ):
            raise SecondaryCoordinatorError(
                "Secondary alias reuse does not match an existing aliased artifact."
            )
    return {label: project.get_artifact(aid) for label, aid in outputs.items()}

def _delegate(
    project: Project, selected: str, message: dict[str, Any],
    *, expected_operation_id: str | None = None,
) -> Mapping[str, Any]:
    _validate_request(message)
    executable = _verified_executable(project, selected)
    request_id = message["request_id"]
    try:
        exit_code, response, stderr = _run(executable, "--run", message)
    except SecondaryCoordinatorError as exc:
        _refresh_parent(project)
        operation = (
            project.catalog.operation_for_coordinator_request(request_id)
            or project.catalog.operation_for_attempt_request(request_id)
        )
        raise SecondaryOperationError(
            "Secondary coordinator outcome is uncertain; do not automatically retry. "
            f"Request {request_id}; operation "
            f"{None if operation is None else operation['operation_id']}. {exc}",
            request_id=request_id,
            operation_id=None if operation is None else operation["operation_id"],
        ) from exc
    _refresh_parent(project)
    tracked = (
        project.catalog.operation_for_coordinator_request(request_id)
        or project.catalog.operation_for_attempt_request(request_id)
    )
    if exit_code != 0 or response is None or response.get("status") != "complete":
        operation_id = (
            response.get("operation_id") if response is not None else None
        ) or (
            tracked["operation_id"] if tracked is not None else expected_operation_id
        )
        reason = response.get("error") if isinstance(response, dict) else None
        phase = response.get("phase") if isinstance(response, dict) else None
        raise SecondaryOperationError(
            f"Secondary {message['kind']} failed "
            f"({phase or 'uncertain'}, exit={exit_code}, request={request_id}, "
            f"operation={operation_id}): {reason or stderr or 'no response'}. "
            "Do not automatically retry an uncertain outcome.",
            request_id=request_id,
            operation_id=operation_id,
        )
    settings = message["settings"] if message["kind"] == "translate" else None
    return _return_artifacts(
        project, selected, response, request_id,
        expected_operation_id=expected_operation_id,
        requested_alias=None if settings is None else settings["alias"],
        overwrite=False if settings is None else settings["overwrite"],
    )


def delegate_translate(
    project: Project,
    selected: str,
    translator: Any,
    sources: Any,
    *,
    workers: int, batch_size: int | None, batch_by: str | None,
    max_outstanding_units: int | None, memo: str | None,
    alias: Any, overwrite: bool, freeze_device: bool,
    params: Mapping[str, Any],
) -> Mapping[str, Any]:
    # Reject unsupported translator/settings before probing a subprocess or
    # creating an operation. Resolve only existing source IDs in the parent.
    state = _validate_translator(translator)
    settings = _execution_settings(
        workers=workers, batch_size=batch_size, batch_by=batch_by,
        max_outstanding_units=max_outstanding_units, memo=memo,
        alias=alias, overwrite=overwrite, freeze_device=freeze_device,
        params=params,
    )
    source_id = _source_id(project, sources)
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "translate", "request_id": uuid4().hex,
        "project_path": str(project.path.resolve()),
        "environment_id": selected, "translator": TRANSLATOR_TYPE,
        "scientific_state": state, "source_ids": {"source": source_id},
        "settings": settings, "operation_id": None,
    }
    return _delegate(project, selected, message)


def delegate_resume(
    project: Project, selected: str, operation_id: str,
) -> Mapping[str, Any]:
    row = project.catalog.get_operation(operation_id)
    if row["operation_type"] != "translate":
        raise SecondaryCoordinatorError("Pilot remote resume supports TextLength translation only.")
    path = project.storage.operation_descriptor_path(operation_id)
    if not path.is_file():
        raise SecondaryCoordinatorError("Remote resume requires an existing operation descriptor.")
    descriptor = _read_json(path.read_text(encoding="utf-8"))
    if descriptor.get("translator_class") != {
        "module": TRANSLATOR_MODULE, "qualname": TRANSLATOR_TYPE
    }:
        origin = (
            "its recorded execution environment"
            if row["environment_id"] is not None
            else "the Project's active default environment"
        )
        raise SecondaryCoordinatorError(
            f"Operation {operation_id!r} requires {origin} {selected!r}. "
            "Cross-interpreter resume is not yet supported for this translator; "
            "reopen the project under the selected environment."
        )
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "resume", "request_id": uuid4().hex,
        "project_path": str(project.path.resolve()),
        "environment_id": selected, "translator": TRANSLATOR_TYPE,
        "scientific_state": None, "source_ids": None,
        "settings": None, "operation_id": operation_id,
    }
    return _delegate(project, selected, message, expected_operation_id=operation_id)


def _child_run(message: dict[str, Any]) -> dict[str, Any]:
    _validate_request(message)
    from text_analysis_lab.core.project import Project
    from text_analysis_lab.translators.text_length import TextLength

    with Project.open(message["project_path"]) as project:
        selected = message["environment_id"]
        specification = project.environments.get(selected)["observation"]
        observed = normalize_environment(observe_environment())
        if specification["kind"] == "portable_locked":
            # Revalidate in the actual child process and explicitly associate
            # its observed inventory with the portable identity before writing.
            project.environments.verify_current_portable(selected)
        elif "env_" + fingerprint_observation(observed) != selected:
            raise EnvironmentRegistryError(
                "Child's installed software differs from the selected observed environment."
            )
        if project.environments.opening_id != selected:
            raise EnvironmentRegistryError(
                "Child's actual opening environment does not match requested environment."
            )
        if message["kind"] == "resume":
            opid = message["operation_id"]
            op = project.catalog.get_operation(opid)
            descriptor = _read_json(
                project.storage.operation_descriptor_path(opid).read_text(encoding="utf-8")
            )
            if (op["environment_id"] not in (None, selected)
                    or op["operation_type"] != "translate"
                    or descriptor.get("translator_class") != {
                "module": TRANSLATOR_MODULE, "qualname": TRANSLATOR_TYPE
            }):
                raise SecondaryCoordinatorError(
                    "Only TextLength operations under the selected software environment can resume."
                )
            outputs = project.resume_operation(
                opid, _coordinator_request_id=message["request_id"],
            )
        else:
            state = message["scientific_state"]
            translator = TextLength.from_json_state(state)
            if translator.to_json_state() != state:
                raise SecondaryCoordinatorError("TextLength scientific state round-trip mismatch.")
            # This is the only operation owner. The parent has not allocated IDs.
            outputs = project.translate(
                translator,
                project.get_artifact(message["source_ids"]["source"]),
                environment=project.environments.opening_id,
                **message["settings"],
                _coordinator_request_id=message["request_id"],
            )
            opid = next(iter(outputs.values())).descriptor["operation_id"]
        operation = project.catalog.get_operation(opid)
        result_kind = "executed"
        if message["kind"] == "translate" and (
            operation["coordinator_request_id"] != message["request_id"]
        ):
            # The normal TeAL engine can return an existing alias without
            # allocating a new operation. Accept that *only* when its durable
            # alias matches the returned artifact and overwrite was disabled.
            alias = message["settings"]["alias"]
            if (
                not isinstance(alias, str) or message["settings"]["overwrite"]
                or set(outputs) != {"output"}
                or project.resolve_artifact_id(alias) != outputs["output"].artifact_id
                or project.catalog.operation_for_coordinator_request(message["request_id"])
                is not None
            ):
                raise SecondaryCoordinatorError(
                    "Child returned an operation without matching request correlation."
                )
            result_kind = "reused"
        if operation["status"] != "complete" or (
            result_kind == "executed"
            and operation["environment_id"] != project.environments.opening_id
        ):
            raise SecondaryCoordinatorError(
                "Child operation did not complete with its executor provenance."
            )
        return {
            "protocol_version": PROTOCOL_VERSION, "kind": "result",
            "request_id": message["request_id"], "status": "complete",
            "operation_id": opid, "environment_id": project.environments.opening_id,
            "platform_id": (
                project.platforms.opening_id if result_kind == "executed"
                else operation["platform_id"]
            ),
            "result_kind": result_kind,
            "output_artifact_ids": {k: v.artifact_id for k, v in outputs.items()},
        }


def main() -> int:
    action = sys.argv[1] if len(sys.argv) == 2 else ""
    try:
        if action == "--probe":
            with contextlib.redirect_stdout(sys.stderr):
                payload = _probe_payload()
        elif action == "--run":
            message = _read_json(sys.stdin.read())
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    payload = _child_run(message)
            except Exception as exc:
                with contextlib.redirect_stdout(sys.stderr):
                    request_id = message.get("request_id")
                    operation_id = None
                    if isinstance(message.get("project_path"), str):
                        # Reopen only to inspect correlated failure history.
                        try:
                            from text_analysis_lab.core.project import Project
                            with Project.open(message["project_path"]) as project:
                                if message.get("kind") == "resume":
                                    operation_id = message.get("operation_id")
                                elif isinstance(request_id, str):
                                    row = project.catalog.operation_for_coordinator_request(request_id)
                                    operation_id = None if row is None else row["operation_id"]
                        except Exception:
                            pass
                payload = {
                    "protocol_version": PROTOCOL_VERSION, "kind": "error",
                    "request_id": request_id, "status": "error",
                    "phase": "after_registration" if operation_id else "pre_execution",
                    "operation_id": operation_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(_json(payload))
                return 1
        else:
            raise SecondaryCoordinatorError("Unsupported coordinator command.")
        print(_json(payload))
        return 0
    except Exception as exc:
        print(f"Secondary coordinator startup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
