"""Local multi-worker world orchestration for ``dryml.dispatch``."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dryml.records.execution import persistence_safe_execution_error
from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldSpec, attach_world_allocation_id, attach_world_id, local_inventory, make_world_allocation_spec, make_world_spec, validate_world_spec

from .backends import LocalSubprocessFuture, build_worker_command
from .errors import DispatchLaunchError, DispatchPlanningError
from .protocol import DispatchResult, ExecutionEnvelope, WorkerHandshakeResponse, WorkerResponse, read_json_file, save_envelope, write_json_file


LOCAL_WORLD_BACKEND_IDENTITY = {"name": "dryml.local_world", "kind": "local_world", "version": "1"}
_FATAL_STATUSES = frozenset({"failed", "timeout", "unsupported", "cancelled"})
_MAX_LOCAL_WORLD_WORKERS = 4096
_MAX_LOCAL_WORLD_CPU_ASSIGNMENTS = 4096


@dataclass(frozen=True, order=True, slots=True)
class WorldWorkerKey:
    """Stable JSON-friendly key for one local world worker.

    Attributes:
        role: Requested world role name.
        replica: Zero-based replica index within the role.
        rank: Global worker rank.
        local_rank: Same-host rank used by local execution.
    """

    role: str
    replica: int
    rank: int
    local_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise DispatchPlanningError("world worker role must be a non-empty string", context={"role": self.role})
        _nonneg_int(self.replica, "replica")
        _nonneg_int(self.rank, "rank")
        _nonneg_int(self.local_rank, "local_rank")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "WorldWorkerKey":
        """Build a worker key from JSON protocol data."""

        if not isinstance(data, Mapping):
            raise DispatchPlanningError("world worker key must be a mapping", context={"type": type(data).__name__})
        role = data.get("role")
        if not isinstance(role, str) or not role:
            raise DispatchPlanningError("world worker role must be a non-empty string", context={"role": role})
        return cls(role, _nonneg_int(data.get("replica"), "replica"), _nonneg_int(data.get("rank"), "rank"), _nonneg_int(data.get("local_rank"), "local_rank"))

    def to_json(self) -> dict[str, Any]:
        """Return a canonical JSON representation."""

        return {"role": self.role, "replica": self.replica, "rank": self.rank, "local_rank": self.local_rank}

    def label(self) -> str:
        """Return a filesystem-safe worker label."""

        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in self.role)
        return f"{safe}-{self.replica}-r{self.rank}"


@dataclass(frozen=True, slots=True)
class LocalWorldAllocationPlan:
    """Expanded deterministic local allocation result.

    Attributes:
        world_spec: Canonical requested-world envelope.
        world_allocation: Concrete assigned resource identifiers.
        world_allocation_spec: Canonical allocation envelope.
        worker_keys: Deterministically ranked role/replica workers.
    """

    world_spec: Mapping[str, Any]
    world_allocation: WorldAllocation
    world_allocation_spec: Mapping[str, Any]
    worker_keys: tuple[WorldWorkerKey, ...]


@dataclass(frozen=True, slots=True)
class WorkerLaunchPlan:
    """Launch plan for one role/replica worker in a local world.

    Attributes:
        key: Worker role, replica, and rank identity.
        dispatch_spec: Shared canonical dispatch intent.
        execution_recipe: Shared backend execution recipe.
        envelope: Worker-specific operation and allocation envelope.
        store: Store used for worker inputs, outputs, and provenance.
    """

    key: WorldWorkerKey
    dispatch_spec: Mapping[str, Any]
    execution_recipe: Mapping[str, Any]
    envelope: ExecutionEnvelope
    store: Any


@dataclass(frozen=True, slots=True)
class LocalWorldPlan:
    """Resolved launch plan for a coordinated local worker group.

    Attributes:
        dispatch_spec: Canonical dispatch intent.
        execution_recipe: Local-world backend recipe.
        operation_spec: Canonical operation executed by each worker.
        world_spec: Requested world kept distinct from allocation.
        world_allocation_spec: Concrete local resource assignment.
        worker_plans: Ordered worker-specific launch plans.
        store: Shared local store used by workers.
        group_work_dir: Optional backend-owned group directory.
        preserve_work_dir: Whether explicit close preserves that directory.
    """

    dispatch_spec: Mapping[str, Any]
    execution_recipe: Mapping[str, Any]
    operation_spec: Mapping[str, Any]
    world_spec: Mapping[str, Any]
    world_allocation_spec: Mapping[str, Any]
    worker_plans: tuple[WorkerLaunchPlan, ...]
    store: Any
    group_work_dir: str | None = None
    preserve_work_dir: bool = False


@dataclass(frozen=True, slots=True)
class WorldDispatchResult:
    """Aggregate result for explicit local-world dispatch.

    Attributes:
        status: Aggregate terminal status.
        dispatch_id: Persisted dispatch identity when available.
        recipe_id: Persisted execution-recipe identity when available.
        world_id: Requested-world identity.
        world_allocation_id: Actual allocation identity.
        primary: Deterministic primary worker result.
        workers: Results keyed by role, replica, and rank.
        execution_record_ids: Worker execution record identities.
        produced_record_ids: Product record identities.
        diagnostics: Aggregate backend diagnostics.
        error: Structured aggregate failure.
        cancellation: Structured cancellation details.
    """

    status: str
    dispatch_id: str | None
    recipe_id: str | None
    world_id: str | None
    world_allocation_id: str | None
    primary: DispatchResult | None
    workers: Mapping[WorldWorkerKey, DispatchResult]
    execution_record_ids: tuple[str, ...] = ()
    produced_record_ids: tuple[str, ...] = ()
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    error: Mapping[str, Any] | None = None
    cancellation: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready aggregate result."""

        return {
            "status": self.status,
            "dispatch_id": self.dispatch_id,
            "recipe_id": self.recipe_id,
            "world_id": self.world_id,
            "world_allocation_id": self.world_allocation_id,
            "primary": self.primary.to_json() if self.primary else None,
            "workers": [{"key": key.to_json(), "result": result.to_json()} for key, result in sorted(self.workers.items())],
            "execution_record_ids": list(self.execution_record_ids),
            "produced_record_ids": list(self.produced_record_ids),
            "diagnostics": list(self.diagnostics),
            "error": self.error,
            "cancellation": self.cancellation,
        }


class LocalWorldBackend:
    """Popen-based coordinator for one same-host local worker group."""

    name = "local_world"

    def __init__(self, *, preserve_work_dir: bool = False, handshake_timeout: float = 10.0, start_timeout: float = 10.0, cancel_grace: float = 0.5):
        self.preserve_work_dir = preserve_work_dir
        self.handshake_timeout = handshake_timeout
        self.start_timeout = start_timeout
        self.cancel_grace = cancel_grace

    def submit(self, plan: LocalWorldPlan) -> "LocalWorldFuture":
        """Launch all workers and return a group future."""

        group_dir = plan.group_work_dir or tempfile.mkdtemp(prefix="dryml-local-world-")
        futures: dict[WorldWorkerKey, LocalSubprocessFuture] = {}
        try:
            workers_dir = os.path.join(group_dir, "workers")
            os.makedirs(workers_dir, exist_ok=True)
            start_path = os.path.join(group_dir, "start.json")
            cancel_path = os.path.join(group_dir, "cancel.json")
            write_json_file(os.path.join(group_dir, "group.json"), {"dispatch_id": plan.dispatch_spec.get("id"), "recipe_id": plan.execution_recipe.get("id"), "world_id": plan.world_spec.get("id"), "world_allocation_id": plan.world_allocation_spec.get("id"), "worker_count": len(plan.worker_plans)})
            for worker_plan in plan.worker_plans:
                worker_dir = os.path.join(workers_dir, worker_plan.key.label())
                os.makedirs(worker_dir, exist_ok=True)
                request_path = os.path.join(worker_dir, "request.json")
                handshake_path = os.path.join(worker_dir, "handshake.json")
                response_path = os.path.join(worker_dir, "response.json")
                stdout_path = os.path.join(worker_dir, "stdout.txt")
                stderr_path = os.path.join(worker_dir, "stderr.txt")
                envelope = _with_coordination(worker_plan.envelope, group_dir=group_dir, start_path=start_path, cancel_path=cancel_path, worker_key=worker_plan.key, start_timeout=self.start_timeout)
                launch_plan = WorkerLaunchPlan(worker_plan.key, worker_plan.dispatch_spec, worker_plan.execution_recipe, envelope, worker_plan.store)
                save_envelope(request_path, envelope)
                cmd, child_env = build_worker_command(envelope.environment_spec)
                child_env.update({str(key): str(value) for key, value in (envelope.allocation_view.get("env") or {}).items()})
                cmd.extend(["-m", "dryml.dispatch.worker", "--request", request_path, "--handshake", handshake_path, "--response", response_path])
                _report("dryml.dispatch.world.launch", "Launching local worker", operation_id=envelope.operation_id, data={"worker": worker_plan.key.to_json(), "work_dir": worker_dir})
                stdout = open(stdout_path, "w", encoding="utf-8")
                stderr = open(stderr_path, "w", encoding="utf-8")
                try:
                    process = subprocess.Popen(cmd, env=child_env, stdout=stdout, stderr=stderr, cwd=worker_dir, start_new_session=(os.name == "posix"))
                finally:
                    stdout.close()
                    stderr.close()
                futures[worker_plan.key] = LocalSubprocessFuture(
                    process,
                    launch_plan,
                    worker_dir,
                    request_path,
                    handshake_path,
                    response_path,
                    stdout_path,
                    stderr_path,
                    True,
                    cancel_grace=self.cancel_grace,
                    handshake_timeout=self.handshake_timeout,
                    process_group=(os.name == "posix"),
                    process_tree=True,
                )
        except BaseException as exc:
            for future in futures.values():
                _cancel_worker_safely(future, grace=self.cancel_grace, reason="launch_failure")
            _cleanup_worker_paths(plan)
            if not self.preserve_work_dir and not plan.preserve_work_dir:
                shutil.rmtree(group_dir, ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            if isinstance(exc, DispatchLaunchError):
                raise
            raise DispatchLaunchError("failed to launch local world worker group", context={"error": str(exc)}) from exc
        return LocalWorldFuture(plan=plan, group_work_dir=group_dir, start_path=start_path, cancel_path=cancel_path, workers=futures, preserve_work_dir=self.preserve_work_dir or plan.preserve_work_dir, handshake_timeout=self.handshake_timeout, start_timeout=self.start_timeout, cancel_grace=self.cancel_grace)


@dataclass(slots=True)
class LocalWorldFuture:
    """Future coordinating handshakes, start barrier, results, and cancellation."""

    plan: LocalWorldPlan
    group_work_dir: str
    start_path: str
    cancel_path: str
    workers: Mapping[WorldWorkerKey, LocalSubprocessFuture]
    preserve_work_dir: bool = False
    handshake_timeout: float = 10.0
    start_timeout: float = 10.0
    cancel_grace: float = 0.5
    _started: bool = False
    _cancelled: bool = False
    _result: WorldDispatchResult | None = None
    _cancel_reason: str | None = None
    _control_failure_status: str | None = None
    _control_diagnostics: tuple[Mapping[str, Any], ...] = ()

    def wait_for_handshakes(self, timeout: float | None = None) -> Mapping[WorldWorkerKey, WorkerHandshakeResponse | None]:
        """Wait for every worker handshake, then release the start barrier."""

        if self._started:
            return {key: future._handshake for key, future in self.workers.items()}
        if self._cancelled:
            return {key: future._handshake for key, future in self.workers.items()}
        _report("dryml.dispatch.world.handshake.wait", "Waiting for worker handshakes", operation_id=self.plan.operation_spec.get("id"), data={"worker_count": len(self.workers)})
        deadline = time.monotonic() + (self.handshake_timeout if timeout is None else timeout)
        handshakes: dict[WorldWorkerKey, WorkerHandshakeResponse | None] = {key: None for key in self.workers}
        while True:
            if self._cancelled:
                return handshakes
            for key, future in self.workers.items():
                if handshakes[key] is not None:
                    continue
                if os.path.exists(future.handshake_path):
                    try:
                        handshake = WorkerHandshakeResponse.from_json(read_json_file(future.handshake_path))
                        _validate_handshake_allocation(key, handshake, self.plan)
                        future._handshake = handshake
                        handshakes[key] = handshake
                    except Exception as exc:
                        handshakes[key] = None
                        self._set_control_failure("failed", "handshake_failed", key=key, error=exc)
                        self.cancel(reason="handshake_failed")
                        return handshakes
                elif future.done():
                    self._set_control_failure("failed", "missing_handshake", key=key)
                    self.cancel(reason="missing_handshake")
                    return handshakes
            bad = {key: handshake for key, handshake in handshakes.items() if handshake is not None and handshake.status != "ok"}
            if bad:
                status = "unsupported" if any(handshake.status == "unsupported" for handshake in bad.values()) else "failed"
                diagnostics = tuple(
                    {"message": "worker handshake was not ok", "reason": "handshake_unsupported" if status == "unsupported" else "handshake_failed", "worker": key.to_json(), "handshake_status": handshake.status, "handshake_diagnostics": list(handshake.diagnostics)}
                    for key, handshake in bad.items()
                )
                self._set_control_failure(status, "handshake_unsupported" if status == "unsupported" else "handshake_failed", diagnostics=diagnostics)
                self.cancel(reason="handshake_unsupported")
                return handshakes
            if all(handshake is not None and handshake.status == "ok" for handshake in handshakes.values()):
                _report("dryml.dispatch.world.start", "Starting local world workers", operation_id=self.plan.operation_spec.get("id"), data={"worker_count": len(self.workers)})
                write_json_file(self.start_path, {"status": "ok", "started_at": time.time()})
                self._started = True
                return handshakes
            if time.monotonic() >= deadline:
                self._set_control_failure("timeout", "handshake_timeout")
                self._timeout_live(message="local world worker handshake timed out")
                return handshakes
            time.sleep(0.01)

    def result(self, timeout: float | None = None) -> WorldDispatchResult:
        """Wait for the group and return an aggregate result."""

        if self._result is not None:
            return self._result
        try:
            handshakes = self.wait_for_handshakes(timeout=self.handshake_timeout)
            if self._cancelled and not self._started:
                status_override = self._control_failure_status
                cancellation = None if status_override is not None else {"requested": True, "reason": self._cancel_reason or "cancelled"}
                self._result = self._aggregate(handshakes=handshakes, status_override=status_override, cancellation=cancellation)
                return self._result
            deadline = None if timeout is None else time.monotonic() + timeout
            fatal_seen = False
            while True:
                for key, future in self.workers.items():
                    if future.done() and future._response is None:
                        future._read_response()
                        if future._response and future._response.status in _FATAL_STATUSES:
                            _report("dryml.dispatch.world.worker.failed", "Worker failed; cancelling local world", operation_id=self.plan.operation_spec.get("id"), data={"worker": key.to_json(), "status": future._response.status})
                            self._cancel_live(reason="sibling_failure")
                            fatal_seen = True
                if all(future.done() or future._response is not None for future in self.workers.values()):
                    break
                if fatal_seen:
                    for future in self.workers.values():
                        try:
                            future.kill()
                            future._wait(self.cancel_grace)
                        except Exception:
                            pass
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._timeout_live()
                    self._result = self._aggregate(handshakes=handshakes, status_override="timeout", cancellation={"requested": True, "reason": "timeout"})
                    return self._result
                time.sleep(0.01)
            self._result = self._aggregate()
            return self._result
        except KeyboardInterrupt:
            self.cancel(reason="KeyboardInterrupt")
            raise
        except BaseException:
            self.cancel(reason="parent_error")
            raise
        finally:
            self._cleanup()

    def cancel(self, reason: str = "user") -> bool:
        """Cancel all live workers and write a cooperative cancel marker."""

        self._cancelled = True
        self._cancel_reason = reason
        _report("dryml.dispatch.world.cancel", "Cancelling local worker group", operation_id=self.plan.operation_spec.get("id"), data={"reason": reason})
        try:
            write_json_file(self.cancel_path, {"cancelled": True, "reason": reason, "time": time.time()})
        except Exception:
            pass
        cancelled = False
        for future in self.workers.values():
            cancelled = _cancel_worker_safely(future, grace=self.cancel_grace, reason=reason) or cancelled
        # Cancellation is terminal for normalized pickle/source payloads even
        # when the caller later asks for the aggregate worker responses.
        _cleanup_worker_paths(self.plan)
        return cancelled

    def close(self, reason: str = "user") -> bool:
        """Cancel the group and remove its work directory without awaiting results."""

        cancelled = self.cancel(reason=reason)
        self._cleanup(force=True)
        return cancelled

    def done(self) -> bool:
        """Return whether every worker is done."""

        return all(future.done() for future in self.workers.values())

    def _cancel_live(self, *, reason: str) -> None:
        self._cancelled = True
        self._cancel_reason = reason
        try:
            write_json_file(self.cancel_path, {"cancelled": True, "reason": reason, "time": time.time()})
        except Exception:
            pass
        for future in self.workers.values():
            _cancel_worker_safely(future, grace=self.cancel_grace, reason=reason)

    def _timeout_live(self, *, message: str = "local world dispatch timed out") -> None:
        self._cancelled = True
        self._cancel_reason = "timeout"
        try:
            write_json_file(self.cancel_path, {"cancelled": True, "reason": "timeout", "time": time.time()})
        except Exception:
            pass
        for future in self.workers.values():
            was_live = not future.done()
            _cancel_worker_safely(future, grace=self.cancel_grace, reason="timeout", record=False)
            if future._response is None:
                future._read_response()
            if was_live or future._response is None or future._response.status == "cancelled":
                future._response = future._parent_failure_response("timeout", error={"type": "TimeoutError", "message": message})

    def _set_control_failure(self, status: str, reason: str, *, key: WorldWorkerKey | None = None, error: BaseException | None = None, diagnostics: tuple[Mapping[str, Any], ...] = ()) -> None:
        if self._control_failure_status is None:
            self._control_failure_status = status
        items = list(self._control_diagnostics)
        if diagnostics:
            items.extend(dict(item) for item in diagnostics)
        else:
            item: dict[str, Any] = {"message": "local world handshake failed", "reason": reason}
            if key is not None:
                item["worker"] = key.to_json()
            if error is not None:
                failure = persistence_safe_execution_error(error)
                item["error_type"] = failure["type"]
                item["code"] = failure["metadata"]["code"]
            items.append(item)
        self._control_diagnostics = tuple(items)

    def _aggregate(self, *, handshakes: Mapping[WorldWorkerKey, WorkerHandshakeResponse | None] | None = None, status_override: str | None = None, cancellation: Mapping[str, Any] | None = None) -> WorldDispatchResult:
        worker_results: dict[WorldWorkerKey, DispatchResult] = {}
        diagnostics: list[Mapping[str, Any]] = [dict(item) for item in self._control_diagnostics]
        first_error: Mapping[str, Any] | None = None
        first_cancel: Mapping[str, Any] | None = dict(cancellation) if cancellation else None
        for key, future in self.workers.items():
            if future._response is None:
                future._read_response()
            if future._response is None:
                future._response = future._parent_failure_response("failed", error={"type": "WorkerProtocolError", "message": "worker produced no response"})
            response = future._response
            future._persist_logs(response.execution_record_id)
            result = DispatchResult.from_worker_response(response)
            worker_results[key] = result
            diagnostics.extend(result.diagnostics)
            if first_error is None and result.error is not None:
                first_error = result.error
            if first_cancel is None and result.cancellation is not None:
                first_cancel = result.cancellation
        status = status_override or _aggregate_status(result.status for result in worker_results.values())
        if status_override is None and handshakes is not None and any(handshake is None or handshake.status != "ok" for handshake in handshakes.values()):
            status = self._control_failure_status or ("cancelled" if self._cancelled else "failed")
        if self._control_failure_status is not None and status == self._control_failure_status:
            first_cancel = None
        primary = _select_primary(worker_results)
        execution_record_ids = tuple(result.execution_record_id for result in worker_results.values() if result.execution_record_id)
        produced_record_ids = tuple(record_id for result in worker_results.values() for record_id in result.produced_record_ids)
        _report("dryml.dispatch.world.complete", "Local world dispatch complete", operation_id=self.plan.operation_spec.get("id"), data={"status": status, "worker_statuses": {key.label(): result.status for key, result in worker_results.items()}, "world_allocation_id": self.plan.world_allocation_spec.get("id"), "execution_record_ids": list(execution_record_ids)})
        return WorldDispatchResult(status=status, dispatch_id=self.plan.dispatch_spec.get("id"), recipe_id=self.plan.execution_recipe.get("id"), world_id=self.plan.world_spec.get("id"), world_allocation_id=self.plan.world_allocation_spec.get("id"), primary=primary, workers=worker_results, execution_record_ids=execution_record_ids, produced_record_ids=produced_record_ids, diagnostics=tuple(diagnostics), error=first_error, cancellation=first_cancel)

    def _cleanup(self, *, force: bool = False) -> None:
        _cleanup_worker_paths(self.plan)
        for future in self.workers.values():
            if isinstance(future, LocalSubprocessFuture):
                future._cleanup()
        if force or not self.preserve_work_dir:
            shutil.rmtree(self.group_work_dir, ignore_errors=True)


def _cleanup_worker_paths(plan: LocalWorldPlan) -> None:
    """Remove per-worker normalization artifacts after an abnormal path."""

    for worker_plan in plan.worker_plans:
        for path in worker_plan.envelope.launch.get("cleanup_paths") or ():
            shutil.rmtree(path, ignore_errors=True)


def _cancel_worker_safely(future: LocalSubprocessFuture, *, grace: float, reason: str, record: bool = True) -> bool:
    """Cancel one worker without allowing persistence failures to strand peers."""

    try:
        return future.cancel(grace=grace, reason=reason, record=record)
    except BaseException:
        try:
            future.kill()
        except BaseException:
            pass
        return False


def normalize_world_spec(world: Mapping[str, Any] | WorldSpec | None) -> Mapping[str, Any]:
    """Normalize a world object, spec envelope, or raw roles mapping."""

    if world is None:
        return attach_world_id(make_world_spec({"worker": {"replicas": 1, "process": {"resources": {"cpus": 0}}}}, backend={"kind": "local_world", "parameters": {}}))
    if isinstance(world, WorldSpec):
        _validate_local_world_backend(world)
        return attach_world_id(make_world_spec(world))
    if not isinstance(world, Mapping):
        raise DispatchPlanningError("world must be a WorldSpec or mapping", context={"type": type(world).__name__})
    if world.get("schema") == "dryml.world.v1":
        validate_world_spec(world)
        _validate_local_world_backend(WorldSpec.from_data(world["payload"]))
        return attach_world_id(world)
    if set(world).issubset({"roles", "backend"}) and "roles" in world:
        world_spec = WorldSpec.from_data(world)
        _validate_local_world_backend(world_spec)
        return attach_world_id(make_world_spec(world_spec))
    return attach_world_id(make_world_spec(world, backend={"kind": "local_world", "parameters": {}}))


def _validate_local_world_backend(world: WorldSpec) -> None:
    """Reject world backends that the same-host local allocator cannot enact."""

    kind = world.backend.get("kind")
    if kind not in {"local", "local_world"}:
        raise DispatchPlanningError(
            "local-world dispatch supports only local or local_world backends",
            context={"backend": dict(world.backend), "kind": kind},
        )
    parameters = world.backend.get("parameters", {})
    if not isinstance(parameters, Mapping) or parameters:
        raise DispatchPlanningError(
            "local-world dispatch cannot enact requested backend parameters",
            context={"backend": dict(world.backend)},
        )


def is_multi_worker_world(world_spec: Mapping[str, Any]) -> bool:
    """Return whether a normalized world spec requests multiple workers."""

    world_obj = WorldSpec.from_data(world_spec["payload"])
    counts = [role.replicas for role in world_obj.roles.values()]
    return len(counts) > 1 or sum(counts) > 1


def validate_local_world_feasibility(world: Mapping[str, Any] | WorldSpec | None, *, inventory: LocalResourceInventory | None = None, oversubscribe: bool = False, allocation_backend_kind: str = "local_world") -> None:
    """Validate whether a local allocator can enact a requested world.

    Args:
        world: Raw, canonical, or enveloped requested world.
        inventory: Optional injected process-visible resource inventory.
        oversubscribe: Permit CPU reuse for explicit advanced local-world plans.
        allocation_backend_kind: ``"local_world"`` or ``"local_subprocess"``.

    Raises:
        DispatchPlanningError: If structure, backend, or capacity is unsupported.
    """

    world_spec = normalize_world_spec(world)
    world_obj = WorldSpec.from_data(world_spec["payload"])
    inv = inventory or local_inventory()
    _validate_local_resource_requests(world_obj, inv, oversubscribe, allocation_backend_kind)


def allocate_local_world(world: Mapping[str, Any] | WorldSpec | None, *, inventory: LocalResourceInventory | None = None, oversubscribe: bool = False, allocation_backend_kind: str = "local_world", requested_world_id: str | None = None) -> LocalWorldAllocationPlan:
    """Expand a feasible world into deterministic local worker assignments.

    Args:
        world: Raw, canonical, or enveloped requested world.
        inventory: Optional injected process-visible resource inventory.
        oversubscribe: Permit CPU reuse for explicit advanced local-world plans.
        allocation_backend_kind: Provenance backend for the allocation.
        requested_world_id: Optional requested-world identity retained in workers.

    Returns:
        Worker keys, canonical requested world, and actual allocation records.

    Raises:
        DispatchPlanningError: If the local allocator cannot enact the world.
    """

    _report("dryml.dispatch.world.allocate", "Allocating local world resources")
    world_spec = normalize_world_spec(world)
    world_obj = WorldSpec.from_data(world_spec["payload"])
    inv = inventory or local_inventory()
    _validate_local_resource_requests(world_obj, inv, oversubscribe, allocation_backend_kind)
    cpu_cursor = 0
    memory_cursor = 0
    accelerator_cursors = {key: 0 for key in inv.accelerators}
    world_size = sum(role.replicas for role in world_obj.roles.values())
    roles: dict[str, list[dict[str, Any]]] = {}
    keys: list[WorldWorkerKey] = []
    rank = 0
    role_sizes = {name: role.replicas for name, role in world_obj.roles.items()}
    for role_name in sorted(world_obj.roles):
        role = world_obj.roles[role_name]
        roles[role_name] = []
        for replica in range(role.replicas):
            key = WorldWorkerKey(role_name, replica, rank, rank)
            keys.append(key)
            requested_cpus = role.process.resources.cpus or 1
            requested_memory = role.process.resources.memory
            if requested_memory is not None:
                memory_cursor += requested_memory
            if oversubscribe:
                cpus = tuple(inv.cpus[(cpu_cursor + idx) % len(inv.cpus)] for idx in range(requested_cpus))
                cpu_cursor = (cpu_cursor + requested_cpus) % len(inv.cpus)
            else:
                cpus = tuple(inv.cpus[cpu_cursor : cpu_cursor + requested_cpus])
                cpu_cursor += requested_cpus
            accelerators: dict[str, tuple[str | int, ...]] = {}
            for acc_name, count in role.process.resources.accelerators.items():
                available = inv.accelerators.get(acc_name, ())
                cursor = accelerator_cursors.get(acc_name, 0)
                accelerators[acc_name] = tuple(available[cursor : cursor + count])
                accelerator_cursors[acc_name] = cursor + count
            env = dict(role.process.env)
            env.update(
                {
                    "DRYML_WORLD_ID": str(requested_world_id or world_spec.get("id")),
                    "DRYML_WORLD_ROLE": role_name,
                    "DRYML_WORLD_REPLICA": str(replica),
                    "DRYML_WORLD_RANK": str(rank),
                    "DRYML_WORLD_LOCAL_RANK": str(rank),
                    "DRYML_WORLD_SIZE": str(world_size),
                    "DRYML_WORLD_ROLE_SIZE": str(role_sizes[role_name]),
                }
            )
            allocation = {
                "replica": replica,
                "rank": rank,
                "local_rank": rank,
                "resources": {"cpus": list(cpus), "accelerators": {name: list(values) for name, values in sorted(accelerators.items())}},
                "environment": role.process.environment,
                "env": env,
                "metadata": {"allocation_policy": "disjoint_local" if not oversubscribe else "oversubscribed_local", "world_size": world_size, "role_size": role_sizes[role_name]},
            }
            if role.process.resources.memory is not None:
                allocation["resources"]["memory"] = role.process.resources.to_data()["memory"]
            roles[role_name].append(allocation)
            rank += 1
    backend = dict(LOCAL_WORLD_BACKEND_IDENTITY)
    if allocation_backend_kind == "local_subprocess":
        backend.update({"name": "dryml.local_subprocess", "kind": "local_subprocess"})
    backend["host"] = os.uname().nodename if hasattr(os, "uname") else "localhost"
    allocation_spec = attach_world_allocation_id(make_world_allocation_spec(roles, backend=backend, kind=f"{allocation_backend_kind}_allocation", metadata={"world_id": requested_world_id or world_spec.get("id")}))
    return LocalWorldAllocationPlan(world_spec=world_spec, world_allocation=WorldAllocation.from_data(allocation_spec["payload"]), world_allocation_spec=allocation_spec, worker_keys=tuple(keys))


def _validate_local_resource_requests(world: WorldSpec, inventory: LocalResourceInventory, oversubscribe: bool, allocation_backend_kind: str) -> None:
    """Check the shared deterministic assignment inputs before allocation."""

    if allocation_backend_kind not in {"local_world", "local_subprocess"}:
        raise DispatchPlanningError("unsupported local allocation backend kind", context={"kind": allocation_backend_kind})
    worker_count = sum(role.replicas for role in world.roles.values())
    if worker_count == 0:
        raise DispatchPlanningError("local world requires at least one worker")
    if worker_count > _MAX_LOCAL_WORLD_WORKERS:
        raise DispatchPlanningError(
            "local world worker count exceeds the bounded limit",
            context={"workers": worker_count, "limit": _MAX_LOCAL_WORLD_WORKERS},
        )
    cpu_assignments = 0
    for role_name, role in world.roles.items():
        process = role.process
        if process.environment is not None or process.runtime is not None or process.metadata:
            raise DispatchPlanningError(
                "local world allocation cannot enact role-specific process environment, runtime, or metadata settings",
                context={"role": role_name, "process": process.to_data()},
            )
        requested_cpus = process.resources.cpus or 1
        cpu_assignments += role.replicas * requested_cpus
    if cpu_assignments > _MAX_LOCAL_WORLD_CPU_ASSIGNMENTS:
        raise DispatchPlanningError(
            "local world CPU assignments exceed the bounded limit",
            context={"cpu_assignments": cpu_assignments, "limit": _MAX_LOCAL_WORLD_CPU_ASSIGNMENTS, "oversubscribe": oversubscribe},
        )
    cpu_cursor = memory_cursor = 0
    accelerator_cursors = {key: 0 for key in inventory.accelerators}
    for role_name in sorted(world.roles):
        role = world.roles[role_name]
        for replica in range(role.replicas):
            resources = role.process.resources
            if _has_positive_unsupported_resource(resources.devices) or _has_positive_unsupported_resource(resources.named):
                raise DispatchPlanningError(
                    "local world allocation does not support named devices or resources",
                    context={"role": role_name, "devices": dict(resources.devices), "named": dict(resources.named)},
                )
            requested_cpus = resources.cpus or 1
            requested_memory = resources.memory
            if requested_memory is not None:
                if requested_memory > 0 and inventory.memory is None:
                    raise DispatchPlanningError("local world memory request cannot be proven against unknown inventory", context={"role": role_name, "requested_memory": requested_memory})
                if not oversubscribe and inventory.memory is not None and memory_cursor + requested_memory > inventory.memory:
                    raise DispatchPlanningError("local world memory requests exceed disjoint inventory", context={"role": role_name, "replica": replica, "requested_memory": requested_memory, "remaining_memory": inventory.memory - memory_cursor})
                memory_cursor += requested_memory
            if not oversubscribe and requested_cpus > len(inventory.cpus):
                raise DispatchPlanningError("local world CPU request exceeds inventory", context={"role": role_name, "replica": replica, "requested_cpus": requested_cpus, "inventory_cpus": len(inventory.cpus)})
            if not oversubscribe and cpu_cursor + requested_cpus > len(inventory.cpus):
                raise DispatchPlanningError("local world CPU requests exceed disjoint inventory", context={"role": role_name, "replica": replica, "requested_cpus": requested_cpus, "remaining_cpus": len(inventory.cpus) - cpu_cursor})
            cpu_cursor = (cpu_cursor + requested_cpus) % len(inventory.cpus) if oversubscribe else cpu_cursor + requested_cpus
            for acc_name, count in resources.accelerators.items():
                available = inventory.accelerators.get(acc_name, ())
                cursor = accelerator_cursors.get(acc_name, 0)
                if cursor + count > len(available):
                    raise DispatchPlanningError("local world accelerator request exceeds explicit inventory", context={"role": role_name, "replica": replica, "accelerator": acc_name, "requested": count, "available": len(available)})
                accelerator_cursors[acc_name] = cursor + count


def _has_positive_unsupported_resource(values: Mapping[str, Any]) -> bool:
    """Return whether a concrete unsupported resource requests backend work."""

    for value in values.values():
        if isinstance(value, Mapping):
            if _has_positive_unsupported_resource(value):
                return True
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return True
        elif value:
            return True
    return False


def _with_coordination(envelope: ExecutionEnvelope, *, group_dir: str, start_path: str, cancel_path: str, worker_key: WorldWorkerKey, start_timeout: float) -> ExecutionEnvelope:
    launch = dict(envelope.launch)
    launch["coordination"] = {"group_id": Path(group_dir).name, "worker_key": worker_key.to_json(), "start_path": os.path.abspath(start_path), "cancel_path": os.path.abspath(cancel_path), "heartbeat_path": os.path.abspath(os.path.join(group_dir, "workers", worker_key.label(), "heartbeat.json")), "start_timeout": float(start_timeout)}
    return ExecutionEnvelope(dispatch_spec=envelope.dispatch_spec, execution_recipe=envelope.execution_recipe, operation_spec=envelope.operation_spec, environment_spec=envelope.environment_spec, runtime_spec=envelope.runtime_spec, allocation_view=envelope.allocation_view, store_refs=envelope.store_refs, transfer=envelope.transfer, result_policy=envelope.result_policy, record_policy=envelope.record_policy, reporting=envelope.reporting, handshake=envelope.handshake, launch=launch)


def _validate_handshake_allocation(key: WorldWorkerKey, handshake: WorkerHandshakeResponse, plan: LocalWorldPlan) -> None:
    expected = key.to_json()
    if handshake.worker_key != expected:
        raise DispatchPlanningError("worker handshake key does not match launch plan", context={"expected": expected, "observed": handshake.worker_key})
    if handshake.world_id != plan.world_spec.get("id"):
        raise DispatchPlanningError("worker handshake world_id does not match launch plan", context={"expected": plan.world_spec.get("id"), "observed": handshake.world_id})
    if handshake.world_allocation_id != plan.world_allocation_spec.get("id"):
        raise DispatchPlanningError("worker handshake world_allocation_id does not match launch plan", context={"expected": plan.world_allocation_spec.get("id"), "observed": handshake.world_allocation_id})


def _aggregate_status(statuses: Any) -> str:
    values = tuple(statuses)
    for status in ("timeout", "failed", "cancelled", "unsupported"):
        if status in values:
            return status
    return "ok" if values and all(status == "ok" for status in values) else "failed"


def _select_primary(results: Mapping[WorldWorkerKey, DispatchResult]) -> DispatchResult | None:
    if not results:
        return None
    for role in ("main", "worker"):
        for key, result in sorted(results.items()):
            if key.role == role and key.replica == 0:
                return result
    return results[sorted(results)[0]]


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DispatchPlanningError(f"{name} must be an integer >= 0", context={"value": value})
    return value


def _report(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


__all__ = [
    "LOCAL_WORLD_BACKEND_IDENTITY",
    "LocalResourceInventory",
    "LocalWorldAllocationPlan",
    "LocalWorldBackend",
    "LocalWorldFuture",
    "LocalWorldPlan",
    "WorkerLaunchPlan",
    "WorldDispatchResult",
    "WorldWorkerKey",
    "allocate_local_world",
    "is_multi_worker_world",
    "normalize_world_spec",
]
