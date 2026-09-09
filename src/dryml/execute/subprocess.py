"""Real one-shot local subprocess backend for additive Execute internals.

The backend accepts only already-spooled calls from :mod:`dryml.execute.executor`.
It never serializes a callable itself: a fresh worker receives the opaque payload
only after the loopback admission handshake grants its one-shot permit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Generic, TypeVar

import dill

from dryml.environments import EnvironmentRecord, EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from dryml.worlds import LocalResourceInventory, WorldAllocation

from ._process import OwnedProcess, _WindowsJob, minimal_environment, run_bounded
from ._protocol import (
    PROTOCOL_VERSION,
    WORKER_PROTOCOL_ID,
    BootstrapDescriptor,
    Correlation,
    FrameError,
    FrameState,
    FrameType,
    OwnerEnvelopeType,
    ProtocolConversation,
    decode_control,
    SocketFrameReader,
    encode_bootstrap_descriptor,
    encode_control,
    encode_owner_envelope,
    encode_frame,
)
from .accounting import RESOURCE_AUTHORITIES, Reservation, ResourceAuthority
from .admission import admit, plan_admission
from .backend import Backend
from .config import BackendConfig
from .discovery import discover_candidates, probe_candidate
from .errors import AdmissionError, CleanupError, ExecutionDeadlineExceeded, ExecutionError, ExecutionUncertainError, RemoteExecutionError
from .future import ExecutionFuture
from .models import DiscoverySnapshot, EnvironmentCandidate, ExecutionIssue, FeasiblePlan, ResourceAmounts, ResourceSnapshot, SubmittedCall
from .output import ExecutionOutput


T = TypeVar("T")
@dataclass(frozen=True, kw_only=True)
class SubProcessConfig(BackendConfig):
    """Describe one inert local subprocess backend.

    Args:
        python_executable: Optional existing direct worker interpreter. ``None`` is
            captured by :class:`~dryml.execute.executor.Executor` at initialization.

    Returns:
        :meth:`create_backend` returns an inert :class:`SubProcessBackend`.

    Failure behavior:
        Invalid paths are rejected at launch, not during construction. The config
        neither probes environments nor starts a worker.
    """

    python_executable: Path | None = None

    def __post_init__(self) -> None:
        """Validate the optional direct interpreter without touching the filesystem."""
        super().__post_init__()
        if self.python_executable is not None and not isinstance(self.python_executable, Path):
            raise TypeError("python_executable must be pathlib.Path or None")

    def create_backend(self) -> "SubProcessBackend":
        """Construct this config's inert local backend without launching work."""
        return SubProcessBackend(self)


class SubProcessFuture(ExecutionFuture[T]):
    """Execution Future with borrowed local launcher and worker identity.

    ``process`` is the real launcher :class:`subprocess.Popen`; ``pid`` is the
    worker-confirmed PID and can differ for wrappers such as ``conda run``. Native
    references are cleared during successful cleanup while common results/output
    remain retained.
    """

    def __init__(self, submission_id: str, *, output: ExecutionOutput, termination_timeout: float) -> None:
        """Create an inert concrete Future with no process association."""
        super().__init__(submission_id, output=output, termination_timeout=termination_timeout, cleanup_scope="owned-group")
        self._native_lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        """Return the borrowed launcher object, or ``None`` before launch/after cleanup."""
        with self._native_lock:
            return self._process

    @property
    def pid(self) -> int | None:
        """Return the worker-confirmed PID without initializing or launching work."""
        return self.snapshot().pid

    def _set_process(self, process: subprocess.Popen[bytes] | None) -> None:
        """Associate one backend-owned launcher without changing the common outcome."""
        with self._native_lock:
            self._process = process


@dataclass(slots=True)
class _Run:
    """Retain one launched owned group and its coordinator socket until cleanup."""

    future: SubProcessFuture[Any]
    owner: OwnedProcess
    connection: socket.socket | None
    lock: Lock
    worker_id: str | None = None
    issued: bool = False
    payload_transfer_started: bool = False
    qualified_terminal: bool = False
    cancelling: bool = False
    outcome_claimed: bool = False
    deadline_expired: bool = False
    terminal_event: Event | None = None
    reservation: Reservation | None = None


class SubProcessBackend(Backend):
    """Launch each accepted call in a fresh owned subprocess group.

    The backend is intentionally one-process-per-submission. It provides local
    loopback transport, output routing, deadline/cancellation escalation, and
    per-call cleanup; it does not inspect source code, provision an environment,
    or fall back to another backend.
    """

    def __init__(self, config: SubProcessConfig) -> None:
        """Store inert config and capture no environment or native process state."""
        if not isinstance(config, SubProcessConfig):
            raise TypeError("config must be a SubProcessConfig")
        self._config = config
        self._runs: dict[str, _Run] = {}
        self._known: set[str] = set()
        self._launching: dict[str, Event] = {}
        self._lock = Lock()
        self._started = False
        self._closed = False
        self._executor_id = secrets.token_hex(16)
        self._authority: ResourceAuthority = RESOURCE_AUTHORITIES.get("subprocess", "same-host-local")

    def start(self) -> None:
        """Mark this local backend initialized without probing or launching workers."""
        if self._closed:
            raise ExecutionError("subprocess backend is closed")
        self._started = True

    def capabilities(self) -> frozenset[str]:
        """Report local environment selection, output, and running cancellation support."""
        return frozenset({"environment_selection", "world_admission", "live_output", "running_cancellation"})

    def create_future(self, submission_id: str, output: ExecutionOutput) -> SubProcessFuture[Any]:
        """Create the stable inert concrete Future required before acceptance."""
        return SubProcessFuture(submission_id, output=output, termination_timeout=self._config.termination_timeout)

    def submit(self, call: SubmittedCall[T], *, future: ExecutionFuture[T]) -> None:
        """Schedule exactly the supplied accepted Future and return ``None``.

        A separate coordinator thread performs admission and launch. Any failure is
        published on this same Future and no alternate candidate is invoked after
        a worker has received GO.
        """
        if not isinstance(future, SubProcessFuture) or future.submission_id != call.submission_id or future.output is not call.output:
            raise ExecutionError("subprocess backend received a mismatched Future or call")
        with self._lock:
            if self._closed:
                raise ExecutionError("subprocess backend is closed")
            if call.submission_id in self._known:
                raise ExecutionError("duplicate subprocess submission ID")
            self._known.add(call.submission_id)
            self._launching[call.submission_id] = Event()
        try:
            Thread(target=self._run, args=(call, future), name="dryml-execute-subprocess", daemon=False).start()
        except BaseException:
            # No worker launch was accepted, so leave neither a duplicate-ID guard
            # nor a pending-cleanup association behind for the executor.
            with self._lock:
                self._known.discard(call.submission_id)
                self._launching.pop(call.submission_id, None)
            raise

    def discover(self, *, environment: EnvironmentRequirement | None = None, world: Any = None, timeout: float) -> DiscoverySnapshot:
        """Return fresh bounded local candidates and resource observations without reservation."""
        deadline = time.monotonic() + timeout
        resources = self._resource_inventory(deadline)
        inventory = resources.inventory
        candidates: list[EnvironmentCandidate] = []
        issues: list[ExecutionIssue] = []
        plans: list[FeasiblePlan] = []
        # Discovery is the sole API that lists allowed environments without a
        # requirement; ordinary unconstrained submission must not scan them.
        if environment is not None or self._config.automatic_environment_discovery or self._config.environment_candidates or self._config.environment_search_roots:
            for spec in discover_candidates(self._config, cwd=self._config.working_directory, interpreter=self._python(), deadline=deadline).specs:
                if time.monotonic() >= deadline:
                    issues.append(ExecutionIssue("discovery_timeout", "environment discovery exceeded its bounded deadline"))
                    break
                candidate = probe_candidate(spec, interpreter=self._python(), timeout=max(0.001, deadline - time.monotonic()), output_limit=self._config.owner_envelope_limit_bytes, deadline=deadline, termination_timeout=self._config.termination_timeout, read_chunk_bytes=self._config.process_read_chunk_bytes, poll_interval=self._config.process_poll_interval)
                candidates.append(candidate)
                if world is not None and inventory is not None and candidate.record is not None:
                    decision = plan_admission(environment=environment, world=world, record=candidate.record, inventory=inventory, deadline=deadline)
                    if decision.feasible and decision.world is not None:
                        plans.append(FeasiblePlan(candidate.key, decision.world, decision.allocation, decision.report))
        if world is not None and environment is None and inventory is not None:
            decision = plan_admission(world=world, inventory=inventory, deadline=deadline)
            if decision.feasible and decision.world is not None:
                plans.append(FeasiblePlan(None, decision.world, decision.allocation, decision.report))
        if time.monotonic() >= deadline:
            issues.append(ExecutionIssue("discovery_timeout", "discovery exceeded its total deadline"))
        return DiscoverySnapshot(datetime.now(timezone.utc), tuple(candidates), resources, tuple(plans), resources.complete and not issues, tuple(issues))

    def resources(self, *, timeout: float) -> ResourceSnapshot:
        """Observe current local inventory without launching or reserving a worker."""
        if timeout <= 0:
            raise TimeoutError("resource observation timeout elapsed")
        return self._resource_inventory(time.monotonic() + timeout)

    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        """Release only a known submission's owned process group and native handle."""
        with self._lock:
            run = self._runs.get(submission_id)
            known = submission_id in self._known
        if run is None and not known:
            raise ExecutionError("unknown subprocess submission")
        if run is None:
            launched = self._launching.get(submission_id)
            if launched is None or not launched.wait(timeout):
                raise CleanupError("subprocess launch ownership is still unresolved")
            with self._lock:
                run = self._runs.get(submission_id)
                known = submission_id in self._known
            if run is None:
                if known:
                    with self._lock:
                        self._known.discard(submission_id)
                        self._launching.pop(submission_id, None)
                return
        if not run.future.done():
            raise RuntimeError("cleanup requires a terminal subprocess execution")
        if not run.owner.reconcile(deadline=time.monotonic() + timeout, poll_interval=self._config.process_poll_interval):
            raise CleanupError("subprocess owned-group cleanup remains incomplete", execution=run.future)
        if run.reservation is not None:
            self._authority.release(
                submission_id,
                generation=run.reservation.generation,
                attempt=run.reservation.attempt,
                worker_id=run.worker_id,
                qualified_terminal=True,
            )
        if run.owner.job is not None:
            run.owner.job.close()
        if run.connection is not None:
            run.connection.close()
            run.connection = None
        run.future._set_process(None)
        run.future._set_association(worker_id=None, pid=None)
        with self._lock:
            self._runs.pop(submission_id, None)
            self._known.discard(submission_id)
            self._launching.pop(submission_id, None)

    def close(self, *, cancel: bool, timeout: float | None) -> None:
        """Reconcile only backend-owned groups, optionally requesting their cancellation."""
        with self._lock:
            self._closed = True
            runs = tuple(self._runs.values())
        deadline = time.monotonic() + (self._config.termination_timeout if timeout is None else timeout)
        for run in runs:
            if cancel and not run.future.done():
                self._cancel_run(run, deadline_error=False)
            if run.future.done():
                self.reconcile_cleanup(run.future.submission_id, timeout=max(0.001, deadline - time.monotonic()))

    def _run(self, call: SubmittedCall[T], future: SubProcessFuture[T]) -> None:
        """Perform one complete handshake, permit, invocation, and outcome receive."""
        if not future._begin_admission():
            return
        listener: socket.socket | None = None
        run: _Run | None = None
        reservation: Reservation | None = None
        try:
            reservation = self._reserve(call, future)
            if reservation is None:
                return
            if reservation.allocation is not None:
                future._set_association(allocation=reservation.allocation)
            executable, candidate = self._select_environment(call)
            if time.monotonic() >= call.admission_deadline:
                raise AdmissionError("admission deadline exceeded")
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(max(0.001, call.admission_deadline - time.monotonic()))
            correlation = Correlation(call.submission_id, 0, 1)
            descriptor = BootstrapDescriptor(correlation, secrets.token_hex(32), "127.0.0.1", listener.getsockname()[1], self._config.control_header_limit_bytes, self._config.owner_envelope_limit_bytes, self._config.admission_message_limit_bytes, self._config.invocation_limit_bytes, self._config.result_limit_bytes, self._config.output_frame_limit_bytes, int(self._config.output_final_timeout * 1000))
            encoded_descriptor = base64.urlsafe_b64encode(encode_bootstrap_descriptor(descriptor)).decode("ascii")
            process = self._launch(executable, encoded_descriptor, candidate)
            owner = OwnedProcess(process)
            try:
                if os.name == "nt":
                    owner.job = _WindowsJob.assign(process)
            except BaseException:
                owner.reconcile(deadline=time.monotonic() + self._config.termination_timeout, poll_interval=self._config.process_poll_interval)
                raise
            run = _Run(future, owner, None, Lock(), terminal_event=Event(), reservation=reservation)
            with self._lock:
                self._runs[call.submission_id] = run
                self._launching[call.submission_id].set()
            if not self._authority.mark_submitted(
                call.submission_id, generation=reservation.generation, attempt=reservation.attempt,
            ):
                raise ExecutionError("subprocess resource reservation was lost before launch")
            future._set_process(process)
            future._set_cancel_requester(lambda: self._request_cancel(run))
            connection, _ = listener.accept()
            connection.settimeout(max(0.001, call.admission_deadline - time.monotonic()))
            run.connection = connection
            self._handshake(call, future, run, descriptor, candidate)
        except BaseException as exc:
            if run is None and reservation is not None:
                self._authority.release(
                    call.submission_id, generation=reservation.generation, attempt=reservation.attempt,
                    worker_id=None, never_submitted=True,
                )
            if not future.done():
                if run is not None and run.issued and run.payload_transfer_started:
                    future._publish_uncertain(ExecutionUncertainError("subprocess failed after payload transfer began"))
                else:
                    future._publish_exception(exc if isinstance(exc, ExecutionError) else ExecutionError("subprocess execution failed"))
        finally:
            with self._lock:
                launched = self._launching.get(call.submission_id)
                if launched is not None:
                    launched.set()
            if listener is not None:
                listener.close()

    def _handshake(self, call: SubmittedCall[T], future: SubProcessFuture[T], run: _Run, descriptor: BootstrapDescriptor, candidate: EnvironmentCandidate | None) -> None:
        """Authorize a ready worker before transmitting the immutable payload bytes."""
        assert run.connection is not None
        conversation = self._conversation(descriptor)
        reader = SocketFrameReader(run.connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=self._limits())
        frame = reader.read()
        conversation.accept_frame(frame)
        hello_control = decode_control(frame, limit_bytes=self._config.control_header_limit_bytes, required_keys={"dill", "implementation", "pid", "protocol", "python", "token", "worker_id"})
        if hello_control["token"] != descriptor.rendezvous_token or hello_control["protocol"] != WORKER_PROTOCOL_ID or hello_control["dill"] != dill.__version__:
            raise AdmissionError("worker bootstrap identity is incompatible")
        if hello_control["implementation"] != sys.implementation.name or hello_control["python"] != list(sys.version_info[:2]):
            raise AdmissionError("worker Python implementation is incompatible")
        pid = hello_control["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise AdmissionError("worker bootstrap reported an invalid PID")
        worker_id = hello_control["worker_id"]
        if not isinstance(worker_id, str) or not worker_id.startswith("subprocess:") or len(worker_id) > 128 or not self._worker_belongs_to_owner(pid, run.owner.process):
            raise AdmissionError("worker bootstrap is not associated with its launcher")
        future._set_association(worker_id=worker_id, pid=pid, environment=candidate)
        run.worker_id = worker_id
        owner_data: list[tuple[OwnerEnvelopeType, bytes]] = []
        if call.environment is not None:
            owner_data.append((OwnerEnvelopeType.ENVIRONMENT, _owner_json(call.environment.to_data(), self._config.owner_envelope_limit_bytes)))
        if call.world is not None:
            owner_data.append((OwnerEnvelopeType.WORLD, _owner_json(call.world.to_data(), self._config.owner_envelope_limit_bytes)))
            assert run.reservation is not None and run.reservation.allocation is not None
            owner_data.append((OwnerEnvelopeType.ALLOCATION, _owner_json(run.reservation.allocation.to_data(), self._config.owner_envelope_limit_bytes)))
        prepare = encode_control(FrameState.PREPARE, descriptor.correlation, {"cwd": str(self._config.working_directory), "deadline": call.admission_deadline, "owners": [owner.value for owner, _ in owner_data]}, header_limit=self._config.control_header_limit_bytes)
        self._send(run.connection, prepare)
        conversation.accept(prepare)
        for owner, data in owner_data:
            envelope = encode_owner_envelope(FrameState.PREPARE, descriptor.correlation, owner, data, header_limit=self._config.control_header_limit_bytes, owner_limit=self._config.owner_envelope_limit_bytes)
            self._send(run.connection, envelope)
            conversation.accept(envelope)
        ready_frame = reader.read()
        conversation.accept_frame(ready_frame)
        if ready_frame.state is FrameState.ERROR:
            raise AdmissionError("worker rejected local admission")
        ready_control = decode_control(ready_frame, limit_bytes=self._config.control_header_limit_bytes, required_keys={"pid", "ready", "worker_id"})
        if ready_control["ready"] is not True or ready_control["pid"] != pid or ready_control["worker_id"] != worker_id:
            raise AdmissionError("worker readiness evidence is invalid")
        if time.monotonic() >= call.admission_deadline:
            self._send_stop(run, descriptor)
            raise AdmissionError("admission deadline exceeded")
        record: EnvironmentRecord | None = None
        if call.environment is not None:
            owner_frame = reader.read()
            conversation.accept_frame(owner_frame)
            if owner_frame.owner is not OwnerEnvelopeType.ENVIRONMENT:
                raise AdmissionError("worker omitted environment evidence")
            record = EnvironmentRecord.from_data(_json(owner_frame.payload, self._config.owner_envelope_limit_bytes))
        if call.world is not None:
            owner_frame = reader.read()
            conversation.accept_frame(owner_frame)
            if owner_frame.owner is not OwnerEnvelopeType.ALLOCATION:
                raise AdmissionError("worker omitted actual resource allocation evidence")
            allocation = WorldAllocation.from_data(_json(owner_frame.payload, self._config.owner_envelope_limit_bytes))
        else:
            allocation = None
        decision = admit(
            environment=call.environment, world=call.world, record=record,
            allocation=allocation, deadline=call.admission_deadline,
        )
        future._set_association(report=decision.report)
        if not decision.go:
            raise AdmissionError("actual worker admission evidence is not admissible", report=decision.report)
        assert run.reservation is not None
        resources = _allocation_amounts(allocation) if allocation is not None else run.reservation.resources
        if not self._authority.confirm_grant(
            call.submission_id, generation=run.reservation.generation, attempt=run.reservation.attempt,
            worker_id=worker_id, resources=resources, pid=pid, allocation=allocation,
        ):
            raise AdmissionError("worker grant no longer matches the held resource reservation")
        if allocation is not None:
            future._set_association(allocation=allocation)
        if not future._authorize(deadline=call.admission_deadline):
            self._send_stop(run, descriptor)
            return
        execution_deadline = None if call.execution_timeout is None else time.monotonic() + call.execution_timeout
        go = encode_control(FrameState.GO, descriptor.correlation, {"deadline": execution_deadline, "permit": descriptor.rendezvous_token}, header_limit=self._config.control_header_limit_bytes)
        self._send(run.connection, go)
        conversation.accept(go)
        with run.lock:
            run.issued = True
            run.terminal_event = run.terminal_event or Event()
        if execution_deadline is not None:
            Thread(target=self._deadline_watch, args=(run, execution_deadline), name="dryml-execute-deadline", daemon=True).start()
        payload = call.payload.path.read_bytes()
        if len(payload) != call.payload.size_bytes or hashlib.sha256(payload).hexdigest() != call.payload.sha256:
            raise ExecutionError("coordinator invocation spool changed before transfer")
        message = encode_frame(FrameState.PAYLOAD, FrameType.PAYLOAD, descriptor.correlation, payload, header_limit=self._config.control_header_limit_bytes)
        with run.lock:
            run.payload_transfer_started = True
        self._send(run.connection, message)
        conversation.accept(message)
        # The admission socket bound cannot turn an explicitly unlimited workload
        # into an invented execution deadline.
        run.connection.settimeout(None)
        self._receive_active(call, future, run, descriptor, conversation)

    def _receive_active(self, call: SubmittedCall[T], future: SubProcessFuture[T], run: _Run, descriptor: BootstrapDescriptor, conversation: ProtocolConversation) -> None:
        """Route ordered output and publish only a validated one-shot worker outcome."""
        assert run.connection is not None
        outcome_seen = False
        reader = SocketFrameReader(run.connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=self._limits())
        while True:
            try:
                frame = reader.read()
                conversation.accept_frame(frame)
            except (OSError, FrameError, EOFError):
                with run.lock:
                    cancelling = run.cancelling
                    outcome_claimed = run.outcome_claimed
                if cancelling:
                    return
                if outcome_claimed:
                    # A validated outcome won; missing output fences are retained
                    # as incomplete capture rather than changing that outcome.
                    return
                if not future.done():
                    future._publish_uncertain(ExecutionUncertainError("subprocess worker disconnected before a validated outcome"))
                return
            if frame.state is FrameState.OUTPUT:
                assert frame.stream is not None and frame.sequence is not None
                call.output._capture(frame.stream, frame.payload, frame.sequence)
                continue
            if frame.state is FrameState.OUTPUT_FINAL:
                control = _json(frame.payload, self._config.control_header_limit_bytes)
                call.output._finalize(control["stream"], control["next_sequence"])
                continue
            if frame.state is FrameState.RESULT:
                if outcome_seen or not self._claim_outcome(run):
                    return
                outcome_seen = True
                try:
                    value = self._receive_result(future, frame.payload)
                except BaseException as exc:
                    future._publish_exception(ExecutionError("worker result could not be decoded"))
                else:
                    future._publish_result(value)
                run.qualified_terminal = True
                continue
            if frame.state is FrameState.ERROR:
                if outcome_seen or not self._claim_outcome(run):
                    return
                outcome_seen = True
                detail = _json(frame.payload, self._config.result_limit_bytes)
                remote_type = str(detail.get("type", "RemoteError"))[:128]
                future._publish_exception(RemoteExecutionError(f"remote subprocess execution failed ({remote_type})", remote_type=remote_type))
                run.qualified_terminal = True
                continue
            if outcome_seen:
                return

    def _receive_result(self, future: SubProcessFuture[T], data: bytes) -> T:
        """Use the executor-owned result slot through its narrow authorized hook."""
        return future._receive_result(data)

    def _select_environment(self, call: SubmittedCall[T]) -> tuple[list[str], EnvironmentCandidate | None]:
        """Select a launchable existing environment only when one was requested."""
        if call.environment is None:
            return [str(self._python())], None
        deadline = call.admission_deadline
        inventory = discover_candidates(self._config, cwd=self._config.working_directory, interpreter=self._python(), deadline=deadline)
        for spec in inventory.specs:
            if time.monotonic() >= deadline:
                break
            candidate = probe_candidate(spec, interpreter=self._python(), timeout=max(0.001, deadline - time.monotonic()), output_limit=self._config.owner_envelope_limit_bytes, deadline=deadline, termination_timeout=self._config.termination_timeout, read_chunk_bytes=self._config.process_read_chunk_bytes, poll_interval=self._config.process_poll_interval)
            if candidate.launchable is not True or candidate.record is None:
                continue
            decision = admit(environment=call.environment, record=candidate.record, deadline=deadline)
            if decision.go:
                return self._command_interpreter(spec), candidate
        raise AdmissionError("no launchable environment satisfies the supplied requirement")

    def _reserve(self, call: SubmittedCall[T], future: SubProcessFuture[T]) -> Reservation | None:
        """Acquire one bounded shared-authority charge before native launch.

        Environment discovery is intentionally absent here when the caller omitted
        that requirement.  World admission alone obtains fresh exact inventory
        through Execute's owned bounded probe and waits only for known contention.
        """
        generation = "subprocess-v1"
        attempt = "0"
        if call.world is None:
            reservation = self._authority.reserve(
                call.submission_id, ResourceAmounts(None, None, {}, {}),
                generation=generation, attempt=attempt, total=ResourceAmounts(None, None, {}, {}),
                executor_id=self._executor_id,
            )
            if reservation is None:
                raise AdmissionError("subprocess resource authority could not record the unconstrained charge")
            return reservation
        while True:
            if future.done():
                return None
            inventory = self._observe_inventory(call.admission_deadline)
            if inventory is None:
                raise AdmissionError("fresh local resource inventory is unavailable")
            plan = plan_admission(world=call.world, inventory=inventory, deadline=call.admission_deadline)
            if not plan.feasible:
                raise AdmissionError("world requirement is unsupported or incompatible with local inventory", report=plan.report)
            reservation = self._authority.reserve_local(
                call.submission_id, call.world, inventory, generation=generation, attempt=attempt,
                executor_id=self._executor_id,
            )
            if reservation is not None:
                return reservation
            if self._authority.has_unknown_charge():
                raise AdmissionError("local resource availability is unknown because another charge is unbound")
            remaining = call.admission_deadline - time.monotonic()
            if remaining <= 0:
                raise AdmissionError("admission deadline elapsed waiting for local resources")
            self._authority.wait_for_change(min(remaining, self._config.process_poll_interval))

    def _command_interpreter(self, spec: Any) -> list[str]:
        """Resolve one selected existing runtime without rewriting Conda launch form."""
        if isinstance(spec, CurrentEnvironmentSpec):
            return [str(self._python())]
        if isinstance(spec, PythonExecutableSpec):
            return [spec.executable]
        if isinstance(spec, CondaEnvironmentSpec):
            if spec.launch_mode == "direct":
                return [spec.direct_python_executable()]
            return [spec.conda_executable, "run", "-p" if spec.prefix else "-n", spec.prefix or spec.name or "", "--no-capture-output", "--", "python"]
        raise AdmissionError("selected environment launch form is unsupported for subprocess execution")

    def _launch(self, runtime: list[str], descriptor: str, candidate: EnvironmentCandidate | None) -> subprocess.Popen[bytes]:
        """Create an owned worker process using a minimal explicit environment."""
        command = [*runtime, "-m", "dryml.execute._worker", "--bootstrap", descriptor]
        overrides: dict[str, str] = dict(self._config.env_vars)
        if candidate is not None and isinstance(candidate.spec, (PythonExecutableSpec, CondaEnvironmentSpec)):
            for key, value in candidate.spec.env.items():
                overrides.setdefault(key, value)
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self._config.working_directory, env=minimal_environment(overrides), start_new_session=(os.name == "posix"), creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0)

    def _request_cancel(self, run: _Run) -> bool:
        """Request and escalate owned-group termination without claiming success early."""
        if run.future.done():
            return False
        with run.lock:
            run.cancelling = True
        Thread(target=self._cancel_run, args=(run,), name="dryml-execute-cancel", daemon=True).start()
        return True

    def _cancel_run(self, run: _Run, deadline_error: bool = False) -> None:
        """Confirm cancellation/deadline only after the exact owned group stops."""
        if run.terminal_event is not None:
            run.terminal_event.set()
        complete = run.owner.reconcile(deadline=time.monotonic() + self._config.termination_timeout, poll_interval=self._config.process_poll_interval)
        if complete:
            if deadline_error:
                run.future._expire(ExecutionDeadlineExceeded("execution deadline exceeded"))
            else:
                run.future._publish_running_cancellation()
        elif not run.future.done():
            run.future._publish_uncertain(ExecutionUncertainError("subprocess owned-group termination could not be confirmed"))

    def _deadline_watch(self, run: _Run, deadline: float) -> None:
        """Escalate an execution deadline only while no validated outcome exists."""
        event = run.terminal_event
        if event is not None and event.wait(max(0.0, deadline - time.monotonic())):
            return
        with run.lock:
            if run.outcome_claimed or run.future.done():
                return
            run.deadline_expired = True
            run.cancelling = True
        self._cancel_run(run, deadline_error=True)

    def _send_stop(self, run: _Run, descriptor: BootstrapDescriptor) -> None:
        """Best-effort STOP closes an unpermitted worker without sending payload bytes."""
        if run.connection is not None:
            try:
                self._send(run.connection, encode_control(FrameState.STOP, descriptor.correlation, {"reason": "rejected"}, header_limit=self._config.control_header_limit_bytes))
            except OSError:
                pass

    def _resource_inventory(self, deadline: float) -> ResourceSnapshot:
        """Refresh accounting through an Execute-owned bounded inventory probe."""
        inventory = self._observe_inventory(deadline)
        if inventory is None:
            total = ResourceAmounts(None, None, {}, {})
            snapshot = self._authority.snapshot(total=total, executor_id=self._executor_id)
            return replace(snapshot, inventory=None)
        total = ResourceAmounts(
            float(len(inventory.cpus)), inventory.memory,
            {kind: float(len(values)) for kind, values in inventory.accelerators.items()}, {},
        )
        snapshot = self._authority.snapshot(total=total, executor_id=self._executor_id)
        return replace(snapshot, inventory=inventory)

    def _observe_inventory(self, deadline: float) -> LocalResourceInventory | None:
        """Run the worlds semantic inventory method in a bounded owned probe."""
        if time.monotonic() >= deadline:
            return None
        command = [
            str(self._python()), "-c",
            "import json; from dryml.worlds import local_inventory; print(json.dumps(local_inventory().to_data()))",
        ]
        try:
            result = run_bounded(
                command, deadline=deadline, output_limit=self._config.owner_envelope_limit_bytes,
                termination_timeout=self._config.termination_timeout,
                read_chunk_bytes=self._config.process_read_chunk_bytes,
                poll_interval=self._config.process_poll_interval,
            )
            if result.returncode != 0 or result.timed_out or result.cancelled or not result.cleanup_complete:
                return None
            return LocalResourceInventory.from_data(json.loads(result.stdout))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _python(self) -> Path:
        """Return the executor-captured direct interpreter after config normalization."""
        assert self._config.python_executable is not None
        return self._config.python_executable

    def _limits(self) -> dict[FrameType, int]:
        """Build one effective typed frame-limit mapping for this config."""
        return {FrameType.CONTROL: self._config.control_header_limit_bytes, FrameType.OWNER: self._config.owner_envelope_limit_bytes, FrameType.PAYLOAD: self._config.invocation_limit_bytes, FrameType.OUTPUT: self._config.output_frame_limit_bytes, FrameType.RESULT: self._config.result_limit_bytes, FrameType.ERROR: self._config.result_limit_bytes}

    def _conversation(self, descriptor: BootstrapDescriptor) -> ProtocolConversation:
        """Create the one correlation-locked protocol state machine for a worker."""
        return ProtocolConversation(descriptor.correlation, header_limit=self._config.control_header_limit_bytes, invocation_limit=self._config.invocation_limit_bytes, result_limit=self._config.result_limit_bytes, output_limit=self._config.output_frame_limit_bytes, owner_limit=self._config.owner_envelope_limit_bytes, admission_limit=self._config.admission_message_limit_bytes)

    def _claim_outcome(self, run: _Run) -> bool:
        """Linearize a validated outcome against deadline-driven group termination."""
        with run.lock:
            if run.deadline_expired or run.outcome_claimed or run.cancelling:
                return False
            run.outcome_claimed = True
            if run.terminal_event is not None:
                run.terminal_event.set()
            return True

    @staticmethod
    def _worker_belongs_to_owner(worker_pid: int, launcher: subprocess.Popen[bytes]) -> bool:
        """Verify a direct or wrapped worker remains in this launcher's owned group."""
        try:
            if os.name == "posix":
                return os.getpgid(worker_pid) == os.getpgid(launcher.pid)
            return worker_pid == launcher.pid
        except OSError:
            return False

    @staticmethod
    def _send(connection: socket.socket, data: bytes) -> None:
        """Write one complete validated frame or propagate the transport failure."""
        connection.sendall(data)


def _owner_json(data: Mapping[str, Any], limit: int) -> bytes:
    """Encode an owner-owned requirement with only that owner's configured cap."""
    try:
        value = canonical_json_bytes(data, max_depth=32, max_nodes=65536, max_entries=4096, max_string=limit, max_int_bits=64)
    except BaseException as exc:
        raise AdmissionError("owner requirement cannot be transported") from exc
    if len(value) > limit:
        raise AdmissionError("owner requirement exceeds configured transport limit")
    return value


def _allocation_amounts(allocation: WorldAllocation) -> ResourceAmounts:
    """Summarize an exact worker allocation for authority grant confirmation."""
    processes = tuple(process for values in allocation.roles.values() for process in values)
    memory = None if all(process.memory is None for process in processes) else sum(process.memory or 0 for process in processes)
    return ResourceAmounts(
        float(sum(len(process.cpus) for process in processes)),
        memory,
        {
            kind: float(sum(len(process.accelerators.get(kind, ())) for process in processes))
            for kind in {kind for process in processes for kind in process.accelerators}
        },
        {},
    )


def _json(data: bytes, limit: int) -> dict[str, Any]:
    """Decode one trusted bounded control/owner JSON mapping without diagnostics."""
    value = canonical_json_load_bytes(data, max_depth=32, max_nodes=65536, max_entries=4096, max_string=limit, max_int_bits=64)
    if not isinstance(value, Mapping):
        raise FrameError("worker control payload is not a mapping")
    return dict(value)


__all__ = ["SubProcessBackend", "SubProcessConfig", "SubProcessFuture"]
