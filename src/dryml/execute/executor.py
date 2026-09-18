"""Reusable Execute lifecycle ownership, fluent controls, and one-off helpers.

This additive module deliberately is not imported by :mod:`dryml.execute` until
the public-facade cutover.  It coordinates only preflight and backend-owned
execution; concrete admission, discovery, and native launch remain backend work.
"""

from __future__ import annotations

import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from threading import Condition, Event, Lock, RLock, Thread
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar

from dryml.environments import EnvironmentRequirement
from dryml.environments.selection import (ResolvedEnvironmentSelection,
                                          resolve_environment_spec)
from dryml.environments.specs import EnvironmentSpec
from dryml.worlds import WorldRequirement

from ._spooling import PayloadSpooler, SpoolBudget, SpoolLease, SpoolReservation, deserialize_result, validate_result
from .backend import Backend
from .config import BackendConfig
from .errors import AdmissionError, CleanupError, ExecutionError
from .future import ExecutionFuture
from .models import DiscoverySnapshot, ResourceSnapshot, SubmittedCall, WorkerSetup
from .output import ExecutionOutput


T = TypeVar("T")
_ExecutorState = Literal["new", "starting", "open", "closing", "cleanup_incomplete", "closed"]


@dataclass(slots=True)
class _Submission(Generic[T]):
    """Keep accepted backend and spool ownership until both cleanup phases finish."""

    future: ExecutionFuture[T]
    payload: Any
    reservation: SpoolReservation
    backend: Backend
    backend_seen: bool = False
    backend_reconciled: bool = False
    spool_disposed: bool = False


class Executor:
    """Own one inert backend configuration and many independently accepted calls.

    Args:
        config: Immutable backend configuration that selects the executor's sole
            backend and all common operational limits.

    Raises:
        TypeError: If ``config`` is not a :class:`BackendConfig`.

    Side Effects:
        Construction captures coordinator paths and interpreter identity only. It
        does not probe/create spool directories, acquire a quota lease, start a
        backend, or invoke workload code. Calls are accepted only after a bounded
        one-graph payload preflight and remain owned until their Future cleanup.
    """

    def __init__(self, config: BackendConfig) -> None:
        """Create an inert reusable executor without starting its backend."""
        if not isinstance(config, BackendConfig):
            raise TypeError("config must be a BackendConfig")
        self._captured_cwd = Path.cwd()
        self._captured_interpreter = Path(sys.executable)
        working_directory = config.working_directory
        resolved_working_directory = self._captured_cwd if working_directory is None else (
            working_directory if working_directory.is_absolute() else self._captured_cwd / working_directory
        )
        configured_spool = config.spool_directory
        resolved_spool_directory = None if configured_spool is None else (
            configured_spool if configured_spool.is_absolute() else self._captured_cwd / configured_spool
        )
        normalized = dict(
            working_directory=resolved_working_directory,
            spool_directory=resolved_spool_directory,
        )
        if "python_executable" in {field.name for field in fields(config)}:
            executable = getattr(config, "python_executable")
            if executable is None:
                normalized["python_executable"] = self._captured_interpreter
            elif isinstance(executable, Path):
                normalized["python_executable"] = executable if executable.is_absolute() else self._captured_cwd / executable
            elif isinstance(executable, str):
                candidate = Path(executable)
                normalized["python_executable"] = executable if candidate.is_absolute() else str(self._captured_cwd / candidate)
        self._config = replace(config, **normalized)
        self._condition = Condition(RLock())
        self._state: _ExecutorState = "new"
        self._preflights = 0
        self._lease: SpoolLease | None = None
        self._spooler: PayloadSpooler | None = None
        self._submissions: dict[ExecutionFuture[Any], _Submission[Any]] = {}
        self._backend: Backend | None = None
        self._backend_creating = False
        self._backend_generation = 0
        self._backend_results: dict[int, BaseException | Backend] = {}
        self._backend_result_waiters: dict[int, int] = {}
        self._backend_starting = False
        self._backend_started = False
        self._backend_start_generation = 0
        self._backend_start_results: dict[int, BaseException | None] = {}
        self._backend_start_result_waiters: dict[int, int] = {}
        self._dispatches = 0
        self._queries = 0
        self._close_active = False
        self._acceptance_hook: Callable[[ExecutionFuture[Any]], None] | None = None
        configured = self._config.spool_directory
        if configured is None:
            self._spool_parent = self._capture_temp_parent(self._captured_cwd)
        else:
            self._spool_parent = configured

    @property
    def state(self) -> _ExecutorState:
        """Return the current lifecycle state without changing backend ownership."""
        with self._condition:
            return self._state

    def start(self) -> "Executor":
        """Initialize this executor's backend exactly once.

        Returns:
            This executor, including when a concurrent close moved it to closing.

        Raises:
            RuntimeError: If the executor is already closed.
            BaseException: If inert backend construction or initialization fails.

        Side Effects:
            May initialize a backend. It never creates a spool lease or accepts a
            workload by itself.
        """
        with self._condition:
            if self._state == "closed":
                raise RuntimeError("executor is closed")
        with self._condition:
            if self._state not in {"new", "starting", "open"}:
                raise RuntimeError("executor is closing or closed")
        self._start_backend(time.monotonic() + self._config.admission_timeout)
        return self

    def __enter__(self) -> "Executor":
        """Start and return this executor for a context-managed lifetime."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close normally without suppressing an exception raised by the context body."""
        try:
            self.close()
        except CleanupError as cleanup_error:
            if exc_value is not None:
                raise exc_value.with_traceback(traceback) from cleanup_error
            raise

    def submit(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        kwargs: Mapping[str, Any] | None = None,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[Callable[[ExecutionFuture[T]], None]] = (),
        output: ExecutionOutput | None = None,
        worker_setup: WorkerSetup | None = None,
    ) -> ExecutionFuture[T]:
        """
        Preflight and asynchronously dispatch one immutable callable snapshot.

                Args:
                    fn: Supported callable root to snapshot once with ``args``
                    and
                        ``kwargs``.
                    args: Ordinary positional workload arguments.
                    kwargs: Ordinary workload keyword mapping, separate from
                    controls.
                    environment: Optional owner-defined environment
                    requirement.
                    environment_spec: Optional exact existing worker selector.
                    It is frozen
                        once here and never becomes candidate-search fallback.
                    world: Optional owner-defined world requirement.
                    execution_timeout: ``"inherit"`` for config default,
                    ``None`` to
                        disable a workload deadline, or a finite positive
                        override.
                    stream_output: Optional per-call live-output override.
                    done_callbacks: Finite coordinator callbacks copied before
                    preflight.
                    output: Optional single-use retained output owner.
                    worker_setup: Optional inert factory/data control entered
                    before worker
                        payload deserialization.

                Returns:
                    The exact concrete Future created by the selected backend.

                Raises:
                    TypeError: If controls, callbacks, output, or workload
                    keywords are
                        malformed.
                    ValueError: If an execution timeout is invalid or spool
                    storage is
                        unusable before acceptance.
                    RuntimeError: If closure linearized before acceptance.
                    ExecutionError: If quota lease/capacity or backend Future
                    identity is
                        invalid. Accepted asynchronous failures are published
                        on Future.

                Side Effects:
                    Serializes one callable/argument graph and creates only an
                    execution-owned spool child after an all-or-nothing
                    reservation.
        """
        self._validate_requirements(environment, world)
        selection = self._resolve_environment_spec(environment_spec)
        call_kwargs, callbacks, effective_timeout, effective_stream, output_owner, setup = self._validate_controls(
            kwargs, execution_timeout, stream_output, done_callbacks, output, worker_setup
        )
        self._register_preflight()
        payload = None
        reservation = None
        spooler = None
        accepted = False
        try:
            lease, spooler = self._prepare_spooler()
            payload, reservation = spooler.snapshot(fn, args, call_kwargs)
            backend = self._get_backend(time.monotonic() + self._config.admission_timeout)
            submission_id = uuid.uuid4().hex
            future = backend.create_future(submission_id, output_owner)
            self._validate_future(future, submission_id, output_owner)
            future._prepare(
                diagnostic_text_limit_bytes=self._config.diagnostic_text_limit_bytes,
                diagnostic_issue_limit=self._config.diagnostic_issue_limit,
                termination_timeout=self._config.termination_timeout,
            )
            call = SubmittedCall(
                submission_id=submission_id,
                admission_deadline=time.monotonic() + self._config.admission_timeout,
                payload=payload,
                environment=environment,
                environment_spec=selection,
                world=world,
                execution_timeout=effective_timeout,
                stream_output=effective_stream,
                output=output_owner,
                worker_setup=setup,
            )
            record = _Submission(future, payload, reservation, backend)
            future._install_initial_callbacks(callbacks)
            with self._condition:
                if self._state not in {"new", "starting", "open"}:
                    raise RuntimeError("executor is closing or closed")
                spooler.accept(payload, reservation)
                future._set_cleanup_reconciler(lambda timeout, target=future: self._cleanup_submission(target, timeout))
                future._set_result_receiver(lambda data, target=future: self._receive_result(target, data))
                try:
                    output_owner._bind(
                        submission_id,
                        output_limit_bytes=self._config.output_limit_bytes,
                        live_output_queue_limit_bytes=self._config.live_output_queue_limit_bytes,
                        stream_output=effective_stream,
                        output_final_timeout=self._config.output_final_timeout,
                        start_live=False,
                    )
                except BaseException:
                    raise
                self._submissions[future] = record
                accepted = True
            hook = self._acceptance_hook
            if hook is not None:
                try:
                    hook(future)
                except BaseException:
                    future._publish_exception(ExecutionError("accepted execution ownership could not be installed"))
                    return future
            if effective_stream:
                output_owner._start_live_delivery()
            self._dispatch(call, future, backend)
            return future
        except BaseException:
            if payload is not None and reservation is not None and spooler is not None and not accepted:
                try:
                    spooler.dispose(payload, reservation)
                except CleanupError:
                    pass
            raise
        finally:
            self._unregister_preflight()

    def run(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        kwargs: Mapping[str, Any] | None = None,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[Callable[[ExecutionFuture[T]], None]] = (),
        output: ExecutionOutput | None = None,
        worker_setup: WorkerSetup | None = None,
    ) -> T:
        """
        Submit one call and return its ordinary result without closing this
        executor.

                Args:
                    fn: Supported workload callable.
                    args: Ordinary positional workload arguments.
                    kwargs: Ordinary workload keyword mapping.
                    environment: Optional owner-defined environment
                    requirement.
                    environment_spec: Optional exact existing selector resolved
                    once at
                        this call's entry; incompatible pins never trigger
                        fallback.
                    world: Optional owner-defined world requirement.
                    execution_timeout: Workload deadline control.
                    stream_output: Optional live-output override.
                    done_callbacks: Copied coordinator completion callbacks.
                    output: Optional single-use retained output owner.
                    worker_setup: Optional inert setup bound before workload
                    keywords.

                Returns:
                    The workload's result.

                Raises:
                    BaseException: Synchronous submission errors or the Future
                    outcome.

                Side Effects:
                    Leaves the accepted Future and its cleanup under this
                    reusable owner.
        """
        return self.submit(
            fn,
            *args,
            kwargs=kwargs,
            environment=environment,
            environment_spec=environment_spec,
            world=world,
            execution_timeout=execution_timeout,
            stream_output=stream_output,
            done_callbacks=done_callbacks,
            output=output,
            worker_setup=worker_setup,
        ).result()

    def with_options(
        self,
        *,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        execution_timeout: float | None | Literal["inherit"] = "inherit",
        stream_output: bool | None = None,
        done_callbacks: Sequence[Callable[[ExecutionFuture[Any]], None]] = (),
        output: ExecutionOutput | None = None,
        worker_setup: WorkerSetup | None = None,
    ) -> "ExecutorView":
        """
        Bind immutable execution controls without initializing or reserving
        work.

                Args:
                    environment: Optional typed environment requirement bound
                    to each view
                        submission.
                    environment_spec: Optional exact selector retained inertly
                    by this view
                        and resolved once when each later call begins.
                    world: Optional typed world requirement bound to each view
                    submission.
                    execution_timeout: ``"inherit"`` for the config default,
                    ``None`` to
                        disable a deadline, or a finite positive seconds
                        override.
                    stream_output: Optional live-output override for view
                    submissions.
                    done_callbacks: Finite callbacks copied for each submitted
                    Future.
                    output: Optional single-use retained output holder bound on
                    submit.
                    worker_setup: Optional inert worker setup bound to every
                    view call.

                Returns:
                    A parent-retaining view whose call keywords are always
                    workload values.

                Raises:
                    TypeError: If control callbacks/output have invalid types.
                    ValueError: If the bound execution timeout is invalid.

                Side Effects:
                    Copies callback registrations only; it does not create
                    backend or spool
                    resources. Later view submissions retain the parent
                    lifecycle.
        """
        _, callbacks, effective_timeout, effective_stream, _, setup = self._validate_controls(
            {}, execution_timeout, stream_output, done_callbacks, output, worker_setup
        )
        self._validate_requirements(environment, world)
        self._validate_environment_spec(environment_spec)
        return ExecutorView(
            executor=self,
            environment=environment,
            world=world,
            execution_timeout=effective_timeout,
            stream_output=effective_stream,
            done_callbacks=callbacks,
            output=output,
            worker_setup=setup,
            environment_spec=environment_spec,
        )

    def discover(
        self,
        *,
        environment: EnvironmentRequirement | None = None,
        environment_spec: EnvironmentSpec | None = None,
        world: WorldRequirement | None = None,
        timeout: float | None = None,
    ) -> DiscoverySnapshot:
        """
        Delegate a bounded discovery query after lazy backend initialization.

                Args:
                    environment: Optional typed environment requirement for
                    candidates and
                        feasibility evidence.
                    environment_spec: Optional exact selector for the sole
                    discovered
                        candidate; it bypasses candidate enumeration.
                    world: Optional typed world requirement for feasibility
                    evidence.
                    timeout: Optional positive query bound in seconds; ``None``
                    uses
                        ``discovery_timeout``.

                Returns:
                    A non-reserving, possibly incomplete backend discovery
                    snapshot.

                Raises:
                    RuntimeError: If closure has begun.
                    TypeError: If requirements or ``timeout`` have invalid
                    types.
                    ValueError: If ``timeout`` is not finite and positive.
                    TimeoutError: If backend initialization or discovery
                    exceeds the bound.
                    ExecutionError: If the selected backend cannot complete
                    discovery.

                Side Effects:
                    Lazily starts the backend and may perform bounded discovery
                    I/O. It
                    does not create a spool, reserve capacity, or invoke
                    workload code.
        """
        budget = self._config.discovery_timeout if timeout is None else self._positive_duration("timeout", timeout)
        self._validate_requirements(environment, world)
        selection = self._resolve_environment_spec(environment_spec)
        deadline = time.monotonic() + budget
        self._require_queryable()
        self._register_query()
        try:
            backend = self._start_backend(deadline)
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                raise TimeoutError("discovery initialization exceeded timeout")
            controls: dict[str, Any] = {
                "environment": environment,
                "world": world,
                "timeout": remaining or budget,
            }
            # Preserve the established unpinned Backend.discover call shape for
            # existing backend implementations.
            if selection is not None:
                controls["environment_spec"] = selection
            return backend.discover(**controls)
        finally:
            self._unregister_query()

    def resources(self, *, timeout: float | None = None) -> ResourceSnapshot:
        """Delegate bounded resource inspection, including during incomplete cleanup.

        Args:
            timeout: Optional positive query bound in seconds; ``None`` uses
                ``discovery_timeout``.

        Returns:
            A backend-scoped, possibly incomplete logical resource snapshot.

        Raises:
            RuntimeError: If the executor is closed or has begun closing without
                an already-started backend.
            TypeError: If ``timeout`` has an invalid type.
            ValueError: If ``timeout`` is not finite and positive.
            TimeoutError: If initialization or observation exceeds the bound.
            ExecutionError: If the selected backend cannot observe resources.

        Side Effects:
            Lazily starts an open backend or uses its retained backend during
            retryable cleanup; may perform bounded native observation I/O.
        """
        budget = self._config.discovery_timeout if timeout is None else self._positive_duration("timeout", timeout)
        deadline = time.monotonic() + budget
        with self._condition:
            if self._state == "closed":
                raise RuntimeError("executor is closed")
            backend = self._backend if self._state in {"closing", "cleanup_incomplete"} else None
            if backend is None and self._state not in {"new", "starting", "open"}:
                raise RuntimeError("executor is closing or closed")
        self._register_query()
        try:
            backend = backend or self._start_backend(deadline)
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                raise TimeoutError("resource initialization exceeded timeout")
            return backend.resources(timeout=remaining or budget)
        finally:
            self._unregister_query()

    def close(self, *, cancel: bool = False, timeout: float | None = None) -> None:
        """Stop admission, reconcile owned calls, then release backend and quota ownership.

        Args:
            cancel: Request cancellation before waiting for outstanding work.
            timeout: Optional finite cleanup budget. Normal closure without one
                waits for accepted work; cancellation and unresolved cleanup use
                ``termination_timeout``.

        Returns:
            ``None`` after accepted work, spools, backend state, and this
            executor's quota lease are safely released.

        Raises:
            TypeError: If controls have unsupported types.
            ValueError: If timeout is non-finite or negative.
            CleanupError: If work, spool disposal, backend close, or quota release
                remains incomplete. The executor stays retryable.

        Side Effects:
            Rejects new preflights. A successful close releases this executor's
            quota lease only after every owned spool is safely disposed.
        """
        if not isinstance(cancel, bool):
            raise TypeError("cancel must be bool")
        if timeout is not None:
            self._nonnegative_duration("timeout", timeout)
        budget = timeout if timeout is not None else (self._config.termination_timeout if cancel else None)
        deadline = None if budget is None else time.monotonic() + float(budget)
        initialization_deadline = deadline
        with self._condition:
            if self._state == "closed":
                return
            while self._close_active:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise CleanupError("executor close is still in progress")
                self._condition.wait(remaining)
                if self._state == "closed":
                    return
            self._close_active = True
            self._state = "closing"
            while self._preflights:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    self._close_active = False
                    self._incomplete_locked("preflight cleanup did not finish")
                self._condition.wait(remaining)
            futures = tuple(self._submissions)
        try:
            worker_cleanup_failure: CleanupError | None = None
            if cancel:
                for future in futures:
                    if not future.done():
                        if not future.cancel():
                            try:
                                future.request_cancel()
                            except ExecutionError:
                                pass
            with self._condition:
                while self._queries or self._dispatches or self._backend_creating or self._backend_starting:
                    if deadline is None and any(not future.done() for future in futures):
                        active_deadline = None
                    else:
                        if initialization_deadline is None:
                            initialization_deadline = time.monotonic() + self._config.termination_timeout
                        active_deadline = initialization_deadline
                    remaining = self._remaining(active_deadline)
                    if remaining is not None and remaining <= 0:
                        self._incomplete_locked("dispatch or initialization cleanup did not finish")
                    self._condition.wait(remaining)
            for future in futures:
                self._wait_terminal(future, deadline)
                try:
                    future.cleanup(timeout=self._cleanup_budget(deadline))
                except CleanupError as exc:
                    if not future._has_unobserved_worker_cleanup():
                        raise
                    # The worker's setup exit cannot be retrospectively observed,
                    # but local process/task and spool ownership can still retire.
                    self._cleanup_submission(future, self._cleanup_budget(deadline))
                    worker_cleanup_failure = exc
            spooler = self._spooler
            if spooler is not None:
                spooler.reconcile_cleanup()
            backend = self._backend
            if backend is not None:
                backend.close(cancel=cancel, timeout=self._cleanup_budget(deadline))
            self._release_lease()
            if worker_cleanup_failure is not None:
                raise worker_cleanup_failure
        except BaseException as exc:
            with self._condition:
                self._state = "cleanup_incomplete"
                self._close_active = False
                self._condition.notify_all()
            if isinstance(exc, CleanupError):
                raise
            raise CleanupError("executor cleanup remains incomplete") from exc
        with self._condition:
            self._state = "closed"
            self._close_active = False
            self._condition.notify_all()

    def _register_preflight(self) -> None:
        """Linearize a producer against close before quota or serialization work."""
        with self._condition:
            if self._state not in {"new", "starting", "open"}:
                raise RuntimeError("executor is closing or closed")
            self._preflights += 1

    def _unregister_preflight(self) -> None:
        """Wake closure exactly once when a registered producer exits."""
        with self._condition:
            self._preflights -= 1
            self._condition.notify_all()

    def _prepare_spooler(self) -> tuple[SpoolLease, PayloadSpooler]:
        """Lazily join the quota generation and retain it through executor close."""
        with self._condition:
            lease = self._lease
        if lease is None:
            candidate = SpoolBudget.acquire(self._config)
            with self._condition:
                if self._lease is None:
                    self._lease = candidate
                    self._spooler = PayloadSpooler(self._config, candidate, parent=self._spool_parent)
                    lease = candidate
                else:
                    lease = self._lease
            if lease is not candidate:
                candidate.release()
        assert lease is not None
        with self._condition:
            spooler = self._spooler
        assert spooler is not None
        return lease, spooler

    def _get_backend(self, deadline: float | None, *, accepted: bool = False) -> Backend:
        """Create one inert backend in a joinable worker for Future construction."""
        with self._condition:
            if self._backend is not None:
                return self._backend
            if self._state not in {"new", "starting", "open"} and not (accepted and self._state == "closing"):
                raise RuntimeError("executor is closing or closed")
            if not self._backend_creating:
                self._backend_generation += 1
                generation = self._backend_generation
                self._backend_creating = True
                try:
                    Thread(target=self._create_backend, args=(generation,), name="dryml-execute-create", daemon=False).start()
                except BaseException as exc:
                    self._backend_creating = False
                    self._condition.notify_all()
                    raise ExecutionError("backend factory worker could not start") from exc
            else:
                generation = self._backend_generation
            self._backend_result_waiters[generation] = self._backend_result_waiters.get(generation, 0) + 1
            try:
                while generation not in self._backend_results:
                    remaining = self._remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("backend construction exceeded timeout")
                    self._condition.wait(remaining)
                result = self._backend_results[generation]
                if isinstance(result, BaseException):
                    raise result
                return result
            finally:
                self._consume_backend_result(generation)

    def _create_backend(self, generation: int) -> None:
        """Run the inert factory outside lifecycle locks and wake all bounded waiters."""
        try:
            backend = self._config.create_backend()
            if not isinstance(backend, Backend):
                raise TypeError("create_backend must return a Backend")
        except BaseException as exc:
            with self._condition:
                self._backend_results[generation] = exc
                self._backend_creating = False
                self._condition.notify_all()
            return
        with self._condition:
            self._backend_results[generation] = backend
            self._backend = backend
            self._backend_creating = False
            self._condition.notify_all()

    def _start_backend(self, deadline: float | None, *, accepted: bool = False) -> Backend:
        """Join one owned factory/start worker, honoring each caller's deadline."""
        with self._condition:
            if self._backend_started:
                assert self._backend is not None
                return self._backend
            if self._state not in {"new", "starting", "open"} and not (accepted and self._state == "closing"):
                raise RuntimeError("executor is closing or closed")
        backend = self._get_backend(deadline, accepted=accepted)
        with self._condition:
            if self._backend_started:
                return backend
            if self._state not in {"new", "starting", "open"} and not (accepted and self._state == "closing"):
                raise RuntimeError("executor is closing or closed")
            if not self._backend_starting:
                self._backend_start_generation += 1
                generation = self._backend_start_generation
                self._backend_starting = True
                if self._state != "closing":
                    self._state = "starting"
                try:
                    Thread(target=self._run_backend_start, args=(backend, generation), name="dryml-execute-start", daemon=False).start()
                except BaseException as exc:
                    self._backend_starting = False
                    self._condition.notify_all()
                    raise ExecutionError("backend initialization worker could not start") from exc
            else:
                generation = self._backend_start_generation
            self._backend_start_result_waiters[generation] = self._backend_start_result_waiters.get(generation, 0) + 1
            try:
                while generation not in self._backend_start_results:
                    remaining = self._remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("backend initialization exceeded timeout")
                    self._condition.wait(remaining)
                result = self._backend_start_results[generation]
                if isinstance(result, BaseException):
                    raise result
                return backend
            finally:
                self._consume_backend_start_result(generation)

    def _run_backend_start(self, backend: Backend, generation: int) -> None:
        """Start an already-created backend outside lifecycle locks for all waiters."""
        try:
            backend.start()
        except BaseException as exc:
            with self._condition:
                self._backend_start_results[generation] = exc
                self._backend_starting = False
                self._condition.notify_all()
            return
        with self._condition:
            self._backend_started = True
            self._backend_start_results[generation] = None
            self._backend_starting = False
            if self._state == "starting":
                self._state = "open"
            self._condition.notify_all()

    def _consume_backend_result(self, generation: int) -> None:
        """Discard one completed factory result after its bound waiter detaches."""
        remaining = self._backend_result_waiters[generation] - 1
        if remaining:
            self._backend_result_waiters[generation] = remaining
            return
        self._backend_result_waiters.pop(generation, None)
        self._backend_results.pop(generation, None)

    def _consume_backend_start_result(self, generation: int) -> None:
        """Discard one completed start result after its bound waiter detaches."""
        remaining = self._backend_start_result_waiters[generation] - 1
        if remaining:
            self._backend_start_result_waiters[generation] = remaining
            return
        self._backend_start_result_waiters.pop(generation, None)
        self._backend_start_results.pop(generation, None)

    def _dispatch(self, call: SubmittedCall[T], future: ExecutionFuture[T], backend: Backend) -> None:
        """Start lazy backend/admission work without delaying the accepted submitter."""
        def dispatch() -> None:
            try:
                try:
                    self._start_backend(call.admission_deadline, accepted=True)
                except TimeoutError:
                    future._publish_exception(AdmissionError("admission deadline exceeded"))
                    return
                if future.done() or time.monotonic() >= call.admission_deadline:
                    future._publish_exception(AdmissionError("admission deadline exceeded"))
                    return
                with self._condition:
                    record = self._submissions.get(future)
                    if record is None or future.done():
                        return
                    # A synchronous backend can publish terminality from submit(),
                    # so association must exist before that call.  Roll it back if
                    # submit raises before accepting the Future.
                    record.backend_seen = True
                try:
                    replacement = backend.submit(call, future=future)
                except BaseException:
                    with self._condition:
                        record = self._submissions.get(future)
                        if record is not None:
                            record.backend_seen = False
                    raise
                if replacement is not None:
                    raise ExecutionError("backend.submit must return None and retain the supplied Future")
            except BaseException as exc:
                future._publish_exception(exc)
            finally:
                with self._condition:
                    self._dispatches -= 1
                    self._condition.notify_all()

        try:
            with self._condition:
                self._dispatches += 1
            Thread(target=dispatch, name="dryml-execute-admission", daemon=False).start()
        except BaseException:
            with self._condition:
                self._dispatches -= 1
                self._condition.notify_all()
            future._publish_exception(ExecutionError("accepted execution dispatch could not start"))

    def _cleanup_submission(self, future: ExecutionFuture[Any], timeout: float) -> None:
        """Reconcile exactly one accepted call without closing unrelated reusable work."""
        with self._condition:
            record = self._submissions.get(future)
        if record is None:
            return
        if record.backend_seen and not record.backend_reconciled:
            record.backend.reconcile_cleanup(record.future.submission_id, timeout=timeout)
            record.backend_reconciled = True
        if not record.spool_disposed:
            assert self._spooler is not None
            self._spooler.dispose(record.payload, record.reservation)
            record.spool_disposed = True
        if future._has_unobserved_worker_cleanup():
            return
        with self._condition:
            self._submissions.pop(future, None)
            self._condition.notify_all()

    def _receive_result(self, future: ExecutionFuture[T], data: bytes) -> T:
        """Publish and decode one backend-validated result through its reserved slot.

        Concrete backends call this narrow accepted-submission hook after protocol
        validation. It keeps result-spool ownership in the Executor so a backend
        cannot bypass the reservation that was held before the invocation launched.

        Args:
            future: The exact accepted Future receiving this result.
            data: One bounded serialized result payload from its worker.

        Returns:
            The decoded ordinary result value.

        Raises:
            ExecutionError: If the Future is unknown or belongs to another owner.
            ValueError or TypeError: If the result cannot pass descriptor/codec
                validation. No successful outcome is published by this hook.
        """
        if not isinstance(data, bytes):
            raise TypeError("serialized result must be bytes")
        with self._condition:
            record = self._submissions.get(future)
            spooler = self._spooler
        if record is None or spooler is None:
            raise ExecutionError("result belongs to no accepted execution")
        result = spooler.receive_result(record.reservation, data)
        validate_result(result, data, limit_bytes=self._config.result_limit_bytes)
        return deserialize_result(data, limit_bytes=self._config.result_limit_bytes)  # type: ignore[return-value]

    def _set_acceptance_hook(self, hook: Callable[[ExecutionFuture[Any]], None]) -> None:
        """Install one private owner hook before this executor can accept work."""
        if not callable(hook):
            raise TypeError("hook must be callable")
        with self._condition:
            if self._submissions or self._preflights or self._acceptance_hook is not None:
                raise RuntimeError("acceptance ownership is already established")
            self._acceptance_hook = hook

    def _validate_controls(
        self,
        kwargs: Mapping[str, Any] | None,
        execution_timeout: float | None | Literal["inherit"],
        stream_output: bool | None,
        callbacks: Sequence[Callable[[ExecutionFuture[Any]], None]],
        output: ExecutionOutput | None,
        worker_setup: WorkerSetup | None,
    ) -> tuple[dict[str, Any], tuple[Callable[[ExecutionFuture[Any]], None], ...], float | None, bool, ExecutionOutput, WorkerSetup | None]:
        """Synchronously copy every control before quota, serializer, or factory work."""
        if kwargs is None:
            copied_kwargs: dict[str, Any] = {}
        elif not isinstance(kwargs, Mapping):
            raise TypeError("kwargs must be a mapping or None")
        else:
            copied_kwargs = dict(kwargs)
        if not all(isinstance(key, str) for key in copied_kwargs):
            raise TypeError("kwargs keys must be strings")
        if not isinstance(callbacks, Sequence):
            raise TypeError("done_callbacks must be a finite sequence")
        copied_callbacks = tuple(callbacks)
        if not all(callable(callback) for callback in copied_callbacks):
            raise TypeError("done_callbacks entries must be callable")
        if execution_timeout == "inherit":
            effective_timeout = self._config.execution_timeout
        elif execution_timeout is None:
            effective_timeout = None
        else:
            effective_timeout = self._positive_duration("execution_timeout", execution_timeout)
        if stream_output is None:
            effective_stream = self._config.stream_output
        elif not isinstance(stream_output, bool):
            raise TypeError("stream_output must be bool or None")
        else:
            effective_stream = stream_output
        if output is not None and not isinstance(output, ExecutionOutput):
            raise TypeError("output must be an ExecutionOutput or None")
        if worker_setup is not None and not isinstance(worker_setup, WorkerSetup):
            raise TypeError("worker_setup must be a WorkerSetup or None")
        if worker_setup is not None:
            # Reject caller-controlled setup data before it can reserve a spool or
            # launch a backend. Backend evidence is checked under the same limits.
            worker_setup.to_data(limit_bytes=min(
                self._config.owner_envelope_limit_bytes,
                self._config.admission_message_limit_bytes,
            ))
        return copied_kwargs, copied_callbacks, effective_timeout, effective_stream, ExecutionOutput() if output is None else output, worker_setup

    @staticmethod
    def _validate_future(future: object, submission_id: str, output: ExecutionOutput) -> None:
        """Reject a native/legacy/replacement Future before it can receive dispatch."""
        if not isinstance(future, ExecutionFuture):
            raise TypeError("create_future must return an execute.future.ExecutionFuture")
        if future.submission_id != submission_id or future.output is not output:
            raise ExecutionError("create_future returned a Future for another submission or output")
        if type(future) is ExecutionFuture or future.done() or future.running() or future.snapshot().state != "pending":
            raise ExecutionError("create_future must return an inert concrete Future")

    def _require_queryable(self) -> None:
        """Reject discovery after closure begins while retaining resource inspection."""
        with self._condition:
            if self._state not in {"new", "starting", "open"}:
                raise RuntimeError("executor is closing or closed")

    def _register_query(self) -> None:
        """Count one bounded backend observation so close cannot race its I/O."""
        with self._condition:
            self._queries += 1

    def _unregister_query(self) -> None:
        """Release one backend observation and wake a waiting close owner."""
        with self._condition:
            self._queries -= 1
            self._condition.notify_all()

    def _wait_terminal(self, future: ExecutionFuture[Any], deadline: float | None) -> None:
        """Wait for a known terminal outcome without changing it."""
        if future.done():
            return
        try:
            future.result(timeout=self._remaining(deadline))
        except TimeoutError as exc:
            if not future.done():
                raise CleanupError("executor close timed out waiting for work", execution=future) from exc
        except BaseException:
            return

    def _release_lease(self) -> None:
        """Release the executor-owned quota lease only after all spool cleanup succeeds."""
        with self._condition:
            lease = self._lease
        if lease is not None:
            lease.release()
            with self._condition:
                self._lease = None
                self._spooler = None

    def _cleanup_budget(self, deadline: float | None) -> float:
        """Choose a finite Future cleanup budget without hiding a close deadline."""
        remaining = self._remaining(deadline)
        if remaining is None:
            return self._config.termination_timeout
        if remaining <= 0:
            raise CleanupError("executor cleanup timed out")
        return remaining

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        """Return remaining monotonic budget without converting ``None`` to a bound."""
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    @staticmethod
    def _positive_duration(name: str, value: object) -> float:
        """Validate a finite positive per-call duration."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number of seconds")
        return float(value)

    @staticmethod
    def _nonnegative_duration(name: str, value: object) -> None:
        """Validate an optional close wait budget."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number of seconds")

    @staticmethod
    def _validate_requirements(environment: EnvironmentRequirement | None, world: WorldRequirement | None) -> None:
        """Reject mutable or foreign requirement controls before preflight ownership."""
        if environment is not None and not isinstance(environment, EnvironmentRequirement):
            raise TypeError("environment must be an EnvironmentRequirement or None")
        if world is not None and not isinstance(world, WorldRequirement):
            raise TypeError("world must be a WorldRequirement or None")

    @staticmethod
    def _resolve_environment_spec(
        spec: EnvironmentSpec | ResolvedEnvironmentSelection | None
    ) -> ResolvedEnvironmentSelection | None:
        """
        Resolve one optional exact selector before any payload preflight.

                ``None`` deliberately retains the prior unpinned execution
                path. A supplied
                selector is resolved exactly once for the operation and cannot
                fall back to
                configured candidates or the backend's default interpreter.
        """
        if spec is None:
            return None
        if isinstance(spec, ResolvedEnvironmentSelection):
            return spec
        if not isinstance(spec, EnvironmentSpec):
            raise TypeError(
                "environment_spec must be an EnvironmentSpec or None")
        return resolve_environment_spec(spec)

    @staticmethod
    def _validate_environment_spec(spec: EnvironmentSpec | None) -> None:
        """
        Validate one inert view selector without resolving or observing it.
        """

        if spec is not None and not isinstance(spec, EnvironmentSpec):
            raise TypeError(
                "environment_spec must be an EnvironmentSpec or None")

    @staticmethod
    def _capture_temp_parent(cwd: Path) -> Path:
        """Choose a stable platform temporary parent without probing it at construction."""
        names = ("TMPDIR", "TEMP", "TMP", "LOCALAPPDATA", "USERPROFILE") if os.name == "nt" else ("TMPDIR",)
        for name in names:
            value = os.environ.get(name)
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = cwd / candidate
                return candidate / "Temp" if name in {"LOCALAPPDATA", "USERPROFILE"} else candidate
        return Path("C:/Temp") if os.name == "nt" else Path("/tmp")

    def _incomplete_locked(self, message: str) -> None:
        """Publish retryable executor cleanup state while holding lifecycle ownership."""
        self._state = "cleanup_incomplete"
        self._condition.notify_all()
        raise CleanupError(message)


@dataclass(frozen=True, slots=True)
class ExecutorView:
    """Immutable controls bound to a parent executor without a second lifecycle.

    Every keyword accepted by :meth:`submit` and :meth:`run` is forwarded to the
    workload. The parent owns backend initialization, quota, Futures, and closure.
    """

    executor: Executor
    environment: EnvironmentRequirement | None
    world: WorldRequirement | None
    execution_timeout: float | None
    stream_output: bool
    done_callbacks: tuple[Callable[[ExecutionFuture[Any]], None], ...]
    output: ExecutionOutput | None
    worker_setup: WorkerSetup | None
    environment_spec: EnvironmentSpec | None = None

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> ExecutionFuture[T]:
        """Submit one workload with bound controls and unmodified workload keywords.

        Args:
            fn: Supported callable function root or importable unbound builtin.
            args: Positional workload arguments.
            kwargs: Workload keyword arguments; none are interpreted as controls.

        Returns:
            The parent backend's exact concrete accepted Future.

        Raises:
            TypeError: If callable transport, arguments, or bound controls are
                invalid.
            ValueError: If serialization or a bound timeout is invalid.
            RuntimeError: If the parent executor is closing or closed.
            ExecutionError: If preflight, quota, or backend Future validation fails.

        Side Effects:
            Delegates to the parent, which serializes one callable graph, reserves
            the shared spool budget, and starts asynchronous backend admission.
        """
        return self.executor.submit(
            fn,
            *args,
            kwargs=kwargs,
            environment=self.environment,
            environment_spec=self.environment_spec,
            world=self.world,
            execution_timeout=self.execution_timeout,
            stream_output=self.stream_output,
            done_callbacks=self.done_callbacks,
            output=self.output,
            worker_setup=self.worker_setup,
        )

    def run(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Submit one workload with bound controls and return its ordinary result.

        Args:
            fn: Supported callable function root or importable unbound builtin.
            args: Positional workload arguments.
            kwargs: Workload keyword arguments; none are interpreted as controls.

        Returns:
            The submitted workload's ordinary result.

        Raises:
            TypeError: If submission controls or callable transport are invalid.
            ValueError: If serialization or a bound timeout is invalid.
            BaseException: Any synchronous submission error or stable Future
                outcome, including admission, backend, and remote failures.

        Side Effects:
            Delegates submission to the parent but does not close the parent or
            clean up unrelated accepted Futures.
        """
        return self.submit(fn, *args, **kwargs).result()


_one_off_lock = Lock()
_one_off_owners: set["_OneOffOwner"] = set()


class _OneOffOwner:
    """Retain a hidden executor until one-off Future and preflight cleanup reconcile."""

    def __init__(self, config: BackendConfig) -> None:
        self.executor = Executor(config)
        self._future: ExecutionFuture[Any] | None = None
        self._monitor_started = False
        self._closing = False
        self._lock = Lock()
        self._monitor_wake = Event()
        self.executor._set_acceptance_hook(self._accept)
        with _one_off_lock:
            _one_off_owners.add(self)

    def submit(self, fn: Callable[..., T], args: tuple[Any, ...], controls: dict[str, Any]) -> ExecutionFuture[T]:
        """Accept one call and extend its cleanup hook to close this private owner."""
        try:
            future = self.executor.submit(fn, *args, **controls)
        except BaseException as workload_error:
            try:
                self._reconcile_owner(cancel=True)
            except CleanupError as cleanup_error:
                self._schedule_monitor(cancel=True)
                raise workload_error from cleanup_error
            raise
        return future

    def _accept(self, future: ExecutionFuture[T]) -> None:
        """Install final cleanup and monitoring before dispatch can publish terminality."""
        with self._lock:
            self._future = future

        def reconcile(timeout: float) -> None:
            self.executor._cleanup_submission(future, timeout)
            with self._lock:
                closing = self._closing
            if not closing:
                self._reconcile_owner(cancel=False, timeout=timeout)

        future._set_cleanup_reconciler(reconcile)
        future.add_done_callback(self._monitor_terminal)

    def _monitor_terminal(self, future: ExecutionFuture[Any]) -> None:
        """Perform bounded automatic one-off cleanup without polling or weak references."""
        deadline = time.monotonic() + self.executor._config.termination_timeout
        for _ in range(self.executor._config.one_off_cleanup_attempts):
            try:
                remaining = self.executor._remaining(deadline)
                if remaining is None or remaining <= 0:
                    break
                future.cleanup(timeout=remaining)
            except CleanupError:
                continue
            return
        self._schedule_monitor(cancel=False)

    def _reconcile_owner(self, *, cancel: bool, timeout: float | None = None) -> None:
        """Bounded preflight recovery retains this owner when storage remains dirty."""
        budget = self.executor._config.termination_timeout if timeout is None else timeout
        deadline = time.monotonic() + budget
        for _ in range(self.executor._config.one_off_cleanup_attempts):
            try:
                with self._lock:
                    self._closing = True
                remaining = self.executor._remaining(deadline)
                if remaining is None or remaining <= 0:
                    break
                self.executor.close(cancel=cancel, timeout=remaining)
            except CleanupError:
                continue
            finally:
                with self._lock:
                    self._closing = False
            self._finish()
            return
        raise CleanupError("one-off owner cleanup remains incomplete", execution=self._future)

    def _schedule_monitor(self, *, cancel: bool) -> None:
        """Keep incomplete hidden ownership active until a later bounded retry succeeds."""
        with self._lock:
            if self._monitor_started:
                self._monitor_wake.set()
                return
            self._monitor_started = True
        try:
            Thread(target=self._monitor_incomplete, args=(cancel,), name="dryml-execute-one-off-cleanup", daemon=True).start()
        except BaseException:
            with self._lock:
                self._monitor_started = False

    def _monitor_incomplete(self, cancel: bool) -> None:
        """Retry retained cleanup on an explicit wakeup or configured cadence."""
        interval = self.executor._config.one_off_cleanup_retry_interval
        while True:
            self._monitor_wake.wait(interval)
            self._monitor_wake.clear()
            with self._lock:
                future = self._future
            try:
                if future is not None and future.done():
                    future.cleanup(timeout=self.executor._config.termination_timeout)
                else:
                    self._reconcile_owner(cancel=cancel)
            except (CleanupError, RuntimeError):
                continue
            return

    def _finish(self) -> None:
        """Release the process-retained manager only after successful owner closure."""
        with _one_off_lock:
            _one_off_owners.discard(self)


def submit(
    fn: Callable[..., T],
    /,
    *args: Any,
    backend: BackendConfig,
    kwargs: Mapping[str, Any] | None = None,
    environment: EnvironmentRequirement | None = None,
    environment_spec: EnvironmentSpec | None = None,
    world: WorldRequirement | None = None,
    execution_timeout: float | None | Literal["inherit"] = "inherit",
    stream_output: bool | None = None,
    done_callbacks: Sequence[Callable[[ExecutionFuture[T]], None]] = (),
    output: ExecutionOutput | None = None,
    worker_setup: WorkerSetup | None = None,
) -> ExecutionFuture[T]:
    """
    Submit one explicit-backend call while retaining a hidden owner through
    cleanup.

        Args:
            fn: Supported workload callable.
            args: Ordinary positional workload arguments.
            backend: Required explicit backend configuration.
            kwargs: Optional workload keyword mapping.
            environment: Optional owner-defined software requirement.
            environment_spec: Optional exact existing selector resolved once at
            call
                entry; an unsupported or mismatched pin never falls back.
            world: Optional owner-defined world requirement.
            execution_timeout: Inherited, disabled, or positive workload
            deadline.
            stream_output: Optional live-output override.
            done_callbacks: Completion callbacks copied before acceptance.
            output: Optional single-use retained output owner.
            worker_setup: Optional inert worker-local setup control.

        Returns:
            The backend's concrete accepted Future.

        Raises:
            TypeError, ValueError, RuntimeError, ExecutionError: The
            corresponding
                :meth:`Executor.submit` validation or preflight failures.

        Side Effects:
            Creates a hidden one-off executor. Future cleanup reconciles this
            call and
            closes that owner without touching reusable callers.
    """
    if not isinstance(backend, BackendConfig):
        raise TypeError("backend must be a BackendConfig")
    owner = _OneOffOwner(backend)
    return owner.submit(
        fn,
        args,
        dict(
            kwargs=kwargs,
            environment=environment,
            environment_spec=environment_spec,
            world=world,
            execution_timeout=execution_timeout,
            stream_output=stream_output,
            done_callbacks=done_callbacks,
            output=output,
            worker_setup=worker_setup,
        ),
    )


def run(
    fn: Callable[..., T],
    /,
    *args: Any,
    backend: BackendConfig,
    kwargs: Mapping[str, Any] | None = None,
    environment: EnvironmentRequirement | None = None,
    environment_spec: EnvironmentSpec | None = None,
    world: WorldRequirement | None = None,
    execution_timeout: float | None | Literal["inherit"] = "inherit",
    stream_output: bool | None = None,
    done_callbacks: Sequence[Callable[[ExecutionFuture[T]], None]] = (),
    output: ExecutionOutput | None = None,
    worker_setup: WorkerSetup | None = None,
) -> T:
    """
    Run one explicit-backend call, then perform bounded hidden-owner cleanup.

        Args are identical to :func:`submit`, including an ``environment_spec``
        that
        resolves once at call entry and never falls back. ``worker_setup``
        enters only
        in the admitted worker.

        Returns:
            The ordinary workload value after hidden-owner cleanup completes.

        Raises:
            BaseException: Submission or workload failures. A cleanup-only
            failure
                raises :class:`CleanupError` retaining the Future; if both
                fail, the
                workload exception stays primary with cleanup as its cause.

        Side Effects:
            Creates and reconciles a hidden one-off executor.
    """
    future = submit(
        fn,
        *args,
        backend=backend,
        kwargs=kwargs,
        environment=environment,
        environment_spec=environment_spec,
        world=world,
        execution_timeout=execution_timeout,
        stream_output=stream_output,
        done_callbacks=done_callbacks,
        output=output,
        worker_setup=worker_setup,
    )
    try:
        result = future.result()
    except BaseException as workload_error:
        try:
            future.cleanup()
        except CleanupError as cleanup_error:
            raise workload_error from cleanup_error
        raise
    future.cleanup()
    return result


__all__ = ["Executor", "ExecutorView", "run", "submit"]
