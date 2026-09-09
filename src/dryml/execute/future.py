"""Thread-safe execution Futures and the temporary legacy orchestration wrapper."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import CancelledError
from threading import Condition, RLock, Thread
from typing import Any, Generic, TypeVar

from .errors import AdmissionError, CleanupError, ExecutionDeadlineExceeded, ExecutionError, ExecutionUncertainError
from .models import ExecutionIssue, ExecutionSnapshot
from .output import ExecutionOutput
from .transfer import restore_result


T = TypeVar("T")
_MISSING = object()


class ExecutionFuture(Generic[T]):
    """Coordinate one execution's stable outcome, callbacks, cleanup, and output.

    Backends construct their concrete subclass before accepting work, install any
    submission callbacks, and use the private lifecycle hooks to authorize work
    and publish its first terminal outcome.  The Future never imports a backend
    or native Future.  Its stored result/error and retained output therefore stay
    inspectable while backend cleanup continues or is retried.

    Args:
        submission_id: Coordinator-unique correlation identifier for this call.
        output: Optional caller-owned retained output object.  A fresh object is
            created when omitted and is bound later by submission acceptance.
        termination_timeout: Default finite positive timeout for ``cleanup()``.
        initial_callbacks: Coordinator-only callbacks copied before acceptance.
        cleanup_scope: Backend-owned worker boundary being reconciled.

    Raises:
        TypeError: If construction controls have unsupported types.
        ValueError: If the identifier or default cleanup timeout is invalid.
    """

    def __init__(
        self,
        submission_id: str,
        *,
        output: ExecutionOutput | None = None,
        termination_timeout: float = 5.0,
        initial_callbacks: Sequence[Callable[["ExecutionFuture[T]"], None]] = (),
        cleanup_scope: str = "worker",
    ) -> None:
        """Create an inert, unstarted execution state with copied callbacks."""
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError("submission_id must be a nonempty string")
        self._validate_duration("termination_timeout", termination_timeout)
        if output is not None and not isinstance(output, ExecutionOutput):
            raise TypeError("output must be an ExecutionOutput or None")
        if cleanup_scope not in {"worker", "owned-group"}:
            raise ValueError("cleanup_scope must be worker or owned-group")
        self._submission_id = submission_id
        self._output = ExecutionOutput() if output is None else output
        self._termination_timeout = float(termination_timeout)
        self._cleanup_scope = cleanup_scope
        self._condition = Condition(RLock())
        self._state = "pending"
        self._go = False
        self._cancel_requested = False
        self._outcome: object = _MISSING
        self._outcome_kind: str | None = None
        self._callbacks = list(initial_callbacks)
        if not all(callable(callback) for callback in self._callbacks):
            raise TypeError("initial_callbacks entries must be callable")
        self._cleanup_state = "pending"
        self._cleanup_issues: list[ExecutionIssue] = []
        self._cleanup_reconciler: Callable[[float], None] | None = None
        self._diagnostic_text_limit_bytes = 65_536
        self._diagnostic_issue_limit = 64
        self._cancel_requester: Callable[[], bool] | None = None
        self._result_receiver: Callable[[bytes], T] | None = None
        self._callback_queue: deque[Callable[[ExecutionFuture[T]], None]] = deque()
        self._backend_job_id: str | None = None
        self._worker_id: str | None = None
        self._pid: int | None = None
        self._environment = None
        self._allocation = None
        self._report = None

    @property
    def submission_id(self) -> str:
        """Return this execution's immutable coordinator correlation identifier."""
        return self._submission_id

    @property
    def output(self) -> ExecutionOutput:
        """Return the retained output owner that survives backend cleanup."""
        return self._output

    def snapshot(self) -> ExecutionSnapshot:
        """Return an immutable execution and cleanup observation.

        Returns:
            A snapshot with stable outcome state and independently changing
            cleanup state, association fields, cancellation request, and report.

        Side Effects:
            None.
        """
        with self._condition:
            return ExecutionSnapshot(
                submission_id=self._submission_id,
                state=self._state,
                cleanup_state=self._cleanup_state,
                cleanup_scope=self._cleanup_scope,
                cleanup_issues=tuple(self._cleanup_issues),
                backend_job_id=self._backend_job_id,
                worker_id=self._worker_id,
                pid=self._pid,
                environment=self._environment,
                allocation=self._allocation,
                cancel_requested=self._cancel_requested,
                report=self._report,
            )

    def done(self) -> bool:
        """Return whether a stable result, error, cancellation, or uncertainty is available."""
        with self._condition:
            return self._outcome is not _MISSING

    def running(self) -> bool:
        """Return whether this Future has passed GO and its workload may be executing."""
        with self._condition:
            return self._state == "running"

    def cancelled(self) -> bool:
        """Return whether pre-invocation cancellation was confirmed."""
        with self._condition:
            return self._state == "cancelled"

    def result(self, timeout: float | None = None) -> T:
        """Wait for and return the result or raise the stable terminal error.

        Args:
            timeout: Optional maximum wait in seconds.  Expiry raises
                ``TimeoutError`` without cancelling or storing an outcome.

        Raises:
            TimeoutError: If no terminal outcome is available in time.
            CancelledError: If cancellation was confirmed by the backend.
            BaseException: The stored execution failure or uncertainty.
        """
        outcome = self._wait_outcome(timeout)
        if self._outcome_kind in {"error", "cancelled"}:
            assert isinstance(outcome, BaseException)
            raise outcome
        return outcome  # type: ignore[return-value]

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Wait for and return the stored failure, if any.

        Args:
            timeout: Optional maximum wait in seconds.  It does not alter work.

        Raises:
            TimeoutError: If no terminal outcome is available in time.
            CancelledError: If cancellation was confirmed by the backend.
        """
        outcome = self._wait_outcome(timeout)
        if self._outcome_kind == "cancelled":
            raise outcome
        return outcome if self._outcome_kind == "error" else None

    def add_done_callback(self, callback: Callable[["ExecutionFuture[T]"], None]) -> None:
        """Schedule a coordinator-local callback once this Future is terminal.

        Args:
            callback: Callable receiving this exact concrete Future instance.

        Raises:
            TypeError: If ``callback`` is not callable.

        Side Effects:
            Callback execution is detached from Future locks and backend readers.
            Callback errors are isolated and never alter the execution outcome.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._outcome is _MISSING:
                self._callbacks.append(callback)
                return
        self._dispatch_callbacks((callback,))

    def cancel(self) -> bool:
        """Confirm cancellation only when invocation has not passed the GO gate.

        Returns:
            ``True`` when this call prevented invocation and published confirmed
            cancellation; ``False`` after authorization or terminal publication.
        """
        callbacks = self._publish_cancel_if_prestart()
        if callbacks is None:
            return False
        self._output._outcome_known()
        self._dispatch_callbacks(callbacks)
        return True

    def request_cancel(self) -> bool:
        """Request backend termination for running work without confirming it.

        Returns:
            ``True`` only when the configured backend requester accepts the
            running-work request.  Acceptance does not change the outcome.
        """
        with self._condition:
            if self._state != "running" or self._outcome is not _MISSING:
                return False
            requester = self._cancel_requester
        if requester is None:
            raise ExecutionError("running cancellation is unavailable for this execution", report=self._report)
        try:
            accepted = requester()
        except BaseException:
            raise ExecutionError("running cancellation request failed", report=self._report) from None
        if not accepted:
            return False
        with self._condition:
            self._cancel_requested = True
        return True

    def cleanup(self, timeout: float | None = None) -> None:
        """Reconcile only this execution's cleanup after it reaches terminality.

        Args:
            timeout: Optional finite positive reconciliation budget.  ``None``
                uses the configured termination timeout.

        Raises:
            RuntimeError: If the workload has not reached a terminal outcome.
            CleanupError: If cleanup is unconfigured, still in progress beyond
                the budget, or fails; its ``execution`` is this Future.
        """
        budget = self._termination_timeout if timeout is None else timeout
        self._validate_duration("timeout", budget)
        deadline = time.monotonic() + float(budget)
        start_reconciler = False
        reconciler: Callable[[float], None] | None = None
        with self._condition:
            if self._outcome is _MISSING:
                raise RuntimeError("cleanup requires a terminal execution outcome")
            if self._cleanup_state == "complete":
                return
            if self._cleanup_state != "reconciling":
                reconciler = self._cleanup_reconciler
                if reconciler is None:
                    self._cleanup_state = "incomplete"
                    self._append_cleanup_issue_locked(RuntimeError("cleanup reconciliation is not configured"))
                    raise self._cleanup_error("cleanup reconciliation is not configured")
                self._cleanup_state = "reconciling"
                start_reconciler = True
        if start_reconciler:
            try:
                Thread(target=self._run_cleanup_reconciler, args=(reconciler, float(budget)), name="dryml-execute-cleanup", daemon=True).start()
            except BaseException as failure:
                with self._condition:
                    self._cleanup_state = "incomplete"
                    self._append_cleanup_issue_locked(failure)
                    self._condition.notify_all()
                raise self._cleanup_error("cleanup reconciliation could not start") from None
        with self._condition:
            while self._cleanup_state == "reconciling":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._cleanup_error("cleanup reconciliation is still running")
                self._condition.wait(remaining)
            if self._cleanup_state == "complete" and time.monotonic() <= deadline:
                return
            if self._cleanup_state == "complete":
                raise self._cleanup_error("cleanup reconciliation completed after the caller budget")
            raise self._cleanup_error("cleanup reconciliation remains incomplete")

    def __await__(self) -> Generator[Any, None, T]:
        """Await this Future through a fresh loop-local waiter without blocking the event loop."""
        return self._await_result().__await__()

    async def _await_result(self) -> T:
        """Bridge terminal callbacks into the current event loop without propagating cancellation."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[T] = loop.create_future()

        def completed(value: ExecutionFuture[T]) -> None:
            def deliver() -> None:
                if waiter.done():
                    return
                try:
                    waiter.set_result(value.result())
                except BaseException as exc:
                    waiter.set_exception(exc)

            try:
                loop.call_soon_threadsafe(deliver)
            except RuntimeError:
                pass

        self.add_done_callback(completed)
        try:
            return await asyncio.shield(waiter)
        finally:
            self._remove_callback(completed)
            if not waiter.done():
                waiter.cancel()

    def _install_initial_callbacks(self, callbacks: Sequence[Callable[["ExecutionFuture[T]"], None]]) -> None:
        """Copy submission callbacks before scheduling can reach terminal publication."""
        copied = tuple(callbacks)
        if not all(callable(callback) for callback in copied):
            raise TypeError("initial callbacks entries must be callable")
        with self._condition:
            if self._outcome is _MISSING:
                self._callbacks.extend(copied)
                return
        self._dispatch_callbacks(copied)

    def _begin_admission(self) -> bool:
        """Move accepted work into admission unless a pre-start terminal state already won."""
        with self._condition:
            if self._outcome is not _MISSING or self._state != "pending":
                return False
            self._state = "admitting"
            return True

    def _authorize(self, *, deadline: float | None = None) -> bool:
        """Atomically choose GO over pre-start cancellation or expired admission."""
        callbacks: tuple[Callable[[ExecutionFuture[T]], None], ...] = ()
        expired = False
        with self._condition:
            if self._outcome is not _MISSING or self._state != "admitting":
                return False
            if deadline is not None and time.monotonic() >= deadline:
                callbacks = self._publish_locked(AdmissionError("admission deadline exceeded"), "failed", "error")
                expired = True
            else:
                self._go = True
                self._state = "running"
                return True
        if expired:
            self._output._outcome_known()
            self._dispatch_callbacks(callbacks)
        return False

    def _publish_result(self, value: T) -> bool:
        """Publish a validated result if no other terminal contender has won."""
        return self._publish_terminal(value, "succeeded")

    def _publish_exception(self, error: BaseException) -> bool:
        """Publish a validated execution failure if no terminal contender has won."""
        if not isinstance(error, BaseException):
            raise TypeError("error must be a BaseException")
        return self._publish_terminal(error, "failed")

    def _publish_uncertain(self, error: BaseException | None = None) -> bool:
        """Publish an unconfirmed terminal outcome without presenting success."""
        if error is None:
            error = ExecutionUncertainError("execution outcome is uncertain")
        if not isinstance(error, BaseException):
            raise TypeError("error must be a BaseException")
        return self._publish_terminal(error, "uncertain")

    def _expire(self, error: ExecutionDeadlineExceeded | None = None) -> bool:
        """Publish confirmed deadline expiry without treating output fences as outcome evidence."""
        return self._publish_terminal(error or ExecutionDeadlineExceeded("execution deadline exceeded"), "failed")

    def _set_cleanup_reconciler(self, reconciler: Callable[[float], None]) -> None:
        """Install the U3/U5 per-submission reconciliation hook before cleanup begins."""
        if not callable(reconciler):
            raise TypeError("reconciler must be callable")
        with self._condition:
            if self._cleanup_state == "reconciling":
                raise RuntimeError("cleanup reconciliation is already running")
            self._cleanup_reconciler = reconciler

    def _prepare(self, *, diagnostic_text_limit_bytes: int, diagnostic_issue_limit: int, termination_timeout: float | None = None) -> None:
        """Install validated effective diagnostic limits before backend admission.

        Args:
            diagnostic_text_limit_bytes: Maximum UTF-8 bytes retained per cleanup
                diagnostic message.
            diagnostic_issue_limit: Maximum retained cleanup diagnostics.

        Raises:
            TypeError: If either limit is not an integer.
            ValueError: If either limit is not positive or preparation is late.
        """
        for name, value in (("diagnostic_text_limit_bytes", diagnostic_text_limit_bytes), ("diagnostic_issue_limit", diagnostic_issue_limit)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if termination_timeout is not None:
            self._validate_duration("termination_timeout", termination_timeout)
        with self._condition:
            if self._state != "pending" or self._outcome is not _MISSING:
                raise RuntimeError("diagnostic limits must be prepared before admission")
            self._diagnostic_text_limit_bytes = diagnostic_text_limit_bytes
            self._diagnostic_issue_limit = diagnostic_issue_limit
            if termination_timeout is not None:
                self._termination_timeout = float(termination_timeout)

    def _set_cancel_requester(self, requester: Callable[[], bool]) -> None:
        """Install the U3/U5 running-work cancellation request hook."""
        if not callable(requester):
            raise TypeError("requester must be callable")
        with self._condition:
            self._cancel_requester = requester

    def _set_result_receiver(self, receiver: Callable[[bytes], T]) -> None:
        """Install the executor-owned reserved result-spool receive hook.

        Backends use this only after validating their transport result frame. The
        receiver remains coordinator-local and is never serialized to a worker.
        """
        if not callable(receiver):
            raise TypeError("result receiver must be callable")
        with self._condition:
            if self._state != "pending" or self._outcome is not _MISSING:
                raise RuntimeError("result receiver must be installed before admission")
            self._result_receiver = receiver

    def _receive_result(self, data: bytes) -> T:
        """Delegate one validated result payload to the accepted executor owner."""
        with self._condition:
            receiver = self._result_receiver
        if receiver is None:
            raise ExecutionError("execution has no authorized result receiver")
        return receiver(data)

    def _set_association(self, *, backend_job_id: str | None | object = _MISSING, worker_id: str | None | object = _MISSING, pid: int | None | object = _MISSING, environment: Any = _MISSING, allocation: Any = _MISSING, report: Any = _MISSING) -> None:
        """Update coordinator-observed U3/U5 association fields without changing outcome."""
        with self._condition:
            if backend_job_id is not _MISSING:
                self._backend_job_id = backend_job_id
            if worker_id is not _MISSING:
                self._worker_id = worker_id
            if pid is not _MISSING:
                self._pid = pid
            if environment is not _MISSING:
                self._environment = environment
            if allocation is not _MISSING:
                self._allocation = allocation
            if report is not _MISSING:
                self._report = report

    def _publish_cancel_if_prestart(self) -> tuple[Callable[["ExecutionFuture[T]"], None], ...] | None:
        """Linearize pre-start cancellation against the private GO authorization hook."""
        with self._condition:
            if self._state not in {"pending", "admitting"} or self._go or self._outcome is not _MISSING:
                return None
            return self._publish_locked(CancelledError(), "cancelled", "cancelled")

    def _publish_running_cancellation(self) -> bool:
        """Publish backend-confirmed running cancellation independently of its request.

        Returns:
            ``True`` only when a running execution had no earlier terminal outcome.
        """
        with self._condition:
            if self._state != "running" or self._outcome is not _MISSING:
                return False
            callbacks = self._publish_locked(CancelledError(), "cancelled", "cancelled")
        self._output._outcome_known()
        self._dispatch_callbacks(callbacks)
        return True

    def _publish_terminal(self, outcome: object, state: str, kind: str | None = None) -> bool:
        """Store exactly one terminal outcome, then notify waiters and detach callbacks."""
        with self._condition:
            if self._outcome is not _MISSING:
                return False
            callbacks = self._publish_locked(outcome, state, kind or ("result" if state == "succeeded" else "error"))
        self._output._outcome_known()
        self._dispatch_callbacks(callbacks)
        return True

    def _publish_locked(self, outcome: object, state: str, kind: str) -> tuple[Callable[["ExecutionFuture[T]"], None], ...]:
        """Set terminal fields while holding the per-submission condition exactly once."""
        self._outcome = outcome
        self._outcome_kind = kind
        self._state = state
        callbacks = tuple(self._callbacks)
        self._callbacks.clear()
        self._condition.notify_all()
        return callbacks

    def _wait_outcome(self, timeout: float | None) -> object:
        """Wait on the condition without turning a caller wait timeout into execution state."""
        if timeout is not None:
            self._validate_nonnegative_duration("timeout", timeout)
        with self._condition:
            if self._outcome is _MISSING and timeout is not None:
                deadline = time.monotonic() + float(timeout)
                while self._outcome is _MISSING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("execution did not finish before timeout")
                    self._condition.wait(remaining)
            while self._outcome is _MISSING:
                self._condition.wait()
            return self._outcome

    def _dispatch_callbacks(self, callbacks: Sequence[Callable[["ExecutionFuture[T]"], None]]) -> None:
        """Dispatch callbacks independently without letting thread launch errors escape.

        A launch is retried once immediately.  If both launches fail, callbacks
        remain owned in the queue and the next registration/publication retries.
        Each successful callback has its own thread, so a blocked callback cannot
        delay a sibling's callback or protocol progress.
        """
        with self._condition:
            pending = tuple(self._callback_queue) + tuple(callbacks)
            self._callback_queue.clear()
        failed: list[Callable[[ExecutionFuture[T]], None]] = []
        for callback in pending:
            for _ in range(2):
                try:
                    Thread(target=self._run_callback, args=(callback,), name="dryml-execute-callback", daemon=True).start()
                    break
                except BaseException:
                    pass
            else:
                failed.append(callback)
        with self._condition:
            self._callback_queue.extend(failed)

    def _remove_callback(self, callback: Callable[["ExecutionFuture[T]"], None]) -> None:
        """Remove a not-yet-dispatched internal await bridge after local cancellation."""
        with self._condition:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                try:
                    self._callback_queue.remove(callback)
                except ValueError:
                    pass

    def _run_callback(self, callback: Callable[["ExecutionFuture[T]"], None]) -> None:
        """Isolate one user callback from execution machinery and sibling callbacks."""
        try:
            callback(self)
        except BaseException:
            pass

    def _run_cleanup_reconciler(self, reconciler: Callable[[float], None], timeout: float) -> None:
        """Run one backend cleanup hook outside the submission lock and wake bounded joiners."""
        try:
            reconciler(timeout)
        except BaseException as failure:
            with self._condition:
                self._cleanup_state = "incomplete"
                self._append_cleanup_issue_locked(failure)
                self._condition.notify_all()
        else:
            with self._condition:
                self._cleanup_state = "complete"
                self._condition.notify_all()

    def _append_cleanup_issue_locked(self, failure: BaseException) -> None:
        """Retain one bounded cleanup diagnostic without exposing arbitrary failure payloads."""
        if len(self._cleanup_issues) < self._diagnostic_issue_limit:
            self._cleanup_issues.append(ExecutionIssue("cleanup_failed", self._bounded_text(type(failure).__name__)))

    def _bounded_text(self, value: str) -> str:
        """Return a UTF-8 byte-bounded diagnostic without retaining arbitrary text."""
        text = value.encode("utf-8")[:self._diagnostic_text_limit_bytes].decode("utf-8", errors="replace")
        while len(text.encode("utf-8")) > self._diagnostic_text_limit_bytes:
            text = text[:-1]
        return text

    def _cleanup_error(self, message: str) -> CleanupError:
        """Create the required recovery error while preserving this Future reference."""
        return CleanupError(message, report=self._report, execution=self)

    @staticmethod
    def _validate_duration(name: str, value: object) -> None:
        """Validate a finite positive cleanup-duration control."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite positive number of seconds")
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite positive number of seconds")

    @staticmethod
    def _validate_nonnegative_duration(name: str, value: object) -> None:
        """Validate a finite nonnegative caller wait duration."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite nonnegative number of seconds")
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite nonnegative number of seconds")


class OrchestratedFuture:
    def __init__(self, *, backend_future, prepared, update_targets, repo=None):
        self.backend_future = backend_future
        self.prepared = prepared
        self.update_targets = update_targets
        self.repo = repo
        self._result = None
        self._has_result = False

    def done(self) -> bool:
        return self.backend_future.done()

    def cancel(self) -> bool:
        return self.backend_future.cancel()

    def exception(self, timeout: float | None = None):
        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None

    def result(self, timeout: float | None = None):
        if self._has_result:
            return self._result

        response = self.backend_future.result(timeout=timeout)
        result = restore_result(
            response,
            repo=self.repo,
            result_store=self.prepared.result_store,
        )
        self._result = result
        self._has_result = True
        return result


__all__ = ["ExecutionFuture", "OrchestratedFuture"]
