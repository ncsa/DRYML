from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from threading import Event

import pytest

from dryml.execute.errors import AdmissionError, CleanupError, ExecutionDeadlineExceeded, ExecutionError, ExecutionUncertainError
from dryml.execute.future import ExecutionFuture


class ConcreteFuture(ExecutionFuture[int]):
    """Concrete test future proving callbacks retain the constructed subtype."""


def test_future_preserves_one_outcome_and_wait_timeouts_leave_it_live():
    """A wait timeout never stores an outcome, and the first terminal publisher wins."""
    future = ConcreteFuture("one")

    with pytest.raises(TimeoutError):
        future.result(timeout=0.001)
    with pytest.raises(TimeoutError):
        future.exception(timeout=0.001)
    assert not future.done()

    assert future._publish_result(3)
    assert not future._publish_exception(ValueError("late"))
    assert future.result() == 3
    assert future.exception() is None
    assert future.snapshot().state == "succeeded"


@pytest.mark.parametrize("value", [ValueError("returned"), CancelledError("returned")])
def test_future_treats_returned_exception_instances_as_values(value: BaseException):
    """A callable may successfully return any object, including an exception instance."""
    future = ExecutionFuture[BaseException]("exception-value")

    assert future._publish_result(value)
    assert future.result() is value
    assert future.exception() is None


def test_cancel_gate_deadline_and_uncertain_outcomes_are_distinct():
    """Cancellation only wins before GO, while deadline and uncertainty retain errors."""
    cancelled = ConcreteFuture("cancelled")
    assert cancelled.cancel()
    assert cancelled.cancelled()
    with pytest.raises(CancelledError):
        cancelled.result()

    running = ConcreteFuture("running")
    assert running._begin_admission()
    assert running._authorize()
    assert not running.cancel()
    running._set_cancel_requester(lambda: True)
    assert running.request_cancel()
    assert running.snapshot().cancel_requested
    assert running._publish_uncertain(ExecutionUncertainError("worker lost"))
    with pytest.raises(ExecutionUncertainError, match="worker lost"):
        running.result()

    expired = ConcreteFuture("expired")
    assert expired._expire(ExecutionDeadlineExceeded("deadline exceeded"))
    with pytest.raises(ExecutionDeadlineExceeded):
        expired.result()


def test_admission_and_running_cancellation_have_one_shot_transitions():
    """Only admitting work crosses GO, while admitted cancellation errors stay actionable."""
    future = ConcreteFuture("admission")
    assert future._begin_admission()
    assert not future._begin_admission()
    assert future._authorize()
    assert not future._authorize()
    assert not future._begin_admission()

    expired = ConcreteFuture("admission-expired")
    assert expired._begin_admission()
    assert not expired._authorize(deadline=0.0)
    with pytest.raises(AdmissionError):
        expired.result()

    unsupported = ConcreteFuture("unsupported-cancel")
    assert unsupported._begin_admission()
    assert unsupported._authorize()
    with pytest.raises(ExecutionError, match="unavailable"):
        unsupported.request_cancel()

    failing = ConcreteFuture("failing-cancel")
    assert failing._begin_admission()
    assert failing._authorize()
    failing._set_cancel_requester(lambda: (_ for _ in ()).throw(RuntimeError("private details")))
    with pytest.raises(ExecutionError, match="failed"):
        failing.request_cancel()

    raced = ConcreteFuture("accepted-cancel")
    assert raced._begin_admission()
    assert raced._authorize()

    def accepted_after_result() -> bool:
        assert raced._publish_result(1)
        return True

    raced._set_cancel_requester(accepted_after_result)
    assert raced.request_cancel()
    assert raced.result() == 1
    assert raced.snapshot().cancel_requested

    confirmed = ConcreteFuture("confirmed-cancel")
    assert confirmed._begin_admission()
    assert confirmed._authorize()
    assert confirmed._publish_running_cancellation()
    with pytest.raises(CancelledError):
        confirmed.result()


def test_callbacks_snapshot_duplicates_reentrancy_and_late_registration():
    """Callback copies, duplicate registrations, and callback failures cannot block outcome use."""
    callbacks = []
    calls: list[tuple[str, ConcreteFuture]] = []
    delivered = Event()

    def callback(value: ConcreteFuture) -> None:
        calls.append(("ok", value))
        assert value.result() == 4
        if len(calls) == 2:
            delivered.set()

    callbacks.extend((callback, callback))
    future = ConcreteFuture("callbacks", initial_callbacks=callbacks)
    callbacks.clear()
    future.add_done_callback(lambda value: (_ for _ in ()).throw(RuntimeError("ignored")))
    assert future._publish_result(4)
    assert delivered.wait(1)

    late = Event()
    future.add_done_callback(lambda value: late.set())
    assert late.wait(1)
    assert calls == [("ok", future), ("ok", future)]
    assert all(isinstance(value, ConcreteFuture) for _, value in calls)


def test_blocked_callback_cannot_prevent_other_callbacks_or_result_progress():
    """A callback waiting on another Future is isolated from completion and sibling callbacks."""
    future = ConcreteFuture("callback-wait")
    dependency = ConcreteFuture("dependency")
    waiting = Event()
    sibling = Event()

    def wait_for_dependency(value: ConcreteFuture) -> None:
        waiting.set()
        assert dependency.result(timeout=1) == 8
        assert value.result() == 5

    future.add_done_callback(wait_for_dependency)
    future.add_done_callback(lambda value: sibling.set())
    assert future._publish_result(5)
    assert waiting.wait(1)
    assert sibling.wait(1)
    assert future.result() == 5
    assert dependency._publish_result(8)


def test_cleanup_is_independent_retryable_and_keeps_known_result():
    """Cleanup joins one reconciler, reports failure with this future, and later retries."""
    future = ConcreteFuture("cleanup", termination_timeout=0.25)
    with pytest.raises(RuntimeError, match="terminal"):
        future.cleanup()

    attempts: list[float] = []

    def reconcile(timeout: float) -> None:
        attempts.append(timeout)
        if len(attempts) == 1:
            raise RuntimeError("still running")

    future._set_cleanup_reconciler(reconcile)
    future._publish_result(7)
    with pytest.raises(CleanupError) as failure:
        future.cleanup()
    assert failure.value.execution is future
    assert future.result() == 7
    assert future.snapshot().cleanup_state == "incomplete"

    future.cleanup(timeout=0.1)
    future.cleanup(timeout=0.1)
    assert attempts == [0.25, 0.1]
    assert future.snapshot().cleanup_state == "complete"


def test_worker_cleanup_evidence_keeps_successful_result_but_blocks_reconciliation():
    """Unobserved worker teardown still releases independent local resources."""
    future = ConcreteFuture("worker-cleanup", termination_timeout=0.1)
    attempts: list[float] = []
    future._set_cleanup_reconciler(lambda timeout: attempts.append(timeout))
    future._record_worker_cleanup_issues(("RuntimeError",))
    future._publish_result(9)

    assert future.result() == 9
    assert future.snapshot().cleanup_state == "incomplete"
    for _ in range(2):
        with pytest.raises(CleanupError) as failure:
            future.cleanup()
        assert failure.value.execution is future
        assert future.snapshot().cleanup_state == "incomplete"
    assert attempts == [0.1, 0.1]


def test_cleanup_start_failure_is_retryable_and_diagnostics_use_prepared_limits(monkeypatch):
    """A failed cleanup worker launch never strands reconciliation or bypasses caps."""
    future = ConcreteFuture("cleanup-start", termination_timeout=0.1)
    future._prepare(diagnostic_text_limit_bytes=3, diagnostic_issue_limit=1)
    future._set_cleanup_reconciler(lambda timeout: None)
    future._publish_result(1)
    original_start = __import__("threading").Thread.start
    calls = 0

    def fail_once(thread):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("thread unavailable")
        return original_start(thread)

    monkeypatch.setattr("dryml.execute.future.Thread.start", fail_once)
    with pytest.raises(CleanupError) as failure:
        future.cleanup()
    assert failure.value.execution is future
    assert future.snapshot().cleanup_state == "incomplete"
    assert future.snapshot().cleanup_issues == (future.snapshot().cleanup_issues[0],)
    assert len(future.snapshot().cleanup_issues[0].message.encode("utf-8")) <= 3
    future.cleanup()
    assert future.snapshot().cleanup_state == "complete"


def test_cleanup_timeout_keeps_reconciliation_pending_until_the_hook_really_succeeds():
    """Caller timeout is not cleanup success; a later completed hook remains retry-observable."""
    future = ConcreteFuture("cleanup-budget", termination_timeout=0.1)
    entered = Event()
    release = Event()
    completed = Event()

    def reconcile(timeout: float) -> None:
        entered.set()
        assert release.wait(1)
        completed.set()

    future._set_cleanup_reconciler(reconcile)
    future._publish_result(1)
    with pytest.raises(CleanupError):
        future.cleanup(timeout=0.01)
    assert entered.is_set()
    assert future.snapshot().cleanup_state == "reconciling"
    release.set()
    assert completed.wait(1)
    future.cleanup(timeout=0.1)
    assert future.snapshot().cleanup_state == "complete"


def test_callback_delivery_retries_a_thread_start_failure_without_rewriting_outcome(monkeypatch):
    """A detached-delivery launch failure retries at the next safe consumer seam."""
    future = ConcreteFuture("callback-start")
    delivered = Event()
    late = Event()
    future.add_done_callback(lambda value: delivered.set())
    original_start = __import__("threading").Thread.start
    calls = 0

    def fail_twice(thread):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("thread unavailable")
        return original_start(thread)

    monkeypatch.setattr("dryml.execute.future.Thread.start", fail_twice)
    assert future._publish_result(2)
    assert future.result() == 2
    assert delivered.wait(1)
    monkeypatch.setattr("dryml.execute.future.Thread.start", original_start)
    future.add_done_callback(lambda value: late.set())
    assert late.wait(1)


def test_two_independent_awaiters_and_cancelled_awaiter_do_not_cancel_execution():
    """Each await uses its own loop waiter and cancellation is local to that await."""
    future = ConcreteFuture("await")

    async def await_and_cancel() -> None:
        started = asyncio.Event()

        async def wait() -> int:
            started.set()
            return await future

        task = asyncio.create_task(wait())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(await_and_cancel())
    assert not future.done()

    two_waiters = ConcreteFuture("two-awaiters")

    async def await_twice() -> tuple[int, int]:
        ready = asyncio.Event()
        count = 0

        async def wait() -> int:
            nonlocal count
            count += 1
            if count == 2:
                ready.set()
            return await two_waiters

        first = asyncio.create_task(wait())
        second = asyncio.create_task(wait())
        await ready.wait()
        assert two_waiters._publish_result(11)
        return await first, await second

    assert asyncio.run(await_twice()) == (11, 11)

    closed_loop = asyncio.new_event_loop()
    started = asyncio.Event()
    closed_future = ConcreteFuture("closed-loop")

    async def abandoned_wait() -> int:
        started.set()
        return await closed_future

    try:
        abandoned = closed_loop.create_task(abandoned_wait())
        closed_loop.run_until_complete(started.wait())
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            closed_loop.run_until_complete(abandoned)
    finally:
        closed_loop.close()
    assert closed_future._publish_result(12)


def test_cancelled_awaiter_releases_its_waiter_before_a_later_failure():
    """A cancelled awaiter cannot retain an unobserved remote failure in its closed loop."""
    future = ConcreteFuture("await-failure")
    loop_errors: list[dict[str, object]] = []

    async def exercise() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: loop_errors.append(context))
        cancelled = asyncio.create_task(future._await_result())
        survivor = asyncio.create_task(future._await_result())
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert future._publish_exception(RuntimeError("remote failure"))
        with pytest.raises(RuntimeError, match="remote failure"):
            await survivor
        await asyncio.sleep(0)

    asyncio.run(exercise())
    assert not loop_errors
