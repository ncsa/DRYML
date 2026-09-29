from __future__ import annotations

from threading import Barrier, Event, Thread

import pytest

from dryml.execute.errors import CleanupError
from dryml.execute.future import ExecutionFuture


def test_simultaneous_terminal_contenders_publish_exactly_one_stable_outcome():
    """A barrier proves result, failure, cancellation, and loss have one linearized winner."""
    future = ExecutionFuture[int]("race")
    barrier = Barrier(4)
    outcomes: list[bool] = []

    def contender(action) -> None:
        barrier.wait()
        outcomes.append(action())

    threads = [
        Thread(target=contender, args=(lambda: future._publish_result(9),)),
        Thread(target=contender, args=(lambda: future._publish_exception(ValueError("failed")),)),
        Thread(target=contender, args=(future.cancel,)),
        Thread(target=contender, args=(lambda: future._publish_uncertain(RuntimeError("lost")),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert outcomes.count(True) == 1
    assert future.done()
    state = future.snapshot().state
    assert state in {"succeeded", "failed", "cancelled", "uncertain"}
    if state == "succeeded":
        assert future.result() == 9
    else:
        with pytest.raises(BaseException):
            future.result()


def test_cleanup_concurrent_join_runs_one_reconciler_and_late_callbacks_progress():
    """A blocked cleanup hook and callback never hold the Future condition lock."""
    future = ExecutionFuture[int]("join")
    entered = Event()
    release = Event()
    callback_started = Event()

    def reconcile(timeout: float) -> None:
        entered.set()
        assert release.wait(timeout)

    future._set_cleanup_reconciler(reconcile)
    future.add_done_callback(lambda value: callback_started.set())
    future._publish_result(2)
    assert callback_started.wait(1)

    failures: list[BaseException] = []

    def clean() -> None:
        try:
            future.cleanup(timeout=0.25)
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=clean)
    second = Thread(target=clean)
    first.start()
    assert entered.wait(1)
    second.start()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    assert not failures
    assert future.result() == 2


def test_cleanup_join_timeout_reports_the_same_future_for_recovery():
    """A joining caller gets a bounded recovery error rather than a false cleanup success."""
    future = ExecutionFuture[int]("join-timeout")
    entered = Event()
    release = Event()

    def reconcile(timeout: float) -> None:
        entered.set()
        release.wait(timeout)

    future._set_cleanup_reconciler(reconcile)
    future._publish_result(1)
    owner = Thread(target=lambda: future.cleanup(timeout=0.5))
    owner.start()
    assert entered.wait(1)
    with pytest.raises(CleanupError) as failure:
        future.cleanup(timeout=0.01)
    assert failure.value.execution is future
    release.set()
    owner.join(timeout=1)
    assert not owner.is_alive()
