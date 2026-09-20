"""Cancellation and deadline proof for owned subprocess groups."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace

import pytest

from dryml.execute import subprocess as subprocess_module
from dryml.execute._protocol import (
    Correlation,
    FrameState,
    FrameType,
    decode_exact_frame,
    encode_frame,
    encode_worker_error,
)
from dryml.execute.errors import ExecutionDeadlineExceeded, ExecutionUncertainError, RemoteExecutionError
from dryml.execute.executor import Executor
from dryml.execute.models import WorkerSetup
from dryml.execute.output import ExecutionOutput
from dryml.execute.subprocess import SubProcessBackend, SubProcessConfig, SubProcessFuture


def _wait() -> None:
    """Keep a worker running until owned-group cancellation reaches it."""
    sleep(30)


class _Reader:
    """Deliver one terminal frame and then close the fake channel."""

    def __init__(self, frame):
        self.frame = frame

    def read(self):
        if self.frame is None:
            raise EOFError
        frame, self.frame = self.frame, None
        return frame


class _Owner:
    """Provide caller-controlled owned-group termination evidence."""

    def __init__(self, confirmed):
        self.confirmed = confirmed
        self.calls = 0

    def reconcile(self, *, deadline, poll_interval):
        self.calls += 1
        return self.confirmed


def _error_frame(*, deadline=False, setup=False, cleanup=()):
    """Build one generated closed worker error terminal."""
    correlation = Correlation("deadline-receiver", 0, 1)
    payload = encode_worker_error(
        None if deadline else "TimeoutError",
        deadline_elapsed=deadline,
        cleanup_types=cleanup,
        setup=setup,
        limit_bytes=4096,
    )
    return decode_exact_frame(
        encode_frame(FrameState.ERROR, FrameType.ERROR, correlation, payload, header_limit=1024),
        header_limit=1024,
        payload_limit=4096,
    )


def _receive_error(monkeypatch, frame, *, confirmed=True, setup=False):
    """Route one generated terminal through the subprocess active receiver."""
    backend = SubProcessBackend(SubProcessConfig(termination_timeout=0.01))
    output = ExecutionOutput()
    future = SubProcessFuture("deadline-receiver", output=output, termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=float("inf"))
    owner = _Owner(confirmed)
    run = subprocess_module._Run(
        future, owner, SimpleNamespace(), Lock(), issued=True,
        terminal_event=Event(),
    )
    monkeypatch.setattr(
        subprocess_module, "SocketFrameReader",
        lambda *args, **kwargs: _Reader(frame),
    )
    call = SimpleNamespace(
        output=output,
        worker_setup=WorkerSetup(factory="tests.execute.test_subprocess_cancellation:unused", data={}) if setup else None,
    )
    descriptor = SimpleNamespace(control_header_limit_bytes=1024)
    conversation = SimpleNamespace(accept_frame=lambda value: value)
    backend._receive_active(call, future, run, descriptor, conversation)
    return backend, future, run, owner


def test_running_cancellation_confirms_only_after_owned_group_exit(tmp_path: Path):
    """A running request uses the owned group rather than presenting request success."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, termination_timeout=1))
    try:
        future = executor.submit(_wait)
        for _ in range(100):
            if future.running():
                break
            sleep(0.01)
        assert future.request_cancel()
        with pytest.raises(CancelledError):
            future.result(timeout=5)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_execution_deadline_terminates_owned_group(tmp_path: Path):
    """An elapsed workload deadline is distinct from an ordinary wait timeout."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, termination_timeout=1))
    try:
        future = executor.submit(_wait, execution_timeout=0.05)
        with pytest.raises(ExecutionDeadlineExceeded):
            future.result(timeout=5)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


@pytest.mark.parametrize("setup", (False, True))
def test_worker_deadline_marker_requires_confirmed_owned_group_termination(monkeypatch, setup):
    """Both terminal variants use lifecycle proof rather than an ordinary error claim."""
    cleanup = ("RuntimeError",) if setup else ()
    _, future, run, owner = _receive_error(
        monkeypatch,
        _error_frame(deadline=True, setup=setup, cleanup=cleanup),
        setup=setup,
    )

    with pytest.raises(ExecutionDeadlineExceeded):
        future.result(timeout=0)
    assert run.deadline_expired and run.cancelling and not run.outcome_claimed
    assert owner.calls == 1
    if setup:
        assert future.snapshot().cleanup_issues[-1].message == "RuntimeError"


def test_worker_deadline_marker_is_uncertain_without_termination_confirmation(monkeypatch):
    """Marker receipt alone cannot prove that the owned process group stopped."""
    _, future, run, owner = _receive_error(
        monkeypatch, _error_frame(deadline=True), confirmed=False,
    )

    with pytest.raises(ExecutionUncertainError):
        future.result(timeout=0)
    assert run.deadline_expired and owner.calls == 1


def test_callable_timeout_error_claims_ordinary_outcome_before_later_deadline(monkeypatch):
    """A genuine callable TimeoutError remains remote even after wall-clock expiry."""
    backend, future, run, owner = _receive_error(monkeypatch, _error_frame())
    backend._deadline_watch(run, 0.0)

    with pytest.raises(RemoteExecutionError) as failure:
        future.result(timeout=0)
    assert failure.value.remote_type == "TimeoutError"
    assert run.outcome_claimed and not run.deadline_expired
    assert owner.calls == 0


def test_cancel_rejects_validated_outcome_claim_before_publication(monkeypatch):
    """Ordinary cancellation cannot overtake a claimed result paused before publish."""
    backend = SubProcessBackend(SubProcessConfig())
    future = SubProcessFuture("claimed-result", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=float("inf"))
    owner = _Owner(True)
    run = subprocess_module._Run(
        future, owner, SimpleNamespace(), Lock(), issued=True,
        terminal_event=Event(),
    )
    started = []

    class UnexpectedThread:
        def __init__(self, *args, **kwargs):
            started.append((args, kwargs))

        def start(self):
            pass

    monkeypatch.setattr(subprocess_module, "Thread", UnexpectedThread)
    assert backend._claim_outcome(run)

    assert not backend._request_cancel(run)
    assert started == []
    assert future._publish_result("preserved")
    assert future.result(timeout=0) == "preserved"
