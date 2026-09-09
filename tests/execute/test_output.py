from __future__ import annotations

from io import StringIO
from threading import Barrier, Event, Thread

import pytest

from dryml.execute.output import ExecutionOutput


def test_output_retains_prefix_truncation_and_immutable_repeated_snapshots():
    """Retained capture stays bounded while drainers continue through truncated data."""
    output = ExecutionOutput()
    output._bind("output-1", output_limit_bytes=4, live_output_queue_limit_bytes=4, stream_output=False)
    output._capture("stdout", b"abcd", 0)
    output._capture("stdout", b"efgh", 1)
    output._capture("stderr", b"bad\xff", 0)
    output._finalize("stdout", 2)
    output._finalize("stderr", 1)
    output._outcome_known()

    snapshot = output.snapshot()
    assert snapshot.stdout == "abcd"
    assert snapshot.stdout_truncated
    assert snapshot.stderr == "bad\ufffd"
    assert snapshot.complete
    with pytest.raises((AttributeError, TypeError)):
        snapshot.stdout = "changed"  # type: ignore[misc]
    assert output.snapshot() == snapshot


def test_output_binding_has_one_cross_executor_winner():
    """Two simultaneous submissions cannot both bind the same caller-owned output."""
    output = ExecutionOutput()
    barrier = Barrier(2)
    results: list[bool] = []

    def bind(identifier: str) -> None:
        barrier.wait()
        try:
            output._bind(identifier, output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=False)
        except RuntimeError:
            results.append(False)
        else:
            results.append(True)

    first = Thread(target=bind, args=("first",))
    second = Thread(target=bind, args=("second",))
    first.start()
    second.start()
    first.join(timeout=1)
    second.join(timeout=1)
    assert results.count(True) == 1
    assert results.count(False) == 1


def test_live_delivery_never_blocks_capture_and_degradation_is_separate():
    """A blocked sink fills only the bounded live queue while retained capture completes."""
    entered = Event()
    release = Event()

    class BlockingSink:
        def write(self, value: str) -> int:
            entered.set()
            assert release.wait(1)
            return len(value)

        def flush(self) -> None:
            return None

    output = ExecutionOutput()
    output._bind(
        "stream",
        output_limit_bytes=128,
        live_output_queue_limit_bytes=2,
        stream_output=True,
        stdout=BlockingSink(),
        stderr=StringIO(),
    )
    output._capture("stdout", b"a", 0)
    assert entered.wait(1)
    for sequence in range(1, 20):
        output._capture("stdout", b"b", sequence)
    output._finalize("stdout", 20)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert output.snapshot().complete
    assert output.snapshot().live_delivery_issue is not None
    release.set()


def test_failing_live_sink_does_not_change_retained_capture():
    """A user sink failure is contained by the delivery worker and remains diagnostic only."""
    delivered = Event()

    class FailingSink:
        def write(self, value: str) -> int:
            delivered.set()
            raise RuntimeError("sink closed")

    output = ExecutionOutput()
    output._bind("failing", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=True, stdout=FailingSink(), stderr=StringIO())
    output._capture("stdout", b"ok", 0)
    output._finalize("stdout", 1)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert delivered.wait(1)
    assert output._live_issue_event.wait(1)
    assert output.snapshot().stdout == "ok"
    assert output.snapshot().complete
    assert output.snapshot().live_delivery_issue is not None


def test_live_sink_failure_disables_delivery_without_repeating_the_failed_sink():
    """A failed sink is detached, so later retained output never retries it."""
    delivered = Event()

    class FailingSink:
        def __init__(self) -> None:
            self.calls = 0

        def write(self, value: str) -> int:
            self.calls += 1
            delivered.set()
            raise RuntimeError("sink closed")

        def flush(self) -> None:
            raise AssertionError("failed write must not be flushed")

    sink = FailingSink()
    output = ExecutionOutput()
    output._bind("one-failure", output_limit_bytes=16, live_output_queue_limit_bytes=16, stream_output=True, stdout=sink, stderr=StringIO())
    output._capture("stdout", b"first", 0)
    assert delivered.wait(1)
    assert output._live_issue_event.wait(1)
    output._capture("stdout", b"second", 1)
    output._finalize("stdout", 2)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert sink.calls == 1
    assert output.snapshot().stdout == "firstsecond"


def test_live_delivery_exits_after_final_fences_or_missing_final_deadline():
    """Delivery workers release queued data and sinks after finals or an idle fence deadline."""
    output = ExecutionOutput()
    output._bind("live-final", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=True, stdout=StringIO(), stderr=StringIO(), output_final_timeout=0.01)
    output._finalize("stdout", 0)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert output._live_stopped.wait(1)
    assert output._live_sinks == {}

    missing = ExecutionOutput()
    missing._bind("live-missing", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=True, stdout=StringIO(), stderr=StringIO(), output_final_timeout=0.01)
    missing._outcome_known()
    assert missing._live_stopped.wait(1)
    assert not missing.snapshot().complete


def test_live_thread_start_failure_degrades_streaming_without_rejecting_binding(monkeypatch):
    """An unavailable delivery thread leaves retained capture usable and records degradation."""
    output = ExecutionOutput()

    def fail_start(thread) -> None:
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr("dryml.execute.output.Thread.start", fail_start)
    output._bind("start-failure", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=True, stdout=StringIO(), stderr=StringIO())
    output._capture("stdout", b"ok", 0)
    output._finalize("stdout", 1)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert output.snapshot().stdout == "ok"
    assert output.snapshot().live_delivery_issue is not None


def test_sequence_fault_and_missing_final_fence_leave_output_incomplete(monkeypatch):
    """Sequence validation and the configured final-fence deadline never rewrite an outcome."""
    now = [10.0]
    monkeypatch.setattr("dryml.execute.output.time.monotonic", lambda: now[0])
    output = ExecutionOutput()
    output._bind("fence", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=False, output_final_timeout=2.5)
    output._capture("stdout", b"a", 0)
    output._capture("stdout", b"b", 2)
    output._outcome_known()
    now[0] += 2.4
    assert not output.snapshot().complete
    now[0] += 0.2
    assert not output.snapshot().complete
    assert output._final_wait_expired


def test_late_invalid_output_cannot_rewrite_an_already_complete_capture():
    """Protocol validation records faults before completion but preserves a settled valid capture."""
    output = ExecutionOutput()
    output._bind("late-frame", output_limit_bytes=8, live_output_queue_limit_bytes=8, stream_output=False)
    output._capture("stdout", b"a", 0)
    output._finalize("stdout", 1)
    output._finalize("stderr", 0)
    output._outcome_known()
    assert output.snapshot().complete
    output._capture("stdout", b"late", 1)
    assert output.snapshot().complete


def test_concurrent_streams_remain_separate_and_flooding_does_not_deadlock():
    """Independent output locks preserve each stream's prefixes under concurrent capture."""
    output = ExecutionOutput()
    output._bind("concurrent", output_limit_bytes=16, live_output_queue_limit_bytes=16, stream_output=False)
    barrier = Barrier(2)

    def capture(stream: str, prefix: bytes) -> None:
        barrier.wait()
        for sequence in range(8):
            output._capture(stream, prefix + bytes([48 + sequence]), sequence)
        output._finalize(stream, 8)

    first = Thread(target=capture, args=("stdout", b"o"))
    second = Thread(target=capture, args=("stderr", b"e"))
    first.start()
    second.start()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    output._outcome_known()
    snapshot = output.snapshot()
    assert snapshot.complete
    assert snapshot.stdout.startswith("o0")
    assert snapshot.stderr.startswith("e0")
