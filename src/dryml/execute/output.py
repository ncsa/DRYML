"""Thread-safe bounded retained and live execution output capture."""

from __future__ import annotations

import codecs
import math
import sys
import time
from collections import deque
from threading import Condition, Event, Lock, Thread
from typing import TextIO

from .models import ExecutionIssue, OutputSnapshot


class ExecutionOutput:
    """Own bounded retained stdout/stderr data for exactly one execution.

    Callers construct this object before submission when a blocking call's output
    must outlive the call.  The executor binds it once after preflight; worker
    drainers then use the private capture hooks to retain bounded byte prefixes
    and optionally mirror chunks to the streams captured at binding time.

    Retained data remains available after worker cleanup.  Live delivery is best
    effort: a blocked or failing sink records an issue without stopping capture.
    """

    def __init__(self) -> None:
        """Create an unbound output owner without capturing or writing streams."""
        self._lock = Lock()
        self._bound_submission_id: str | None = None
        self._output_limit_bytes = 0
        self._final_timeout = 0.0
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._truncated = {"stdout": False, "stderr": False}
        self._next_sequence = {"stdout": 0, "stderr": 0}
        self._final = {"stdout": False, "stderr": False}
        self._sequence_valid = True
        self._outcome_known_at: float | None = None
        self._final_wait_expired = False
        self._live_issue: ExecutionIssue | None = None
        self._live_issue_event = Event()
        self._live_enabled = False
        self._live_limit = 0
        self._live_bytes = 0
        self._live_inflight_bytes = 0
        self._live_queue: deque[tuple[str, bytes | None]] = deque()
        self._live_condition = Condition(Lock())
        self._live_sinks: dict[str, TextIO] = {}
        self._live_thread: Thread | None = None
        self._live_stop_deadline: float | None = None
        self._live_shutdown = False
        self._live_stopped = Event()

    def snapshot(self) -> OutputSnapshot:
        """Return an immutable view of captured output and delivery health.

        Returns:
            An output snapshot with decoded bounded prefixes, truncation flags,
            capture completion, and any live-delivery issue.

        Side Effects:
            None.  Invalid UTF-8 retained bytes decode with replacement.
        """
        with self._lock:
            return OutputSnapshot(
                stdout=bytes(self._stdout).decode("utf-8", errors="replace"),
                stderr=bytes(self._stderr).decode("utf-8", errors="replace"),
                stdout_truncated=self._truncated["stdout"],
                stderr_truncated=self._truncated["stderr"],
                complete=self._capture_complete_locked(),
                live_delivery_issue=self._live_issue,
            )

    def _bind(
        self,
        submission_id: str,
        *,
        output_limit_bytes: int,
        live_output_queue_limit_bytes: int,
        stream_output: bool,
        output_final_timeout: float = 5.0,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        start_live: bool = True,
    ) -> None:
        """Atomically bind this owner to a successful submission preflight.

        This coordinator-only hook captures sink objects at acceptance.  A second
        bind always fails, including from another executor, leaving the winner's
        retained output unchanged.
        """
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError("submission_id must be a nonempty string")
        for name, value in (
            ("output_limit_bytes", output_limit_bytes),
            ("live_output_queue_limit_bytes", live_output_queue_limit_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(stream_output, bool):
            raise TypeError("stream_output must be bool")
        try:
            valid_final_timeout = not isinstance(output_final_timeout, bool) and isinstance(output_final_timeout, (int, float)) and math.isfinite(output_final_timeout) and output_final_timeout > 0
        except OverflowError:
            valid_final_timeout = False
        if not valid_final_timeout:
            raise ValueError("output_final_timeout must be a positive duration")
        thread: Thread | None = None
        with self._lock:
            if self._bound_submission_id is not None:
                raise RuntimeError("ExecutionOutput is already bound to a submission")
            self._bound_submission_id = submission_id
            self._output_limit_bytes = output_limit_bytes
            self._final_timeout = float(output_final_timeout)
            self._live_enabled = stream_output
            if stream_output:
                self._live_limit = live_output_queue_limit_bytes
                self._live_sinks = {"stdout": sys.stdout if stdout is None else stdout, "stderr": sys.stderr if stderr is None else stderr}
                self._live_stopped.clear()
                thread = Thread(target=self._deliver_live, name="dryml-execute-output", daemon=True)
                self._live_thread = thread
        if thread is not None and start_live:
            try:
                thread.start()
            except BaseException:
                self._disable_live("live_delivery_unavailable", "live output delivery could not start")

    def _start_live_delivery(self) -> None:
        """Start a previously accepted live-delivery thread outside lifecycle locks."""
        with self._lock:
            thread = self._live_thread
        if thread is not None and not thread.is_alive():
            try:
                thread.start()
            except BaseException:
                self._disable_live("live_delivery_unavailable", "live output delivery could not start")

    def _capture(self, stream: str, data: bytes, sequence: int) -> None:
        """Retain one protocol-ordered byte chunk and enqueue best-effort live delivery."""
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        if not isinstance(data, bytes):
            raise TypeError("output data must be bytes")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("output sequence must be a nonnegative integer")
        with self._lock:
            self._require_bound_locked()
            self._expire_final_wait_locked()
            if self._capture_complete_locked():
                return
            if sequence != self._next_sequence[stream] or self._final[stream]:
                self._sequence_valid = False
                return
            self._next_sequence[stream] += 1
            retained = self._stdout if stream == "stdout" else self._stderr
            remaining = self._output_limit_bytes - len(retained)
            if remaining > 0:
                retained.extend(data[:remaining])
            if len(data) > remaining:
                self._truncated[stream] = True
            if self._live_enabled:
                self._enqueue_live_locked(stream, data)

    def _finalize(self, stream: str, sequence: int) -> None:
        """Record a validated final fence for one captured stream."""
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("output sequence must be a nonnegative integer")
        with self._lock:
            self._require_bound_locked()
            self._expire_final_wait_locked()
            if self._capture_complete_locked():
                return
            if sequence != self._next_sequence[stream] or self._final[stream]:
                self._sequence_valid = False
                return
            self._final[stream] = True
            if self._live_enabled:
                with self._live_condition:
                    self._live_queue.append((stream, None))
                    self._live_condition.notify()

    def _outcome_known(self) -> None:
        """Start the configured bounded wait for final output fences after outcome publication."""
        with self._lock:
            if self._bound_submission_id is not None and self._outcome_known_at is None:
                self._outcome_known_at = time.monotonic()
                with self._live_condition:
                    self._live_stop_deadline = self._outcome_known_at + self._final_timeout
                    self._live_condition.notify_all()

    def _capture_complete_locked(self) -> bool:
        """Return whether known terminal work has complete validated stream fences."""
        if self._bound_submission_id is None or self._outcome_known_at is None:
            return False
        if self._final_wait_expired:
            return False
        if self._sequence_valid and self._final["stdout"] and self._final["stderr"]:
            return True
        self._expire_final_wait_locked()
        return False

    def _expire_final_wait_locked(self) -> None:
        """Make a missing fence permanently incomplete when its configured wait expires."""
        if self._outcome_known_at is not None and time.monotonic() >= self._outcome_known_at + self._final_timeout:
            self._final_wait_expired = True

    def _require_bound_locked(self) -> None:
        """Reject worker messages that were not associated with an accepted submission."""
        if self._bound_submission_id is None:
            raise RuntimeError("ExecutionOutput is not bound to a submission")

    def _enqueue_live_locked(self, stream: str, data: bytes) -> None:
        """Offer bytes to a bounded independent queue without delaying capture."""
        if not data:
            return
        with self._live_condition:
            if self._live_bytes + self._live_inflight_bytes + len(data) > self._live_limit:
                self._record_live_issue_locked("live_delivery_dropped", "live output delivery fell behind retained capture")
                self._live_enabled = False
                self._live_queue.clear()
                self._live_bytes = 0
                self._live_shutdown = True
                self._live_condition.notify_all()
                return
            self._live_queue.append((stream, data))
            self._live_bytes += len(data)
            self._live_condition.notify()

    def _record_live_issue_locked(self, code: str, message: str) -> None:
        """Keep the first bounded delivery failure while preserving retained capture."""
        if self._live_issue is None:
            self._live_issue = ExecutionIssue(code, message)
            self._live_issue_event.set()

    def _deliver_live(self) -> None:
        """Write queued chunks outside capture locks, then release delivery ownership."""
        decoders = {"stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"), "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace")}
        finalized: set[str] = set()
        try:
            while True:
                with self._live_condition:
                    while not self._live_queue and not self._live_shutdown:
                        if finalized == {"stdout", "stderr"}:
                            return
                        timeout = None if self._live_stop_deadline is None else self._live_stop_deadline - time.monotonic()
                        if timeout is not None and timeout <= 0:
                            self._live_shutdown = True
                            return
                        self._live_condition.wait(timeout)
                    if self._live_shutdown:
                        return
                    stream, data = self._live_queue.popleft()
                    if data is not None:
                        self._live_bytes -= len(data)
                        self._live_inflight_bytes += len(data)
                try:
                    text = decoders[stream].decode(b"" if data is None else data, final=data is None)
                    if text:
                        self._live_sinks[stream].write(text)
                        flush = getattr(self._live_sinks[stream], "flush", None)
                        if flush is not None:
                            flush()
                except BaseException:
                    self._disable_live("live_delivery_failed", "live output sink failed")
                    return
                finally:
                    if data is not None:
                        with self._live_condition:
                            self._live_inflight_bytes -= len(data)
                            self._live_condition.notify_all()
                if data is None:
                    finalized.add(stream)
        finally:
            with self._live_condition:
                self._live_queue.clear()
                self._live_bytes = 0
                self._live_inflight_bytes = 0
                self._live_shutdown = True
                self._live_sinks.clear()
                self._live_thread = None
                self._live_condition.notify_all()
            self._live_stopped.set()

    def _disable_live(self, code: str, message: str) -> None:
        """Disable streaming after a launch, sink, or queue failure without delaying capture."""
        with self._lock:
            self._record_live_issue_locked(code, message)
            self._live_enabled = False
            with self._live_condition:
                self._live_queue.clear()
                self._live_bytes = 0
                self._live_shutdown = True
                self._live_sinks.clear()
                self._live_thread = None
                self._live_condition.notify_all()
        self._live_stopped.set()

    def _shutdown_live_delivery(self) -> None:
        """Request nonblocking private delivery shutdown without joining user sinks."""
        self._disable_live("live_delivery_stopped", "live output delivery was stopped")


__all__ = ["ExecutionOutput"]
