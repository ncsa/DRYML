"""Output flood proof for the subprocess protocol drainers."""

from __future__ import annotations

from pathlib import Path

from dryml.execute.executor import Executor
from dryml.execute.output import ExecutionOutput
from dryml.execute.subprocess import SubProcessConfig


def _flood() -> str:
    """Write larger-than-capture data through both raw worker descriptors."""
    import os

    os.write(1, b"a" * 32_768)
    os.write(2, b"b" * 32_768)
    return "done"


def _buffered_and_raw_output() -> str:
    """Exercise Python buffering, raw descriptors, and a split UTF-8 sequence."""
    import os

    print("buffered stdout")
    print("buffered stderr", file=__import__("sys").stderr)
    os.write(1, b"split \xe2")
    os.write(1, b"\x82\xac\n")
    os.write(2, b"raw stderr\n")
    return "captured"


def test_fd_output_flood_is_drained_after_retained_prefix_truncates(tmp_path: Path):
    """Raw descriptor output cannot block completion when retained capture is bounded."""
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, output_limit_bytes=64, output_frame_limit_bytes=1024, live_output_queue_limit_bytes=1024))
    try:
        future = executor.submit(_flood, output=output)
        assert future.result(timeout=10) == "done"
        snapshot = output.snapshot()
        assert snapshot.stdout_truncated
        assert snapshot.stderr_truncated
        assert snapshot.complete
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_python_buffered_and_raw_descriptor_output_are_flushed_before_restore(tmp_path: Path):
    """Redirect restoration cannot discard buffered Python output or raw writes."""
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    try:
        assert executor.run(_buffered_and_raw_output, output=output) == "captured"
        snapshot = output.snapshot()
        assert "buffered stdout" in snapshot.stdout
        assert "split \u20ac" in snapshot.stdout
        assert "buffered stderr" in snapshot.stderr
        assert "raw stderr" in snapshot.stderr
        assert snapshot.complete
    finally:
        executor.close(cancel=True, timeout=5)
