"""Actual subprocess concurrency conformance without synthetic backend transport."""

from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

from dryml.execute.executor import Executor
from dryml.execute.output import ExecutionOutput
from dryml.execute.subprocess import SubProcessConfig


def _hold(marker: str, release: str, value: str) -> str:
    """Publish a worker marker and retain one independently captured execution."""
    import os
    from pathlib import Path
    from time import sleep

    Path(marker).write_text("running", encoding="ascii")
    os.write(1, f"output-{value}\n".encode("ascii"))
    while not Path(release).exists():
        sleep(0.01)
    return value


def _wait_for(path: Path) -> None:
    """Wait for one real worker marker with a bounded diagnostic deadline."""
    deadline = monotonic() + 10
    while not path.exists():
        if monotonic() >= deadline:
            raise AssertionError(f"worker did not create {path.name}")
        sleep(0.01)


def test_concurrent_submissions_keep_worker_identity_and_output_isolated(tmp_path: Path):
    """Two actual workers retain distinct associations and never mix retained output."""
    spool = tmp_path / "spool"
    spool.mkdir()
    release = tmp_path / "release"
    first_output = ExecutionOutput()
    second_output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=spool))
    try:
        first = executor.submit(_hold, str(tmp_path / "first"), str(release), "first", output=first_output)
        second = executor.submit(_hold, str(tmp_path / "second"), str(release), "second", output=second_output)
        _wait_for(tmp_path / "first")
        _wait_for(tmp_path / "second")
        assert first.pid is not None and second.pid is not None and first.pid != second.pid
        release.write_text("release", encoding="ascii")
        assert first.result(timeout=10) == "first"
        assert second.result(timeout=10) == "second"
        assert first_output.snapshot().stdout == "output-first\n"
        assert second_output.snapshot().stdout == "output-second\n"
        first.cleanup(timeout=5)
        second.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)
