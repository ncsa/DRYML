"""Cancellation and deadline proof for owned subprocess groups."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from time import sleep

import pytest

from dryml.execute.errors import ExecutionDeadlineExceeded
from dryml.execute.executor import Executor
from dryml.execute.subprocess import SubProcessConfig


def _wait() -> None:
    """Keep a worker running until owned-group cancellation reaches it."""
    sleep(30)


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
