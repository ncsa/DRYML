"""Real subprocess integration conformance for the core Execute adapter."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from dryml.core import Executor as CoreExecutor
from dryml.core import Repo
from dryml.core.execute import CoreOptions, PreparedCoreCall
from dryml.core.store.dir import DirStore
from dryml.execute import Executor, ExecutionOutput, WorkerSetup
from dryml.execute.errors import RemoteExecutionError
from dryml.execute.subprocess import SubProcessConfig


def _core_value(value):
    """Return an ordinary core result after worker-local setup has completed."""
    return value + 1


def _payload_marker(path):
    """Leave durable proof only if generic Execute delivers the invocation payload."""
    Path(path).write_text("invoked", encoding="ascii")
    return "invoked"


def _generic_value():
    """Return one ordinary value after the setup output flood is drained."""
    return "delivered"


@contextmanager
def setup_output_flood(_context, _data):
    """Emit setup and teardown output larger than a pipe before yielding payload access."""
    os.write(1, b"s" * 131_072)
    try:
        yield None
    finally:
        os.write(2, b"t" * 131_072)


def test_core_setup_failure_withholds_payload_from_a_real_subprocess(tmp_path, monkeypatch):
    """Malformed core setup reaches terminal failure before the worker receives call bytes."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    marker = tmp_path / "payload-marker"
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False),
    )

    def malformed_setup(self, runtime=None, *, cache="weak"):
        """Return an importable core factory with invalid data for this accepted call."""
        return WorkerSetup(factory="dryml.core.execute:core_worker_setup", data={"payload": {}})

    monkeypatch.setattr(PreparedCoreCall, "worker_setup", malformed_setup)
    try:
        future = executor.submit(_payload_marker, str(marker))
        with pytest.raises(RemoteExecutionError):
            future.result(timeout=10)
        assert future.snapshot().phase == "backend"
        assert not marker.exists()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_real_setup_and_teardown_output_flood_drains_before_payload_completion(tmp_path):
    """Setup output beyond pipe capacity is bounded, drained, and cannot deadlock delivery."""
    spool = tmp_path / "spool"
    spool.mkdir()
    output = ExecutionOutput()
    executor = Executor(
        SubProcessConfig(
            spool_directory=spool, output_limit_bytes=128, output_frame_limit_bytes=1024,
            live_output_queue_limit_bytes=1024,
        ),
    )
    try:
        future = executor.submit(
            _generic_value,
            worker_setup=WorkerSetup(
                factory="tests.core.test_execute_integration:setup_output_flood", data={},
            ),
            output=output,
        )
        assert future.result(timeout=10) == "delivered"
        snapshot = output.snapshot()
        assert snapshot.stdout_truncated
        assert snapshot.stderr_truncated
        assert snapshot.complete
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_real_core_submission_freezes_default_repo_before_worker_delivery(tmp_path):
    """An accepted core call uses its retained Repo even after the caller changes defaults."""
    first = Repo(DirStore(tmp_path / "first", query_index="none"))
    second = Repo(DirStore(tmp_path / "second", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=first, return_objects=False),
    )
    try:
        future = executor.submit(_core_value, 4)
        executor._core = CoreOptions(repo=second, return_objects=False)
        assert future.result(timeout=10) == 5
        assert future._storage.recovery_repo.to_definition().to_data() == first.to_definition().to_data()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)
