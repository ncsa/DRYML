"""Real subprocess integration conformance for the core Execute adapter."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Executor as CoreExecutor
from dryml.core import Object, ObjectRef, Repo, Serializable, StateRef, function
from dryml.core.execute import CoreExecutionError, CoreOptions, PreparedCoreCall
from dryml.core.store.dir import DirStore
from dryml.execute import Executor, ExecutionOutput, WorkerSetup
from dryml.execute.errors import RemoteExecutionError
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import managed_operation
from dryml.methods import Method


def _core_value(value):
    """Return an ordinary core result after worker-local setup has completed."""
    return value + 1


def _near_boundary_publication(value):
    """Publish durable state while returning ordinary bytes near the core outcome bound."""
    return TrainingValue(11), b"x" * value


def _payload_marker(path):
    """Leave durable proof only if generic Execute delivers the invocation payload."""
    Path(path).write_text("invoked", encoding="ascii")
    return "invoked"


def _generic_value():
    """Return one ordinary value after the setup output flood is drained."""
    return "delivered"


class TrainingValue(Serializable):
    """Small stateful value used to prove real worker publication and refresh."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Persist the deterministic scalar state for cross-process recovery."""
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore the deterministic scalar state published by the worker."""
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class StatelessResult(Object):
    """Definition-only result fixture that requires no state publication."""


class TrainingRoot(Serializable):
    """Retain a nested trainable value for descendant snapshot assertions."""

    def __init__(self, child):
        self.child = child


def _train_and_report(value):
    """Update a transported value and return stateful, stateless, and raw results."""
    value.value += 4
    return TrainingValue(value.value), StatelessResult(), value.object_ref, value.value / 2


def _update_child_and_return_it(root):
    """Return an updated descendant so result and refresh share one root snapshot."""
    root.child.value += 1
    return root.child


def _control_store_path():
    """Return the worker-selected control Store role without opening another Store."""
    from dryml.core.execute import current_context

    control_store = current_context().control_store
    return None if control_store is None else control_store.base_dir


def _worker_cache_snapshot():
    """Inspect worker setup resources without opening another worker Store."""
    from dryml import session
    from dryml.core.execute import current_context

    context = current_context()
    cache = session.current_resource_cache()
    return {
        "cache_active": cache is not None,
        "repo_cached": cache is not None and context.repo in cache.repos,
        "repo_store_cached": cache is not None and context.repo.stores[0] in cache.stores,
    }


def _wait_for_release(marker, release):
    """Publish a post-GO marker and wait until cancellation terminates this worker."""
    marker.write_text("running", encoding="ascii")
    while not release.exists():
        time.sleep(0.01)


def _increment(value):
    """Return a representative raw metric through each supported callable owner."""
    return value + 1


@function
def _signature_owned_increment(value):
    """Exercise the signatures-owned callable boundary in a real subprocess."""
    return value + 2


class _BoundCallable:
    """Retain an ordinary instance capture for bound and callable-owner coverage."""

    def __init__(self, offset):
        self.offset = offset

    def add(self, value):
        """Return one bound-method result from the retained receiver."""
        return value + self.offset

    def __call__(self, value):
        """Return one callable-instance result from the retained receiver."""
        return value + self.offset


class _MethodOwner(Method):
    """Use the methods-owned invocation seam without an outer wrapper."""

    def __call__(self, value):
        """Return one raw method result for owner-specific normalization."""
        return value + 5


class _ManagedOwner(Serializable):
    """Provide one managed owner whose mutation stays isolated in the worker."""

    def __init__(self, calls=0):
        self.calls = calls

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Persist managed call count for ordinary core object reconstruction."""
        Path(dest_dir, "calls").write_text(str(self.calls), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore managed call count before its worker-owned invocation."""
        self.calls = int(Path(src_dir, "calls").read_text(encoding="ascii"))

    @managed_operation()
    def increment(self, value, *, managed):
        """Return one managed raw metric while retaining owner lifecycle control."""
        self.calls += 1
        return value + self.calls


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


def test_real_subprocess_core_publishes_results_refreshes_nested_updates_and_keeps_control_store_separate(tmp_path):
    """Exercise core publication/recovery through a real worker and separate Stores."""
    repo_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo(repo_store)
    value = TrainingValue(2, repo=repo)
    repo.save(value, deep_capture=True)
    root = TrainingRoot(TrainingValue(7, repo=repo), repo=repo)
    repo.save(root, deep_capture=True)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(
            spool_directory=spool, execution_timeout=10,
            invocation_limit_bytes=1_000_000, result_limit_bytes=1_000_000,
        ),
        core=CoreOptions(repo=repo, control_store=control_store, return_objects=False, update_args=True),
    )
    try:
        future = executor.submit(_train_and_report, value)
        stateful, stateless, existing, metric = future.result(timeout=15)
        assert isinstance(stateful, StateRef)
        assert isinstance(stateless, ConcreteDefinition)
        assert isinstance(existing, ObjectRef)
        assert metric == 3.0
        assert value.value == 6
        assert value.last_state_ref is not None
        future.cleanup(timeout=5)

        nested = executor.submit(_update_child_and_return_it, root)
        returned = nested.result(timeout=15)
        assert isinstance(returned, StateRef)
        assert root.child.value == 8
        assert returned == root.last_state_ref.at(next(
            path for path, child in root._runtime_projection.items() if child is root.child
        ))
        nested.cleanup(timeout=5)

        control = executor.submit(_control_store_path)
        assert control.result(timeout=15) == str(control_store.base_dir)
        control.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)


def test_real_subprocess_core_setup_exposes_its_session_resource_cache(tmp_path):
    """A real worker retains setup resources in its active Session cache."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_worker_cache_snapshot)
        assert future.result(timeout=10) == {
            "cache_active": True,
            "repo_cached": True,
            "repo_store_cached": True,
        }
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_real_subprocess_core_retains_publication_evidence_when_setup_budget_drops_its_result(tmp_path):
    """A setup-safe core budget returns exact evidence instead of a generic terminal error."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(
            spool_directory=spool, invocation_limit_bytes=4096, result_limit_bytes=4096,
        ),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_near_boundary_publication, 2_000)
        assert isinstance(future.backend_future.result(timeout=10), bytes)
        with pytest.raises(CoreExecutionError, match="result outcome exceeds configured bound") as raised:
            future.result(timeout=10)
        assert raised.value.evidence.publications
        assert future.snapshot().evidence == raised.value.evidence
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_real_subprocess_core_pre_and_post_go_cancellation_are_not_reported_as_results(tmp_path):
    """Cancel before and after worker authorization, then reconcile every owned worker."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool, termination_timeout=5),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        pre_go = executor.submit(_increment, 1)
        assert pre_go.cancel()
        with pytest.raises(BaseException):
            pre_go.result(timeout=10)
        pre_go.cleanup(timeout=5)

        marker = tmp_path / "running"
        running = executor.submit(_wait_for_release, marker, tmp_path / "release")
        deadline = time.monotonic() + 10
        while not marker.exists():
            if running.done():
                failure = running.exception()
                if failure is not None:
                    raise failure
                pytest.fail("worker returned before publishing its running marker")
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert running.request_cancel()
        with pytest.raises(BaseException):
            running.result(timeout=15)
        assert running.snapshot().backend.cancel_requested
        running.cleanup(timeout=10)
    finally:
        executor.close(cancel=True, timeout=10)


def test_real_subprocess_core_invokes_each_callable_owner_once(tmp_path):
    """Cover plain, closure, bound, callable, signature, Method, and managed owners."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False),
    )
    managed = _ManagedOwner(repo=repo)
    repo.save(managed, deep_capture=True)
    owner = _BoundCallable(3)
    try:
        def closure(offset):
            """Return a closure whose captured value is only reconstructed in the worker."""
            return lambda value: value + offset

        calls = (
            (_increment, (4,), 5),
            (closure(2), (4,), 6),
            (owner.add, (4,), 7),
            (owner, (4,), 7),
            (_signature_owned_increment, (4,), 6),
            (_MethodOwner(), (4,), 9),
            (managed.increment, (4,), 5),
        )
        for fn, args, expected in calls:
            future = executor.submit(fn, *args)
            assert future.result(timeout=15) == expected
            future.cleanup(timeout=5)
        assert managed.calls == 0
    finally:
        executor.close(cancel=True, timeout=10)
