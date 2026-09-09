from __future__ import annotations

from dataclasses import replace
import concurrent.futures
import errno
import os
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import socket
import threading
from pathlib import Path

import pytest

from dryml.execute._spooling import PayloadSpooler, SpoolBudget, deserialize_call, deserialize_result, serialize_call, serialize_result, validate_payload, validate_result
from dryml.execute.future import ExecutionFuture
from dryml.execute.config import BackendConfig
from dryml.execute.errors import CleanupError, ExecutionError


class FakeConfig(BackendConfig):
    """Small inert configuration for deterministic spool tests."""

    def create_backend(self):
        raise AssertionError


def test_payload_spool_is_complete_descriptor_and_preserves_caller_parent(tmp_path: Path):
    """A preflight snapshot owns only its private child and exposes no live graph."""
    parent = tmp_path / "parent"
    parent.mkdir()
    sibling = parent / "caller.txt"
    sibling.write_text("keep")
    config = FakeConfig(spool_directory=parent, spool_limit_bytes=10_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=4)
    lease = SpoolBudget.acquire(config)
    try:
        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda x: x + 1, (3,), {})
        assert payload.path.parent.parent == parent
        assert payload.size_bytes <= config.invocation_limit_bytes
        assert set(payload.__dataclass_fields__) == {"path", "size_bytes", "sha256", "serializer", "serializer_version", "python_implementation", "python_version", "pickle_protocol"}
        encoded = payload.path.read_bytes()
        validate_payload(payload, encoded, limit_bytes=config.invocation_limit_bytes)
        with pytest.raises(ValueError, match="digest"):
            validate_payload(replace(payload, sha256="0" * 64), encoded, limit_bytes=config.invocation_limit_bytes)
        with pytest.raises(TypeError, match="serializer"):
            validate_payload(replace(payload, serializer="pickle"), encoded, limit_bytes=config.invocation_limit_bytes)
        spooler.dispose(payload, reservation)
        assert sibling.read_text() == "keep"
    finally:
        lease.release()


def test_failed_unaccepted_snapshot_cleans_owned_child_and_keeps_no_output_owner(tmp_path: Path):
    """Serialization overflow rejects before acceptance and removes only owned storage."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_100, invocation_limit_bytes=100, result_limit_bytes=2_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    try:
        with pytest.raises(ValueError, match="invocation"):
            PayloadSpooler(config, lease).snapshot(lambda: "x" * 1000, (), {})
        assert not list(tmp_path.glob("dryml-execute-*"))
    finally:
        lease.release()


def test_short_write_cleans_preflight_and_failed_disposal_retains_its_charge(tmp_path: Path, monkeypatch):
    """Storage failures never refund capacity before the owned child is removed."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=10_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=4)
    lease = SpoolBudget.acquire(config)
    original_open = Path.open

    class ShortWriter:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, data):
            return len(data) - 1

        def flush(self):
            pass

        def fileno(self):
            return 0

    def short_open(path, *args, **kwargs):
        if path.name == "invocation.dill":
            return ShortWriter()
        return original_open(path, *args, **kwargs)

    try:
        monkeypatch.setattr(Path, "open", short_open)
        with pytest.raises(OSError, match="short write"):
            PayloadSpooler(config, lease).snapshot(lambda: 1, (), {})
        monkeypatch.setattr(Path, "open", original_open)

        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda: 1, (), {})
        original_rmdir = Path.rmdir
        monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")) if path == payload.path.parent else original_rmdir(path))
        with pytest.raises(CleanupError):
            spooler.dispose(payload, reservation)
        assert SpoolBudget.snapshot().reserved_bytes == payload.size_bytes + config.result_limit_bytes
        monkeypatch.setattr(Path, "rmdir", original_rmdir)
        spooler.dispose(payload, reservation)
    finally:
        lease.release()


def test_disk_full_and_permission_preflight_failures_release_unpublished_reservations(tmp_path: Path, monkeypatch):
    """Unpublished storage failures leave no child or process-wide quota charge."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=10_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=4)
    lease = SpoolBudget.acquire(config)
    original_open = Path.open
    original_mkdir = Path.mkdir
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", lambda path, *args, **kwargs: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")) if path.name == "invocation.dill" else original_open(path, *args, **kwargs))
            with pytest.raises(OSError) as full:
                PayloadSpooler(config, lease).snapshot(lambda: 1, (), {})
            assert full.value.errno == errno.ENOSPC
        assert SpoolBudget.snapshot().reserved_bytes == 0

        with monkeypatch.context() as patch:
            patch.setattr(Path, "mkdir", lambda path, *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")) if path.name.startswith("dryml-execute-") else original_mkdir(path, *args, **kwargs))
            with pytest.raises(PermissionError):
                PayloadSpooler(config, lease).snapshot(lambda: 1, (), {})
        assert SpoolBudget.snapshot().reserved_bytes == 0
    finally:
        lease.release()


def test_result_is_bounded_received_in_reserved_slot_and_disposed_with_payload(tmp_path: Path):
    """Result bytes use the reserved second file and retain its capacity until cleanup."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    try:
        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda: 1, (), {})
        spooler.accept(payload, reservation)
        data = serialize_result({"answer": 42}, limit_bytes=config.result_limit_bytes)
        result = spooler.receive_result(reservation, data)
        assert result.path.name == "result.dill"
        validate_result(result, result.path.read_bytes(), limit_bytes=config.result_limit_bytes)
        assert deserialize_result(result.path.read_bytes(), limit_bytes=config.result_limit_bytes) == {"answer": 42}
        with pytest.raises(ExecutionError, match="already has a result"):
            spooler.receive_result(reservation, data)
        spooler.dispose(payload, reservation)
    finally:
        lease.release()


def test_serializer_rejects_resource_subclasses_attrs_defaults_and_closures():
    """Pickler-time checking reaches graph values hidden outside top-level containers."""
    class LockBox:
        def __init__(self, value):
            self.value = value

    class ExecutorSubclass(concurrent.futures.ThreadPoolExecutor):
        pass

    resources = [LockBox(threading.RLock()), socket.socket(), ExecutorSubclass(max_workers=1)]
    try:
        for resource in resources:
            with pytest.raises(TypeError, match="live resource"):
                serialize_call(lambda value: value, (resource,), {}, limit_bytes=1_000_000)
        lock = threading.RLock()
        def defaulted(value=lock):
            return value
        with pytest.raises(TypeError, match="live resource"):
            serialize_call(defaulted, (), {}, limit_bytes=1_000_000)
        captured = socket.socket()
        try:
            def closed():
                return captured
            with pytest.raises(TypeError, match="live resource"):
                serialize_call(closed, (), {}, limit_bytes=1_000_000)
        finally:
            captured.close()
    finally:
        resources[1].close()
        resources[2].shutdown()


def test_cleanup_retry_preserves_unrelated_child_contents_and_payload_matching(tmp_path: Path, monkeypatch):
    """Only known files are removed; failed cleanup stays recoverable and charged."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    try:
        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda: 1, (), {})
        unrelated = payload.path.parent / "caller.txt"
        unrelated.write_text("keep")
        with pytest.raises(ExecutionError, match="does not match"):
            spooler.dispose(replace(payload, sha256="0" * 64), reservation)
        with pytest.raises(CleanupError) as failure:
            spooler.dispose(payload, reservation)
        assert unrelated.read_text() == "keep"
        assert SpoolBudget.snapshot().reserved_bytes > 0
        unrelated.unlink()
        spooler.reconcile_cleanup()
        spooler.dispose(payload, reservation)
    finally:
        lease.release()


def test_preflight_cleanup_failure_exposes_recoverable_owner_and_releases_slot(tmp_path: Path, monkeypatch):
    """A no-Future rejection retains a cleanup owner but no active producer slot."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_100, invocation_limit_bytes=100, result_limit_bytes=2_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    original_rmdir = Path.rmdir
    try:
        monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")) if path.name.startswith("dryml-execute-") else original_rmdir(path))
        spooler = PayloadSpooler(config, lease)
        with pytest.raises(CleanupError) as failure:
            spooler.snapshot(lambda: "x" * 10_000, (), {})
        assert SpoolBudget.snapshot().active_preflights == 0
        monkeypatch.setattr(Path, "rmdir", original_rmdir)
        spooler.reconcile_cleanup()
        assert SpoolBudget.snapshot().reserved_bytes == 0
    finally:
        lease.release()


def test_interrupted_snapshot_after_child_creation_retains_targeted_cleanup(tmp_path: Path, monkeypatch):
    """An interrupt after mkdir preserves the exact spool identity for recovery."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    original_rmdir = Path.rmdir
    try:
        spooler = PayloadSpooler(config, lease)
        monkeypatch.setattr(spooler, "_write_new", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
        monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")) if path.name.startswith("dryml-execute-") else original_rmdir(path))
        with pytest.raises(CleanupError):
            spooler.snapshot(lambda: 1, (), {})
        assert SpoolBudget.snapshot().active_preflights == 0
        assert SpoolBudget.snapshot().reserved_bytes > 0
        monkeypatch.setattr(Path, "rmdir", original_rmdir)
        spooler.reconcile_cleanup()
        assert SpoolBudget.snapshot().reserved_bytes == 0
    finally:
        lease.release()


def test_snapshot_holds_preflight_until_accepted_and_reconciliation_preserves_active_spools(tmp_path: Path):
    """Only failed/rejected spools are cleanup candidates; accepted work stays owned."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=4_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=4)
    lease = SpoolBudget.acquire(config)
    try:
        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda: 1, (), {})
        assert SpoolBudget.snapshot().active_preflights == 1
        spooler.reconcile_cleanup()
        assert payload.path.exists()
        assert SpoolBudget.snapshot().reserved_bytes == payload.size_bytes + config.result_limit_bytes
        spooler.accept(payload, reservation)
        assert SpoolBudget.snapshot().active_preflights == 0
        spooler.reconcile_cleanup()
        assert payload.path.exists()
        spooler.dispose(payload, reservation)
        spooler.dispose(payload, reservation)
    finally:
        lease.release()


def test_result_writer_and_disposal_share_only_the_operation_lock(tmp_path: Path):
    """Disposal cannot release a reservation while its result writer still owns it."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_000, invocation_limit_bytes=1_000, result_limit_bytes=1_000, spool_file_limit=2)
    lease = SpoolBudget.acquire(config)
    try:
        spooler = PayloadSpooler(config, lease)
        payload, reservation = spooler.snapshot(lambda: 1, (), {})
        spooler.accept(payload, reservation)
        entered = threading.Event()
        unblock = threading.Event()
        original_write = spooler._write_new

        def delayed_write(path, data, label):
            if label == "result":
                entered.set()
                assert unblock.wait(timeout=2)
            return original_write(path, data, label)

        spooler._write_new = delayed_write
        writer = threading.Thread(target=spooler.receive_result, args=(reservation, serialize_result(1, limit_bytes=1_000)))
        writer.start()
        assert entered.wait(timeout=2)
        disposed = threading.Event()
        disposer = threading.Thread(target=lambda: (spooler.dispose(payload, reservation), disposed.set()))
        disposer.start()
        assert not disposed.wait(timeout=0.1)
        unblock.set()
        writer.join(timeout=2)
        disposer.join(timeout=2)
        assert not writer.is_alive()
        assert not disposer.is_alive()
        assert SpoolBudget.snapshot().reserved_bytes == 0
    finally:
        lease.release()


def test_snapshot_copies_aliases_defaults_and_closures_before_caller_mutation():
    """Dill snapshots the supported graph rather than retaining coordinator globals."""
    shared = ["before"]

    def captured(default=shared):
        return default

    graph = [shared, shared]
    payload = serialize_call(captured, (graph,), {"again": shared}, limit_bytes=1_000_000)
    shared[0] = "after"
    fn, args, kwargs = deserialize_call(payload, limit_bytes=1_000_000)
    assert fn() == ["before"]
    assert args[0][0] is args[0][1]
    assert args[0][0] is kwargs["again"]
    assert args[0][0] == ["before"]


def test_concurrent_disposal_releases_one_reservation(tmp_path, monkeypatch):
    """Overlapping cleanup callers may share an owner but release it only once."""
    config = FakeConfig(spool_directory=tmp_path, spool_limit_bytes=2_000,
                        invocation_limit_bytes=1_000, result_limit_bytes=1_000)
    lease = SpoolBudget.acquire(config)
    spooler = PayloadSpooler(config, lease)
    payload, reservation = spooler.snapshot(lambda: 1, (), {})
    barrier = threading.Barrier(2)
    lookup = spooler._owned_for
    errors = []

    def overlapping_lookup(reservation):
        owned = lookup(reservation)
        barrier.wait(timeout=2)
        return owned

    def dispose():
        try:
            spooler.dispose(payload, reservation)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(spooler, "_owned_for", overlapping_lookup)
    threads = [threading.Thread(target=dispose) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        assert all(not thread.is_alive() for thread in threads)
        assert not errors
        assert SpoolBudget.snapshot().reserved_bytes == 0
    finally:
        lease.release()


def test_serializer_rejects_execute_and_connection_subclasses_and_bound_builtins():
    """Known live-resource inheritance cannot bypass preflight root checks."""
    class FutureSubclass(ExecutionFuture):
        def done(self):
            return False

        def result(self, timeout=None):
            raise AssertionError

        def exception(self, timeout=None):
            raise AssertionError

        def cancel(self):
            return False

    class ConnectionSubclass(Connection):
        pass

    resource = FutureSubclass("future")
    with pytest.raises(TypeError, match="live resource"):
        serialize_call(lambda value: value, (resource,), {}, limit_bytes=1_000_000)
    with pytest.raises(TypeError, match="bound builtin"):
        serialize_call([].append, (), {}, limit_bytes=1_000_000)
    parent, child = Pipe()
    connection = ConnectionSubclass(os.dup(parent.fileno()))
    try:
        with pytest.raises(TypeError, match="Connection"):
            serialize_call(lambda value: value, (connection,), {}, limit_bytes=1_000_000)
    finally:
        connection.close()
        parent.close()
        child.close()
    fn, args, kwargs = deserialize_call(serialize_call(len, ([1, 2],), {}, limit_bytes=1_000_000), limit_bytes=1_000_000)
    assert fn(*args, **kwargs) == 2
