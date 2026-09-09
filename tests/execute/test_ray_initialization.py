"""Deterministic fake-SDK checks for Ray connection ownership and timeouts."""

from __future__ import annotations

import socket
from threading import Event, Thread
from time import monotonic

import pytest

from dryml.execute import ray as ray_module
from dryml.execute.accounting import ResourceAuthority
from dryml.execute.errors import BackendUnavailableError, CleanupError, ExecutionError
from dryml.execute.models import ResourceAmounts
from dryml.execute.output import ExecutionOutput
from dryml.execute.ray import RayBackendConfig, RayFuture


class _FakeRemote:
    """Record options and descriptor-only native task submissions."""

    def options(self, **options):
        """Retain one options projection for assertions."""
        self.options_value = options
        return self

    def remote(self, descriptor):
        """Reject anything other than the closed bootstrap descriptor bytes."""
        assert isinstance(descriptor, bytes)
        return object()


class _FakeSDK:
    """Minimal fakeable adapter with optional blocked initialization."""

    connect_entered = Event()
    connect_release = Event()
    connects: list[tuple[str, str | None]] = []
    disconnects = 0
    disconnected = Event()

    def __init__(self):
        """Expose only the facade surface used during start."""
        self.ray = type("Ray", (), {"get_runtime_context": staticmethod(lambda: type("Context", (), {"namespace": None})())})()
        self.node_affinity = lambda node_id, soft: (node_id, soft)

    def initialized(self):
        """Model an unborrowed driver."""
        return False

    def connect(self, address, namespace):
        """Optionally block so independent caller budgets can expire."""
        type(self).connects.append((address, namespace))
        type(self).connect_entered.set()
        assert type(self).connect_release.wait(2)

    def disconnect(self):
        """Record only an owned-driver disconnect."""
        type(self).disconnects += 1
        type(self).disconnected.set()

    def connection_identity(self):
        """Provide the connected GCS identity without runtime-context creation."""
        return "127.0.0.1:6379", None, "cluster"

    def nodes(self):
        """Return the one local node required by the backend."""
        return [{"Alive": True, "NodeID": "node", "NodeManagerAddress": "127.0.0.1", "NodeManagerPort": 6379}]

    def remote_bootstrap(self):
        """Supply an inert descriptor-only remote function."""
        return _FakeRemote()


def _reset(monkeypatch):
    """Install an isolated registry and fresh fake state for every test."""
    _FakeSDK.connect_entered = Event()
    _FakeSDK.connect_release = Event()
    _FakeSDK.connects = []
    _FakeSDK.disconnects = 0
    _FakeSDK.disconnected = Event()
    monkeypatch.setattr(ray_module, "_CONNECTIONS", ray_module._ConnectionRegistry())
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", _FakeSDK)


def test_concurrent_start_uses_one_initializer_and_owned_final_disconnect(monkeypatch):
    """Compatible executors share exactly one SDK initialization generation."""
    _reset(monkeypatch)
    first = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    second = RayBackendConfig(address="127.0.0.1:6379").create_backend()

    from threading import Thread

    first_thread = Thread(target=first.start)
    second_thread = Thread(target=second.start)
    first_thread.start()
    assert _FakeSDK.connect_entered.wait(1)
    second_thread.start()
    _FakeSDK.connect_release.set()
    first_thread.join(1)
    second_thread.join(1)

    assert _FakeSDK.connects == [("127.0.0.1:6379", None)]
    first.close(cancel=False, timeout=1)
    assert _FakeSDK.disconnects == 0
    second.close(cancel=False, timeout=1)
    assert _FakeSDK.disconnects == 1


def test_expired_waiter_does_not_start_another_initializer(monkeypatch):
    """A caller timeout detaches permanently while its sole initializer remains owned."""
    _reset(monkeypatch)
    backend = RayBackendConfig(address="127.0.0.1:6379", connect_timeout=0.01).create_backend()

    started = monotonic()
    with pytest.raises(TimeoutError):
        backend.start()
    assert monotonic() - started < 0.5
    assert len(_FakeSDK.connects) == 1
    _FakeSDK.connect_release.set()
    assert _FakeSDK.disconnected.wait(1)
    assert _FakeSDK.disconnects == 1


def test_incompatible_active_endpoint_never_falls_back_to_auto(monkeypatch):
    """A different address is a deterministic conflict, not a second init attempt."""
    _reset(monkeypatch)
    first = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    from threading import Thread

    thread = Thread(target=first.start)
    thread.start()
    assert _FakeSDK.connect_entered.wait(1)
    with pytest.raises(ExecutionError, match="conflicts"):
        RayBackendConfig(address="auto").create_backend().start()
    _FakeSDK.connect_release.set()
    thread.join(1)
    first.close(cancel=False, timeout=1)


def test_incompatible_borrowed_connection_is_never_disconnected(monkeypatch):
    """Borrow conflicts fail without shutting down a caller-owned Ray driver."""
    _reset(monkeypatch)

    class _BorrowedSDK(_FakeSDK):
        def initialized(self):
            return True

        def connection_identity(self):
            return "127.0.0.1:6379", "caller", "cluster"

    monkeypatch.setattr(ray_module, "_SDK_FACTORY", _BorrowedSDK)
    with pytest.raises(ExecutionError, match="namespace"):
        RayBackendConfig(address="127.0.0.1:6379", namespace="execute").create_backend().start()
    assert _BorrowedSDK.disconnects == 0


def test_start_thread_failure_does_not_leave_a_stuck_generation(monkeypatch):
    """A launcher-thread failure wakes the caller and permits a later generation."""
    _reset(monkeypatch)

    class _FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(ray_module, "Thread", _FailingThread)
    with pytest.raises(ExecutionError, match="could not start"):
        RayBackendConfig(address="127.0.0.1:6379").create_backend().start()
    assert ray_module._CONNECTIONS._current is None


def test_resource_observation_uses_one_bounded_monitor_and_refreshes_after_completion(monkeypatch):
    """Timed-out readers join one SDK call, while a completed refresh is never cached."""
    registry = ray_module._ConnectionRegistry()
    monkeypatch.setattr(ray_module, "_CONNECTIONS", registry)

    class BlockingSDK:
        def __init__(self):
            self.entered = Event()
            self.release = Event()
            self.calls = 0
            self.capacity = 2.0
            self.block = True

        def cluster_resources(self):
            self.calls += 1
            if self.block:
                self.entered.set()
                assert self.release.wait(1)
            return {"CPU": self.capacity}

        def available_resources(self):
            return {"CPU": self.capacity}

    sdk = BlockingSDK()
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=sdk, leases=1)
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    backend._authority = ResourceAuthority()

    started = monotonic()
    with pytest.raises(TimeoutError, match="resource observation"):
        backend.resources(timeout=0.02)
    assert monotonic() - started < 0.5
    assert sdk.entered.wait(1)
    assert sdk.calls == 1

    joined: list[object] = []
    reader = Thread(target=lambda: joined.append(backend.resources(timeout=0.5)))
    reader.start()
    sdk.release.set()
    reader.join(1)
    assert not reader.is_alive()
    assert sdk.calls == 1
    assert joined[0].total.cpus == 2.0

    sdk.capacity = 3.0
    sdk.block = False
    assert backend.resources(timeout=0.5).total.cpus == 3.0
    assert sdk.calls == 2


def test_close_retains_an_inflight_resource_generation_until_the_monitor_returns(monkeypatch):
    """A timed-out query cannot let close disconnect its still-running SDK call."""
    registry = ray_module._ConnectionRegistry()
    monkeypatch.setattr(ray_module, "_CONNECTIONS", registry)

    class BlockingSDK:
        def __init__(self):
            self.entered = Event()
            self.release = Event()

        def cluster_resources(self):
            self.entered.set()
            assert self.release.wait(1)
            return {"CPU": 2.0}

        def available_resources(self):
            return {"CPU": 2.0}

    sdk = BlockingSDK()
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=sdk, leases=1)
    registry._current = connection
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    backend._authority = ResourceAuthority()

    with pytest.raises(TimeoutError):
        backend.resources(timeout=0.02)
    assert sdk.entered.wait(1)
    with pytest.raises(CleanupError, match="observation"):
        backend.close(cancel=False, timeout=0.02)
    assert backend._closed
    assert backend._connection is connection
    assert not connection.closing

    sdk.release.set()
    deadline = monotonic() + 1
    while connection.observations and monotonic() < deadline:
        Event().wait(0.005)
    assert not connection.observations
    backend.close(cancel=False, timeout=0.5)
    assert backend._connection is None


def test_native_failure_before_hello_wakes_only_its_listener_and_retains_its_reference(monkeypatch):
    """An exact task failure releases the pending accept without inventing worker identity."""
    class RayTaskError(Exception):
        pass

    class FailingSDK:
        def get(self, reference, *, timeout):
            raise RayTaskError("bootstrap failed")

    registry = ray_module._ConnectionRegistry()
    monkeypatch.setattr(ray_module, "_CONNECTIONS", registry)
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=FailingSDK())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("failed-bootstrap", output=ExecutionOutput(), termination_timeout=1)
    reference = object()
    future._set_native(reference, "127.0.0.1:6379", "node")
    reservation = authority.reserve(
        future.submission_id, ResourceAmounts(1.0, None, {}, {}),
        generation=connection.generation, attempt="0", total=ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    assert authority.mark_submitted(future.submission_id, generation=connection.generation, attempt="0")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    run = ray_module._Run(future, None, reservation, listener=listener, reference=reference, native_submission_attempted=True)
    backend._runs[future.submission_id] = run
    accept_done = Event()

    def accept() -> None:
        try:
            backend._accept_worker(listener, run, monotonic() + 10)
        except ExecutionError:
            pass
        finally:
            accept_done.set()

    waiter = Thread(target=accept)
    waiter.start()
    backend._watch_native(run)
    assert accept_done.wait(1)
    assert run.terminal_before_hello
    assert not run.issued
    assert future.object_ref is reference
    backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert future.object_ref is None
    waiter.join(1)


def test_ambiguous_native_observation_does_not_release_a_submitted_ray_charge(monkeypatch):
    """Connection-level SDK errors retain the submitted charge without inventing terminal proof."""
    class ConnectionLostError(Exception):
        pass

    class LostSDK:
        def get(self, reference, *, timeout):
            raise ConnectionLostError("driver disconnected")

    monkeypatch.setattr(ray_module, "_CONNECTIONS", ray_module._ConnectionRegistry())
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=LostSDK())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("ambiguous-bootstrap", output=ExecutionOutput(), termination_timeout=1)
    reservation = authority.reserve(
        future.submission_id, ResourceAmounts(1.0, None, {}, {}),
        generation=connection.generation, attempt="0", total=ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    assert authority.mark_submitted(future.submission_id, generation=connection.generation, attempt="0")
    run = ray_module._Run(future, None, reservation, reference=object(), native_submission_attempted=True)
    backend._runs[future.submission_id] = run

    backend._watch_native(run)
    assert not run.terminal_before_hello
    with pytest.raises(CleanupError, match="release remains unconfirmed"):
        backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert authority.snapshot(total=ResourceAmounts(1.0, None, {}, {})).allocations[0].state == "unconfirmed"
