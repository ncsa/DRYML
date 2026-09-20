"""Deterministic fake-SDK checks for Ray connection ownership and timeouts."""

from __future__ import annotations

import socket
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import pytest

from dryml.execute import ray as ray_module
from dryml.execute.accounting import Reservation, ResourceAuthority
from dryml.execute.errors import BackendUnavailableError, CleanupError, ExecutionDeadlineExceeded, ExecutionError, ExecutionUncertainError, RemoteExecutionError
from dryml.execute.models import ResourceAmounts
from dryml.execute.output import ExecutionOutput
from dryml.execute.ray import RayBackendConfig, RayFuture
from dryml.execute.subprocess import SubProcessBackend, SubProcessConfig
from dryml.execute._protocol import BootstrapDescriptor, Correlation, FrameState, FrameType, decode_control, decode_exact_frame, encode_control, encode_frame, encode_worker_error


def test_address_normalization_failure_does_not_chain_sdk_details(monkeypatch):
    """New SDK normalization preserves the fixed diagnostic boundary."""
    import sys
    import traceback
    from types import ModuleType

    services = ModuleType("ray._private.services")

    def fail(_address):
        raise ValueError("token=synthetic-private-value")

    services.canonicalize_bootstrap_address = fail
    monkeypatch.setitem(sys.modules, "ray._private.services", services)
    sdk = object.__new__(ray_module._RaySDK)
    with pytest.raises(ExecutionError) as failure:
        sdk.canonical_address("localhost:6379")
    assert failure.value.__cause__ is None
    assert "synthetic-private-value" not in "".join(traceback.format_exception(failure.value))


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

    def canonical_address(self, address):
        """Keep fake endpoints literal unless a test models Ray alias handling."""
        return address

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


def test_borrowed_loopback_aliases_match_rays_canonical_gcs_address(monkeypatch):
    """Accept only Ray-normalized loopback aliases for a borrowed driver."""
    class BorrowedAliasSDK(_FakeSDK):
        def initialized(self):
            return True

        def connection_identity(self):
            return "10.1.0.248:6379", "caller", "cluster"

        def canonical_address(self, address):
            assert address in {"127.0.0.1:6379", "localhost:6379"}
            return "10.1.0.248:6379"

        def disconnect(self):
            raise AssertionError("borrowed driver must not be disconnected")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", BorrowedAliasSDK)
    for address in ("127.0.0.1:6379", "localhost:6379"):
        backend = RayBackendConfig(address=address, namespace="caller").create_backend()
        backend.start()
        assert backend._connection is not None
        assert backend._connection.resolved_address == "10.1.0.248:6379"
        backend.close(cancel=False, timeout=1)
    assert BorrowedAliasSDK.connects == []
    assert BorrowedAliasSDK.disconnects == 0


@pytest.mark.parametrize(
    ("address", "canonical"),
    [
        ("127.0.0.1:6380", "10.1.0.248:6380"),
        ("192.168.2.31:6379", "192.168.2.31:6379"),
        ("10.1.0.249:6379", "10.1.0.249:6379"),
    ],
    ids=("different-port", "nonlocal-endpoint", "different-cluster-endpoint"),
)
def test_borrowed_address_mismatch_rejects_before_creating_a_task(monkeypatch, address, canonical):
    """Reject nonmatching port, nonlocal, and cluster endpoints before task setup."""
    class BorrowedMismatchSDK(_FakeSDK):
        remote_calls = 0

        def initialized(self):
            return True

        def connection_identity(self):
            return "10.1.0.248:6379", "caller", "active-cluster"

        def canonical_address(self, requested):
            assert requested == address
            return canonical

        def remote_bootstrap(self):
            type(self).remote_calls += 1
            raise AssertionError("address mismatch must reject before task setup")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", BorrowedMismatchSDK)
    with pytest.raises(ExecutionError, match="address is incompatible"):
        RayBackendConfig(address=address, namespace="caller").create_backend().start()
    assert BorrowedMismatchSDK.remote_calls == 0
    assert BorrowedMismatchSDK.connects == []
    assert BorrowedMismatchSDK.disconnects == 0


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
    descriptor = backend._descriptor(SimpleNamespace(submission_id=future.submission_id), listener)
    accept_done = Event()

    def accept() -> None:
        try:
            backend._accept_worker(listener, run, descriptor, monotonic() + 10)
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


def test_native_diagnostics_discard_initialization_and_resource_exception_text(monkeypatch):
    """SDK failures retain only their type, never credential or local-path values."""
    class CredentialError(Exception):
        pass

    class InitializationSDK(_FakeSDK):
        def connect(self, address, namespace):
            raise CredentialError("token=dummy-secret path=/private/dryml/credentials")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", InitializationSDK)
    with pytest.raises(ExecutionError) as initialization:
        RayBackendConfig(address="127.0.0.1:6379").create_backend().start()
    assert "CredentialError" in str(initialization.value)
    assert "dummy-secret" not in str(initialization.value)
    assert "/private/dryml" not in str(initialization.value)

    class ObservationSDK:
        def cluster_resources(self):
            raise CredentialError("uri=ray://user:dummy-secret@host/private/dryml")

        def available_resources(self):
            return {"CPU": 1.0}

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=ObservationSDK(), leases=1)
    backend._authority = ResourceAuthority()
    with pytest.raises(BackendUnavailableError) as observation:
        backend.resources(timeout=0.5)
    assert "CredentialError" in str(observation.value)
    assert "dummy-secret" not in str(observation.value)
    assert "/private/dryml" not in str(observation.value)


def test_owned_disconnect_failure_retains_the_final_lease_for_retry(monkeypatch):
    """A failed owned disconnect remains the same backend's cleanup responsibility."""
    class RetrySDK(_FakeSDK):
        attempts = 0

        def connect(self, address, namespace):
            return None

        def disconnect(self):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("token=dummy-secret")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", RetrySDK)
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend.start()
    connection = backend._connection
    assert connection is not None

    with pytest.raises(ExecutionError, match="disconnect failed"):
        backend.close(cancel=False, timeout=1)
    assert connection.leases == 1
    assert ray_module._CONNECTIONS._current is connection

    backend.close(cancel=False, timeout=1)
    assert RetrySDK.attempts == 2
    assert ray_module._CONNECTIONS._current is None


def test_overlapping_owned_disconnects_never_touch_a_replacement_generation(monkeypatch):
    """One disconnect owner serializes overlapping closes before a later acquire."""
    class BlockingDisconnectSDK(_FakeSDK):
        disconnect_entered = Event()
        disconnect_release = Event()
        attempts = 0

        def connect(self, address, namespace):
            return None

        def disconnect(self):
            type(self).attempts += 1
            type(self).disconnect_entered.set()
            assert type(self).disconnect_release.wait(1)

    _reset(monkeypatch)
    BlockingDisconnectSDK.disconnect_entered = Event()
    BlockingDisconnectSDK.disconnect_release = Event()
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", BlockingDisconnectSDK)
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend.start()
    old_connection = backend._connection
    assert old_connection is not None
    first = Thread(target=lambda: backend.close(cancel=False, timeout=1))
    second = Thread(target=lambda: backend.close(cancel=False, timeout=1))
    first.start()
    assert BlockingDisconnectSDK.disconnect_entered.wait(1)
    second.start()
    BlockingDisconnectSDK.disconnect_release.set()
    first.join(1)
    second.join(1)
    assert not first.is_alive() and not second.is_alive()
    assert BlockingDisconnectSDK.attempts == 1

    replacement = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    replacement.start()
    assert replacement._connection is not old_connection
    replacement.close(cancel=False, timeout=1)


def test_late_initializer_disconnect_failure_is_retried_before_a_new_generation(monkeypatch):
    """An unleased late initializer cannot be forgotten or race a replacement driver."""
    class LateRetrySDK(_FakeSDK):
        attempts = 0
        disconnect_attempted = Event()

        def connect(self, address, namespace):
            type(self).connect_entered.set()
            assert type(self).connect_release.wait(1)

        def disconnect(self):
            type(self).attempts += 1
            type(self).disconnect_attempted.set()
            if type(self).attempts == 1:
                raise RuntimeError("token=dummy-secret")

    _reset(monkeypatch)
    LateRetrySDK.disconnect_attempted = Event()
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", LateRetrySDK)
    timed_out = RayBackendConfig(address="127.0.0.1:6379", connect_timeout=0.01).create_backend()
    with pytest.raises(TimeoutError):
        timed_out.start()
    assert LateRetrySDK.connect_entered.wait(1)
    LateRetrySDK.connect_release.set()
    assert LateRetrySDK.disconnect_attempted.wait(1)
    retained = ray_module._CONNECTIONS._current
    assert retained is not None and retained.closing

    replacement = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    replacement.start()
    assert LateRetrySDK.attempts == 2
    assert replacement._connection is not retained
    replacement.close(cancel=False, timeout=1)


def test_failed_initializer_retains_a_disconnect_failure_for_recovery(monkeypatch):
    """A failed post-connect initializer cannot discard its owned driver generation."""
    class FailedInitializationSDK(_FakeSDK):
        attempts = 0
        connected = False

        def connect(self, address, namespace):
            type(self).connected = True
            return None

        def initialized(self):
            return type(self).connected

        def connection_identity(self):
            if type(self).attempts == 0:
                raise RuntimeError("token=dummy-secret")
            return super().connection_identity()

        def disconnect(self):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("token=dummy-secret")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", FailedInitializationSDK)
    with pytest.raises(ExecutionError, match="initialization failed"):
        RayBackendConfig(address="127.0.0.1:6379").create_backend().start()
    retained = ray_module._CONNECTIONS._current
    assert retained is not None and retained.closing and retained.owned

    replacement = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    replacement.start()
    assert FailedInitializationSDK.attempts == 2
    replacement.close(cancel=False, timeout=1)


def test_last_borrowed_lease_drops_stale_driver_identity(monkeypatch):
    """A caller restart is revalidated instead of reusing a cached borrowed driver."""
    class BorrowedSDK(_FakeSDK):
        namespace = "first"
        cluster = "first-cluster"

        def initialized(self):
            return True

        def connection_identity(self):
            return "127.0.0.1:6379", type(self).namespace, type(self).cluster

        def disconnect(self):
            raise AssertionError("borrowed driver must not be disconnected")

    _reset(monkeypatch)
    monkeypatch.setattr(ray_module, "_SDK_FACTORY", BorrowedSDK)
    first = RayBackendConfig(address="127.0.0.1:6379", namespace="first").create_backend()
    first.start()
    first.close(cancel=False, timeout=1)
    assert ray_module._CONNECTIONS._current is None

    BorrowedSDK.namespace = "second"
    BorrowedSDK.cluster = "replacement-cluster"
    second = RayBackendConfig(address="127.0.0.1:6379", namespace="second").create_backend()
    second.start()
    assert second._connection is not None
    assert second._connection.cluster_id == "replacement-cluster"
    second.close(cancel=False, timeout=1)


def test_settled_pre_admission_cancellation_reconciles_without_a_native_run(monkeypatch):
    """A cancelled accepted submission settles its launch marker before cleanup."""
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    future = RayFuture("cancel-before-admission", output=ExecutionOutput(), termination_timeout=1)
    assert future.cancel()
    backend._known.add(future.submission_id)
    backend._launching[future.submission_id] = Event()

    backend._run(SimpleNamespace(submission_id=future.submission_id), future)
    backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert future.submission_id not in backend._known


def test_environment_rejection_settles_known_cleanup_without_a_native_run(monkeypatch):
    """A pre-native admission rejection leaves an accepted submission cleanable."""
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=object())
    future = RayFuture("rejected-before-native", output=ExecutionOutput(), termination_timeout=1)
    backend._known.add(future.submission_id)
    backend._launching[future.submission_id] = Event()

    def reject(call):
        raise ExecutionError("environment rejected")

    monkeypatch.setattr(backend, "_select_environment", reject)
    backend._run(SimpleNamespace(submission_id=future.submission_id), future)
    with pytest.raises(ExecutionError, match="environment rejected"):
        future.result(timeout=0)
    backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert future.submission_id not in backend._known


def test_cancel_startup_failures_roll_back_for_a_retry(monkeypatch):
    """SDK and monitor-start failures do not claim cancellation before it starts."""
    class CancelSDK:
        fail_cancel = True
        calls = 0

        def cancel(self, reference):
            type(self).calls += 1
            if type(self).fail_cancel:
                raise RuntimeError("cancel unavailable")

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=CancelSDK())
    future = RayFuture("retry-cancel", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=monotonic() + 1)
    run = ray_module._Run(future, None, Reservation("retry-cancel", "generation", "0", ResourceAmounts(1.0, None, {}, {})), reference=object())
    future._set_cancel_requester(lambda: backend._request_cancel(run))

    with pytest.raises(ExecutionError, match="cancellation request failed"):
        future.request_cancel()
    assert not run.cancelling and not run.cancellation_starting

    CancelSDK.fail_cancel = False
    original_thread = ray_module.Thread

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(ray_module, "Thread", FailingThread)
    assert future.request_cancel()
    assert run.cancelling and not run.cancellation_starting
    with pytest.raises(ExecutionUncertainError, match="could not be confirmed"):
        future.result(timeout=0)
    assert CancelSDK.calls == 2
    monkeypatch.setattr(ray_module, "Thread", original_thread)


def test_deadline_cancellation_start_failure_is_uncertain_not_a_late_result():
    """A deadline that cannot start exact cancellation publishes a retryable uncertainty."""
    class FailingCancelSDK:
        def cancel(self, reference):
            raise RuntimeError("cancel unavailable")

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=FailingCancelSDK())
    future = RayFuture("deadline-cancel", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=monotonic() + 1)
    run = ray_module._Run(future, None, Reservation("deadline-cancel", "generation", "0", ResourceAmounts(1.0, None, {}, {})), reference=object())

    backend._deadline_watch(run, monotonic())
    with pytest.raises(ExecutionUncertainError, match="deadline cancellation could not start"):
        future.result(timeout=0)
    assert run.deadline_expired and not run.cancelling


def test_ray_deadline_claim_order_is_stable_without_result_deserialization():
    """Validated frames win before expiry, while expiry rejects later frames."""
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    first = RayFuture("result-first", output=ExecutionOutput(), termination_timeout=1)
    first_run = ray_module._Run(first, None, Reservation("result-first", "generation", "0", ResourceAmounts(1.0, None, {}, {})), terminal_event=Event())
    assert backend._claim_outcome(first_run)
    backend._deadline_watch(first_run, monotonic())
    assert first_run.outcome_claimed and first_run.terminal_event.is_set() and not first_run.deadline_expired

    second = RayFuture("deadline-first", output=ExecutionOutput(), termination_timeout=1)
    second_run = ray_module._Run(second, None, Reservation("deadline-first", "generation", "0", ResourceAmounts(1.0, None, {}, {})), terminal_event=Event())
    backend._deadline_watch(second_run, monotonic())
    assert second_run.deadline_expired
    assert not backend._claim_outcome(second_run)


def test_ray_cancel_rejects_validated_outcome_claim_before_publication(monkeypatch):
    """Ordinary native cancellation cannot overtake a claimed paused result."""
    class CancelSDK:
        calls = 0

        def cancel(self, reference):
            type(self).calls += 1

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection(
        "127.0.0.1:6379", None, 256, sdk=CancelSDK(),
    )
    future = RayFuture("claimed-ray-result", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=monotonic() + 1)
    run = ray_module._Run(
        future, None,
        Reservation(future.submission_id, "generation", "0", ResourceAmounts(1.0, None, {}, {})),
        reference=object(), terminal_event=Event(),
    )
    started = []

    class UnexpectedThread:
        def __init__(self, *args, **kwargs):
            started.append((args, kwargs))

        def start(self):
            pass

    monkeypatch.setattr(ray_module, "Thread", UnexpectedThread)
    assert backend._claim_outcome(run)

    assert not backend._request_cancel(run)
    assert CancelSDK.calls == 0
    assert started == []
    assert future._publish_result("preserved")
    assert future.result(timeout=0) == "preserved"
    assert backend._request_cancel(run, allow_terminal=True)
    assert CancelSDK.calls == 1
    assert len(started) == 1


class _TerminalReader:
    """Deliver one generated channel terminal and then close."""

    def __init__(self, frame):
        self.frame = frame

    def read(self):
        if self.frame is None:
            raise EOFError
        frame, self.frame = self.frame, None
        return frame


def _ray_error_frame(*, deadline=False, setup=False, cleanup=()):
    """Build one closed generated Ray worker error terminal."""
    correlation = Correlation("ray-deadline-receiver", 0, 1)
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


def _receive_ray_error(monkeypatch, frame, *, setup=False, backend=None):
    """Route one generated terminal through the dependency-free Ray receiver."""
    backend = backend or RayBackendConfig(address="127.0.0.1:6379").create_backend()
    output = ExecutionOutput()
    future = RayFuture("ray-deadline-receiver", output=output, termination_timeout=1)
    assert future._begin_admission()
    assert future._authorize(deadline=float("inf"))
    run = ray_module._Run(
        future, SimpleNamespace(),
        Reservation(future.submission_id, "generation", "0", ResourceAmounts(1.0, None, {}, {})),
        reference=object(), terminal_event=Event(), issued=True,
    )
    monkeypatch.setattr(ray_module, "SocketFrameReader", lambda *args, **kwargs: _TerminalReader(frame))
    call = SimpleNamespace(output=output, worker_setup=object() if setup else None)
    descriptor = SimpleNamespace(control_header_limit_bytes=1024)
    conversation = SimpleNamespace(accept_frame=lambda value: value)
    backend._receive_active(call, future, run, descriptor, conversation)
    return backend, future, run


@pytest.mark.parametrize("setup", (False, True))
def test_ray_receiver_routes_worker_deadline_marker_to_native_reconciliation(monkeypatch, setup):
    """Both terminal variants defer expiry publication to native stop proof."""
    cleanup = ("RuntimeError",) if setup else ()
    frame = _ray_error_frame(deadline=True, setup=setup, cleanup=cleanup)
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()

    def confirm(run):
        run.cancelling = True
        run.termination_qualified = True
        run.future._expire(ExecutionDeadlineExceeded("execution deadline exceeded"))
        return True

    monkeypatch.setattr(backend, "_request_cancel", confirm)
    _, future, run = _receive_ray_error(monkeypatch, frame, setup=setup, backend=backend)

    with pytest.raises(ExecutionDeadlineExceeded):
        future.result(timeout=0)
    assert run.deadline_expired and run.outcome_validated and not run.outcome_claimed
    if setup:
        assert future.snapshot().cleanup_issues[-1].message == "RuntimeError"


def test_ray_callable_timeout_error_remains_remote_after_later_deadline(monkeypatch):
    """The remote type name cannot forge native deadline reconciliation."""
    backend, future, run = _receive_ray_error(monkeypatch, _ray_error_frame())
    backend._deadline_watch(run, 0.0)

    with pytest.raises(RemoteExecutionError) as failure:
        future.result(timeout=0)
    assert failure.value.remote_type == "TimeoutError"
    assert run.outcome_claimed and not run.deadline_expired


def test_ray_deadline_monitor_wakes_when_a_terminal_event_is_signalled():
    """A long deadline does not retain a completed call's watcher thread."""
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    future = RayFuture("wake-deadline", output=ExecutionOutput(), termination_timeout=1)
    run = ray_module._Run(future, None, Reservation("wake-deadline", "generation", "0", ResourceAmounts(1.0, None, {}, {})), terminal_event=Event())
    watcher = Thread(target=backend._deadline_watch, args=(run, monotonic() + 3600))
    watcher.start()
    run.terminal_event.set()
    watcher.join(1)
    assert not watcher.is_alive()


def test_uncertain_native_submission_never_releases_its_reservation():
    """A submission exception without an ObjectRef stays actionable, not cleanable."""
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=object())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("uncertain-native-submit", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    future._publish_uncertain(ExecutionUncertainError("Ray native submission is uncertain; native task identity is unavailable"))
    reservation = authority.reserve(future.submission_id, ResourceAmounts(1.0, None, {}, {}), generation=connection.generation, attempt="0", total=ResourceAmounts(1.0, None, {}, {}))
    assert reservation is not None
    run = ray_module._Run(future, None, reservation, native_submission_uncertain=True)
    backend._runs[future.submission_id] = run

    with pytest.raises(CleanupError, match="submission is uncertain"):
        backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert authority.snapshot(total=ResourceAmounts(1.0, None, {}, {})).allocated.cpus == 1.0


def test_pre_go_admission_failure_cancels_the_exact_terminal_future_reference():
    """A terminal admission failure still cancels its concrete queued ObjectRef."""
    class CancelSDK:
        cancelled: list[object] = []

        def cancel(self, reference):
            type(self).cancelled.append(reference)

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=CancelSDK())
    future = RayFuture("terminal-admission-failure", output=ExecutionOutput(), termination_timeout=1)
    assert future._begin_admission()
    future._publish_exception(ExecutionError("admission failed"))
    reference = object()
    run = ray_module._Run(future, None, Reservation("terminal-admission-failure", "generation", "0", ResourceAmounts(1.0, None, {}, {})), reference=reference)
    run.native_done.set()

    assert backend._request_cancel(run, allow_terminal=True)
    assert CancelSDK.cancelled == [reference]


def test_native_submission_exception_is_redacted_and_retained_as_uncertain():
    """A fake native submit failure cannot leak text or release an unknown task."""
    class NativeSubmitCredentialError(Exception):
        pass

    class FailingRemote:
        def options(self, **options):
            return self

        def remote(self, descriptor):
            raise NativeSubmitCredentialError("token=dummy-secret path=/private/dryml")

    class SDK:
        def node_affinity(self, node_id, soft):
            return (node_id, soft)

    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=SDK(), node_id="node")
    authority = ResourceAuthority()
    backend._connection = connection
    backend._authority = authority
    backend._remote = FailingRemote()
    future = RayFuture("native-submit-failure", output=ExecutionOutput(), termination_timeout=1)
    reservation = authority.reserve(future.submission_id, ResourceAmounts(1.0, None, {}, {}), generation=connection.generation, attempt="0", total=ResourceAmounts(1.0, None, {}, {}))
    assert reservation is not None
    backend._reserve = lambda call: reservation
    call = SimpleNamespace(
        submission_id=future.submission_id, admission_deadline=monotonic() + 1,
        world=None, environment=None, environment_spec=None,
    )

    backend._run(call, future)
    with pytest.raises(ExecutionUncertainError) as failure:
        future.result(timeout=0)
    assert "dummy-secret" not in str(failure.value)
    assert "/private/dryml" not in str(failure.value)
    with pytest.raises(CleanupError, match="submission is uncertain"):
        backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert authority.snapshot(total=ResourceAmounts(1.0, None, {}, {})).allocated.cpus == 1.0


@pytest.mark.parametrize("backend_kind", ("ray", "subprocess"))
def test_listener_skips_a_stale_hello_before_accepting_the_matching_worker(tmp_path, backend_kind):
    """A wrong-generation connector cannot consume a later call's listener."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    descriptor = BootstrapDescriptor(
        Correlation("fresh-submission", 0, 1), "fresh-token", "127.0.0.1", listener.getsockname()[1],
        1024, 1024, 4096, 4096, 4096, 1024, 1.0,
    )
    accepted: list[object] = []
    if backend_kind == "ray":
        backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
        future = RayFuture("fresh-submission", output=ExecutionOutput(), termination_timeout=1)
        run = ray_module._Run(future, None, Reservation("fresh-submission", "generation", "0", ResourceAmounts(1.0, None, {}, {})))
        waiter = Thread(target=lambda: accepted.extend(backend._accept_worker(listener, run, descriptor, monotonic() + 1)))
    else:
        backend = SubProcessBackend(SubProcessConfig(spool_directory=tmp_path))
        waiter = Thread(target=lambda: accepted.extend(backend._accept_worker(listener, descriptor, monotonic() + 1)))
    waiter.start()
    stale = socket.create_connection(listener.getsockname())
    stale.sendall(encode_control(FrameState.HELLO, Correlation("stale-submission", 0, 1), {"token": "stale-token"}, header_limit=1024))
    stale.close()
    hello_control = {
        "dill": ray_module.dill.__version__, "implementation": ray_module.sys.implementation.name,
        "pid": 123, "protocol": ray_module.WORKER_PROTOCOL_ID,
        "python": list(ray_module.sys.version_info[:2]),
        "token": descriptor.rendezvous_token, "worker_id": f"{backend_kind}:worker",
    }
    if backend_kind == "ray":
        hello_control["native"] = {"node_id": "node", "worker_id": "worker"}
    matching = socket.create_connection(listener.getsockname())
    matching.sendall(encode_control(FrameState.HELLO, descriptor.correlation, hello_control, header_limit=1024))
    waiter.join(1)
    matching.close()
    listener.close()
    assert not waiter.is_alive()
    assert accepted[0] is not None
    assert decode_control(accepted[1], limit_bytes=1024, required_keys=set(hello_control)) == hello_control
    accepted[0].close()


def test_pre_go_correlated_terminal_receipt_qualifies_cleanup():
    """A STOP-confirmed exact worker receipt can release an unissued task attempt."""
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=object())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("pre-go-receipt", output=ExecutionOutput(), termination_timeout=1)
    assert future.cancel()
    reservation = authority.reserve(
        future.submission_id, ResourceAmounts(1.0, None, {}, {}), generation=connection.generation,
        attempt="0", total=ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    assert authority.mark_submitted(future.submission_id, generation=connection.generation, attempt="0")
    run = ray_module._Run(
        future, None, reservation, reference=object(), native_submission_attempted=True,
        worker_id="ray:worker", native_node_id="node", native_task_id="task",
        native_receipt={"marker": ray_module._BOOTSTRAP_MARKER, "worker_id": "worker", "node_id": "node", "task_id": "task"},
    )
    ray_module.RayBackend._qualify_native_terminal(run)
    assert run.native_terminal
    run.native_done.set()
    backend._runs[future.submission_id] = run

    backend.reconcile_cleanup(future.submission_id, timeout=0.5)
    assert authority.snapshot(total=ResourceAmounts(1.0, None, {}, {})).allocated.cpus == 0.0


def test_pre_go_receipt_with_the_wrong_worker_remains_unconfirmed():
    """A terminal receipt remains unusable unless it names the exact HELLO worker."""
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=object())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("wrong-pre-go-receipt", output=ExecutionOutput(), termination_timeout=1)
    assert future.cancel()
    reservation = authority.reserve(
        future.submission_id, ResourceAmounts(1.0, None, {}, {}), generation=connection.generation,
        attempt="0", total=ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    assert authority.mark_submitted(future.submission_id, generation=connection.generation, attempt="0")
    run = ray_module._Run(
        future, None, reservation, reference=object(), native_submission_attempted=True,
        worker_id="ray:worker", native_node_id="node", native_task_id="task",
        native_receipt={"marker": ray_module._BOOTSTRAP_MARKER, "worker_id": "other-worker", "node_id": "node", "task_id": "task"},
    )
    ray_module.RayBackend._qualify_native_terminal(run)
    assert not run.native_terminal
    run.native_done.set()
    backend._runs[future.submission_id] = run

    with pytest.raises(CleanupError, match="release remains unconfirmed"):
        backend.reconcile_cleanup(future.submission_id, timeout=0.5)


def test_pre_go_receipt_with_the_wrong_generation_remains_unconfirmed():
    """A matching worker receipt cannot release a reservation from another generation."""
    connection = ray_module._Connection("127.0.0.1:6379", None, 256, sdk=object())
    backend = RayBackendConfig(address="127.0.0.1:6379").create_backend()
    backend._connection = connection
    authority = ResourceAuthority()
    backend._authority = authority
    future = RayFuture("stale-pre-go-receipt", output=ExecutionOutput(), termination_timeout=1)
    assert future.cancel()
    reservation = authority.reserve(
        future.submission_id, ResourceAmounts(1.0, None, {}, {}), generation="stale-generation",
        attempt="0", total=ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    assert authority.mark_submitted(future.submission_id, generation="stale-generation", attempt="0")
    run = ray_module._Run(
        future, None, reservation, reference=object(), native_submission_attempted=True,
        worker_id="ray:worker", native_node_id="node", native_task_id="task",
        native_receipt={"marker": ray_module._BOOTSTRAP_MARKER, "worker_id": "worker", "node_id": "node", "task_id": "task"},
    )
    ray_module.RayBackend._qualify_native_terminal(run)
    assert run.native_terminal
    run.native_done.set()
    backend._runs[future.submission_id] = run

    with pytest.raises(CleanupError, match="release remains unconfirmed"):
        backend.reconcile_cleanup(future.submission_id, timeout=0.5)
