"""Existing-deployment Ray backend for the additive Execute API.

The module is dependency-light at import time.  Ray is imported only by
``RayBackend.start`` through a narrow SDK facade, so installing or selecting the
optional backend never changes the local process until an executor starts it.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, Event, Lock, RLock, Thread
from typing import Any, TypeVar

import dill

from dryml.environments import EnvironmentRecord, EnvironmentRequirement
from dryml.environments.specs import CondaEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes

from ._protocol import (
    BootstrapDescriptor,
    Correlation,
    FrameError,
    FrameState,
    FrameType,
    OwnerEnvelopeType,
    ProtocolConversation,
    SocketFrameReader,
    WORKER_PROTOCOL_ID,
    decode_control,
    decode_setup_terminal,
    encode_bootstrap_descriptor,
    encode_control,
    encode_frame_parts,
    encode_owner_envelope,
)
from .accounting import RESOURCE_AUTHORITIES, Reservation, ResourceAuthority
from .admission import _admit_observed_logical, admit
from .backend import Backend
from .config import BackendConfig, _positive_duration
from .discovery import discover_candidates, probe_candidate
from .errors import AdmissionError, BackendUnavailableError, CleanupError, ExecutionDeadlineExceeded, ExecutionError, ExecutionUncertainError, RemoteExecutionError
from .future import ExecutionFuture
from .models import DiscoverySnapshot, EnvironmentCandidate, ExecutionIssue, FeasiblePlan, ResourceAmounts, ResourceSnapshot, SubmittedCall
from .output import ExecutionOutput
from dryml.worlds import ProcessSpec, ResourceSpec, RoleSpec, WorldSpec


T = TypeVar("T")
_RAY_VERSION = "2.56.0"
_BOOTSTRAP_MARKER = "dryml.execute.ray.bootstrap.v1"
_HELLO_FIELDS = {"dill", "implementation", "native", "pid", "protocol", "python", "token", "worker_id"}


@dataclass(frozen=True, kw_only=True)
class RayBackendConfig(BackendConfig):
    """Describe one inert connection to an existing same-host Ray deployment.

    Args:
        address: ``"auto"`` or an existing textual ``host:port`` endpoint.
            Explicit endpoints never fall back to discovery.
        namespace: Optional existing Ray namespace.  It must match a borrowed
            caller connection.
        connect_timeout: Positive caller wait bound for SDK initialization.

    Returns:
        :meth:`create_backend` creates an unstarted :class:`RayBackend`.

    Raises:
        TypeError: If endpoint, namespace, or timeout types are invalid.
        ValueError: If an endpoint is a URI/provisioning form or timeout is not
            finite and positive.

    Side Effects:
        Construction validates values only; it neither imports Ray nor contacts a
        deployment.
    """

    address: str = "auto"
    namespace: str | None = None
    connect_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Validate closed existing-server connection selectors without I/O."""
        super().__post_init__()
        if not isinstance(self.address, str):
            raise TypeError("address must be text")
        if self.address != "auto":
            host, separator, port = self.address.rpartition(":")
            if not host or not separator or not port.isdecimal() or "://" in self.address or any(character.isspace() for character in self.address):
                raise ValueError("address must be 'auto' or an existing host:port endpoint")
            if not 0 < int(port) <= 65535:
                raise ValueError("address port must be between 1 and 65535")
        if self.namespace is not None and (not isinstance(self.namespace, str) or not self.namespace):
            raise TypeError("namespace must be non-empty text or None")
        _positive_duration("connect_timeout", self.connect_timeout)
        if {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "XLA_VISIBLE_DEVICES"} & self.env_vars.keys():
            raise ValueError("Ray env_vars cannot override native device visibility")

    def create_backend(self) -> "RayBackend":
        """Create an inert backend without importing or initializing Ray."""
        return RayBackend(self)


class _RaySDK:
    """Narrow, replaceable Ray 2.56.0 SDK adapter used by the connection registry."""

    def __init__(self) -> None:
        """Import exactly the pinned optional dependency on first backend start."""
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        except ImportError as exc:
            raise ExecutionError("Ray backend requires ray[default]==2.56.0") from exc
        if ray.__version__ != _RAY_VERSION:
            raise ExecutionError(f"Ray backend requires Ray {_RAY_VERSION}")
        self.ray = ray
        self.node_affinity = NodeAffinitySchedulingStrategy

    def initialized(self) -> bool:
        """Return Ray driver's observable connection state."""
        return bool(self.ray.is_initialized())

    def connect(self, address: str, namespace: str | None) -> Any:
        """Attach to an existing server without calling a cluster-creation mode."""
        return self.ray.init(address=address, namespace=namespace, ignore_reinit_error=False)

    def disconnect(self) -> None:
        """Disconnect only this process's Execute-owned Ray driver connection."""
        self.ray.shutdown()

    def nodes(self) -> list[Mapping[str, Any]]:
        """Return public Ray node evidence for same-host validation."""
        return list(self.ray.nodes())

    def cluster_resources(self) -> Mapping[str, float]:
        """Return public cluster capacity observations."""
        return self.ray.cluster_resources()

    def available_resources(self) -> Mapping[str, float]:
        """Return public net scheduler availability observations."""
        return self.ray.available_resources()

    def remote_bootstrap(self) -> Any:
        """Lazily define the fixed one-shot native task with retries disabled."""
        from ._worker import run_ray_bootstrap

        return self.ray.remote(max_calls=1, max_retries=0, retry_exceptions=False)(run_ray_bootstrap)

    def cancel(self, reference: Any) -> None:
        """Request cancellation only for one exact bootstrap task reference."""
        self.ray.cancel(reference, force=True, recursive=False)

    def get(self, reference: Any, *, timeout: float) -> Any:
        """Wait for one bootstrap marker for a caller-owned finite interval."""
        return self.ray.get(reference, timeout=timeout)

    def connection_identity(self) -> tuple[str, str | None, str]:
        """Return the initialized driver's actual GCS endpoint, namespace, and cluster ID.

        This reads driver connection state only after initialization and never
        creates a runtime context or schedules work.
        """
        worker = self.ray._private.worker.global_worker
        node = self.ray._private.worker._global_node
        address = node.gcs_address
        cluster_id = node.cluster_id.hex()
        namespace = worker.namespace
        if not isinstance(address, str) or not address or not isinstance(cluster_id, str) or not cluster_id:
            raise ExecutionError("Ray driver did not expose a connected GCS identity")
        if namespace is not None and not isinstance(namespace, str):
            raise ExecutionError("Ray driver did not expose a valid namespace")
        return address, namespace, cluster_id

    def canonical_address(self, address: str) -> str:
        """Normalize one explicit address using Ray's pinned connection rules.

        Args:
            address: Validated explicit ``host:port`` endpoint selected by the
                caller.

        Returns:
            Ray's canonical existing-cluster endpoint. Loopback aliases become
            the Ray-visible address of this host.

        Raises:
            ExecutionError: If Ray cannot produce a concrete endpoint.

        Side Effects:
            May inspect local network configuration as Ray does for an explicit
            ``ray.init(address=...)`` connection. It neither initializes a
            driver nor contacts or changes a deployment.
        """
        try:
            from ray._private.services import canonicalize_bootstrap_address

            canonical = canonicalize_bootstrap_address(address)
        except BaseException:
            raise ExecutionError("Ray could not normalize the requested address") from None
        if not isinstance(canonical, str) or not canonical:
            raise ExecutionError("Ray did not normalize the requested address")
        return canonical


_SDK_FACTORY = _RaySDK


@dataclass(slots=True)
class _Connection:
    """Retain one process-global Ray SDK generation and its executor leases."""

    address: str
    namespace: str | None
    diagnostic_limit: int
    event: Event = field(default_factory=Event)
    sdk: _RaySDK | None = None
    error: BaseException | None = None
    owned: bool = False
    resolved_address: str | None = None
    cluster_id: str | None = None
    node_id: str | None = None
    generation: str = field(default_factory=lambda: secrets.token_hex(16))
    leases: int = 0
    waiters: int = 0
    closing: bool = False
    disconnecting: bool = False
    observations: dict[str, "_Observation"] = field(default_factory=dict)


@dataclass(slots=True)
class _Observation:
    """Retain one in-flight SDK observation until its native calls actually return."""

    event: Event = field(default_factory=Event)
    result: tuple[Mapping[str, float], Mapping[str, float]] | None = None
    error: BaseException | None = None


class _ConnectionRegistry:
    """Serialize compatible process-global Ray initialization and owned leases."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._current: _Connection | None = None

    def acquire(self, config: RayBackendConfig, deadline: float) -> _Connection:
        """Join one compatible initializer and acquire a lease before returning it."""
        while True:
            disconnect: _RaySDK | None = None
            with self._condition:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Ray connection initialization exceeded timeout")
                current = self._current
                if current is not None and current.closing:
                    if current.address != config.address or current.namespace != config.namespace or current.sdk is None:
                        raise ExecutionError("Ray connection conflicts with the active address or namespace")
                    disconnect = current.sdk
                elif current is None:
                    current = _Connection(config.address, config.namespace, config.diagnostic_text_limit_bytes)
                    self._current = current
                    try:
                        Thread(target=self._initialize, args=(current,), name="dryml-execute-ray-connect", daemon=True).start()
                    except BaseException as exc:
                        current.error = ExecutionError(_native_error("Ray initialization could not start", exc, current.diagnostic_limit))
                        current.event.set()
                        self._current = None
                        self._condition.notify_all()
                elif not self._compatible(current, config):
                    raise ExecutionError("Ray connection conflicts with the active address or namespace")
                if disconnect is None:
                    current.waiters += 1
                    try:
                        while not current.event.is_set():
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("Ray connection initialization exceeded timeout")
                            self._condition.wait(remaining)
                        if current.error is not None:
                            if self._current is current and not current.closing:
                                self._current = None
                            raise current.error
                        if current.closing:
                            continue
                        current.leases += 1
                        return current
                    finally:
                        current.waiters -= 1
                        self._condition.notify_all()
            # A failed owned disconnect keeps this generation current.  A new
            # initializer may begin only after retrying that exact connection.
            assert current is not None and disconnect is not None
            self._disconnect_pending(current, disconnect, deadline)

    def release(self, connection: _Connection, *, deadline: float | None = None) -> None:
        """Release one executor lease and disconnect only an owned idle generation."""
        sdk: _RaySDK | None = None
        with self._condition:
            if connection.leases <= 0:
                return
            if connection.observations:
                raise CleanupError("Ray resource observation remains in flight")
            if connection.leases > 1:
                connection.leases -= 1
            elif not connection.owned:
                connection.leases = 0
                if self._current is connection:
                    # Caller-owned drivers can restart outside Execute.  Retain
                    # no identity cache after their final borrower departs.
                    self._current = None
            elif connection.event.is_set() and connection.error is None:
                connection.closing = True
                sdk = connection.sdk
            else:
                connection.leases -= 1
            self._condition.notify_all()
        if sdk is not None:
            self._disconnect_pending(connection, sdk, deadline)

    def _disconnect_pending(self, connection: _Connection, sdk: _RaySDK, deadline: float | None) -> None:
        """Serialize a retained generation's disconnect without touching a replacement.

        A concurrent releaser joins the exact in-flight disconnect rather than
        calling the SDK again.  A failed attempt remains owned by this generation
        and a later compatible caller may retry it before a new driver starts.
        """
        with self._condition:
            while connection.disconnecting:
                if deadline is None:
                    raise CleanupError("Ray driver disconnect remains in flight")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CleanupError("Ray driver disconnect remains in flight")
                self._condition.wait(remaining)
            if self._current is not connection:
                return
            connection.disconnecting = True
        try:
            # Ray shutdown disconnects this owned driver, not the existing server.
            sdk.disconnect()
        except BaseException as exc:
            with self._condition:
                connection.disconnecting = False
                self._condition.notify_all()
            raise ExecutionError(_native_error("Ray disconnect failed", exc, connection.diagnostic_limit)) from None
        with self._condition:
            connection.disconnecting = False
            connection.leases = 0
            if self._current is connection:
                self._current = None
            self._condition.notify_all()

    def _initialize(self, connection: _Connection) -> None:
        """Perform exactly one SDK operation, retaining late completion ownership."""
        sdk: _RaySDK | None = None
        owned = False
        disconnect_failed = False
        try:
            sdk = _SDK_FACTORY()
            borrowed = sdk.initialized()
            owned = not borrowed
            if not borrowed:
                sdk.connect(connection.address, connection.namespace)
            resolved, namespace, cluster_id = sdk.connection_identity()
            if borrowed and connection.address != "auto" and sdk.canonical_address(connection.address) != resolved:
                raise ExecutionError("borrowed Ray connection address is incompatible")
            if borrowed and connection.namespace != namespace:
                raise ExecutionError("borrowed Ray connection namespace is incompatible")
            node_id = _local_node(sdk)
            with self._condition:
                connection.sdk = sdk
                connection.owned = owned
                connection.node_id = node_id
                connection.resolved_address = resolved
                connection.cluster_id = cluster_id
        except BaseException as exc:
            if sdk is not None and owned and sdk.initialized():
                try:
                    sdk.disconnect()
                except BaseException:
                    disconnect_failed = True
            with self._condition:
                if disconnect_failed:
                    # A setup failure after connecting still owns this driver;
                    # retain it until a later compatible acquisition disconnects it.
                    connection.sdk = sdk
                    connection.owned = True
                    connection.closing = True
                connection.error = exc if isinstance(exc, ExecutionError) else ExecutionError(_native_error("Ray initialization failed", exc, connection.diagnostic_limit))
        finally:
            disconnect: _RaySDK | None = None
            with self._condition:
                connection.event.set()
                if connection.error is not None and self._current is connection and not connection.closing:
                    self._current = None
                if connection.error is None and connection.owned and connection.leases == 0 and connection.waiters == 0:
                    # Every waiter detached before this owned init completed. Do
                    # not leave an unleased driver connection behind indefinitely.
                    connection.closing = True
                    disconnect = connection.sdk
                self._condition.notify_all()
            if disconnect is not None:
                try:
                    self._disconnect_pending(connection, disconnect, None)
                except ExecutionError:
                    # With no eligible executor lease, the next compatible
                    # acquisition retries this retained generation's disconnect.
                    pass

    def resources(self, connection: _Connection, deadline: float) -> tuple[Mapping[str, float], Mapping[str, float]]:
        """Join one live capacity refresh or start a fresh bounded caller observation."""
        key = "resources"
        with self._condition:
            if connection.closing or connection.sdk is None:
                raise BackendUnavailableError("Ray connection is unavailable for resource observation")
            observation = connection.observations.get(key)
            if observation is None:
                observation = _Observation()
                connection.observations[key] = observation
                try:
                    Thread(target=self._observe_resources, args=(connection, key, observation), name="dryml-execute-ray-resources", daemon=True).start()
                except BaseException as exc:
                    observation.error = BackendUnavailableError(_native_error("Ray resource observation could not start", exc, connection.diagnostic_limit))
                    observation.event.set()
                    connection.observations.pop(key, None)
                    self._condition.notify_all()
            while not observation.event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Ray resource observation exceeded timeout")
                self._condition.wait(remaining)
            if observation.error is not None:
                raise observation.error
            assert observation.result is not None
            return observation.result

    def _observe_resources(self, connection: _Connection, key: str, observation: _Observation) -> None:
        """Call the blocking SDK outside registry and backend lifecycle locks."""
        try:
            sdk = connection.sdk
            if sdk is None:
                raise BackendUnavailableError("Ray connection is unavailable for resource observation")
            observation.result = (sdk.cluster_resources(), sdk.available_resources())
        except BaseException as exc:
            observation.error = exc if isinstance(exc, BackendUnavailableError) else BackendUnavailableError(
                _native_error("Ray resource observation failed", exc, connection.diagnostic_limit)
            )
        finally:
            with self._condition:
                observation.event.set()
                if connection.observations.get(key) is observation:
                    connection.observations.pop(key, None)
                self._condition.notify_all()

    @staticmethod
    def _compatible(connection: _Connection, config: RayBackendConfig) -> bool:
        return connection.address == config.address and connection.namespace == config.namespace and not connection.closing


_CONNECTIONS = _ConnectionRegistry()


class RayFuture(ExecutionFuture[T]):
    """Concrete Future retaining only this generation's Ray bootstrap reference.

    Native references are intentionally separate from the common managed result;
    cleanup clears them only after qualified backend release evidence is observed.
    Accessing native properties never imports Ray, contacts the server, or changes
    task cleanup ownership.
    """

    def __init__(self, submission_id: str, *, output: ExecutionOutput, termination_timeout: float) -> None:
        """Create the required inert Future without importing Ray."""
        super().__init__(submission_id, output=output, termination_timeout=termination_timeout, cleanup_scope="worker")
        self._native_lock = Lock()
        self._object_ref: Any | None = None
        self._server_address: str | None = None
        self._node_id: str | None = None

    @property
    def object_ref(self) -> Any | None:
        """Return this call's real bootstrap ObjectRef until qualified cleanup.

        Returns:
            The backend-created one-shot bootstrap reference, or ``None`` before
            submission and after qualified cleanup. It is not the user result.

        Side Effects:
            None. Access is inert and does not call Ray or retain the task.
        """
        with self._native_lock:
            return self._object_ref

    @property
    def server_address(self) -> str | None:
        """Return the generation-qualified resolved server address when known.

        Returns:
            The validated existing server address for this task generation, or
            ``None`` before association and after qualified cleanup.

        Side Effects:
            None. Access does not initialize or reconnect Ray.
        """
        with self._native_lock:
            return self._server_address

    @property
    def worker_pid(self) -> int | None:
        """Return the worker-confirmed PID without inferring identity from it.

        Returns:
            Positive PID after validated worker HELLO, otherwise ``None``; it is
            cleared during qualified cleanup.

        Side Effects:
            None. This is an inert observed association, not a native probe.
        """
        return self.snapshot().pid

    @property
    def worker_id(self) -> str | None:
        """Return the qualified Ray worker identity when it passed HELLO.

        Returns:
            The verified worker identity, or ``None`` before validation and after
            qualified cleanup.

        Side Effects:
            None. Access does not contact Ray or change task cleanup.
        """
        return self.snapshot().worker_id

    @property
    def node_id(self) -> str | None:
        """Return the verified native node identity for this bootstrap task.

        Returns:
            The task's verified same-host node identity, or ``None`` before
            native submission and after qualified cleanup.

        Side Effects:
            None. Access is inert and does not query native node state.
        """
        with self._native_lock:
            return self._node_id

    def _set_native(self, reference: Any, address: str, node_id: str) -> None:
        """Bind one actual native reference after this Future has been accepted."""
        with self._native_lock:
            self._object_ref = reference
            self._server_address = address
            self._node_id = node_id

    def _clear_native(self) -> None:
        """Expire backend-native access without rewriting retained common results."""
        with self._native_lock:
            self._object_ref = None
            self._server_address = None
            self._node_id = None


@dataclass(slots=True)
class _Run:
    """Own one Ray task attempt and its coordinator-only channel state."""

    future: RayFuture[Any]
    connection: socket.socket | None
    reservation: Reservation
    listener: socket.socket | None = None
    reference: Any | None = None
    native_submission_attempted: bool = False
    native_submission_uncertain: bool = False
    worker_id: str | None = None
    worker_pid: int | None = None
    worker_create_time: float | None = None
    native_node_id: str | None = None
    native_task_id: str | None = None
    issued: bool = False
    outcome_validated: bool = False
    native_terminal: bool = False
    native_cancelled: bool = False
    terminal_before_hello: bool = False
    termination_qualified: bool = False
    native_receipt: Mapping[str, object] | None = None
    native_done: Event = field(default_factory=Event)
    cancellation_starting: bool = False
    cancelling: bool = False
    outcome_claimed: bool = False
    deadline_expired: bool = False
    terminal_event: Event | None = None
    lock: Lock = field(default_factory=Lock)


class RayBackend(Backend):
    """Run accepted descriptor payloads on one verified existing local Ray node.

    The backend never provisions a cluster or transmits callable graphs to Ray.
    A pinned one-shot bootstrap task carries only the common channel descriptor;
    common protocol evidence still gates payload transfer and result publication.
    """

    def __init__(self, config: RayBackendConfig) -> None:
        """Store immutable policy without importing Ray or creating a connection."""
        if not isinstance(config, RayBackendConfig):
            raise TypeError("config must be a RayBackendConfig")
        self._config = config
        self._connection: _Connection | None = None
        self._remote: Any | None = None
        self._runs: dict[str, _Run] = {}
        self._pre_run_reservations: dict[str, Reservation] = {}
        self._known: set[str] = set()
        self._launching: dict[str, Event] = {}
        self._lock = RLock()
        self._closed = False
        self._executor_id = secrets.token_hex(16)
        self._authority: ResourceAuthority | None = None

    def start(self) -> None:
        """Attach to an existing compatible Ray deployment within caller bounds.

        Returns:
            ``None`` after attaching to one compatible same-host deployment.

        Raises:
            ExecutionError: If the backend is closed, Ray is unavailable, its
                version/identity is incompatible, or connection setup fails.
            TimeoutError: If the caller's configured connection wait expires.

        Side Effects:
            Lazily imports Ray and may attach an Execute-owned driver to an
            existing deployment. It never provisions or stops the server.
        """
        with self._lock:
            if self._closed:
                raise ExecutionError("Ray backend is closed")
            if self._connection is not None:
                return
        deadline = time.monotonic() + self._config.connect_timeout
        connection = _CONNECTIONS.acquire(self._config, deadline)
        assert connection.sdk is not None and connection.cluster_id is not None and connection.node_id is not None
        with self._lock:
            closed = self._closed
        if closed:
            _CONNECTIONS.release(connection)
            raise ExecutionError("Ray backend closed during initialization")
        try:
            remote = connection.sdk.remote_bootstrap()
        except BaseException:
            _CONNECTIONS.release(connection)
            raise
        with self._lock:
            closed = self._closed
            if not closed:
                self._connection = connection
                self._remote = remote
                self._authority = RESOURCE_AUTHORITIES.get("ray", connection.cluster_id)
        if closed:
            _CONNECTIONS.release(connection)
            raise ExecutionError("Ray backend closed during initialization")

    def capabilities(self) -> frozenset[str]:
        """Report the controls supported by the same-host pinned implementation.

        Returns:
            The immutable common capability set plus ``ray_existing_deployment``.
            ``world_admission`` is limited to one role with logical CPU, memory,
            and GPU quantities.

        Side Effects:
            None. This inert query does not import Ray or inspect a cluster.
        """
        return frozenset({"environment_selection", "world_admission", "live_output", "running_cancellation", "ray_existing_deployment"})

    def create_future(self, submission_id: str, output: ExecutionOutput) -> RayFuture[Any]:
        """Create the exact inert concrete Future before workload acceptance.

        Args:
            submission_id: Nonempty executor-generated submission identifier.
            output: Exact unbound output holder for that submission.

        Returns:
            An inert :class:`RayFuture` retaining the supplied identity.

        Raises:
            TypeError: If Future construction controls are invalid.
            ValueError: If ``submission_id`` is invalid.

        Side Effects:
            Does not import Ray, attach a driver, bind output, or schedule a task.
        """
        return RayFuture(submission_id, output=output, termination_timeout=self._config.termination_timeout)

    def submit(self, call: SubmittedCall[T], *, future: ExecutionFuture[T]) -> None:
        """Schedule the supplied Future only; native arguments never include user code.

        Args:
            call: Exact accepted spool and admission metadata.
            future: Exact matching inert :class:`RayFuture`.

        Returns:
            ``None``; asynchronous admission outcomes publish on ``future``.

        Raises:
            ExecutionError: If Ray was not started, the backend is closed, or
                call/Future identity is mismatched or duplicated.

        Side Effects:
            Starts one coordinator thread that may schedule one descriptor-only
            native bootstrap task; user callable bytes never enter native options.
        """
        if not isinstance(future, RayFuture) or future.submission_id != call.submission_id or future.output is not call.output:
            raise ExecutionError("Ray backend received a mismatched Future or call")
        with self._lock:
            if self._closed or self._connection is None:
                raise ExecutionError("Ray backend is not initialized")
            if call.submission_id in self._known:
                raise ExecutionError("duplicate Ray submission ID")
            self._known.add(call.submission_id)
            self._launching[call.submission_id] = Event()
        try:
            Thread(target=self._run, args=(call, future), name="dryml-execute-ray", daemon=False).start()
        except BaseException:
            with self._lock:
                self._known.discard(call.submission_id)
                self._launching.pop(call.submission_id, None)
            raise

    def discover(self, *, environment: EnvironmentRequirement | None = None, world: Any = None, timeout: float) -> DiscoverySnapshot:
        """Return bounded existing-environment and native-resource observations.

        Args:
            environment: Optional environment requirement for candidate evidence.
            world: Optional supported one-role world requirement for plans.
            timeout: Positive total observation budget in seconds.

        Returns:
            A non-reserving Ray discovery snapshot, possibly incomplete.

        Raises:
            TimeoutError: If the supplied budget has elapsed.
            BackendUnavailableError: If the existing Ray connection cannot observe
                its resources.

        Side Effects:
            May perform bounded candidate probes and Ray capacity reads; it does
            not provision a server, reserve capacity, or invoke workload code.
        """
        if timeout <= 0:
            raise TimeoutError("Ray discovery timeout elapsed")
        deadline = time.monotonic() + timeout
        resources = self._resources_until(deadline)
        issues: list[ExecutionIssue] = []
        candidates: list[EnvironmentCandidate] = []
        plans: list[FeasiblePlan] = []
        inventory_complete = True
        if environment is not None or self._config.automatic_environment_discovery or self._config.environment_candidates or self._config.environment_search_roots:
            discovered = discover_candidates(self._config, cwd=self._config.working_directory, interpreter=Path(sys.executable), deadline=deadline)
            inventory_complete = discovered.complete
            issues.extend(discovered.issues)
            for spec in discovered.specs:
                if time.monotonic() >= deadline:
                    issues.append(ExecutionIssue("discovery_timeout", "Ray environment discovery exceeded its deadline"))
                    break
                candidate = probe_candidate(spec, interpreter=Path(sys.executable), timeout=max(0.001, deadline - time.monotonic()), output_limit=self._config.owner_envelope_limit_bytes, deadline=deadline, termination_timeout=self._config.termination_timeout, read_chunk_bytes=self._config.process_read_chunk_bytes, poll_interval=self._config.process_poll_interval)
                candidates.append(candidate)
                if world is not None and candidate.record is not None:
                    decision = _logical_plan(environment, world, candidate.record, resources.total, deadline)
                    if decision.go and decision.world is not None:
                        plans.append(FeasiblePlan(candidate.key, decision.world, None, decision.report))
        if world is not None and environment is None:
            decision = _logical_plan(None, world, None, resources.total, deadline)
            if decision.go and decision.world is not None:
                plans.append(FeasiblePlan(None, decision.world, None, decision.report))
        return DiscoverySnapshot(datetime.now(timezone.utc), tuple(candidates), resources, tuple(plans), resources.complete and inventory_complete and not issues, tuple(issues))

    def resources(self, *, timeout: float) -> ResourceSnapshot:
        """Observe native Ray capacity and subtract only unrepresented reservations.

        Args:
            timeout: Positive total native observation budget in seconds.

        Returns:
            A backend-scoped logical Ray resource snapshot, possibly incomplete.

        Raises:
            TimeoutError: If the supplied budget has elapsed.
            BackendUnavailableError: If the existing Ray connection cannot observe
                scheduler capacity.

        Side Effects:
            Calls existing Ray capacity APIs but does not schedule, reserve, or
            provision native work.
        """
        if timeout <= 0:
            raise TimeoutError("Ray resource observation timeout elapsed")
        return self._resources_until(time.monotonic() + timeout)

    def _resources_until(self, deadline: float) -> ResourceSnapshot:
        """Build one fresh capacity snapshot within an already-established deadline."""
        if time.monotonic() >= deadline:
            raise TimeoutError("Ray resource observation timeout elapsed")
        connection = self._require_connection()
        assert self._authority is not None
        total_values, available_values = _CONNECTIONS.resources(connection, deadline)
        total = _resource_amounts(total_values)
        available = _resource_amounts(available_values)
        return self._authority.snapshot(total=total, native_available=available, native_available_is_net=True, executor_id=self._executor_id)

    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        """Release only a terminal attempt with matching worker and native evidence.

        Args:
            submission_id: Exact accepted Ray submission identifier.
            timeout: Positive maximum task reconciliation time in seconds.

        Returns:
            ``None`` after qualified native task, socket, and charge release.

        Raises:
            ExecutionError: If the submission is unknown.
            RuntimeError: If the matching Future is not terminal.
            CleanupError: If launch, native task, or resource release evidence is
                unavailable; state remains retryable.

        Side Effects:
            May close only this submission's channel and clear its native Future
            associations. It releases no caller-owned Ray driver or server.
        """
        with self._lock:
            run = self._runs.get(submission_id)
            known = submission_id in self._known
            launching = self._launching.get(submission_id)
        if run is None and not known:
            raise ExecutionError("unknown Ray submission")
        if run is None:
            if launching is None or not launching.wait(timeout):
                raise CleanupError("Ray launch ownership is still unresolved")
            with self._lock:
                run = self._runs.get(submission_id)
                known = submission_id in self._known
            if run is None:
                if known:
                    with self._lock:
                        reservation = self._pre_run_reservations.get(submission_id)
                    if reservation is not None:
                        assert self._authority is not None
                        if not self._authority.release(
                            submission_id, generation=reservation.generation, attempt=reservation.attempt,
                            worker_id=None, never_submitted=True,
                        ):
                            raise CleanupError("Ray pre-submission release remains unconfirmed")
                    with self._lock:
                        self._known.discard(submission_id)
                        self._launching.pop(submission_id, None)
                        self._pre_run_reservations.pop(submission_id, None)
                return
        if not run.future.done():
            raise RuntimeError("cleanup requires a terminal Ray execution")
        deadline = time.monotonic() + timeout
        with run.lock:
            if run.native_submission_uncertain:
                raise CleanupError("Ray native submission is uncertain; native task identity is unavailable", execution=run.future)
        if run.native_submission_attempted and not run.native_done.wait(max(0.0, deadline - time.monotonic())):
            raise CleanupError("Ray task terminal evidence remains unavailable", execution=run.future)
        with run.lock:
            qualified = run.worker_id is not None and (
                run.outcome_validated and run.native_terminal
                or run.termination_qualified
                or run.native_terminal and not run.issued
            ) or run.terminal_before_hello and not run.issued
        assert self._authority is not None
        if not run.native_submission_attempted:
            released = self._authority.release(submission_id, generation=self._generation(), attempt="0", worker_id=None, never_submitted=True)
            if not released:
                raise CleanupError("Ray pre-submission release remains unconfirmed", execution=run.future)
        elif not qualified or not self._authority.release(submission_id, generation=self._generation(), attempt="0", worker_id=run.worker_id, qualified_terminal=True):
            self._authority.release(submission_id, generation=self._generation(), attempt="0", worker_id=run.worker_id)
            raise CleanupError("Ray task release remains unconfirmed", execution=run.future)
        if run.connection is not None:
            run.connection.close()
            run.connection = None
        if run.listener is not None:
            run.listener.close()
            run.listener = None
        run.future._clear_native()
        run.future._set_association(worker_id=None, pid=None)
        if run.terminal_event is not None:
            run.terminal_event.set()
        with self._lock:
            self._runs.pop(submission_id, None)
            self._known.discard(submission_id)
            self._launching.pop(submission_id, None)
            self._pre_run_reservations.pop(submission_id, None)

    def close(self, *, cancel: bool, timeout: float | None) -> None:
        """Cancel/reconcile only owned tasks, then release this executor's SDK lease.

        Args:
            cancel: Whether to request cancellation of outstanding owned tasks.
            timeout: Optional cleanup budget; ``None`` uses termination policy.

        Returns:
            ``None`` after owned tasks and this executor's connection lease close.

        Raises:
            CleanupError: If an owned task or the Execute-owned driver disconnect
                remains incomplete; retry ownership is preserved.

        Side Effects:
            Stops backend admission, may cancel exact native tasks, and releases
            only an Execute-owned driver lease, never the caller-owned server.
        """
        with self._lock:
            self._closed = True
            runs = tuple(self._runs.values())
            connection = self._connection
        deadline = time.monotonic() + (self._config.termination_timeout if timeout is None else timeout)
        unresolved: list[_Run] = []
        for run in runs:
            if cancel and run.reference is not None:
                self._request_cancel(run, allow_terminal=True)
            if run.future.done():
                self.reconcile_cleanup(run.future.submission_id, timeout=max(0.001, deadline - time.monotonic()))
            else:
                unresolved.append(run)
        if unresolved:
            raise CleanupError("Ray backend retains unresolved owned tasks", execution=unresolved[0].future)
        if connection is not None:
            _CONNECTIONS.release(connection, deadline=deadline)
            with self._lock:
                self._connection = None

    def _run(self, call: SubmittedCall[T], future: RayFuture[T]) -> None:
        """Perform one local-node schedule, common handshake, and no-retry receive."""
        listener: socket.socket | None = None
        run: _Run | None = None
        try:
            if not future._begin_admission():
                return
            connection = self._require_connection()
            candidate, runtime_env = self._select_environment(call)
            reservation = self._reserve(call)
            with self._lock:
                self._pre_run_reservations[call.submission_id] = reservation
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(max(0.001, call.admission_deadline - time.monotonic()))
            descriptor = self._descriptor(call, listener)
            encoded = encode_bootstrap_descriptor(descriptor)
            options = {"scheduling_strategy": connection.sdk.node_affinity(connection.node_id, soft=False)}
            options.update(_native_options(call.world))
            effective_runtime_env = self._runtime_environment(runtime_env, candidate)
            if effective_runtime_env:
                options["runtime_env"] = effective_runtime_env
            run = _Run(future, None, reservation, listener=listener)
            with self._lock:
                self._runs[call.submission_id] = run
                self._pre_run_reservations.pop(call.submission_id, None)
            future._set_cancel_requester(lambda: self._request_cancel(run))
            if future.done():
                return
            try:
                reference = self._remote.options(**options).remote(encoded)
            except BaseException:
                # Ray does not promise that a raised submission call was never
                # accepted. There is no ObjectRef to monitor or cancel, so retain
                # the charge as an explicitly recoverable uncertainty.
                with run.lock:
                    run.native_submission_uncertain = True
                run.native_done.set()
                raise
            run.reference = reference
            run.native_submission_attempted = True
            future._set_native(reference, connection.resolved_address or self._config.address, connection.node_id or "")
            future._set_association(backend_job_id=_reference_id(reference))
            # Observe this exact ObjectRef as soon as it exists: bootstrap failure
            # must not wait for an unrelated listener timeout.
            Thread(target=self._watch_native, args=(run,), name="dryml-execute-ray-terminal", daemon=True).start()
            if not self._authority.mark_submitted(call.submission_id, generation=self._generation(), attempt="0", backend_job_id=_reference_id(reference)):
                raise ExecutionError("Ray resource reservation was lost before native submission")
            worker_connection, hello, conversation = self._accept_worker(listener, run, descriptor, call.admission_deadline)
            worker_connection.settimeout(max(0.001, call.admission_deadline - time.monotonic()))
            run.connection = worker_connection
            self._handshake(call, future, run, descriptor, candidate, connection, hello, conversation)
        except BaseException as exc:
            if run is None:
                try:
                    reservation = locals().get("reservation")
                    if reservation is not None and self._authority is not None:
                        released = self._authority.release(call.submission_id, generation=reservation.generation, attempt=reservation.attempt, worker_id=None, never_submitted=True)
                        if released:
                            with self._lock:
                                self._pre_run_reservations.pop(call.submission_id, None)
                except BaseException:
                    pass
            if run is not None and run.reference is not None and not run.issued:
                try:
                    # Admission failures after a concrete native submission must
                    # retire that exact queued task even after the common Future
                    # has received its admission outcome.
                    self._request_cancel(run, allow_terminal=True)
                except BaseException:
                    pass
            if not future.done():
                if run is not None and run.issued:
                    future._publish_uncertain(ExecutionUncertainError("Ray execution failed after payload transfer began"))
                elif run is not None and run.native_submission_uncertain:
                    future._publish_uncertain(ExecutionUncertainError("Ray native submission is uncertain; native task identity is unavailable"))
                else:
                    future._publish_exception(exc if isinstance(exc, ExecutionError) else ExecutionError("Ray execution failed"))
                if run is not None and run.terminal_event is not None:
                    run.terminal_event.set()
        finally:
            with self._lock:
                launching = self._launching.get(call.submission_id)
                if launching is not None:
                    launching.set()
            if listener is not None:
                listener.close()
            if run is not None:
                with run.lock:
                    if run.listener is listener:
                        run.listener = None

    def _handshake(self, call: SubmittedCall[T], future: RayFuture[T], run: _Run, descriptor: BootstrapDescriptor, candidate: EnvironmentCandidate | None, connection: _Connection, hello: Any, conversation: ProtocolConversation) -> None:
        """Validate native node/worker locality before owner controls or payload egress."""
        assert run.connection is not None
        reader = SocketFrameReader(run.connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=self._limits())
        control = decode_control(hello, limit_bytes=self._config.control_header_limit_bytes, required_keys=_HELLO_FIELDS)
        if control["token"] != descriptor.rendezvous_token or control["protocol"] != WORKER_PROTOCOL_ID or control["dill"] != dill.__version__ or control["implementation"] != sys.implementation.name or control["python"] != list(sys.version_info[:2]):
            raise AdmissionError("Ray worker runtime is incompatible")
        native = control["native"]
        if not isinstance(native, Mapping) or native.get("node_id") != connection.node_id or native.get("worker_id") is None or native.get("task_id") is None or native.get("ray") != _RAY_VERSION or native.get("python") != list(sys.version_info[:3]):
            raise AdmissionError("Ray task was not placed on the verified local node")
        pid = control["pid"]
        worker_id = control["worker_id"]
        create_time = native.get("process_create_time")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not isinstance(worker_id, str) or not worker_id.startswith("ray:") or isinstance(create_time, bool) or not isinstance(create_time, (int, float)) or not math.isfinite(create_time) or create_time <= 0:
            raise AdmissionError("Ray worker identity is invalid")
        run.worker_id, run.worker_pid, run.worker_create_time = worker_id, pid, float(create_time)
        run.native_node_id = str(native["node_id"])
        run.native_task_id = str(native["task_id"])
        self._qualify_native_terminal(run)
        future._set_association(worker_id=worker_id, pid=pid, environment=candidate)
        owners: list[tuple[OwnerEnvelopeType, bytes]] = []
        if call.environment is not None:
            owners.append((OwnerEnvelopeType.ENVIRONMENT, _owner_json(call.environment.to_data(), self._config.owner_envelope_limit_bytes)))
        if call.world is not None:
            owners.append((OwnerEnvelopeType.WORLD, _owner_json(call.world.to_data(), self._config.owner_envelope_limit_bytes)))
        prepare = encode_control(FrameState.PREPARE, descriptor.correlation, {"cwd": str(self._config.working_directory), "deadline": call.admission_deadline, "owners": [kind.value for kind, _ in owners]}, header_limit=self._config.control_header_limit_bytes)
        self._send(run.connection, prepare)
        conversation.accept(prepare)
        for kind, data in owners:
            frame = encode_owner_envelope(FrameState.PREPARE, descriptor.correlation, kind, data, header_limit=self._config.control_header_limit_bytes, owner_limit=self._config.owner_envelope_limit_bytes)
            self._send(run.connection, frame)
            conversation.accept(frame)
        ready = reader.read()
        conversation.accept_frame(ready)
        ready_data = decode_control(ready, limit_bytes=self._config.control_header_limit_bytes, required_keys={"pid", "ready", "worker_id"})
        if ready_data["ready"] is not True or ready_data["pid"] != pid or ready_data["worker_id"] != worker_id:
            raise AdmissionError("Ray worker readiness evidence is invalid")
        record: EnvironmentRecord | None = None
        if call.environment is not None:
            evidence = reader.read()
            conversation.accept_frame(evidence)
            if evidence.owner is not OwnerEnvelopeType.ENVIRONMENT:
                raise AdmissionError("Ray worker omitted environment evidence")
            record = EnvironmentRecord.from_data(_json(evidence.payload, self._config.owner_envelope_limit_bytes))
        observed_world, controls = _logical_world(call.world, native)
        decision = _admit_observed_logical(
            environment=call.environment, world=call.world, record=record,
            observed_world=observed_world, controls=controls,
            deadline=call.admission_deadline,
        )
        future._set_association(report=decision.report)
        if not decision.go:
            raise AdmissionError("Ray worker admission evidence is not admissible", report=decision.report)
        actual = _resource_amounts(native.get("assigned_resources", {}))
        if not self._authority.confirm_grant(call.submission_id, generation=self._generation(), attempt="0", worker_id=worker_id, resources=actual, backend_job_id=_reference_id(run.reference), pid=pid):
            raise AdmissionError("Ray worker grant does not match its reservation")
        if not future._authorize(deadline=call.admission_deadline):
            self._send_stop(run, descriptor)
            return
        execution_deadline = None if call.execution_timeout is None else time.monotonic() + call.execution_timeout
        go = encode_control(FrameState.GO, descriptor.correlation, {"deadline": execution_deadline, "permit": descriptor.rendezvous_token}, header_limit=self._config.control_header_limit_bytes)
        self._send(run.connection, go)
        conversation.accept(go)
        # A post-GO setup is workload execution, not an extension of admission.
        run.connection.settimeout(None)
        with run.lock:
            run.issued = True
            run.terminal_event = run.terminal_event or Event()
        if execution_deadline is not None:
            Thread(target=self._deadline_watch, args=(run, execution_deadline), name="dryml-execute-ray-deadline", daemon=True).start()
        if call.worker_setup is not None:
            if not self._send_setup(call, future, run, descriptor, conversation, reader, record, native):
                return
        payload = call.payload.path.read_bytes()
        if len(payload) != call.payload.size_bytes or hashlib.sha256(payload).hexdigest() != call.payload.sha256:
            raise ExecutionError("coordinator invocation spool changed before Ray transfer")
        payload_frame, payload_prefix, payload_view = encode_frame_parts(FrameState.PAYLOAD, FrameType.PAYLOAD, descriptor.correlation, payload, header_limit=self._config.control_header_limit_bytes)
        self._send(run.connection, (payload_prefix, payload_view))
        conversation.accept_frame(payload_frame)
        self._receive_active(call, future, run, descriptor, conversation)

    def _send_setup(self, call: SubmittedCall[T], future: RayFuture[T], run: _Run, descriptor: BootstrapDescriptor, conversation: ProtocolConversation, reader: SocketFrameReader, record: EnvironmentRecord | None, native: Mapping[str, object]) -> bool:
        """Send qualified Ray grant evidence and drain setup output before payload."""
        assert run.connection is not None and call.worker_setup is not None
        native_grant = {
            "kind": "ray",
            "node_id": run.native_node_id,
            "task_id": run.native_task_id,
            "worker_id": run.worker_id,
            "resources": dict(native.get("assigned_resources", {})) if isinstance(native.get("assigned_resources"), Mapping) else {},
        }
        context = {
            "submission_id": call.submission_id,
            "backend": "ray",
            "environment": None if record is None else record.to_data(),
            "allocation": None,
            "native_grant": native_grant,
        }
        try:
            encoded = call.worker_setup.to_envelope(
                context, owner_limit_bytes=self._config.owner_envelope_limit_bytes,
                admission_limit_bytes=self._config.admission_message_limit_bytes,
            )
        except BaseException as exc:
            raise ExecutionError("Ray worker setup could not be encoded") from exc
        frame = encode_owner_envelope(FrameState.SETUP, descriptor.correlation, OwnerEnvelopeType.SETUP, encoded, header_limit=self._config.control_header_limit_bytes, owner_limit=self._config.owner_envelope_limit_bytes)
        self._send(run.connection, frame)
        conversation.accept(frame)
        while True:
            frame = reader.read()
            conversation.accept_frame(frame)
            if frame.state is FrameState.OUTPUT:
                assert frame.stream is not None and frame.sequence is not None
                call.output._capture(frame.stream, frame.payload, frame.sequence)
                continue
            if frame.state is FrameState.OUTPUT_FINAL:
                data = _json(frame.payload, self._config.control_header_limit_bytes)
                call.output._finalize(data["stream"], data["next_sequence"])
                continue
            if frame.state is FrameState.SETUP_READY:
                ready = decode_control(frame, limit_bytes=self._config.control_header_limit_bytes, required_keys={"ready"})
                if ready["ready"] is not True:
                    raise ExecutionError("Ray worker setup readiness evidence is invalid")
                return True
            if frame.state is FrameState.ERROR:
                _, remote_type, issues = decode_setup_terminal(frame, limit_bytes=self._config.result_limit_bytes)
                future._record_worker_cleanup_issues(issues)
                future._publish_exception(RemoteExecutionError(f"remote Ray setup failed ({remote_type})", remote_type=remote_type or "RemoteError"))
                run.outcome_validated = True
                self._drain_setup_failure(call, run, reader, conversation)
                return False
            raise FrameError("Ray worker setup sent an invalid frame")

    def _drain_setup_failure(self, call: SubmittedCall[T], run: _Run, reader: SocketFrameReader, conversation: ProtocolConversation) -> None:
        """Drain only bounded final output after setup fails before payload delivery."""
        assert run.connection is not None
        deadline = time.monotonic() + self._config.output_final_timeout
        try:
            while time.monotonic() < deadline:
                run.connection.settimeout(max(0.001, deadline - time.monotonic()))
                try:
                    frame = reader.read()
                    conversation.accept_frame(frame)
                except (OSError, EOFError, FrameError):
                    return
                if frame.state is FrameState.OUTPUT:
                    assert frame.stream is not None and frame.sequence is not None
                    call.output._capture(frame.stream, frame.payload, frame.sequence)
                elif frame.state is FrameState.OUTPUT_FINAL:
                    data = _json(frame.payload, self._config.control_header_limit_bytes)
                    call.output._finalize(data["stream"], data["next_sequence"])
        finally:
            run.connection.settimeout(None)

    def _receive_active(self, call: SubmittedCall[T], future: RayFuture[T], run: _Run, descriptor: BootstrapDescriptor, conversation: ProtocolConversation) -> None:
        """Route common output and publish only a validated channel terminal state."""
        assert run.connection is not None
        reader = SocketFrameReader(run.connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=self._limits())
        outcome_seen = False
        while True:
            try:
                frame = reader.read()
                conversation.accept_frame(frame)
            except (OSError, EOFError, FrameError):
                if outcome_seen or future.done() or run.cancelling:
                    return
                if not future.done():
                    future._publish_uncertain(ExecutionUncertainError("Ray worker disconnected before a validated outcome"))
                    if run.terminal_event is not None:
                        run.terminal_event.set()
                return
            if frame.state is FrameState.OUTPUT:
                assert frame.stream is not None and frame.sequence is not None
                call.output._capture(frame.stream, frame.payload, frame.sequence)
            elif frame.state is FrameState.OUTPUT_FINAL:
                data = _json(frame.payload, self._config.control_header_limit_bytes)
                call.output._finalize(data["stream"], data["next_sequence"])
            elif frame.state is FrameState.RESULT:
                if outcome_seen or not self._claim_outcome(run):
                    return
                outcome_seen = True
                try:
                    payload = frame.payload
                    if call.worker_setup is not None:
                        payload, _, issues = decode_setup_terminal(frame, limit_bytes=self._config.result_limit_bytes)
                        assert payload is not None
                        future._record_worker_cleanup_issues(issues)
                    future._publish_result(future._receive_result(payload))
                except BaseException:
                    future._publish_exception(ExecutionError("Ray worker result could not be decoded"))
                run.outcome_validated = True
            elif frame.state is FrameState.ERROR:
                if outcome_seen or not self._claim_outcome(run):
                    return
                outcome_seen = True
                if call.worker_setup is not None:
                    _, remote_type, issues = decode_setup_terminal(frame, limit_bytes=self._config.result_limit_bytes)
                    future._record_worker_cleanup_issues(issues)
                else:
                    data = _json(frame.payload, self._config.result_limit_bytes)
                    remote_type = str(data.get("type", "RemoteError"))[:self._config.diagnostic_text_limit_bytes]
                future._publish_exception(RemoteExecutionError(f"remote Ray execution failed ({remote_type})", remote_type=remote_type))
                run.outcome_validated = True

    def _watch_native(self, run: _Run) -> None:
        """Pair the exact task's marker with channel evidence for qualified release."""
        try:
            while True:
                try:
                    receipt = self._require_connection().sdk.get(run.reference, timeout=self._config.output_final_timeout)
                    break
                except BaseException as exc:
                    if type(exc).__name__ == "GetTimeoutError":
                        continue
                    task_cancelled = type(exc).__name__ == "TaskCancelledError"
                    terminal_task = _terminal_task_error(exc)
                    if task_cancelled:
                        with run.lock:
                            run.native_cancelled = True
                            cancelling = run.cancelling or run.cancellation_starting
                            before_hello = run.worker_id is None and not run.issued
                            if before_hello and terminal_task:
                                run.terminal_before_hello = True
                        if before_hello:
                            self._wake_pending_accept(run)
                        if cancelling:
                            return
                    if not run.future.done():
                        detail = type(exc).__name__[:self._config.diagnostic_text_limit_bytes]
                        if run.issued:
                            run.future._publish_uncertain(ExecutionUncertainError("Ray bootstrap failed after payload transfer began"))
                        else:
                            run.future._publish_exception(ExecutionError(f"Ray bootstrap failed ({detail})"))
                    with run.lock:
                        if terminal_task and run.worker_id is None and not run.issued:
                            run.terminal_before_hello = True
                    if terminal_task:
                        self._wake_pending_accept(run)
                    return
            with run.lock:
                run.native_receipt = receipt if isinstance(receipt, Mapping) else None
            self._qualify_native_terminal(run)
            if run.cancelling and run.native_terminal and not run.future.done():
                if run.deadline_expired:
                    run.future._expire(ExecutionDeadlineExceeded("execution deadline exceeded"))
                else:
                    run.future._publish_running_cancellation()
        finally:
            if run.future.done() and run.terminal_event is not None:
                run.terminal_event.set()
            run.native_done.set()

    @staticmethod
    def _wake_pending_accept(run: _Run) -> None:
        """Close only this failed task's listener; polling also covers close races."""
        with run.lock:
            listener = run.listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _accept_worker(self, listener: socket.socket, run: _Run, descriptor: BootstrapDescriptor, deadline: float) -> tuple[socket.socket, Any, ProtocolConversation]:
        """Return only a correlation/token-validated HELLO before payload transfer."""
        while True:
            with run.lock:
                if run.terminal_before_hello:
                    raise ExecutionError("Ray bootstrap terminated before worker handshake")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Ray worker handshake exceeded admission deadline")
            listener.settimeout(min(remaining, self._config.process_poll_interval))
            try:
                connection, _ = listener.accept()
                try:
                    # Poll listener acceptance frequently so native terminal
                    # evidence wakes promptly, but permit a complete bounded
                    # HELLO to use the remaining admission budget.
                    connection.settimeout(max(0.001, remaining))
                    hello = SocketFrameReader(connection, header_limit=descriptor.control_header_limit_bytes, payload_limit=self._limits()).read()
                    conversation = self._conversation(descriptor)
                    conversation.accept_frame(hello)
                    if decode_control(hello, limit_bytes=self._config.control_header_limit_bytes, required_keys=_HELLO_FIELDS)["token"] != descriptor.rendezvous_token:
                        raise FrameError("Ray worker rendezvous token does not match")
                    return connection, hello, conversation
                except (OSError, EOFError, FrameError, ValueError):
                    connection.close()
                    continue
            except socket.timeout:
                continue
            except OSError as exc:
                with run.lock:
                    if run.terminal_before_hello:
                        raise ExecutionError("Ray bootstrap terminated before worker handshake") from exc
                raise

    @staticmethod
    def _qualify_native_terminal(run: _Run) -> None:
        """Accept a terminal receipt only when it names this socket worker and task."""
        with run.lock:
            receipt = run.native_receipt
            run.native_terminal = bool(
                isinstance(receipt, Mapping)
                and receipt.get("marker") == _BOOTSTRAP_MARKER
                and run.worker_id is not None
                and run.native_node_id is not None
                and run.native_task_id is not None
                and receipt.get("worker_id") == run.worker_id.removeprefix("ray:")
                and receipt.get("node_id") == run.native_node_id
                and receipt.get("task_id") == run.native_task_id
            )

    def _request_cancel(self, run: _Run, *, allow_terminal: bool = False) -> bool:
        """Request cancellation for exactly this task without claiming release early."""
        if run.reference is None or run.future.done() and not allow_terminal:
            return False
        with run.lock:
            if run.cancelling:
                return True
            if run.cancellation_starting:
                return False
            run.cancellation_starting = True
        try:
            self._require_connection().sdk.cancel(run.reference)
        except BaseException:
            with run.lock:
                run.cancellation_starting = False
            raise
        try:
            Thread(target=self._cancel_monitor, args=(run,), name="dryml-execute-ray-cancel", daemon=True).start()
        except BaseException:
            with run.lock:
                # The SDK may have synchronously caused native terminal evidence
                # before thread startup failed. Keep one synchronous monitor owner
                # rather than reporting a cancellation that nobody can confirm.
                run.cancellation_starting = False
                run.cancelling = True
            self._cancel_monitor(run)
            return True
        with run.lock:
            run.cancellation_starting = False
            run.cancelling = True
        return True

    def _cancel_monitor(self, run: _Run) -> None:
        """Publish cancellation only after the exact native task reaches a known stop."""
        if not run.native_done.wait(self._config.termination_timeout):
            if not run.future.done():
                run.future._publish_uncertain(ExecutionUncertainError("Ray task cancellation could not be confirmed"))
                if run.terminal_event is not None:
                    run.terminal_event.set()
            return
        with run.lock:
            confirmed = run.native_terminal
            cancelled = run.native_cancelled
            deadline_expired = run.deadline_expired
        if cancelled:
            deadline = time.monotonic() + self._config.termination_timeout
            while not _worker_lifetime_ended(run):
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            confirmed = _worker_lifetime_ended(run)
        if confirmed:
            with run.lock:
                run.termination_qualified = True
                if run.terminal_event is not None:
                    run.terminal_event.set()
        if not confirmed or run.future.done():
            if not run.future.done():
                run.future._publish_uncertain(ExecutionUncertainError("Ray task termination evidence is inconclusive"))
                if run.terminal_event is not None:
                    run.terminal_event.set()
        elif deadline_expired:
            run.future._expire(ExecutionDeadlineExceeded("execution deadline exceeded"))
        else:
            run.future._publish_running_cancellation()

    def _deadline_watch(self, run: _Run, deadline: float) -> None:
        """Request exact-task cancellation after a workload deadline without replay."""
        event = run.terminal_event
        if event is not None and event.wait(max(0.0, deadline - time.monotonic())):
            return
        with run.lock:
            if run.outcome_claimed or run.future.done():
                return
            run.deadline_expired = True
        try:
            accepted = self._request_cancel(run)
        except BaseException:
            accepted = False
        if not accepted and not run.future.done():
            run.future._publish_uncertain(ExecutionUncertainError("Ray execution deadline cancellation could not start"))
            if run.terminal_event is not None:
                run.terminal_event.set()

    @staticmethod
    def _claim_outcome(run: _Run) -> bool:
        """Linearize a validated frame against deadline cancellation without decoding under lock."""
        with run.lock:
            if run.deadline_expired or run.outcome_claimed or run.cancelling or run.cancellation_starting:
                return False
            run.outcome_claimed = True
            if run.terminal_event is not None:
                run.terminal_event.set()
            return True

    def _select_environment(self, call: SubmittedCall[T]) -> tuple[EnvironmentCandidate | None, Mapping[str, str] | None]:
        """Select only pre-existing Conda or experimental venv runtime forms."""
        if call.environment is None:
            return None, None
        deadline = call.admission_deadline
        for spec in discover_candidates(self._config, cwd=self._config.working_directory, interpreter=Path(sys.executable), deadline=deadline).specs:
            candidate = probe_candidate(spec, interpreter=Path(sys.executable), timeout=max(0.001, deadline - time.monotonic()), output_limit=self._config.owner_envelope_limit_bytes, deadline=deadline, termination_timeout=self._config.termination_timeout, read_chunk_bytes=self._config.process_read_chunk_bytes, poll_interval=self._config.process_poll_interval)
            if candidate.record is None or candidate.launchable is not True or not _compatible_runtime(spec):
                continue
            if not admit(environment=call.environment, record=candidate.record, deadline=deadline).go:
                continue
            if isinstance(spec, CondaEnvironmentSpec):
                return candidate, {"conda": spec.prefix or spec.name or ""}
            if isinstance(spec, PythonExecutableSpec):
                return candidate, {"py_executable": spec.executable}
            if isinstance(spec, CurrentEnvironmentSpec):
                return candidate, None
        raise AdmissionError("no existing Ray runtime satisfies the supplied environment requirement")

    def _runtime_environment(self, selector: Mapping[str, str] | None, candidate: EnvironmentCandidate | None) -> dict[str, object]:
        """Merge selected runtime form with explicit worker overrides without ambient state.

        Candidate selector fields such as Conda names and Python executables remain
        intact. Candidate environment declarations apply first, and the explicit
        backend configuration wins where both declare the same variable.
        """
        runtime_env: dict[str, object] = dict(selector or {})
        worker_env: dict[str, str] = {}
        if candidate is not None and isinstance(candidate.spec, (CondaEnvironmentSpec, PythonExecutableSpec)):
            worker_env.update(candidate.spec.env)
        worker_env.update(self._config.env_vars)
        if worker_env:
            runtime_env["env_vars"] = worker_env
        return runtime_env

    def _reserve(self, call: SubmittedCall[T]) -> Reservation:
        """Reserve known native capacity before task scheduling, not after its grant."""
        connection = self._require_connection()
        assert self._authority is not None
        requested = _requested_amounts(call.world)
        total = _resource_amounts(connection.sdk.cluster_resources())
        if not _amounts_cover(total, requested):
            raise AdmissionError("Ray resource requirement is unsupported, unavailable, or incompatible")
        while True:
            revision = self._authority.revision()
            reservation = self._authority.reserve(call.submission_id, requested, generation=self._generation(), attempt="0", total=total, executor_id=self._executor_id)
            if reservation is not None:
                return reservation
            remaining = call.admission_deadline - time.monotonic()
            if remaining <= 0:
                raise AdmissionError("admission deadline elapsed waiting for Ray resources")
            self._authority.wait_for_change(revision, min(remaining, 0.05))

    def _descriptor(self, call: SubmittedCall[T], listener: socket.socket) -> BootstrapDescriptor:
        """Create the sole bounded native task argument using effective config limits."""
        return BootstrapDescriptor(Correlation(call.submission_id, 0, 1), secrets.token_hex(32), "127.0.0.1", listener.getsockname()[1], self._config.control_header_limit_bytes, self._config.owner_envelope_limit_bytes, self._config.admission_message_limit_bytes, self._config.invocation_limit_bytes, self._config.result_limit_bytes, self._config.output_frame_limit_bytes, self._config.output_final_timeout)

    def _require_connection(self) -> _Connection:
        """Return the started connection or reject uninitialized backend use."""
        with self._lock:
            if self._connection is None or self._connection.sdk is None:
                raise ExecutionError("Ray backend is not initialized")
            return self._connection

    def _generation(self) -> str:
        """Return this driver's unique lifecycle generation within its cluster authority."""
        return self._require_connection().generation

    def _limits(self) -> dict[FrameType, int]:
        """Build effective typed limits for common socket frame validation."""
        return {FrameType.CONTROL: self._config.control_header_limit_bytes, FrameType.OWNER: self._config.owner_envelope_limit_bytes, FrameType.PAYLOAD: self._config.invocation_limit_bytes, FrameType.OUTPUT: self._config.output_frame_limit_bytes, FrameType.RESULT: self._config.result_limit_bytes, FrameType.ERROR: self._config.result_limit_bytes}

    def _conversation(self, descriptor: BootstrapDescriptor) -> ProtocolConversation:
        """Return a correlation-locked common protocol conversation."""
        return ProtocolConversation(descriptor.correlation, header_limit=descriptor.control_header_limit_bytes, invocation_limit=descriptor.invocation_limit_bytes, result_limit=descriptor.result_limit_bytes, output_limit=descriptor.output_frame_limit_bytes, owner_limit=descriptor.owner_envelope_limit_bytes, admission_limit=descriptor.admission_message_limit_bytes)

    @staticmethod
    def _send(connection: socket.socket, data: bytes | tuple[bytes, memoryview]) -> None:
        """Write one validated common frame without logging private controls."""
        if isinstance(data, tuple):
            connection.sendall(data[0])
            connection.sendall(data[1])
        else:
            connection.sendall(data)

    def _send_stop(self, run: _Run, descriptor: BootstrapDescriptor) -> None:
        """Best-effort rejection before payload transfer."""
        if run.connection is not None:
            try:
                self._send(run.connection, encode_control(FrameState.STOP, descriptor.correlation, {"reason": "rejected"}, header_limit=self._config.control_header_limit_bytes))
            except OSError:
                pass


def _local_node(sdk: _RaySDK) -> str:
    """Require exactly one alive same-host node before a task can be submitted."""
    alive = [node for node in sdk.nodes() if node.get("Alive") is True]
    if len(alive) != 1:
        raise ExecutionError("Ray backend requires exactly one alive local node")
    node = alive[0]
    node_id = node.get("NodeID")
    host = node.get("NodeManagerAddress")
    if not isinstance(node_id, str) or not node_id or not isinstance(host, str) or not _same_host(host):
        raise ExecutionError("Ray deployment is not proven to be same-host")
    return node_id


def _same_host(host: str) -> bool:
    """Require direct local-interface evidence for every resolved node address.

    Hostname lookup alone is not locality evidence in containers: it commonly
    yields only loopback while the Ray node advertises a bridged interface. A UDP
    connect selects the actual local source interface without sending traffic; a
    matching source and destination address is bounded proof that the advertised
    address belongs to this network namespace.
    """
    try:
        remote = {item[4][0] for item in socket.getaddrinfo(host, 9, type=socket.SOCK_DGRAM)}
    except OSError:
        return False
    for address in tuple(remote)[:8]:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        probe = socket.socket(family, socket.SOCK_DGRAM)
        try:
            probe.settimeout(0.2)
            probe.connect((address, 9))
            if probe.getsockname()[0].split("%", 1)[0] == address.split("%", 1)[0]:
                return True
        except OSError:
            pass
        finally:
            probe.close()
    return False


def _resource_amounts(values: Mapping[str, object]) -> ResourceAmounts:
    """Convert public Ray logical-resource mappings without inventing device IDs."""
    if not isinstance(values, Mapping):
        return ResourceAmounts(None, None, {}, {})
    cpu = values.get("CPU")
    memory = values.get("memory")
    accelerators = {"gpu": float(value) for key, value in values.items() if isinstance(key, str) and key.lower() == "gpu" and isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0}
    named = {key: float(value) for key, value in values.items() if isinstance(key, str) and key not in {"CPU", "memory"} and key.lower() != "gpu" and isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0}
    return ResourceAmounts(float(cpu) if isinstance(cpu, (int, float)) and not isinstance(cpu, bool) and cpu >= 0 else None, int(memory) if isinstance(memory, (int, float)) and not isinstance(memory, bool) and memory >= 0 else None, accelerators, named)


def _requested_amounts(world: Any) -> ResourceAmounts:
    """Summarize supported single-task Ray logical requests before scheduler grant."""
    if world is None:
        # Ray's native default task request is one CPU even when no Execute world
        # was supplied.  Preserve unknown memory rather than inventing a zero grant.
        return ResourceAmounts(1.0, None, {}, {})
    if len(world.roles) != 1:
        raise AdmissionError("Ray supports one Execute role per task")
    role = next(iter(world.roles.values()))
    if not role.replicas.satisfied_by(1) or role.topology or role.resources.devices or role.resources.named or role.resources.accelerator_memory:
        raise AdmissionError("Ray world requirement contains unsupported exact controls")
    resources = role.resources
    requested_cpus = max(1, resources.cpus.min or 0)
    if not resources.cpus.satisfied_by(requested_cpus):
        raise AdmissionError("Ray CPU constraint is incompatible with Ray's one-CPU floor")
    accelerators: dict[str, float] = {}
    for key, constraint in resources.accelerators.items():
        if key.lower() != "gpu":
            raise AdmissionError("Ray supports only native GPU accelerator requirements")
        requested = float(constraint.min or 0)
        if constraint.max is not None and requested > constraint.max:
            raise AdmissionError("Ray GPU requirement is internally incompatible")
        accelerators["gpu"] = requested
    memory = resources.memory.min
    if memory is not None and resources.memory.max is not None and memory > resources.memory.max:
        raise AdmissionError("Ray memory requirement is internally incompatible")
    return ResourceAmounts(float(requested_cpus), memory, accelerators, {})


def _native_options(world: Any) -> dict[str, object]:
    """Translate only supported logical requests into exact Ray task options."""
    if world is None:
        return {}
    requested = _requested_amounts(world)
    options: dict[str, object] = {}
    if requested.cpus:
        options["num_cpus"] = requested.cpus
    if requested.accelerators.get("gpu"):
        options["num_gpus"] = requested.accelerators["gpu"]
    if requested.memory_bytes:
        options["memory"] = requested.memory_bytes
    return options


def _logical_world(world: Any, native: Mapping[str, object]) -> tuple[WorldSpec | None, dict[str, str]]:
    """Build an owner-checkable shape from one worker's raw Ray grant evidence."""
    if world is None:
        return None, {}
    assigned = native.get("assigned_resources")
    accelerator_ids = native.get("accelerator_ids")
    if not isinstance(assigned, Mapping) or not isinstance(accelerator_ids, Mapping):
        return None, {}
    amounts = _resource_amounts(assigned)
    gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("gpu"))
    controls = _logical_controls(amounts, gpu_ids)
    return _world_from_logical_amounts(world, amounts), controls


def _logical_plan(environment: EnvironmentRequirement | None, world: Any, record: EnvironmentRecord | None, amounts: ResourceAmounts, deadline: float):
    """Evaluate a query-only feasible Ray logical plan without reserving capacity."""
    observed = _world_from_logical_amounts(world, amounts)
    controls = _logical_controls(amounts, None, query=True)
    return _admit_observed_logical(
        environment=environment, world=world, record=record,
        observed_world=observed, controls=controls, deadline=deadline,
    )


def _world_from_logical_amounts(requirement: Any, amounts: ResourceAmounts) -> WorldSpec | None:
    """Represent one observed logical worker without fabricating exact bindings."""
    if requirement is None or not hasattr(requirement, "roles") or len(requirement.roles) != 1:
        return None
    cpu = _whole_amount(amounts.cpus)
    memory = _whole_amount(amounts.memory_bytes)
    gpu = _whole_amount(amounts.accelerators.get("gpu"))
    accelerators = {} if gpu is None else {"gpu": gpu}
    role_name = next(iter(requirement.roles))
    return WorldSpec(
        {role_name: RoleSpec(1, ProcessSpec(ResourceSpec(cpus=cpu or 0, memory=memory, accelerators=accelerators)))},
        backend={"kind": "ray", "logical_grant": True},
    )


def _logical_controls(amounts: ResourceAmounts, gpu_ids: object, *, query: bool = False) -> dict[str, str]:
    """Return only resource controls supported by the supplied raw evidence."""
    controls: dict[str, str] = {}
    if _whole_amount(amounts.cpus) is not None:
        controls["cpus"] = "logical"
    if _whole_amount(amounts.memory_bytes) is not None:
        controls["memory"] = "logical"
    gpu = _whole_amount(amounts.accelerators.get("gpu"))
    if gpu is not None and (query or isinstance(gpu_ids, (list, tuple)) and not isinstance(gpu_ids, (str, bytes)) and len(gpu_ids) == gpu):
        controls["accelerators"] = "logical"
    return controls


def _whole_amount(value: object) -> int | None:
    """Return a nonnegative integral logical amount or preserve unavailable evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or int(value) != value:
        return None
    return int(value)


def _amounts_cover(total: ResourceAmounts, requested: ResourceAmounts) -> bool:
    """Fail closed when a requested Ray resource has unknown or insufficient total capacity."""
    def covers(available: int | float | None, required: int | float | None) -> bool:
        return required in (None, 0) or available is not None and available >= required

    return (
        covers(total.cpus, requested.cpus)
        and covers(total.memory_bytes, requested.memory_bytes)
        and all(covers(total.accelerators.get(name), amount) for name, amount in requested.accelerators.items())
        and all(covers(total.named.get(name), amount) for name, amount in requested.named.items())
    )


def _compatible_runtime(spec: object) -> bool:
    """Limit Ray runtime selections to existing Conda strings or venv executables."""
    return isinstance(spec, (CondaEnvironmentSpec, PythonExecutableSpec, CurrentEnvironmentSpec))


def _reference_id(reference: Any) -> str | None:
    """Return a bounded diagnostic handle without relying on it for release proof."""
    value = str(reference)
    return value[:128] if value else None


def _worker_lifetime_ended(run: _Run) -> bool:
    """Verify that the HELLO-qualified worker lifetime no longer exists.

    Ray's task cancellation exception identifies only the exact ObjectRef. It does
    not establish that the corresponding worker process ended, and PID reuse makes
    a PID-only check unsafe. The worker supplies its psutil creation time in the
    native HELLO record; absent evidence deliberately leaves cancellation
    unconfirmed.
    """
    if run.worker_pid is None or run.worker_create_time is None:
        return False
    try:
        import psutil
    except ImportError:
        return False
    try:
        return psutil.Process(run.worker_pid).create_time() != run.worker_create_time
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False


def _terminal_task_error(error: BaseException) -> bool:
    """Recognize only public Ray exceptions that prove this ObjectRef is terminal."""
    return type(error).__name__ in {
        "RayTaskError", "TaskCancelledError", "TaskUnschedulableError", "WorkerCrashedError",
    }


def _native_error(prefix: str, error: BaseException, limit: int) -> str:
    """Render only a configured-bounded native exception type for diagnostics."""
    value = f"{prefix} ({type(error).__name__})"
    return value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _owner_json(data: Mapping[str, Any], limit: int) -> bytes:
    """Encode one owner requirement under its own configured bound."""
    payload = canonical_json_bytes(data, max_depth=32, max_nodes=65536, max_entries=4096, max_string=limit, max_int_bits=64)
    if len(payload) > limit:
        raise AdmissionError("owner requirement exceeds configured transport limit")
    return payload


def _json(data: bytes, limit: int) -> dict[str, Any]:
    """Decode trusted bounded common control data without retaining raw payload text."""
    value = canonical_json_load_bytes(data, max_depth=32, max_nodes=65536, max_entries=4096, max_string=limit, max_int_bits=64)
    if not isinstance(value, Mapping):
        raise FrameError("Ray worker control payload is not a mapping")
    return dict(value)


__all__ = ["RayBackend", "RayBackendConfig", "RayFuture"]
