"""Bounded coordinator-owned invocation and result spools for Execute.

Workers use bounded memory buffers only.  This module owns the two reserved
coordinator files and never launches a worker or imports DRYML core.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import inspect
import io
from multiprocessing.connection import Connection
import os
import platform
import socket
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import BuiltinFunctionType, CoroutineType, GeneratorType, ModuleType
from typing import TYPE_CHECKING

import dill

from .errors import CleanupError, ExecutionError
from .models import PayloadSpool, ResultSpool

if TYPE_CHECKING:
    from .config import BackendConfig


_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))
_EXECUTE_RESOURCE_BASES = frozenset({
    "Backend", "ExecutionFuture", "ExecutionOutput", "Executor", "ExecutorView",
})


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Expose immutable process-wide quota counters for inspection and tests."""

    generation: int | None
    limit_bytes: int
    file_limit: int
    preflight_limit: int
    reserved_bytes: int
    reserved_files: int
    active_preflights: int
    leases: int


@dataclass(slots=True)
class _BudgetState:
    """Hold one generation's mutable counters behind ``SpoolBudget._lock``."""

    generation: int
    quota: tuple[int, int, int]
    leases: int = 0
    reserved_bytes: int = 0
    reserved_files: int = 0
    active_preflights: int = 0


class SpoolReservation:
    """Own one generation-qualified invocation/result capacity reservation."""

    def __init__(self, lease: "SpoolLease", identity: str, invocation_bytes: int, result_bytes: int) -> None:
        self._lease = lease
        self.identity = identity
        self._invocation_bytes = invocation_bytes
        self._result_bytes = result_bytes
        self._released = False
        self._preflight_released = False
        self._invocation_refunded = False
        self._result_refunded = False

    @property
    def reserved_bytes(self) -> int:
        """Return this reservation's current aggregate byte charge."""
        return self._invocation_bytes + self._result_bytes

    def refund_invocation(self, actual_bytes: int) -> None:
        """Refund unused invocation capacity once after immutable publication."""
        self._refund("invocation", actual_bytes)

    def refund_result(self, actual_bytes: int) -> None:
        """Refund unused result capacity once after immutable result publication."""
        self._refund("result", actual_bytes)

    def _refund(self, kind: str, actual_bytes: int) -> None:
        with SpoolBudget._lock:
            state = self._state()
            allocated = self._invocation_bytes if kind == "invocation" else self._result_bytes
            refunded = self._invocation_refunded if kind == "invocation" else self._result_refunded
            if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int) or not 0 <= actual_bytes <= allocated:
                raise ExecutionError(f"invalid {kind} refund")
            if refunded:
                raise ExecutionError(f"{kind} capacity was already refunded")
            state.reserved_bytes -= allocated - actual_bytes
            if kind == "invocation":
                self._invocation_bytes = actual_bytes
                self._invocation_refunded = True
            else:
                self._result_bytes = actual_bytes
                self._result_refunded = True

    def release_preflight(self) -> None:
        """Release this producer's preflight slot once it stops serializing."""
        with SpoolBudget._lock:
            state = self._state()
            if not self._preflight_released:
                state.active_preflights -= 1
                self._preflight_released = True

    def release(self) -> None:
        """Release all charges exactly once after owned files are disposed."""
        with SpoolBudget._lock:
            state = self._state()
            state.reserved_bytes -= self.reserved_bytes
            state.reserved_files -= 2
            if not self._preflight_released:
                state.active_preflights -= 1
                self._preflight_released = True
            self._released = True
            self._lease._reservations.discard(self)

    def _state(self) -> _BudgetState:
        """Return this reservation's matching generation or fail closed."""
        if self._released:
            raise ExecutionError("spool reservation is already released")
        state = SpoolBudget._state
        if state is None or state.generation != self._lease.generation:
            raise ExecutionError("spool reservation belongs to a stale generation")
        return state


class SpoolLease:
    """Keep an executor's matching quota generation active through cleanup."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self._released = False
        self._reservations: set[SpoolReservation] = set()

    def reserve(self, identity: str, *, invocation_bytes: int, result_bytes: int) -> SpoolReservation:
        """Atomically reserve two files, bytes, and one active preflight slot."""
        if not isinstance(identity, str) or not identity:
            raise ExecutionError("spool reservation identity is invalid")
        for name, value in (("invocation_bytes", invocation_bytes), ("result_bytes", result_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExecutionError(f"{name} is invalid")
        with SpoolBudget._lock:
            if self._released:
                raise ExecutionError("spool lease is released")
            state = SpoolBudget._state
            if state is None or state.generation != self.generation:
                raise ExecutionError("spool lease belongs to a stale generation")
            total = invocation_bytes + result_bytes
            byte_limit, file_limit, preflight_limit = state.quota
            if state.reserved_bytes + total > byte_limit or state.reserved_files + 2 > file_limit or state.active_preflights + 1 > preflight_limit:
                raise ExecutionError("spool_capacity_exhausted")
            state.reserved_bytes += total
            state.reserved_files += 2
            state.active_preflights += 1
            reservation = SpoolReservation(self, identity, invocation_bytes, result_bytes)
            self._reservations.add(reservation)
            return reservation

    def release(self) -> None:
        """Release this lease after all of its reservations have been disposed."""
        with SpoolBudget._lock:
            if self._released:
                raise ExecutionError("spool lease is already released")
            if self._reservations:
                raise ExecutionError("spool lease still owns reservations")
            state = SpoolBudget._state
            if state is None or state.generation != self.generation:
                raise ExecutionError("spool lease belongs to a stale generation")
            state.leases -= 1
            self._released = True
            if state.leases == 0:
                if state.reserved_bytes or state.reserved_files or state.active_preflights:
                    raise ExecutionError("spool budget has unreconciled charges")
                SpoolBudget._state = None


class SpoolBudget:
    """Coordinate one in-memory aggregate quota generation for local Execute."""

    _lock = threading.Lock()
    _state: _BudgetState | None = None
    _next_generation = 1

    @classmethod
    def acquire(cls, config: "BackendConfig") -> SpoolLease:
        """Join a compatible quota generation without serialization or disk I/O."""
        requested = (config.spool_limit_bytes, config.spool_file_limit, config.preflight_limit)
        with cls._lock:
            if cls._state is None:
                cls._state = _BudgetState(cls._next_generation, requested)
                cls._next_generation += 1
            if cls._state.quota != requested:
                names = ("spool_limit_bytes", "spool_file_limit", "preflight_limit")
                differences = ", ".join(f"{name}: active={active}, requested={wanted}" for name, active, wanted in zip(names, cls._state.quota, requested) if active != wanted)
                raise ExecutionError(f"spool_configuration_conflict ({differences})")
            cls._state.leases += 1
            return SpoolLease(cls._state.generation)

    @classmethod
    def snapshot(cls) -> BudgetSnapshot:
        """Return current counters without reserving capacity or changing a lease."""
        with cls._lock:
            if cls._state is None:
                return BudgetSnapshot(None, 0, 0, 0, 0, 0, 0, 0)
            state = cls._state
            return BudgetSnapshot(state.generation, state.quota[0], state.quota[1], state.quota[2], state.reserved_bytes, state.reserved_files, state.active_preflights, state.leases)


def _reject_live_resource(value: object) -> None:
    """Reject a live or semantic object at the point dill actually visits it."""
    value_type = type(value)
    if any(cls.__module__ == "dryml.core" or cls.__module__.startswith("dryml.core.") for cls in value_type.__mro__):
        raise TypeError("live resource: core semantic values are unsupported by Execute transport")
    if isinstance(value, (io.IOBase, _LOCK_TYPES, GeneratorType, CoroutineType, socket.socket, threading.Thread, concurrent.futures.Executor, concurrent.futures.Future)):
        raise TypeError(f"live resource: {value_type.__name__} is unsupported by Execute transport")
    if isinstance(value, Connection):
        raise TypeError("live resource: Connection is unsupported by Execute transport")
    if any(cls.__module__.startswith("dryml.execute") and cls.__name__ in _EXECUTE_RESOURCE_BASES for cls in value_type.__mro__):
        raise TypeError(f"live resource: {value_type.__name__} is unsupported by Execute transport")


class _CheckedPickler(dill.Pickler):
    """Apply resource rejection to every object dill chooses to serialize."""

    def persistent_id(self, obj: object) -> object | None:
        """Check the actual pickler graph without manually guessing globals/attrs."""
        _reject_live_resource(obj)
        return None


class _BoundedWriter(io.RawIOBase):
    """Accept dill writes only until an explicit maximum is reached."""

    def __init__(self, limit: int, label: str) -> None:
        self._limit = limit
        self._label = label
        self._buffer = bytearray()

    def writable(self) -> bool:
        """Declare this in-memory serializer sink writable."""
        return True

    def write(self, data: bytes) -> int:
        """Append one dill chunk or reject an oversized serialization."""
        if len(self._buffer) + len(data) > self._limit:
            raise ValueError(f"serialized {self._label} exceeds {self._label} limit")
        self._buffer.extend(data)
        return len(data)

    def bytes(self) -> bytes:
        """Return the complete bounded serialization after dill finishes."""
        return bytes(self._buffer)


def _serialize(value: object, *, limit_bytes: int, label: str, checked: bool) -> bytes:
    if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes <= 0:
        raise ValueError(f"{label} limit must be positive")
    writer = _BoundedWriter(limit_bytes, label)
    try:
        pickler = _CheckedPickler(writer, protocol=5, byref=False, recurse=True) if checked else dill.Pickler(writer, protocol=5, byref=False, recurse=True)
        pickler.dump(value)
    except ValueError:
        raise
    except TypeError as exc:
        if str(exc).startswith("live resource:"):
            raise
        raise TypeError(f"{label} graph cannot be serialized for Execute transport") from exc
    except Exception as exc:
        raise TypeError(f"{label} graph cannot be serialized for Execute transport") from exc
    return writer.bytes()


def serialize_call(fn: object, args: tuple[object, ...], kwargs: dict[str, object], *, limit_bytes: int) -> bytes:
    """Snapshot one supported callable and its complete argument graph with dill.

    Functions and importable builtins are supported.  Stateful bound methods and
    callable instances are rejected before serialization; resource checks then
    run inside the pickler over defaults, closures, globals, and object attrs
    that dill actually serializes.
    """
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        raise TypeError("call arguments must be a tuple and keyword dictionary")
    if not all(isinstance(key, str) for key in kwargs):
        raise TypeError("call keyword keys must be strings")
    if inspect.ismethod(fn) and getattr(fn, "__self__", None) is not None:
        raise TypeError("stateful bound methods are unsupported by Execute transport")
    if isinstance(fn, BuiltinFunctionType) and getattr(fn, "__self__", None) is not None and not isinstance(fn.__self__, ModuleType):
        raise TypeError("bound builtin methods are unsupported by Execute transport")
    if not isinstance(fn, (BuiltinFunctionType,)) and not inspect.isfunction(fn):
        if callable(fn):
            raise TypeError("callable instances are unsupported by Execute transport")
        raise TypeError("callable root is required")
    return _serialize((fn, args, kwargs), limit_bytes=limit_bytes, label="invocation", checked=True)


def deserialize_call(payload: bytes, *, limit_bytes: int) -> tuple[object, tuple[object, ...], dict[str, object]]:
    """Load a bounded trusted call graph after validating its declared memory cap."""
    if not isinstance(payload, bytes) or len(payload) > limit_bytes:
        raise ValueError("serialized invocation exceeds invocation limit")
    try:
        fn, args, kwargs = dill.loads(payload)
    except Exception as exc:
        raise TypeError("call graph cannot be deserialized for Execute transport") from exc
    if not callable(fn) or not isinstance(args, tuple) or not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
        raise TypeError("serialized call has an invalid shape")
    return fn, args, kwargs


def serialize_result(value: object, *, limit_bytes: int) -> bytes:
    """Serialize one worker result into a bounded in-memory coordinator payload."""
    return _serialize(value, limit_bytes=limit_bytes, label="result", checked=True)


def deserialize_result(payload: bytes, *, limit_bytes: int) -> object:
    """Load one trusted bounded result payload without creating a worker spool."""
    if not isinstance(payload, bytes) or len(payload) > limit_bytes:
        raise ValueError("serialized result exceeds result limit")
    try:
        return dill.loads(payload)
    except Exception as exc:
        raise TypeError("result graph cannot be deserialized for Execute transport") from exc


def _validate_descriptor(payload: PayloadSpool | ResultSpool, data: bytes, *, limit_bytes: int, kind: str) -> None:
    if not isinstance(payload, (PayloadSpool, ResultSpool)) or not isinstance(data, bytes):
        raise TypeError("payload descriptor and bytes are required")
    if payload.serializer != "dill" or payload.serializer_version != dill.__version__ or payload.pickle_protocol != 5:
        raise TypeError("payload serializer identity is unsupported")
    if payload.python_implementation != platform.python_implementation() or payload.python_version[:2] != sys.version_info[:2]:
        raise TypeError("payload Python identity is unsupported")
    if len(data) > limit_bytes:
        raise ValueError(f"serialized {kind} exceeds {kind} limit")
    if payload.size_bytes != len(data):
        raise ValueError("payload length does not match descriptor")
    if not hmac.compare_digest(payload.sha256, hashlib.sha256(data).hexdigest()):
        raise ValueError("payload digest does not match descriptor")


def validate_payload(payload: PayloadSpool, data: bytes, *, limit_bytes: int) -> None:
    """Validate an invocation spool descriptor's codec identity, length, and digest."""
    _validate_descriptor(payload, data, limit_bytes=limit_bytes, kind="invocation")


def validate_result(payload: ResultSpool, data: bytes, *, limit_bytes: int) -> None:
    """Validate a result spool descriptor's codec identity, length, and digest."""
    _validate_descriptor(payload, data, limit_bytes=limit_bytes, kind="result")


@dataclass(slots=True)
class _OwnedSpool:
    """Keep exact known paths and descriptors for one reservation's cleanup."""

    child: Path
    invocation_path: Path
    result_path: Path
    payload: PayloadSpool | None = None
    result: ResultSpool | None = None
    accepted: bool = False
    cleanup_pending: bool = False
    disposed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class PayloadSpooler:
    """Publish, receive, and reconcile two bounded files inside one owned child."""

    def __init__(self, config: "BackendConfig", lease: SpoolLease, *, parent: Path | None = None) -> None:
        """Capture the selected spool parent once without inspecting its existence."""
        self._config = config
        self._lease = lease
        configured = config.spool_directory
        self._parent = parent if parent is not None else (Path(tempfile.gettempdir()) if configured is None else (Path.cwd() / configured if not configured.is_absolute() else configured))
        self._owned: dict[SpoolReservation, _OwnedSpool] = {}
        self._lock = threading.RLock()

    def snapshot(self, fn: object, args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[PayloadSpool, SpoolReservation]:
        """Reserve, serialize, and atomically publish one immutable invocation spool.

        Failed pre-acceptance cleanup remains recoverable through this owner's
        :meth:`reconcile_cleanup`; public errors expose no internal owner.
        """
        reservation = self._lease.reserve(uuid.uuid4().hex, invocation_bytes=self._config.invocation_limit_bytes, result_bytes=self._config.result_limit_bytes)
        published = False
        try:
            if not self._parent.exists() or not self._parent.is_dir():
                raise ValueError("spool_directory must be an existing directory")
            child = self._parent / f"dryml-execute-{uuid.uuid4().hex}"
            owned = _OwnedSpool(child, child / "invocation.dill", child / "result.dill")
            with self._lock:
                self._owned[reservation] = owned
            # Register the cleanup identity before mkdir so interruption after a
            # successful directory creation cannot orphan its reservation.
            child.mkdir(mode=0o700)
            data = serialize_call(fn, args, kwargs, limit_bytes=self._config.invocation_limit_bytes)
            self._write_new(owned.invocation_path, data, "invocation")
            reservation.refund_invocation(len(data))
            payload = self._descriptor(PayloadSpool, owned.invocation_path, data)
            with self._lock:
                owned.payload = payload
            published = True
            return payload, reservation
        except BaseException as exc:
            try:
                with self._lock:
                    tracked = reservation in self._owned
                if tracked:
                    self._dispose_child(reservation, expected_payload=None)
                else:
                    reservation.release()
            except CleanupError as cleanup_error:
                raise CleanupError(str(cleanup_error)) from exc
            raise
        finally:
            if not published and not reservation._released:
                reservation.release_preflight()

    def accept(self, payload: PayloadSpool, reservation: SpoolReservation) -> None:
        """Transfer one exact preflight spool to an accepted submission.

        The preflight slot remains charged until this transfer or a rejected
        disposal, so a completed snapshot cannot silently outlive executor close.
        Repeating the same transfer is idempotent; another payload or spooler is
        rejected rather than releasing a foreign reservation.
        """
        if not isinstance(payload, PayloadSpool):
            raise TypeError("payload must be a PayloadSpool")
        owned = self._owned_for(reservation)
        with owned.lock:
            if owned.payload != payload:
                raise ExecutionError("payload does not match spool reservation")
            if owned.disposed:
                raise ExecutionError("spool reservation is already disposed")
            if owned.cleanup_pending:
                raise CleanupError("unable to accept spool pending cleanup")
            if not owned.accepted:
                reservation.release_preflight()
                owned.accepted = True

    def receive_result(self, reservation: SpoolReservation, data: bytes) -> ResultSpool:
        """Write validated bounded result bytes into the reservation's result slot."""
        if not isinstance(data, bytes) or len(data) > self._config.result_limit_bytes:
            raise ValueError("serialized result exceeds result limit")
        owned = self._owned_for(reservation)
        with owned.lock:
            if owned.disposed or owned.payload is None or not owned.accepted or owned.cleanup_pending:
                raise ExecutionError("spool reservation is not an active invocation")
            if owned.result is not None:
                raise ExecutionError("spool reservation already has a result")
            try:
                self._write_new(owned.result_path, data, "result")
                result = self._descriptor(ResultSpool, owned.result_path, data)
                owned.result = result
                # Keep the operation lock through the refund so disposal cannot
                # release this reservation between publication and accounting.
                reservation.refund_result(len(data))
                return result
            except BaseException:
                owned.cleanup_pending = True
                raise

    def dispose(self, payload: PayloadSpool, reservation: SpoolReservation) -> None:
        """Remove this exact invocation/result group, then release its charge."""
        if not isinstance(payload, PayloadSpool):
            raise TypeError("payload must be a PayloadSpool")
        self._dispose_child(reservation, expected_payload=payload)

    def reconcile_cleanup(self) -> None:
        """Retry only failed or rejected cleanup without touching active spools."""
        with self._lock:
            reservations = tuple(reservation for reservation, owned in self._owned.items() if owned.cleanup_pending)
        failures: list[BaseException] = []
        for reservation in reservations:
            try:
                self._dispose_child(reservation, expected_payload=None)
            except CleanupError as exc:
                failures.append(exc)
        if failures:
            raise CleanupError("unable to dispose Execute-owned spool storage") from failures[0]

    def _write_new(self, path: Path, data: bytes, label: str) -> None:
        """Publish one known file completely before exposing its descriptor."""
        with path.open("xb") as stream:
            written = stream.write(data)
            if written != len(data):
                raise OSError(f"short write while publishing {label} spool")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _descriptor(descriptor_type: type[PayloadSpool] | type[ResultSpool], path: Path, data: bytes) -> PayloadSpool | ResultSpool:
        """Build the complete local dill descriptor for a fully written file."""
        return descriptor_type(path, len(data), hashlib.sha256(data).hexdigest(), "dill", dill.__version__, platform.python_implementation(), sys.version_info[:3], 5)

    def _dispose_child(self, reservation: SpoolReservation, *, expected_payload: PayloadSpool | None) -> None:
        """Dispose only tracked file paths; unrelated child contents retain charge."""
        try:
            owned = self._owned_for(reservation)
        except ExecutionError:
            if expected_payload is not None and reservation._released:
                return
            raise
        with owned.lock:
            if expected_payload is not None and owned.payload != expected_payload:
                raise ExecutionError("payload does not match spool reservation")
            if owned.disposed:
                return
            try:
                for path in (owned.invocation_path, owned.result_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    owned.child.rmdir()
                except FileNotFoundError:
                    pass
            except BaseException as exc:
                owned.cleanup_pending = True
                raise CleanupError("unable to dispose Execute-owned spool storage") from exc
            with self._lock:
                del self._owned[reservation]
            reservation.release()
            owned.disposed = True

    def _owned_for(self, reservation: SpoolReservation) -> _OwnedSpool:
        """Return this spooler's active owner without holding a global I/O lock."""
        with self._lock:
            owned = self._owned.get(reservation)
        if owned is None:
            raise ExecutionError("spool reservation is not owned by this spooler")
        return owned


__all__ = ["BudgetSnapshot", "PayloadSpooler", "SpoolBudget", "SpoolLease", "SpoolReservation", "deserialize_call", "deserialize_result", "serialize_call", "serialize_result", "validate_payload", "validate_result"]
