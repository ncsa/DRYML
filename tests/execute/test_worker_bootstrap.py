"""Proof that the private worker bootstrap remains descriptor-only."""

from __future__ import annotations

import inspect
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dryml.execute import _worker
from dryml.execute._protocol import (
    PROTOCOL_VERSION,
    WORKER_PROTOCOL_ID,
    BootstrapDescriptor,
    Correlation,
    FrameState,
    FrameType,
    decode_exact_frame,
    decode_worker_error,
    encode_frame,
)


class _Connection:
    """Collect complete worker frames without opening a socket."""

    def __init__(self):
        self.frames = []

    def sendall(self, data):
        self.frames.append(bytes(data))


class _Thread:
    """Avoid descriptor drainer activity in direct invocation tests."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def _invoke_at(monkeypatch, clock, *, setup=None, manager=None):
    """Invoke the worker with inert descriptors and return its terminal frame."""
    correlation = Correlation("worker-deadline", 0, 1)
    descriptor = BootstrapDescriptor(
        correlation, "token", "127.0.0.1", 43123,
        1024, 1024, 4096, 4096, 4096, 1024,
    )
    connection = _Connection()
    monkeypatch.setattr(_worker.os, "dup", lambda fd: fd)
    monkeypatch.setattr(_worker.os, "pipe", lambda: (10, 11))
    monkeypatch.setattr(_worker.os, "dup2", lambda source, target: None)
    monkeypatch.setattr(_worker.os, "close", lambda fd: None)
    monkeypatch.setattr(_worker, "Thread", _Thread, raising=False)
    monkeypatch.setattr(_worker.threading, "Thread", _Thread)
    monkeypatch.setattr(_worker, "_flush_standard_streams", lambda: None)
    monkeypatch.setattr(_worker.time, "monotonic", lambda: clock[0])
    reader = None
    conversation = None
    if setup is not None:
        monkeypatch.setattr(_worker, "_resolve_setup", lambda value: manager)
        payload = encode_frame(
            FrameState.PAYLOAD, FrameType.PAYLOAD, correlation, b"payload",
            header_limit=1024,
        )
        frame = decode_exact_frame(payload, header_limit=1024, payload_limit=4096)
        reader = SimpleNamespace(read=lambda: frame)
        conversation = SimpleNamespace(accept=lambda value: value, accept_frame=lambda value: value)
    _worker._invoke(
        connection, Lock(), descriptor, None if setup is not None else b"payload", 1.0,
        setup=setup, reader=reader, conversation=conversation,
    )
    return decode_exact_frame(connection.frames[-1], header_limit=1024, payload_limit=4096)


def test_worker_bootstrap_imports_the_execute_protocol_marker_directly():
    """Launchability depends on the source bootstrap marker, not package metadata."""
    source = inspect.getsource(_worker)
    assert "deserialize_call(payload" in source
    assert "go_frame.state" in source
    assert "dryml.execute.v0.3" not in source
    assert PROTOCOL_VERSION == 4
    assert WORKER_PROTOCOL_ID == "dryml.execute.worker.v4"


def test_worker_marks_deadline_before_deserialization_without_loading_workload(monkeypatch):
    """An expired pre-deserialization check emits worker-owned deadline evidence."""
    deserialize = Mock(side_effect=AssertionError("payload must not be deserialized"))
    monkeypatch.setattr(_worker, "deserialize_call", deserialize)

    frame = _invoke_at(monkeypatch, [2.0])

    assert decode_worker_error(frame, limit_bytes=4096, setup=False) == (None, True, ())
    deserialize.assert_not_called()


def test_worker_marks_deadline_before_invocation_without_calling_workload(monkeypatch):
    """Expiry after decoding emits the marker and never invokes the callable."""
    clock = [0.0]
    workload = Mock(side_effect=AssertionError("workload must not be invoked"))

    def deserialize(payload, *, limit_bytes):
        clock[0] = 2.0
        return workload, (), {}

    monkeypatch.setattr(_worker, "deserialize_call", deserialize)
    frame = _invoke_at(monkeypatch, clock)

    assert decode_worker_error(frame, limit_bytes=4096, setup=False) == (None, True, ())
    workload.assert_not_called()


def test_callable_timeout_error_is_not_a_worker_deadline_marker(monkeypatch):
    """A callable's own timeout type remains an ordinary remote error."""
    def deserialize(payload, *, limit_bytes):
        def workload():
            raise TimeoutError("credential=dummy must not cross the channel")

        return workload, (), {}

    monkeypatch.setattr(_worker, "deserialize_call", deserialize)
    frame = _invoke_at(monkeypatch, [0.0])

    assert decode_worker_error(frame, limit_bytes=4096, setup=False) == (
        "TimeoutError", False, (),
    )
    assert b"credential" not in frame.payload


def test_setup_deadline_marker_preserves_cleanup_failure(monkeypatch):
    """Setup teardown evidence remains attached to a pre-invocation deadline."""
    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, error_type, error, traceback):
            assert error_type is not None
            raise RuntimeError("cleanup detail must not cross the channel")

    monkeypatch.setattr(
        _worker,
        "deserialize_call",
        Mock(side_effect=AssertionError("payload must not be deserialized")),
    )
    frame = _invoke_at(monkeypatch, [2.0], setup={}, manager=Manager())

    assert decode_worker_error(frame, limit_bytes=4096, setup=True) == (
        None, True, ("RuntimeError",),
    )
    assert b"cleanup detail" not in frame.payload


@pytest.mark.parametrize("result_limit, expected", ((257, 0), (276, 0), (280, 0)))
def test_setup_result_bytes_limit_rejects_tiny_transport_budgets(result_limit, expected):
    """Tiny setup terminals leave no serializable core-outcome byte capacity."""
    assert _worker.setup_result_bytes_limit(result_limit) == expected


@pytest.mark.parametrize("byte_length", (1, 255, 256, 65_535, 65_536))
def test_setup_result_bytes_limit_fits_serializer_opcode_boundaries(byte_length):
    """Reserved generic-result capacity fits dill bytes framing at every opcode boundary."""
    payload_limit = byte_length + 18
    result_limit = 256 + (payload_limit * 4 + 2) // 3

    usable = _worker.setup_result_bytes_limit(result_limit)

    assert usable == byte_length
    assert len(_worker.serialize_result(bytes(usable), limit_bytes=payload_limit)) <= payload_limit


def test_setup_result_bytes_limit_never_probes_proportional_to_configured_limit(monkeypatch):
    """A large configured result limit still uses only fixed-size serializer probes."""
    original = _worker.serialize_result
    probe_sizes = []

    def observe(value, *, limit_bytes):
        """Record only dummy bytes supplied to the serializer probe."""
        probe_sizes.append(len(value))
        return original(value, limit_bytes=limit_bytes)

    _worker._setup_result_serialization_overhead.cache_clear()
    monkeypatch.setattr(_worker, "serialize_result", observe)
    try:
        assert _worker.setup_result_bytes_limit(48 * 1024 * 1024) > 0
    finally:
        _worker._setup_result_serialization_overhead.cache_clear()

    assert probe_sizes
    assert max(probe_sizes) == 65_536
