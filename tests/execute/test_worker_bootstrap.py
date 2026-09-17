"""Proof that the private worker bootstrap remains descriptor-only."""

from __future__ import annotations

import inspect

import pytest

from dryml.execute import _worker
from dryml.execute._protocol import PROTOCOL_VERSION


def test_worker_bootstrap_imports_the_execute_protocol_marker_directly():
    """Launchability depends on the source bootstrap marker, not package metadata."""
    source = inspect.getsource(_worker)
    assert "deserialize_call(payload" in source
    assert "go_frame.state" in source
    assert "dryml.execute.v0.3" not in source
    assert PROTOCOL_VERSION == 2


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
