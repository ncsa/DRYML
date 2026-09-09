"""Proof that the private worker bootstrap remains descriptor-only."""

from __future__ import annotations

import inspect

from dryml.execute import _worker
from dryml.execute._protocol import PROTOCOL_VERSION


def test_worker_bootstrap_imports_the_execute_protocol_marker_directly():
    """Launchability depends on the source bootstrap marker, not package metadata."""
    source = inspect.getsource(_worker)
    assert "deserialize_call(payload" in source
    assert "go_frame.state" in source
    assert "dryml.execute.v0.3" not in source
    assert PROTOCOL_VERSION == 1
