"""Proof for the closed projected Dispatch probe transport."""

from __future__ import annotations

import contextvars
import time

import pytest

from dryml.code import capture_inspection
from dryml.dispatch import ProbeOptions
from dryml.dispatch._probe import run_probe
from dryml.dispatch._probe_protocol import (
    ProbeProtocolError,
    _decode_request,
    _result_from_data,
    _safe_diagnostics,
    _snapshot_data,
    _snapshot_from_data,
    build_request,
    decode_result,
    execute_probe,
)
from dryml.environments import EnvironmentRequirement
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from dryml.environments import (
    EnvironmentRequirementsKernel,
    req as environment_req,
)
from dryml.environments.kernel import _view_to_data as environment_view_data
from dryml.worlds import WorldRequirementsKernel, req as world_req
from dryml.worlds.kernel import _view_to_data as world_view_data


@environment_req(requirements=("demo>=1",), source="root")
@world_req(cpus=1, source="root")
def _root() -> None:
    """Reach one declared helper for transport parity proof."""

    _helper()


@environment_req(tags=("helper",), source="helper")
def _helper() -> None:
    """Supply a distinct requirement declaration to static discovery."""


def _request():
    """Build one projected request without a Store or user transport."""

    capture = capture_inspection(_root)
    environment = EnvironmentRequirementsKernel._from_capture(capture)
    world = WorldRequirementsKernel._from_capture(capture)
    return build_request(
        capture.target,
        environment_view=environment_view_data(environment._view),
        world_view=world_view_data(world._view),
        max_targets=256,
        max_depth=32,
    )


def test_fixed_protocol_round_trips_projected_owner_results() -> None:
    """The worker consumes canonical projections and normal kernel outcomes."""

    request = _request()
    result = decode_result(execute_probe(request.data), request)

    assert result.complete
    assert result.environment.value.requirements == ("demo>=1",)
    assert result.environment.value.tags == ("helper",)
    assert result.world.value is not None
    text = request.data.decode("utf-8")
    assert "filename" not in text
    assert "arguments" not in text


def test_protocol_rejects_digest_root_and_oversized_transport() -> None:
    """Malformed transport never becomes an empty incomplete domain result."""

    request = _request()
    result = execute_probe(request.data)
    with pytest.raises(ProbeProtocolError):
        execute_probe(request.data.replace(b'"digest":"', b'"digest":"0', 1))
    with pytest.raises(ProbeProtocolError):
        decode_result(
            result.replace(b'"root_id":"t0000"', b'"root_id":"wrong"'), request
        )
    with pytest.raises(ProbeProtocolError):
        decode_result(b"x" * (4 * 1024 * 1024 + 1), request)


def test_inline_probe_uses_fresh_context_and_the_scheduler_kernel_dag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context-local defaults cannot leak into a static kernel call."""

    poisoned = contextvars.ContextVar("probe-poison", default="clean")
    observed: list[str] = []
    from dryml.code.static_dependencies import StaticDependenciesKernel

    original = StaticDependenciesKernel.run

    def instrument(self, *args, **kwargs):
        observed.append(poisoned.get())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(StaticDependenciesKernel, "run", instrument)
    token = poisoned.set("poisoned")
    try:
        result = run_probe(_root, options=ProbeOptions(placement="in_process"))
    finally:
        poisoned.reset(token)

    assert result.placement == "in_process"
    assert observed == ["clean"]


def test_protocol_rejects_successful_value_with_conflict_issues() -> None:
    """A malformed owner result cannot claim both success and a conflict."""

    request = _request()
    payload = canonical_json_load_bytes(execute_probe(request.data))
    payload = dict(payload)
    payload["environment"] = {
        "value": payload["environment"]["value"],
        "issues": [
            {
                "code": "environment.requirements_conflict",
                "message": "requirement conflict",
                "path": None,
                "sources": [],
            }
        ],
    }

    with pytest.raises(ProbeProtocolError):
        decode_result(canonical_json_bytes(payload), request)


def test_protocol_rejects_raw_conflict_diagnostics() -> None:
    """Wire diagnostics cannot retain source text or arbitrary values."""

    with pytest.raises(ProbeProtocolError):
        _result_from_data(
            {
                "value": None,
                "issues": [
                    {
                        "code": "environment.requirements_conflict",
                        "message": "token=private",
                        "path": None,
                        "sources": [],
                    }
                ],
            },
            EnvironmentRequirement,
        )
    with pytest.raises(ProbeProtocolError):
        _safe_diagnostics(("token=private",))


def test_protocol_rejects_boolean_snapshot_version_early() -> None:
    """A Boolean cannot impersonate the exact integer snapshot version."""

    request = _request()
    snapshot = _snapshot_data(_decode_request(request.data)[0])
    snapshot["version"] = True

    with pytest.raises(ProbeProtocolError):
        _snapshot_from_data(snapshot)


def test_inline_timeout_covers_capture_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cooperative inline timeout includes passive call-entry capture work."""

    from dryml.dispatch import _probe

    original = _probe._capture_description

    def delayed(description):
        time.sleep(0.01)
        return original(description)

    monkeypatch.setattr(_probe, "_capture_description", delayed)

    with pytest.raises(TimeoutError, match="cooperative timeout"):
        run_probe(_root, options=ProbeOptions(execution_timeout=0.001))
