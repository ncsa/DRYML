"""Placement selection tests for independent Dispatch probes."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dryml.dispatch import ProbeOptions
from dryml.dispatch._probe import ProbePlacementError, run_probe
from dryml.execute.config import BackendConfig
from dryml.execute.subprocess import SubProcessConfig
from dryml.environments import EnvironmentRequirement, req as environment_req
from dryml.environments.specs import CurrentEnvironmentSpec


@environment_req(tags=("placement",), source="placement")
def _target() -> None:
    """Supply a target for placement-only tests."""


@dataclass(frozen=True, kw_only=True)
class FakeRayProbeConfig(BackendConfig):
    """Name an existing-target Ray-like configuration without importing Ray."""

    def create_backend(self):
        """Fail if selection tries to initialize this fake backend."""

        raise AssertionError(
            "placement selection must not initialize the fake backend"
        )


def test_auto_prefers_inline_only_without_probe_bootstrap_constraints() -> (
    None
):
    """The default avoids Execute when current inspection is compatible."""

    assert run_probe(_target).placement == "in_process"


def test_explicit_backend_always_selects_execute_without_workload_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected probe backend is honored even if inline is possible."""

    from dryml.dispatch import _probe

    chosen: list[BackendConfig] = []

    def remote(request, options, backend):
        chosen.append(backend)
        return _probe.execute_probe(request)

    config = FakeRayProbeConfig()
    monkeypatch.setattr(_probe, "_execute_remote", remote)
    result = run_probe(_target, options=ProbeOptions(backend=config))

    assert result.placement == "execute"
    assert chosen == [config]


def test_execute_and_forced_inline_reject_without_fallback() -> None:
    """Forced inline rejects conflicting controls before probing."""

    with pytest.raises(ProbePlacementError, match="cannot use"):
        run_probe(
            _target,
            options=ProbeOptions(
                placement="in_process", backend=SubProcessConfig()
            ),
        )
    with pytest.raises(ProbePlacementError, match="not compatible"):
        run_probe(
            _target,
            options=ProbeOptions(
                placement="in_process",
                environment=EnvironmentRequirement(tags=("bootstrap",)),
            ),
        )


def test_forced_inline_admits_current_environment_constraints() -> None:
    """Current-process evidence admits a compatible bootstrap request."""

    result = run_probe(
        _target,
        options=ProbeOptions(
            placement="in_process",
            environment=EnvironmentRequirement(),
            environment_spec=CurrentEnvironmentSpec(),
        ),
    )

    assert result.placement == "in_process"


def test_probe_pin_is_resolved_once_and_handed_to_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen selector cannot be re-resolved by a probe backend."""

    from dryml.dispatch import _probe

    resolved: list[object] = []
    handed_off: list[object] = []
    original = _probe.resolve_environment_spec

    def resolve(spec):
        value = original(spec)
        resolved.append(value)
        return value

    def remote(request, options, backend, selection):
        handed_off.append(selection)
        return _probe.execute_probe(request)

    monkeypatch.setattr(_probe, "resolve_environment_spec", resolve)
    monkeypatch.setattr(_probe, "_execute_remote", remote)

    result = run_probe(
        _target,
        options=ProbeOptions(
            placement="execute",
            backend=FakeRayProbeConfig(),
            environment_spec=CurrentEnvironmentSpec(),
        ),
    )

    assert result.placement == "execute"
    assert len(resolved) == 1
    assert handed_off == resolved


def test_named_probe_backend_waits_for_the_later_dispatch_registry() -> None:
    """U5 never guesses a backend for an unresolved registered name."""

    with pytest.raises(ProbePlacementError, match="registry"):
        run_probe(_target, options=ProbeOptions(backend="existing-ray"))
