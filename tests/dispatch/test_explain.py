"""Non-submitting Dispatch explanation tests."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

import dryml.dispatch as dispatch
from dryml.core.execute import CoreOptions
from dryml.environments import inspect_current
from dryml.environments import req as environment_req
from dryml.environments.specs import (
    CurrentEnvironmentSpec,
    PythonExecutableSpec,
)
from dryml.execute.backend import Backend
from dryml.execute.config import BackendConfig
from dryml.execute.errors import BackendUnavailableError
from dryml.execute.models import EnvironmentCandidate
from dryml.execute.subprocess import SubProcessConfig
from dryml.worlds import ProcessAllocation
from dryml.worlds import req as world_req


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Isolate default selection for each explanation."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


def test_explain_probes_without_invoking_workload_or_reserving_resources() -> (
    None
):
    """Explanation returns bounded evidence rather than an execution ticket."""

    invoked = False

    def workload(*args, **kwargs):
        """Record prohibited direct workload execution."""

        nonlocal invoked
        invoked = True
        return args, kwargs

    dispatch.set_execute_backend_default(SubProcessConfig())
    report = dispatch.explain(
        workload,
        env="workload data",
        backend="workload data",
        probe="workload data",
    )

    assert not invoked
    assert report.workload_placement == "execute"
    assert report.supported_methods == frozenset({"run", "submit"})
    assert report.probe_placement in {"in_process", "execute"}
    assert isinstance(report.diagnostics, tuple)
    assert isinstance(report.warnings, tuple)


def test_preflight_failure_is_reported_or_raised_before_execution() -> None:
    """Represent trustworthy probe setup failure before accepting work."""

    dispatch.set_execute_backend_default(SubProcessConfig())
    dispatch.set_probe_default(
        dispatch.ProbeOptions(
            placement="in_process", backend=SubProcessConfig()
        )
    )
    report = dispatch.explain(lambda: None)
    assert not report.eligible
    with pytest.raises(dispatch.DispatchError) as error:
        dispatch.submit(lambda: None)
    assert error.value.report == report


def test_safe_capture_failure_becomes_an_ineligible_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep classified capture errors on the Dispatch report boundary."""

    from dryml.code.errors import InvalidTargetError

    dispatch.set_execute_backend_default(SubProcessConfig())

    def fail_capture(*_args, **_kwargs):
        """Provide one classified value-free capture failure."""

        raise InvalidTargetError()

    monkeypatch.setattr(dispatch._preflight, "prepare_probe", fail_capture)
    report = dispatch.explain(lambda: None)
    assert not report.eligible
    assert report.diagnostics == ("target.invalid",)
    with pytest.raises(dispatch.DispatchError) as error:
        dispatch.run(lambda: None)
    assert error.value.report.diagnostics == ("target.invalid",)


def test_valid_preflight_hands_the_selected_backend_to_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful U6 preflight reaches the backend owner without fallback."""

    dispatch.set_execute_backend_default(SubProcessConfig())
    accepted = []

    class _Future:
        """Minimal recovered future used to retain run's ownership boundary."""

        def result(self):
            """Return the configured recovered value."""

            return "accepted"

        def cleanup(self):
            """Retain the ordinary successful cleanup contract."""

    def submit_backend(prepared, fn, args, kwargs):
        """Capture the retained selected configuration at the private seam."""

        accepted.append((prepared.options.backend, fn, args, kwargs))
        return _Future()

    monkeypatch.setattr(dispatch.api, "_submit_backend", submit_backend)
    future = dispatch.submit(lambda: "never invoked")
    assert isinstance(future, _Future)
    assert dispatch.run(lambda: "never invoked") == "accepted"
    assert len(accepted) == 2


def test_explain_does_not_emit_coverage_warning() -> None:
    """Keep incomplete coverage as report evidence until later execution."""

    dispatch.set_execute_backend_default(SubProcessConfig())
    with warnings.catch_warnings(record=True) as captured:
        dispatch.explain(lambda: None)
    assert not [
        item
        for item in captured
        if issubclass(item.category, dispatch.DispatchCoverageWarning)
    ]


class DiscoveryBackend(Backend):
    """Return controlled non-reserving discovery evidence."""

    def __init__(
        self, *, available: bool = True, launchable: bool | None = True
    ):
        """Record discovery lifecycle calls and evidence availability."""

        self.available = available
        self.launchable = launchable
        self.started = 0
        self.discoveries: list[tuple[object, object, object]] = []
        self.closed = 0

    def start(self) -> None:
        """Record selected-backend startup without accepting a workload."""

        self.started += 1
        if not self.available:
            raise BackendUnavailableError("selected fake backend unavailable")

    def capabilities(self) -> frozenset[str]:
        """Return no submission capabilities for this discovery-only fake."""

        return frozenset()

    def create_future(self, submission_id, output):
        """Reject workload creation because U6 must never reach it."""

        raise AssertionError("U6 must not create a workload future")

    def submit(self, call, *, future) -> None:
        """Reject workload submission because U6 must never reach it."""

        raise AssertionError("U6 must not submit a workload")

    def discover(
        self, *, environment=None, environment_spec=None, world=None, timeout
    ):
        """Return one selected candidate and no synthetic world grant."""

        del timeout
        self.discoveries.append((environment, environment_spec, world))
        return SimpleNamespace(
            complete=True,
            issues=(),
            environments=(
                EnvironmentCandidate(
                    "candidate",
                    CurrentEnvironmentSpec(),
                    inspect_current(),
                    None,
                    self.launchable,
                    (),
                ),
            ),
            plans=(),
        )

    def resources(self, *, timeout):
        """Reject resource observation outside discovery evidence."""

        raise AssertionError("U6 must not reserve resources")

    def reconcile_cleanup(self, submission_id, *, timeout) -> None:
        """Reject cleanup for a submission that U6 never creates."""

        raise AssertionError("U6 must not create a submission")

    def close(self, *, cancel, timeout) -> None:
        """Record independent discovery-owner cleanup."""

        assert not cancel
        assert timeout is not None
        self.closed += 1


@dataclass(frozen=True, kw_only=True)
class DiscoveryConfig(BackendConfig):
    """Bind a test backend to one inert Dispatch Execute configuration."""

    backend: DiscoveryBackend = field(repr=False, compare=False)

    def create_backend(self) -> Backend:
        """Return the exact selected test backend without fallback."""

        return self.backend


def test_explain_discovers_only_the_selected_backend_and_cleans_up() -> None:
    """Use actual selected discovery while leaving other names inert."""

    unavailable = DiscoveryBackend(available=False)
    viable = DiscoveryBackend()
    dispatch.register_backend("viable", DiscoveryConfig(backend=viable))
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=unavailable))

    report = dispatch.explain(lambda: None)

    assert not report.eligible
    assert "dispatch.backend_discovery_unavailable" in report.diagnostics
    assert unavailable.started == 1
    assert unavailable.closed == 1
    assert viable.started == 0
    with pytest.raises(dispatch.DispatchError) as error:
        dispatch.run(lambda: None)
    assert (
        "dispatch.backend_discovery_unavailable"
        in error.value.report.diagnostics
    )
    assert viable.started == 0


@pytest.mark.parametrize(
    "launchable, expected",
    [
        (False, "dispatch.backend_environment_incompatible"),
        (None, "dispatch.backend_environment_incompatible"),
    ],
)
def test_explain_requires_affirmative_backend_environment_evidence(
    launchable: bool | None, expected: str
) -> None:
    """Treat failed or unknown selected-candidate grants as ineligible."""

    backend = DiscoveryBackend(launchable=launchable)
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))

    @environment_req(tags=("dispatch-test-unavailable",))
    def workload() -> None:
        """Provide a hard requirement the discovered record cannot satisfy."""

    report = dispatch.explain(workload)

    assert not report.eligible
    assert expected in report.diagnostics
    assert backend.discoveries[0][0] is not None
    assert backend.closed == 1


def test_explain_accepts_affirmative_selected_backend_evidence() -> None:
    """Accept complete compatible evidence from the one selected backend."""

    backend = DiscoveryBackend()
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))

    @environment_req()
    def workload() -> None:
        """Provide an empty hard environment requirement for owner checking."""

    report = dispatch.explain(workload)

    assert report.eligible
    assert backend.discoveries[0][0] is not None
    assert backend.closed == 1


def test_explain_revalidates_target_after_backend_discovery() -> None:
    """Reject declaration drift observed after selected discovery completes."""

    backend = DiscoveryBackend()
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))

    def workload() -> None:
        """Provide a target whose declaration changes during discovery."""

    discover = backend.discover

    def mutate_during_discovery(**kwargs):
        """Change a captured declaration before returning genuine evidence."""

        environment_req(tags=("changed-during-discovery",))(workload)
        return discover(**kwargs)

    backend.discover = mutate_during_discovery
    report = dispatch.explain(workload)

    assert not report.eligible
    assert "dispatch.target_changed" in report.diagnostics


def test_explain_checks_the_frozen_selected_environment_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a discovered candidate that mismatches the exact frozen pin."""

    backend = DiscoveryBackend()
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))
    resolved = dispatch._preflight.resolve_environment_spec(
        CurrentEnvironmentSpec()
    )
    mismatched = replace(resolved, expected_executable="/different/python")
    monkeypatch.setattr(
        dispatch._preflight,
        "resolve_environment_spec",
        lambda _spec: mismatched,
    )

    report = dispatch.with_options(python=CurrentEnvironmentSpec()).explain(
        lambda: None
    )

    assert not report.eligible
    assert "dispatch.backend_selection_mismatch" in report.diagnostics
    assert backend.discoveries[0][1] is mismatched


def test_invalid_selector_is_a_safe_ineligible_report() -> None:
    """Keep known selector failures on the Dispatch report boundary."""

    backend = DiscoveryBackend()
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))
    view = dispatch.with_options(
        python=PythonExecutableSpec(executable="/missing/dispatch-python")
    )

    report = view.explain(lambda: None)

    assert not report.eligible
    assert report.diagnostics == ("dispatch.selector_invalid",)
    assert backend.started == 0
    with pytest.raises(dispatch.DispatchError) as error:
        view.run(lambda: None)
    assert error.value.report.diagnostics == report.diagnostics


@pytest.mark.parametrize("operation", ("explain", "run", "submit"))
def test_unavailable_probe_selector_stays_on_the_safe_report_boundary(
    operation: str,
) -> None:
    """Keep valid unavailable probe pins distinct from malformed API values."""

    unavailable = "/missing/dispatch-probe-python-SECRET"
    invoked = False

    def workload() -> None:
        """Record any prohibited workload invocation during preflight."""

        nonlocal invoked
        invoked = True

    view = dispatch.with_options(
        backend=SubProcessConfig(),
        probe=dispatch.ProbeOptions(
            environment_spec=PythonExecutableSpec(executable=unavailable)
        ),
    )

    if operation == "explain":
        report = view.explain(workload)
    else:
        with pytest.raises(dispatch.DispatchError) as raised:
            getattr(view, operation)(workload)
        report = raised.value.report

    assert not report.eligible
    assert report.probe_placement is None
    assert report.coverage is None
    assert report.probe_reason == "selected probe environment is unavailable"
    assert report.diagnostics == ("dispatch.probe_selector_invalid",)
    assert unavailable not in repr(report)
    assert not invoked


def test_malformed_probe_options_remain_a_type_error() -> None:
    """Reject malformed selector data rather than report it as unavailable."""

    with pytest.raises(TypeError, match="environment_spec"):
        dispatch.ProbeOptions(environment_spec=object())


def test_explain_requires_backend_world_feasibility() -> None:
    """Reject a selected backend that supplies no affirmative world plan."""

    backend = DiscoveryBackend()
    dispatch.set_execute_backend_default(DiscoveryConfig(backend=backend))

    @world_req(cpus=1)
    def workload() -> None:
        """Provide a world requirement needing feasibility evidence."""

    report = dispatch.explain(workload)

    assert not report.eligible
    assert "dispatch.backend_world_incompatible" in report.diagnostics
    assert backend.discoveries[0][2] is not None


def test_in_process_explain_checks_current_constraints_without_core_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject local constraints without core execution state."""

    monkeypatch.setattr(
        dispatch._preflight,
        "_capture_frozen_core_controls",
        lambda *_args, **_kwargs: pytest.fail(
            "local route must not capture core"
        ),
    )

    @environment_req(tags=("dispatch-test-unavailable",))
    def workload() -> None:
        """Provide a local hard environment requirement."""

    report = dispatch.with_options(
        backend=dispatch.InProcess()
    ).explain(workload)

    assert not report.eligible
    assert (
        "dispatch.in_process_environment_incompatible" in report.diagnostics
    )
    assert report.probe_reason == "current process satisfies probe controls"


def test_in_process_explain_allows_no_world_requirement() -> None:
    """Keep an unrequested local world compatible without inventing a grant."""

    invoked = False

    def workload() -> None:
        """Record that explanation must not call the local workload."""

        nonlocal invoked
        invoked = True

    report = dispatch.with_options(
        backend=dispatch.InProcess()
    ).explain(workload)

    assert report.eligible
    assert not invoked


def test_in_process_core_override_is_ineligible_without_core_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a worker-only local core control without ambient core state."""

    monkeypatch.setattr(
        dispatch._preflight,
        "_capture_frozen_core_controls",
        lambda *_args, **_kwargs: pytest.fail(
            "local route must not capture core"
        ),
    )

    report = dispatch.with_options(
        backend=dispatch.InProcess(), core=CoreOptions(return_objects=False)
    ).explain(lambda: None)

    assert not report.eligible
    assert "dispatch.in_process_core_override" in report.diagnostics


def test_in_process_orchestrator_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an orchestrator generation before any later direct invocation."""

    monkeypatch.setattr(
        dispatch._admission,
        "snapshot_for_generation",
        lambda _generation: SimpleNamespace(
            health="healthy", mode="orchestrator", allocation=None
        ),
    )

    report = dispatch.with_options(backend=dispatch.InProcess()).explain(
        lambda: None
    )

    assert not report.eligible
    assert "dispatch.in_process_orchestrator" in report.diagnostics


def test_in_process_pin_mismatch_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare a frozen local selector against fresh current evidence."""

    resolved = dispatch._preflight.resolve_environment_spec(
        CurrentEnvironmentSpec()
    )
    mismatched = replace(resolved, expected_executable="/different/python")
    monkeypatch.setattr(
        dispatch._preflight,
        "resolve_environment_spec",
        lambda _spec: mismatched,
    )

    report = dispatch.with_options(
        backend=dispatch.InProcess(), python=CurrentEnvironmentSpec()
    ).explain(lambda: None)

    assert not report.eligible
    assert "dispatch.in_process_selection_mismatch" in report.diagnostics


def test_in_process_ignores_disabled_automatic_environment_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require direct owner checking even when automatic axes are disabled."""

    monkeypatch.setattr(
        dispatch._admission,
        "snapshot_for_generation",
        lambda _generation: SimpleNamespace(
            health="healthy",
            mode="python",
            allocation=None,
            requirement_axes={"environment": False},
        ),
    )

    @environment_req(tags=("dispatch-test-unavailable",))
    def workload() -> None:
        """Provide a requirement independent of session axes."""

    report = dispatch.with_options(
        backend=dispatch.InProcess()
    ).explain(workload)

    assert not report.eligible
    assert "dispatch.in_process_environment_incompatible" in report.diagnostics


def test_in_process_unknown_memory_evidence_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when a selected process has no memory evidence."""

    allocation = SimpleNamespace(
        role="main",
        process=ProcessAllocation(
            replica=0, rank=0, local_rank=0, memory=None
        ),
    )
    monkeypatch.setattr(
        dispatch._admission,
        "snapshot_for_generation",
        lambda _generation: SimpleNamespace(
            health="healthy", mode="python", allocation=allocation
        ),
    )

    @world_req(memory=1)
    def workload() -> None:
        """Require memory that the selected process cannot evidence."""

    report = dispatch.with_options(
        backend=dispatch.InProcess()
    ).explain(workload)

    assert not report.eligible
    assert "dispatch.in_process_world_incompatible" in report.diagnostics
