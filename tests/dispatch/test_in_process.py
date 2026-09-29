"""Blocking in-process Dispatch admission and direct-call coverage."""

from __future__ import annotations

import threading
import warnings
from dataclasses import replace

import pytest

import dryml.dispatch as dispatch
from dryml.core import Serializable, function
from dryml.core.execute import CoreOptions, SharedDirStoreStrategy
from dryml.environments import CurrentEnvironmentSpec
from dryml.environments import req as environment_req
from dryml.runtime import PublicationBusyError
from dryml.session import reset as reset_session
from dryml.session import set_mode


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Keep process-local Dispatch defaults out of direct-call assertions."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


class _LocalObject(Serializable):
    """A no-Store Object fixture with one directly mutable receiver."""

    def __init__(self) -> None:
        """Initialize a locally owned mutation log without a Repo or Store."""

        self.values: list[object] = []

    def record(self, value: object, *, env: object) -> object:
        """Mutate this live receiver and return the exact workload argument."""

        self.values.append(env)
        return value

    def save_state_to_dir_imp(
        self, dest_dir, *, codec
    ) -> None:
        """Provide the Serializable hook without using it in this test."""

        del dest_dir, codec

    def restore_state_from_dir_imp(
        self, src_dir, *, codec
    ) -> None:
        """Provide the Serializable hook without using it in this test."""

        del src_dir, codec


@function
def _core_owned_identity(value: object, *, backend: object) -> object:
    """Return the exact input through an established core function wrapper."""

    del backend
    return value


def _local_view():
    """Return the explicit local route used by the U8 test workload calls."""

    return dispatch.with_options(backend=dispatch.InProcess())


def test_in_process_invokes_original_function_wrapper_and_object_without_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve direct receiver, argument, wrapper, and return identity."""

    monkeypatch.setattr(
        dispatch.api,
        "_submit_backend",
        lambda *_args: pytest.fail(
            "local execution must not create Execute work"
        ),
    )
    value = object()
    receiver = _LocalObject()
    calls: list[object] = []

    def ordinary(token: object, *, world: object) -> object:
        """Record one ordinary direct call using a control-named keyword."""

        calls.append(world)
        return token

    assert _local_view().run(ordinary, value, world=receiver) is value
    assert calls == [receiver]
    assert (
        _local_view().run(_core_owned_identity, value, backend=receiver)
        is value
    )
    assert _local_view().run(receiver.record, value, env=receiver) is value
    assert receiver.values == [receiver]


def test_in_process_propagates_interruption_identity_and_releases_lease(
) -> None:
    """Leave direct interruptions unchanged and permit a later publication."""

    interruption = KeyboardInterrupt()

    def workload() -> None:
        """Raise the exact interruption object from the direct invocation."""

        raise interruption

    with pytest.raises(KeyboardInterrupt) as raised:
        _local_view().run(workload)
    assert raised.value is interruption
    try:
        assert set_mode("orchestrator").mode == "orchestrator"
    finally:
        reset_session()


def test_in_process_returns_deferred_data_without_driving_it() -> None:
    """Return an untouched awaitable and close it after the assertion."""

    advanced = False

    async def deferred() -> None:
        """Record execution if Dispatch awaited the returned coroutine."""

        nonlocal advanced
        advanced = True

    def workload():
        """Return a coroutine as ordinary synchronous result data."""

        return deferred()

    result = _local_view().run(workload)
    try:
        assert not advanced
        assert result.cr_running is False
    finally:
        result.close()


def test_in_process_rechecks_runtime_after_probe_before_direct_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a new orchestrator generation instead of stale U6 evidence."""

    original = dispatch.api._warn_coverage

    def publish_after_preflight(prepared) -> None:
        """Publish an incompatible generation between probe and admission."""

        original(prepared)
        set_mode("orchestrator")

    monkeypatch.setattr(
        dispatch.api, "_warn_coverage", publish_after_preflight
    )
    try:
        with pytest.raises(dispatch.DispatchError) as raised:
            _local_view().run(
                lambda: pytest.fail("workload must not run")
            )
        assert (
            "dispatch.in_process_orchestrator"
            in raised.value.report.diagnostics
        )
    finally:
        reset_session()


def test_in_process_final_target_guard_runs_under_held_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert target mutation after fresh evidence into admission failure."""

    called = False

    def workload() -> None:
        """Remain uncalled after its declaration carrier changes at guard."""

        nonlocal called
        called = True

    original = dispatch.api.check_current_compatibility

    def check_and_mutate(*args, **kwargs):
        """Use real evidence, then create an observed guard mismatch."""

        result = original(*args, **kwargs)
        environment_req(tags=("u8-final-guard",))(workload)
        return result

    monkeypatch.setattr(
        dispatch.api, "check_current_compatibility", check_and_mutate
    )
    with pytest.raises(dispatch.DispatchError) as raised:
        _local_view().run(workload)
    assert "dispatch.target_changed" in raised.value.report.diagnostics
    assert not called


def test_in_process_callable_instance_rebinding_rejects_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reject a changed raw call descriptor before direct special-method lookup.
    """

    called = False

    class Workload:
        """
        Provide a callable instance whose class binding changes after probe.
        """

        def __call__(self) -> str:
            """Remain uncalled after the guard observes a replacement."""

            nonlocal called
            called = True
            return "original"

    def replacement(self) -> str:
        """Remain uncalled when final admission rejects the new descriptor."""

        nonlocal called
        del self
        called = True
        return "replacement"

    workload = Workload()
    original = Workload.__dict__["__call__"]
    check = dispatch.api.check_current_compatibility

    def check_and_rebind(*args, **kwargs):
        """Rebind after fresh evidence but before final target validation."""

        result = check(*args, **kwargs)
        Workload.__call__ = replacement
        return result

    monkeypatch.setattr(
        dispatch.api, "check_current_compatibility", check_and_rebind
    )
    try:
        with pytest.raises(dispatch.DispatchError) as raised:
            _local_view().run(workload)
    finally:
        Workload.__call__ = original

    assert "dispatch.target_changed" in raised.value.report.diagnostics
    assert not called


def test_in_process_releases_failed_admission_and_rejects_local_core_controls(
) -> None:
    """Release failed leases and leave inherited core options inert."""

    @environment_req(tags=("dispatch-u8-missing-environment",))
    def incompatible() -> None:
        """Require environment evidence unavailable from this interpreter."""

    with pytest.raises(dispatch.DispatchError):
        _local_view().run(incompatible)
    try:
        assert set_mode("orchestrator").mode == "orchestrator"
    finally:
        reset_session()

    assert (
        _local_view().with_options(core=CoreOptions()).run(lambda: "ok")
        == "ok"
    )
    with pytest.raises(dispatch.DispatchError) as raised:
        _local_view().with_options(core=CoreOptions(return_objects=False)).run(
            lambda: None
        )
    assert (
        "dispatch.in_process_core_override" in raised.value.report.diagnostics
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repo", None),
        ("control_store", None),
        ("runtime", None),
        ("cache", "none"),
        ("marshalling", SharedDirStoreStrategy),
        ("return_objects", False),
        ("update_args", False),
    ),
)
def test_every_explicit_core_option_is_ineligible_locally(
    field, value
) -> None:
    """Reject every worker-only CoreOptions field rather than ignoring it."""

    report = _local_view().with_options(
        core=CoreOptions(**{field: value})
    ).explain(lambda: None)
    assert not report.eligible
    assert "dispatch.in_process_core_override" in report.diagnostics


def test_in_process_probe_can_be_remote_while_local_admission_rejects(
) -> None:
    """Keep remote probing independent of current-process admission."""

    @environment_req(tags=("dispatch-u8-remote-probe-missing",))
    def workload() -> None:
        """Declare a requirement that a local process cannot satisfy."""

    view = dispatch.with_options(
        backend=dispatch.InProcess(),
        probe=dispatch.ProbeOptions(placement="execute"),
    )
    report = view.explain(workload)
    assert report.probe_placement == "execute"
    assert not report.eligible
    with pytest.raises(dispatch.DispatchError):
        view.run(workload)


def test_in_process_exact_pin_mismatch_rejects_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat a non-current Python selection as local assertion failure."""

    resolved = dispatch._preflight.resolve_environment_spec(
        CurrentEnvironmentSpec()
    )
    monkeypatch.setattr(
        dispatch._preflight,
        "resolve_environment_spec",
        lambda _spec: replace(
            resolved, expected_executable="/u8/not-current"
        ),
    )
    with pytest.raises(dispatch.DispatchError) as raised:
        _local_view().with_options(python=CurrentEnvironmentSpec()).run(
            lambda: pytest.fail("mismatched pin must not run")
        )
    assert (
        "dispatch.in_process_selection_mismatch"
        in raised.value.report.diagnostics
    )


def test_incompatible_publication_fails_while_direct_call_holds_lease(
) -> None:
    """Use threads to prove a held local call fences publication changes."""

    entered = threading.Event()
    complete = threading.Event()
    result: list[BaseException] = []

    def transition() -> None:
        """Attempt incompatible publication while U8 holds its lease."""

        assert entered.wait(timeout=5)
        try:
            set_mode("orchestrator")
        except BaseException as error:
            result.append(error)
        finally:
            complete.set()

    def workload() -> str:
        """Hold direct execution until concurrent publication has returned."""

        entered.set()
        assert complete.wait(timeout=5)
        return "local"

    worker = threading.Thread(target=transition)
    worker.start()
    assert _local_view().run(workload) == "local"
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], PublicationBusyError)


def test_incomplete_coverage_warns_but_known_conflict_is_ineligible() -> None:
    """Preserve advisory incomplete coverage while stopping hard conflicts."""

    def incomplete(callback):
        """Keep a parameter call unresolved while returning valid data."""

        return callback()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert (
            _local_view().run(incomplete, lambda: "complete enough")
            == "complete enough"
        )
    assert any(
        item.category is dispatch.DispatchCoverageWarning for item in captured
    )

    @environment_req(python=">=4")
    @environment_req(python="<3")
    def conflict() -> None:
        """Attach two irreconcilable known environment declarations."""

    with pytest.raises(dispatch.DispatchError) as raised:
        _local_view().run(conflict)
    assert "dispatch.requirements_conflict" in raised.value.report.diagnostics
