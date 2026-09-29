"""Dispatch composition coverage for managed callable stacks."""

from __future__ import annotations

import sys

import pytest

import dryml.dispatch as dispatch
from dryml.core import Repo, StateRef
from dryml.core.execute import CoreOptions
from dryml.core.object import Pickleable
from dryml.core.session import config as core_config
from dryml.core.store.dir import DirStore
from dryml.dispatch import ProbeOptions
from dryml.dispatch._probe import run_probe
from dryml.environments.specs import PythonExecutableSpec
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig, managed_operation
from tests.managed.execution_fixtures import (
    ConflictingManagedDiscoveryValue,
    ConflictingManagedWorldDiscoveryValue,
    DispatchDiscoveryValue,
    InnerFunctionWrappedManagedValue,
    ManagedMatrixValue,
    ManagedWorldMatrixValue,
    MatrixArgument,
    ORDERS,
    OuterWrappedManagedValue,
    matrix_config,
    declared_managed_call,
)


class DispatchExplicitConfigValue(Pickleable):
    """Non-composite receiver used to verify Dispatch config transport."""

    def __init__(self) -> None:
        """Initialize the state and callback observations for each invocation."""

        self.value = 0
        self.checkpoint_callbacks = 0

    @managed_operation()
    def advance(self, *, managed) -> int:
        """Advance once and create a callback-visible checkpoint."""

        self.value += 1
        managed.checkpoint()
        return self.value


def _count_dispatch_checkpoint(value, context) -> None:
    """Record worker checkpoint callback delivery on the restored receiver."""

    del context
    value.checkpoint_callbacks += 1


@pytest.fixture(autouse=True)
def _reset_dispatch() -> None:
    """Keep every placement independent of process-global Dispatch defaults."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


@pytest.mark.parametrize(("member", "written_order"), ORDERS)
@pytest.mark.parametrize("placement", ("direct", "in_process", "subprocess"))
def test_dispatch_preserves_every_managed_decorator_order_and_declaration(
        tmp_path, member, written_order, placement):
    """Run direct and Dispatch placements with passive requirement discovery."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    argument_state = repo.save_object(argument, deep_capture=True)
    operation = getattr(subject, member)
    config = matrix_config(repo, control_store)

    if placement == "direct":
        result = operation(argument_state, managed=config)
        assert result is argument
    elif placement == "in_process":
        view = dispatch.with_options(backend=dispatch.InProcess())
        result = view.run(operation, argument_state, managed=config)
        assert result is argument
    else:
        repo.save_object(subject, deep_capture=True)
        spool = tmp_path / "spool"
        spool.mkdir()
        view = dispatch.with_options(
            backend=SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store,
                return_objects=False,
            ),
            python=PythonExecutableSpec(sys.executable),
        )
        assert isinstance(view.run(operation, argument_state), StateRef)

    status = operation.status(
        state_repo=repo, control_store=control_store,
    )
    assert status.state == "completed"
    restored = repo.load_state_ref(status.final_state_ref, reuse_live="never")
    assert (restored.calls, restored.value) == (1, 3)


@pytest.mark.parametrize(("member", "written_order"), ORDERS)
def test_dispatch_explains_complete_nonempty_declarations_for_managed_orders(
        tmp_path, member, written_order):
    """Every bound A/F/M order retains its passive requirement carrier."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = ManagedMatrixValue(repo=repo)
    operation = getattr(subject, member)
    view = dispatch.with_options(backend=dispatch.InProcess())

    report = view.explain(operation)

    assert report.coverage == "complete"
    assert report.environment is not None
    assert report.environment.value is not None
    assert report.environment.value.requirements == ()
    assert report.environment.value.details["sources"] == (
        "1: managed-execution-matrix",
    )


@pytest.mark.parametrize(("member", "written_order"), ORDERS)
def test_dispatch_explains_managed_direct_world_declarations(
        member, written_order):
    """Every A/F/M descriptor carrier retains its nonempty world requirement."""

    del written_order
    report = dispatch.with_options(backend=dispatch.InProcess()).explain(
        getattr(ManagedWorldMatrixValue(), member),
    )

    assert report.probe_placement == "in_process"
    assert report.world is not None
    assert report.world.value is not None
    assert report.world.value.roles["main"].resources.cpus.min == 1


@pytest.mark.parametrize(("member", "written_order"), ORDERS)
@pytest.mark.parametrize("placement", ("in_process", "execute"))
def test_dispatch_probe_placements_preserve_managed_world_declarations(
        member, written_order, placement):
    """Both probe placements retain nonempty direct A/F/M world declarations."""

    del written_order
    options = ProbeOptions(placement=placement)
    if placement == "execute":
        options = ProbeOptions(
            placement=placement, backend=SubProcessConfig(),
        )
    result = run_probe(
        getattr(ManagedWorldMatrixValue(), member), options=options,
    )

    assert result.placement == placement
    assert result.world.value is not None
    assert result.world.value.roles["main"].resources.cpus.min == 1


def test_dispatch_rejects_conflicting_managed_declarations_before_run(
        tmp_path):
    """A bound managed carrier conflict stops Dispatch before lifecycle mutation."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    subject = ConflictingManagedDiscoveryValue(repo=repo)
    view = dispatch.with_options(backend=dispatch.InProcess())
    with pytest.raises(dispatch.DispatchError) as raised:
        view.run(subject.advance, managed=matrix_config(repo, control_store))

    assert "dispatch.requirements_conflict" in raised.value.report.diagnostics
    assert subject.calls == 0


def test_dispatch_rejects_conflicting_managed_world_declarations_before_run(
        tmp_path):
    """A managed world conflict stops Dispatch before lifecycle mutation."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = ConflictingManagedWorldDiscoveryValue(repo=repo)
    view = dispatch.with_options(backend=dispatch.InProcess())

    with pytest.raises(dispatch.DispatchError) as raised:
        view.run(subject.advance)

    assert "dispatch.requirements_conflict" in raised.value.report.diagnostics
    assert subject.calls == 0


def test_dispatch_subprocess_retains_explicit_direct_managed_config(tmp_path):
    """Dispatch forwards direct state, control, callback, and rerun policy to workers."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = DispatchExplicitConfigValue(repo=repo)
    repo.save_object(subject, deep_capture=True)
    initial = ManagedConfig(
        state_repo=repo, control_store=control_store,
        callbacks=[_count_dispatch_checkpoint],
    )
    assert subject.advance(managed=initial) == 1
    rerun = ManagedConfig(
        state_repo=repo, control_store=control_store, rerun=True,
        callbacks=[_count_dispatch_checkpoint],
    )
    spool = tmp_path / "spool"
    spool.mkdir()
    view = dispatch.with_options(
        backend=SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, control_store=control_store, return_objects=False),
        python=PythonExecutableSpec(sys.executable),
    )

    assert view.run(subject.advance, managed=rerun) == 2

    status = subject.advance.status(state_repo=repo, control_store=control_store)
    restored = repo.load_state_ref(status.final_state_ref, reuse_live="never")
    assert (status.state, restored.value, restored.checkpoint_callbacks) == (
        "completed", 2, 2,
    )


@pytest.mark.parametrize("placement", ("in_process", "subprocess"))
def test_dispatch_discovers_declarations_on_a_managed_function_stack(
        tmp_path, placement):
    """Dispatch discovers then invokes the declared ordinary-to-managed call."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    subject = DispatchDiscoveryValue(repo=repo)
    if placement == "in_process":
        view = dispatch.with_options(backend=dispatch.InProcess())
    else:
        spool = tmp_path / "spool"
        spool.mkdir()
        view = dispatch.with_options(
            backend=SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store,
                return_objects=False,
            ),
            python=PythonExecutableSpec(sys.executable),
        )
    report = view.explain(declared_managed_call)
    assert report.environment is not None
    assert report.environment.value is not None
    assert report.environment.value.requirements == ()
    assert report.environment.value.details["sources"] == (
        "1: managed-dispatch-discovery",
    )
    if placement == "in_process":
        with core_config(repo=repo):
            assert view.run(declared_managed_call, subject, 3) == 3
    else:
        repo.save_object(subject, deep_capture=True)
        assert view.run(declared_managed_call, subject, 3) == 3
    status_kwargs = {"state_repo": repo}
    if placement == "subprocess":
        status_kwargs["control_store"] = control_store
    assert subject.advance.status(**status_kwargs).state == "completed"


@pytest.mark.parametrize("placement", ("in_process", "subprocess"))
@pytest.mark.parametrize(
    ("subject_type", "events"),
    (
        (OuterWrappedManagedValue, ["before", "body"]),
        (InnerFunctionWrappedManagedValue, ["before", "body", "after"]),
    ),
)
def test_dispatch_runs_ordinary_managed_wrappers(
        tmp_path, placement, subject_type, events):
    """Dispatch preserves ordinary wrapper bodies on local and subprocess routes."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = subject_type(repo=repo)
    if placement == "in_process":
        view = dispatch.with_options(backend=dispatch.InProcess())
        with core_config(repo=repo):
            assert view.run(subject.advance, 2) == 12
    else:
        repo.save_object(subject, deep_capture=True)
        spool = tmp_path / "spool"
        spool.mkdir()
        view = dispatch.with_options(
            backend=SubProcessConfig(spool_directory=spool),
            core=CoreOptions(repo=repo, return_objects=False),
            python=PythonExecutableSpec(sys.executable),
        )
        assert view.run(subject.advance, 2) == 12
    restored = repo.load_state_ref(
        subject.advance.status(state_repo=repo).final_state_ref,
        reuse_live="never",
    )
    assert restored.events == events
    assert restored.value == 2
