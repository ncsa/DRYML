"""End-to-end core Execute coverage for managed-operation transport."""

from __future__ import annotations

import functools

import pytest

from dryml.core import Executor as CoreExecutor
from dryml.core import Repo, StateRef, function, signatures
from dryml.core.execute import CoreOptions
from dryml.core.execute_codec import CoreCallCodecError, decode_outcome, encode_invocation, invoke_invocation
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig, managed_operation
from tests.managed.execution_fixtures import (
    InnerFunctionWrappedManagedValue,
    ManagedMatrixValue,
    MatrixArgument,
    ORDERS,
    OuterWrappedManagedValue,
    matrix_config,
    nested_managed_call,
    nested_managed_advance,
    nested_unrelated_function,
)


def _add_wrapper(target):
    """Retain a visible ordinary wrapper around one managed declaration."""

    @functools.wraps(target)
    def wrapped(self, value, *, managed):
        self.events.append("before")
        try:
            return target(self, value, managed=managed) + 10
        finally:
            self.events.append("after")

    return wrapped


def _default_wrapper(target):
    """Retain a managed target in a keyword-only default for transport coverage."""

    @functools.wraps(target)
    def wrapped(self, value, *, managed, _target=target):
        return _target(self, value, managed=managed) + 10

    return wrapped


class FrozenWrapperValue(Pickleable):
    """Managed receiver whose accepted outer wrapper must remain transport-stable."""

    def __init__(self, value=0):
        """Initialize deterministic state and wrapper observations."""

        self.value = value
        self.events = []

    @_add_wrapper
    @managed_operation()
    def advance(self, value, *, managed):
        """Mutate once through the inner managed lifecycle."""

        self.events.append("body")
        self.value += value
        return self.value


class DefaultCapturedWrapperValue(Pickleable):
    """Fixture whose outer wrapper retains its managed target in a default."""

    def __init__(self):
        """Initialize the state changed by the retained managed body."""

        self.value = 0

    @_default_wrapper
    @managed_operation()
    def advance(self, value, *, managed):
        """Mutate once before the default-captured wrapper adjusts the result."""

        self.value += value
        return self.value


class DirectConfigValue(Pickleable):
    """Non-composite managed receiver used to prove direct config transport."""

    def __init__(self) -> None:
        """Initialize state and callback observations for each fresh attempt."""

        self.value = 0
        self.checkpoint_callbacks = 0

    @managed_operation()
    def advance(self, *, managed) -> int:
        """Advance once and create the callback-visible checkpoint."""

        self.value += 1
        managed.checkpoint()
        return self.value


def _count_checkpoint_callback(value, context) -> None:
    """Record callback delivery on the checkpointed managed receiver."""

    del context
    value.checkpoint_callbacks += 1


class InheritedDirectConfigValue(DirectConfigValue):
    """Child fixture retaining its base class's normal inherited declaration."""


def test_accepted_composite_uses_captured_wrapper_after_class_mutation(tmp_path):
    """A caller mutation after encoding cannot replace the accepted wrapper graph."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = FrozenWrapperValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    invocation = encode_invocation(
        value.advance, (2,), {"managed": ManagedConfig(state_repo=repo)}, repo=repo,
    )

    def replacement(self, value, *, managed):
        """Fail if decoder performs a late receiver-member lookup."""

        raise AssertionError("accepted wrapper was bypassed")

    original = FrozenWrapperValue.__dict__["advance"]
    type.__setattr__(FrozenWrapperValue, "advance", replacement)
    try:
        outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    finally:
        type.__setattr__(FrozenWrapperValue, "advance", original)

    assert outcome["success"]
    assert outcome["result"] == 12


@pytest.mark.parametrize(
    ("subject_type", "capture"),
    ((FrozenWrapperValue, "closure"), (DefaultCapturedWrapperValue, "default")),
)
def test_accepted_composite_snapshots_closure_and_default_wrapper_captures(
        tmp_path, subject_type, capture):
    """Later wrapper-capture mutation cannot alter a previously encoded call."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = subject_type(repo=repo)
    repo.save_object(value, deep_capture=True)
    invocation = encode_invocation(
        value.advance, (2,), {"managed": ManagedConfig(state_repo=repo)}, repo=repo,
    )
    outer = subject_type.__dict__["advance"]._outer
    target = outer.__wrapped__

    def replacement(self, value, *, managed):
        """Fail if the decoder reads a mutable source wrapper after acceptance."""

        raise AssertionError("accepted wrapper capture was bypassed")

    if capture == "closure":
        cell = next(cell for cell in outer.__closure__ if cell.cell_contents is target)
        cell.cell_contents = replacement
        restore = lambda: setattr(cell, "cell_contents", target)
    else:
        defaults = outer.__kwdefaults__
        assert defaults is not None
        defaults["_target"] = replacement
        restore = lambda: defaults.__setitem__("_target", target)
    try:
        outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    finally:
        restore()

    assert outcome["success"]
    assert outcome["result"] == 12


@pytest.mark.parametrize(
    ("member", "written_order"), ORDERS,
)
@pytest.mark.parametrize("placement", ("local", "subprocess"))
def test_managed_decorator_orders_materialize_once_and_publish_one_final_state(
        tmp_path, member, written_order, placement,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest):
    """Each order has one Mat boundary; subprocess rows retain host observation."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    if placement == "local":
        request.getfixturevalue("fixed_managed_snapshot_environment")
    argument_state = repo.save_object(argument, deep_capture=True)
    operation = getattr(subject, member)
    from dryml.annotations import annotations_for_method

    annotations = annotations_for_method(ManagedMatrixValue, member)
    assert ("test.integration", "present") in [
        (annotation.key, annotation.value) for annotation in annotations
    ]
    if placement == "local":
        observed = []
        original_args = signatures.SignaturePlan._prepare_bound
        original_return = signatures.SignaturePlan._prepare_return

        def observe_args(plan, arguments, *, partial, controls):
            observed.append("args")
            return original_args(
                plan, arguments, partial=partial, controls=controls,
            )

        def observe_return(plan, value, controls, **kwargs):
            observed.append("return")
            return original_return(plan, value, controls, **kwargs)

        monkeypatch.setattr(
            signatures.SignaturePlan, "_prepare_bound", observe_args,
        )
        monkeypatch.setattr(
            signatures.SignaturePlan, "_prepare_return", observe_return,
        )
        result = operation(
            argument_state, managed=matrix_config(repo, control_store),
        )
        assert result is argument
        assert result.value == 3
        assert (subject.calls, subject.value) == (1, 3)
        assert subject.save_calls == 1
        assert observed == ["args", "return"]
    else:
        repo.save_object(subject, deep_capture=True)
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(operation, argument_state)
            assert isinstance(future.result(timeout=15), StateRef)
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)
        restored = repo.load_state_ref(
            operation.status(
                state_repo=repo, control_store=control_store,
            ).final_state_ref,
        )
        assert (restored.calls, restored.value, restored.save_calls) == (1, 3, 2)
    assert operation.status(
        state_repo=repo, control_store=control_store,
    ).state == "completed"


@pytest.mark.parametrize(
    ("subject_type", "persisted_events"),
    (
        (FrozenWrapperValue, ["before", "body"]),
        (InnerFunctionWrappedManagedValue, ["before", "body", "after"]),
    ),
)
def test_ordinary_wrappers_run_in_authored_order_on_first_subprocess_call(
        tmp_path, subject_type, persisted_events):
    """W(M(method)) and M(W(F(method))) retain wrapper bodies before local use."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = subject_type(repo=repo)
    repo.save_object(subject, deep_capture=True)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(subject.advance, 2)
        assert future.result(timeout=15) == 12
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    restored = repo.load_state_ref(
        subject.advance.status(state_repo=repo).final_state_ref, reuse_live="never",
    )
    assert restored.events == persisted_events
    assert restored.value == 2


@pytest.mark.parametrize("placement", ("codec", "subprocess"))
@pytest.mark.parametrize("subject_type", (DirectConfigValue, InheritedDirectConfigValue))
def test_direct_bound_operation_transports_explicit_managed_config(
        tmp_path, placement: str, subject_type: type[DirectConfigValue]) -> None:
    """Direct bound worker calls retain explicit authority, callbacks, and rerun."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = subject_type(repo=repo)
    repo.save_object(subject, deep_capture=True)
    initial = ManagedConfig(
        state_repo=repo, control_store=control_store,
        callbacks=[_count_checkpoint_callback],
    )
    assert subject.advance(managed=initial) == 1
    rerun = ManagedConfig(
        state_repo=repo, control_store=control_store, rerun=True,
        callbacks=[_count_checkpoint_callback],
    )

    if placement == "codec":
        outcome = decode_outcome(
            invoke_invocation(
                encode_invocation(subject.advance, (), {"managed": rerun}, repo=repo),
                repo=repo,
            ),
            repo=repo,
        )
        assert outcome["success"]
        assert outcome["result"] == 2
    else:
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(subject.advance, kwargs={"managed": rerun})
            assert future.result(timeout=15) == 2
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)

    status = subject.advance.status(state_repo=repo, control_store=control_store)
    restored = repo.load_state_ref(status.final_state_ref, reuse_live="never")
    assert status.state == "completed"
    assert (restored.value, restored.checkpoint_callbacks) == (2, 2)


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_direct_bound_operation_rejects_malformed_config_before_mutation(tmp_path) -> None:
    """Invalid explicit policy is rejected during capture before a worker can mutate."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = DirectConfigValue(repo=repo)
    repo.save_object(subject, deep_capture=True)
    malformed = ManagedConfig(state_repo=repo)
    object.__setattr__(malformed, "rerun", 1)

    with pytest.raises(CoreCallCodecError, match="malformed managed config"):
        encode_invocation(subject.advance, (), {"managed": malformed}, repo=repo)

    assert subject.value == 0
    assert subject.advance.status(state_repo=repo).state == "not_started"


def test_unrelated_nested_function_keeps_its_own_normalization_boundary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A managed handoff never suppresses a separate nested function boundary."""

    observed = []
    original_args = signatures.SignaturePlan._prepare_ambient_args
    original_return = signatures.SignaturePlan._prepare_ambient_return

    def observe_args(plan, args, kwargs):
        observed.append("args")
        return original_args(plan, args, kwargs)

    def observe_return(plan, value, **kwargs):
        observed.append("return")
        return original_return(plan, value, **kwargs)

    monkeypatch.setattr(
        signatures.SignaturePlan, "_prepare_ambient_args", observe_args,
    )
    monkeypatch.setattr(
        signatures.SignaturePlan, "_prepare_ambient_return", observe_return,
    )
    assert nested_unrelated_function(2) == 3
    assert observed == ["args", "return"]


@pytest.mark.parametrize("placement", ("local", "subprocess"))
def test_nested_managed_config_preserves_selected_state_ref_authority(
        tmp_path, placement: str) -> None:
    """Nested local and worker calls retain explicit config and StateRef authority."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    subject = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    argument_state = repo.save_object(argument, deep_capture=True)
    repo.save_object(subject, deep_capture=True)
    nested_config = {"config": [matrix_config(repo, control_store)]}
    if placement == "local":
        assert nested_managed_call(subject, argument_state, nested_config) == argument_state
    else:
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(
                nested_managed_call, subject, argument_state, nested_config,
            )
            assert future.result(timeout=15) == argument_state
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)
    status = subject.operation_afm.status(
        state_repo=repo, control_store=control_store,
    )
    assert status.state == "completed"
    restored = repo.load_state_ref(status.final_state_ref, reuse_live="never")
    assert (restored.calls, restored.value) == (1, 3)
