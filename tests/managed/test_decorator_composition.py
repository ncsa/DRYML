"""Managed declaration composition with function and passive annotations."""

from __future__ import annotations

import functools
import inspect
import contextvars
import threading
import asyncio

import dill
import pytest

from dryml.annotations import Annotation, annotations_for_method, attach_annotation
from dryml.core import Repo, Serializable, function
from dryml.core import signatures
from dryml.core.signatures import function_normalization_handoff
from dryml.core.execute import (
    ExecutionContext, SharedDirStoreStrategy, invoke_prepared_call, worker_context,
)
from dryml.core.store.dir import DirStore
from dryml.execute import Executor
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig, managed_operation


def _annotate(target):
    """Attach one identity-preserving declaration used by each matrix row."""

    return attach_annotation(target, Annotation("test.composition", "present"))


def _recording_wrapper(target):
    """Keep observable before/error/finally/after behavior around a declaration."""

    @functools.wraps(target)
    def wrapped(*args, **kwargs):
        receiver = args[0]
        receiver.events.append("before")
        try:
            result = target(*args, **kwargs)
        except BaseException:
            receiver.events.append("error")
            raise
        finally:
            receiver.events.append("finally")
        receiver.events.append("after")
        return result + 10

    return wrapped


def _default_captured_wrapper(target):
    """Forward through an exact target retained in a function default."""

    def wrapped(*args, __target=target, **kwargs):
        return __target(*args, **kwargs) + 1

    return functools.wraps(target)(wrapped)


def _chain_wrapper(target):
    """Record a distinct preserved ordinary wrapper layer for chain coverage."""

    @functools.wraps(target)
    def wrapped(*args, **kwargs):
        args[0].events.append("wrapper:before")
        try:
            return target(*args, **kwargs)
        finally:
            args[0].events.append("wrapper:finally")

    return wrapped


class OuterManaged(Serializable):
    """Importable fixture with an ordinary wrapper outside managed lifecycle."""

    def __init__(self, value=0):
        self.value = value
        self.events = []

    @_recording_wrapper
    @managed_operation()
    def operation(self, value: int, *, managed) -> int:
        """Mutate once or fail after the wrapper's before behavior."""

        self.events.append("body")
        if value < 0:
            raise ValueError("negative")
        self.value += value
        return self.value


class InnerFunctionManaged(Serializable):
    """Fixture with managed owning an ordinary wrapper around ``function``."""

    def __init__(self, value=0):
        self.value = value
        self.events = []

    @managed_operation()
    @_recording_wrapper
    @function
    def operation(self, value: int, *, managed) -> int:
        """Exercise the exact function-owner handoff beneath a real wrapper."""

        self.events.append("body")
        if value < 0:
            raise ValueError("negative")
        self.value += value
        return self.value


class ChainFunctionManaged(Serializable):
    """Importable M(W(W(F))) fixture retaining all ordinary wrapper bodies."""

    def __init__(self, value=0):
        self.value = value
        self.events = []

    @managed_operation()
    @_chain_wrapper
    @_chain_wrapper
    @function
    def operation(self, value: int, *, managed) -> int:
        """Run after both wrappers without adding another F boundary."""

        self.events.append("body")
        self.value += value
        return self.value


@pytest.mark.parametrize(
    "written_order",
    (
        ("A", "F", "M"), ("A", "M", "F"), ("F", "A", "M"),
        ("F", "M", "A"), ("M", "A", "F"), ("M", "F", "A"),
    ),
)
def test_all_decorator_orders_preserve_one_managed_boundary(tmp_path, written_order):
    """Every supported written A/F/M order binds, declares, and runs once."""

    calls = []

    def operation(self, value: int, *, managed) -> int:
        calls.append((self, value, managed))
        self.value += value
        return self.value

    decorators = {"A": _annotate, "F": function, "M": managed_operation()}
    member = operation
    for name in reversed(written_order):
        member = decorators[name](member)

    class Subject(Serializable):
        """Stateful receiver assembled from one matrix declaration."""

        def __init__(self):
            self.value = 0

    Subject.operation = member
    member.__set_name__(Subject, "operation")
    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    subject = Subject(repo=repo)

    assert inspect.signature(subject.operation).parameters["value"].annotation == "int"
    annotations = annotations_for_method(Subject, "operation")
    assert len(annotations) == 1
    assert (annotations[0].key, annotations[0].value) == (
        "test.composition", "present",
    )
    assert subject.operation(3, managed=ManagedConfig(state_repo=repo)) == 3
    assert calls and calls[0][0] is subject
    assert len(calls) == 1
    assert subject.operation.status(state_repo=repo).state == "completed"


def test_outer_wrapper_exposes_managed_controls_and_preserves_its_behavior(tmp_path):
    """W(M(method)) has controls before call and retains outer result processing."""

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = OuterManaged(repo=repo)

    assert subject.operation.status(state_repo=repo).state == "not_started"
    assert inspect.signature(subject.operation).parameters["value"].annotation == "int"
    assert subject.operation(2, managed=ManagedConfig(state_repo=repo)) == 12
    assert subject.events == ["before", "body", "finally", "after"]
    assert subject.operation.status(state_repo=repo).state == "completed"

    failing = OuterManaged(repo=repo)
    with pytest.raises(ValueError, match="negative"):
        failing.operation(-1, managed=ManagedConfig(state_repo=repo))
    assert failing.events == ["before", "body", "error", "finally"]


def test_outer_wrapper_reconstructs_as_one_managed_target_in_a_worker(tmp_path):
    """A fresh core worker restores the hidden composite before its first call."""

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = OuterManaged(repo=repo)
    repo.save(subject, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        subject.operation, (3,), {}, repo=repo, control_store=None,
        update_args=False,
    )
    assert {
        "managed_declaration", "managed_composite", "managed_composite_target",
    } <= {
        node["tag"] for node in dill.loads(prepared.invocation)["nodes"]
    }

    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)

    assert strategy.recover(
        output, prepared, repo=repo, args=(3,), kwargs={},
        return_objects=False, update_args=False,
    ) == 13


def test_managed_wrapper_function_handoff_runs_once_and_resets_after_failure(tmp_path):
    """M(W(F(method))) skips only its inner F boundary and never leaks it."""

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = InnerFunctionManaged(repo=repo)

    assert subject.operation(2, managed=ManagedConfig(state_repo=repo)) == 12
    assert subject.events == ["before", "body", "finally", "after"]
    assert subject.operation.status(state_repo=repo).state == "completed"

    failing = InnerFunctionManaged(repo=repo)
    with pytest.raises(ValueError, match="negative"):
        failing.operation(-1, managed=ManagedConfig(state_repo=repo))
    assert failing.events == ["before", "body", "error", "finally"]
    assert function(lambda value: value + 1)(1) == 2


def test_outer_managed_composite_reconstructs_in_a_fresh_subprocess(tmp_path):
    """Subprocess reconstruction finalizes W(M(method)) before invocation."""

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = OuterManaged(repo=repo)
    repo.save(subject, deep_capture=True)
    prepared = SharedDirStoreStrategy().prepare(
        subject.operation, (4,), {}, repo=repo, control_store=None,
        update_args=False,
    )
    spool = tmp_path / "spool"
    spool.mkdir()

    with Executor(SubProcessConfig(spool_directory=spool)) as executor:
        assert executor.run(
            invoke_prepared_call, prepared.invocation,
            worker_setup=prepared.worker_setup(),
        ) == 14


def test_hidden_managed_aliases_and_copied_wrapper_evidence_are_rejected():
    """Only one member and an executable closure can finalize hidden managed code."""

    declaration = managed_operation()(lambda self, *, managed: None)

    with pytest.raises(Exception, match="multiple member names"):
        class Ambiguous(Serializable):
            """Attempt to expose one hidden declaration under two member names."""

            first = _recording_wrapper(declaration)
            second = _recording_wrapper(declaration)

    def raw(self, *, managed):
        """Provide a native declaration for forged wrapper evidence."""

    def copied(self, *, managed):
        """Do not retain the function wrapper it claims to describe."""

    copied.__wrapped__ = function(raw)
    with pytest.raises(Exception, match="signature-preserving wrapper"):
        managed_operation()(copied)


def test_managed_wrapper_chain_preserves_every_wrapper_and_one_function_boundary(tmp_path):
    """M(W(W(F(method))) retains both ordinary bodies and bypasses only F once."""

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = ChainFunctionManaged(repo=repo)

    assert subject.operation(2, managed=ManagedConfig(state_repo=repo)) == 2
    assert subject.events == [
        "wrapper:before", "wrapper:before", "body", "wrapper:finally",
        "wrapper:finally",
    ]


@pytest.mark.parametrize("kind", ("opaque", "async", "inconsistent"))
def test_hidden_managed_finalization_rejects_unproven_outer_relationship(kind):
    """W(M(method)) requires synchronous closure/default forwarding evidence."""

    declaration = managed_operation()(lambda self, value, *, managed: value)

    if kind == "opaque":
        def outer(self, value, *, managed):
            return value

        outer.__wrapped__ = declaration
    elif kind == "async":
        def decorate(target):
            @functools.wraps(target)
            async def outer(*args, **kwargs):
                return target(*args, **kwargs)

            return outer

        outer = decorate(declaration)
    else:
        def decorate(target):
            @functools.wraps(target)
            def outer(self, value, extra, *, managed):
                return target(self, value, managed=managed) + extra

            return outer

        outer = decorate(declaration)

    with pytest.raises(Exception, match="(closure|synchronous|signature)"):
        class Subject(Serializable):
            """An invalid hidden declaration must fail while class creation runs."""

            operation = outer


def test_hidden_managed_finalization_accepts_default_captured_target(tmp_path):
    """A wrapper that keeps its exact target in a default remains transportable."""

    class Subject(Serializable):
        """Use default capture rather than a closure for the hidden declaration."""

        def __init__(self):
            self.value = 0

        @_default_captured_wrapper
        @managed_operation()
        def operation(self, value: int, *, managed) -> int:
            """Return a value the outer wrapper visibly transforms."""

            self.value += value
            return self.value

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = Subject(repo=repo)
    assert subject.operation(2, managed=ManagedConfig(state_repo=repo)) == 3


def test_function_handoff_lease_expires_and_is_shared_once_across_context_copies(monkeypatch):
    """One mutable owner lease rejects stale/cross-thread reuse without leaking F."""

    calls = []
    original_args = signatures.SignaturePlan._prepare_ambient_args
    original_return = signatures.SignaturePlan._prepare_ambient_return

    def observe_args(self, args, kwargs):
        calls.append("args")
        return original_args(self, args, kwargs)

    def observe_return(self, value, **kwargs):
        calls.append("return")
        return original_return(self, value, **kwargs)

    monkeypatch.setattr(signatures.SignaturePlan, "_prepare_ambient_args", observe_args)
    monkeypatch.setattr(signatures.SignaturePlan, "_prepare_ambient_return", observe_return)
    inner = function(lambda value: value + 1)
    unrelated = function(lambda value: value + 2)
    owner = inner.__dryml_function_invocation_owner__
    copied = None
    with function_normalization_handoff(owner):
        copied = contextvars.copy_context()
        assert copied.run(inner, 1) == 2
        assert inner(1) == 2
        assert unrelated(1) == 3

        async def consume_in_task():
            return inner(1)

        assert asyncio.run(consume_in_task()) == 2

    # The copied context retains the ContextVar value but the exited lease is dead.
    assert copied is not None
    assert copied.run(inner, 1) == 2
    cross_thread = []
    ready = threading.Event()

    def consume_in_thread():
        with function_normalization_handoff(owner):
            context = contextvars.copy_context()
        ready.set()
        cross_thread.append(context.run(inner, 1))

    thread = threading.Thread(target=consume_in_thread)
    thread.start()
    thread.join()
    assert ready.is_set()
    assert cross_thread == [2]
    # First copied consumption bypasses only one boundary.  Every later call,
    # including the stale copied context and unrelated F call, normalizes.
    assert calls == ["args", "return"] * 5


def test_codec_transports_a_nested_callback_capturing_managed_declaration_and_config(tmp_path):
    """Invocation graph v2 retains a hidden declaration/config capture by node."""

    declaration = OuterManaged.__dict__["operation"]._descriptor
    config = ManagedConfig()

    def callback(receiver, value, *, _declaration=declaration,
                 _first=config, _second=config):
        return _declaration(receiver, value, managed=_first) + int(
            _first is _second
        )

    repo = Repo((DirStore(tmp_path / "state"),))
    subject = OuterManaged(repo=repo)
    repo.save(subject, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        callback, (subject, 2), {}, repo=repo, control_store=None,
        update_args=False,
    )
    tags = {node["tag"] for node in dill.loads(prepared.invocation)["nodes"]}

    assert "managed_declaration" in tags
    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(
            prepared.invocation, repo=repo, update_args=False,
        )
    assert strategy.recover(
        output, prepared, repo=repo, args=(subject, 2), kwargs={},
        return_objects=False, update_args=False,
    ) == 3
