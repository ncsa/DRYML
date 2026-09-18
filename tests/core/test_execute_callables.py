"""Callable-owner and capture coverage for core Execute."""

from __future__ import annotations

import functools
import os
from pathlib import Path

from dryml.core import ObjectRef, Repo, Serializable, function
from dryml.core.execute import ExecutionContext, SharedDirStoreStrategy, invoke_prepared_call, worker_context
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore
from dryml.execute import Executor
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import managed_operation
from dryml.methods import Method


def _invoke(fn, args, repo):
    """Run one core Execute call in a worker context and decode its ordinary result."""
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(fn, args, {}, repo=repo, control_store=None, update_args=False)
    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)
    return strategy.recover(
        output, prepared, repo=repo, args=args, kwargs={},
        return_objects=False, update_args=False,
    )


@function
def decorated(value):
    """Exercise the signatures-owned raw-result seam."""
    return value + 1


class CallableInstance:
    """Callable fixture retaining ordinary instance state."""

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


class SlotCallableInstance:
    """Callable fixture whose slots capture participates in argument aliasing."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __call__(self, other):
        return self.value.value + other.value, self.value is other


class CapturedValue(Serializable):
    """Durable capture fixture whose receiver and argument must share one restore."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class DirectMethod(Method):
    """Stateless Method fixture using its owner-specific single-call seam."""

    def __call__(self, value):
        return value + 4


class DirectRawResult(Method):
    """Method fixture proving Execute intercepts the raw return before encoding."""

    def __call__(self):
        Path(os.environ["DRYML_U5_RAW_MARKER"]).write_text("once", encoding="ascii")
        return CapturedValue()


class ManagedValue(Serializable):
    """Managed receiver fixture retaining normal lifecycle publication behavior."""

    def __init__(self, calls=0):
        self.calls = calls

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "calls").write_text(str(self.calls), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.calls = int(Path(src_dir, "calls").read_text(encoding="ascii"))

    @managed_operation()
    def increment(self, value, *, managed):
        self.calls += 1
        return value + self.calls

    @managed_operation()
    def raw_result(self, *, managed):
        Path(os.environ["DRYML_U5_RAW_MARKER"]).write_text("once", encoding="ascii")
        return CapturedValue()


def test_function_closure_lambda_bound_method_callable_instance_and_decorator_invoke_once(tmp_path):
    """Every promised ordinary callable shape preserves its receiver/capture behavior."""
    repo = Repo(DirStore(tmp_path / "state"))
    calls = []

    def closure(offset):
        return lambda value: calls.append(value) or value + offset

    class Receiver:
        def add(self, value):
            return value + 3

    assert _invoke(closure(2), (5,), repo) == 7
    assert _invoke(lambda value: value * 2, (5,), repo) == 10
    assert _invoke(Receiver().add, (5,), repo) == 8
    assert _invoke(CallableInstance(6), (5,), repo) == 11
    assert _invoke(decorated, (5,), repo) == 6
    assert calls == []


def test_method_and_managed_owners_invoke_once_without_a_second_outer_boundary(tmp_path):
    """Method selection and managed lifecycle remain owned by their existing owners."""
    repo = Repo(DirStore(tmp_path / "state"))
    assert _invoke(DirectMethod(), (5,), repo) == 9

    receiver = ManagedValue(repo=repo)
    repo.save_object(receiver, deep_capture=True)
    assert _invoke(receiver.increment, (5,), repo) == 6
    assert receiver.calls == 0


def test_method_and_managed_raw_returns_are_published_once_before_return_encoding(tmp_path, monkeypatch):
    """Owner seams publish live raw results after one body call before return handling."""
    repo = Repo(DirStore(tmp_path / "state"))
    strategy = SharedDirStoreStrategy()
    direct_marker = tmp_path / "direct-marker"
    monkeypatch.setenv("DRYML_U5_RAW_MARKER", str(direct_marker))
    direct = strategy.prepare(
        DirectRawResult(), (), {}, repo=repo,
        control_store=None, update_args=False,
    )
    direct_output = strategy.invoke(direct.invocation, repo=repo, update_args=False)
    assert strategy.recover(
        direct_output, direct, repo=repo, args=(), kwargs={},
        return_objects=False, update_args=False,
    ).object_id is not None
    assert direct_marker.read_text(encoding="ascii") == "once"

    receiver = ManagedValue(repo=repo)
    repo.save_object(receiver, deep_capture=True)
    managed_marker = tmp_path / "managed-marker"
    monkeypatch.setenv("DRYML_U5_RAW_MARKER", str(managed_marker))
    managed = strategy.prepare(
        receiver.raw_result, (), {}, repo=repo,
        control_store=None, update_args=False,
    )
    managed_output = strategy.invoke(managed.invocation, repo=repo, update_args=False)
    assert strategy.recover(
        managed_output, managed, repo=repo, args=(), kwargs={},
        return_objects=False, update_args=False,
    ).object_id is not None
    assert managed_marker.read_text(encoding="ascii") == "once"


def _subprocess_invoke(fn, args, repo, spool):
    """Transport one prepared core call through a fresh generic subprocess worker."""
    prepared = SharedDirStoreStrategy().prepare(
        fn, args, {}, repo=repo, control_store=None, update_args=False,
    )
    with Executor(SubProcessConfig(spool_directory=spool)) as executor:
        return executor.run(
            invoke_prepared_call, prepared.invocation,
            worker_setup=prepared.worker_setup(),
        )


def _worker_pid(value):
    """Return a worker fact while preserving an ordinary value argument."""
    return value, os.getpid()


def _reference_only(value: Ref[ObjectRef]):
    """Prove a Ref input remains authority data in the worker."""
    return value.digest()


def test_fresh_subprocess_transports_every_callable_owner_and_shared_capture_graph(tmp_path):
    """Core calls execute after generic setup, not through coordinator imports."""
    repo = Repo(DirStore(tmp_path / "state"))
    calls = []

    def closure(offset):
        return lambda value: calls.append(value) or value + offset

    class Receiver:
        def add(self, value):
            return value + 3

    spool = tmp_path / "spool"
    spool.mkdir()
    observed_value, worker_pid = _subprocess_invoke(_worker_pid, (5,), repo, spool)
    assert observed_value == 5
    assert worker_pid != os.getpid()
    assert _subprocess_invoke(closure(2), (5,), repo, spool) == 7
    assert _subprocess_invoke(lambda value: value * 2, (5,), repo, spool) == 10
    assert _subprocess_invoke(Receiver().add, (5,), repo, spool) == 8
    assert _subprocess_invoke(CallableInstance(6), (5,), repo, spool) == 11
    assert _subprocess_invoke(decorated, (5,), repo, spool) == 6
    assert _subprocess_invoke(DirectMethod(), (5,), repo, spool) == 9

    receiver = ManagedValue(repo=repo)
    repo.save_object(receiver, deep_capture=True)
    assert _subprocess_invoke(receiver.increment, (5,), repo, spool) == 6

    value = CapturedValue(7, repo=repo)
    older = repo.save_object(value, deep_capture=True)
    value.value = 11
    repo.save_object(value, deep_capture=True)
    assert _subprocess_invoke(SlotCallableInstance(older), (older,), repo, spool) == (14, True)
    assert _subprocess_invoke(_reference_only, (older.object,), repo, spool) == older.object.digest()
    assert calls == []


def test_copied_function_wrapper_metadata_rebuilds_its_established_owner(
        tmp_path):
    """
    Ordinary wrappers retain their body while captured function owners rebuild.
    """

    repo = Repo(DirStore(tmp_path / "state"))
    spool = tmp_path / "spool"
    spool.mkdir()

    @functools.wraps(decorated)
    def copied(value):
        """
        Forward through the wrapped function without becoming its raw target.
        """

        return decorated(value)

    assert _subprocess_invoke(copied, (5,), repo, spool) == 6
