"""Deterministic core Execute failure ownership and recovery conformance."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

import dryml.core.execute as execute_module
from dryml.core import Repo, Serializable
from dryml.core.execute import CoreExecutionError, CoreOptions, SharedDirStoreStrategy
from dryml.core.execute_codec import decode_outcome
from dryml.core.store.dir import DirStore
from dryml.execute.errors import CleanupError
from dryml.execute.subprocess import SubProcessConfig


class MatrixValue(Serializable):
    """Small durable value used to prove refresh failure does not alter callers."""

    def __init__(self, value=0):
        """Retain the visible state restored through a StateRef."""
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Persist the scalar state using the test Store codec contract."""
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Restore the scalar state from the durable test fixture."""
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def _mutate_pair(first, second):
    """Mutate two independently selected worker values before update publication."""
    first.value += 1
    second.value += 1
    return None


def _raise_secret():
    """Raise a diagnostic-bearing workload error that transport must redact."""
    raise RuntimeError("token=matrix-secret path=/private/matrix payload=captured")


def test_preparation_failure_preserves_primary_error_and_safe_cleanup_evidence(tmp_path, monkeypatch):
    """A failed snapshot close cannot replace the pre-acceptance preparation failure."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    closed = []

    class Storage:
        """Minimal owned snapshot that deterministically fails its release."""

        frozen_storage = None

        def close(self):
            """Record a cleanup attempt and fail without exposing the detail publicly."""
            closed.append(True)
            raise RuntimeError("token=matrix-secret")

    monkeypatch.setattr(execute_module, "prepare_shared_storage", lambda **_: Storage())
    monkeypatch.setattr(
        SharedDirStoreStrategy,
        "prepare",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("call preparation failed")),
    )

    with pytest.raises(ValueError, match="call preparation failed") as raised:
        execute_module._submit_core_call(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not submit")),
            SubProcessConfig(spool_directory=tmp_path / "spool"), CoreOptions(repo=repo),
            _raise_secret, (), kwargs=None, core=None, environment=None, world=None,
            execution_timeout="inherit", stream_output=None, done_callbacks=(), output=None,
            one_off=False,
        )

    assert closed == [True]
    assert isinstance(raised.value.__cause__, CleanupError)
    assert "matrix-secret" not in str(raised.value)
    assert "matrix-secret" not in str(raised.value.__cause__)
    assert raised.value.__context__ is None or "matrix-secret" not in str(raised.value.__context__)


def test_worker_failure_keeps_secret_payload_out_of_core_outcome_and_publishes_nothing(tmp_path):
    """Invocation failure has a typed reason, no authority, and no workload repr."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _raise_secret, (), {}, repo=repo, control_store=None, update_args=False,
    )

    outcome = decode_outcome(
        strategy.invoke(prepared.invocation, repo=repo, update_args=False), repo=repo,
    )

    assert not outcome["success"]
    assert outcome["reason"] == "RuntimeError"
    assert outcome["publications"] == []
    with pytest.raises(CoreExecutionError) as raised:
        strategy.recover(
            strategy.invoke(prepared.invocation, repo=repo, update_args=False), prepared,
            repo=repo, args=(), kwargs={}, return_objects=False, update_args=False,
        )
    assert raised.value.phase == "invoke"
    assert "matrix-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("failure_index", "statuses"),
    ((0, ["preflight_failed", "skipped"]), (1, ["pending", "preflight_failed"])),
)
def test_refresh_preflight_failure_skips_all_restores_and_retains_exact_ledger(
        tmp_path, monkeypatch, failure_index, statuses,
):
    """Each preflight position leaves caller payloads untouched and is never retried."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    first = MatrixValue(1, repo=repo)
    second = MatrixValue(2, repo=repo)
    repo.save(first, deep_capture=True)
    repo.save(second, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _mutate_pair, (first, second), {}, repo=repo, control_store=None, update_args=True,
    )
    recovery = strategy.bind_recovery(prepared, args=(first, second), kwargs={})
    outcome = strategy.invoke(prepared.invocation, repo=repo, update_args=True)
    load = repo.load_state_ref
    restores = []
    unavailable = (first, second)[failure_index]

    def fail_second(state, **kwargs):
        """Reject the second required snapshot before any caller restore begins."""
        if state.object == unavailable.object_ref:
            raise OSError("StateRef unavailable")
        return load(state, **kwargs)

    monkeypatch.setattr(repo, "load_state_ref", fail_second)
    monkeypatch.setattr(repo, "restore_state_ref_into", lambda *args: restores.append(args))

    with pytest.raises(CoreExecutionError) as raised:
        strategy.recover(
            outcome, prepared, repo=repo, args=(first, second), kwargs={},
            return_objects=False, update_args=True, _recovery=recovery,
        )

    assert raised.value.phase == "refresh"
    assert [entry.status for entry in raised.value.evidence.refreshes] == statuses
    assert restores == []
    assert (first.value, second.value) == (1, 2)
    with pytest.raises(CoreExecutionError) as repeated:
        strategy.recover(
            outcome, prepared, repo=repo, args=(first, second), kwargs={},
            return_objects=False, update_args=True, _recovery=recovery,
        )
    assert repeated.value is raised.value
