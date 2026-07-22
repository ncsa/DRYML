from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from dryml.core.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedLeaseConflictError,
    ManagedOperationStore,
    ManagedTakeoverRequiredError,
    OperationKey,
    StaleManagedLeaseError,
)


KEY = OperationKey(format_cdef_id("5" * 64), "compute")
FP = "managed-declaration-v1-" + "6" * 64


def _operation(path):
    return ManagedOperationStore(DirStore(path)).operation(KEY, FP)


def _holder(path, *, exit_without_release=False):
    script = r"""
import json
import sys
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedOperationStore, OperationKey

path, producer, fingerprint, abrupt = sys.argv[1:]
operation = ManagedOperationStore(DirStore(path)).operation(
    OperationKey(producer, "compute"), fingerprint
)
lease = operation.acquire()
print(json.dumps({"epoch": lease.epoch}), flush=True)
command = sys.stdin.readline().strip()
if abrupt == "yes":
    import os
    os._exit(0)
lease.release()
"""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            KEY.producer_cdef_id,
            FP,
            "yes" if exit_without_release else "no",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_first_acquisition_release_and_monotonic_fence(tmp_path):
    operation = _operation(tmp_path / "store")
    first = operation.acquire()
    assert first.epoch == 1
    first.release()

    with operation.acquire() as second:
        assert second.epoch == 2


def test_concurrent_process_gets_explicit_conflict_and_live_takeover_cannot_break_lock(tmp_path):
    path = tmp_path / "store"
    holder = _holder(path)
    first = json.loads(holder.stdout.readline())
    try:
        with pytest.raises(ManagedLeaseConflictError, match="already owned"):
            _operation(path).acquire()
        with pytest.raises(ManagedLeaseConflictError, match="stop the current owner"):
            _operation(path).acquire(takeover=True)
    finally:
        holder.stdin.write("release\n")
        holder.stdin.flush()
        stdout, stderr = holder.communicate(timeout=10)
        assert holder.returncode == 0, stdout + stderr

    with _operation(path).acquire() as recovered:
        assert recovered.epoch > first["epoch"]


def test_process_death_releases_lock_and_recovery_advances_fence(tmp_path):
    path = tmp_path / "store"
    holder = _holder(path, exit_without_release=True)
    first = json.loads(holder.stdout.readline())
    holder.stdin.write("die\n")
    holder.stdin.flush()
    stdout, stderr = holder.communicate(timeout=10)
    assert holder.returncode == 0, stdout + stderr

    with _operation(path).acquire() as recovered:
        assert recovered.epoch == first["epoch"] + 1


def test_live_released_owner_requires_explicit_operator_takeover(tmp_path):
    operation = _operation(tmp_path / "store")
    former = operation.acquire()
    former._lock.release()

    with pytest.raises(ManagedTakeoverRequiredError, match="operator takeover"):
        operation.acquire()
    current = operation.acquire(takeover=True)
    try:
        with pytest.raises(StaleManagedLeaseError):
            former.prepare(resumable=True)
        assert current.prepare(resumable=True).action == "start"
    finally:
        current.release()
        with pytest.raises(StaleManagedLeaseError):
            former.release()


def test_stale_former_writer_is_rejected_after_new_fence(tmp_path):
    operation = _operation(tmp_path / "store")
    former = operation.acquire()
    former.release()
    current = operation.acquire()
    try:
        with pytest.raises(StaleManagedLeaseError):
            former.prepare(resumable=True)
        decision = current.prepare(resumable=True)
        assert decision.action == "start"
    finally:
        current.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process fork")
def test_forked_worker_cannot_inherit_coordinator_mutation_authority(tmp_path):
    operation = _operation(tmp_path / "store")
    lease = operation.acquire()
    pid = os.fork()
    if pid == 0:
        try:
            lease.prepare(resumable=True)
        except StaleManagedLeaseError:
            os._exit(0)
        except Exception:
            os._exit(2)
        os._exit(1)
    _, status = os.waitpid(pid, 0)
    try:
        assert os.waitstatus_to_exitcode(status) == 0
        assert operation.history() == ()
    finally:
        lease.release()
