"""Process and fork lifecycle coverage for retained advisory leases."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from dryml.locking import FileLock


def _hold_then_exit(path: str, ready, release) -> None:
    """Hold one lease until directed to exit without user-space cleanup."""

    lock = FileLock(path)
    assert lock.acquire()
    ready.set()
    assert release.wait(10)
    os._exit(0)


def test_spawned_owner_exit_releases_the_kernel_lease(tmp_path):
    """A spawned owner's process exit makes its retained lock available again."""

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    owner = context.Process(target=_hold_then_exit, args=(str(tmp_path / "owner.lock"), ready, release))
    owner.start()
    try:
        assert ready.wait(10)
        contender = FileLock(tmp_path / "owner.lock")
        assert not contender.acquire(blocking=False)
        release.set()
        owner.join(10)
        assert owner.exitcode == 0
        assert contender.acquire(blocking=False)
        contender.release()
    finally:
        release.set()
        if owner.is_alive():
            owner.terminate()
            owner.join(5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this host")
def test_fork_child_closes_inherited_lease_without_unlocking_parent(tmp_path):
    """Fork cleanup drops child descriptors but leaves the parent lease effective."""

    path = tmp_path / "fork.lock"
    owner = FileLock(path)
    assert owner.acquire()
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            acquired_after_cleanup = owner.acquire(blocking=False)
            contender = FileLock(path)
            blocked_by_parent = not contender.acquire(blocking=False)
            contender.release()
            os.write(write_fd, f"{acquired_after_cleanup}:{blocked_by_parent}".encode("ascii"))
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        assert os.read(read_fd, 32) == b"False:True"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        contender = FileLock(path)
        assert not contender.acquire(blocking=False)
        contender.release()
    finally:
        os.close(read_fd)
        owner.release()
