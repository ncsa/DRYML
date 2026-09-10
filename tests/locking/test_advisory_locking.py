"""Contract tests for the dependency-light shared advisory-lock owner."""

from __future__ import annotations

import errno
import os
import threading

import pytest

import dryml.locking as locking
from dryml.locking import (
    FileLock,
    LockError,
    interprocess_lock,
    interprocess_read_lock,
    supports_advisory_locking,
    try_lock_file,
    unlock_file,
)


def test_store_compatibility_locks_preserve_writer_recursion_and_shared_reads(tmp_path):
    """Store wrappers retain same-thread recursion and POSIX shared-reader overlap."""

    path = tmp_path / "store.lock"
    with interprocess_lock(path):
        with interprocess_lock(path):
            assert path.exists()

    with interprocess_read_lock(path):
        if locking.fcntl is not None:
            with interprocess_read_lock(path):
                pass


def test_file_lock_is_non_reentrant_and_release_is_idempotent(tmp_path):
    """A retained lease rejects repeat acquisition and closes only its own handle."""

    lock = FileLock(tmp_path / "lease.lock")

    assert lock.acquire()
    with pytest.raises(LockError, match="already acquired"):
        lock.acquire()
    lock.release()
    lock.release()

    with FileLock(tmp_path / "lease.lock") as held:
        assert held.acquire is not None
    after_context = FileLock(tmp_path / "lease.lock")
    assert after_context.acquire(blocking=False)
    after_context.release()


def test_nonblocking_file_lock_reports_only_contention(tmp_path):
    """Separate same-process leases contend without treating contention as failure."""

    path = tmp_path / "lease.lock"
    owner = FileLock(path)
    contender = FileLock(path)
    assert owner.acquire()
    try:
        assert not contender.acquire(blocking=False)
    finally:
        contender.release()
        owner.release()

    assert contender.acquire(blocking=False)
    contender.release()


def test_borrowed_descriptor_lock_never_closes_or_reopens_descriptor(tmp_path):
    """Borrowed-FD calls leave descriptor lifecycle and path ownership to callers."""

    path = tmp_path / "claim.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        assert try_lock_file(fd)
        assert os.fstat(fd).st_ino == path.stat().st_ino
        contender = FileLock(path)
        try:
            assert not contender.acquire(blocking=False)
        finally:
            contender.release()
        unlock_file(fd)
        assert os.fstat(fd).st_ino == path.stat().st_ino
    finally:
        os.close(fd)


def test_lock_adapter_classifies_permission_as_an_error(tmp_path, monkeypatch):
    """POSIX permission errors raise rather than masquerading as contention."""

    class FailingFcntl:
        LOCK_EX = 1
        LOCK_SH = 2
        LOCK_NB = 4
        LOCK_UN = 8

        @staticmethod
        def flock(fd, operation):
            raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(locking, "fcntl", FailingFcntl)
    fd = os.open(tmp_path / "claim.lock", os.O_CREAT | os.O_RDWR)
    try:
        with pytest.raises(LockError, match="permission"):
            try_lock_file(fd)
    finally:
        os.close(fd)


def test_windows_adapter_does_not_classify_preparation_permission_error_as_contention(tmp_path, monkeypatch):
    """Only the Windows native lock call may report nonblocking contention."""

    class FakeMSVCRT:
        LK_NBLCK = 1

        @staticmethod
        def locking(fd, operation, size):
            raise AssertionError("Windows lock preparation must fail before locking")

    def deny_lseek(fd, offset, whence):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(locking, "fcntl", None)
    monkeypatch.setattr(locking, "msvcrt", FakeMSVCRT)
    monkeypatch.setattr(locking.os, "lseek", deny_lseek)
    fd = os.open(tmp_path / "claim.lock", os.O_CREAT | os.O_RDWR)
    try:
        with pytest.raises(LockError, match="preparation failed"):
            try_lock_file(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_acquire_releases_its_local_reservation_and_descriptor(tmp_path, monkeypatch, interruption):
    """An interrupted native acquire leaves the path available to a new lease."""

    path = tmp_path / "interrupted.lock"

    def interrupt(fd, *, shared, blocking):
        raise interruption("injected native acquire interruption")

    monkeypatch.setattr(locking, "_lock_file", interrupt)
    with pytest.raises(interruption, match="native acquire"):
        FileLock(path).acquire()
    monkeypatch.undo()

    recovered = FileLock(path)
    assert recovered.acquire(blocking=False)
    recovered.release()


def test_windows_adapter_treats_native_eacces_as_contention(tmp_path, monkeypatch):
    """The Windows locking call retains its native EACCES contention behavior."""

    class FakeMSVCRT:
        LK_NBLCK = 1

        @staticmethod
        def locking(fd, operation, size):
            raise OSError(errno.EACCES, "already locked")

    monkeypatch.setattr(locking, "fcntl", None)
    monkeypatch.setattr(locking, "msvcrt", FakeMSVCRT)
    fd = os.open(tmp_path / "claim.lock", os.O_CREAT | os.O_RDWR)
    try:
        assert not try_lock_file(fd)
    finally:
        os.close(fd)


def test_windows_adapter_does_not_treat_other_winerrors_as_contention(tmp_path, monkeypatch):
    """Windows access-denied errors with a non-contention code remain failures."""

    class FakeMSVCRT:
        LK_NBLCK = 1

        @staticmethod
        def locking(fd, operation, size):
            error = OSError(errno.EACCES, "access denied")
            error.winerror = 5
            raise error

    monkeypatch.setattr(locking, "fcntl", None)
    monkeypatch.setattr(locking, "msvcrt", FakeMSVCRT)
    fd = os.open(tmp_path / "claim.lock", os.O_CREAT | os.O_RDWR)
    try:
        with pytest.raises(LockError, match="Windows advisory lock failed"):
            try_lock_file(fd)
    finally:
        os.close(fd)


def test_cross_thread_release_preserves_the_acquiring_thread_slot(tmp_path):
    """A different thread cannot release a lease or its local reservation."""

    path = tmp_path / "owner.lock"
    owner = FileLock(path)
    assert owner.acquire()
    errors = []

    def release_from_other_thread():
        try:
            owner.release()
        except LockError as error:
            errors.append(error)

    releaser = threading.Thread(target=release_from_other_thread)
    releaser.start()
    releaser.join(10)

    assert not releaser.is_alive()
    assert len(errors) == 1
    contender = FileLock(path)
    assert not contender.acquire(blocking=False)
    owner.release()
    assert contender.acquire(blocking=False)
    contender.release()


def test_concurrent_acquire_on_one_instance_keeps_one_lease(tmp_path):
    """One FileLock instance cannot retain duplicate descriptors across threads."""

    lock = FileLock(tmp_path / "concurrent.lock")
    start = threading.Barrier(3)
    acquired = threading.Event()
    rejected = threading.Event()
    release = threading.Event()

    def acquire_and_release():
        start.wait()
        try:
            assert lock.acquire()
        except LockError:
            rejected.set()
            return
        acquired.set()
        assert release.wait(10)
        lock.release()

    first = threading.Thread(target=acquire_and_release)
    second = threading.Thread(target=acquire_and_release)
    first.start()
    second.start()
    start.wait()
    assert acquired.wait(10)
    assert rejected.wait(10)
    release.set()
    first.join(10)
    second.join(10)

    assert not first.is_alive()
    assert not second.is_alive()
    replacement = FileLock(tmp_path / "concurrent.lock")
    assert replacement.acquire(blocking=False)
    replacement.release()


def test_same_thread_exclusive_then_shared_fails_without_waiting(tmp_path):
    """A non-reentrant incompatible request raises instead of waiting forever."""

    path = tmp_path / "incompatible.lock"
    exclusive = FileLock(path)
    shared = FileLock(path, shared=True)
    assert exclusive.acquire()
    try:
        with pytest.raises(LockError, match="already owns"):
            shared.acquire()
    finally:
        exclusive.release()


def test_windows_adapter_locks_beyond_eof_without_writing(tmp_path, monkeypatch):
    """Windows locking leaves an empty lock file unchanged while locking byte zero."""

    class FakeMSVCRT:
        LK_LOCK = 1
        LK_NBLCK = 2
        LK_UNLCK = 3
        calls = []

        @classmethod
        def locking(cls, fd, operation, size):
            cls.calls.append((fd, operation, size))

    class WindowsOSProxy:
        SEEK_SET = os.SEEK_SET
        fstat = staticmethod(os.fstat)
        lseek = staticmethod(os.lseek)

        @staticmethod
        def write(fd, data):
            raise OSError(errno.EACCES, "lock-file writes are forbidden")

    monkeypatch.setattr(locking, "fcntl", None)
    monkeypatch.setattr(locking, "msvcrt", FakeMSVCRT)
    monkeypatch.setattr(locking, "os", WindowsOSProxy)
    path = tmp_path / "windows.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        assert try_lock_file(fd)
        unlock_file(fd)
        assert os.fstat(fd).st_size == 0
    finally:
        os.close(fd)

    assert [call[1] for call in FakeMSVCRT.calls] == [FakeMSVCRT.LK_NBLCK, FakeMSVCRT.LK_UNLCK]
    assert [call[2] for call in FakeMSVCRT.calls] == [1, 1]


def test_suitability_check_is_limited_to_a_local_directory_shape(tmp_path):
    """Path suitability is not advertised as a filesystem locking guarantee."""

    assert supports_advisory_locking(tmp_path / "lock")
    assert not supports_advisory_locking(tmp_path / "missing" / "lock")
