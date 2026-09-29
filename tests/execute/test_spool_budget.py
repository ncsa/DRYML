from __future__ import annotations

import threading

import pytest

from dryml.execute._spooling import SpoolBudget
from dryml.execute.config import BackendConfig
from dryml.execute.errors import ExecutionError


class FakeConfig(BackendConfig):
    """Config with no backend side effects."""

    def create_backend(self):
        raise AssertionError


def config(**overrides):
    """Return a small internally consistent budget configuration."""
    values = dict(spool_limit_bytes=300, spool_file_limit=6, preflight_limit=2, invocation_limit_bytes=100, result_limit_bytes=50)
    values.update(overrides)
    return FakeConfig(**values)


def test_matching_leases_share_quota_and_conflicts_preserve_active_generation():
    """A process-wide generation rejects incompatible aggregate settings without reset."""
    first = SpoolBudget.acquire(config())
    second = SpoolBudget.acquire(config())
    try:
        reservation = first.reserve("one", invocation_bytes=100, result_bytes=50)
        with pytest.raises(ExecutionError, match="spool_configuration_conflict"):
            SpoolBudget.acquire(config(spool_limit_bytes=400))
        assert SpoolBudget.snapshot().reserved_bytes == 150
        reservation.release()
    finally:
        second.release()
        first.release()

    replacement = SpoolBudget.acquire(config(spool_limit_bytes=400))
    replacement.release()


def test_reservations_are_atomic_concurrent_and_retain_result_capacity():
    """Concurrent preflights cannot partially exceed byte, file, or preflight quota."""
    lease = SpoolBudget.acquire(config(spool_limit_bytes=300, spool_file_limit=4, preflight_limit=2))
    successes = []
    failures = []

    def reserve(index: int) -> None:
        try:
            successes.append(lease.reserve(f"job-{index}", invocation_bytes=100, result_bytes=50))
        except ExecutionError:
            failures.append(index)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert len(successes) == 2
        assert len(failures) == 2
        assert SpoolBudget.snapshot().reserved_bytes == 300
        for reservation in successes:
            reservation.refund_invocation(10)
        assert SpoolBudget.snapshot().reserved_bytes == 120
    finally:
        for reservation in successes:
            reservation.release()
        lease.release()


def test_stale_reservation_cannot_release_new_generation():
    """Generation identity prevents delayed cleanup from charging a later budget."""
    lease = SpoolBudget.acquire(config())
    reservation = lease.reserve("one", invocation_bytes=100, result_bytes=50)
    reservation.release()
    lease.release()
    later = SpoolBudget.acquire(config(spool_limit_bytes=400))
    try:
        with pytest.raises(ExecutionError, match="already released"):
            reservation.release()
    finally:
        later.release()


def test_barrier_race_has_one_atomic_refund_and_preserves_preflight_limits():
    """Concurrent refund attempts cannot double-credit a live reservation."""
    lease = SpoolBudget.acquire(config(spool_limit_bytes=150, spool_file_limit=2, preflight_limit=1))
    reservation = lease.reserve("one", invocation_bytes=100, result_bytes=50)
    barrier = threading.Barrier(3)
    failures: list[ExecutionError] = []

    def refund() -> None:
        barrier.wait()
        try:
            reservation.refund_invocation(10)
        except ExecutionError as exc:
            failures.append(exc)

    threads = [threading.Thread(target=refund) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    try:
        assert len(failures) == 1
        assert SpoolBudget.snapshot().reserved_bytes == 60
        reservation.release_preflight()
        assert SpoolBudget.snapshot().active_preflights == 0
    finally:
        reservation.release()
        lease.release()
