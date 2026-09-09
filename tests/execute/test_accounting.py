"""Proof for coordinator-scoped backend resource accounting."""

from __future__ import annotations

from threading import Barrier, Thread

from dryml.execute.accounting import ResourceAuthorityRegistry
from dryml.execute.models import ResourceAmounts


def amounts(cpus: float | None) -> ResourceAmounts:
    """Build a compact CPU-only resource observation."""
    return ResourceAmounts(cpus, None, {}, {})


def test_shared_authority_prevents_sibling_double_spend_and_release_is_qualified():
    """Reservations share one backend authority and PID alone cannot release them."""
    registry = ResourceAuthorityRegistry()
    left = registry.get("subprocess", "local")
    right = registry.get("subprocess", "local")
    reservation = left.reserve("one", amounts(2), generation="g1", attempt="a1", total=amounts(4))
    assert reservation is not None
    assert right.reserve("two", amounts(3), generation="g1", attempt="a2", total=amounts(4)) is None
    assert not left.release("one", generation="g1", attempt="a1", worker_id=None, never_submitted=False)
    assert left.release("one", generation="g1", attempt="a1", worker_id=None, never_submitted=True)


def test_backend_types_have_no_common_allocator_and_unknown_is_not_free():
    """Different backends are explicitly independent and unknown capacity stays unknown."""
    registry = ResourceAuthorityRegistry()
    local = registry.get("subprocess", "local")
    ray = registry.get("ray", "cluster")
    assert local.reserve("local", amounts(4), generation="g", attempt="a", total=amounts(4))
    assert ray.reserve("ray", amounts(4), generation="g", attempt="a", total=amounts(4))
    snapshot = local.snapshot(total=amounts(None))
    assert snapshot.available.cpus is None
    assert snapshot.accounting_scope == "coordinator-and-backend"


def test_known_sparse_accelerator_and_named_capacity_accepts_first_reservation():
    """An empty charge ledger leaves known accelerator and named capacity usable."""
    authority = ResourceAuthorityRegistry().get("ray", "cluster")
    total = ResourceAmounts(4, None, {"gpu": 2}, {"license": 3})

    snapshot = authority.snapshot(total=total)
    assert snapshot.available.cpus == 4
    assert snapshot.available.accelerators["gpu"] == 2
    assert snapshot.available.named["license"] == 3

    reservation = authority.reserve(
        "first-gpu",
        ResourceAmounts(0, None, {"gpu": 1}, {"license": 1}),
        generation="g",
        attempt="a",
        total=total,
    )
    assert reservation is not None


def test_explicit_unknown_charge_remains_unknown_while_sparse_charge_is_zero():
    """Only an explicit unknown charge hides known capacity in that dimension."""
    authority = ResourceAuthorityRegistry().get("ray", "cluster")
    total = ResourceAmounts(4, None, {"gpu": 2}, {"license": 3})
    assert authority.reserve(
        "unknown-gpu",
        ResourceAmounts(1, None, {"gpu": None}, {}),
        generation="g",
        attempt="a",
        total=total,
    )

    snapshot = authority.snapshot(total=total)
    assert snapshot.available.cpus == 3
    assert snapshot.available.accelerators["gpu"] is None
    assert snapshot.available.named["license"] == 3


def test_native_net_capacity_does_not_subtract_running_charge_twice():
    """A native net observation deducts only reservations it cannot already include."""
    authority = ResourceAuthorityRegistry().get("ray", "cluster")
    assert authority.reserve("running", amounts(2), generation="g", attempt="a", total=amounts(4))
    assert authority.confirm_grant("running", generation="g", attempt="a", worker_id="worker", resources=amounts(2))
    snapshot = authority.snapshot(total=amounts(4), native_available=amounts(2), native_available_is_net=True)
    assert snapshot.available.cpus == 2
    assert not authority.release("running", generation="g", attempt="a", worker_id="other", qualified_terminal=True)


def test_unknown_total_refuses_reservation_and_grants_cannot_enlarge_or_rebind():
    """Unknown capacity, mismatched workers, and larger grants retain the charge."""
    authority = ResourceAuthorityRegistry().get("subprocess", "local")
    assert authority.reserve("unknown", amounts(1), generation="g", attempt="a") is None
    assert authority.reserve("one", amounts(1), generation="g", attempt="a", total=amounts(2))
    assert not authority.confirm_grant("one", generation="g", attempt="a", worker_id="worker", resources=amounts(2))
    assert authority.confirm_grant("one", generation="g", attempt="a", worker_id="worker", resources=amounts(1))
    assert not authority.confirm_grant("one", generation="g", attempt="a", worker_id="other", resources=amounts(1))


def test_concurrent_reservations_do_not_double_spend_known_capacity():
    """The authority lock serializes competing capacity checks and charges."""
    authority = ResourceAuthorityRegistry().get("subprocess", "local")
    barrier = Barrier(2)
    results = []

    def reserve(submission_id: str) -> None:
        barrier.wait()
        results.append(authority.reserve(submission_id, amounts(1), generation="g", attempt=submission_id, total=amounts(1)))

    threads = [Thread(target=reserve, args=(name,)) for name in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(value is not None for value in results) == 1
