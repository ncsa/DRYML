"""Dependency-free accounting checks for generation-qualified Ray charges."""

from __future__ import annotations

from dryml.execute.accounting import ResourceAuthority
from dryml.execute.models import ResourceAmounts


def test_ray_net_availability_does_not_double_subtract_confirmed_charge():
    """Ray scheduler availability is net, so only unrepresented reservations subtract."""
    authority = ResourceAuthority()
    total = ResourceAmounts(4.0, None, {}, {})
    charge = authority.reserve("submission", ResourceAmounts(2.0, None, {}, {}), generation="cluster", attempt="0", total=total)
    assert charge is not None
    assert authority.mark_submitted("submission", generation="cluster", attempt="0")
    assert authority.confirm_grant("submission", generation="cluster", attempt="0", worker_id="ray:worker", resources=ResourceAmounts(2.0, None, {}, {}))

    snapshot = authority.snapshot(total=total, native_available=ResourceAmounts(2.0, None, {}, {}), native_available_is_net=True)

    assert snapshot.available.cpus == 2.0


def test_ray_net_availability_caps_sparse_accelerator_and_named_capacity():
    """Ray net capacity stays capped without subtracting confirmed sparse charges twice."""
    authority = ResourceAuthority()
    total = ResourceAmounts(4.0, None, {"gpu": 4.0}, {"license": 3.0})
    requested = ResourceAmounts(1.0, None, {"gpu": 1.0}, {"license": 1.0})
    charge = authority.reserve("submission", requested, generation="cluster", attempt="0", total=total)
    assert charge is not None
    assert authority.mark_submitted("submission", generation="cluster", attempt="0")
    assert authority.confirm_grant("submission", generation="cluster", attempt="0", worker_id="ray:worker", resources=requested)

    snapshot = authority.snapshot(
        total=total,
        native_available=ResourceAmounts(3.0, None, {"gpu": 4.0}, {"license": 3.0}),
        native_available_is_net=True,
    )

    assert snapshot.available.cpus == 3.0
    assert snapshot.available.accelerators["gpu"] == 3.0
    assert snapshot.available.named["license"] == 2.0


def test_ray_unqualified_release_retains_an_unconfirmed_charge():
    """Capacity changes cannot release a Ray reservation without worker evidence."""
    authority = ResourceAuthority()
    total = ResourceAmounts(1.0, None, {}, {})
    assert authority.reserve("submission", ResourceAmounts(1.0, None, {}, {}), generation="cluster", attempt="0", total=total)
    assert authority.mark_submitted("submission", generation="cluster", attempt="0")

    assert not authority.release("submission", generation="cluster", attempt="0", worker_id=None)
    assert authority.snapshot(total=total).allocations[0].state == "unconfirmed"
