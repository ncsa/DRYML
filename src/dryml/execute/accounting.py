"""Coordinator-scoped in-memory resource reservations for one backend authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import Condition, RLock

from dryml.worlds import LocalResourceInventory, WorldAllocation, WorldRequirement, assign_local_world, synthesize

from .models import ActiveAllocation, ExecutionIssue, ResourceAmounts, ResourceSnapshot


@dataclass(frozen=True, slots=True)
class Reservation:
    """Identify one generation-qualified backend resource charge and its evidence."""

    submission_id: str
    generation: str
    attempt: str
    resources: ResourceAmounts
    executor_id: str | None = None
    worker_id: str | None = None
    backend_job_id: str | None = None
    pid: int | None = None
    allocation: WorldAllocation | None = None
    submitted: bool = False
    state: str = "reserved"


class ResourceAuthority:
    """Track one backend identity's live coordinator charges without disk state.

    The authority is process-local and backend-specific. It provides conservative
    accounting, not physical exclusion across coordinators or backend types.
    """

    def __init__(self) -> None:
        """Create an empty in-memory authority with no native side effects."""
        self._lock = RLock()
        self._changed = Condition(self._lock)
        self._reservations: dict[str, Reservation] = {}
        self._revision = 0

    def reserve(
        self,
        submission_id: str,
        resources: ResourceAmounts,
        *,
        generation: str,
        attempt: str,
        total: ResourceAmounts | None = None,
        executor_id: str | None = None,
    ) -> Reservation | None:
        """Acquire a charge only when explicit native capacity can cover it.

        Args:
            submission_id: Exact backend submission identifier.
            resources: Requested charge with unknown dimensions preserved.
            generation: Exact backend generation identifier.
            attempt: Exact submission-attempt identifier.
            total: Fresh owner-proven native capacity. ``None`` refuses rather
                than treating unknown capacity as available.
            executor_id: Optional exact owner ID used to filter a caller view.

        Returns:
            The reservation, or ``None`` for duplicate, unknown, or insufficient
            capacity.

        Side Effects:
            Retains an in-memory coordinator charge on success.
        """
        _handle("submission_id", submission_id)
        _handle("generation", generation)
        _handle("attempt", attempt)
        if executor_id is not None:
            _handle("executor_id", executor_id)
        if not isinstance(resources, ResourceAmounts):
            raise TypeError("resources must be ResourceAmounts")
        if total is not None and not isinstance(total, ResourceAmounts):
            raise TypeError("total must be ResourceAmounts or None")
        if total is None:
            return None
        with self._changed:
            if submission_id in self._reservations:
                return None
            if not _fits(_subtract(total, _sum(item.resources for item in self._reservations.values())), resources):
                return None
            reservation = Reservation(submission_id, generation, attempt, resources, executor_id=executor_id)
            self._reservations[submission_id] = reservation
            self._transition()
            return reservation

    def reserve_local(
        self,
        submission_id: str,
        world: WorldRequirement,
        inventory: LocalResourceInventory,
        *,
        generation: str,
        attempt: str,
        executor_id: str | None = None,
    ) -> Reservation | None:
        """Atomically select an exact free local allocation and charge it.

        The supplied inventory is a fresh Execute-owned observation.  Selection is
        pure and occurs while the authority lock is held, so sibling subprocess
        executors cannot choose the same CPU or accelerator ID.  An outstanding
        unconstrained/unknown charge deliberately blocks constrained admission: it
        cannot be treated as free affinity capacity.
        """
        _handle("submission_id", submission_id)
        _handle("generation", generation)
        _handle("attempt", attempt)
        if executor_id is not None:
            _handle("executor_id", executor_id)
        if not isinstance(world, WorldRequirement) or not isinstance(inventory, LocalResourceInventory):
            raise TypeError("world and inventory must be owner-defined local values")
        with self._changed:
            if submission_id in self._reservations or any(_unknown_charge(item) for item in self._reservations.values()):
                return None
            used_cpus = {cpu for item in self._reservations.values() if item.allocation is not None for process in _processes(item.allocation) for cpu in process.cpus}
            used_accelerators = {
                (kind, device)
                for item in self._reservations.values()
                if item.allocation is not None
                for process in _processes(item.allocation)
                for kind, devices in process.accelerators.items()
                for device in devices
            }
            remaining_cpus = tuple(cpu for cpu in inventory.cpus if cpu not in used_cpus)
            # LocalResourceInventory intentionally cannot represent zero CPUs. A
            # constrained local worker always needs a concrete CPU binding.
            if not remaining_cpus:
                return None
            remaining = LocalResourceInventory(
                remaining_cpus,
                {
                    kind: tuple(device for device in devices if (kind, device) not in used_accelerators)
                    for kind, devices in inventory.accelerators.items()
                },
                inventory.memory,
                {
                    kind: {
                        device: amount
                        for device, amount in values.items()
                        if (kind, device) not in used_accelerators
                    }
                    for kind, values in inventory.accelerator_memory.items()
                },
                inventory.metadata,
            )
            synthesis = synthesize(world, inventory=remaining)
            if not synthesis.ok or synthesis.world is None:
                return None
            try:
                allocation = assign_local_world(synthesis.world, inventory=remaining)
            except Exception:
                return None
            resources = _allocation_amounts(allocation)
            reservation = Reservation(
                submission_id, generation, attempt, resources,
                executor_id=executor_id, allocation=allocation,
            )
            self._reservations[submission_id] = reservation
            self._transition()
            return reservation

    def mark_submitted(self, submission_id: str, *, generation: str, attempt: str, backend_job_id: str | None = None) -> bool:
        """Record that a native submission may exist before a worker grant arrives.

        Returns false for mismatched handles, preventing a queued native task from
        later being released through the never-submitted path.
        """
        _handle("submission_id", submission_id)
        _handle("generation", generation)
        _handle("attempt", attempt)
        if backend_job_id is not None:
            _handle("backend_job_id", backend_job_id)
        with self._changed:
            current = self._matching(submission_id, generation, attempt)
            if current is None or current.submitted:
                return False
            self._reservations[submission_id] = replace(current, submitted=True, backend_job_id=backend_job_id, state="reserved")
            self._transition()
            return True

    def confirm_grant(
        self,
        submission_id: str,
        *,
        generation: str,
        attempt: str,
        worker_id: str,
        resources: ResourceAmounts,
        backend_job_id: str | None = None,
        pid: int | None = None,
        allocation: WorldAllocation | None = None,
    ) -> bool:
        """Confirm an actual grant without rebinding or enlarging a charge.

        A grant proves submission. It may reduce a reservation but cannot increase
        any known request, overwrite a worker identity, or replace a completed
        grant with a different attempt.
        """
        _handle("submission_id", submission_id)
        _handle("generation", generation)
        _handle("attempt", attempt)
        _handle("worker_id", worker_id)
        if backend_job_id is not None:
            _handle("backend_job_id", backend_job_id)
        if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
            raise ValueError("pid must be a positive integer or None")
        if not isinstance(resources, ResourceAmounts):
            raise TypeError("resources must be ResourceAmounts")
        if allocation is not None and not isinstance(allocation, WorldAllocation):
            raise TypeError("allocation must be WorldAllocation or None")
        with self._changed:
            current = self._matching(submission_id, generation, attempt)
            if current is None or current.state != "reserved" or (current.worker_id is not None and current.worker_id != worker_id):
                return False
            if not _fits(current.resources, resources):
                return False
            if current.allocation is not None and (allocation is None or not _same_bindings(current.allocation, allocation)):
                return False
            self._reservations[submission_id] = replace(current, submitted=True, worker_id=worker_id, backend_job_id=backend_job_id or current.backend_job_id, pid=pid, allocation=allocation, resources=resources, state="running")
            self._transition()
            return True

    def release(
        self,
        submission_id: str,
        *,
        generation: str,
        attempt: str,
        worker_id: str | None,
        never_submitted: bool = False,
        qualified_terminal: bool = False,
    ) -> bool:
        """Release only matching never-submitted or qualified terminal charges.

        A cancellation request, absent PID, queued task, or unqualified terminal
        response retains an ``unconfirmed`` charge for conservative reconciliation.
        """
        _handle("submission_id", submission_id)
        _handle("generation", generation)
        _handle("attempt", attempt)
        if worker_id is not None:
            _handle("worker_id", worker_id)
        if not isinstance(never_submitted, bool) or not isinstance(qualified_terminal, bool):
            raise TypeError("release qualification flags must be bool")
        with self._changed:
            current = self._matching(submission_id, generation, attempt)
            if current is None:
                return False
            if never_submitted and not current.submitted:
                del self._reservations[submission_id]
                self._transition()
                return True
            # A locally owned group that stopped before READY has no confirmed
            # worker identity, but it is still qualified release evidence for its
            # own pre-launch reservation.  PID/cancellation alone remains invalid.
            if qualified_terminal and (current.worker_id == worker_id or current.worker_id is None):
                del self._reservations[submission_id]
                self._transition()
                return True
            if current.state != "unconfirmed":
                self._reservations[submission_id] = replace(current, state="unconfirmed")
                self._transition()
            return False

    def revision(self) -> int:
        """Return the current reservation-transition revision without probing I/O."""
        with self._lock:
            return self._revision

    def wait_for_change(self, revision: int, timeout: float) -> bool:
        """Wait for a transition after ``revision`` and report whether one occurred.

        Callers can use a short explicit polling timeout to notice cancellation or
        admission expiry without repeating an external inventory observation.
        """
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("timeout must be a nonnegative duration")
        with self._changed:
            return self._changed.wait_for(lambda: self._revision != revision, timeout)

    def has_unknown_charge(self) -> bool:
        """Return whether a live unbound charge prevents safe exact allocation."""
        with self._lock:
            return any(_unknown_charge(item) for item in self._reservations.values())

    def snapshot(
        self,
        *,
        total: ResourceAmounts,
        native_available: ResourceAmounts | None = None,
        native_available_is_net: bool = False,
        executor_id: str | None = None,
    ) -> ResourceSnapshot:
        """Return a fresh backend-scoped capacity observation without reserving.

        ``available`` always accounts for every authority charge, while
        ``allocated`` and ``allocations`` can be filtered to one executor owner.
        A net native observation is capped by total-minus-all-charges and deducts
        only unrepresented reserved work, so advisory capacity cannot free an
        unresolved grant.
        """
        if not isinstance(total, ResourceAmounts):
            raise TypeError("total must be ResourceAmounts")
        if native_available is not None and not isinstance(native_available, ResourceAmounts):
            raise TypeError("native_available must be ResourceAmounts or None")
        if not isinstance(native_available_is_net, bool):
            raise TypeError("native_available_is_net must be bool")
        if executor_id is not None:
            _handle("executor_id", executor_id)
        with self._lock:
            reservations = tuple(self._reservations.values())
        charged = _sum(item.resources for item in reservations)
        visible = reservations if executor_id is None else tuple(item for item in reservations if item.executor_id == executor_id)
        allocated = _sum(item.resources for item in visible)
        total_limited = _subtract(total, charged)
        if native_available is None:
            available = total_limited
        elif native_available_is_net:
            unrepresented = _sum(item.resources for item in reservations if item.state == "reserved")
            available = _minimum(total_limited, _subtract(native_available, unrepresented))
        else:
            available = _minimum(total_limited, _subtract(native_available, charged))
        allocations = tuple(
            ActiveAllocation(item.submission_id, item.backend_job_id, item.worker_id, item.pid, item.resources, item.allocation, item.state if item.state in {"reserved", "running", "unconfirmed"} else "unconfirmed")
            for item in visible
        )
        complete = _amounts_complete(total) and (native_available is None or _amounts_complete(native_available)) and all(item.state != "unconfirmed" and _amounts_complete(item.resources) for item in reservations)
        issues = () if complete else (ExecutionIssue("resource_evidence_incomplete", "resource capacity or outstanding charge evidence is incomplete"),)
        return ResourceSnapshot(datetime.now(timezone.utc), "coordinator-and-backend", None, total, allocated, available, allocations, complete, issues)

    def _matching(self, submission_id: str, generation: str, attempt: str) -> Reservation | None:
        """Return a reservation only for its exact generation-qualified handles."""
        current = self._reservations.get(submission_id)
        return current if current is not None and current.generation == generation and current.attempt == attempt else None

    def _transition(self) -> None:
        """Publish one actual reservation-state transition to waiting admitters."""
        self._revision += 1
        self._changed.notify_all()


class ResourceAuthorityRegistry:
    """Share authorities only among equal backend type and identity in one process."""

    def __init__(self) -> None:
        """Create an empty coordinator-local authority registry."""
        self._lock = RLock()
        self._authorities: dict[tuple[str, str], ResourceAuthority] = {}

    def get(self, backend_type: str, backend_identity: str) -> ResourceAuthority:
        """Return the shared authority for one exact backend type and identity."""
        _handle("backend_type", backend_type)
        _handle("backend_identity", backend_identity)
        key = (backend_type, backend_identity)
        with self._lock:
            if key not in self._authorities:
                self._authorities[key] = ResourceAuthority()
            return self._authorities[key]


RESOURCE_AUTHORITIES = ResourceAuthorityRegistry()
"""Process-global registry future backend instances use for common authorities."""


def _handle(name: str, value: str) -> None:
    """Require an exact non-empty text owner handle."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _sum(values: Iterable[ResourceAmounts]) -> ResourceAmounts:
    """Sum resource observations while preserving absent dimensions as unknown."""
    items = tuple(values)
    return ResourceAmounts(_sum_scalar(item.cpus for item in items), _sum_scalar(item.memory_bytes for item in items), _sum_mapping(items, "accelerators"), _sum_mapping(items, "named"))


def _sum_scalar(values: Iterable[float | int | None]) -> float | int | None:
    """Sum a scalar only when every contributing observation is known."""
    items = tuple(values)
    return None if any(value is None for value in items) else sum(items)


def _sum_mapping(values: Iterable[ResourceAmounts], name: str) -> dict[str, float | None]:
    """Sum mappings without converting absent resource dimensions to zero."""
    items = tuple(values)
    keys = {key for item in items for key in getattr(item, name)}
    return {key: _sum_scalar(getattr(item, name).get(key) if key in getattr(item, name) else None for item in items) for key in keys}


def _subtract(total: ResourceAmounts, charges: ResourceAmounts) -> ResourceAmounts:
    """Subtract charges while preserving unknown capacity and explicit unknown charges."""
    def sub(left: float | int | None, right: float | int | None) -> float | int | None:
        return None if left is None or right is None else max(0, left - right)

    def mappings(left: Mapping[str, float | None], right: Mapping[str, float | None]) -> dict[str, float | None]:
        return {key: sub(left.get(key) if key in left else None, right.get(key, 0)) for key in set(left) | set(right)}

    return ResourceAmounts(sub(total.cpus, charges.cpus), sub(total.memory_bytes, charges.memory_bytes), mappings(total.accelerators, charges.accelerators), mappings(total.named, charges.named))


def _minimum(left: ResourceAmounts, right: ResourceAmounts) -> ResourceAmounts:
    """Return conservative availability, preserving an unknown operand as unknown."""
    def minimum(first: float | int | None, second: float | int | None) -> float | int | None:
        return None if first is None or second is None else min(first, second)

    def mappings(first: Mapping[str, float | None], second: Mapping[str, float | None]) -> dict[str, float | None]:
        return {key: minimum(first.get(key) if key in first else None, second.get(key) if key in second else None) for key in set(first) | set(second)}

    return ResourceAmounts(minimum(left.cpus, right.cpus), minimum(left.memory_bytes, right.memory_bytes), mappings(left.accelerators, right.accelerators), mappings(left.named, right.named))


def _fits(available: ResourceAmounts, requested: ResourceAmounts) -> bool:
    """Require known capacity for every known requested resource dimension."""
    def fits(left: float | int | None, right: float | int | None) -> bool:
        return right in (None, 0) or (left is not None and left >= right)

    return fits(available.cpus, requested.cpus) and fits(available.memory_bytes, requested.memory_bytes) and all(fits(available.accelerators.get(key) if key in available.accelerators else None, value) for key, value in requested.accelerators.items()) and all(fits(available.named.get(key) if key in available.named else None, value) for key, value in requested.named.items())


def _amounts_complete(amounts: ResourceAmounts) -> bool:
    """Return whether every represented capacity or charge value is known."""
    return amounts.cpus is not None and amounts.memory_bytes is not None and all(value is not None for value in amounts.accelerators.values()) and all(value is not None for value in amounts.named.values())


def _processes(allocation: WorldAllocation):
    """Yield the exact assigned processes without interpreting requirement policy."""
    return (process for processes in allocation.roles.values() for process in processes)


def _allocation_amounts(allocation: WorldAllocation) -> ResourceAmounts:
    """Summarize exact local bindings for the authority's scalar charge view."""
    processes = tuple(_processes(allocation))
    return ResourceAmounts(
        float(sum(len(process.cpus) for process in processes)),
        None if all(process.memory is None for process in processes) else sum(process.memory or 0 for process in processes),
        {
            kind: float(sum(len(process.accelerators.get(kind, ())) for process in processes))
            for kind in {kind for process in processes for kind in process.accelerators}
        },
        {},
    )


def _same_bindings(expected: WorldAllocation, observed: WorldAllocation) -> bool:
    """Compare exclusive IDs while allowing worker control evidence to differ."""
    return tuple(
        (name, tuple((process.cpus, tuple((kind, devices) for kind, devices in process.accelerators.items())) for process in processes))
        for name, processes in expected.roles.items()
    ) == tuple(
        (name, tuple((process.cpus, tuple((kind, devices) for kind, devices in process.accelerators.items())) for process in processes))
        for name, processes in observed.roles.items()
    )


def _unknown_charge(reservation: Reservation) -> bool:
    """Keep only an unbound charge from becoming fabricated exact-ID capacity.

    An exact local allocation safely identifies its occupied CPUs/accelerators even
    when unrelated dimensions, such as unenforced memory, remain unknown.  Treating
    that unrelated uncertainty as an unknown CPU charge would unnecessarily
    serialize independently affinitized workers.
    """
    return reservation.allocation is None


__all__ = ["RESOURCE_AUTHORITIES", "Reservation", "ResourceAuthority", "ResourceAuthorityRegistry"]
