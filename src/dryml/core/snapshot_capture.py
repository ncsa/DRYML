"""In-memory lifecycle and environment evidence capture for future snapshots.

This module builds detached :class:`SnapshotCapture` values from already
materialized save plans. It deliberately owns neither Store access nor snapshot
publication, so a later persistence boundary can reuse one capture unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Mapping, TYPE_CHECKING

from dryml.environments import EnvironmentRecord
from dryml.environments.combination import _requirements_for_classes

from .metadata import LineageMetadata, SnapshotCapture
from .reference_values import ObjectId
from .utils.graph.path import GraphPath

if TYPE_CHECKING:
    from .repo_plan import SavePlan


_ENVIRONMENT_UNAVAILABLE = (
    "dryml.environments.observation_unavailable",
    "current environment observation unavailable",
)


def current_utc_time() -> datetime:
    """Return the current aware UTC timestamp for allocation and capture facts."""

    return datetime.now(timezone.utc)


def install_lineage_fact(obj, object_id: ObjectId, created_at: datetime | None) -> None:
    """Attach one framework-owned creation fact to a live Object identity.

    Args:
        obj: Live Object whose exact ``object_id`` is being attached.
        object_id: Current exact lineage identity for ``obj``.
        created_at: Aware UTC allocation instant, or ``None`` for known-unknown
            historical evidence.

    Side Effects:
        Replaces only framework runtime evidence. User attributes, serialized
        state, and reference identity are not inspected or changed.
    """

    obj._lineage_fact_object_id = object_id
    obj._lineage_created_at = created_at


def capture_lineages(plan: "SavePlan") -> Mapping[GraphPath, LineageMetadata]:
    """Project live creation evidence into the primary paths of one save plan.

    Args:
        plan: Complete retained save evidence for one live root.

    Returns:
        Detached lineage metadata for the root and every canonical ObjectId path.
        A root without an ObjectId and identities without a live creation fact are
        represented as unknown.

    Raises:
        TypeError: If ``plan`` is not a save-plan-like object with an ObjectRef.
        ValueError: If retained bindings cannot resolve a primary ObjectId path.

    Side Effects:
        None. This reads framework runtime evidence only.
    """

    reference = getattr(plan, "object_ref", None)
    root = getattr(getattr(plan, "binding", None), "roots", (None,))[0]
    root_object = getattr(root, "obj", None)
    if root_object is None or reference is None:
        raise TypeError("snapshot lineage capture requires a complete live SavePlan")
    lineages = {}
    for path in (GraphPath(), *reference.objects):
        target = reference if not path else reference.at(path)
        created_at = None
        if target.object_id is not None:
            try:
                bound = root_object.graph_at(path)
            except Exception as error:
                raise ValueError(f"snapshot lineage path {path!s} has no live binding") from error
            if getattr(bound, "_lineage_fact_object_id", None) == target.object_id:
                created_at = getattr(bound, "_lineage_created_at", None)
        lineages[path] = LineageMetadata(
            target,
            "known" if created_at is not None else "unknown",
            created_at,
        )
    return lineages


def _materializing_classes(plan: "SavePlan") -> tuple[object, ...]:
    """Return unique live classes projected by materializing save-plan snapshots."""

    classes = []
    seen = set()
    for snapshot in plan.snapshots:
        obj = snapshot.obj
        if obj is None:
            # A seeded exact reference has no live class association in this
            # process, so its declaration coverage is honestly incomplete.
            classes.append(None)
            continue
        cls = type(obj)
        if id(cls) not in seen:
            seen.add(id(cls))
            classes.append(cls)
    return tuple(classes)


def capture_snapshot(
    plan: "SavePlan",
    *,
    observer: Callable[[], EnvironmentRecord] | None = None,
    clock: Callable[[], datetime] = current_utc_time,
) -> SnapshotCapture:
    """Capture detached lifecycle, environment, and requirement evidence once.

    Args:
        plan: Complete live save plan whose materializing snapshots define the
            declaration scope.
        observer: Optional current-process environment observer. Omitted uses the
            existing lazy environment inspector.
        clock: Optional aware-UTC timestamp source for deterministic callers.

    Returns:
        A valid :class:`SnapshotCapture` with independent environment and
        requirement outcome/coverage fields.

    Raises:
        KeyboardInterrupt: Propagated without converting cancellation into
        unavailable evidence.
        SystemExit: Propagated without converting termination into unavailable
        evidence.
        TypeError: If the plan, observer result, or clock result is invalid.

    Side Effects:
        Performs at most one current-environment observation. It never probes,
        opens Stores, reads payloads, or publishes evidence.
    """

    return _capture_snapshot_evidence(
        capture_lineages(plan), _materializing_classes(plan),
        observer=observer, clock=clock,
    )


def _capture_snapshot_evidence(
        lineages, classes, *, observer=None, clock=current_utc_time) -> SnapshotCapture:
    """Capture fresh process evidence for already-selected lineage and classes."""

    if observer is None:
        from dryml.environments.introspection import inspect_current

        observer = inspect_current
    try:
        environment = observer()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        environment = None
        environment_status = "unavailable"
        environment_diagnostics = (_ENVIRONMENT_UNAVAILABLE,)
    else:
        if not isinstance(environment, EnvironmentRecord):
            raise TypeError("current environment observer must return an EnvironmentRecord")
        environment_status = "known"
        environment_diagnostics = ()

    requirements = _requirements_for_classes(tuple(classes))
    return SnapshotCapture(
        lineages,
        clock(),
        environment,
        environment_status,
        requirements.value,
        requirements.status,
        requirements.coverage,
        (*environment_diagnostics, *requirements.diagnostics),
    )


__all__: list[str] = []
