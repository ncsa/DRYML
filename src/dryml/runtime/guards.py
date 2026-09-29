"""PID-safe admission for live Object materialization operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
import warnings
from typing import Iterator

from .context import active_runtime, publication
from .enforcement import RuntimeEnforcement
from .errors import RuntimeTransitionError
from .modes import RuntimeMode

MaterializationAction = RuntimeEnforcement
_ACTION: ContextVar[RuntimeEnforcement] = ContextVar("dryml_runtime_materialization_action", default=RuntimeEnforcement.STRICT)


@dataclass(slots=True)
class _Admission:
    """Task-bound state for one leased top-level materialization operation."""

    control_epoch: int
    owner: tuple[int, int | None]
    active: bool = True
    warned: bool = False


_ADMISSION: ContextVar[_Admission | None] = ContextVar("dryml_runtime_materialization_admission", default=None)


def _owner_identity() -> tuple[int, int | None]:
    """Return the current thread/task identity without creating an async task."""

    try:
        import asyncio

        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), None if task is None else id(task)


def internal_construction_admitted() -> bool:
    """Return whether the current task holds a valid private construction admission.

    This intentionally does not expose a public object-mode escape hatch.  A
    copied context in a sibling task or thread cannot reuse an admission.
    """

    generation = publication.current()
    admission = _ADMISSION.get()
    return (
        admission is not None
        and admission.active
        and admission.control_epoch == int(generation.metadata.get("control_epoch", generation.number))
        and admission.owner == _owner_identity()
    )


def materialization_action() -> RuntimeEnforcement:
    """Return the innermost materialization action without changing runtime state."""
    return _ACTION.get()


def _enforce_orchestrator_action(admission: _Admission, operation: str) -> None:
    """Apply the innermost action without acquiring another lease."""

    if active_runtime().mode is not RuntimeMode.ORCHESTRATOR:
        return
    action = materialization_action()
    if action is RuntimeEnforcement.STRICT:
        raise RuntimeTransitionError(
            "Orchestration mode prohibits Object materialization",
            context={
                "operation": operation,
                "mode": "orchestrator",
                "fix": "use Definition/CDef APIs, a fresh managed process, or a future explicit dispatch",
            },
        )
    if action is RuntimeEnforcement.WARN and not admission.warned:
        warnings.warn(
            f"Orchestration mode permits Object materialization only through explicit warn scope ({operation})",
            RuntimeWarning,
            stacklevel=3,
        )
        admission.warned = True


@contextmanager
def materialization_scope(action: RuntimeEnforcement | str = "strict") -> Iterator[RuntimeEnforcement]:
    """Temporarily select a nested materialization guard action.

    Args:
        action: Exactly ``strict``, ``warn``, or ``off``.

    Yields:
        The normalized enforcement action while the override is active.

    Raises:
        ValueError: If ``action`` is outside the closed vocabulary.

    Side Effects:
        Changes only task-local guard action and restores it after all exits.
    """
    resolved = RuntimeEnforcement.coerce(action)
    token = _ACTION.set(resolved)
    try:
        yield resolved
    finally:
        _ACTION.reset(token)


@contextmanager
def materialization_admission(*, operation: str = "materialization") -> Iterator[None]:
    """Admit one live Object operation under the current publication generation.

    Args:
        operation: Stable diagnostic name for the attempted live-object action.

    Yields:
        ``None`` while the operation owns, or reuses, its publication lease.

    Raises:
        RuntimeTransitionError: Before any materialization side effect when an
            orchestrator uses the default strict action.

    Side Effects:
        Pins the current runtime generation for a top-level operation.  In
        orchestrator mode, an explicit ``warn`` scope warns once and permits the
        operation; ``off`` permits it silently.
    """

    # This checks the PID before consulting a ContextVar that could have been
    # inherited across fork, and before any inherited synchronization is used.
    current = publication.current()
    existing = _ADMISSION.get()
    if (
        existing is not None
        and existing.active
        and existing.control_epoch == int(current.metadata.get("control_epoch", current.number))
        and existing.owner == _owner_identity()
    ):
        _enforce_orchestrator_action(existing, operation)
        yield
        return

    with publication.lease() as generation:
        admission = _Admission(
            int(generation.metadata.get("control_epoch", generation.number)),
            _owner_identity(),
        )
        token = _ADMISSION.set(admission)
        try:
            _enforce_orchestrator_action(admission, operation)
            yield
        finally:
            admission.active = False
            _ADMISSION.reset(token)


__all__ = ["MaterializationAction", "internal_construction_admitted", "materialization_action", "materialization_admission", "materialization_scope"]
