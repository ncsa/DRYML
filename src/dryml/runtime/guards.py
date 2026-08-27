"""Context-local materialization action state and publication lease admission."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .context import publication
from .enforcement import RuntimeEnforcement

MaterializationAction = RuntimeEnforcement
_ACTION: ContextVar[RuntimeEnforcement] = ContextVar("dryml_runtime_materialization_action", default=RuntimeEnforcement.STRICT)


def materialization_action() -> RuntimeEnforcement:
    """Return the innermost materialization action without changing runtime state."""
    return _ACTION.get()


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
def materialization_admission() -> Iterator[None]:
    """Hold a generation lease for U8 materialization integration.

    U5 deliberately does not decide whether an operation materializes an
    object; U8 invokes this seam before its core entry points.
    """
    with publication.lease():
        yield


__all__ = ["MaterializationAction", "materialization_action", "materialization_admission", "materialization_scope"]
