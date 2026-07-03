"""Active process-local runtime state."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator

from .allocation import NoAllocation, RuntimeAllocationView, is_no_allocation
from .errors import RuntimeTransitionError
from .modes import RuntimeMode
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Current process-local runtime mode, spec, and allocation view."""

    mode: RuntimeMode = RuntimeMode.ORCHESTRATOR
    allocation: RuntimeAllocationView | Any = NoAllocation
    spec: RuntimeContextSpec | None = None


_DEFAULT_RUNTIME = RuntimeState()
_ACTIVE_RUNTIME: ContextVar[RuntimeState] = ContextVar("dryml_active_runtime", default=_DEFAULT_RUNTIME)


def active_runtime() -> RuntimeState:
    """Return the current active runtime state."""

    return _ACTIVE_RUNTIME.get()


def active_runtime_mode() -> RuntimeMode:
    """Return the current active runtime mode."""

    return active_runtime().mode


@contextmanager
def enter_runtime(mode: RuntimeMode | str, allocation: RuntimeAllocationView | Any = NoAllocation, spec: RuntimeContextSpec | None = None) -> Iterator[RuntimeState]:
    """Temporarily enter a runtime mode with reset/token semantics."""

    state = _make_state(mode, allocation, spec)
    token = _ACTIVE_RUNTIME.set(state)
    try:
        yield state
    finally:
        _ACTIVE_RUNTIME.reset(token)


def set_runtime(state: RuntimeState) -> Token[RuntimeState]:
    """Set the active runtime and return a reset token."""

    _validate_state(state)
    return _ACTIVE_RUNTIME.set(state)


def reset_runtime(token: Token[RuntimeState] | None = None) -> None:
    """Reset active runtime to a token or the safe default."""

    if token is None:
        _ACTIVE_RUNTIME.set(_DEFAULT_RUNTIME)
    else:
        _ACTIVE_RUNTIME.reset(token)


def _make_state(mode: RuntimeMode | str, allocation: RuntimeAllocationView | Any, spec: RuntimeContextSpec | None) -> RuntimeState:
    state = RuntimeState(mode=RuntimeMode.coerce(mode), allocation=allocation, spec=spec)
    _validate_state(state)
    return state


def _validate_state(state: RuntimeState) -> None:
    if state.mode in {RuntimeMode.WORKER, RuntimeMode.INLINE} and is_no_allocation(state.allocation):
        raise RuntimeTransitionError(
            "worker/inline runtime requires an explicit allocation",
            context={"mode": state.mode.value, "allocation": repr(state.allocation), "fix": "enter worker/inline runtime with RuntimeAllocationView"},
        )
    if state.mode in {RuntimeMode.ORCHESTRATOR, RuntimeMode.PROBE} and not is_no_allocation(state.allocation):
        raise RuntimeTransitionError(
            "orchestrator/probe runtime must not hold workload allocation",
            context={"mode": state.mode.value, "allocation": repr(state.allocation), "fix": "use RuntimeMode.WORKER or RuntimeMode.INLINE for workload resources"},
        )


__all__ = ["RuntimeState", "active_runtime", "active_runtime_mode", "enter_runtime", "reset_runtime", "set_runtime"]
