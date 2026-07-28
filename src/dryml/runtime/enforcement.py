"""Runtime enforcement policy for DRYML guard behavior."""

from __future__ import annotations

import os
import warnings
from enum import Enum
from typing import Any


ENV_VAR = "DRYML_RUNTIME_ENFORCEMENT"


class RuntimeEnforcement(str, Enum):
    """Policy controlling how DRYML runtime/environment/world guards react."""

    STRICT = "strict"
    WARN = "warn"
    OFF = "off"


def normalize_enforcement(policy: RuntimeEnforcement | str) -> RuntimeEnforcement:
    """Return *policy* as a ``RuntimeEnforcement`` member.

    Args:
        policy: A ``RuntimeEnforcement`` member or one of ``strict``, ``warn``,
            or ``off``. String values are matched case-insensitively.

    Returns:
        The normalized runtime enforcement policy.

    Raises:
        ValueError: If *policy* is not one of the supported values.
    """

    if isinstance(policy, RuntimeEnforcement):
        return policy
    if not isinstance(policy, str):
        raise ValueError(f"unsupported runtime enforcement policy {policy!r}; expected strict, warn, or off")
    value = policy.strip().lower()
    try:
        return RuntimeEnforcement(value)
    except ValueError as exc:
        raise ValueError(f"unsupported runtime enforcement policy {policy!r}; expected strict, warn, or off") from exc


def startup_enforcement_from_env(environ: Any = None) -> tuple[RuntimeEnforcement, bool]:
    """Return the startup policy and whether it was explicitly supplied.

    Missing and malformed values deliberately select unchecked Python behavior.
    A valid non-off value remains visible as an advanced low-level override until
    a process baseline is later published.
    """

    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_VAR)
    if raw is None:
        return RuntimeEnforcement.OFF, False
    try:
        return normalize_enforcement(raw), True
    except ValueError as exc:
        warnings.warn(f"{ENV_VAR}={raw!r} is invalid; falling back to off: {exc}", RuntimeWarning, stacklevel=2)
        return RuntimeEnforcement.OFF, False


def default_enforcement_from_env(environ: Any = None) -> RuntimeEnforcement:
    """Return the initial enforcement policy from ``DRYML_RUNTIME_ENFORCEMENT``.

    Args:
        environ: Optional mapping-like environment used by tests. Defaults to
            ``os.environ``.

    Returns:
        The configured policy, or ``RuntimeEnforcement.OFF`` when unset or
        invalid. Invalid values emit ``RuntimeWarning`` to keep imports usable.
    """

    return startup_enforcement_from_env(environ)[0]


__all__ = ["ENV_VAR", "RuntimeEnforcement", "default_enforcement_from_env", "normalize_enforcement", "startup_enforcement_from_env"]
