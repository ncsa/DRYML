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


def default_enforcement_from_env(environ: Any = None) -> RuntimeEnforcement:
    """Return the initial enforcement policy from ``DRYML_RUNTIME_ENFORCEMENT``.

    Args:
        environ: Optional mapping-like environment used by tests. Defaults to
            ``os.environ``.

    Returns:
        The configured policy, or ``RuntimeEnforcement.STRICT`` when unset or
        invalid. Invalid values emit ``RuntimeWarning`` to keep imports usable.
    """

    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_VAR)
    if raw is None:
        return RuntimeEnforcement.STRICT
    try:
        return normalize_enforcement(raw)
    except ValueError as exc:
        warnings.warn(f"{ENV_VAR}={raw!r} is invalid; falling back to strict: {exc}", RuntimeWarning, stacklevel=2)
        return RuntimeEnforcement.STRICT


__all__ = ["ENV_VAR", "RuntimeEnforcement", "default_enforcement_from_env", "normalize_enforcement"]
