"""Runtime enforcement policy for DRYML guard behavior."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


ENV_VAR = "DRYML_RUNTIME_ENFORCEMENT"
REQUIREMENT_AXIS_NAMES = ("environment", "world", "runtime")


@dataclass(frozen=True, slots=True)
class RequirementAxes:
    """Canonical immutable subset of requirement compatibility axes.

    The value identifies which of the ``environment``, ``world``, and
    ``runtime`` requirement checks participate in compatibility collection.
    It has no effect on runtime role, allocation, visibility, protocol, or
    lifecycle validation. Enabled names are stored in stable canonical order.

    Args:
        enabled: Iterable of enabled canonical axis names.

    Raises:
        ValueError: If an axis is unknown, duplicated, or not in canonical
            order.
    """

    enabled: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the enabled subset in canonical order."""

        enabled = tuple(self.enabled)
        if len(set(enabled)) != len(enabled) or any(axis not in REQUIREMENT_AXIS_NAMES for axis in enabled):
            raise ValueError("requirement axes must contain unique environment, world, and runtime names")
        canonical = tuple(axis for axis in REQUIREMENT_AXIS_NAMES if axis in enabled)
        if enabled != canonical:
            raise ValueError("requirement axes must use canonical environment, world, runtime order")
        object.__setattr__(self, "enabled", canonical)

    @classmethod
    def all(cls) -> "RequirementAxes":
        """Return the immutable mask with every supported axis enabled."""

        return cls(REQUIREMENT_AXIS_NAMES)

    @classmethod
    def from_mapping(cls, value: Mapping[str, bool]) -> "RequirementAxes":
        """Normalize one complete exact-boolean axis mapping.

        Args:
            value: Mapping containing exactly ``environment``, ``world``, and
                ``runtime`` with literal boolean values.

        Returns:
            The canonical immutable enabled subset.

        Raises:
            ValueError: If the mapping is incomplete, has unknown names, or
                contains a non-boolean value.
        """

        if not isinstance(value, Mapping) or set(value) != set(REQUIREMENT_AXIS_NAMES):
            raise ValueError("requirement axes must contain exactly environment, world, and runtime")
        if any(type(value[axis]) is not bool for axis in REQUIREMENT_AXIS_NAMES):
            raise ValueError("requirement axis values must be booleans")
        return cls(tuple(axis for axis in REQUIREMENT_AXIS_NAMES if value[axis]))

    def to_data(self) -> list[str]:
        """Return the enabled names as a fresh canonical JSON-ready list."""

        return list(self.enabled)


def normalize_requirement_axes(value: Mapping[str, bool]) -> RequirementAxes:
    """Normalize one complete requirement-axis mapping.

    Args:
        value: Exact mapping accepted by :meth:`RequirementAxes.from_mapping`.

    Returns:
        A canonical immutable requirement-axis mask.

    Raises:
        ValueError: If the mapping is malformed.
    """

    return RequirementAxes.from_mapping(value)


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


__all__ = ["ENV_VAR", "REQUIREMENT_AXIS_NAMES", "RequirementAxes", "RuntimeEnforcement", "default_enforcement_from_env", "normalize_enforcement", "normalize_requirement_axes", "startup_enforcement_from_env"]
