"""Closed runtime role declarations for this non-dispatch port."""

from __future__ import annotations

from enum import Enum


class RuntimeMode(str, Enum):
    """Process-global role with no worker, probe, or launcher variants."""

    NONE = "none"
    ORCHESTRATOR = "orchestrator"
    INLINE = "inline"

    @classmethod
    def coerce(cls, value: "RuntimeMode | str") -> "RuntimeMode":
        """Normalize a declared mode.

        Args:
            value: A member or its exact lower-case wire value.

        Returns:
            The matching closed mode.

        Raises:
            ValueError: If the value is not a supported runtime mode.
        """
        return value if isinstance(value, cls) else cls(value)


__all__ = ["RuntimeMode"]
