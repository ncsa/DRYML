"""Runtime mode names and semantics."""

from __future__ import annotations

from enum import Enum


class RuntimeMode(str, Enum):
    """Process-local runtime mode."""

    ORCHESTRATOR = "orchestrator"
    PROBE = "probe"
    WORKER = "worker"
    INLINE = "inline"

    @classmethod
    def coerce(cls, value: "RuntimeMode | str") -> "RuntimeMode":
        """Return *value* as a ``RuntimeMode``."""

        if isinstance(value, RuntimeMode):
            return value
        return cls(str(value))


__all__ = ["RuntimeMode"]
