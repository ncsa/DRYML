"""Flat, persistent process-session configuration facade."""

from .configuration import normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, SessionSnapshot
from .state import allocate_world, configure, current, manage, mode, request_world, require_env, reset, set_mode

__all__ = [
    "SelectedWorldAllocation",
    "SessionConfiguration",
    "SessionConfigurationError",
    "SessionSnapshot",
    "allocate_world",
    "configure",
    "current",
    "manage",
    "mode",
    "normalize_configuration",
    "request_world",
    "require_env",
    "reset",
    "select_world_allocation",
    "set_mode",
]
