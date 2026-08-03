"""Flat, persistent process-session configuration facade."""

from .configuration import normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, SessionSnapshot
from .state import allocate_world, configure, current, enforce_requirements, manage, mode, require_env, reset, set_mode, worker_env_request, worker_world_request

__all__ = [
    "SelectedWorldAllocation",
    "SessionConfiguration",
    "SessionConfigurationError",
    "SessionSnapshot",
    "allocate_world",
    "configure",
    "current",
    "enforce_requirements",
    "manage",
    "mode",
    "normalize_configuration",
    "require_env",
    "reset",
    "select_world_allocation",
    "set_mode",
    "worker_world_request",
    "worker_env_request",
]
