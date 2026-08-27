"""Persistent public process-session configuration facade.

This facade is separate from :mod:`dryml.core.session`. It owns no session
pointer: every successful mutation is a generation on runtime publication.
"""

from .configuration import normalize_configuration, select_world_allocation
from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, SessionSnapshot
from .state import allocate_world, configure, current, enforce_requirements, manage, mode, require_env, reset, set_mode

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
]
