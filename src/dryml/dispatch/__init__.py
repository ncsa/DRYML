"""Dispatch configuration, bounded requirement probing, and execution seams.

Importing this package creates no backend, Store, worker, resource reservation,
or workload execution. It exposes immutable settings, bounded requirement
probing, and explicit backend or in-process execution choices.
"""

from .api import (
    backends,
    explain,
    register_backend,
    run,
    set_execute_backend_default,
    set_probe_default,
    set_worker_environment_default,
    set_worker_python_default,
    set_worker_world_default,
    submit,
    unregister_backend,
    with_options,
)
from .errors import DispatchError
from .models import (
    BackendChoice,
    DispatchCoverageWarning,
    DispatchReport,
    DispatchView,
    InProcess,
    ProbeOptions,
)

__all__ = [
    "BackendChoice",
    "DispatchCoverageWarning",
    "DispatchError",
    "DispatchReport",
    "DispatchView",
    "InProcess",
    "ProbeOptions",
    "backends",
    "explain",
    "register_backend",
    "run",
    "set_execute_backend_default",
    "set_probe_default",
    "set_worker_environment_default",
    "set_worker_python_default",
    "set_worker_world_default",
    "submit",
    "unregister_backend",
    "with_options",
]
