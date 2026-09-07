"""Public managed and core-state API integration contracts."""

from __future__ import annotations

import json
import subprocess
import sys


_MANAGED_EXPORTS = {
    "InterruptRequestResult",
    "ManagedConfig",
    "ManagedContext",
    "ManagedConfigError",
    "ManagedConflictError",
    "ManagedContextError",
    "ManagedControlError",
    "ManagedDeclarationError",
    "ManagedError",
    "ManagedInterrupted",
    "ManagedPublicationError",
    "ManagedRecoveryError",
    "ManagedRerunRequiredError",
    "ManagedStatus",
    "ManagedStoreError",
    "argument_digest",
    "managed_operation",
    "operation_digest",
}


def test_root_managed_is_lazy_and_core_owns_state_graph_reservation():
    """Expose managed lazily while keeping the reservation in its core owner."""

    completed = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
import dryml
before = 'dryml.managed' in sys.modules
managed = dryml.managed
import dryml.core
from dryml.core import StateGraphReservation
print(json.dumps({
    'before': before,
    'root_exports': 'managed' in dryml.__all__,
    'managed_exports': sorted(managed.__all__),
    'core_reservation': 'StateGraphReservation' in dryml.core.__all__,
    'reservation_owner': StateGraphReservation.__module__,
    'root_reservation': hasattr(dryml, 'StateGraphReservation'),
}))
"""],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    assert data == {
        "before": False,
        "root_exports": True,
        "managed_exports": sorted(_MANAGED_EXPORTS),
        "core_reservation": True,
        "reservation_owner": "dryml.core.state",
        "root_reservation": False,
    }


def test_managed_import_does_not_load_execution_or_optional_consumers():
    """Keep the standalone lifecycle API free of execution and plugin imports."""

    completed = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
import dryml.managed
forbidden = ('dryml.dispatch', 'dryml.execute', 'dryml.records', 'tensorflow',
             'torch', 'jax', 'jaxlib', 'ray')
print(json.dumps(sorted(set(forbidden) & sys.modules.keys())))
"""],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
