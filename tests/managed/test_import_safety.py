"""Import-safety tests for the managed declaration boundary."""

from __future__ import annotations

import json
import subprocess
import sys


def test_managed_exports_are_closed_and_import_avoids_lifecycle_consumers():
    """U3 imports declaration helpers but not execute, dispatch, records, or plugins."""

    script = """
import json
import sys
import dryml.managed

assert set(dryml.managed.__all__) == {
    'InterruptRequestResult', 'ManagedConfig', 'ManagedConfigError',
    'ManagedConflictError', 'ManagedContextError', 'ManagedControlError',
    'ManagedDeclarationError', 'ManagedError', 'ManagedInterrupted',
    'ManagedRecoveryError', 'ManagedRerunRequiredError', 'ManagedStatus',
    'ManagedStoreError', 'argument_digest', 'managed_operation', 'operation_digest',
}
forbidden = ('dryml.dispatch', 'dryml.execute', 'dryml.records', 'dryml.operations',
             'tensorflow', 'torch', 'jax', 'jaxlib', 'ray')
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
