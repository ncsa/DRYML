"""Tests for the dependency-light public Method package boundary."""

import subprocess
import sys


def test_fresh_methods_import_has_no_consumer_runtime_or_optional_side_effects():
    """Importing Method declarations stays below code, consumer, and runtime layers."""

    script = """
import json
import sys
import dryml.methods
forbidden = (
    'dryml.code', 'dryml.data', 'dryml.models', 'dryml.artifacts',
    'dryml.requirements', 'dryml.environments', 'dryml.worlds',
    'dryml.runtime', 'dryml.session', 'dryml.dispatch', 'dryml.execute',
    'dryml.managed', 'dryml.operations', 'dryml.records', 'dryml.store',
    'dryml.core.repo', 'dryml.core.query', 'dryml.core.store',
    'dryml.core.session',
    'tensorflow', 'torch', 'jax', 'jaxlib', 'ray',
)
print(json.dumps(sorted(name for name in sys.modules if name in forbidden)))
"""

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
