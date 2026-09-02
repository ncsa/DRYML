"""Tests for the dependency-light public Method package boundary."""

import subprocess
import sys

_EXPECTED_METHOD_EXPORTS = {
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "Method",
    "MethodCallMode",
    "MethodCallNode",
    "MethodCallNodeKind",
    "MethodCallSignature",
    "MethodError",
    "MethodImplementation",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
    "Traits",
    "traits",
}


def test_methods_public_manifest_and_retired_code_imports_are_exact():
    """Methods own the exact public IR surface and retired code imports fail."""

    script = """
import importlib
import json

import dryml.code
import dryml.methods

assert set(dryml.methods.__all__) == {
    'ImplementationDeclarationError', 'ImplementationSelectionError', 'Method',
    'MethodCallMode', 'MethodCallNode', 'MethodCallNodeKind',
    'MethodCallSignature', 'MethodError', 'MethodImplementation',
    'PreparedCallMismatchError', 'SelectionFailureReason', 'SelectionTraitName',
    'Traits', 'traits',
}
assert not {'Method', 'Traits', 'traits'} & set(dryml.code.__all__)
for statement in (
    'from dryml.code import Method',
    'from dryml.code import Traits',
    'from dryml.code import traits',
):
    try:
        exec(statement, {})
    except ImportError:
        pass
    else:
        raise AssertionError(f'retired import succeeded: {statement}')
for module in ('dryml.code.method', 'dryml.code.traits'):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        assert error.name == module
    else:
        raise AssertionError(f'retired module remains importable: {module}')
print(json.dumps(sorted(dryml.methods.__all__)))
"""

    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert set(__import__("json").loads(completed.stdout)) == _EXPECTED_METHOD_EXPORTS


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
