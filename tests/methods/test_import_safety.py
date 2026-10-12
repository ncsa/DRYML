"""Tests for the dependency-light public Method package boundary."""

import subprocess
import sys
import importlib.util
import pytest

_EXPECTED_METHOD_EXPORTS = {
    "Accumulator",
    "AccumulatorGroup",
    "ImplementationDeclarationError",
    "ImplementationSelectionError",
    "Method",
    "MethodGraph",
    "MethodGraphNode",
    "MethodGraphNodeKind",
    "MethodCallMode",
    "MethodCallNode",
    "MethodCallNodeKind",
    "MethodCallSignature",
    "ConversionEdge",
    "MethodError",
    "MethodImplementation",
    "IteratorPort",
    "MethodPort",
    "MethodPortKind",
    "PreparedCallMismatchError",
    "SelectionFailureReason",
    "SelectionTraitName",
    "StreamGraphCursor",
    "StreamNode",
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
    'Accumulator', 'AccumulatorGroup', 'ImplementationDeclarationError',
    'ImplementationSelectionError', 'Method', 'MethodGraph', 'MethodGraphNode', 'ConversionEdge',
    'MethodGraphNodeKind',
    'MethodCallMode', 'MethodCallNode', 'MethodCallNodeKind',
    'MethodCallSignature', 'MethodError', 'MethodImplementation', 'IteratorPort', 'MethodPort',
    'MethodPortKind',
    'PreparedCallMismatchError', 'SelectionFailureReason', 'SelectionTraitName',
    'StreamGraphCursor', 'StreamNode',
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


def test_stream_shortfall_stays_below_data_and_data_reexports_same_exception():
    """Runtime skip/Take errors need no reverse imports and retain Data identity."""
    script = """
import sys
import dryml.methods
from dryml.methods.stream import DatasetExhaustedError, StreamGraphCursor, _StreamOp, _TakeIterator
class Source:
    def iterator(self):
        return iter((1,))
cursor = StreamGraphCursor(_StreamOp('source', dataset=Source()))
try:
    cursor.skip(2)
except DatasetExhaustedError as error:
    assert (error.requested, error.yielded) == (2, 1)
else:
    raise AssertionError('skip did not report shortfall')
cursor = StreamGraphCursor(_StreamOp('source', dataset=Source()))
try:
    next(_TakeIterator(cursor, iter(()), 2))
except DatasetExhaustedError as error:
    assert (error.requested, error.yielded) == (2, 0)
else:
    raise AssertionError('Take did not report shortfall')
assert not any(name in sys.modules for name in ('dryml.data', 'dryml.artifacts', 'dryml.models', 'dryml.managed'))
assert not hasattr(dryml.methods, 'DatasetExhaustedError')
from dryml.data import DatasetExhaustedError as public
from dryml.data.dataset import DatasetExhaustedError as established
assert public is established is DatasetExhaustedError
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("backend,dependency", (("torch", "torch"), ("tf", "tensorflow"), ("jax", "jax")))
def test_fresh_runtime_crossing_activates_adapters_after_passive_planning(backend, dependency, tmp_path):
    """Installed framework crossings plan inertly then activate both endpoints."""
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    script = f'''
import sys
import numpy as np
from dryml.core import TensorSpec
from dryml.methods import Method, traits
from dryml.data import Dataset, Map
from dryml.methods.conversion import convert, make_edge
class Target(Method):
    @traits(backend={backend!r})
    def native(self, value):
        return value
    def infer_output_spec(self, spec):
        return spec
class Source(Dataset):
    def __iter__(self):
        raise AssertionError('planning opened source')
spec = TensorSpec('float32', shape=(2,), backend='numpy')
method = Target()
selected = method.find_implementation(spec)
mapped = Map(Source(spec=spec), method)
assert mapped.spec.backend.value == {backend!r}
method.learn(spec)
assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax'))
result = method(np.array([1, 2], dtype=np.float32))
assert {dependency!r} in sys.modules
assert 'dryml.' + {backend!r} in sys.modules
restored = convert(make_edge(TensorSpec('float32', shape=(2,), backend={backend!r}), 'numpy'), result)
np.testing.assert_array_equal(restored, [1, 2])
'''
    path = tmp_path / "crossing.py"
    path.write_text(script)
    completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_fresh_native_learning_activates_declared_adapter_only_at_runtime(tmp_path):
    """Removing eager registration must preserve native first-call learning."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    script = '''
import sys
from dryml.core import TensorSpec
import torch
from dryml.models.torch import Model
import dryml.methods.conversion as conversion
model = Model(torch.nn.Identity, output_spec=TensorSpec('float32', shape=(2,)))
model.find_implementation(TensorSpec('float32', shape=(2,), backend='torch'))
model.learn()
assert 'dryml.torch' not in sys.modules
def fail(*args):
    raise AssertionError('same-native learning crossed frameworks')
conversion.convert = fail
result = model(torch.ones(2))
assert result.shape == (2,)
assert model.call_mode == 'cached'
assert 'dryml.torch' in sys.modules
assert model.method_graph().conversion_edges == ()
'''
    path = tmp_path / "native-learning.py"
    path.write_text(script)
    completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
