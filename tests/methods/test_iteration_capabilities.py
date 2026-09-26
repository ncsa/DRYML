import numpy as np

from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArgMax, Cast, Flatten, Pipe, Project, Scale, Select
from dryml.methods import Method


class ExplicitIndependent(Method):
    """Test Method that explicitly declares omitted calls history-independent."""

    def __init__(self):
        self.calls = 0

    @property
    def iteration_independent(self):
        return True

    def __call__(self, value):
        self.calls += 1
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


class ChangedIndependent(ExplicitIndependent):
    """Subclass that changes behavior without re-declaring the capability."""

    def __call__(self, value):
        return value + 1


def test_iteration_independence_is_explicit_and_capability_reads_are_inert():
    method = ExplicitIndependent()

    assert method.iteration_independent is True
    assert method.calls == 0
    assert ChangedIndependent().iteration_independent is False
    assert Method().iteration_independent is False


def test_builtin_and_composed_iteration_independence_is_conservative():
    leaves = (Select(0), Cast("float32"), Scale(), Flatten(), ArgMax())

    assert all(method.iteration_independent for method in leaves)
    assert Pipe(Select(0), Cast("float32")).iteration_independent is True
    assert Project(left=Select(0), right=Cast("float32")).iteration_independent is True
    assert Pipe(Select(0), ExplicitIndependent()).iteration_independent is True
    assert Project(left=Select(0), right=ChangedIndependent()).iteration_independent is False

    spec = TensorSpec("int64", shape=(1,), backend="numpy")
    assert ExplicitIndependent().infer_output_spec(spec) == spec
    np.testing.assert_array_equal(ExplicitIndependent()(np.array([1])), np.array([1]))
