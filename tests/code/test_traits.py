import numpy as np

import dryml.numpy
from dryml.code import Method, traits
from dryml.core2.tensor_spec import TensorSpec


class Double(Method):
    @traits(backend="numpy")
    def numpy_call(self, x):
        return x * 2


class BatchAware(Method):
    @traits(backend="numpy", batch_mode="element")
    def numpy_element(self, x):
        return "element"

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, x):
        return "batched"


def test_traits_register_and_dispatch_numpy_impl():
    x = np.array([1, 2, 3])

    assert Double()(x).tolist() == [2, 4, 6]


def test_traits_resolve_with_batch_mode_from_spec():
    method = BatchAware()
    x = np.zeros((4, 2), dtype=np.float32)
    spec = TensorSpec("float32", shape=(2,), batch=4, backend="numpy")

    impl = method.resolve_impl_for(x, input_spec=spec)

    assert impl(x) == "batched"


def test_traits_resolve_element_mode_from_spec():
    method = BatchAware()
    x = np.zeros((2,), dtype=np.float32)
    spec = TensorSpec("float32", shape=(2,), backend="numpy")

    impl = method.resolve_impl_for(x, input_spec=spec)

    assert impl(x) == "element"
