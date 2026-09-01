"""Contract tests for immutable call signatures and selected-call validation."""

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.methods import ImplementationSelectionError, Method, traits


class Checked(Method):
    """Method fixture whose target must run only after input validation."""

    calls = 0

    @traits(backend="numpy", batch_mode="batched")
    def numpy_batched(self, value, extra, *, option):
        """Return all logical arguments after successful pre-invocation validation."""

        self.calls += 1
        return value, extra, option


def test_selected_callable_directionally_validates_nested_dynamic_first_input_only():
    """Known spec facts reject conflicts while Dynamic dimensions accept concrete values."""

    method = Checked()
    specification = {
        "left": TensorSpec("float32", shape=(Dynamic, 3), batch=Dynamic, backend="numpy"),
        "right": (TensorSpec("int64", shape=(), batch=Dynamic, backend="numpy"),),
    }
    selected = method.find_implementation(input_spec=specification)
    value = {
        "left": np.ones((2, 4, 3), dtype=np.float32),
        "right": (np.ones((2,), dtype=np.int64),),
    }

    assert selected(value, ["unvalidated", "later"], option={"also": "forwarded"})[0] is value
    assert method.calls == 1

    bad = dict(value)
    bad["left"] = np.ones((2, 4, 2), dtype=np.float32)
    with pytest.raises(ImplementationSelectionError) as error:
        selected(bad, None, option=None)
    assert error.value.reason == "conflict"
    assert method.calls == 1

    with pytest.raises(ImplementationSelectionError):
        selected()
    assert method.calls == 1


def test_input_spec_constraints_conflicts_and_selected_calls_do_not_touch_preparation_state():
    """Selection APIs reject incompatible known traits without reading Method cache state."""

    method = Checked()
    method.default_batched = False
    selected = method.find_implementation(
        input_spec=TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy")
    )
    method.learn()

    with pytest.raises(ImplementationSelectionError) as error:
        method.find_implementation(
            input_spec=TensorSpec("float32", shape=(3,), backend="numpy"),
            batch_mode="batched",
        )
    assert error.value.reason == "conflict"
    assert method.call_mode == "learning"
    assert method.default_batched is False

    selected(np.ones((2, 3), dtype=np.float32), None, option=None)
    assert method.call_mode == "learning"
    assert method.cached_signature is None
