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


class BackendChecked(Method):
    """Select a NumPy transition while recording successful target calls."""

    calls = 0

    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation, carry):
        """Return the carry only after selected-call validation succeeds."""

        type(self).calls += 1
        return carry


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


def test_selected_callable_validates_all_positional_specs_and_raw_output_before_callbacks():
    """Carry and result constraints fail at their selected-call boundaries."""

    method = Checked()
    observation = TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy")
    carry = TensorSpec("float32", shape=(3,), backend="numpy")
    selected = method.find_implementation(observation, carry, output_spec=carry)

    with pytest.raises(ImplementationSelectionError) as missing:
        selected(np.ones((2, 3), dtype=np.float32))
    assert missing.value.reason == "conflict"
    assert method.calls == 0

    with pytest.raises(ImplementationSelectionError) as bad_carry:
        selected(
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            option=None,
        )
    assert bad_carry.value.reason == "conflict"
    assert method.calls == 0

    callbacks = []
    with pytest.raises(ImplementationSelectionError) as bad_output:
        selected.invoke_with_raw_result(
            (np.ones((1, 3), dtype=np.float32), np.ones((3,), dtype=np.float32)),
            {"option": None},
            callbacks.append,
        )
    assert bad_output.value.reason == "conflict"
    assert method.calls == 1
    assert callbacks == []


def test_extra_specs_require_a_first_spec_and_malformed_specs_fail_before_selection():
    """Selected contracts reject invalid static input layouts before targets run."""

    method = Checked()
    carry = TensorSpec("float32", shape=(3,), backend="numpy")

    with pytest.raises(ImplementationSelectionError) as missing_first:
        method.find_implementation(None, carry)
    assert missing_first.value.reason == "conflict"

    with pytest.raises(ImplementationSelectionError) as malformed:
        method.find_implementation(carry, object())
    assert malformed.value.reason == "conflict"


def test_additional_known_backends_conflict_without_selecting_or_ranking_from_carry():
    """Carry backends validate against observation/explicit backend without driving selection."""

    method = BackendChecked()
    BackendChecked.calls = 0
    observation = TensorSpec("float32", shape=(2,), batch=Dynamic, backend="numpy")
    unknown_observation = TensorSpec("float32", shape=(2,), batch=Dynamic)
    torch_carry = TensorSpec("float32", shape=(), backend="torch")
    unknown_carry = TensorSpec("float32", shape=())

    with pytest.raises(ImplementationSelectionError) as compatible:
        method.compatible_implementations(observation, torch_carry)
    assert compatible.value.reason == "conflict"

    with pytest.raises(ImplementationSelectionError) as selected_conflict:
        method.find_implementation(observation, torch_carry)
    assert selected_conflict.value.reason == "conflict"

    with pytest.raises(ImplementationSelectionError) as explicit_conflict:
        method.find_implementation(unknown_observation, torch_carry, backend="numpy")
    assert explicit_conflict.value.reason == "conflict"
    assert BackendChecked.calls == 0

    selected = method.find_implementation(observation, unknown_carry)
    carry = np.array(2.0, dtype=np.float32)
    assert selected(np.ones((1, 2), dtype=np.float32), carry) is carry
    assert BackendChecked.calls == 1
