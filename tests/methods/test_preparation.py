"""Contract tests for Method learning and exact cached calls."""

import numpy as np
import pytest
from dataclasses import replace

from dryml.core.backend import Backend, backend_testers
from dryml.core.tensor_spec import Dynamic, Layout, TensorSpec
from dryml.methods import ImplementationSelectionError, Method, MethodError, PreparedCallMismatchError, traits


class Prepared(Method):
    """Method fixture that records which selected target ran."""

    calls: list[str] = []

    @traits(backend="numpy")
    def numpy(self, value):
        """Record a NumPy invocation."""

        self.calls.append("numpy")
        return value


class SimplePrepared(Method):
    """Simple direct-call fixture for preparation parity."""

    def __call__(self, value):
        """Return a supported value through the captured direct target."""

        return value


class ExactPrepared(Method):
    """Generic fixture used to isolate exact signature comparison."""

    @traits()
    def generic(self, *args, **kwargs):
        """Return the unchanged logical call layout."""

        return args, kwargs


def test_learning_publishes_an_immutable_signature_and_cached_call_bypasses_catalog():
    """The learned target remains available after publication and exact calls reuse it."""

    method = Prepared()
    method.learn()
    assert method.call_mode == "learning"

    first = np.ones((2, 3), dtype=np.float32)
    assert method(first) is first
    signature = method.cached_signature
    assert method.call_mode == "cached"
    assert signature is not None

    method.implementations = lambda: (_ for _ in ()).throw(AssertionError("must not inspect"))
    second = np.ones((2, 3), dtype=np.float32)
    assert method(second) is second

    with pytest.raises(PreparedCallMismatchError) as mismatch:
        method(np.ones((3, 3), dtype=np.float32))
    assert mismatch.value.expected == signature
    assert method.cached_signature == signature


def test_simple_method_learning_reuses_its_direct_target_without_catalog_discovery():
    """Simple Methods support the same explicit learning and exact reuse contract."""

    method = SimplePrepared()
    method.learn()
    first = np.ones((2, 3), dtype=np.float32)

    assert method(first) is first
    assert method.call_mode == "cached"
    signature = method.cached_signature
    assert signature is not None

    method.implementations = lambda: (_ for _ in ()).throw(AssertionError("must not inspect"))
    second = np.ones((2, 3), dtype=np.float32)
    assert method(second) is second

    with pytest.raises(PreparedCallMismatchError):
        method(np.ones((3, 3), dtype=np.float32))
    assert method.cached_signature == signature


def test_default_batched_is_exact_eager_only_and_survives_eager_reset():
    """Batch defaults are local preferences, not cached or logical-call controls."""

    method = Prepared()
    method.default_batched = True
    assert method.default_batched is True
    method.learn()
    with pytest.raises(RuntimeError):
        method.default_batched = False
    method.eager()
    assert method.call_mode == "eager"
    assert method.default_batched is True
    with pytest.raises(TypeError):
        method.default_batched = 1
    assert method.default_batched is True


def test_learning_failures_leave_no_partial_cache_but_target_failure_keeps_publication():
    """Only successful normalization/selection publishes learning state before the target."""

    class Failing(Method):
        @traits(backend="numpy")
        def numpy(self, value):
            raise ValueError("target failure")

    method = object.__new__(Failing)
    method.learn()
    with pytest.raises(MethodError):
        method(object())
    assert method.call_mode == "learning"
    assert method.cached_signature is None

    with pytest.raises(ValueError, match="target failure"):
        method(np.ones((2,), dtype=np.float32))
    assert method.call_mode == "cached"
    assert method.cached_signature is not None


def test_valid_selection_failure_leaves_learning_without_a_partial_cache():
    """A supported call may retry learning after candidate selection fails."""

    class TorchOnly(Method):
        @traits(backend="torch")
        def torch(self, value):
            raise AssertionError("incompatible target must not run")

    method = object.__new__(TorchOnly)
    method.learn()

    with pytest.raises(ImplementationSelectionError) as error:
        method(np.ones((2,), dtype=np.float32))
    assert error.value.reason == "no_candidate"
    assert method.call_mode == "learning"
    assert method.cached_signature is None


def test_cached_signature_detects_every_exact_tensor_field_and_preserves_cache():
    """Backend and normalized TensorSpec metadata all participate in exact reuse."""

    baseline = TensorSpec(
        "float32",
        shape=(Dynamic, 3),
        backend="numpy",
        layout=Layout.RAGGED,
        axis_names=("tokens", "features"),
        ragged_rank=1,
        row_splits_dtype="int64",
    )
    changes = (
        replace(baseline, dtype="float64"),
        replace(baseline, shape=(Dynamic, 4)),
        replace(baseline, backend="torch"),
        replace(baseline, layout=Layout.DENSE),
        replace(baseline, axis_names=("items", "features")),
        replace(baseline, ragged_rank=2),
        replace(baseline, row_splits_dtype="int32"),
    )
    method = ExactPrepared()
    method.learn()
    method(baseline)
    signature = method.cached_signature

    for changed in changes:
        with pytest.raises(PreparedCallMismatchError) as error:
            method(changed)
        assert error.value.expected == signature
        assert method.cached_signature == signature

    assert method(baseline)[0] == (baseline,)


def test_cached_signature_handles_unknown_backend_and_bounds_conflicting_facts():
    method = ExactPrepared()
    unknown = TensorSpec("float32", shape=(3,))
    method.learn()
    method(unknown, unknown)
    signature = method.cached_signature

    with pytest.raises(PreparedCallMismatchError):
        method(TensorSpec("float32", shape=(3,), backend="numpy"), unknown)
    with pytest.raises(PreparedCallMismatchError):
        method(
            TensorSpec("float32", shape=(3,), backend="numpy"),
            TensorSpec("float32", shape=(3,), backend="torch"),
        )
    assert method.cached_signature == signature


def test_python_scalar_options_are_backend_neutral_in_eager_and_cached_calls():
    class BackendOption(Method):
        @traits(backend="tf")
        def tf(self, value, *, training):
            return value, training

    value = TensorSpec("float32", shape=(3,), backend="tf")
    method = object.__new__(BackendOption)
    assert method(value, training=False) == (value, False)

    method.learn()
    assert method(value, training=False) == (value, False)
    assert method(value, training=False) == (value, False)


def test_backend_detector_failures_remain_bounded_and_observable(monkeypatch):
    class DetectorInput:
        pass

    class Generic(Method):
        calls = 0

        @traits()
        def generic(self, value):
            self.calls += 1
            return value

    def fail_detector(value):
        raise RuntimeError("detector failed")

    monkeypatch.setitem(backend_testers, Backend.jax, fail_detector)
    method = object.__new__(Generic)
    with pytest.raises(MethodError, match="detector failed") as error:
        method(DetectorInput())

    assert isinstance(error.value.__cause__, RuntimeError)
    assert method.calls == 0


def test_cached_signature_copies_containers_and_preserves_call_layout():
    """Caller mutation cannot change a cache, while container and argument layout remain exact."""

    spec = TensorSpec("float32", shape=(3,), backend="numpy")
    source = [spec]
    options = {"value": spec}
    method = ExactPrepared()
    method.learn()
    method(source, option=options)
    signature = method.cached_signature
    source.append(spec)
    options["extra"] = spec

    assert method([spec], option={"value": spec})[0] == ([spec],)
    assert method.cached_signature == signature
    with pytest.raises(PreparedCallMismatchError):
        method(source, option={"value": spec})
    with pytest.raises(PreparedCallMismatchError):
        method([spec], options={"value": spec})
    with pytest.raises(PreparedCallMismatchError):
        method(value=[spec], option={"value": spec})
    assert method.cached_signature == signature
