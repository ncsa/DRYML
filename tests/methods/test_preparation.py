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


def test_known_spec_learning_selects_without_a_body_and_validates_dynamic_batches():
    """Known specs prepare once, admit declared dynamic batches, and reject fixed drift."""

    class Counting(Method):
        def __init__(self):
            self.calls = 0

        @traits(backend="numpy", batch_mode="batched")
        def numpy(self, value):
            self.calls += 1
            return value

    method = Counting()
    method.learn(TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy"))

    assert method.call_mode == "cached"
    assert method.calls == 0
    method.implementations = lambda: (_ for _ in ()).throw(AssertionError("must not rediscover"))
    assert method(np.ones((64, 3), dtype=np.float32)).shape == (64, 3)
    assert method(np.ones((17, 3), dtype=np.float32)).shape == (17, 3)
    assert method.calls == 2

    for changed in (
        np.ones((17, 4), dtype=np.float32),
        np.ones((17, 3, 1), dtype=np.float32),
        np.ones((17, 3), dtype=np.float64),
        TensorSpec("float32", shape=(3,), batch=17, backend="torch"),
        {"value": np.ones((17, 3), dtype=np.float32)},
    ):
        with pytest.raises(ImplementationSelectionError):
            method(changed)
    assert method.calls == 2


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


@pytest.mark.parametrize("case", (
    "valid", "shape", "dtype", "empty", "batch", "nested", "subclass", "metadata", "raw",
))
def test_prepared_numpy_validation_work_counts_and_failure_boundaries(monkeypatch, case):
    """Exact dense inputs avoid metadata reconstruction; fallbacks and outputs still validate."""
    from collections import Counter
    import dryml.core.tensor_spec as tensor_spec
    import dryml.methods.implementation as implementation

    class Identity(Method):
        def __init__(self, invalid_output=False):
            self.calls = 0
            self.invalid_output = invalid_output

        def __call__(self, value):
            self.calls += 1
            return np.ones((2, 4), dtype="float32") if self.invalid_output else value

    spec = TensorSpec("float32", shape=(3,), batch=Dynamic, backend="numpy",
                      axis_names=("features",), batch_axis_name="examples")
    value = np.ones((2, 3), dtype="float32")
    if case == "shape":
        value = np.ones((2, 4), dtype="float32")
    elif case == "dtype":
        value = value.astype("float64")
    elif case == "empty":
        value = np.ones((0, 3), dtype="float32")
    elif case == "batch":
        spec = replace(spec, batch=4)
    elif case == "nested":
        spec, value = {"x": spec}, {"x": value}
    elif case == "subclass":
        class Array(np.ndarray):
            pass
        value = value.view(Array)
    elif case == "metadata":
        value = np.ones((2, 3), dtype=np.dtype("float32", metadata={"unit": "m"}))

    method = Identity(invalid_output=case == "raw")
    invoker = method.find_implementation(spec, output_spec=spec if case == "raw" else None).prepared_invoker()
    counts = Counter()
    normalize = tensor_spec.normalize_dtype
    runtime = implementation.runtime_node_for_constraint
    initialize_spec = TensorSpec.__post_init__
    initialize_node = implementation.MethodCallNode.__init__

    def counted_dtype(*args, **kwargs):
        counts["dtype"] += 1
        return normalize(*args, **kwargs)

    def counted_node(*args, **kwargs):
        counts["runtime_node"] += 1
        return runtime(*args, **kwargs)

    def counted_spec(self):
        counts["TensorSpec"] += 1
        initialize_spec(self)

    def counted_node_init(self, *args, **kwargs):
        counts["MethodCallNode"] += 1
        initialize_node(self, *args, **kwargs)

    monkeypatch.setattr(tensor_spec, "normalize_dtype", counted_dtype)
    monkeypatch.setattr(implementation, "runtime_node_for_constraint", counted_node)
    monkeypatch.setattr(TensorSpec, "__post_init__", counted_spec)
    monkeypatch.setattr(implementation.MethodCallNode, "__init__", counted_node_init)
    rejected = case in {"shape", "dtype", "batch", "raw"}
    for _ in range(2):
        if rejected:
            with pytest.raises(ImplementationSelectionError, match="conflict"):
                invoker.invoke((value,), {}, on_raw_result=lambda result: pytest.fail("invalid result published"))
        else:
            assert invoker(value) is value
    assert method.calls == (0 if case in {"shape", "dtype", "batch"} else 2)
    if case in {"nested", "subclass", "metadata", "raw"}:
        assert counts["runtime_node"] == 2
        assert counts["dtype"] > 0
        assert counts["TensorSpec"] > 0
        assert counts["MethodCallNode"] > 0
    else:
        assert counts == Counter()


def test_valid_selection_failure_leaves_learning_without_a_partial_cache():
    """A supported call may retry learning after candidate selection fails."""

    class TorchOnly(Method):
        @traits(backend="torch")
        def torch(self, value):
            raise AssertionError("incompatible target must not run")

    method = object.__new__(TorchOnly)
    method.learn()

    with pytest.raises(ImplementationSelectionError) as error:
        method(np.asarray(["not convertible to Torch"]))
    assert error.value.reason == "no_candidate"
    assert method.call_mode == "learning"
    assert method.cached_signature is None


def test_cached_signature_detects_every_exact_tensor_field_and_preserves_cache():
    """Backend and normalized TensorSpec metadata all participate in exact reuse."""

    baseline = TensorSpec(
        "float32",
        shape=(Dynamic, 3),
        batch=Dynamic,
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


def test_failed_direct_reprepare_clears_old_plan_without_partial_publication():
    """Direct learn retains its clearing semantics, unlike Dataset replacement."""
    method = SimplePrepared()
    spec = TensorSpec("float32", shape=(2,), backend="numpy")
    method.learn(spec)
    with pytest.raises(ImplementationSelectionError):
        method.learn(spec, output_spec=object())
    assert method.call_mode == "learning"
    assert method.cached_signature is None
    assert all(node.selected is None for node in method.method_graph().method_nodes)


def test_learning_conversion_failure_keeps_published_signature(monkeypatch):
    """A retained handoff is target work, after learning's input admission/publication."""
    class Converted(Method):
        @traits(backend="torch")
        def torch(self, value):
            pytest.fail("a failed handoff reached the target")

    import dryml.methods.conversion as conversion

    def fail(*args):
        raise RuntimeError("handoff failed")

    monkeypatch.setattr(conversion, "convert", fail)
    method = object.__new__(Converted)
    method.default_batched = False
    method.learn()
    with pytest.raises(RuntimeError, match="handoff failed"):
        method(np.ones(2, dtype=np.float32))
    assert method.call_mode == "cached"
    assert method.cached_signature is not None


def _numpy_validation_cases(group):
    """Supply bounded marginal cases against the independent generic oracle."""
    base = TensorSpec("float32", shape=(2, 3), backend="numpy")
    value = np.ones((2, 3), dtype="float32")
    if group == "dtypes":
        for name in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                     "float16", "float32", "float64", "complex64", "complex128"):
            for order in ("<", ">"):
                array = np.zeros((2, 3), dtype=np.dtype(name).newbyteorder(order))
                yield replace(base, dtype=name), array
                yield base, array
        for dtype in ("U0", "U7", "O", "S3", "V4", "datetime64[ns]", "timedelta64[D]",
                      np.longdouble, np.clongdouble,
                      np.dtype("float32", metadata={}),
                      np.dtype("float32", metadata={"unit": "m"}),
                      np.dtype([("field", "f4")]),
                      np.dtype([("field", "f4", (2,))], align=True)):
            yield base, np.zeros((2, 3), dtype=dtype)
        yield replace(base, dtype="string"), np.zeros((2, 3), dtype="U7")
        yield replace(base, dtype="object"), np.zeros((2, 3), dtype=object)
        for dtype in (np.longdouble, np.clongdouble):
            yield replace(base, dtype=np.dtype(dtype)), np.zeros((2, 3), dtype=dtype)
        # NumPy expands subarray dtypes into physical axes before validation.
        array = np.zeros((2, 3), dtype=np.dtype(("float32", (2,))))
        yield base, array
        yield replace(base, shape=(2, 3, 2)), array
    elif group == "shapes":
        for shape, batch, physical in (
            ((2, 3), None, (2, 3)), ((2, 3), None, (3, 2)),
            ((Dynamic, 3), None, (0, 3)), ((Dynamic, 3), None, (7, 3)),
            ((Dynamic, 3), None, (7, 4)), ((Dynamic, 3), None, (1, 7, 3)),
            (None, None, ()), (None, None, (0, 3, 2)),
            ((), None, ()), ((), None, (1,)),
            ((3,), Dynamic, (0, 3)), ((3,), Dynamic, (1, 3)), ((3,), Dynamic, (5, 3)),
            ((3,), 2, (2, 3)), ((3,), 2, (1, 3)), ((3,), 0, (0, 3)),
            ((), Dynamic, ()), ((), Dynamic, (0,)), ((), 2, (2,)),
            (None, Dynamic, ()), (None, Dynamic, (0,)), (None, Dynamic, (5, 7, 3)),
            (None, 2, (2, 7, 3)), (None, 2, (1, 7, 3)),
        ):
            spec = replace(base, shape=shape, batch=batch)
            array = np.zeros(physical, dtype="float32")
            yield spec, array
            if shape is not None:
                yield replace(spec, axis_names=tuple(f"axis{i}" for i in range(len(shape))),
                              batch_axis_name="examples" if batch is not None else None), array
    elif group == "layouts":
        for changes in ({"backend": None}, {"backend": "torch"},
                        {"layout": "ragged", "ragged_rank": 1}, {"layout": "sparse"},
                        {"ragged_rank": 1}, {"row_splits_dtype": "int32"},
                        {"sparse_format": "coo"}, {"batch_axis_name": "custom"}):
            yield replace(base, **changes), value
        for array in (value.T, value[:, ::-1], np.asfortranarray(value),
                      np.broadcast_to(value[:1], (2, 3))):
            yield replace(base, shape=array.shape), array
        readonly = value.copy()
        readonly.flags.writeable = False
        yield base, readonly
    elif group == "fallbacks":
        class Array(np.ndarray):
            pass

        for runtime in (value.view(Array), np.float32(1), 1.0, True, object(),
                        base, replace(base, axis_names=("different", "axes")),
                        replace(base, backend="torch"), [value], {"x": value}):
            yield base, runtime
        opaque = TensorSpec("object", layout="python")
        yield opaque, {object(): value}
        yield opaque.with_batch(Dynamic), [value, object()]
        yield opaque.with_batch(0), []
        yield opaque.with_batch(2), [object()]
        yield opaque.with_batch(Dynamic), value
    else:
        spec = {"x": base, "y": (base, [base])}
        runtime = {"x": value, "y": (value, [value])}
        yield spec, runtime
        yield spec, {"y": (value, [value]), "x": value}
        yield spec, {"x": value}
        yield spec, {"x": value, "y": (value, [value, value])}
        yield spec, {"x": value, "y": [value, [value]]}
        yield spec, {**runtime, object(): value}
        yield (base,), [value]
        yield [base], (value,)


@pytest.mark.parametrize("group", ("dtypes", "shapes", "layouts", "fallbacks", "structures"))
def test_compiled_numpy_validation_matches_generic_oracle(group):
    """Compare acceptance and exact normalization errors, including all fallback families."""
    from dryml.methods.signature import _compile_numpy_constraint, runtime_node_for_constraint, satisfies, spec_node

    def outcome(call):
        try:
            return ("result", call())
        except Exception as error:
            return ("error", type(error), str(error))

    for spec, value in _numpy_validation_cases(group):
        constraint = spec_node(spec)
        matcher = _compile_numpy_constraint(constraint)

        def generic():
            return satisfies(constraint, runtime_node_for_constraint(value, constraint))

        def compiled():
            result = None if matcher is None else matcher(value)
            return generic() if result is None else result

        assert outcome(compiled) == outcome(generic), (spec, type(value), getattr(value, "dtype", None))


def test_prepared_numpy_rereads_mutated_metadata_and_keeps_no_values_or_receiver():
    """Reuse validates changed ndarray metadata; immutable predicates permit concurrent calls."""
    from concurrent.futures import ThreadPoolExecutor
    import gc
    import weakref

    method = object.__new__(SimplePrepared)
    spec = TensorSpec("float32", shape=(2, 3), backend="numpy")
    invoker = method.find_implementation(spec).prepared_invoker()
    value = np.zeros((2, 3), dtype="float32")
    assert invoker(value) is value
    value.shape = (3, 2)
    with pytest.raises(ImplementationSelectionError):
        invoker(value)
    value.shape = (2, 3)
    value.dtype = np.int32
    with pytest.raises(ImplementationSelectionError):
        invoker(value)
    value.dtype = np.float32
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoker, np.ones((2, 3), dtype="float32")) for _ in range(4)]
        assert all(future.result().shape == (2, 3) for future in futures)
    value_ref, receiver_ref = weakref.ref(value), weakref.ref(method)
    del value, method
    gc.collect()
    assert value_ref() is receiver_ref() is None
    with pytest.raises(MethodError, match="no longer live"):
        invoker(np.ones((2, 3), dtype="float32"))


def test_compiled_numpy_preserves_activation_order_and_fallback_errors(monkeypatch):
    """All declared adapters activate before short-circuiting validation, as before."""
    from dryml.methods.implementation import _validate_input_specs
    from dryml.methods.signature import _compile_numpy_constraint, spec_node

    constraints = tuple(spec_node(TensorSpec("float32", shape=(3,), backend=backend))
                        for backend in ("numpy", "torch"))
    matchers = tuple(_compile_numpy_constraint(node) for node in constraints)
    events = []

    def activate(backend):
        events.append(backend.value)
        if backend is Backend.torch:
            raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(Backend, "module", property(activate))
    for compiled in ((), matchers):
        events.clear()
        with pytest.raises(RuntimeError, match="adapter unavailable"):
            _validate_input_specs(constraints, (np.zeros(4), object()), compiled)
        assert events == ["numpy", "torch"]
        events.clear()
        with pytest.raises(ImplementationSelectionError):
            _validate_input_specs(constraints, (), compiled)
        assert events == []

    for value in (np.zeros(3, dtype="S2"), np.zeros(3, dtype=[("x", "i4")])):
        errors = []
        for compiled in ((), matchers[:1]):
            with pytest.raises(Exception) as caught:
                _validate_input_specs(constraints[:1], (value,), compiled)
            error = caught.value
            cause = error.__cause__
            errors.append((type(error), str(error), type(cause), str(cause)))
        assert errors[0] == errors[1]


def test_prepared_map_batch_eliminates_per_row_input_reconstruction(monkeypatch):
    """The Dataset hotpath validates each selected array without runtime IR construction."""
    from dryml.data import ArrayDataset, Batch, Map
    import dryml.methods.implementation as implementation

    class Identity(Method):
        def __init__(self):
            self.shapes = []

        def __call__(self, value):
            self.shapes.append(value.shape)
            return value

        def infer_output_spec(self, spec):
            return spec

    values = np.arange(66, dtype="float32").reshape(11, 3, 2)
    spec = TensorSpec("float32", shape=(2,), batch=3, backend="numpy")
    method = Identity()
    prepared = Batch(Map(ArrayDataset(values, spec=spec), method, preserves_examples=True), 4).prepare()

    def forbidden(*args, **kwargs):
        pytest.fail("selected Map input reconstructed runtime tensor metadata")

    monkeypatch.setattr(implementation, "runtime_node_for_constraint", forbidden)
    with prepared.open() as traversal:
        batches = list(traversal.next_epoch())
    assert [batch.shape for batch in batches] == [(4, 3, 2), (4, 3, 2), (3, 3, 2)]
    np.testing.assert_array_equal(np.concatenate(batches), values)
    assert method.shapes == [(3, 2)] * 11
    assert method.call_mode == "eager"
