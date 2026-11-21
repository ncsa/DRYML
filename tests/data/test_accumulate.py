"""Streaming carry ownership, retained selection, and exact continuation."""

import numpy as np
import pytest

import dryml.data as data
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import TensorSpec
from dryml.methods import Accumulator, ImplementationSelectionError, Method, traits


SCALAR = TensorSpec("int64", shape=(), backend="numpy")


class Zero(Method):
    """Initialize fresh scalar carry from a real observation only."""

    def __init__(self):
        self.calls = self.selections = 0

    def find_implementation(self, *args, **kwargs):
        self.selections += 1
        return super().find_implementation(*args, **kwargs)

    @traits(backend="numpy")
    def numpy(self, observation):
        self.calls += 1
        return np.int64(0)

    def infer_output_spec(self, observation_spec):
        return SCALAR


class Sum(Accumulator):
    """Record observation-first transitions without storing execution carry."""

    def __init__(self):
        self.calls = self.selections = 0

    def find_implementation(self, *args, **kwargs):
        self.selections += 1
        return super().find_implementation(*args, **kwargs)

    @traits(backend="numpy")
    def numpy(self, observation, state):
        self.calls += 1
        return state + observation.sum(dtype=np.int64)

    def infer_output_spec(self, observation_spec, state_spec):
        return state_spec


def test_capture_restores_carry_without_repeating_accepted_transitions(tmp_path):
    """Restore both midsection and boundary cuts, including persisted bytes."""
    initial, transition = Zero(), Sum()
    dataset = data.Accumulate(data.Epoch(data.ArrayDataset(np.arange(1, 7)), len=3),
                              initial_state=initial, accumulator=transition)
    prepared = dataset.prepare()
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        assert next(view) == 1
        middle = traversal.capture_state()
        middle.save(tmp_path / "middle")
        assert list(view) == [3, 6]
        boundary = traversal.capture_state()
        assert list(traversal.next_epoch()) == [10, 15, 21]
    assert (initial.calls, transition.calls) == (1, 6)
    for record, suffix in ((data.DatasetState.load(tmp_path / "middle"), [3, 6, 10, 15, 21]),
                           (boundary, [10, 15, 21])):
        before = transition.calls
        with prepared.open(state=record) as resumed:
            assert [value for _, view in resumed.epochs() for value in view] == suffix
        assert transition.calls - before == len(suffix)
        assert initial.calls == 1


def totals(source=None, *, initial=None, transition=None):
    return data.Accumulate(data.ArrayDataset(np.arange(1, 4)) if source is None else source,
                           initial_state=Zero() if initial is None else initial,
                           accumulator=Sum() if transition is None else transition)


def test_ae13_one_post_transition_output_per_input_and_empty_never_initializes():
    initial, transition = Zero(), Sum()
    dataset = totals(initial=initial, transition=transition)
    assert dataset.yield_cardinality() == dataset.example_cardinality() == Cardinality.finite(3)
    assert dataset.spec == SCALAR
    assert list(dataset) == [1, 3, 6]
    assert (initial.calls, transition.calls) == (1, 3)
    empty = totals(data.ArrayDataset(np.array([], dtype=np.int64)), initial=initial, transition=transition)
    assert list(empty) == []
    with empty.open() as traversal:
        assert list(traversal.epochs()) == []
    assert (initial.calls, transition.calls) == (1, 3)


class First(Zero):
    @traits(backend="numpy")
    def numpy(self, observation):
        self.calls += 1
        return observation


def test_first_observation_initializes_once_and_then_transitions_once():
    initial, transition = First(), Sum()
    assert list(totals(initial=initial, transition=transition)) == [2, 4, 7]
    assert (initial.calls, transition.calls) == (1, 3)


def test_two_sessions_share_prepared_selections_and_implicit_passes_reset(monkeypatch):
    from dryml.core import signatures

    initial, transition = Zero(), Sum()
    dataset = totals(initial=initial, transition=transition)
    prepared = dataset.prepare()
    assert (initial.selections, transition.selections) == (1, 1)
    assert (initial.calls, transition.calls) == (0, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("shared prepared roles recompiled during execution")

    monkeypatch.setattr(signatures, "compile_signature", forbidden)
    with prepared.open() as left, prepared.open() as right:
        lview, rview = left.next_epoch(), right.next_epoch()
        assert next(lview) == next(rview) == 1
        assert list(lview) == [3, 6]
        assert list(rview) == [3, 6]
        assert list(left.next_epoch()) == [1, 3, 6]
    assert (initial.selections, transition.selections) == (1, 1)
    assert (initial.calls, transition.calls) == (3, 9)
    assert not hasattr(transition, "carry")


class MutableZero(Zero):
    @traits(backend="numpy")
    def numpy(self, observation):
        self.calls += 1
        return np.array(0, dtype=np.int64)


class MutableSum(Sum):
    @traits(backend="numpy")
    def numpy(self, observation, state):
        self.calls += 1
        state += observation.sum(dtype=np.int64)
        return state


def test_mutating_carry_preserves_emissions_records_and_independent_restores():
    initial, transition = MutableZero(), MutableSum()
    prepared = totals(initial=initial, transition=transition).prepare()
    with prepared.open() as live, prepared.open() as sibling:
        view = live.next_epoch()
        first = next(view)
        record = live.capture_state()
        first[...] = 99
        tail = list(view)
        assert [int(v) for v in tail] == [3, 6]
        assert list(sibling.next_epoch()) == [1, 3, 6]
    with prepared.open(state=record) as left, prepared.open(state=record) as right:
        lv, rv = left.next_epoch(), right.next_epoch()
        lvalue = next(lv)
        lvalue[...] = -1
        assert next(rv) == 3
        assert next(lv) == next(rv) == 6
    assert initial.calls == 2  # restoring a carry does not initialize again


class StructuredZero(Zero):
    @traits(backend="numpy")
    def numpy(self, observation):
        self.calls += 1
        return {"sum": np.int64(0), "count": np.int64(0)}

    def infer_output_spec(self, observation_spec):
        return {"sum": SCALAR, "count": SCALAR}


class StructuredSum(Sum):
    @traits(backend="numpy")
    def numpy(self, observation, state):
        self.calls += 1
        return {"sum": state["sum"] + observation.sum(dtype=np.int64),
                "count": state["count"] + observation.size}


@pytest.mark.parametrize("structured", [False, True])
def test_batched_vectors_have_declared_scalar_or_structured_carry(structured):
    source = data.Batch(data.ArrayDataset(np.arange(1, 7).reshape(3, 2)), 2)
    initial, transition = (StructuredZero(), StructuredSum()) if structured else (Zero(), Sum())
    dataset = totals(source, initial=initial, transition=transition)
    prepared = dataset.prepare()
    assert prepared.spec == initial.infer_output_spec(source.spec)
    assert dataset.yield_cardinality() == Cardinality.finite(2)
    assert dataset.example_cardinality() == Cardinality.finite(2)
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        first = next(view)
        record = traversal.capture_state()
        assert list(view) == ([{"sum": 21, "count": 6}] if structured else [21])
    assert first == ({"sum": 10, "count": 4} if structured else 10)
    with prepared.open(state=record) as resumed:
        assert list(resumed.next_epoch()) == ([{"sum": 21, "count": 6}] if structured else [21])


_acquisitions = []


def observations():
    _acquisitions.append("open")
    yield np.int64(1)


class BadInitial(Zero):
    @traits(backend="numpy", batch_mode="batched")
    def numpy(self, observation):
        raise AssertionError("incompatible initializer executed")


class BadTransition(Sum):
    def infer_output_spec(self, observation_spec, state_spec):
        return TensorSpec("int64", shape=(2,), backend="numpy")


@pytest.mark.parametrize("role", ["initializer", "transition"])
def test_incompatible_roles_fail_before_source_acquisition(role):
    _acquisitions.clear()
    source = data.GeneratorDataset(observations, spec=SCALAR)
    dataset = totals(source, initial=BadInitial() if role == "initializer" else Zero(),
                     transition=BadTransition() if role == "transition" else Sum())
    with pytest.raises(ImplementationSelectionError):
        dataset.prepare()
    assert _acquisitions == []


def test_unqualified_custom_composition_fails_explicitly_without_recursive_fallback():
    class Legacy(data.Dataset):
        def __init__(self, src):
            self.src = src
            super().__init__(spec=src.spec)

        def __iter__(self):
            return iter(self.src)

    _acquisitions.clear()
    dataset = totals(Legacy(data.GeneratorDataset(observations, spec=SCALAR)))
    with pytest.raises(NotImplementedError, match="planning does not support"):
        iter(dataset)
    assert _acquisitions == []


def test_repeat_occurrence_order_and_empty_leading_sections():
    src = data.ArrayDataset(np.array([1, 2], dtype=np.int64))
    for dataset, expected in ((data.Repeat(totals(src), 2), [1, 3, 1, 3]),
                              (totals(data.Repeat(src, 2)), [1, 3, 4, 6])):
        assert list(dataset) == expected
        assert list(dataset.prepare().iterator()) == expected
    initial, transition = Zero(), Sum()
    source = data.Skip(data.Epoch(data.ArrayDataset(np.arange(1, 7)), len=2), 2)
    prepared = totals(source, initial=initial, transition=transition).prepare()
    with prepared.open() as traversal:
        assert list(traversal.next_epoch()) == []
        assert (initial.calls, transition.calls) == (0, 0)
        record = traversal.capture_state()
        assert list(traversal.next_epoch()) == [3, 7]
        assert list(traversal.next_epoch()) == [12, 18]
    with prepared.open(state=record) as resumed:
        assert [v for _, view in resumed.epochs() for v in view] == [3, 7, 12, 18]
    assert initial.calls == 2


def test_immutable_carry_live_path_does_not_snapshot_encode_or_clone(monkeypatch):
    import copy
    import dryml.data.continuation as codec
    import dryml.data.recovery as recovery
    prepared = totals().prepare()

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery-only work during live accumulation")

    monkeypatch.setattr(codec, "encode_value", forbidden)
    monkeypatch.setattr(recovery, "_capture_execution", forbidden)
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    monkeypatch.setattr(np, "copy", forbidden)
    with prepared.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        outputs = []
        for _ in range(3):
            pending = view.reserve()
            assert pending.value is traversal._execution.carry
            traversal.accept(pending)
            outputs.append(pending.value)
        assert outputs == [1, 3, 6]


OPAQUE = TensorSpec("object", layout="python")


class DynamicInitial(Method):
    def __call__(self, observation):
        return 0

    def infer_output_spec(self, observation_spec):
        return OPAQUE


class DynamicTransition(Accumulator):
    def __call__(self, observation, state):
        return int(observation) if observation < 2 else object()

    def infer_output_spec(self, observation_spec, state_spec):
        return state_spec


def test_unsupported_dynamic_carry_fails_only_capture_without_falsifying_progress():
    prepared = totals(initial=DynamicInitial(), transition=DynamicTransition()).prepare()
    with prepared.open(acceptance="explicit") as traversal:
        view = traversal.next_epoch()
        candidate = view.reserve()
        traversal.accept(candidate)
        earlier = traversal.capture_state()
        candidate = view.reserve()
        traversal.accept(candidate)
        before = traversal.position
        with pytest.raises(data.UnsupportedContinuationError):
            traversal.capture_state()
        assert traversal.position == before
        assert traversal.position.sequence == 2
        assert earlier.position.sequence == 1
        candidate = view.reserve()
        traversal.accept(candidate)
        assert traversal.position.sequence == 3
    with prepared.open(state=earlier) as resumed:
        assert len(list(resumed.next_epoch())) == 2


BATCH_CARRY = TensorSpec("int64", shape=(), batch=2, backend="numpy")


class VectorZero(Zero):
    @traits(backend="numpy")
    def numpy(self, observation):
        self.calls += 1
        return np.zeros(2, dtype=np.int64)

    def infer_output_spec(self, observation_spec):
        return BATCH_CARRY


class VectorSum(Sum):
    @traits(backend="numpy")
    def numpy(self, observation, state):
        self.calls += 1
        state += observation
        return state


@pytest.mark.parametrize("buffer", ["shuffle", "unbatch"])
def test_pending_outputs_restore_without_duplicate_transitions_or_mutable_aliasing(buffer):
    initial, transition = VectorZero(), VectorSum()
    source = totals(data.ArrayDataset(np.arange(1, 13).reshape(6, 2)), initial=initial, transition=transition)
    dataset = data.Shuffle(source, 3, seed=7) if buffer == "shuffle" else data.Unbatch(source)
    prepared = dataset.prepare()
    convert = lambda value: np.asarray(value).tolist()
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        accepted_transitions = transition.calls
        record = traversal.capture_state()
        expected = [convert(v) for v in view]
    for _ in range(2):
        before = transition.calls
        with prepared.open(state=record) as resumed:
            view = resumed.next_epoch()
            first = next(view)
            actual = [convert(first)]
            if isinstance(first, np.ndarray):
                first[...] = -99
            actual.extend(convert(v) for v in view)
            assert actual == expected
        assert transition.calls - before == 6 - accepted_transitions
    assert initial.calls == 1


@pytest.mark.parametrize("mutation", ["flag", "role", "extra"])
def test_accumulate_state_rejects_malformed_carry_and_role_records(mutation):
    from dryml.data.continuation import encode_value
    with totals().open() as traversal:
        next(traversal.next_epoch())
        state = traversal.capture_state()._data()
    if mutation == "flag":
        state["execution"]["carry_initialized"] = False
    elif mutation == "role":
        state["target"][0]["selection"]["roles"].pop()
    else:
        state["execution"]["unknown_carry"] = 1
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState._from_bytes(*encode_value(state))


_fetched = []


def replay_observations():
    for value in range(1, 7):
        _fetched.append(value)
        yield np.int64(value)


def test_source_replay_does_not_reexecute_accepted_accumulator_transitions():
    initial, transition = Zero(), Sum()
    prepared = totals(data.GeneratorDataset(replay_observations, spec=SCALAR),
                      initial=initial, transition=transition).prepare()
    with prepared.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        assert [next(view), next(view), next(view)] == [1, 3, 6]
        record = traversal.capture_state()
    _fetched.clear()
    with prepared.open(state=record) as resumed:
        assert list(resumed.next_epoch()) == [10, 15, 21]
    assert _fetched == list(range(1, 7))
    assert (initial.calls, transition.calls) == (1, 6)


class NativeInitial(Method):
    @traits(backend="jax")
    def numpy(self, observation):
        import jax.numpy as jnp
        return jnp.zeros((), dtype=jnp.int32)

    def infer_output_spec(self, observation_spec):
        return TensorSpec("int32", shape=(), backend="jax")


class NativeSum(Accumulator):
    def __init__(self):
        import jax
        self.advance = jax.jit(lambda observation, state: state + observation)

    @traits(backend="jax")
    def numpy(self, observation, state):
        return self.advance(observation, state)

    def infer_output_spec(self, observation_spec, state_spec):
        return state_spec


def native_observations():
    import jax.numpy as jnp
    for value in range(1, 4):
        yield jnp.asarray(value, dtype=jnp.int32)


def test_native_functional_carry_has_no_recovery_clone_encoding_or_host_conversion(monkeypatch):
    """A real CPU JAX carry stays native on the live path, including compiled updates."""
    jax = pytest.importorskip("jax")
    import dryml.jax  # noqa: F401 - explicitly activate the selected native adapter
    import copy
    import dryml.data.continuation as codec
    import dryml.data.recovery as recovery

    prepared = totals(data.GeneratorDataset(native_observations,
                                           spec=TensorSpec("int32", shape=(), backend="jax")),
                      initial=NativeInitial(), transition=NativeSum()).prepare()
    with prepared.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        view = traversal.next_epoch()
        first = next(view)
        native_type = type(first)

        def forbidden(*args, **kwargs):
            raise AssertionError("native carry recovery-only clone/snapshot/host conversion")

        with monkeypatch.context() as patch:
            patch.setattr(native_type, "__array__", forbidden)
            patch.setattr(native_type, "copy", forbidden)
            patch.setattr(copy, "deepcopy", forbidden)
            patch.setattr(codec, "encode_value", forbidden)
            patch.setattr(recovery, "_capture_execution", forbidden)
            outputs = [first, next(view), next(view)]
            assert outputs[-1] is traversal._execution.carry
        assert all(isinstance(value, jax.Array) for value in outputs)
        assert [int(value) for value in outputs] == [1, 3, 6]
