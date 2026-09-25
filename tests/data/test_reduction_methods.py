"""Native numerical Method and reservoir-reduction conformance tests."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.core.store.dir import DirStore
from dryml.data import Dataset
from dryml.managed import ManagedConfig
from tests.fixtures import require_optional_backend


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_threefry_known_answer_vectors(backend):
    """The private native counter primitive matches Random123 v1.14 vectors."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import _threefry2x32_20

    def tensor(words):
        if backend == "numpy":
            return np.asarray(words, dtype=np.int64)
        if backend == "torch":
            import dryml.torch
            import torch

            return torch.tensor(words, dtype=torch.int64)
        import dryml.tf
        import tensorflow as tf

        return tf.constant(words, dtype=tf.int64)

    def words(value):
        if backend == "numpy":
            return tuple(int(item) for item in value)
        if backend == "torch":
            return tuple(int(item) for item in value.tolist())
        return tuple(int(item) for item in value.numpy().tolist())

    assert words(_threefry2x32_20(tensor((0, 0)), tensor((0, 0)))) == (
        0x6B200159, 0x99BA4EFE,
    )
    assert words(_threefry2x32_20(tensor((0xFFFFFFFF, 0xFFFFFFFF)), tensor((0xFFFFFFFF, 0xFFFFFFFF)))) == (
        0x1CB996FC, 0xBB002BE7,
    )
    assert words(_threefry2x32_20(tensor((0x243F6A88, 0x85A308D3)), tensor((0x13198A2E, 0x03707344)))) == (
        0xC4923A9C, 0x483DF7A0,
    )


def test_element_methods_promote_before_arithmetic_and_reject_broadcasting():
    """Diff/Abs/Squared/Equal keep their explicit numerical contracts."""

    from dryml.data import Abs, Diff, Equal, Squared

    item = {"x": np.array([127, -128], dtype=np.int8), "y": np.array([-1, 1], dtype=np.int8)}
    difference = Diff()(item)

    assert difference.dtype == np.float64
    assert np.array_equal(difference, np.array([128.0, -129.0]))
    assert np.array_equal(Squared()(difference), np.array([16384.0, 16641.0]))
    assert np.array_equal(Abs()(difference), np.array([128.0, 129.0]))
    assert np.array_equal(Equal()(item), np.array([False, False]))
    with pytest.raises(ValueError, match="broadcasting"):
        Diff()({"x": np.ones((2, 1)), "y": np.ones((2,))})
    with pytest.raises(TypeError, match="dtype"):
        Abs()(np.array([True]))


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_native_primitives_reject_nonfinite_intermediate_outputs(backend):
    """Every native arithmetic primitive rejects overflow before returning it."""

    require_optional_backend(backend)
    from dryml.data import Diff, Squared

    if backend == "numpy":
        large = np.array([1e308], dtype=np.float64)
    elif backend == "torch":
        import dryml.torch
        import torch

        large = torch.tensor([1e308], dtype=torch.float64)
    else:
        import dryml.tf
        import tensorflow as tf

        large = tf.constant([1e308], dtype=tf.float64)

    with pytest.raises(ValueError, match="non-finite"):
        Diff()({"x": large, "y": -large})
    with pytest.raises(ValueError, match="non-finite"):
        Squared()(large)


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_native_element_primitives_preserve_promoted_values(backend):
    """The reusable primitive values agree across all required native backends."""

    require_optional_backend(backend)
    from dryml.data import Abs, Diff, Equal, Squared

    left = _native_tensor(backend, [127, -128], dtype="int64")
    right = _native_tensor(backend, [-1, 1], dtype="int64")
    difference = Diff()({"x": left, "y": right})

    assert np.array_equal(_host_values(difference), [128.0, -129.0])
    assert np.array_equal(_host_values(Abs()(difference)), [128.0, 129.0])
    assert np.array_equal(_host_values(Squared()(difference)), [16384.0, 16641.0])
    assert np.array_equal(_host_values(Equal()({"x": left, "y": right})), [False, False])


def test_exact_array_reductions_validate_requests_and_preserve_plural_order():
    """Bounded array reductions use float64 linear semantics without deduplication."""

    from dryml.data import ArrayMean, ArrayQuantile

    values = np.array([1, 3, 9, 11], dtype=np.int16)
    assert ArrayMean() (values) == 6.0
    assert np.allclose(ArrayQuantile((1.0, 0.25, 0.25, 0.0))(values), [11.0, 2.5, 2.5, 1.0], rtol=1e-12, atol=1e-12)
    matrix = np.array([[1, 11], [3, 13], [9, 19]], dtype=np.int16)
    assert np.allclose(ArrayQuantile(0.5, axis=0)(matrix), [3.0, 13.0], rtol=1e-12, atol=1e-12)
    with pytest.raises(TypeError):
        ArrayQuantile(True)
    with pytest.raises(ValueError):
        ArrayQuantile((0.5,))(np.array([], dtype=np.float64))
    with pytest.raises(TypeError):
        ArrayQuantile(0.5)(np.array([True]))


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_array_reductions_have_native_axis_and_dtype_conformance(backend):
    """Array mean/quantile normalize axes and preserve native numerical semantics."""

    require_optional_backend(backend)
    from dryml.data import ArrayMean, ArrayQuantile

    values = np.array([[1, 11], [3, 13], [9, 19]], dtype=np.int16)
    if backend == "torch":
        import dryml.torch
        import torch

        values = torch.tensor(values)
    elif backend == "tf":
        import dryml.tf
        import tensorflow as tf

        values = tf.constant(values)

    assert np.allclose(ArrayMean(axis=-1)(values), [6.0, 8.0, 14.0])
    assert np.allclose(ArrayQuantile((0.0, 0.5, 1.0), axis=(-1,))(values), [[1.0, 3.0, 9.0], [6.0, 8.0, 14.0], [11.0, 13.0, 19.0]])
    with pytest.raises(ValueError, match="duplicate"):
        ArrayMean(axis=(0, -2))(values)
    with pytest.raises(ValueError, match="invalid"):
        ArrayQuantile(0.5, axis=2)(values)


def _native_tensor(backend, values, *, dtype="float64"):
    """Build a test-only CPU tensor without using the reduction implementation."""

    if backend == "numpy":
        return np.asarray(values, dtype=np.float64 if dtype == "float64" else np.int64)
    if backend == "torch":
        import dryml.torch
        import torch

        return torch.tensor(values, dtype=torch.float64 if dtype == "float64" else torch.int64)
    import dryml.tf
    import tensorflow as tf

    return tf.constant(values, dtype=tf.float64 if dtype == "float64" else tf.int64)


def _host_values(value):
    """Convert a completed test result at the explicit host-observation boundary."""

    if isinstance(value, (np.ndarray, np.generic)):
        return value
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return value.numpy()


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_coordinate_element_mean_updates_each_coordinate_once(backend):
    """Unbatched coordinate mean never reduces a whole observation to one scalar."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import MeanFinalize, _mean_initial, _mean_update

    first = _native_tensor(backend, [1, 10])
    state = _mean_initial(first, "coordinate", batched=False)
    state = _mean_update(first, state, "coordinate", batched=False)
    state = _mean_update(_native_tensor(backend, [3, 30]), state, "coordinate", batched=False)

    assert np.array_equal(_host_values(state[0]), [4.0, 40.0])
    assert int(_host_values(state[1])) == 2
    assert np.array_equal(_host_values(MeanFinalize()(state)), [2.0, 20.0])


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_mean_transition_rejects_nonfinite_carry_and_count_wrap(backend):
    """Mean validates native carry before arithmetic can return a bad next state."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import _MAX_INT64, _mean_update

    observation = _native_tensor(backend, [1, 2])
    with pytest.raises(ValueError, match="non-finite"):
        _mean_update(
            observation,
            (_native_tensor(backend, [float("inf"), 0]), _native_tensor(backend, 0, dtype="int64")),
            "coordinate",
            batched=False,
        )
    with pytest.raises(OverflowError, match="count"):
        _mean_update(
            observation,
            (_native_tensor(backend, [0, 0]), _native_tensor(backend, _MAX_INT64, dtype="int64")),
            "coordinate",
            batched=False,
        )


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_mean_accepts_boolean_observations_while_quantile_rejects_them(backend):
    """Boolean equality rates are mean-only; quantile numerical inputs exclude bool."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import _mean_initial, _mean_update, _reservoir_initial

    values = _native_tensor(backend, [True, False])
    if backend == "numpy":
        values = values.astype(np.bool_)
    elif backend == "torch":
        values = values.bool()
    else:
        import tensorflow as tf

        values = tf.cast(values, tf.bool)
    state = _mean_initial(values, "coordinate", batched=False)
    state = _mean_update(values, state, "coordinate", batched=False)
    assert np.array_equal(_host_values(state[0]), [1.0, 0.0])
    with pytest.raises(TypeError, match="dtype"):
        _reservoir_initial(values, "coordinate", 2, 0, batched=False)


class BatchDataset(Dataset):
    """Small explicit batched NumPy source for inert Fold factory tests."""

    def __init__(self, batches):
        self.batches = tuple(batches)
        super().__init__(spec=TensorSpec("float64", shape=(2,), batch=Dynamic, backend="numpy"))

    def __iter__(self):
        """Yield every configured uneven batch in the declared logical order."""

        yield from self.batches


class NativeBatchDataset(Dataset):
    """Generate fixed CPU-native batches only when a Fold starts traversal."""

    def __init__(self, backend, batches):
        self.backend = backend
        self.batches = tuple(tuple(tuple(row) for row in batch) for batch in batches)
        super().__init__(spec=TensorSpec("float64", shape=(2,), batch=Dynamic, backend=backend))

    def __iter__(self):
        """Create fresh backend values without retaining tensors in the declaration."""

        for batch in self.batches:
            if self.backend == "numpy":
                yield np.asarray(batch, dtype=np.float64)
            elif self.backend == "torch":
                import dryml.torch
                import torch

                yield torch.tensor(batch, dtype=torch.float64)
            else:
                import dryml.tf
                import tensorflow as tf

                yield tf.constant(batch, dtype=tf.float64)


class PinnedPopulationDataset(Dataset):
    """Generate the fixed U6 coordinate fixture in bounded unequal working batches."""

    def __init__(self, backend, *, batch_size=127):
        self.backend = backend
        self.batch_size = batch_size
        super().__init__(spec=TensorSpec("float64", shape=(4,), batch=Dynamic, backend=backend))

    def __iter__(self):
        """Yield the four fixed populations without retaining the full population."""

        for start in range(0, 4096, self.batch_size):
            index = np.arange(start, min(start + self.batch_size, 4096), dtype=np.float64)
            u = (index + 0.5) / 4096
            p = (109 * index.astype(np.int64) + 37) % 4096
            values = np.column_stack((u, (p + 0.5) / 4096, np.exp(8 * u), 1 / np.sqrt(1 - u)))
            if self.backend == "numpy":
                yield values
            elif self.backend == "torch":
                import dryml.torch
                import torch

                yield torch.tensor(values, dtype=torch.float64)
            else:
                import dryml.tf
                import tensorflow as tf

                yield tf.constant(values, dtype=tf.float64)


class PinnedGlobalPopulationDataset(Dataset):
    """Generate one independently declared fixed U6 scalar population."""

    def __init__(self, backend, column, *, batch_size=127):
        self.backend = backend
        self.column = column
        self.batch_size = batch_size
        super().__init__(spec=TensorSpec("float64", shape=(), batch=Dynamic, backend=backend))

    def __iter__(self):
        """Yield only the selected pinned recipe in deterministic uneven batches."""

        for start in range(0, 4096, self.batch_size):
            index = np.arange(start, min(start + self.batch_size, 4096), dtype=np.float64)
            u = (index + 0.5) / 4096
            p = (109 * index.astype(np.int64) + 37) % 4096
            columns = (u, (p + 0.5) / 4096, np.exp(8 * u), 1 / np.sqrt(1 - u))
            values = columns[self.column]
            if self.backend == "numpy":
                yield values
            elif self.backend == "torch":
                import dryml.torch
                import torch

                yield torch.tensor(values, dtype=torch.float64)
            else:
                import dryml.tf
                import tensorflow as tf

                yield tf.constant(values, dtype=tf.float64)


def test_mean_factory_is_inert_and_weights_uneven_batches(tmp_path):
    """Global and coordinate means use true scalar/row counts rather than batch means."""

    from dryml.artifacts import mean

    source = BatchDataset((np.array([[1.0, 10.0], [3.0, 30.0]]), np.array([[9.0, 90.0]])))
    global_fold = mean(source, mode="global")
    coordinate_fold = mean(source, mode="coordinate")

    assert not global_fold.ready and not coordinate_fold.ready
    global_fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "global")))
    coordinate_fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "coordinate")))

    assert global_fold.value() == pytest.approx(143 / 6, rel=1e-10, abs=1e-12)
    assert np.allclose(coordinate_fold.value(), [13 / 3, 130 / 3], rtol=1e-10, atol=1e-12)


def test_reduction_factories_normalize_sources_at_the_function_boundary():
    """Factory declarations retain inert source references and reject soft Definitions."""

    from dryml.artifacts import mean
    from dryml.core import Definition, SignatureError

    source = BatchDataset((np.array([[1.0, 2.0]]),))
    fold = mean(source, mode="global")

    assert fold.src == source.definition
    assert not fold.ready
    with pytest.raises(SignatureError, match="authority is unavailable"):
        mean(Definition(BatchDataset, ()), mode="global")


def test_quantile_factory_is_exact_within_capacity_and_reuses_one_summary(tmp_path):
    """Within capacity Algorithm R retains every item and plural requests share it."""

    from dryml.artifacts import quantile

    source = BatchDataset((np.array([[1.0, 10.0], [3.0, 30.0]]), np.array([[9.0, 90.0]])))
    fold = quantile(source, (1.0, 0.5, 0.5, 0.0), mode="coordinate", capacity=3, seed=0)

    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "quantile")))

    assert np.allclose(fold.value(), [[9.0, 90.0], [3.0, 30.0], [3.0, 30.0], [1.0, 10.0]], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_native_mean_and_quantile_fold_conformance_on_uneven_cpu_batches(tmp_path, backend):
    """All required CPU backends retain native transitions and finalization semantics."""

    require_optional_backend(backend)
    from dryml.artifacts import mean, quantile

    batches = (((1.0, 10.0), (3.0, 30.0)), ((9.0, 90.0),))
    mean_fold = mean(NativeBatchDataset(backend, batches), mode="coordinate")
    quantile_fold = quantile(NativeBatchDataset(backend, batches), 0.5, mode="coordinate", capacity=3, seed=0)

    mean_fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / f"mean-{backend}")))
    quantile_fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / f"quantile-{backend}")))

    assert np.allclose(mean_fold.value(), [13 / 3, 130 / 3], rtol=1e-10, atol=1e-12)
    assert np.allclose(quantile_fold.value(), [3.0, 30.0], rtol=1e-12, atol=1e-12)


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
@pytest.mark.parametrize("n", (1, 2, 8, (1 << 62) + 1, (1 << 63) - 1))
@pytest.mark.parametrize("candidate_kind", ("limit_minus_one", "limit", "signed_max"))
def test_bounded_index_controlled_boundary_candidates_advance_exactly_once(
    monkeypatch, backend, n, candidate_kind,
):
    """Every sampler backend handles approved rejection boundaries without wrapping."""

    require_optional_backend(backend)
    import dryml.data.reduction_methods as reductions

    limit = ((reductions._MAX_INT64 // n) * n)
    candidate = {
        "limit_minus_one": limit - 1,
        "limit": limit,
        "signed_max": reductions._MAX_INT64,
    }[candidate_kind]
    attempts = (candidate, 0)

    def controlled_threefry(counter, _key):
        if backend == "numpy":
            draw = counter[0] * (1 << 32) + counter[1]
            selected = np.take(np.asarray(attempts, dtype=np.int64), draw)
            return np.stack((selected // (1 << 32), selected % (1 << 32)))
        if backend == "torch":
            import torch

            draw = counter[0] * (1 << 32) + counter[1]
            selected = torch.tensor(attempts, dtype=torch.int64)[draw]
            return torch.stack((selected // (1 << 32), selected % (1 << 32)))
        import tensorflow as tf

        draw = counter[0] * (1 << 32) + counter[1]
        selected = tf.gather(tf.constant(attempts, dtype=tf.int64), tf.cast(draw, tf.int32))
        return tf.stack((selected // (1 << 32), selected % (1 << 32)))

    monkeypatch.setattr(reductions, "_threefry2x32_20", controlled_threefry)
    key = _native_tensor(backend, (0, 0), dtype="int64")
    counter = _native_tensor(backend, 0, dtype="int64")
    population = _native_tensor(backend, n, dtype="int64")
    if backend == "tf":
        index, next_counter = reductions._tf_bounded_index(key, counter, population)
    else:
        index, next_counter = reductions._bounded_index(key, counter, population, backend)

    assert int(_host_values(index)) == (candidate % n if candidate < limit else 0)
    assert int(_host_values(next_counter)) == (1 if candidate < limit else 2)


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_reservoir_fill_consumes_no_draw_even_at_exhausted_counter(monkeypatch, backend):
    """The first capacity items never evaluate the sampler or advance its counter."""

    require_optional_backend(backend)
    import dryml.data.reduction_methods as reductions

    observation = _native_tensor(backend, [1.0])
    state = reductions._reservoir_initial(observation, "global", 2, 0, batched=False)
    state = (state[0], state[1], _native_tensor(backend, reductions._MAX_INT64, dtype="int64"), state[3])

    def fail_draw(*_args):
        raise AssertionError("fill must not evaluate a sampler draw")

    if backend == "tf":
        reductions._tf_reservoir_loop.cache_clear()
        monkeypatch.setattr(reductions, "_tf_bounded_index", fail_draw)
    else:
        monkeypatch.setattr(reductions, "_bounded_index", fail_draw)

    next_state = reductions._reservoir_update(observation, state, "global", 2, batched=False)
    assert int(_host_values(next_state[1])) == 1
    assert int(_host_values(next_state[2])) == reductions._MAX_INT64


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
@pytest.mark.parametrize(
    ("candidates", "expected_reservoir", "expected_counter"),
    (
        ((0,), [9.0, 2.0], 1),
        ((2,), [1.0, 2.0], 1),
        ((((1 << 63) - 1) // 3 * 3, 1), [1.0, 9.0], 2),
    ),
)
def test_reservoir_replacement_and_rejection_consume_the_pinned_draw_schedule(
    monkeypatch, backend, candidates, expected_reservoir, expected_counter,
):
    """Accepted replacement, non-replacement, and rejection update draw counts exactly."""

    require_optional_backend(backend)
    import dryml.data.reduction_methods as reductions

    def controlled_threefry(counter, _key):
        if backend == "numpy":
            draw = counter[0] * (1 << 32) + counter[1]
            selected = np.take(np.asarray(candidates, dtype=np.int64), draw)
            return np.stack((selected // (1 << 32), selected % (1 << 32)))
        if backend == "torch":
            import torch

            draw = counter[0] * (1 << 32) + counter[1]
            selected = torch.tensor(candidates, dtype=torch.int64)[draw]
            return torch.stack((selected // (1 << 32), selected % (1 << 32)))
        import tensorflow as tf

        draw = counter[0] * (1 << 32) + counter[1]
        selected = tf.gather(tf.constant(candidates, dtype=tf.int64), tf.cast(draw, tf.int32))
        return tf.stack((selected // (1 << 32), selected % (1 << 32)))

    monkeypatch.setattr(reductions, "_threefry2x32_20", controlled_threefry)
    if backend == "tf":
        reductions._tf_reservoir_loop.cache_clear()
    initial = reductions._reservoir_initial(_native_tensor(backend, [1.0, 2.0]), "global", 2, 0, batched=False)
    filled = reductions._reservoir_update(_native_tensor(backend, [1.0, 2.0]), initial, "global", 2, batched=False)
    next_state = reductions._reservoir_update(_native_tensor(backend, [9.0]), filled, "global", 2, batched=False)

    assert np.array_equal(_host_values(next_state[0]), expected_reservoir)
    assert int(_host_values(next_state[2])) == expected_counter


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_reservoir_population_and_draw_overflow_fail_before_native_wrap(backend):
    """Population and draw counters reject their signed-int64 terminal values."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import _MAX_INT64, _reservoir_initial, _reservoir_update

    observation = _native_tensor(backend, [1.0])
    initial = _reservoir_initial(observation, "global", 1, 0, batched=False)
    population_max = (initial[0], _native_tensor(backend, _MAX_INT64, dtype="int64"), initial[2], initial[3])
    with pytest.raises(OverflowError, match="population"):
        _reservoir_update(observation, population_max, "global", 1, batched=False)
    draw_max = (initial[0], _native_tensor(backend, 1, dtype="int64"), _native_tensor(backend, _MAX_INT64, dtype="int64"), initial[3])
    with pytest.raises(OverflowError, match="draw"):
        _reservoir_update(observation, draw_max, "global", 1, batched=False)


def test_reservoir_storage_is_bounded_and_partitions_preserve_the_same_sample(tmp_path):
    """Fixed capacity bounds carry dimensions and item-based draws ignore batch cuts."""

    from dryml.artifacts import quantile
    from dryml.data.reduction_methods import _reservoir_initial, _reservoir_update

    prototype = np.ones((2, 4), dtype=np.float64)
    short = _reservoir_initial(prototype, "coordinate", 5, 0, batched=True)
    long = _reservoir_update(np.ones((100, 4), dtype=np.float64), short, "coordinate", 5, batched=True)
    assert short[0].shape == long[0].shape == (5, 4)
    assert short[1].shape == long[1].shape == ()
    values = np.arange(20, dtype=np.float64).reshape(10, 2)
    one = quantile(BatchDataset((values,)), (0.1, 0.5, 0.9), mode="coordinate", capacity=5, seed=1)
    uneven = quantile(BatchDataset((values[:3], values[3:9], values[9:])), (0.1, 0.5, 0.9), mode="coordinate", capacity=5, seed=1)
    one.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "one")))
    uneven.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "uneven")))
    assert np.array_equal(one.value(), uneven.value())
    direct = _reservoir_initial(values, "coordinate", 5, 1, batched=True)
    direct = _reservoir_update(values, direct, "coordinate", 5, batched=True)
    partitioned = _reservoir_initial(values[:3], "coordinate", 5, 1, batched=True)
    for batch in (values[:3], values[3:9], values[9:]):
        partitioned = _reservoir_update(batch, partitioned, "coordinate", 5, batched=True)
    for direct_value, partitioned_value in zip(direct, partitioned):
        assert np.array_equal(direct_value, partitioned_value)


@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_coordinate_reservoir_element_and_uneven_batches_have_identical_membership(backend):
    """Coordinate row reservoirs keep the same sample across element/batch partitions."""

    require_optional_backend(backend)
    from dryml.data.reduction_methods import _reservoir_initial, _reservoir_update

    rows = ((1.0, 10.0), (3.0, 30.0), (9.0, 90.0))
    element = _reservoir_initial(_native_tensor(backend, rows[0]), "coordinate", 2, 1, batched=False)
    for row in rows:
        element = _reservoir_update(_native_tensor(backend, row), element, "coordinate", 2, batched=False)
    batched = _reservoir_initial(_native_tensor(backend, rows[:2]), "coordinate", 2, 1, batched=True)
    for batch in (rows[:2], rows[2:]):
        batched = _reservoir_update(_native_tensor(backend, batch), batched, "coordinate", 2, batched=True)
    for element_value, batched_value in zip(element, batched):
        assert np.array_equal(_host_values(element_value), _host_values(batched_value))


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_native_reduction_transitions_do_not_extract_numerical_values(monkeypatch, backend):
    """Transitions/finalizers keep native data native; host conversion is terminal only."""

    require_optional_backend(backend)
    import dryml.data.reduction_methods as reductions

    observation = _native_tensor(backend, [1.0, 2.0, 3.0])
    state = reductions._reservoir_initial(observation, "global", 1, 0, batched=False)
    def fail_asarray(*_args, **_kwargs):
        raise AssertionError("transition must not convert a native tensor with np.asarray")

    monkeypatch.setattr(np, "asarray", fail_asarray)
    if backend == "torch":
        import torch

        monkeypatch.setattr(torch.Tensor, "numpy", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected Tensor.numpy")))
        monkeypatch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected Tensor.cpu")))
        monkeypatch.setattr(torch.Tensor, "item", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected Tensor.item")))
    else:
        original_numpy = type(observation).numpy

        def guard_direct_numpy(value, *_args, **_kwargs):
            """Reject direct reducer extraction but allow TensorFlow's own control predicates."""

            import inspect

            caller = inspect.currentframe().f_back
            if caller is not None and caller.f_code.co_filename.endswith("reduction_methods.py"):
                raise AssertionError("unexpected direct Tensor.numpy")
            return original_numpy(value, *_args, **_kwargs)

        monkeypatch.setattr(type(observation), "numpy", guard_direct_numpy)

    next_state = reductions._reservoir_update(observation, state, "global", 1, batched=False)
    native_result = reductions._reservoir_quantile(next_state[0], next_state[1], backend, (0.5,), plural=False)

    monkeypatch.undo()
    assert np.isfinite(_host_values(native_result)).all()


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_fold_terminal_normalization_converts_only_after_native_traversal(monkeypatch, tmp_path, backend):
    """Fold observes its one host conversion after, not during, all native transitions."""

    require_optional_backend(backend)
    import importlib

    from dryml.artifacts import mean

    module = importlib.import_module("dryml.artifacts.fold")
    source = NativeBatchDataset(backend, (((1.0, 10.0),), ((3.0, 30.0),)))
    fold = mean(source, mode="coordinate")
    original_iter = NativeBatchDataset.__iter__
    traversal = {"finished": False}

    def observe_iter(instance):
        """Observe exhaustion of the materialized source iterator."""

        yield from original_iter(instance)
        traversal["finished"] = True

    monkeypatch.setattr(NativeBatchDataset, "__iter__", observe_iter)
    original_terminal = module._terminal_host_value
    calls = []

    def observe_terminal(value):
        """Record the explicit post-loop conversion boundary."""

        assert traversal["finished"]
        calls.append(type(value).__module__)
        return original_terminal(value)

    monkeypatch.setattr(module, "_terminal_host_value", observe_terminal)
    fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / backend)))

    assert len(calls) == 1
    assert np.allclose(fold.value(), [2.0, 20.0])


def test_quantile_rejects_invalid_configuration_and_nonfinite_fold_input(tmp_path):
    """Invalid q/capacity/seed and data fail before a Value payload is installed."""

    from dryml.artifacts import quantile

    source = BatchDataset((np.array([[1.0, 2.0]]),))
    for q, capacity, seed in ((True, 2, 0), ((0.5,), 0, 0), ((0.5,), 2, -1)):
        with pytest.raises((TypeError, ValueError)):
            quantile(source, q, mode="global", capacity=capacity, seed=seed)
    invalid = quantile(BatchDataset((np.array([[np.inf, 2.0]]),)), 0.5, mode="global", capacity=2)
    with pytest.raises(ValueError, match="non-finite"):
        invalid.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / "invalid")))
    assert not invalid.ready


def _pinned_reference():
    """Build the plan-pinned reference values independently of reduction code."""

    index = np.arange(4096, dtype=np.float64)
    u = (index + 0.5) / 4096
    p = (109 * index.astype(np.int64) + 37) % 4096
    return np.column_stack((u, (p + 0.5) / 4096, np.exp(8 * u), 1 / np.sqrt(1 - u)))


def _rank_interval_distance(ordered, value, q):
    """Measure distance to the duplicate-safe empirical CDF interval at ``value``."""

    lower = np.searchsorted(ordered, value, side="left") / len(ordered)
    upper = np.searchsorted(ordered, value, side="right") / len(ordered)
    return max(lower - q, q - upper, 0.0)


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
def test_reservoir_fixed_empirical_coordinate_qualification(tmp_path, backend):
    """Pinned U6 fixtures meet the approved rank interval threshold without tuning."""

    require_optional_backend(backend)
    from dryml.artifacts import quantile

    reference = _pinned_reference()
    q = (0.01, 0.5, 0.99)
    for seed in (0, 1, 0x243F6A8885A308D3):
        fold = quantile(PinnedPopulationDataset(backend), q, mode="coordinate", capacity=256, seed=seed)
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / f"{backend}-{seed}")))
        actual = fold.value()
        for column in range(4):
            ordered = np.sort(reference[:, column])
            for request, value in zip(q, actual[:, column]):
                assert _rank_interval_distance(ordered, value, request) <= 0.125


@pytest.mark.exhaustive_only
@pytest.mark.parametrize("backend", ("numpy", "torch", "tf"))
@pytest.mark.parametrize("column", range(4))
def test_reservoir_fixed_empirical_global_qualification(tmp_path, backend, column):
    """Every pinned recipe also qualifies through the independent global path."""

    require_optional_backend(backend)
    from dryml.artifacts import quantile

    reference = _pinned_reference()[:, column]
    q = (0.01, 0.5, 0.99)
    for seed in (0, 1, 0x243F6A8885A308D3):
        fold = quantile(PinnedGlobalPopulationDataset(backend, column), q, mode="global", capacity=256, seed=seed)
        fold.compute(managed=ManagedConfig(state_repo=DirStore(tmp_path / f"{backend}-{column}-{seed}")))
        for request, value in zip(q, fold.value()):
            assert _rank_interval_distance(np.sort(reference), value, request) <= 0.125
