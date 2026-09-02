import subprocess
import sys

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.backend import Backend
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data.dataset import Dataset, Map
from dryml.methods import ImplementationSelectionError, Method, PreparedCallMismatchError, traits
from dryml.data import (
    ArgMax,
    Batch,
    Cast,
    Flatten,
    Pipe,
    Project,
    Repeat,
    Scale,
    Select,
    Shuffle,
    Skip,
    Take,
    Unbatch,
    Zip,
    Chain,
)


class ListDataset(Dataset):
    def __init__(self, items, spec):
        self.items = list(items)
        super().__init__(spec=spec)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return Cardinality.finite(len(self.items))


class CountingCast(Cast):
    def __init__(self, dtype):
        super().__init__(dtype)
        self.dispatch_count = 0

    def find_implementation(self, *args, **kwargs):
        self.dispatch_count += 1
        return super().find_implementation(*args, **kwargs)

    def _prepare_implementation(self, *args, **kwargs):
        self.dispatch_count += 1
        return super()._prepare_implementation(*args, **kwargs)


class OneShotDataset(Dataset):
    """A deliberately one-shot source that records each attempted consumption."""

    def __init__(self, items, spec):
        self._items = iter(items)
        self.next_calls = 0
        super().__init__(spec=spec)

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        return next(self._items)

    def __len__(self):
        return Cardinality.UNKNOWN


class CountingNumpy(Method):
    """Count target calls so selected-call validation can be observed."""

    def __init__(self):
        self.calls = 0

    @traits(backend="numpy")
    def numpy(self, value):
        self.calls += 1
        return value

    def infer_output_spec(self, input_spec):
        return input_spec


class CountingArgMax(ArgMax):
    """Record local child specialization without changing ArgMax behavior."""

    def __init__(self):
        super().__init__()
        self.dispatch_count = 0

    def find_implementation(self, *args, **kwargs):
        self.dispatch_count += 1
        return super().find_implementation(*args, **kwargs)


def test_cast_infer_output_spec_and_iteration():
    src = ListDataset(
        [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Map(src, Cast("float32"))
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert ds.spec.backend is Backend.numpy
    assert [item.dtype for item in out] == [np.dtype("float32"), np.dtype("float32")]


def test_select_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)},
            {"x": np.array([4, 5], dtype=np.int32), "y": np.array([6], dtype=np.int32)},
        ],
        {
            "x": TensorSpec("int32", shape=(2,), backend="numpy"),
            "y": TensorSpec("int32", shape=(1,), backend="numpy"),
        },
    )

    ds = Map(src, Select("x"))

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[1, 2], [4, 5]]


def test_select_accepts_multiple_indices():
    src = ListDataset(
        [
            {"x": (np.array([1, 2], dtype=np.int32),)},
            {"x": (np.array([3, 4], dtype=np.int32),)},
        ],
        {"x": (TensorSpec("int32", shape=(2,), backend="numpy"),)},
    )

    ds = Map(src, Select("x", 0))

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[1, 2], [3, 4]]


def test_select_accepts_tuple_path():
    src = ListDataset(
        [
            {"x": (np.array([1, 2], dtype=np.int32),)},
            {"x": (np.array([3, 4], dtype=np.int32),)},
        ],
        {"x": (TensorSpec("int32", shape=(2,), backend="numpy"),)},
    )

    ds = Map(src, Select(("x", 0)))

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in ds] == [[1, 2], [3, 4]]


def test_elementwise_dataset_resolves_dispatch_once_per_iterator():
    transform = CountingCast("float32")
    src = ListDataset(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Map(src, transform)

    list(ds)
    list(ds)

    assert transform.dispatch_count == 2


def test_map_selects_before_consuming_complete_specs():
    transform = CountingCast("float32")
    src = OneShotDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    assert [item.dtype for item in Map(src, transform)] == [np.dtype("float32")]
    assert transform.dispatch_count == 1


def test_empty_complete_spec_map_selects_without_invoking_a_target():
    transform = CountingCast("float32")
    src = ListDataset([], TensorSpec("int32", shape=(1,), backend="numpy"))

    assert list(Map(src, transform)) == []
    assert transform.dispatch_count == 1


def test_map_backend_fallback_reads_only_the_first_value_before_selection():
    src = OneShotDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend=None),
    )

    class ObservingCast(CountingCast):
        def __init__(self):
            super().__init__("float32")
            self.next_counts = []

        def find_implementation(self, *args, **kwargs):
            self.next_counts.append(self.source.next_calls)
            return super().find_implementation(*args, **kwargs)

    transform = ObservingCast()
    transform.source = src
    assert [item.tolist() for item in Map(src, transform)] == [[1.0], [2.0]]
    assert transform.dispatch_count == 2
    assert transform.next_counts == [0, 1]


def test_map_empty_missing_backend_only_checks_exhaustion():
    transform = CountingCast("float32")
    src = OneShotDataset([], TensorSpec("int32", shape=(1,), backend=None))

    assert list(Map(src, transform)) == []
    assert transform.dispatch_count == 1
    assert src.next_calls == 1


def test_map_selected_callable_rejects_later_backend_conflicts_before_target():
    transform = CountingNumpy()
    src = ListDataset(
        [
            np.array([1], dtype=np.int32),
            TensorSpec("int32", shape=(1,), backend="torch"),
        ],
        TensorSpec("int32", shape=(1,), backend=None),
    )

    with pytest.raises(ImplementationSelectionError) as error:
        list(Map(src, transform))
    assert error.value.reason == "conflict"
    assert transform.calls == 1


@pytest.mark.parametrize(
    "error",
    (
        ImplementationSelectionError("ambiguous"),
        ImplementationSelectionError("conflict"),
        ImplementationSelectionError("no_candidate"),
        ImplementationSelectionError("unknown_traits", ("batch_mode",)),
        ImplementationSelectionError("unknown_traits", ("backend", "batch_mode")),
        ImplementationSelectionError("unknown_traits", ("backend", "malformed")),
    ),
)
def test_map_only_falls_back_for_the_exact_missing_backend_error(error):
    class FailingMethod(Method):
        def __call__(self, value):
            raise AssertionError("Map must not invoke a failed selection")

        def infer_output_spec(self, input_spec):
            return input_spec

        def find_implementation(self, *args, **kwargs):
            raise self.error

    src = OneShotDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend=None),
    )

    method = FailingMethod()
    method.error = error
    with pytest.raises(ImplementationSelectionError) as observed:
        list(Map(src, method))
    assert observed.value is error
    assert src.next_calls == 0


def test_pipe_infer_output_spec_and_call():
    pipe = Pipe(Select("x"), Cast("float32"))
    spec = {
        "x": TensorSpec("int32", shape=(2,), backend="numpy"),
        "y": TensorSpec("int32", shape=(1,), backend="numpy"),
    }
    x = {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)}

    out = pipe(x)

    assert pipe.infer_output_spec(spec) == TensorSpec("float32", shape=(2,), backend="numpy")
    assert out.dtype == np.dtype("float32")
    assert out.tolist() == [1.0, 2.0]


def test_project_positional_branches_use_one_input_element():
    project = Project(Select("x"), Select("y"))
    spec = {
        "x": TensorSpec("int32", shape=(2,), backend="numpy"),
        "y": TensorSpec("int32", shape=(1,), backend="numpy"),
    }
    value = {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)}

    out = project(value)

    assert project.infer_output_spec(spec) == (spec["x"], spec["y"])
    assert out[0].tolist() == [1, 2]
    assert out[1].tolist() == [3]


def test_project_named_branches_use_one_input_element():
    src = ListDataset(
        [
            {"x": np.array([1, 2], dtype=np.int32), "y": np.array([3], dtype=np.int32)},
            {"x": np.array([4, 5], dtype=np.int32), "y": np.array([6], dtype=np.int32)},
        ],
        {
            "x": TensorSpec("int32", shape=(2,), backend="numpy"),
            "y": TensorSpec("int32", shape=(1,), backend="numpy"),
        },
    )

    ds = Map(src, Project(x=Select("x"), y=Select("y")))

    assert ds.spec == {"x": src.spec["x"], "y": src.spec["y"]}
    assert [item["x"].tolist() for item in ds] == [[1, 2], [4, 5]]


def test_map_accepts_multiple_transforms_as_pipe():
    transform = CountingCast("float32")
    src = ListDataset(
        [
            {"x": np.array([1, 2], dtype=np.int32)},
            {"x": np.array([3, 4], dtype=np.int32)},
        ],
        {"x": TensorSpec("int32", shape=(2,), backend="numpy")},
    )

    ds = Map(src, Select("x"), transform)
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    assert [item.dtype for item in out] == [np.dtype("float32"), np.dtype("float32")]
    assert transform.dispatch_count == 1


def test_project_and_pipe_select_children_once_from_structural_specs():
    first = CountingCast("float32")
    second = CountingCast("float64")
    src = ListDataset(
        [{"x": np.array([1], dtype=np.int32)}, {"x": np.array([2], dtype=np.int32)}],
        {"x": TensorSpec("int32", shape=(1,), backend="numpy")},
    )

    projected = Map(src, Project(left=Select("x"), right=Pipe(Select("x"), first)))
    piped = Map(src, Pipe(Select("x"), second))

    assert [item["right"].dtype for item in projected] == [np.dtype("float32")] * 2
    assert [item.dtype for item in piped] == [np.dtype("float64")] * 2
    assert first.dispatch_count == 1
    assert second.dispatch_count == 1

    argmax = CountingArgMax()
    vector_src = ListDataset(
        [np.array([[0.1, 0.9]], dtype=np.float32)],
        TensorSpec("float32", shape=(1, 2), backend="numpy"),
    )
    assert [int(item) for item in Map(vector_src, Pipe(Flatten(), argmax))] == [1]
    assert argmax.dispatch_count == 1


def test_flatten_and_scale_infer_output_spec_and_iteration():
    src = ListDataset(
        [np.array([[0, 255]], dtype=np.uint8)],
        TensorSpec("uint8", shape=(1, 2), backend="numpy"),
    )

    ds = Map(src, Scale.from_range(0, 255), Flatten())
    out = list(ds)

    assert ds.spec == TensorSpec("float32", shape=(2,), backend="numpy")
    np.testing.assert_allclose(out[0], np.array([0.0, 1.0], dtype=np.float32))


def test_argmax_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            np.array([0.1, 0.9, 0.0], dtype=np.float32),
            np.array([0.8, 0.1, 0.1], dtype=np.float32),
        ],
        TensorSpec("float32", shape=(3,), backend="numpy"),
    )

    ds = Map(src, ArgMax())
    out = list(ds)

    assert ds.spec == TensorSpec("int64", shape=(), backend="numpy")
    assert [int(item) for item in out] == [1, 0]


def test_argmax_batched_preserves_batch_axis():
    src = ListDataset(
        [
            np.array([0.1, 0.9, 0.0], dtype=np.float32),
            np.array([0.8, 0.1, 0.1], dtype=np.float32),
        ],
        TensorSpec("float32", shape=(3,), backend="numpy"),
    )

    ds = Map(Batch(src, 2), ArgMax())
    out = list(ds)

    assert ds.spec == TensorSpec("int64", shape=(), batch=Dynamic, backend="numpy")
    assert [item.tolist() for item in out] == [[1, 0]]


def test_migrated_data_methods_support_preparation_and_explicit_batch_defaults():
    cast = Cast("float32")
    first = np.array([1, 2], dtype=np.int32)
    cast.learn()
    assert cast(first).dtype == np.dtype("float32")
    cached = cast.cached_signature
    cast.implementations = lambda: (_ for _ in ()).throw(AssertionError("must stay cached"))
    assert cast(np.array([3, 4], dtype=np.int32)).dtype == np.dtype("float32")
    with pytest.raises(PreparedCallMismatchError):
        cast(np.array([3, 4, 5], dtype=np.int32))
    assert cast.cached_signature == cached
    cast.eager()
    assert cast.call_mode == "eager"

    argmax = ArgMax()
    value = np.array([[0.1, 0.9]], dtype=np.float32)
    with pytest.raises(ImplementationSelectionError) as unknown:
        argmax(value)
    assert unknown.value.unknown_traits == ("batch_mode",)
    argmax.default_batched = False
    assert int(argmax(value)[0]) == 1
    argmax.default_batched = True
    assert argmax(value).tolist() == [1]
    argmax.eager()
    argmax.default_batched = False
    argmax.learn()
    assert int(argmax(value)[0]) == 1
    cached_argmax = argmax.cached_signature
    assert int(argmax(value)[0]) == 1
    assert argmax.cached_signature == cached_argmax
    argmax.eager()

    flatten = Flatten()
    with pytest.raises(ImplementationSelectionError) as flatten_unknown:
        flatten(value)
    assert flatten_unknown.value.unknown_traits == ("batch_mode",)
    flatten.default_batched = False
    assert flatten(value).shape == (2,)
    flatten.default_batched = True
    assert flatten(value).shape == (1, 2)


def test_learned_pipe_caches_locally_selected_children():
    """Composite cached calls bypass both top-level and child candidate discovery."""

    first = CountingCast("float32")
    second = CountingCast("float64")
    pipe = Pipe(first, second)
    value = np.array([1, 2], dtype=np.int32)
    pipe.learn()

    assert pipe(value).dtype == np.dtype("float64")
    assert (first.dispatch_count, second.dispatch_count) == (1, 1)

    first.implementations = lambda: (_ for _ in ()).throw(AssertionError("first must stay selected"))
    second.implementations = lambda: (_ for _ in ()).throw(AssertionError("second must stay selected"))
    assert pipe(np.array([3, 4], dtype=np.int32)).dtype == np.dtype("float64")
    assert (first.dispatch_count, second.dispatch_count) == (1, 1)


def test_data_import_keeps_optional_backends_unloaded():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.data; "
            "assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_batch_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Batch(src, 2)
    out = list(ds)

    assert ds.spec == TensorSpec("int32", shape=(2,), batch=Dynamic, backend="numpy")
    assert ds.spec.backend is Backend.numpy
    assert [item.shape for item in out] == [(2, 2), (1, 2)]
    assert out[0].tolist() == [[1, 2], [3, 4]]
    assert out[1].tolist() == [[5, 6]]


def test_unbatch_infer_output_spec_and_iteration():
    src = ListDataset(
        [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32),
        ],
        TensorSpec("int32", shape=(2,), batch=2, backend="numpy"),
    )

    ds = Unbatch(src)
    out = list(ds)

    assert ds.spec == TensorSpec("int32", shape=(2,), backend="numpy")
    assert [item.tolist() for item in out] == [[1, 2], [3, 4], [5, 6]]


def test_take_skip_and_repeat_cardinality_and_iteration():
    src = ListDataset(
        [np.array([i], dtype=np.int32) for i in range(5)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    taken = Take(src, 3)
    skipped = Skip(src, 2)
    repeated = Repeat(Take(src, 2), 3)

    assert taken.__len__() == Cardinality.finite(3)
    assert skipped.__len__() == Cardinality.finite(3)
    assert repeated.__len__() == Cardinality.finite(6)
    assert [item.item() for item in taken] == [0, 1, 2]
    assert [item.item() for item in skipped] == [2, 3, 4]
    assert [item.item() for item in repeated] == [0, 1, 0, 1, 0, 1]


def test_take_zero_has_zero_cardinality():
    src = ListDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    ds = Take(src, 0)

    assert ds.__len__() == Cardinality.finite(0)
    assert list(ds) == []


def test_shuffle_is_seeded_and_preserves_elements():
    src = ListDataset(
        [np.array([i], dtype=np.int32) for i in range(5)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    first = [item.item() for item in Shuffle(src, 5, seed=7)]
    second = [item.item() for item in Shuffle(src, 5, seed=7)]

    assert first == second
    assert sorted(first) == [0, 1, 2, 3, 4]


def test_zip_positional_infer_output_spec_and_iteration():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3, 4], dtype=np.float32), np.array([5, 6], dtype=np.float32)],
        TensorSpec("float32", shape=(2,), backend="numpy"),
    )

    ds = Zip(left, right)
    out = list(ds)

    assert ds.spec == (left.spec, right.spec)
    assert [(a.tolist(), b.tolist()) for a, b in out] == [([1], [3.0, 4.0]), ([2], [5.0, 6.0])]


def test_zip_nested_tree_with_int_key():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3], dtype=np.float32)],
        TensorSpec("float32", shape=(1,), backend="numpy"),
    )

    ds = Zip({1: left, "b": {"2": right}})
    out = list(ds)

    assert ds.spec == {1: left.spec, "b": {"2": right.spec}}
    assert len(out) == 1
    assert out[0][1].tolist() == [1]
    assert out[0]["b"]["2"].tolist() == [3.0]


def test_nested_zip_is_dataset_leaf_for_outer_zip():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3], dtype=np.float32), np.array([4], dtype=np.float32)],
        TensorSpec("float32", shape=(1,), backend="numpy"),
    )

    inner = Zip({1: left, "test": right})
    outer = Zip(inner, left)
    out = list(outer)

    assert outer.spec == (inner.spec, left.spec)
    assert out[0][0][1].tolist() == [1]
    assert out[0][0]["test"].tolist() == [3.0]
    assert out[0][1].tolist() == [1]


def test_zip_rejects_noncanonical_dict_key():
    src = ListDataset(
        [np.array([1], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )

    with pytest.raises(TypeError):
        Zip({object(): src})


def test_chain_merges_specs_and_concatenates_sources():
    left = ListDataset(
        [np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)],
        TensorSpec("int32", shape=(1,), backend="numpy"),
    )
    right = ListDataset(
        [np.array([3, 4], dtype=np.int32)],
        TensorSpec("int32", shape=(2,), backend="numpy"),
    )

    ds = Chain(left, right)

    assert ds.spec == TensorSpec("int32", shape=(Dynamic,), backend="numpy")
    assert ds.__len__() == Cardinality.finite(3)
    assert [item.tolist() for item in ds] == [[1], [2], [3, 4]]
