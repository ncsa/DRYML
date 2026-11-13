import subprocess
import sys
import json

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Batch, Chain, Dataset, GeneratorDataset, Map, Repeat, Select, Shuffle, Skip, Take, Unbatch, Zip
from dryml.methods import Method


class DeclaredDataset(Dataset):
    """Small re-iterable source with independently declared yield cardinality."""

    def __init__(self, values, spec, cardinality):
        self.values = tuple(values)
        self.cardinality = cardinality
        self.opens = 0
        super().__init__(spec=spec)

    def __iter__(self):
        self.opens += 1
        yield from self.values

    def __len__(self):
        return self.cardinality


def _spec(*, batch=None):
    return TensorSpec("float32", shape=(2,), batch=batch, backend="numpy")


def dynamic_batches_factory():
    return iter((np.ones((3, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)))


def short_dynamic_batches_factory():
    return iter(np.ones((size, 2), dtype=np.float32) for size in (2, 2, 1))


def short_object_batches_factory():
    return iter([object() for _ in range(size)] for size in (2, 2, 1))


def seeded_values_factory(*, seed):
    generator = np.random.default_rng(seed)
    while True:
        yield int(generator.integers(0, 2**31))


@pytest.mark.parametrize(
    ("declared", "expected"),
    (
        (3, Cardinality.finite(3)),
        (Cardinality.UNKNOWN, Cardinality.UNKNOWN),
        (Cardinality.INFINITE, Cardinality.INFINITE),
    ),
)
def test_yield_cardinality_normalizes_legacy_declarations_without_iteration(declared, expected):
    dataset = DeclaredDataset((), _spec(), declared)

    assert dataset.yield_cardinality() == expected
    assert dataset.opens == 0


@pytest.mark.parametrize("declared", (True, -1, 1.5, Cardinality.finite(True)))
def test_yield_cardinality_rejects_malformed_legacy_declarations(declared):
    dataset = DeclaredDataset((), _spec(), declared)

    with pytest.raises((TypeError, ValueError)):
        dataset.yield_cardinality()
    assert dataset.opens == 0


def test_example_cardinality_separates_dynamic_batches_from_yields_without_iteration():
    dataset = DeclaredDataset((), _spec(batch=Dynamic), Cardinality.finite(3))

    assert dataset.yield_cardinality() == Cardinality.finite(3)
    assert dataset.example_cardinality() == Cardinality.UNKNOWN
    assert dataset.opens == 0


def test_example_cardinality_uses_fixed_declared_batch_and_zero_yields():
    fixed = DeclaredDataset((), _spec(batch=2), Cardinality.finite(3))
    empty = DeclaredDataset((), _spec(batch=Dynamic), Cardinality.finite(0))
    infinite = DeclaredDataset((), _spec(batch=2), Cardinality.INFINITE)

    assert fixed.example_cardinality() == Cardinality.finite(6)
    assert empty.example_cardinality() == Cardinality.finite(0)
    assert infinite.example_cardinality() == Cardinality.INFINITE


def test_zero_yields_prove_zero_examples_without_a_countable_spec():
    dataset = DeclaredDataset((), "generic", Cardinality.finite(0))

    assert dataset.example_cardinality() == Cardinality.finite(0)


def test_unknown_spec_has_unknown_example_cardinality_without_iteration():
    dataset = DeclaredDataset((), None, Cardinality.finite(2))

    assert dataset.example_cardinality() == Cardinality.UNKNOWN
    assert dataset.opens == 0


def test_derived_yield_count_rejects_a_malformed_source_declaration():
    source = DeclaredDataset((), _spec(), True)

    with pytest.raises(TypeError):
        Batch(source, 2).yield_cardinality()


def test_example_cardinality_rejects_conflicting_fixed_batch_declarations():
    dataset = DeclaredDataset(
        (),
        (_spec(batch=2), _spec(batch=3)),
        Cardinality.finite(1),
    )

    with pytest.raises(ValueError, match="disagree"):
        dataset.example_cardinality()


def test_metadata_counting_does_not_import_optional_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import numpy as np; from dryml.core import TensorSpec; "
            "from dryml.data import ArrayDataset, Batch; "
            "dataset = Batch(ArrayDataset(np.ones((2, 1)), "
            "spec=TensorSpec('float64', shape=(1,), backend='numpy')), 2); "
            "dataset.yield_cardinality(); dataset.example_cardinality(); "
            "assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_examples_in_counts_nested_supervised_dynamic_short_batches():
    spec = {
        "x": (_spec(batch=Dynamic), {"extra": _spec(batch=Dynamic)}),
        "y": _spec(batch=Dynamic),
    }
    dataset = DeclaredDataset((), spec, Cardinality.finite(1))
    value = {
        "x": (
            np.zeros((1, 2), dtype=np.float32),
            {"extra": np.zeros((1, 2), dtype=np.float32)},
        ),
        "y": np.zeros((1, 2), dtype=np.float32),
    }

    assert dataset.examples_in(value) == 1


def test_examples_in_rejects_mismatched_fixed_batches_zero_batches_and_rank_zero_values():
    spec = (_spec(batch=2), _spec(batch=2))
    dataset = DeclaredDataset((), spec, Cardinality.finite(1))

    with pytest.raises(ValueError, match="declared batch size"):
        dataset.examples_in((np.zeros((1, 2)), np.zeros((1, 2))))

    dynamic_pair = DeclaredDataset(
        (),
        (_spec(batch=Dynamic), _spec(batch=Dynamic)),
        Cardinality.finite(1),
    )
    with pytest.raises(ValueError, match="disagree"):
        dynamic_pair.examples_in((np.zeros((2, 2)), np.zeros((3, 2))))

    dynamic = DeclaredDataset((), _spec(batch=Dynamic), Cardinality.finite(1))
    with pytest.raises(ValueError, match="zero examples"):
        dynamic.examples_in(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="rank-0"):
        dynamic.examples_in(np.asarray(1.0))


def test_examples_in_returns_one_for_a_coherent_unbatched_supervised_value():
    dataset = DeclaredDataset((), (_spec(), _spec()), Cardinality.finite(1))

    assert dataset.examples_in((np.zeros((2,)), np.zeros((2,)))) == 1


def test_take_count_remains_a_strict_yield_count_when_source_declaration_is_short():
    source = DeclaredDataset((1, 2), _spec(), Cardinality.finite(2))
    taken = Take(source, 3)

    assert taken.yield_cardinality() == Cardinality.finite(3)
    cursor = taken.iterator()
    assert [next(cursor), next(cursor)] == [1, 2]
    with pytest.raises(ValueError, match="Requested 3"):
        next(cursor)


def test_take_does_not_truncate_values_to_a_descriptive_source_total():
    source = DeclaredDataset((1, 2, 3), _spec(), Cardinality.finite(1))

    assert list(Take(source, 3)) == [1, 2, 3]


def test_structural_nodes_propagate_proved_example_counts_without_iteration():
    source = DeclaredDataset(
        tuple(np.full((2,), value, dtype=np.float32) for value in range(5)),
        _spec(),
        Cardinality.finite(5),
    )
    batches = Batch(source, 2)

    assert batches.example_cardinality() == Cardinality.finite(5)
    assert Batch(source, 2, drop_remainder=True).example_cardinality() == Cardinality.finite(4)
    assert Unbatch(batches).yield_cardinality() == Cardinality.finite(5)
    assert Unbatch(batches).example_cardinality() == Cardinality.finite(5)
    assert Take(batches, 2).example_cardinality() == Cardinality.finite(4)
    assert Skip(Take(batches, 2), 1).example_cardinality() == Cardinality.finite(2)
    assert Skip(batches, 2).example_cardinality() == Cardinality.finite(1)
    assert source.opens == 0


def test_shuffle_keeps_total_examples_but_invalidates_prefix_facts_without_iteration():
    source = DeclaredDataset(
        tuple(np.full((2,), value, dtype=np.float32) for value in range(6)),
        _spec(),
        Cardinality.finite(6),
    )
    shuffled = Shuffle(Batch(source, 2), 2, seed=4)

    assert shuffled.example_cardinality() == Cardinality.finite(6)
    assert Take(shuffled, 2).example_cardinality() == Cardinality.UNKNOWN
    assert source.opens == 0


def test_map_requires_explicit_checked_example_preservation_declaration():
    source = DeclaredDataset(
        (np.ones((2,), dtype=np.float32),), _spec(), Cardinality.finite(1),
    )

    assert Map(source, Select()).example_cardinality() == Cardinality.UNKNOWN
    preserved = Map(source, Select(), preserves_examples=True)
    assert preserved.example_cardinality() == Cardinality.finite(1)
    assert [item.tolist() for item in preserved] == [[1.0, 1.0]]

    class DropOne(Method):
        def __call__(self, value):
            return value[:-1]

        def infer_output_spec(self, input_spec):
            return input_spec

    batched = DeclaredDataset(
        (np.ones((2, 2), dtype=np.float32),), _spec(batch=Dynamic), Cardinality.finite(1),
    )
    invalid = Map(batched, DropOne(), preserves_examples=True)
    with pytest.raises(ValueError, match="preserves_examples"):
        list(invalid)
    graph = invalid.method_graph()
    graph.learn()
    with pytest.raises(ValueError, match="preserves_examples"):
        list(graph.iterator())


def test_opaque_dynamic_generator_can_declare_its_total_examples_without_reading():
    dataset = GeneratorDataset(
        dynamic_batches_factory,
        cardinality=Cardinality.finite(2),
        example_count=Cardinality.finite(5),
        spec=_spec(batch=Dynamic),
    )

    assert dataset.example_cardinality() == Cardinality.finite(5)
    assert Take(dataset, 1).example_cardinality() == Cardinality.UNKNOWN


@pytest.mark.parametrize("factory,spec", (
    (short_dynamic_batches_factory, _spec(batch=Dynamic)),
    (short_object_batches_factory, TensorSpec("object", layout="python", batch=Dynamic)),
))
def test_preserving_map_keeps_explicit_dynamic_batch_total_without_opening_source(factory, spec, monkeypatch):
    import dryml.data.source as source_module

    source = GeneratorDataset(factory, cardinality=3, example_count=5, spec=spec)
    mapped = Map(source, Select(), preserves_examples=True)
    graph = mapped.method_graph()

    def forbidden(*args, **kwargs):
        pytest.fail("example metadata or preparation opened the generator")

    with monkeypatch.context() as patch:
        patch.setattr(source_module, "_fresh_iterable", forbidden)
        assert source.example_cardinality() == Cardinality.finite(5)
        assert mapped.yield_cardinality() == Cardinality.finite(3)
        assert mapped.example_cardinality() == Cardinality.finite(5)
        assert Map(mapped, Select(), preserves_examples=True).example_cardinality() == Cardinality.finite(5)
        assert Map(source, Select()).example_cardinality() == Cardinality.UNKNOWN
        assert Unbatch(mapped).yield_cardinality() == Cardinality.finite(5)
        graph.learn()

    for opener in (mapped.iterator, graph.iterator):
        with opener() as cursor:
            batches = list(cursor)
        assert [len(batch) for batch in batches] == [2, 2, 1]


def test_explicit_generator_total_proves_whole_ranges_but_not_dynamic_batch_subsets(monkeypatch):
    import dryml.data.source as source_module

    source = GeneratorDataset(
        short_dynamic_batches_factory, cardinality=3, example_count=5, spec=_spec(batch=Dynamic),
    )
    mapped = Map(source, Select(), preserves_examples=True)

    def forbidden(*args, **kwargs):
        pytest.fail("range metadata opened the generator")

    monkeypatch.setattr(source_module, "_fresh_iterable", forbidden)
    for dataset in (source, mapped):
        assert Take(dataset, 3).example_cardinality() == Cardinality.finite(5)
        assert Skip(dataset, 0).example_cardinality() == Cardinality.finite(5)
        assert dataset._range_example_cardinality(0, 4) == Cardinality.finite(5)
        assert Take(dataset, 2).example_cardinality() == Cardinality.UNKNOWN
        assert Skip(dataset, 1).example_cardinality() == Cardinality.UNKNOWN
        assert dataset._range_example_cardinality(0, 0) == Cardinality.finite(0)
        assert dataset._range_example_cardinality(3) == Cardinality.finite(0)
        with pytest.raises(ValueError):
            dataset._range_example_cardinality(0, -1)


def test_generator_range_proofs_preserve_unknown_empty_and_infinite_cases():
    unknown_yields = GeneratorDataset(
        short_dynamic_batches_factory, example_count=5, spec=_spec(batch=Dynamic),
    )
    unknown_examples = GeneratorDataset(
        short_dynamic_batches_factory, cardinality=3,
        example_count=Cardinality.UNKNOWN, spec=_spec(batch=Dynamic),
    )
    empty = GeneratorDataset(
        short_dynamic_batches_factory, cardinality=0, example_count=0, spec=_spec(batch=Dynamic),
    )
    infinite = GeneratorDataset(
        short_dynamic_batches_factory, cardinality=Cardinality.INFINITE, spec=_spec(batch=2),
    )

    assert Map(unknown_yields, Select(), preserves_examples=True).example_cardinality() == Cardinality.finite(5)
    assert Take(unknown_yields, 3).example_cardinality() == Cardinality.UNKNOWN
    assert Map(unknown_examples, Select(), preserves_examples=True).example_cardinality() == Cardinality.UNKNOWN
    assert Map(empty, Select(), preserves_examples=True).example_cardinality() == Cardinality.finite(0)
    assert Map(infinite, Select(), preserves_examples=True).example_cardinality() == Cardinality.INFINITE


def test_generator_example_count_rejects_provable_and_observed_conflicts():
    declared_conflict = GeneratorDataset(
        dynamic_batches_factory,
        cardinality=Cardinality.finite(2),
        example_count=Cardinality.finite(4),
        spec=_spec(batch=3),
    )
    observed_conflict = GeneratorDataset(
        dynamic_batches_factory,
        cardinality=Cardinality.finite(2),
        example_count=Cardinality.finite(4),
        spec=_spec(batch=Dynamic),
    )

    with pytest.raises(ValueError, match="conflicts"):
        declared_conflict.example_cardinality()
    with pytest.raises(ValueError, match="conflicts"):
        Map(declared_conflict, Select(), preserves_examples=True).example_cardinality()
    with pytest.raises(ValueError, match="observed"):
        list(observed_conflict)
    with pytest.raises(ValueError, match="observed"):
        list(Map(observed_conflict, Select(), preserves_examples=True))


def test_zip_proves_only_aligned_examples_and_an_empty_input_proves_zero():
    left = DeclaredDataset((np.ones((2,), dtype=np.float32),) * 2, _spec(), Cardinality.finite(2))
    right = DeclaredDataset((np.ones((2,), dtype=np.float32),) * 3, _spec(), Cardinality.finite(3))
    unknown = DeclaredDataset((), _spec(), Cardinality.UNKNOWN)
    empty = DeclaredDataset((), _spec(), Cardinality.finite(0))

    assert Zip(left, right).example_cardinality() == Cardinality.UNKNOWN
    assert Zip(unknown, empty).yield_cardinality() == Cardinality.finite(0)
    assert Zip(unknown, empty).example_cardinality() == Cardinality.finite(0)


def test_chain_sums_compatible_example_totals():
    left = DeclaredDataset((np.ones((2,), dtype=np.float32),) * 2, _spec(), Cardinality.finite(2))
    right = DeclaredDataset((np.ones((2,), dtype=np.float32),) * 3, _spec(), Cardinality.finite(3))

    assert Chain(left, right).example_cardinality() == Cardinality.finite(5)


def test_infinite_repeat_of_empty_source_terminates_after_one_acquisition():
    source = DeclaredDataset((), _spec(), Cardinality.finite(0))

    assert list(Repeat(source)) == []
    assert source.opens == 1


def test_seed_aware_generator_epochs_are_direct_reproducible_and_optionally_fixed():
    source = GeneratorDataset(
        seeded_values_factory,
        cardinality=Cardinality.INFINITE,
        spec=TensorSpec("int64", shape=()),
        seed=37,
        seed_aware=True,
    )

    epoch_two = [*Take(source, 3, epoch=2)]
    assert epoch_two == [*Take(source, 3, epoch=2)]
    repeated = [*Take(Repeat(Take(source, 2)), 4)]
    fixed = [*Take(Repeat(Take(source, 2, fixed_prefix=True)), 4)]
    assert repeated[:2] != repeated[2:]
    assert fixed[:2] == fixed[2:]
    finite_repeat = [*Repeat(Take(source, 2), 2)]
    assert finite_repeat[:2] != finite_repeat[2:]

    reopened = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from dryml.core.cardinality import Cardinality; "
            "from dryml.core.tensor_spec import TensorSpec; "
            "from dryml.data import GeneratorDataset, Take; "
            "from tests.data.test_example_counts import seeded_values_factory; "
            "source = GeneratorDataset(seeded_values_factory, cardinality=Cardinality.INFINITE, "
            "spec=TensorSpec('int64', shape=()), seed=37, seed_aware=True); "
            "print(json.dumps(list(Take(source, 3, epoch=2))))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reopened.returncode == 0, reopened.stderr
    assert json.loads(reopened.stdout) == epoch_two
