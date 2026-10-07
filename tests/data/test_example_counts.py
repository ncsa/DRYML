import subprocess
import sys

import numpy as np
import pytest

from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import Dynamic, TensorSpec
from dryml.data import Batch, Dataset, Take


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
