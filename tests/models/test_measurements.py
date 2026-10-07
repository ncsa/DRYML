import numpy as np
import os
import subprocess
import sys

import pytest

from dryml.core.cardinality import Cardinality
from dryml.data import ArrayDataset, Batch, Take, Unbatch
from dryml.models import MeasurementUnavailableError, ParameterCounts
from dryml.models.measurements import dataset_size, parameter_counts_from_parameters


class Parameter:
    def __init__(self, shape):
        self.shape = shape


class CountingDataset:
    def __init__(self, cardinality):
        self.cardinality = cardinality
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(())

    def __len__(self):
        return self.cardinality


def test_parameter_counts_deduplicate_identity_and_preserve_effective_trainability():
    shared = Parameter((2, 3))
    frozen = Parameter((4,))

    counts = parameter_counts_from_parameters(
        (shared, frozen, shared),
        (shared,),
    )
    reversed_counts = parameter_counts_from_parameters(
        (frozen, shared),
        (shared, shared),
    )

    assert counts == ParameterCounts(total=10, trainable=6)
    assert reversed_counts == counts


def test_dataset_size_requires_the_public_dataset_example_contract_without_iteration():
    for cardinality in (Cardinality.finite(17), Cardinality.UNKNOWN, Cardinality.INFINITE):
        dataset = CountingDataset(cardinality)

        assert dataset_size(dataset) is Cardinality.UNKNOWN
        assert dataset.iterations == 0


def test_dataset_size_keeps_known_selected_examples_distinct_from_batches():
    source = ArrayDataset(np.arange(5, dtype=np.float32).reshape(5, 1))
    selected = Take(source, 4)
    batches = Batch(selected, 3)

    assert batches.__len__().require_finite() == 2
    assert dataset_size(batches) == Cardinality.finite(4)
    assert dataset_size(Unbatch(batches)) == Cardinality.finite(4)


def test_dataset_size_never_relabels_native_batch_count_as_example_count():
    from dryml.data import Dataset
    from dryml.core.tensor_spec import TensorSpec

    class NativeBatches(Dataset):
        def __init__(self, examples=None):
            self.examples = examples
            super().__init__(TensorSpec("float32", shape=(1,), batch=3, backend="numpy"))

        def __iter__(self):
            raise AssertionError("dataset_size must not open native batches")

        def __len__(self):
            return Cardinality.finite(2)

        def example_cardinality(self):
            return Cardinality.UNKNOWN if self.examples is None else Cardinality.finite(self.examples)

    assert dataset_size(Unbatch(NativeBatches())) is Cardinality.UNKNOWN
    assert dataset_size(Unbatch(NativeBatches(5))) == Cardinality.finite(5)


def test_torch_native_parameter_counts_cover_shared_frozen_and_lazy_models():
    torch = pytest.importorskip("torch")
    from dryml.torch.measurements import parameter_counts

    shared = torch.nn.Parameter(torch.zeros(2, 3))

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = shared
            self.frozen = shared
            self.other = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def forward(self, value):
            raise AssertionError("measurement must not invoke the model")

    assert parameter_counts(Shared()) == ParameterCounts(total=10, trainable=6)
    with pytest.raises(MeasurementUnavailableError):
        parameter_counts(torch.nn.LazyLinear(2))


def test_composite_measurement_unions_trainable_paths_for_shared_torch_parameter():
    from dryml.models import AutoEncoder
    from dryml.models.torch import Model

    torch = pytest.importorskip("torch")
    shared = torch.nn.Linear(2, 3, bias=False)

    trainable = Model(torch.nn.Linear, 2, 3)
    frozen = Model(torch.nn.Linear, 3, 2)
    for wrapper in (trainable, frozen):
        wrapper.obj = shared
        wrapper.module = shared
        wrapper.mdl = shared
    frozen.trainable_parameters = lambda backend=None: ()

    assert AutoEncoder(trainable, frozen).parameter_counts() == ParameterCounts(total=6, trainable=6)
    assert AutoEncoder(frozen, trainable).parameter_counts() == ParameterCounts(total=6, trainable=6)


def test_tf_native_parameter_counts_distinguish_unbuilt_and_zero_parameter_models():
    tf = pytest.importorskip("tensorflow")
    from dryml.tf.measurements import parameter_counts

    unbuilt = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    with pytest.raises(MeasurementUnavailableError):
        parameter_counts(unbuilt)

    inputs = tf.keras.Input((3,))
    zero = tf.keras.Model(inputs, tf.keras.layers.Activation("linear")(inputs))
    assert parameter_counts(zero) == ParameterCounts(total=0, trainable=0)


def test_tf_native_parameter_counts_deduplicate_shared_and_include_frozen_parameters():
    tf = pytest.importorskip("tensorflow")
    from dryml.tf.measurements import parameter_counts

    class SharedAndFrozen(tf.keras.Model):
        def __init__(self):
            super().__init__()
            shared = tf.keras.layers.Dense(3, use_bias=False)
            self.left = shared
            self.right = shared
            self.frozen = self.add_weight(name="frozen", shape=(4,), trainable=False)

        def call(self, value):
            return self.left(value) + self.right(value)

    native = SharedAndFrozen()
    native(tf.zeros((1, 2)))

    assert parameter_counts(native) == ParameterCounts(total=10, trainable=6)


def test_tf_composite_measurement_unions_shared_parameter_in_both_orders():
    tf = pytest.importorskip("tensorflow")
    from dryml.models import AutoEncoder
    from dryml.models.tf import Model
    from dryml.core.tensor_spec import TensorSpec

    inputs = tf.keras.Input((2,))
    native = tf.keras.Model(inputs, tf.keras.layers.Dense(3, use_bias=False)(inputs))
    # Distinct wrapper definitions ensure Repo traversal visits both paths.
    trainable = Model(tf.keras.Sequential, output_spec=TensorSpec("float32", shape=(3,), backend="tf"))
    frozen = Model(tf.keras.Sequential, output_spec=TensorSpec("float32", shape=(4,), backend="tf"))
    for wrapper in (trainable, frozen):
        wrapper.obj = native
        wrapper.model = native
        wrapper.mdl = native
    frozen.trainable_parameters = lambda backend=None: ()

    expected = ParameterCounts(total=6, trainable=6)
    assert AutoEncoder(trainable, frozen).parameter_counts() == expected
    assert AutoEncoder(frozen, trainable).parameter_counts() == expected


def test_generic_measurement_exports_are_lightweight_in_a_fresh_process():
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    environment = dict(os.environ, PYTHONPATH=source_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dryml.models import ParameterCounts, MeasurementUnavailableError; "
            "assert ParameterCounts(0, 0).total == 0; "
            "assert not ({'tensorflow', 'torch', 'pandas'} & set(sys.modules))",
        ],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
