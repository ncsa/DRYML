"""Persistence compatibility breaks for Dataset-owned trainer input."""

import inspect

import numpy as np
import pytest

from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.materialization import project_cdef_call
from dryml.core.symbol import ImportRef


@pytest.mark.parametrize(
    ("module", "name"),
    (
        ("dryml.models.tf.base", "BasicTraining"),
        ("dryml.models.tf.base", "BasicEarlyStoppingTraining"),
        ("dryml.models.tf.base", "Training"),
        ("dryml.models.torch.base", "Training"),
        ("dryml.models.sklearn.base", "BasicTraining"),
    ),
)
def test_saved_retired_trainer_controls_fail_during_definition_projection(module, name):
    parameters = (("batch_size", 32),)
    if name == "BasicEarlyStoppingTraining":
        parameters = (("args", FrozenTuple(())), ("kwargs", FrozenDict({"batch_size": 32})), ("patience", 3),
                      ("monitor", "val_loss"), ("restore_best_weights", True))
    cdef = ConcreteDefinition._from_persisted_record(
        ImportRef(module, name),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments(parameters),
    )

    with pytest.raises(TypeError, match="retired trainer parameters.*Rebuild the trainer"):
        project_cdef_call(cdef)


@pytest.mark.parametrize(
    ("module", "name"),
    (
        ("dryml.models.tf.base", "BasicTraining"),
        ("dryml.models.tf.base", "BasicEarlyStoppingTraining"),
        ("dryml.models.tf.base", "Training"),
        ("dryml.models.torch.base", "Training"),
        ("dryml.models.sklearn.base", "BasicTraining"),
    ),
)
def test_maintained_trainers_do_not_advertise_retired_pipeline_controls(module, name):
    trainer = getattr(__import__(module, fromlist=[name]), name)
    parameters = inspect.signature(trainer).parameters

    assert not {"batch_size", "num_examples", "shuffle", "shuffle_seed", "shuffle_buffer_size", "x_path", "y_path"}.intersection(parameters)


def test_tensorflow_dataset_owned_batching_and_reserved_fit_options():
    tf = pytest.importorskip("tensorflow")
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.models import Experiment
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    class Native(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = tf.keras.layers.Dense(1)

        def call(self, value):
            return self.layer(value)

    source = as_supervised(ArrayDataset((
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
    )), 0, 1)
    unbatched = Experiment(
        Model(Native),
        BasicTraining(optimizer=Optimizer(tf.keras.optimizers.SGD), loss=Loss(tf.keras.losses.MeanSquaredError), verbose=0),
        train_data=source,
    )
    with pytest.raises(ValueError, match="explicitly batched"):
        unbatched.train_fn(unbatched)
    assert unbatched.state.step == 0

    trainer = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD),
        loss=Loss(tf.keras.losses.MeanSquaredError), verbose=0,
        fit_kwargs={"shuffle": True},
    )
    rejected = Experiment(Model(Native), trainer, train_data=Batch(source, 2))
    with pytest.raises(ValueError, match="Dataset inputs or iteration"):
        trainer(rejected)
    assert rejected.state.step == 0

    trained = Experiment(
        Model(Native),
        BasicTraining(optimizer=Optimizer(tf.keras.optimizers.SGD), loss=Loss(tf.keras.losses.MeanSquaredError), verbose=0),
        train_data=Batch(source, 2),
    )
    trained.train_fn(trained)
    assert (trained.state.step, trained.state.examples_seen) == (2, 3)


def test_torch_dataset_owned_batches_weight_short_final_update():
    torch = pytest.importorskip("torch")
    from dryml.data import ArrayDataset, Batch, as_supervised
    from dryml.models import Experiment
    from dryml.models.torch import Model, Optimizer, Training

    source = as_supervised(ArrayDataset((
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
    )), 0, 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss, verbose=0,
    )
    exp = Experiment(model, trainer, train_data=Batch(source, 2))

    trainer(exp)

    assert (exp.state.step, exp.state.examples_seen, exp.state.loss_denominator) == (2, 3, 3)
