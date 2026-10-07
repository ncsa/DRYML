"""Credential-free public Dataset training representatives across backends."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dryml import F
from dryml.core import Mat, Object, Repo, StateRef, TensorSpec
from dryml.core.repo import default_repo
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.data import ArrayDataset, Batch, Select, as_supervised
from dryml.managed import ManagedConfig
from dryml.models import Experiment, ExperimentData, Model, TrainFunction


class CountingModel(Model):
    """Minimal backend-neutral Model used only for Dataset authority evidence."""

    def __call__(self, value):
        """Return ``value`` unchanged without backend work."""

        return value


class CountingTrainer(TrainFunction):
    """Record only public Dataset cardinality and one submitted fit transition."""

    supports_safe_points = False

    def __init__(self):
        """Initialize retained cardinality observations before training."""

        self.yields = None
        self.examples = None

    def __call__(self, exp, *, callbacks=()):
        """Count the finalized Dataset without inspecting its operator graph.

        Args:
            exp: Experiment providing the canonical supervised Dataset.
            callbacks: Unsupported intermediate safe-point callbacks.

        Returns:
            The number of examples represented by the submitted Dataset.

        Raises:
            NotImplementedError: If intermediate callbacks are requested.
            ValueError: If either public cardinality is not finite.

        Side Effects:
            Records one coarse successful fit transition on ``exp.state``.
        """

        if callbacks:
            raise NotImplementedError("CountingTrainer has no intermediate safe points.")
        self.yields = exp.train_data.yield_cardinality().require_finite()
        self.examples = exp.train_data.example_cardinality().require_finite()
        exp.state.record_fit(examples=self.examples)
        exp.state.finish_epoch()
        return self.examples


def _linear_init(key):
    """Return one scalar JAX linear parameter and no mutable state."""

    import jax

    del key
    return {"weight": jax.numpy.zeros((1, 1), dtype=jax.numpy.float32)}, {}


def _linear_apply(parameters, mutable_state, rng, value, training):
    """Apply the tiny functional JAX model and advance its owned RNG."""

    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    return value @ parameters["weight"], mutable_state, next_rng


def _linear_init_factory():
    """Build the definition-compatible tiny JAX initializer."""

    return _linear_init


def _linear_apply_factory():
    """Build the definition-compatible tiny JAX apply callable."""

    return _linear_apply


def _adam_factory():
    """Build the tiny Optax recipe only in an admitted JAX runtime."""

    import optax

    return optax.adam(0.1)


def _mse(predictions, targets):
    """Return one scalar mean JAX loss."""

    import jax

    return jax.numpy.mean((predictions - targets) ** 2)


def _pair_data(*, repo=None):
    """Return five explicitly projected examples in authored 2, 2, 1 batches."""

    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    source = ArrayDataset({
        "left": x,
        "right": x + 1.0,
        "label": x * 2.0,
        "metadata": np.arange(5, dtype=np.int64),
    }, repo=repo)
    pairs = as_supervised(
        source,
        {
            "left": Select.from_path(("left",)),
            "right": Select.from_path(("right",)),
        },
        "label",
    )
    return Batch(pairs, 2, repo=repo)


def _simple_pair_data(*, repo=None):
    """Return five scalar-regression examples in authored 2, 2, 1 batches."""

    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    return Batch(ArrayDataset((x, x * 2.0), repo=repo), 2, repo=repo)


def _jax_experiment(*, repo=None):
    """Build one tiny managed JAX Experiment with independently owned state."""

    from dryml.models.jax import Model as JaxModel
    from dryml.models.jax import Optimizer, Training

    model = JaxModel(
        F(_linear_init_factory), F(_linear_apply_factory),
        output_spec=TensorSpec("float32", shape=(1,), backend="jax"),
        seed=17, repo=repo,
    )
    return Experiment(
        model,
        Training(
            optimizer=Optimizer(F(_adam_factory), repo=repo),
            loss=_mse, epochs=1, verbose=0, repo=repo,
        ),
        train_data=_simple_pair_data(repo=repo), artifacts={}, repo=repo,
    )


def _run_jax_train(experiment: Mat[Object]):
    """Invoke managed JAX training inside a Core worker."""

    return experiment.train()


def test_generic_trainer_and_empty_artifacts_use_final_dataset_cardinality(tmp_path):
    """History and a generic trainer agree on the projected Dataset's five examples."""

    repo = Repo(DirStore(tmp_path / "state"))
    trainer = CountingTrainer(repo=repo)
    exp = Experiment(
        CountingModel(repo=repo), trainer, train_data=_pair_data(repo=repo),
        artifacts={}, repo=repo,
    )

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[0]

    assert (trainer.yields, trainer.examples) == (3, 5)
    assert (row.dataset_size_kind, row.dataset_size) == ("finite", 5)
    assert row.examples_seen == 5
    assert row.expected_artifacts == []
    assert row.eval_artifacts == {}
    assert row.evaluation_status == "completed"


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_native_trainers_consume_authored_dataset_batches(backend):
    """Torch and Keras retain exactly three updates and five examples."""

    data = _simple_pair_data()
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from dryml.models.torch import Model as TorchModel
        from dryml.models.torch import Optimizer, Training

        model = TorchModel(torch.nn.Linear, 1, 1)
        exp = Experiment(
            model,
            Training(
                optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.1),
                loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
            ),
            train_data=data,
        )
    else:
        tf = pytest.importorskip("tensorflow")
        from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential

        exp = Experiment(
            Sequential(layer_defs=(F("Dense", units=1),)),
            BasicTraining(
                optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.1),
                loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, verbose=0,
            ),
            train_data=data,
        )

    with default_repo(Repo()):
        exp.train_fn(exp)

    assert data.yield_cardinality().require_finite() == 3
    assert data.example_cardinality().require_finite() == 5
    assert (exp.state.step, exp.state.examples_seen) == (3, 5)
    assert exp.train_fn.training_preparation.consumer_specs[0].backend.value == backend


def test_jax_trainer_consumes_authored_dataset_batches():
    """JAX retains the authored three updates and five submitted examples."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    exp = _jax_experiment()

    with default_repo(Repo()):
        exp.train_fn(exp)

    assert exp.train_data.yield_cardinality().require_finite() == 3
    assert exp.train_data.example_cardinality().require_finite() == 5
    assert (exp.state.step, exp.state.examples_seen) == (3, 5)
    assert exp.train_fn.training_preparation.consumer_specs[0].backend.value == "jax"


@pytest.mark.parametrize("workload", ("W1", "W3"))
def test_jax_qualification_model_runs_on_authored_dataset_batches(workload):
    """Qualification JAX factories execute authored batches and supported labels."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from tests.qualification.ml_workflow_workloads import (
        _model_and_training,
        mnist_pipeline,
        native_device_evidence,
        native_device_observations,
        observe_jax_training_tensors,
    )

    model, trainer = _model_and_training("jax", workload, seed=7105)
    if workload == "W1":
        source = ArrayDataset((
            np.zeros((5, 28, 28, 1), dtype=np.uint8),
            np.arange(5, dtype=np.int64),
        ))
        train_data = Batch(mnist_pipeline(source, label_dtype="int32"), 2)
        expected = (3, 5)
    else:
        train_data = _simple_pair_data()
        expected = (30, 50)
    exp = Experiment(model, trainer, train_data=train_data)
    training_tensors, execution_tensors = [], []
    with default_repo(Repo()), observe_jax_training_tensors(
        trainer,
        training_tensors=training_tensors,
        execution_tensors=execution_tensors,
    ):
        trainer(exp)

    assert (exp.state.step, exp.state.examples_seen) == expected
    assert trainer.training_preparation.consumer_specs[0].backend.value == "jax"
    assert native_device_evidence(model, training_tensors=training_tensors) == "cpu"
    assert native_device_observations(
        model,
        training_tensors=training_tensors,
        execution_tensors=execution_tensors,
    )["observed_device"] == "cpu"


def test_jax_managed_local_publishes_exact_history_and_independent_owner_state(tmp_path):
    """A local managed JAX run publishes exact terminal, history, and owner receipts."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    repo = Repo(DirStore(tmp_path / "state"))
    exp = _jax_experiment(repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")
    history = ExperimentData.find(final.object_projection(), repo=repo)
    row = history.data.iloc[-1]

    assert status.final_state_ref == final == row.state_ref
    assert (restored.state.step, restored.state.examples_seen) == (3, 5)
    model_ref = final.at(GraphPath((Parameter("model"),)))
    optimizer_ref = final.at(GraphPath((Parameter("train_fn"), Parameter("optimizer"))))
    assert model_ref.object != optimizer_ref.object
    assert model_ref.states != optimizer_ref.states
    assert row.dataset_size == 5
    assert row.eval_artifacts == {}


def test_jax_managed_subprocess_restores_exact_graph(tmp_path):
    """Core subprocess JAX training returns and restores its exact terminal graph."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    import dryml.dispatch as dispatch
    from dryml.core.execute import CoreOptions
    from dryml.environments import PythonExecutableSpec
    from dryml.execute.subprocess import SubProcessConfig

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    exp = _jax_experiment(repo=repo)
    initial = repo.save(exp, deep_capture=True)
    spool = tmp_path / "spool"
    spool.mkdir()
    checkout = Path(__file__).parents[2]
    view = dispatch.with_options(
        backend=SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
        python=PythonExecutableSpec(
            sys.executable,
            pythonpath_policy="explicit",
            extra_pythonpath=(str(checkout / "src"), str(checkout)),
        ),
    )
    final = view.run(_run_jax_train, initial)

    assert isinstance(final, StateRef)
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")
    row = ExperimentData.find(final.object_projection(), repo=repo).data.iloc[-1]
    assert status.final_state_ref == final == row.state_ref
    assert (restored.state.step, restored.state.examples_seen) == (3, 5)
    model_ref = final.at(GraphPath((Parameter("model"),)))
    optimizer_ref = final.at(GraphPath((Parameter("train_fn"), Parameter("optimizer"))))
    assert model_ref.object != optimizer_ref.object
    assert model_ref.states != optimizer_ref.states
    assert row.dataset_size == 5
