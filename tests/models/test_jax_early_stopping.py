"""Managed completed-epoch early stopping for experimental JAX training."""

import numpy as np
import pytest

from dryml import F
from dryml.core import Repo, TensorSpec
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset, Batch
from dryml.managed import ManagedConfig
from dryml.models import Experiment


def _init(key):
    import jax

    del key
    return {"weight": jax.numpy.zeros((1, 1), dtype=jax.numpy.float32)}, {}


def _apply(parameters, mutable_state, rng, value, training):
    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    return value @ parameters["weight"], mutable_state, next_rng


def _init_factory():
    return _init


def _apply_factory():
    return _apply


def _zero_sgd_factory():
    import optax

    return optax.sgd(0.0)


def _adam_factory():
    import optax

    return optax.adam(0.1)


def _mse(predictions, targets):
    import jax

    return jax.numpy.mean((predictions - targets) ** 2)


def _experiment(*, restore_best_weights=False):
    from dryml.models.jax import EarlyStoppingTraining, Model, Optimizer

    model = Model(
        F(_init_factory),
        F(_apply_factory),
        output_spec=TensorSpec("float32", shape=(1,), backend="jax"),
    )
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(F(_zero_sgd_factory)),
        loss=_mse,
        epochs=5,
        monitor="loss",
        patience=0,
        mode="min",
        min_delta=0.0,
        restore_best_weights=restore_best_weights,
        verbose=0,
    )
    data = Batch(ArrayDataset((
        np.ones((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )), 1)
    return Experiment(model, trainer, train_data=data)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"monitor": ""}, ValueError),
        ({"monitor": "accuracy"}, ValueError),
        ({"patience": True}, TypeError),
        ({"patience": -1}, ValueError),
        ({"mode": "auto"}, ValueError),
        ({"min_delta": -1.0}, ValueError),
        ({"min_delta": float("nan")}, ValueError),
    ],
)
def test_jax_early_stopping_rejects_invalid_configuration(kwargs, error):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import EarlyStoppingTraining, Optimizer

    with pytest.raises(error):
        EarlyStoppingTraining(
            optimizer=Optimizer(F(_zero_sgd_factory)),
            loss=_mse,
            epochs=3,
            verbose=0,
            **kwargs,
        )


def test_jax_early_stopping_accepts_shortened_target_and_keeps_stop_state():
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    exp.train_fn(exp)

    assert exp.state.epoch == 2
    assert exp.state.target_epoch is None
    assert exp.train_fn.early_stopping.best_epoch == 0
    assert exp.train_fn.early_stopping.wait == 1
    assert exp.train_fn.early_stopping.accepted_target == 2
    assert exp.state.step == 4


def test_jax_early_stopping_progress_retry_does_not_repeat_decision(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    import dryml.models.jax.base as jax_base

    class Progress:
        attempts = 0

        def __init__(self, **kwargs):
            del kwargs

        def update(self, *args, **kwargs):
            del args, kwargs

        def epoch_end(self, *args, **kwargs):
            del args, kwargs
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("progress postlude")

        def close(self):
            pass

    monkeypatch.setattr(jax_base, "TrainingProgress", Progress)
    exp = _experiment()

    with pytest.raises(RuntimeError, match="progress postlude"):
        exp.train_fn(exp)
    assert exp.train_fn.early_stopping.wait == 1
    assert exp.state.pending_epoch_postlude == 1

    repo = Repo(stores=tmp_path)
    checkpoint = repo.save_object(exp, deep_capture=True)
    resumed = repo.load_state_ref(checkpoint, reuse_live="never")
    resumed.train_fn(resumed)
    assert resumed.train_fn.early_stopping.wait == 1
    assert resumed.state.target_epoch is None


def test_jax_early_stopping_restores_model_but_not_optimizer_or_rng(monkeypatch):
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import EarlyStoppingTraining, Model, Optimizer

    model = Model(
        F(_init_factory),
        F(_apply_factory),
        output_spec=TensorSpec("float32", shape=(1,), backend="jax"),
    )
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(F(_adam_factory)),
        loss=_mse,
        epochs=5,
        monitor="val_loss",
        patience=0,
        restore_best_weights=True,
        verbose=0,
    )
    data = Batch(ArrayDataset((
        np.ones((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )), 1)
    exp = Experiment(model, trainer, train_data=data, val_data=data)
    validation = iter((1.0, 2.0))
    monkeypatch.setattr(
        trainer,
        "_evaluate",
        lambda *args: {"loss": next(validation)},
    )
    observed = {}
    original = trainer._finish_training_behavior_epoch

    def finish(jax_module, exp, epoch, metrics):
        observed[epoch] = (
            np.asarray(model.parameters["weight"]).copy(),
            np.asarray(jax.random.key_data(model.rng)).copy(),
        )
        return original(jax_module, exp, epoch, metrics)

    monkeypatch.setattr(trainer, "_finish_training_behavior_epoch", finish)

    trainer(exp)

    np.testing.assert_array_equal(model.parameters["weight"], observed[0][0])
    np.testing.assert_array_equal(jax.random.key_data(model.rng), observed[1][1])
    assert np.asarray(trainer.optimizer.state[0].count).item() == 4
    assert trainer.early_stopping.restored is True


def test_jax_early_stopping_round_trip_retains_base_continuation(tmp_path):
    """Specialized persistence retains inherited TrainFunction-owned state."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    trainer = _experiment().train_fn
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    trainer.save_state_to_dir(state_dir, codec="pkl")
    restored = _experiment().train_fn
    restored.restore_state_from_dir(state_dir, codec="pkl")

    assert (state_dir / "train-function-state.json").is_file()
    assert restored.continuation == {}


def test_jax_managed_early_stopping_returns_exact_terminal_state(tmp_path):
    """Managed completion associates the shortened mixed-point graph itself."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(restore_best_weights=True)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")

    assert final == status.final_state_ref == exp.last_state_ref
    assert (restored.state.epoch, restored.state.step, restored.state.target_epoch) == (2, 4, None)
    assert restored.train_fn.early_stopping.accepted_target == 2
    assert restored.train_fn.early_stopping.restored is True
