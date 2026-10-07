"""Focused experimental JAX Training delivery, accounting, and recovery tests."""

from __future__ import annotations

import base64
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from dryml import F
from dryml.core import Repo, TensorSpec
from dryml.core.cardinality import Cardinality
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset, Batch, GeneratorDataset, Take
from dryml.managed import ManagedConfig
from dryml.methods import Method
from dryml.models import Experiment
from dryml.models.utils import TrainingPreparation


UNKNOWN_SOURCE_OPENS = []


def _unknown_source():
    """Record source acquisition for the unknown-cardinality callback boundary."""

    UNKNOWN_SOURCE_OPENS.append(True)
    yield np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)


class PreparationOwner(Method):
    """Minimal Method owner for inspecting the JAX handoff preparation path."""

    def __call__(self, value):
        """Return the unused test value unchanged."""

        return value


def _linear_init(key):
    """Return one scalar functional linear parameter and no mutable state."""
    import jax

    del key
    return {"weight": jax.numpy.zeros((1, 1), dtype=jax.numpy.float32)}, {}


def _linear_apply(parameters, mutable_state, rng, value, training):
    """Apply one deterministic functional linear candidate."""
    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    return value @ parameters["weight"], mutable_state, next_rng


def _linear_init_factory():
    """Build the definition-compatible test initializer."""
    return _linear_init


def _linear_apply_factory():
    """Build the definition-compatible test apply callable."""
    return _linear_apply


def _stateful_init(key):
    """Build parameters plus mutable state for RNG-continuation recovery proof."""
    import jax

    del key
    return (
        {"weight": jax.numpy.zeros((1, 1), dtype=jax.numpy.float32)},
        {"updates": jax.numpy.asarray(0, dtype=jax.numpy.int32)},
    )


def _stateful_apply(parameters, mutable_state, rng, value, training):
    """Use a stochastic training candidate and advance mutable state once per apply."""
    import jax

    next_rng, dropout_rng = jax.random.split(rng)
    noise = jax.random.bernoulli(dropout_rng, 0.5, value.shape).astype(value.dtype)
    prediction = value @ parameters["weight"] + noise * 0.01
    updates = mutable_state["updates"] + int(training)
    return prediction, {"updates": updates}, next_rng


def _stateful_init_factory():
    """Build the stateful recovery-test initializer."""
    return _stateful_init


def _stateful_apply_factory():
    """Build the stateful recovery-test apply callable."""
    return _stateful_apply


def _mse(predictions, targets):
    """Return the required scalar mean loss for tiny training tests."""
    import jax

    return jax.numpy.mean((predictions - targets) ** 2)


def _raising_loss_factory():
    """Fail if zero-work training incorrectly constructs the caller loss."""

    raise AssertionError("loss factory executed")


def _raising_loss(predictions, targets):
    """Raise from the traced loss body for pre-commit failure coverage."""

    del predictions, targets
    raise RuntimeError("loss failed")


def _nondifferentiable_loss(predictions, targets):
    """Return an integer scalar so JAX rejects differentiation before commit."""

    import jax

    del targets
    return jax.numpy.argmax(predictions)


def _adam_factory():
    """Build an Optax Adam recipe with slots for persistence coverage."""
    import optax

    return optax.adam(0.1)


def _adamw_factory():
    """Build AdamW so frozen leaves prove decoupled decay is masked too."""
    import optax

    return optax.adamw(0.1, weight_decay=0.5)


def _two_weight_init(key):
    """Return separate leaves for mixed frozen/trainable update coverage."""
    import jax

    del key
    return {
        "frozen": jax.numpy.ones((1, 1), dtype=jax.numpy.float32),
        "trainable": jax.numpy.ones((1, 1), dtype=jax.numpy.float32),
    }, {}


def _two_weight_apply(parameters, mutable_state, rng, value, training):
    """Use both test leaves so decay and gradient updates are observable."""
    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    return value @ (parameters["frozen"] + parameters["trainable"]), mutable_state, next_rng


def _shared_init(key):
    """Build a functional parameter tree with one intentionally shared leaf."""

    import jax

    del key
    shared = jax.numpy.ones((1, 1), dtype=jax.numpy.float32)
    return {"left": shared, "right": shared}, {}


def _shared_apply(parameters, mutable_state, rng, value, training):
    """Use both aliases so unsupported shared-gradient training is observable."""

    import jax

    del training
    next_rng, _ = jax.random.split(rng)
    return value @ (parameters["left"] + parameters["right"]), mutable_state, next_rng


def _two_weight_init_factory():
    """Build the mixed-mask test initializer."""
    return _two_weight_init


def _two_weight_apply_factory():
    """Build the mixed-mask test apply callable."""
    return _two_weight_apply


def _shared_init_factory():
    """Build the shared-leaf initializer factory."""

    return _shared_init


def _shared_apply_factory():
    """Build the shared-leaf apply factory."""

    return _shared_apply


def _nnx_linear_factory(*, rngs):
    """Build the tiny NNX module used by the shared trainer test."""
    from flax import nnx

    return nnx.Linear(1, 1, rngs=rngs)


def _seeded_pairs(*, seed):
    """Yield deterministic prebatched pairs for a selected GeneratorDataset epoch."""
    rng = np.random.default_rng(seed)
    for _ in range(3):
        x = rng.normal(size=(1, 1)).astype(np.float32)
        yield x, x * 2.0


def _seeded_elements(*, seed):
    """Yield deterministic unbatched pairs for wrapped Take epoch selection."""

    rng = np.random.default_rng(seed)
    for _ in range(3):
        x = rng.normal(size=(1,)).astype(np.float32)
        yield x, x * 2.0


def _empty_pairs():
    """Return a named empty generator factory for cardinality admission tests."""
    return iter(())


def _spec():
    """Return the explicit one-feature JAX output specification."""
    return TensorSpec("float32", shape=(1,), backend="jax")


def _experiment(*, repo=None, epochs=1):
    """Build a five-example JAX Experiment with an authored short final batch."""
    from dryml.models.jax import Model, Optimizer, Training

    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    y = x * 2.0
    model = Model(F(_linear_init_factory), F(_linear_apply_factory), output_spec=_spec(), repo=repo)
    trainer = Training(
        optimizer=Optimizer(F(_adam_factory), repo=repo), loss=_mse,
        epochs=epochs, verbose=0, repo=repo,
    )
    return Experiment(
        model, trainer,
        train_data=Batch(ArrayDataset((x, y), repo=repo), 2, repo=repo), repo=repo,
    )


def test_jax_native_iterator_yields_prepared_authored_batches():
    """The JAX bridge yields native batches and closes an exhausted traversal."""
    jax = pytest.importorskip("jax")
    import dryml.jax
    from dryml.data.native import PreparedDataset
    from dryml.jax.training_data import iter_training_batches

    data = Batch(ArrayDataset((
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
    )), 2)
    preparation = TrainingPreparation.from_specs(
        PreparationOwner(), data.spec[0], data.spec[1], "jax"
    )

    values = list(iter_training_batches(PreparedDataset(data), preparation))

    assert [tuple(x.shape) for x, _ in values] == [(2, 1), (1, 1)]
    assert all(isinstance(value, jax.Array) for pair in values for value in pair)
    assert [np.asarray(y)[:, 0].tolist() for _, y in values] == [[0.0, 2.0], [4.0]]


def test_jax_training_learns_and_accounts_for_a_short_final_batch():
    """Accepted updates retain three yielded batches and five exact examples."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    losses = exp.train_fn(exp)

    assert len(losses) == 3
    assert exp.state.step == 3
    assert exp.state.examples_seen == exp.state.loss_denominator == 5
    assert (exp.state.epoch, exp.state.next_batch, exp.state.target_epoch) == (1, 0, None)
    assert float(np.asarray(exp.model.parameters["weight"])[0, 0]) > 0.0


def test_jax_training_executes_one_model_apply_per_accepted_update(monkeypatch):
    """The traced transition has no eager preflight apply in addition to an update."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    import jax

    exp = _experiment()
    calls = []
    original = exp.model._apply

    def counted(*args):
        result = original(*args)
        jax.debug.callback(lambda _: calls.append(None), result[0])
        return result

    monkeypatch.setattr(exp.model, "_apply", counted)
    exp.train_fn(exp)

    assert len(calls) == exp.state.step == 3


def test_jax_training_rejects_unbatched_dataset_before_optax_binding():
    """Canonical pair admission rejects missing authored batching without native mutation."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Model, Optimizer, Training

    model = Model(F(_linear_init_factory), F(_linear_apply_factory), output_spec=_spec())
    optimizer = Optimizer(F(_adam_factory))
    exp = Experiment(
        model, Training(optimizer=optimizer, loss=_mse, verbose=0),
        train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))),
    )

    with pytest.raises(ValueError, match="explicitly batched"):
        exp.train_fn(exp)

    assert optimizer.obj is optimizer.state is None
    assert (exp.state.step, exp.state.examples_seen, exp.state.target_epoch) == (0, 0, None)


def test_jax_training_rejects_shared_functional_parameters_before_optimizer_binding():
    """Functional aliases cannot silently use first-path optimizer-mask semantics."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Model, Optimizer, Training

    x = np.ones((1, 1), dtype=np.float32)
    optimizer = Optimizer(F(_adam_factory))
    exp = Experiment(
        Model(F(_shared_init_factory), F(_shared_apply_factory), output_spec=_spec()),
        Training(optimizer=optimizer, loss=_mse, verbose=0),
        train_data=Batch(ArrayDataset((x, x)), 1),
    )

    with pytest.raises(ValueError, match="shared parameter"):
        exp.train_fn(exp)

    assert optimizer.obj is optimizer.state is None
    assert (exp.state.step, exp.state.examples_seen, exp.state.target_epoch) == (0, 0, None)


def test_jax_training_mask_preserves_frozen_weight_against_adamw_decay():
    """A retained false mask leaf stays bitwise equal despite Optax weight decay."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Model, Optimizer, Training

    model = Model(
        F(_two_weight_init_factory), F(_two_weight_apply_factory), output_spec=_spec(),
        trainable_mask={"frozen": False, "trainable": True},
    )
    exp = Experiment(
        model,
        Training(optimizer=Optimizer(F(_adamw_factory)), loss=_mse, verbose=0),
        train_data=Batch(ArrayDataset((np.ones((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32))), 1),
    )

    exp.train_fn(exp)

    np.testing.assert_array_equal(model.parameters["frozen"], np.ones((1, 1), dtype=np.float32))
    assert not np.array_equal(model.parameters["trainable"], np.ones((1, 1), dtype=np.float32))


def test_jax_zero_epoch_is_terminal_without_source_or_optimizer_binding(monkeypatch):
    """Zero epochs validate metadata but do not prepare, read, bind, or update native state."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment(epochs=0)

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("zero epoch must not prepare or bind")

    monkeypatch.setattr(exp.train_data, "prepare", fail)
    monkeypatch.setattr(exp.train_fn.optimizer, "bind", fail)
    assert exp.train_fn(exp) == []
    assert exp.train_fn.optimizer.obj is exp.train_fn.optimizer.state is None
    assert (exp.state.step, exp.state.epoch, exp.state.target_epoch) == (0, 0, None)


def test_jax_zero_epoch_does_not_build_the_loss_factory():
    """Zero work completes after Dataset admission without executing caller loss code."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Training

    exp = _experiment(epochs=0)
    exp.train_fn = Training(
        optimizer=exp.train_fn.optimizer,
        loss=F(_raising_loss_factory),
        epochs=0,
        verbose=0,
    )

    assert exp.train_fn(exp) == []
    assert exp.train_fn.optimizer.obj is exp.train_fn.optimizer.state is None


@pytest.mark.parametrize("value", (object(), ()))
def test_jax_training_rejects_non_dataset_train_input_before_preparation_or_binding(value):
    """Malformed train input gets the documented type error without native mutation."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    exp.train_data = value
    with pytest.raises(TypeError, match="Dataset"):
        exp.train_fn(exp)
    assert exp.train_fn.optimizer.obj is exp.train_fn.optimizer.state is None
    assert exp.train_fn.training_preparation is None


def test_jax_training_rejects_empty_and_infinite_validation_before_binding():
    """Validation cardinality admission never allocates optimizer slots on rejection."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    exp.val_data = Batch(
        GeneratorDataset(_empty_pairs, cardinality=Cardinality.INFINITE, spec=(spec, spec)), 1,
    )
    with pytest.raises(ValueError, match="infinite"):
        exp.train_fn(exp)
    assert exp.train_fn.optimizer.obj is exp.train_fn.optimizer.state is None


def test_jax_training_admission_matrix_preserves_preacquisition_boundaries():
    """Declared empty, infinite, unknown-callback, and noncanonical inputs fail early."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")

    infinite = _experiment()
    infinite.train_data = GeneratorDataset(
        _seeded_pairs, cardinality=Cardinality.INFINITE, spec=(spec, spec), seed=7, seed_aware=True,
    )
    with pytest.raises(ValueError, match="infinite"):
        infinite.train_fn(infinite)
    assert infinite.train_fn.optimizer.obj is infinite.train_fn.optimizer.state is None

    UNKNOWN_SOURCE_OPENS.clear()
    callback_blocked = _experiment()
    callback_blocked.train_data = GeneratorDataset(
        _unknown_source, cardinality=Cardinality.UNKNOWN, spec=(spec, spec),
    )
    with pytest.raises(ValueError, match="finite deterministic"):
        callback_blocked.train_fn(callback_blocked, callbacks=(lambda: None,))
    assert UNKNOWN_SOURCE_OPENS == []
    assert callback_blocked.train_fn.optimizer.obj is callback_blocked.train_fn.optimizer.state is None

    known_empty = _experiment()
    zeros = np.zeros((0, 1), dtype=np.float32)
    known_empty.train_data = Batch(ArrayDataset((zeros, zeros)), 1)
    with pytest.raises(ValueError, match="empty"):
        known_empty.train_fn(known_empty)
    assert known_empty.train_fn.optimizer.obj is known_empty.train_fn.optimizer.state is None

    noncanonical = _experiment()
    noncanonical.train_data = Batch(ArrayDataset(np.zeros((1, 1), dtype=np.float32)), 1)
    with pytest.raises(ValueError, match="canonical"):
        noncanonical.train_fn(noncanonical)
    assert noncanonical.train_fn.optimizer.obj is noncanonical.train_fn.optimizer.state is None


def test_jax_unknown_sources_exhaust_normally_or_fail_truthfully_when_empty():
    """Unknown sources may complete without callbacks but never report empty work as training."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")
    completed = _experiment()
    completed.train_data = GeneratorDataset(
        _seeded_pairs, cardinality=Cardinality.UNKNOWN, spec=(spec, spec), seed=8, seed_aware=True,
    )
    completed.train_fn(completed)
    assert (completed.state.step, completed.state.epoch, completed.state.target_epoch) == (3, 1, None)

    empty = _experiment()
    empty.train_data = GeneratorDataset(_empty_pairs, cardinality=Cardinality.UNKNOWN, spec=(spec, spec))
    before = np.asarray(empty.model.parameters["weight"]).copy()
    with pytest.raises(ValueError, match="empty"):
        empty.train_fn(empty)
    np.testing.assert_array_equal(empty.model.parameters["weight"], before)
    assert (empty.state.step, empty.state.epoch, empty.state.target_epoch) == (0, 0, None)


def test_jax_training_callback_interruption_resumes_without_duplicate_update(tmp_path):
    """A saved accepted update reopens its epoch and skips it before conversion."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    repo = Repo(DirStore(tmp_path / "store"))
    interrupted = _experiment(repo=repo)
    seen = []

    def pause():
        seen.append((interrupted.state.step, interrupted.state.epoch, interrupted.state.next_batch))
        raise RuntimeError("pause after accepted JAX update")

    with pytest.raises(RuntimeError, match="accepted JAX update"):
        interrupted.train_fn(interrupted, callbacks=(pause,))
    checkpoint = repo.save_object(interrupted, deep_capture=True)
    resumed = Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(
        checkpoint, reuse_live="never",
    )
    resumed.train_fn(resumed)

    assert seen == [(1, 0, 1)]
    assert (resumed.state.step, resumed.state.examples_seen) == (3, 5)
    assert (resumed.state.epoch, resumed.state.next_batch, resumed.state.target_epoch) == (1, 0, None)
    assert np.asarray(resumed.train_fn.optimizer.state[0].count).item() == 3


def test_jax_managed_checkpoint_failure_retains_accepted_update_for_retry(tmp_path):
    """A post-commit managed callback failure keeps the cadenced checkpoint truthful."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(repo=repo)
    exp.checkpoint_every_steps = 1
    failures = []

    def fail_once(obj, context):
        del context
        if obj.state.step == 1 and not failures:
            failures.append(obj.state.step)
            raise RuntimeError("post-commit checkpoint callback failed")

    with pytest.raises(RuntimeError, match="post-commit checkpoint callback failed"):
        exp.train(managed=ManagedConfig(state_repo=repo, callbacks=[fail_once]))

    checkpoint = exp.train.status(state_repo=repo).checkpoint_state_ref
    assert failures == [1]
    assert checkpoint is not None
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch, exp.state.target_epoch) == (1, 2, 1, 1)
    assert np.asarray(exp.train_fn.optimizer.state[0].count).item() == 1

    # Managed recovery replays the retained checkpoint before it re-enters the
    # trainer; the live receiver remains the managed retry target here.
    resumed = exp
    resumed.train(managed=ManagedConfig(state_repo=repo))

    assert (resumed.state.step, resumed.state.examples_seen, resumed.state.next_batch, resumed.state.target_epoch) == (3, 5, 0, None)
    assert np.asarray(resumed.train_fn.optimizer.state[0].count).item() == 3


def test_jax_final_update_retry_runs_postlude_once_then_next_invocation_adds_epochs():
    """A final safe-point interruption retains one pending postlude without repeating updates."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()

    def pause_on_final_update():
        if exp.state.step == 3:
            raise RuntimeError("final update accepted")

    with pytest.raises(RuntimeError, match="final update accepted"):
        exp.train_fn(exp, callbacks=(pause_on_final_update,))
    assert (exp.state.step, exp.state.epoch, exp.state.next_batch, exp.state.pending_epoch_postlude) == (3, 1, 0, 0)

    exp.train_fn(exp)
    assert (exp.state.step, exp.state.epoch, exp.state.target_epoch, exp.state.pending_epoch_postlude) == (3, 1, None, None)
    exp.train_fn(exp)
    assert (exp.state.step, exp.state.epoch, exp.state.target_epoch) == (6, 2, None)


def test_jax_validation_retry_discards_candidate_mutable_and_rng_state(monkeypatch):
    """Validation failure never installs its candidate state and retries no optimizer update."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")

    baseline = _experiment()
    baseline.train_fn(baseline)
    exp = _experiment()
    exp.val_data = exp.train_data
    calls = []
    original = exp.train_fn._evaluate

    def fail_once(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("validation failed")
        return original(*args)

    monkeypatch.setattr(exp.train_fn, "_evaluate", fail_once)
    with pytest.raises(RuntimeError, match="validation failed"):
        exp.train_fn(exp)
    np.testing.assert_allclose(exp.model.parameters["weight"], baseline.model.parameters["weight"])
    np.testing.assert_array_equal(jax.random.key_data(exp.model.rng), jax.random.key_data(baseline.model.rng))
    assert (exp.state.step, exp.state.pending_epoch_postlude) == (3, 0)

    exp.train_fn(exp)
    assert calls == [True, True]
    assert (exp.state.step, exp.state.pending_epoch_postlude, exp.state.target_epoch) == (3, None, None)


def test_jax_experiments_do_not_share_native_runtime_owners():
    """Separate Experiment instances retain independent models, slots, and progress."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    first = _experiment()
    second = _experiment()
    second_before = np.asarray(second.model.parameters["weight"]).copy()
    first.train_fn(first)

    assert first.train_fn.optimizer.state is not None
    np.testing.assert_array_equal(second.model.parameters["weight"], second_before)
    assert second.train_fn.optimizer.obj is second.train_fn.optimizer.state is None
    assert (second.state.step, second.state.examples_seen, second.state.epoch) == (0, 0, 0)


@pytest.mark.parametrize("stage", ("before_model", "after_model", "after_optimizer", "after_accounting"))
def test_jax_candidate_commit_interruptions_repair_all_owners(monkeypatch, stage):
    """Any interruption before accounting restores model slots and TrainState together."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    import dryml.models.jax.base as jax_base

    exp = _experiment()
    before = np.asarray(exp.model.parameters["weight"]).copy()
    exp.train_fn.continuation = {"marker": jax.numpy.asarray(7, dtype=jax.numpy.int32)}

    def interrupt(observed):
        if observed == stage:
            raise RuntimeError(stage)

    monkeypatch.setattr(jax_base, "_training_commit_boundary", interrupt)
    with pytest.raises(RuntimeError, match=stage):
        exp.train_fn(exp)

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    np.testing.assert_array_equal(exp.train_fn.continuation["marker"], np.asarray(7, dtype=np.int32))
    assert exp.train_fn.optimizer.state is not None
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)


def test_jax_training_apply_contract_fails_before_owner_installation():
    """Malformed native apply output reports its own contract before candidate commit."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    before = np.asarray(exp.model.parameters["weight"]).copy()
    exp.model._apply = lambda *args: (args[3], {},)
    with pytest.raises(TypeError, match="training apply must return exactly"):
        exp.train_fn(exp)
    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert exp.state.step == 0


@pytest.mark.parametrize("loss", (lambda jax: jax.numpy.asarray([1.0]), lambda jax: jax.numpy.asarray(float("nan"))))
def test_jax_precommit_bad_loss_candidates_leave_all_owned_progress_unaccepted(monkeypatch, loss):
    """Nonscalar and nonfinite candidates fail before model, slots, or state commit."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    before = np.asarray(exp.model.parameters["weight"]).copy()

    monkeypatch.setattr(
        exp.train_fn,
        "_native_update",
        lambda *ignored: lambda parameters, mutable, rng, slots, x, y: (
            loss(jax), parameters, mutable, rng, slots,
        ),
    )
    with pytest.raises(ValueError, match="scalar|finite"):
        exp.train_fn(exp)

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert np.asarray(exp.train_fn.optimizer.state[0].count).item() == 0
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)


@pytest.mark.parametrize("loss", (_raising_loss, _nondifferentiable_loss))
def test_jax_loss_and_differentiation_failures_leave_progress_unaccepted(loss):
    """A traced loss error cannot publish a partially accepted update."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Training

    exp = _experiment()
    exp.train_fn = Training(optimizer=exp.train_fn.optimizer, loss=loss, verbose=0)
    before = np.asarray(exp.model.parameters["weight"]).copy()
    with pytest.raises((RuntimeError, TypeError), match="loss failed|grad"):
        exp.train_fn(exp)

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)


@pytest.mark.parametrize("candidate", ("slots", "model"))
def test_jax_malformed_native_candidates_leave_prior_owners_unchanged(monkeypatch, candidate):
    """Model and Optax candidate validation completes before any eager installation."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment()
    before = np.asarray(exp.model.parameters["weight"]).copy()

    def update(parameters, mutable, rng, slots, x, y):
        del x, y
        return (
            np.asarray(0.0, dtype=np.float32), parameters,
            {"wrong": mutable} if candidate == "model" else mutable,
            rng, () if candidate == "slots" else slots,
        )

    monkeypatch.setattr(exp.train_fn, "_native_update", lambda *ignored: update)
    with pytest.raises(TypeError, match="candidate"):
        exp.train_fn(exp)

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert np.asarray(exp.train_fn.optimizer.state[0].count).item() == 0
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)


def test_jax_synchronization_failure_leaves_prior_owners_unchanged(monkeypatch):
    """A device synchronization failure occurs before candidate installation or accounting."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    import dryml.models.jax.base as jax_base

    exp = _experiment()
    before = np.asarray(exp.model.parameters["weight"]).copy()
    monkeypatch.setattr(jax_base, "_synchronize_tree", lambda *ignored: (_ for _ in ()).throw(RuntimeError("sync failed")))
    with pytest.raises(RuntimeError, match="sync failed"):
        exp.train_fn(exp)

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert np.asarray(exp.train_fn.optimizer.state[0].count).item() == 0
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)


def test_jax_training_updates_first_class_flax_nnx_model():
    """The shared Training loop learns through partitioned NNX candidate state."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    pytest.importorskip("flax.nnx")
    from dryml.models.jax import NNXModel, Optimizer, Training

    x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = x * 2.0
    model = NNXModel(F(_nnx_linear_factory), output_spec=_spec(), seed=3)
    exp = Experiment(
        model,
        Training(optimizer=Optimizer(F(_adam_factory)), loss=_mse, epochs=1, verbose=0),
        train_data=Batch(ArrayDataset((x, y)), 1),
    )

    losses = exp.train_fn(exp)

    assert len(losses) == exp.state.step == 2
    assert exp.state.examples_seen == 2


def test_jax_seeded_take_resume_reopens_the_saved_epoch_before_skip(tmp_path):
    """Seed-aware Take epoch selection keeps resume equivalent to uninterrupted training."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Model, Optimizer, Training

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")

    def make_experiment(*, repo=None):
        data = Batch(Take(GeneratorDataset(
            _seeded_elements, cardinality=Cardinality.INFINITE, spec=(spec, spec),
            seed=91, seed_aware=True, repo=repo,
        ), 3, repo=repo), 1, repo=repo)
        model = Model(F(_linear_init_factory), F(_linear_apply_factory), output_spec=_spec(), repo=repo)
        return Experiment(
            model,
            Training(optimizer=Optimizer(F(_adam_factory), repo=repo), loss=_mse, epochs=2, verbose=0, repo=repo),
            train_data=data, repo=repo,
        )

    baseline = make_experiment()
    baseline.train_fn(baseline)

    repo = Repo(DirStore(tmp_path / "store"))
    interrupted = make_experiment(repo=repo)

    def pause_in_epoch_one():
        if (interrupted.state.epoch, interrupted.state.next_batch) == (1, 1):
            raise RuntimeError("pause in generated epoch")

    with pytest.raises(RuntimeError, match="generated epoch"):
        interrupted.train_fn(interrupted, callbacks=(pause_in_epoch_one,))
    checkpoint = repo.save_object(interrupted, deep_capture=True)
    resumed = Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(
        checkpoint, reuse_live="never",
    )
    resumed.train_fn(resumed)

    np.testing.assert_allclose(resumed.model.parameters["weight"], baseline.model.parameters["weight"])
    np.testing.assert_array_equal(jax.random.key_data(resumed.model.rng), jax.random.key_data(baseline.model.rng))
    assert np.asarray(resumed.train_fn.optimizer.state[0].count).item() == 6
    assert (resumed.state.step, resumed.state.epoch, resumed.state.next_batch) == (6, 2, 0)


def test_jax_training_supports_deterministic_wrapped_take_for_multiple_epochs():
    """Batch around a deterministic Take keeps each prepared epoch independently bounded."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    exp = _experiment(epochs=2)
    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    exp.train_data = Batch(Take(ArrayDataset((x, x * 2.0)), 2), 1)
    exp.train_fn(exp)

    assert (exp.state.step, exp.state.epoch, exp.state.next_batch) == (4, 2, 0)


def test_jax_rng_and_mutable_state_resume_match_uninterrupted_adam(tmp_path):
    """A stochastic mutable functional update restores parameters, slots, and next RNG."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models.jax import Model, Optimizer, Training

    def make_experiment(*, repo=None):
        x = np.arange(3, dtype=np.float32).reshape(-1, 1)
        model = Model(
            F(_stateful_init_factory), F(_stateful_apply_factory),
            output_spec=_spec(), seed=17, repo=repo,
        )
        data = Batch(ArrayDataset((x, x * 2.0), repo=repo), 1, repo=repo)
        return Experiment(
            model,
            Training(optimizer=Optimizer(F(_adam_factory), repo=repo), loss=_mse, epochs=1, verbose=0, repo=repo),
            train_data=data, val_data=data, repo=repo,
        )

    baseline = make_experiment()
    baseline.train_fn(baseline)
    repo = Repo(DirStore(tmp_path / "store"))
    interrupted = make_experiment(repo=repo)

    def pause():
        raise RuntimeError("stochastic pause")

    with pytest.raises(RuntimeError, match="stochastic pause"):
        interrupted.train_fn(interrupted, callbacks=(pause,))
    checkpoint = repo.save_object(interrupted, deep_capture=True)
    resumed = Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(
        checkpoint, reuse_live="never",
    )
    resumed.train_fn(resumed)

    np.testing.assert_allclose(resumed.model.parameters["weight"], baseline.model.parameters["weight"])
    np.testing.assert_array_equal(resumed.model.mutable_state["updates"], baseline.model.mutable_state["updates"])
    assert np.asarray(resumed.model.mutable_state["updates"]).item() == 3
    np.testing.assert_array_equal(jax.random.key_data(resumed.model.rng), jax.random.key_data(baseline.model.rng))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed.train_fn.optimizer.state),
        jax.tree_util.tree_leaves(baseline.train_fn.optimizer.state),
    ):
        np.testing.assert_allclose(actual, expected)


def test_jax_fresh_process_checkpoint_resume_matches_uninterrupted_state(tmp_path):
    """A separate interpreter restores one saved mid-epoch Experiment without replay."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    script = """
import base64
import pickle
import sys
import numpy as np
from dryml import F
from dryml.core import Repo, TensorSpec
from dryml.core.store.dir import DirStore
from dryml.core.symbol import SourceSpec
from dryml.data import ArrayDataset, Batch
from dryml.models import Experiment
from dryml.models.jax import Model, Optimizer, Training

def experiment(repo=None):
    init = SourceSpec.from_source("lambda: lambda key: ({'weight': __import__('jax').numpy.zeros((1, 1), dtype=__import__('jax').numpy.float32)}, {'updates': __import__('jax').numpy.asarray(0, dtype=__import__('jax').numpy.int32)})")
    apply = SourceSpec.from_source("lambda: lambda p, m, rng, x, training: (x @ p['weight'], {'updates': m['updates'] + int(training)}, __import__('jax').random.split(rng)[0])")
    optimizer = SourceSpec.from_source("lambda: __import__('optax').adam(0.1)")
    loss = SourceSpec.from_source("lambda: lambda predictions, targets: __import__('jax').numpy.mean((predictions - targets) ** 2)")
    x = np.arange(5, dtype=np.float32).reshape(-1, 1)
    return Experiment(
        Model(F(init), F(apply), output_spec=TensorSpec('float32', shape=(1,), backend='jax'), seed=23, repo=repo),
        Training(optimizer=Optimizer(F(optimizer), repo=repo), loss=F(loss), epochs=1, verbose=0, repo=repo),
        train_data=Batch(ArrayDataset((x, x * 2.0), repo=repo), 2, repo=repo), repo=repo,
    )

def evidence(exp):
    import jax
    return {
        'parameters': np.asarray(exp.model.parameters['weight']),
        'mutable': np.asarray(exp.model.mutable_state['updates']),
        'rng': np.asarray(jax.random.key_data(exp.model.rng)),
        'slots': [np.asarray(value) for value in jax.tree_util.tree_leaves(exp.train_fn.optimizer.state)],
        'train_state': exp.state.__getstate__(),
    }

if sys.argv[1] == 'save':
    baseline = experiment()
    baseline.train_fn(baseline)
    repo = Repo(DirStore(sys.argv[2]))
    interrupted = experiment(repo)
    def pause():
        raise RuntimeError('pause after accepted update')
    try:
        interrupted.train_fn(interrupted, callbacks=(pause,))
    except RuntimeError as error:
        assert str(error) == 'pause after accepted update'
    state = repo.save_object(interrupted, deep_capture=True)
    print(base64.b64encode(pickle.dumps((state, evidence(baseline)))).decode('ascii'))
else:
    state, expected = pickle.loads(base64.b64decode(sys.argv[3]))
    resumed = Repo(DirStore.open_existing(sys.argv[2])).load_state_ref(state, reuse_live='never')
    resumed.train_fn(resumed)
    actual = evidence(resumed)
    np.testing.assert_allclose(actual['parameters'], expected['parameters'])
    np.testing.assert_array_equal(actual['mutable'], expected['mutable'])
    np.testing.assert_array_equal(actual['rng'], expected['rng'])
    for actual_slot, expected_slot in zip(actual['slots'], expected['slots']):
        np.testing.assert_allclose(actual_slot, expected_slot)
    assert actual['train_state'] == expected['train_state']
    assert resumed.state.step == 3 and resumed.state.examples_seen == 5
"""
    environment = dict(
        os.environ,
        PYTHONPATH=str(__import__("pathlib").Path(__file__).parents[2] / "src"),
        JAX_PLATFORMS="cpu",
    )
    store = tmp_path / "store"
    saved = subprocess.run(
        [sys.executable, "-c", script, "save", str(store)],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert saved.returncode == 0, saved.stderr
    resumed = subprocess.run(
        [sys.executable, "-c", script, "resume", str(store), saved.stdout.strip()],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
