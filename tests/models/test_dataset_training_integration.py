"""Credential-free Dataset/native regressions with test-body backend imports."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dryml import F
from dryml.core import Mat, Node, Object, Repo, TensorSpec
from dryml.core.node import require_saved
from dryml.core.repo import default_repo
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.data import ArrayDataset, Batch, RecoverableGenerator, Select, as_supervised
from dryml.managed import ManagedConfig
from dryml.models import Experiment, ExperimentData, Model, TrainFunction


class FailingClosePairs:
    """Import-referencable NumPy source recording cleanup before it raises."""

    closes = []

    def __init__(self):
        """Retain three pairs for an accepted update followed by update failure."""
        self.values = iter(((np.array([1.], dtype=np.float32), np.array([2.], dtype=np.float32)),) * 3)

    def __iter__(self):
        """Return this independently positioned source iterator."""
        return self

    def __next__(self):
        """Return the next pair, raising StopIteration after all three pairs."""
        return next(self.values)

    def close(self):
        """Record one close attempt and raise the secondary cleanup failure."""
        self.closes.append(True)
        raise RuntimeError("source close failed")


_dataset_continuation_events = []
_dataset_continuation_direct_captures = []


class DatasetContinuationDirectPairs(RecoverableGenerator):
    """Tiny infinite direct producer; state extraction is independently counted."""

    def __init__(self):
        self.index = 0

    def __next__(self):
        self.index += 1
        value = np.asarray([self.index], dtype=np.float32)
        return value, value * 2

    def capture_state(self):
        _dataset_continuation_direct_captures.append(self.index)
        return self.index

    def restore_state(self, state):
        self.index = state


def _dataset_continuation_pairs(*, seed):
    """Record a tiny reproducible source's acquisitions and delivered identifiers."""
    _dataset_continuation_events.append("open")
    rng = np.random.default_rng(seed)
    for identifier in rng.permutation(4):
        value = np.asarray([identifier + 1], dtype=np.float32)
        _dataset_continuation_events.append(int(value[0]))
        yield value, value * 2


def _dataset_continuation_short_pairs(actual):
    """Yield a configurable tiny physical input with no captured environment."""
    for _ in range(actual):
        value = np.ones((1,), dtype=np.float32)
        yield value, value


def _dataset_continuation_experiment(backend, *, repo=None, sectioned=True, epochs=2, validation=False):
    """Build independently persisted native owners and four tiny paired inputs."""
    from dryml.data import Epoch, GeneratorDataset
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    data = Batch(GeneratorDataset(_dataset_continuation_pairs, seed=7, seed_aware=True,
                                 spec=(spec, spec), repo=repo), 1, repo=repo)
    if sectioned:
        data = Epoch(data, len=2, repo=repo)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from dryml.models.torch import Model, Optimizer, Training
        model = Model(torch.nn.Linear, 1, 1, repo=repo)
        with torch.no_grad():
            model.obj.weight.zero_()
            model.obj.bias.zero_()
        trainer = Training(optimizer=Optimizer(torch.optim.SGD, target=model, lr=.001,
                                               momentum=.5, repo=repo),
                           loss_cls=torch.nn.MSELoss, epochs=epochs, verbose=0, repo=repo)
    elif backend in ("tf", "keras"):
        tf = pytest.importorskip("tensorflow")
        from dryml.models.tf import Model, Optimizer, Training
        if backend == "keras":
            from dryml.models.tf import BasicTraining as Training
        from tests.models.test_tf_training import ZeroKerasModel
        model = Model(ZeroKerasModel, repo=repo)
        model.obj(tf.zeros((1, 1)))
        trainer = Training(optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=.001,
                                               momentum=.5, repo=repo),
                           epochs=epochs, verbose=0, repo=repo)
    else:
        pytest.importorskip("jax")
        pytest.importorskip("optax", reason="real Optax is required for native JAX qualification")
        from dryml.models.jax import Model, Optimizer, Training
        model = Model(F(_linear_init_factory), F(_linear_apply_factory),
                      output_spec=TensorSpec("float32", shape=(1,), backend="jax"), repo=repo)
        trainer = Training(optimizer=Optimizer(F(_adam_factory), repo=repo), loss=_mse,
                           epochs=epochs, verbose=0, repo=repo)
    return Experiment(model, trainer, train_data=data, val_data=data if validation else None,
                      artifacts={}, repo=repo)


@pytest.mark.parametrize("model_kind", ("model", "sequential"))
def test_tf_training_retains_raw_selection_signature_and_native_sgd(model_kind, monkeypatch):
    """Exact built-ins select once per invocation and still deliver every batch."""
    tf = pytest.importorskip("tensorflow")
    from dryml.core import signatures
    from dryml.core.backend import Backend
    from dryml.methods.implementation import SelectedDescriptorAdapter
    from dryml.models.tf import Model, Optimizer, Sequential, Training

    model = (Model(tf.keras.layers.Dense, 1, kernel_initializer="ones")
             if model_kind == "model" else
             Sequential((F("Dense", 1, kernel_initializer="ones"),)))
    x = np.arange(5, dtype=np.float32).reshape(-1, 1) / 10
    y = x * 2
    model.obj(tf.zeros((1, 1)))
    native = tf.keras.layers.Dense(1)
    native(tf.zeros((1, 1)))
    native.set_weights(model.obj.get_weights())
    native_optimizer = tf.keras.optimizers.SGD(learning_rate=.01)
    loss_fn = tf.keras.losses.MeanSquaredError()
    exp = Experiment(model, Training(optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=.01),
                                    epochs=1, verbose=0),
                     train_data=Batch(ArrayDataset((x, y)), 2))
    graph = model.method_graph()
    eager = model._select(Backend.tf, None)
    canonical = model.find_implementation(backend="tf")
    assert canonical.name == eager.name == "raw_call"
    assert canonical.target is eager.target
    assert canonical.conversion_edge is eager.conversion_edge is None
    selections, catalogs, compiles, deliveries = [], [], [], []
    original_select, original_catalog = Model.find_implementation, Model.implementations
    original_compile = signatures.compile_signature
    original_prepare, original_invoke = SelectedDescriptorAdapter.prepare, SelectedDescriptorAdapter.invoke_prepared

    def select(owner, *args, **kwargs):
        if owner is model:
            selections.append(kwargs)
        return original_select(owner, *args, **kwargs)

    def catalog(owner):
        if owner is model:
            catalogs.append(owner)
        return original_catalog(owner)

    def compile(target):
        if getattr(target, "__name__", None) == "raw_call":
            compiles.append(target)
        return original_compile(target)

    def prepare(adapter, args, kwargs):
        if adapter.name == "raw_call":
            deliveries.append("args")
        return original_prepare(adapter, args, kwargs)

    def invoke(adapter, args, kwargs, **controls):
        result = original_invoke(adapter, args, kwargs, **controls)
        if adapter.name == "raw_call":
            deliveries.append("return")
        return result

    monkeypatch.setattr(Model, "find_implementation", select)
    monkeypatch.setattr(Model, "implementations", catalog)
    monkeypatch.setattr(signatures, "compile_signature", compile)
    monkeypatch.setattr(SelectedDescriptorAdapter, "prepare", prepare)
    monkeypatch.setattr(SelectedDescriptorAdapter, "invoke_prepared", invoke)
    observed = []
    for invocation in range(2):
        expected = []
        for start in range(0, len(x), 2):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(loss_fn(y[start:start + 2], native(x[start:start + 2])))
            native_optimizer.apply_gradients(zip(tape.gradient(loss, native.trainable_variables),
                                                native.trainable_variables))
            expected.append(float(loss.numpy()))
        actual = exp.train_fn(exp, callbacks=(lambda: observed.append(exp.state.step),))
        assert actual == pytest.approx(expected)
        for value, expected_value in zip(model.obj.get_weights(), native.get_weights()):
            np.testing.assert_allclose(value, expected_value)
        assert len(compiles) == invocation + 1
        assert selections == [{"backend": "tf"}] * (invocation + 1)
        # Canonical selection reads compatible candidates, then selects from the catalog.
        assert len(catalogs) == 2 * (invocation + 1)
        assert model.call_mode == "eager" and model.cached_signature is None
        assert model.default_batched is None
        assert model.method_graph().nodes == graph.nodes
        assert model.method_graph().conversion_edges == ()
    assert deliveries == ["args", "return"] * 6
    assert observed == list(range(1, 7))
    assert exp.state.examples_seen == 10 and exp.state.epoch == 2


@pytest.mark.parametrize("backend", ("torch", "tf", "jax", "keras"))
def test_dataset_continuation_native_sections_share_one_source(backend):
    """Two logical sections retain one factory, RNG, conversion plan and input owner."""
    exp = _dataset_continuation_experiment(backend)
    _dataset_continuation_events.clear()
    exp.train_fn(exp)
    assert _dataset_continuation_events.count("open") == 1
    assert len(_dataset_continuation_events) == 5
    assert (exp.state.step, exp.state.epoch, exp.state.examples_seen) == (4, 2, 4)
    assert exp.state.input_recovery == "exact"
    assert exp.state.dataset_state.position == exp.state.input_position
    assert exp.state._input_provider is None


@pytest.mark.parametrize("backend", ("torch", "tf", "jax"))
def test_dataset_continuation_native_checkpoint_suffix(backend, tmp_path, monkeypatch):
    """An actual saved Experiment resumes its accepted input cut and native slots."""
    repo = Repo(DirStore(tmp_path / "native"))
    exp = _dataset_continuation_experiment(backend, repo=repo)
    _dataset_continuation_events.clear()
    saved = []
    def checkpoint():
        if exp.state.step == 1:
            saved.append(repo.save(exp, deep_capture=True))
        if exp.state.step == 2:
            raise RuntimeError("unsaved update")
    with pytest.raises(RuntimeError, match="unsaved update"):
        exp.train_fn(exp, callbacks=(checkpoint,))
    loaded = repo.load_state_ref(saved[0], reuse_live="never")
    assert loaded.state.step == 1
    from dryml.models.utils import TrainingPreparation
    received = []
    original_prepare = TrainingPreparation.prepare
    def prepare(owner, x, y):
        received.append(int(x.item()))
        return original_prepare(owner, x, y)
    monkeypatch.setattr(TrainingPreparation, "prepare", prepare)
    _dataset_continuation_events.clear()
    loaded.train_fn(loaded)
    assert _dataset_continuation_events.count("open") == 1
    assert len(_dataset_continuation_events) == 5  # one replayed prefix, three new yields
    suffix = list(received)
    received.clear()
    recovered = loaded.model
    baseline = _dataset_continuation_experiment(backend)
    baseline.train_fn(baseline)
    assert suffix == received[1:]
    assert loaded.state.step == baseline.state.step == 4
    if backend == "torch":
        import torch
        torch.testing.assert_close(recovered.obj.state_dict(), baseline.model.obj.state_dict())
        torch.testing.assert_close(loaded.train_fn.optimizer.obj.state_dict(),
                                   baseline.train_fn.optimizer.obj.state_dict())
    elif backend == "tf":
        for actual, expected in zip(recovered.obj.weights, baseline.model.obj.weights):
            np.testing.assert_allclose(actual.numpy(), expected.numpy())
        for actual, expected in zip(loaded.train_fn.optimizer.obj.variables,
                                    baseline.train_fn.optimizer.obj.variables):
            np.testing.assert_allclose(actual.numpy(), expected.numpy())
    else:
        np.testing.assert_allclose(recovered.parameters["weight"], baseline.model.parameters["weight"])


@pytest.mark.parametrize("backend", ("torch", "tf", "jax", "keras"))
def test_dataset_continuation_native_implicit_passes(backend):
    """Finite physical passes reopen exactly once per logical epoch."""
    exp = _dataset_continuation_experiment(backend, sectioned=False)
    _dataset_continuation_events.clear()
    exp.train_fn(exp)
    assert _dataset_continuation_events.count("open") == 2
    assert len(_dataset_continuation_events) == 10
    assert (exp.state.step, exp.state.epoch, exp.state.input_sequence) == (8, 2, 10)


@pytest.mark.parametrize("backend", ("torch", "tf"))
@pytest.mark.parametrize("actual", (0, 2))
def test_dataset_continuation_native_declared_shortfall_rejects_completion(backend, actual):
    """Actual explicit trainers cannot finish declared-three, short physical input."""
    from dryml.data import DatasetExhaustedError, GeneratorDataset
    exp = _dataset_continuation_experiment(backend, sectioned=False, epochs=1)
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    exp.train_data = Batch(GeneratorDataset(_dataset_continuation_short_pairs, actual,
                                          cardinality=3, spec=(spec, spec)), 1)
    with pytest.raises(DatasetExhaustedError) as caught:
        exp.train_fn(exp)
    assert (caught.value.requested, caught.value.yielded) == (3, actual)
    assert (exp.state.step, exp.state.examples_seen, exp.state.epoch) == (actual, actual, 0)
    assert exp.state._input_provider is None


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_dataset_continuation_native_finite_final_callback_sees_normalized_epoch(backend):
    """A finite generator's last callback sees the accepted next-epoch frontier."""
    from dryml.data import GeneratorDataset
    exp = _dataset_continuation_experiment(backend, sectioned=False, epochs=1)
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    exp.train_data = Batch(GeneratorDataset(_dataset_continuation_pairs, seed=7, seed_aware=True,
                                          cardinality=4, spec=(spec, spec)), 1)
    _dataset_continuation_events.clear()
    seen = []
    exp.train_fn(exp, callbacks=(lambda: seen.append((exp.state.epoch, exp.state.next_batch)),))
    assert seen == [(0, 1), (0, 2), (0, 3), (1, 0)]
    assert _dataset_continuation_events.count("open") == 1
    assert len(_dataset_continuation_events) == 5
    assert exp.state.step == exp.state.input_sequence == 4


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_dataset_continuation_native_unknown_eof_postlude_reload(backend, tmp_path, monkeypatch):
    """Saved last-value EOF normalizes once; validation precedes next-pass demand."""
    import importlib
    module = importlib.import_module(f"dryml.models.{backend}.base")
    repo = Repo(DirStore(tmp_path / "eof"))
    exp = _dataset_continuation_experiment(backend, repo=repo, sectioned=False, validation=True)
    saved = []
    def checkpoint():
        if exp.state.step == 4:
            saved.append(repo.save(exp, deep_capture=True))
            raise RuntimeError("saved last value")
    with pytest.raises(RuntimeError, match="saved last value"):
        exp.train_fn(exp, callbacks=(checkpoint,))
    loaded = repo.load_state_ref(saved[0], reuse_live="never")
    assert (loaded.state.epoch, loaded.state.next_batch, loaded.state.input_sequence) == (0, 4, 4)
    evaluations = []
    monkeypatch.setattr(type(loaded.train_fn), "_evaluate",
                        lambda *a, **k: evaluations.append("validation") or {"loss": 3.})
    postludes = []
    class Progress:
        def __init__(self, **kwargs):
            pass
        def update(self, *a, **k):
            pass
        def close(self):
            pass
        def epoch_end(self, epoch, **kwargs):
            postludes.append(epoch)
            if len(postludes) == 1:
                assert _dataset_continuation_events.count("open") == 1  # saved prefix replay only
                assert loaded.state.input_sequence == 5
                assert loaded.state.pending_epoch_postlude_phase == "progress"
                assert loaded.state.pending_epoch_metrics["loss"] > 0
                saved.append(repo.save(loaded, deep_capture=True))
                raise RuntimeError("boundary postlude")
    monkeypatch.setattr(module, "TrainingProgress", Progress)
    _dataset_continuation_events.clear()
    with pytest.raises(RuntimeError, match="boundary postlude"):
        loaded.train_fn(loaded)
    resumed = repo.load_state_ref(saved[1], reuse_live="never")
    assert resumed.state.input_sequence == 5
    _dataset_continuation_events.clear()
    resumed.train_fn(resumed)
    assert evaluations == ["validation", "validation"]  # once per completed epoch
    assert postludes == [1, 1, 2]
    assert _dataset_continuation_events.count("open") == 1
    assert (resumed.state.step, resumed.state.epoch, resumed.state.input_sequence) == (8, 2, 10)
    assert resumed.state.dataset_state.position == resumed.state.input_position


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_dataset_continuation_native_stale_retry_precedes_setup(backend, monkeypatch):
    """A callback alone leaves owners valid but cannot authorize uncaptured retry."""
    exp = _dataset_continuation_experiment(backend)
    def pause():
        raise RuntimeError("uncaptured callback")
    with pytest.raises(RuntimeError, match="uncaptured callback"):
        exp.train_fn(exp, callbacks=(pause,))
    assert exp.state.step == exp.state.input_sequence == 1
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("stale input prepared"))
    monkeypatch.setattr(exp.train_fn, "_make_loss", lambda *a: pytest.fail("stale retry built loss"))
    with pytest.raises(ValueError, match="current matching record.*durable reload"):
        exp.train_fn(exp)
    assert exp.state._input_provider is None


@pytest.mark.parametrize("backend", ("torch", "tf"))
@pytest.mark.parametrize("failure_step", (1, 2))
def test_dataset_continuation_native_mutation_failure_invalidates_owners(backend, failure_step, tmp_path, monkeypatch):
    """Model and independently saved momentum slots require durable reload after failure."""
    from dryml.core.repo import RepoSaveError
    repo = Repo(DirStore(tmp_path / "mutation"))
    exp = _dataset_continuation_experiment(backend, repo=repo)
    checkpoint = []
    def capture():
        if exp.state.step == 1:
            checkpoint.append(repo.save(exp, deep_capture=True))
    optimizer = exp.train_fn.optimizer.obj
    name = "step" if backend == "torch" else "apply_gradients"
    original = getattr(type(optimizer), name)
    calls = []
    def fail(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        calls.append(True)
        if len(calls) == failure_step:
            raise RuntimeError("mutated model and slots")
        return result
    monkeypatch.setattr(type(optimizer), name, fail)
    with pytest.raises(RuntimeError, match="mutated model and slots"):
        exp.train_fn(exp, callbacks=(capture,))
    assert exp.state.step == exp.state.input_sequence == failure_step - 1
    for owner in (exp, exp.model, exp.train_fn.optimizer):
        assert owner._restore_failed
        with pytest.raises(RepoSaveError, match="invalidated"):
            repo.save(owner, deep_capture=True)
    with pytest.raises(RepoSaveError, match="invalidated"):
        exp.train_fn(exp)
    if failure_step == 1:
        assert checkpoint == []  # no usable durable recovery cut was fabricated
        return
    monkeypatch.setattr(type(optimizer), name, original)
    loaded = repo.load_state_ref(checkpoint[0], reuse_live="never")
    assert loaded.state.step == loaded.state.input_sequence == 1
    loaded.train_fn(loaded)
    assert loaded.state.step == 4


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_dataset_continuation_native_direct_hooks_only_at_final_capture(backend, monkeypatch):
    """Infinite directly recoverable input is bounded by sections without per-step hooks."""
    from dryml.core.cardinality import Cardinality
    from dryml.data import Epoch, GeneratorDataset
    exp = _dataset_continuation_experiment(backend)
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    exp.train_data = Epoch(Batch(GeneratorDataset(DatasetContinuationDirectPairs,
        cardinality=Cardinality.INFINITE, spec=(spec, spec)), 1), len=2)
    _dataset_continuation_direct_captures.clear()
    exp.train_fn(exp, callbacks=(lambda: _dataset_continuation_direct_captures == []
                                or pytest.fail("direct source snapshot per update"),))
    assert _dataset_continuation_direct_captures == [4]
    assert exp.state.input_replay_policy == "disabled"
    assert exp.state.step == exp.state.input_sequence == 4
    record = exp.state.dataset_state
    exp.train_fn.epochs = 0
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("started zero call prepared"))
    exp.train_fn(exp)
    assert exp.state.dataset_state is record and exp.state.input_recovery == "exact"


@pytest.mark.parametrize("backend", ("torch", "tf"))
def test_dataset_continuation_native_conversion_failure_preserves_primary(backend, monkeypatch):
    """Conversion precedes mutation; a failing source close is only secondary evidence."""
    from dryml.data import GeneratorDataset
    from dryml.models.utils import TrainingPreparation
    exp = _dataset_continuation_experiment(backend)
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    exp.train_data = Batch(GeneratorDataset(FailingClosePairs, spec=(spec, spec)), 1)
    FailingClosePairs.closes.clear()
    def fail(*args):
        raise ValueError("conversion before mutation")
    monkeypatch.setattr(TrainingPreparation, "prepare", fail)
    with pytest.raises(ValueError, match="conversion before mutation") as caught:
        exp.train_fn(exp)
    assert FailingClosePairs.closes == [True]
    assert exp.state.step == exp.state.input_sequence == exp.state.examples_seen == 0
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    assert exp.state._input_provider is None
    assert any("source close failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("backend,kind", (("torch", "native"), ("tf", "native"),
                                         ("torch", "cache"), ("torch", "transient"),
                                         ("tf", "transient")))
@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_dataset_continuation_positional_compatibility(backend, kind, tmp_path, monkeypatch):
    """Unsupported sources/transient owners keep explicitly positional same-live retry."""
    from dryml.data import UnsupportedContinuationError
    exp = _dataset_continuation_experiment(backend, sectioned=False)
    values = np.arange(3, dtype=np.float32).reshape(-1, 1)
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    if kind == "native":
        if backend == "torch":
            import torch
            from dryml.data import TorchDatasetAdapter
            from tests.models.test_torch_training import DatasetContinuationTorchPairs
            native = DatasetContinuationTorchPairs(values)
            native_spec = TensorSpec("float32", shape=(1,), backend="torch")
            source = TorchDatasetAdapter(native, spec=(native_spec, native_spec))
        else:
            import types
            import tensorflow as tf
            from dryml.data import TFDSAdapter
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            native = tf.data.Dataset.from_tensor_slices((values, values)).with_options(options)
            monkeypatch.setitem(sys.modules, "tensorflow_datasets", types.SimpleNamespace(load=lambda *a, **k: native))
            source = TFDSAdapter("local-test", spec=(spec, spec), as_numpy=True, download=False)
        exp.train_data = Batch(source, 1)
    elif kind == "cache":
        from dryml.artifacts import CachedDataset
        repo = Repo(DirStore(tmp_path / "cache"))
        source = CachedDataset(ArrayDataset((values, values)))
        source.compute(codec="numpy", managed=ManagedConfig(state_repo=repo))
        assert source.ready
        exp.train_data = Batch(source, 1)
    else:
        if backend == "torch":
            from dryml.models.torch import Training
        else:
            from dryml.models.tf import Training
        exp.train_fn = Training(epochs=2, verbose=0)
        exp.train_data = Batch(ArrayDataset((values, values)), 1)
    with ArrayDataset(values).open() as traversal:
        record = traversal.capture_state()
    if kind != "transient":
        with pytest.raises(UnsupportedContinuationError):
            exp.train_data.open(state=record)
    def pause():
        raise RuntimeError("positional callback")
    with pytest.raises(RuntimeError, match="positional callback"):
        exp.train_fn(exp, callbacks=(pause,))
    assert exp.state.input_recovery == "positional" and exp.state.dataset_state is None
    assert not exp.model._restore_failed
    exp.train_fn(exp)
    assert exp.state.step == exp.state.examples_seen == 6
    assert exp.state.epoch == 2 and exp.state.dataset_state is None


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
    """Invoke managed JAX training and return its exact final Node from a worker."""

    return experiment.train().final_state_ref


def test_generic_trainer_and_empty_artifacts_use_final_dataset_cardinality(tmp_path):
    """History and a generic trainer agree on the projected Dataset's five examples."""

    repo = Repo(DirStore(tmp_path / "state"))
    trainer = CountingTrainer(repo=repo)
    exp = Experiment(
        CountingModel(repo=repo), trainer, train_data=_pair_data(repo=repo),
        artifacts={}, repo=repo,
    )

    result = exp.train(managed=ManagedConfig(state_repo=repo))
    assert result.value is None
    final = result.final_state_ref
    row = ExperimentData.find(final.object_view(), repo=repo).data.iloc[0]

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


def test_torch_update_failure_survives_source_cleanup_with_accepted_prefix(monkeypatch):
    """A failed second update keeps the accepted first update and its primary error."""
    torch = pytest.importorskip("torch")
    from dryml.data import GeneratorDataset
    from dryml.models.torch import Model, Optimizer, Training

    FailingClosePairs.closes.clear()
    spec = TensorSpec("float32", shape=(1,), backend="numpy")
    dataset = Batch(GeneratorDataset(FailingClosePairs, spec=(spec, spec)), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.01),
        loss_cls=torch.nn.MSELoss, verbose=0,
    )
    exp = Experiment(model, trainer, train_data=dataset)
    original = torch.optim.SGD.step
    calls = []

    def step(optimizer, *args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("update failed")
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", step)
    with pytest.raises(ValueError, match="update failed"):
        trainer(exp)
    assert FailingClosePairs.closes == [True]
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (1, 1, 1)


def test_tf_selected_element_and_batched_calls_preserve_batch_boundaries_and_cache(monkeypatch):
    """Selected TF calls cross once and retain element/batch and cache semantics."""
    tf = pytest.importorskip("tensorflow")
    from dryml.models.tf import Sequential

    model = Sequential(layer_defs=(F("Dense", units=2),))
    element_spec = TensorSpec("float32", shape=(3,), backend="numpy")
    batch_spec = TensorSpec("float32", shape=(3,), batch=2, backend="numpy")
    element = model.find_implementation(input_spec=element_spec)
    batched = model.find_implementation(input_spec=batch_spec)
    import dryml.methods.conversion as conversion
    crossings = []
    original_convert = conversion.convert
    monkeypatch.setattr(conversion, "convert",
                        lambda edge, value: (crossings.append(edge), original_convert(edge, value))[1])

    assert tuple(element(np.zeros((3,), dtype=np.float32)).shape) == (2,)
    assert tuple(batched(np.zeros((2, 3), dtype=np.float32)).shape) == (2, 2)
    assert crossings == [element.conversion_edge, batched.conversion_edge]

    model.learn()
    direct = model(tf.zeros((1, 3), dtype=tf.float32))
    assert model.call_mode == "cached"
    assert tuple(model(tf.zeros((1, 3), dtype=tf.float32)).shape) == tuple(direct.shape)
    model.eager()
    assert model.call_mode == "eager"


def test_torch_selected_element_and_batched_calls_preserve_batch_boundaries_and_cache(monkeypatch):
    """Selected Torch calls cross once and retain element/batch and cache semantics."""
    torch = pytest.importorskip("torch")
    from dryml.models.torch import Sequential

    model = Sequential(layer_defs=(F("Linear", 3, 2),))
    model.prep_eval()
    element_spec = TensorSpec("float32", shape=(3,), backend="numpy")
    batch_spec = TensorSpec("float32", shape=(3,), batch=2, backend="numpy")
    element = model.find_implementation(input_spec=element_spec)
    batched = model.find_implementation(input_spec=batch_spec)
    import dryml.methods.conversion as conversion
    crossings = []
    original_convert = conversion.convert
    monkeypatch.setattr(conversion, "convert",
                        lambda edge, value: (crossings.append(edge), original_convert(edge, value))[1])

    assert tuple(element(np.zeros((3,), dtype=np.float32)).shape) == (2,)
    assert tuple(batched(np.zeros((2, 3), dtype=np.float32)).shape) == (2, 2)
    assert crossings == [element.conversion_edge, batched.conversion_edge]

    model.learn()
    direct = model(torch.zeros((1, 3), dtype=torch.float32))
    assert model.call_mode == "cached"
    assert tuple(model(torch.zeros((1, 3), dtype=torch.float32)).shape) == tuple(direct.shape)
    model.eager()
    assert model.call_mode == "eager"


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


def test_jax_candidate_accounting_does_not_use_train_state_persistence(monkeypatch):
    """The U5 v3 codec cannot introduce per-update continuation serialization."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.models import TrainState
    exp = _jax_experiment()
    monkeypatch.setattr(TrainState, "__getstate__", lambda _: pytest.fail("candidate persistence encoding"))
    with default_repo(Repo()):
        exp.train_fn(exp)
    assert (exp.state.step, exp.state.examples_seen, exp.state.epoch) == (3, 5, 1)
    assert exp.state.input_started is True
    assert exp.state.dataset_state.position == exp.state.input_position


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

    result = exp.train(managed=ManagedConfig(state_repo=repo))
    assert result.value is None
    final = require_saved(result.final_state_ref)
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")
    history = ExperimentData.find(final.object_view(), repo=repo)
    row = history.data.iloc[-1]

    assert final.graph_equal(status.final)
    assert final.graph_equal(row.state_ref)
    assert (restored.state.step, restored.state.examples_seen) == (3, 5)
    model_ref = final.at(GraphPath((Parameter("model"),)))
    optimizer_ref = final.at(GraphPath((Parameter("train_fn"), Parameter("optimizer"))))
    assert model_ref.object_id != optimizer_ref.object_id
    assert model_ref.state is not None and optimizer_ref.state is not None
    assert model_ref.state != optimizer_ref.state
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
            extra_pythonpath=(
                str(checkout / "src"), str(checkout), str(Path(__file__).parent),
            ),
        ),
    )
    final = view.run(_run_jax_train, initial)

    assert isinstance(final, Node)
    require_saved(final)
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")
    row = ExperimentData.find(final.object_view(), repo=repo).data.iloc[-1]
    assert final.graph_equal(status.final)
    assert final.graph_equal(row.state_ref)
    assert (restored.state.step, restored.state.examples_seen) == (3, 5)
    model_ref = final.at(GraphPath((Parameter("model"),)))
    optimizer_ref = final.at(GraphPath((Parameter("train_fn"), Parameter("optimizer"))))
    assert model_ref.object_id != optimizer_ref.object_id
    assert model_ref.state is not None and optimizer_ref.state is not None
    assert model_ref.state != optimizer_ref.state
    assert row.dataset_size == 5
